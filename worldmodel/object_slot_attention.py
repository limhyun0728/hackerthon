"""DEVS 객체 slot을 action-conditioned predictor로 상호작용시키는 월드모델 모듈.

Le-WM의 `ViT encoder -> action encoder -> predictor` 흐름에서 ViT encoder를
DEVS typed object slot encoder로 교체한 구조다. 여기서는 이미지에서 slot을
발견하지 않는다. DEVS state가 이미 객체를 알고 있으므로, raw slot은 타입별
projection만 거치고 객체 간 상호작용은 action-conditioned predictor에서 학습한다.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from hackerthon.worldmodel.actions import ACTION_DIM
from hackerthon.worldmodel.slots import (
    MAX_FEATURE_DIM,
    MISSION_FEATURE_NAMES,
    TERRAIN_FEATURE_NAMES,
    UNIT_FEATURE_NAMES,
    ObjectType,
    TeamId,
)


# 지형·임무 slot을 predictor query에서 빼고 key/value로만 둔다.
#
# predictor는 시간축과 객체축을 펼쳐 6층 full self-attention을 돈다. slot의 89%가
# 지형이라(강남역 장애물 142 대 유닛 16) attention 비용의 대부분이 정지한 건물을
# 예측하는 데 쓰인다. query에서 빼면 L_query x L_key가 되어 10~20배 줄어든다.
#
# 한 층이면 self-attention이 행마다 독립이라 그냥 빼도 결과가 같지만, 6층이면 다음
# 층의 지형 key가 이 층의 지형 출력이라 그럴 수 없다. 그래서 지형 token을 입력
# 임베딩 상태로 고정해 모든 층이 같은 것을 참조하게 한다 — 정지 객체라 층을 거치며
# 정제할 내용이 없다는 가정이고, 예측 오차로 확인해야 한다.
#
# 0으로 두면 이전 경로(전체 slot이 query)라 같은 코드로 A/B를 돌릴 수 있다.
STATIC_TERRAIN_KV = os.environ.get("CJEPA_STATIC_TERRAIN_KV", "1") not in ("0", "false", "False")

# rollout 예측에 물리 제약을 씌운다. 학습 손실에는 관여하지 않고 planning/표시에만 쓴다.
#
# 월드모델은 한 번에 6프레임을 내놓을 뿐 프레임 간 연속성이나 이동 한계를 모른다.
# 실측에서 프레임 간 이동이 중앙 2.5m인데 최대 24.9m로, 한 스텝 한계 15m를 1.2%가
# 넘겼다. 전사자는 더 심해서 hp=0인데 6스텝에 44m를 움직였다.
UNIT_HP_FEATURE_INDEX, UNIT_X_FEATURE_INDEX, UNIT_Y_FEATURE_INDEX = 1, 3, 4
# 한 스텝 최대 이동(월드 유닛). cem_planner.MAX_MOVE_PER_STEP, devs next_waypoint와 같다.
MAX_MOVE_PER_STEP_UNITS = 1.0
ENFORCE_ROLLOUT_PHYSICS = os.environ.get("CJEPA_ROLLOUT_PHYSICS", "1") not in ("0", "false", "False")

# 유닛 위치를 절대 좌표가 아니라 **마지막 관측 위치로부터의 변화량**으로 예측한다.
#
# 실측(DEVS 60 에피소드, RED 19,014표본): 유닛이 실제로 움직이는 거리는 t+1s에 평균
# 3.9m / 중앙 0.0m다. 절반은 1초 동안 그 자리에 있다. 그런데 절대 좌표를 디코딩하면
# "안 움직였다"를 표현하는 데도 디코더가 입력 좌표를 토큰 병목으로 재현해야 하고,
# 그 복원 오차만 2.02m가 깔린다. 실제 예측 오차는 9.6m로 **정지 가정(3.9m)보다 2.5배
# 나빴다.** 지형은 이미 같은 이유로 마지막 관측값을 그대로 쓴다.
#
# delta head는 영-초기화한다. 그래서 켠 직후 모델은 정확히 "안 움직임"을 예측하고,
# 학습은 그 기준선 위에서 실제 변화량만 배우면 된다.
#
# 주의: 이 플래그를 켜고 학습한 checkpoint는 켠 상태로 써야 한다. 끄면 delta head가
# 무시돼 위치가 전부 어긋난다.
POSITION_RESIDUAL = os.environ.get("CJEPA_POSITION_RESIDUAL", "0") not in ("0", "false", "False")

# 속성별 블록. 토큰을 통짜로 두면 위치 정보가 embedding 어디에나 중복해서 실릴 수
# 있고, 전체의 38%를 차지하는 latent 손실이 그걸 자유롭게 회전·혼합시킨다. 실측에서
# 위치 손실을 20% 비중까지 올려도 실제 오차가 안 내려간 이유가 이것으로 보인다.
#
# 인코더가 속성을 정해진 블록에 넣고 디코더가 같은 블록에서만 읽으면, EMA 타깃의
# 블록에도 같은 속성이 들어가므로 latent 손실이 "위치 블록을 위치 블록에 맞춰라"가
# 되어 위치 손실과 방향이 정렬된다.
#
# 좌표는 세 타입 모두 갖고 있어(UNIT x/y, TERRAIN x/y, MISSION objective_x/y) 위치
# 블록만은 타입 공용 인코더를 쓴다. 그래야 유닛-건물, 유닛-목표 기하가 같은 축에서
# 비교된다.
#
# transformer는 여전히 전체 차원을 섞는다. 제약은 입출력 경계에만 건다 — dynamics가
# "hp가 0이면 안 움직인다" 같은 관계를 쓸 수 있어야 하기 때문이다.
PROPERTY_BLOCKS = os.environ.get("CJEPA_PROPERTY_BLOCKS", "0") not in ("0", "false", "False")

# embedding_dim을 이 비율로 나눠 블록을 만든다. 합이 1이 아니면 나머지는 정체 블록에
# 붙는다. 위치에 가장 큰 몫을 준다 — 우리가 못 맞히고 있는 축이고, 세 타입이 공유한다.
BLOCK_RATIO = {"position": 0.375, "state": 0.25, "heading": 0.125}

# 타입별로 어느 feature가 어느 블록에 들어가는지. 인코더/디코더가 같은 표를 쓴다.
BLOCK_FEATURE_INDEX: Mapping[int, Mapping[str, tuple[int, ...]]] = {
    int(ObjectType.UNIT): {
        "position": (3, 4),          # x_norm, y_norm
        "state": (1, 2, 7),          # hp_ratio, ammo_ratio, alive
        "heading": (5, 6),           # heading_cos, heading_sin
        "identity": (0,),            # team
    },
    int(ObjectType.TERRAIN): {
        "position": (1, 2),          # x_norm, y_norm
        "state": (5, 6, 7, 8),       # traversability, movement_cost, cover_value, los_block
        "heading": (),
        "identity": (0, 3, 4),       # terrain_type, width_norm, height_norm
    },
    int(ObjectType.MISSION): {
        "position": (1, 2),          # objective_x_norm, objective_y_norm
        "state": (3, 4),             # time_remaining_ratio, completion_flag
        "heading": (),
        "identity": (0,),            # mission_type
    },
}


def block_layout(embedding_dim: int) -> dict[str, slice]:
    """embedding을 속성 블록 slice로 나눈다. 나머지는 identity가 흡수한다."""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim은 0보다 커야 한다")
    sizes = {name: max(1, int(embedding_dim * ratio)) for name, ratio in BLOCK_RATIO.items()}
    used = sum(sizes.values())
    if used >= embedding_dim:
        raise ValueError(f"블록 합({used})이 embedding_dim({embedding_dim}) 이상이다")
    sizes["identity"] = embedding_dim - used
    layout: dict[str, slice] = {}
    start = 0
    for name in ("position", "state", "heading", "identity"):
        layout[name] = slice(start, start + sizes[name])
        start += sizes[name]
    return layout


TEAM_EMBEDDING_INDEX: Mapping[int, int] = {
    int(TeamId.NONE): 0,
    int(TeamId.BLUE): 1,
    int(TeamId.RED): 2,
}


@dataclass(frozen=True)
class ObjectSlotModelConfig:
    """객체 중심 월드모델의 최소 구조 설정."""

    embedding_dim: int = 128
    hidden_dim: int = 2048
    num_encoder_layers: int = 0
    num_predictor_layers: int = 6
    num_heads: int = 16
    dropout: float = 0.1
    history_frames: int = 3
    pred_frames: int = 1
    num_masked_slots: int = 2
    self_state_dim: int = 32
    mask_seed: int = 42
    maskable_type_ids: tuple[int, ...] = (int(ObjectType.UNIT),)
    mask_team_strategy: str = "random_team"
    mask_count_min: int = 1
    mask_count_max: int = 5
    blue_team_mask_probability: float = 1.0
    blue_team_mask_count_min: int = 1
    blue_team_mask_count_max: int = 5
    ema_momentum: float = 0.996

    def __post_init__(self) -> None:
        """학습 구조의 핵심 shape 계약을 즉시 확인한다."""
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim은 0보다 커야 한다")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim은 0보다 커야 한다")
        if self.num_encoder_layers < 0:
            raise ValueError("num_encoder_layers는 음수일 수 없다")
        if self.num_predictor_layers <= 0:
            raise ValueError("num_predictor_layers는 0보다 커야 한다")
        if self.num_heads <= 0:
            raise ValueError("num_heads는 0보다 커야 한다")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim은 num_heads로 나누어떨어져야 한다")
        if self.history_frames <= 0:
            raise ValueError("history_frames는 0보다 커야 한다")
        if self.pred_frames <= 0:
            raise ValueError("pred_frames는 0보다 커야 한다")
        if self.num_masked_slots < 0:
            raise ValueError("num_masked_slots는 음수일 수 없다")
        if self.self_state_dim <= 0:
            raise ValueError("self_state_dim은 0보다 커야 한다")
        if self.self_state_dim > self.embedding_dim:
            raise ValueError("self_state_dim은 embedding_dim보다 클 수 없다")
        known_type_ids = {int(object_type) for object_type in ObjectType}
        invalid = sorted(set(self.maskable_type_ids) - known_type_ids)
        if invalid:
            raise ValueError(f"maskable_type_ids에 정의되지 않은 객체 타입이 있다: {invalid}")
        if self.mask_team_strategy not in ("random_team", "blue", "red", "all"):
            raise ValueError("mask_team_strategy는 random_team, blue, red, all 중 하나여야 한다")
        if self.mask_count_min <= 0:
            raise ValueError("mask_count_min은 0보다 커야 한다")
        if self.mask_count_max < self.mask_count_min:
            raise ValueError("mask_count_max는 mask_count_min보다 작을 수 없다")
        if not 0.0 <= self.blue_team_mask_probability <= 1.0:
            raise ValueError("blue_team_mask_probability는 [0, 1] 범위여야 한다")
        if self.blue_team_mask_count_min <= 0:
            raise ValueError("blue_team_mask_count_min은 0보다 커야 한다")
        if self.blue_team_mask_count_max < self.blue_team_mask_count_min:
            raise ValueError("blue_team_mask_count_max는 blue_team_mask_count_min보다 작을 수 없다")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum은 [0, 1) 범위여야 한다")


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """slot feature를 공통 embedding 차원으로 올리는 작은 MLP."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _expect_rank(name: str, value: torch.Tensor, rank: int) -> None:
    """텐서 rank 계약을 즉시 검증한다."""
    if value.ndim != rank:
        raise ValueError(f"{name} rank는 {rank}이어야 한다: shape={tuple(value.shape)}")


def _expect_bool(name: str, value: torch.Tensor) -> None:
    """mask 텐서는 bool dtype만 허용한다."""
    if value.dtype != torch.bool:
        raise TypeError(f"{name} dtype은 torch.bool이어야 한다: dtype={value.dtype}")


def _ensure_known_type_ids(type_ids: torch.Tensor) -> None:
    """정의되지 않은 객체 타입이 들어오면 학습을 멈춘다."""
    known = (
        (type_ids == int(ObjectType.UNIT))
        | (type_ids == int(ObjectType.TERRAIN))
        | (type_ids == int(ObjectType.MISSION))
    )
    if not torch.all(known):
        bad = type_ids[~known].detach().cpu().tolist()
        raise ValueError(f"정의되지 않은 object type id가 있다: {bad}")


def _team_embedding_indices(team_ids: torch.Tensor) -> torch.Tensor:
    """TeamId.NONE=-1을 embedding table의 0번 index로 옮긴다."""
    indices = torch.empty_like(team_ids, dtype=torch.long)
    known = torch.zeros_like(team_ids, dtype=torch.bool)
    for team_id, embedding_index in TEAM_EMBEDDING_INDEX.items():
        matched = team_ids == team_id
        indices = torch.where(matched, torch.full_like(indices, embedding_index), indices)
        known = known | matched
    if not torch.all(known):
        bad = team_ids[~known].detach().cpu().tolist()
        raise ValueError(f"정의되지 않은 team id가 있다: {bad}")
    return indices


def build_object_attention_mask(type_ids: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
    """객체 간 attention 허용 행렬을 만든다.

    반환값은 shape `(B, N, N)`이고 True가 attention 허용을 뜻한다. 사망 유닛은
    key/value 문맥에서 제외하지만, 자기 자신은 항상 볼 수 있게 둬서 해당 slot의
    dead 상태를 유지·예측할 수 있게 한다.
    """
    _expect_rank("type_ids", type_ids, 2)
    _expect_rank("alive_mask", alive_mask, 2)
    _expect_bool("alive_mask", alive_mask)
    if type_ids.shape != alive_mask.shape:
        raise ValueError("type_ids와 alive_mask shape가 같아야 한다")
    _ensure_known_type_ids(type_ids)

    batch_size, num_slots = type_ids.shape
    is_unit = type_ids == int(ObjectType.UNIT)
    key_is_active = (~is_unit) | alive_mask
    allowed = key_is_active.unsqueeze(1).expand(batch_size, num_slots, num_slots).clone()

    # 각 query slot은 자기 상태를 항상 읽는다. dead unit도 자기 dead 상태는 필요하다.
    eye = torch.eye(num_slots, dtype=torch.bool, device=type_ids.device).unsqueeze(0)
    allowed = allowed | eye

    empty_rows = allowed.sum(dim=-1) == 0
    if torch.any(empty_rows):
        bad = torch.nonzero(empty_rows, as_tuple=False).detach().cpu().tolist()
        raise ValueError(f"attention 가능한 key가 없는 query slot이 있다: {bad}")
    return allowed


def build_maskable_object_mask(type_ids: torch.Tensor, maskable_type_ids: tuple[int, ...]) -> torch.Tensor:
    """C-JEPA식 object-level masking 후보 slot을 고른다.

    현재 설정에서는 unit slot만 mask 대상으로 둔다. 지형과 임무는 전장 조건으로
    계속 보이게 두고, 숨겨진 유닛 상태를 다른 유닛·지형·임무 context로 복원하게 한다.
    """
    _expect_rank("type_ids", type_ids, 2)
    _ensure_known_type_ids(type_ids)
    maskable = torch.zeros_like(type_ids, dtype=torch.bool)
    for object_type_id in maskable_type_ids:
        maskable = maskable | (type_ids == int(object_type_id))
    if not torch.any(maskable):
        raise ValueError("maskable object slot이 하나도 없다")
    return maskable


def select_cjepa_masked_slots(
    *,
    type_ids: torch.Tensor,
    team_ids: torch.Tensor | None = None,
    maskable_type_ids: tuple[int, ...],
    num_masked_slots: int,
    mask_team_strategy: str = "random_team",
    mask_count_min: int = 1,
    mask_count_max: int = 5,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C-JEPA와 같이 같은 layout batch에서 object slot index를 선택한다.

    DEVS slot은 entity id 순서로 고정돼 있어서, 호출마다 rng를 전진시켜
    다른 slot 조합이 마스킹되게 한다. seed 고정 rng를 매번 새로 만들면
    학습 내내 같은 유닛만 마스킹되므로 반드시 지속 rng를 넘겨야 한다.

    반환값:
    - masked_slot_mask: `(B, N)` bool, True가 숨겨진 target slot
    - masked_indices: `(M,)` long, batch 전체에 공통으로 쓰는 slot index
    """
    if num_masked_slots < 0:
        raise ValueError("num_masked_slots는 음수일 수 없다")
    if mask_team_strategy not in ("random_team", "blue", "red", "all"):
        raise ValueError("mask_team_strategy는 random_team, blue, red, all 중 하나여야 한다")
    if mask_count_min <= 0:
        raise ValueError("mask_count_min은 0보다 커야 한다")
    if mask_count_max < mask_count_min:
        raise ValueError("mask_count_max는 mask_count_min보다 작을 수 없다")
    if team_ids is not None:
        _expect_rank("team_ids", team_ids, 2)
        if team_ids.shape != type_ids.shape:
            raise ValueError("team_ids shape는 type_ids shape와 같아야 한다")
    maskable = build_maskable_object_mask(type_ids, maskable_type_ids)
    batch_size, num_slots = type_ids.shape
    if num_masked_slots == 0:
        return torch.zeros_like(maskable), torch.empty(0, dtype=torch.long, device=type_ids.device)

    reference_mask = maskable[0]
    for batch_index in range(1, batch_size):
        if not torch.equal(maskable[batch_index], reference_mask):
            raise ValueError("C-JEPA mask는 같은 slot layout batch에서만 선택한다")

    def team_candidates(team: TeamId) -> torch.Tensor:
        if team_ids is None:
            raise ValueError("team 기반 mask를 쓰려면 team_ids가 필요하다")
        reference_team_mask = reference_mask & (team_ids[0] == int(team))
        for batch_index in range(1, batch_size):
            batch_team_mask = maskable[batch_index] & (team_ids[batch_index] == int(team))
            if not torch.equal(batch_team_mask, reference_team_mask):
                raise ValueError("team mask는 같은 slot layout batch에서만 선택한다")
        return torch.nonzero(reference_team_mask, as_tuple=False).flatten()

    if mask_team_strategy == "random_team":
        candidate_groups = tuple(
            candidates
            for candidates in (team_candidates(TeamId.BLUE), team_candidates(TeamId.RED))
            if candidates.numel() > 0
        )
        if not candidate_groups:
            raise ValueError("BLUE/RED mask 후보 slot이 하나도 없다")
        candidates = candidate_groups[int(rng.integers(0, len(candidate_groups)))]
    elif mask_team_strategy == "blue":
        candidates = team_candidates(TeamId.BLUE)
    elif mask_team_strategy == "red":
        candidates = team_candidates(TeamId.RED)
    else:
        candidates = torch.nonzero(reference_mask, as_tuple=False).flatten()

    if candidates.numel() == 0:
        raise ValueError(f"{mask_team_strategy} mask 후보 slot이 하나도 없다")

    max_count = min(int(mask_count_max), int(candidates.numel()))
    min_count = min(int(mask_count_min), max_count)
    sample_count = int(rng.integers(min_count, max_count + 1))
    selected_offsets = rng.choice(candidates.numel(), sample_count, replace=False)
    selected_offsets = torch.as_tensor(selected_offsets, dtype=torch.long, device=type_ids.device)
    masked_indices = candidates[selected_offsets]
    masked_slot_mask = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=type_ids.device)
    masked_slot_mask[:, masked_indices] = True
    return masked_slot_mask, masked_indices


class BlockObjectSlotEncoder(nn.Module):
    """속성별 블록으로 나눠 인코딩한다. 위치 블록은 타입 공용이다.

    타입별 통짜 MLP(8 -> 128)와 달리, 각 속성을 자기 블록에만 쓴다. 위치는 세 타입이
    같은 인코더를 공유하므로 유닛/건물/목표의 좌표가 embedding의 같은 축에 놓인다.
    """

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.embedding_dim
        hidden = config.hidden_dim
        self.layout = block_layout(dim)
        self.position_encoder = _mlp(2, hidden, self.layout["position"].stop - self.layout["position"].start,
                                     config.dropout)
        state_dim = self.layout["state"].stop - self.layout["state"].start
        heading_dim = self.layout["heading"].stop - self.layout["heading"].start
        self.state_encoders = nn.ModuleDict()
        for type_id, spec in BLOCK_FEATURE_INDEX.items():
            if spec["state"]:
                self.state_encoders[str(type_id)] = _mlp(len(spec["state"]), hidden, state_dim, config.dropout)
        self.heading_encoder = _mlp(2, hidden, heading_dim, config.dropout)
        # 방향이 없는 타입(지형·임무)은 학습된 상수로 채운다. 0으로 두면 그 축이
        # 죽은 채로 attention에 들어가 타입 구분이 흐려진다.
        self.heading_default = nn.Parameter(torch.zeros(heading_dim))
        identity_dim = self.layout["identity"].stop - self.layout["identity"].start
        self.type_embedding = nn.Embedding(len(ObjectType), identity_dim)
        self.team_embedding = nn.Embedding(len(TEAM_EMBEDDING_INDEX), identity_dim)
        self.identity_encoders = nn.ModuleDict()
        for type_id, spec in BLOCK_FEATURE_INDEX.items():
            if spec["identity"]:
                self.identity_encoders[str(type_id)] = _mlp(
                    len(spec["identity"]), hidden, identity_dim, config.dropout
                )
        # 정규화도 블록별로 건다. 전체 차원에 LayerNorm을 걸면 평균·분산이 모든 블록에
        # 걸쳐 계산되어 블록이 다시 섞이고, 입력 쪽 고정이 무의미해진다. 검증에서
        # hp만 바꿨는데 position 블록이 0.46 움직이는 것으로 확인했다.
        self.embedding_dim = dim
        self.block_norms = nn.ModuleDict(
            {name: nn.LayerNorm(sl.stop - sl.start) for name, sl in self.layout.items()}
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
    ) -> torch.Tensor:
        """object feature를 속성 블록에 배치한 token으로 만든다."""
        _expect_rank("features", features, 3)
        _expect_rank("type_ids", type_ids, 2)
        _expect_bool("feature_mask", feature_mask)
        if features.shape[:2] != type_ids.shape or type_ids.shape != team_ids.shape:
            raise ValueError("features/type_ids/team_ids의 batch, slot 차원이 같아야 한다")
        _ensure_known_type_ids(type_ids)

        batch_size, num_slots, _ = features.shape
        token = features.new_zeros((batch_size, num_slots, self.embedding_dim))
        heading_slice = self.layout["heading"]
        token[..., heading_slice] = self.heading_default

        team_index = _team_embedding_indices(team_ids)
        identity_slice = self.layout["identity"]
        token[..., identity_slice] = (
            self.type_embedding(type_ids.long()) + self.team_embedding(team_index)
        )

        for type_id, spec in BLOCK_FEATURE_INDEX.items():
            selected = type_ids == int(type_id)
            if not bool(torch.any(selected)):
                continue
            chosen = features[selected]
            # 디코더와 같은 이유로 mask 인덱싱과 블록 slice를 분리한다.
            block = token[selected]
            block[:, self.layout["position"]] = self.position_encoder(
                chosen[:, list(spec["position"])]
            )
            if spec["state"]:
                block[:, self.layout["state"]] = self.state_encoders[str(type_id)](
                    chosen[:, list(spec["state"])]
                )
            if spec["heading"]:
                block[:, heading_slice] = self.heading_encoder(chosen[:, list(spec["heading"])])
            if spec["identity"]:
                block[:, identity_slice] = block[:, identity_slice] + (
                    self.identity_encoders[str(type_id)](chosen[:, list(spec["identity"])])
                )
            token[selected] = block
        # 블록마다 따로 정규화해야 블록 간 독립이 유지된다.
        parts = [self.block_norms[name](token[..., sl]) for name, sl in self.layout.items()]
        return torch.cat(parts, dim=-1)


class BlockObjectStateDecoder(nn.Module):
    """각 속성을 자기 블록에서만 읽어 feature로 되돌린다."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.embedding_dim
        hidden = config.hidden_dim
        self.layout = block_layout(dim)
        self.position_decoder = _mlp(
            self.layout["position"].stop - self.layout["position"].start, hidden, 2, config.dropout
        )
        self.heading_decoder = _mlp(
            self.layout["heading"].stop - self.layout["heading"].start, hidden, 2, config.dropout
        )
        self.state_decoders = nn.ModuleDict()
        self.identity_decoders = nn.ModuleDict()
        for type_id, spec in BLOCK_FEATURE_INDEX.items():
            if spec["state"]:
                self.state_decoders[str(type_id)] = _mlp(
                    self.layout["state"].stop - self.layout["state"].start, hidden,
                    len(spec["state"]), config.dropout,
                )
            if spec["identity"]:
                self.identity_decoders[str(type_id)] = _mlp(
                    self.layout["identity"].stop - self.layout["identity"].start, hidden,
                    len(spec["identity"]), config.dropout,
                )

    def forward(
        self,
        tokens: torch.Tensor,
        type_ids: torch.Tensor,
        *,
        only_types: tuple[ObjectType, ...] | None = None,
    ) -> torch.Tensor:
        """`only_types`를 주면 그 타입만 디코딩하고 나머지 slot은 0으로 남긴다."""
        _expect_rank("tokens", tokens, 3)
        _expect_rank("type_ids", type_ids, 2)
        if tokens.shape[:2] != type_ids.shape:
            raise ValueError("tokens와 type_ids의 batch, slot 차원이 같아야 한다")
        _ensure_known_type_ids(type_ids)

        batch_size, num_slots, _ = tokens.shape
        decoded = tokens.new_zeros((batch_size, num_slots, MAX_FEATURE_DIM))
        for type_id, spec in BLOCK_FEATURE_INDEX.items():
            if only_types is not None and ObjectType(type_id) not in only_types:
                continue
            selected = type_ids == int(type_id)
            if not bool(torch.any(selected)):
                continue
            chosen = tokens[selected]
            # bool mask와 feature index를 한 번에 쓰면 브로드캐스트가 깨진다
            # (shape [N], [N], [2]). mask로 뽑은 (N, MAX_FEATURE_DIM) 버퍼를 채운 뒤
            # 통째로 되돌려 넣는다.
            filled = decoded.new_zeros((chosen.shape[0], MAX_FEATURE_DIM))
            filled[:, list(spec["position"])] = self.position_decoder(
                chosen[:, self.layout["position"]]
            )
            if spec["state"]:
                filled[:, list(spec["state"])] = self.state_decoders[str(type_id)](
                    chosen[:, self.layout["state"]]
                )
            if spec["heading"]:
                filled[:, list(spec["heading"])] = self.heading_decoder(
                    chosen[:, self.layout["heading"]]
                )
            if spec["identity"]:
                filled[:, list(spec["identity"])] = self.identity_decoders[str(type_id)](
                    chosen[:, self.layout["identity"]]
                )
            decoded[selected] = filled
        return decoded


class TypedObjectSlotEncoder(nn.Module):
    """유닛, 지형, 임무 slot을 같은 embedding 공간으로 사상한다."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.embedding_dim
        self.unit_dim = len(UNIT_FEATURE_NAMES)
        self.terrain_dim = len(TERRAIN_FEATURE_NAMES)
        self.mission_dim = len(MISSION_FEATURE_NAMES)

        self.unit_encoder = _mlp(self.unit_dim, config.hidden_dim, dim, config.dropout)
        self.terrain_encoder = _mlp(self.terrain_dim, config.hidden_dim, dim, config.dropout)
        self.mission_encoder = _mlp(self.mission_dim, config.hidden_dim, dim, config.dropout)
        self.type_embedding = nn.Embedding(len(ObjectType), dim)
        self.team_embedding = nn.Embedding(len(TEAM_EMBEDDING_INDEX), dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
    ) -> torch.Tensor:
        """padding된 object feature를 type-aware token으로 변환한다."""
        _expect_rank("features", features, 3)
        _expect_rank("feature_mask", feature_mask, 3)
        _expect_rank("type_ids", type_ids, 2)
        _expect_rank("team_ids", team_ids, 2)
        _expect_bool("feature_mask", feature_mask)
        if features.shape != feature_mask.shape:
            raise ValueError("features와 feature_mask shape가 같아야 한다")
        if features.shape[-1] != MAX_FEATURE_DIM:
            raise ValueError(f"features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")
        if features.shape[:2] != type_ids.shape or type_ids.shape != team_ids.shape:
            raise ValueError("features/type_ids/team_ids의 batch, slot 차원이 같아야 한다")
        _ensure_known_type_ids(type_ids)

        batch_size, num_slots, _ = features.shape
        encoded = features.new_zeros((batch_size, num_slots, self.output_norm.normalized_shape[0]))
        type_specs = (
            (ObjectType.UNIT, self.unit_dim, self.unit_encoder),
            (ObjectType.TERRAIN, self.terrain_dim, self.terrain_encoder),
            (ObjectType.MISSION, self.mission_dim, self.mission_encoder),
        )

        for object_type, feature_dim, encoder in type_specs:
            selected = type_ids == int(object_type)
            if torch.any(selected):
                required_mask = feature_mask[selected, :feature_dim]
                if not torch.all(required_mask):
                    bad = torch.nonzero(~required_mask, as_tuple=False).detach().cpu().tolist()
                    raise ValueError(f"{object_type.name} 필수 feature mask가 비어 있다: {bad}")
                compact = features[selected, :feature_dim].float()
                encoded[selected] = encoder(compact)

        tokens = encoded + self.type_embedding(type_ids.long())
        tokens = tokens + self.team_embedding(_team_embedding_indices(team_ids))
        return self.output_norm(tokens)


class ObjectSelfAttentionBlock(nn.Module):
    """객체 slot 간 masked self-attention block."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_allowed: torch.Tensor,
        *,
        static_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """attention_allowed=True인 slot 쌍만 self-attention에 사용한다.

        `static_context`를 주면 그 token들을 key/value로만 붙인다. 지형처럼 정지한
        객체를 query에서 빼기 위한 것이고, `attention_allowed`의 key 축이 그만큼
        길어야 한다. predictor 쪽과 같은 이유다 — 층이 여러 개면 다음 층의 지형 key가
        이 층의 지형 출력이라, 고정하지 않으면 query에서 뺄 수 없다.
        """
        _expect_rank("tokens", tokens, 3)
        _expect_rank("attention_allowed", attention_allowed, 3)
        _expect_bool("attention_allowed", attention_allowed)
        batch_size, num_query, _ = tokens.shape
        num_key = num_query if static_context is None else num_query + static_context.shape[1]
        if attention_allowed.shape != (batch_size, num_query, num_key):
            raise ValueError("attention_allowed shape는 (B, N_query, N_key)이어야 한다")

        blocked = ~attention_allowed
        attn_mask = blocked.unsqueeze(1).expand(
            batch_size,
            self.num_heads,
            num_query,
            num_key,
        )
        attn_mask = attn_mask.reshape(batch_size * self.num_heads, num_query, num_key)

        attn_input = self.attn_norm(tokens)
        if static_context is None:
            key_value = attn_input
        else:
            _expect_rank("static_context", static_context, 3)
            key_value = torch.cat([attn_input, self.attn_norm(static_context)], dim=1)
        attn_out, _ = self.attn(
            attn_input,
            key_value,
            key_value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class ObjectSlotTransformer(nn.Module):
    """여러 masked self-attention block을 쌓은 객체 관계 encoder."""

    def __init__(self, *, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ObjectSelfAttentionBlock(embedding_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_allowed: torch.Tensor,
        *,
        static_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """객체 slot들의 관계 표현을 갱신한다.

        `static_context`는 층마다 갱신하지 않고 그대로 넘긴다.
        """
        for layer in self.layers:
            tokens = layer(tokens, attention_allowed, static_context=static_context)
        return self.output_norm(tokens)


class FullSelfAttentionBlock(nn.Module):
    """C-JEPA predictor에서 쓰는 시간 마스크 self-attention block."""

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        static_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """시간축과 객체축을 펼친 token sequence에 causal attention을 적용한다.

        `static_context`를 주면 그 token들을 **key/value로만** 쓴다. 지형·임무처럼
        정지한 객체를 query에서 빼기 위한 것이다. self-attention은 행마다 독립이라
        query를 빼도 남은 행의 출력이 변하지 않지만, 층이 여러 개면 다음 층의 지형
        key가 이 층의 지형 출력이라 그냥 뺄 수 없다. 그래서 지형 token은 입력
        임베딩 상태로 고정해 모든 층이 같은 것을 참조한다.
        """
        _expect_rank("tokens", tokens, 3)
        _expect_rank("attn_mask", attn_mask, 2)
        _expect_bool("attn_mask", attn_mask)
        query_length = tokens.shape[1]
        key_length = query_length if static_context is None else query_length + static_context.shape[1]
        if attn_mask.shape != (query_length, key_length):
            raise ValueError("attn_mask shape는 (L_query, L_key)이어야 한다")
        attn_input = self.attn_norm(tokens)
        if static_context is None:
            key_value = attn_input
        else:
            _expect_rank("static_context", static_context, 3)
            key_value = torch.cat([attn_input, self.attn_norm(static_context)], dim=1)
        attn_out, _ = self.attn(
            attn_input,
            key_value,
            key_value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class TemporalCausalObjectTransformer(nn.Module):
    """각 query frame이 자기 시점까지의 token만 보는 predictor transformer."""

    def __init__(self, *, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FullSelfAttentionBlock(embedding_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        static_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """mask token, 실제 history token, future query token을 causal하게 처리한다.

        `static_context`는 층마다 갱신하지 않고 그대로 넘긴다 — 정지 객체라 층을
        거치며 정제할 내용이 없다는 가정이다.
        """
        for layer in self.layers:
            tokens = layer(tokens, attn_mask=attn_mask, static_context=static_context)
        return self.output_norm(tokens)


class CausalMaskedObjectPredictor(nn.Module):
    """C-JEPA 방식의 object-level masked JEPA predictor.

    입력 token shape는 `(B, T_hist, N, D)`이다. t=0의 모든 객체는 anchor로
    사용하고, 선택된 object slot은 t=1 이후 history에서도 query token으로
    바꾼다. 미래 시점은 모든 slot이 query token이며, transformer는 숨겨진
    객체 상태와 미래 상태를 동시에 복원한다.
    """

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        self.config = config
        # 학습 step마다 다른 slot을 마스킹하기 위한 지속 rng. seed로 시퀀스가 재현된다.
        self._mask_rng = np.random.default_rng(config.mask_seed)
        dim = config.embedding_dim
        total_frames = config.history_frames + config.pred_frames
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.time_pos_embedding = nn.Parameter(torch.randn(1, total_frames, 1, dim))
        self.anchor_projector = nn.Linear(dim, dim)
        self.transformer = TemporalCausalObjectTransformer(
            embedding_dim=dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_predictor_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.output_projection = nn.Linear(dim, dim)

    @property
    def total_frames(self) -> int:
        """history와 future를 합친 predictor 입력 길이."""
        return self.config.history_frames + self.config.pred_frames

    @staticmethod
    def temporal_attention_mask(
        *,
        total_frames: int,
        total_token_slots: int,
        device: torch.device,
        query_slots: int | None = None,
        static_slots: int = 0,
    ) -> torch.Tensor:
        """query frame이 미래 frame token을 보지 못하게 하는 attention mask.

        `query_slots`를 주면 query 축만 그 수로 줄이고 key 축은 `total_token_slots`
        전체를 유지한다. 그 뒤에 정지 객체 `static_slots`개가 프레임마다 key로 붙는다.
        반환 shape는 `(T*query_slots, T*total_token_slots + T*static_slots)`이다.
        """
        if total_frames <= 0:
            raise ValueError("total_frames는 0보다 커야 한다")
        if total_token_slots <= 0:
            raise ValueError("total_token_slots는 0보다 커야 한다")
        if query_slots is None:
            query_slots = total_token_slots
        if query_slots <= 0:
            raise ValueError("query_slots는 0보다 커야 한다")
        if static_slots < 0:
            raise ValueError("static_slots는 음수일 수 없다")
        arange = torch.arange(total_frames, device=device)
        query_frames = arange.repeat_interleave(query_slots)
        key_frames = arange.repeat_interleave(total_token_slots)
        if static_slots > 0:
            key_frames = torch.cat([key_frames, arange.repeat_interleave(static_slots)])
        # PyTorch MultiheadAttention의 bool attn_mask는 True가 차단을 뜻한다.
        return key_frames.unsqueeze(0) > query_frames.unsqueeze(1)

    def _run_transformer(
        self,
        model_input: torch.Tensor,
        *,
        type_ids: torch.Tensor | None,
        num_object_slots: int,
    ) -> torch.Tensor:
        """predictor transformer를 돌린다. 정지 객체는 query에서 빼고 key/value로만 둔다.

        `model_input`은 `(B, T, S_total, D)`이고 S_total = object slot + action token이다.
        `type_ids`가 `(B, N_object)`로 주어지면 unit이 아닌 slot을 정지 객체로 보고
        query에서 제외한다. 없으면 전체를 query로 두는 이전 동작이다.

        attention 비용이 `L_query x L_key`이므로 slot의 대부분(실측 89%)인 지형을
        query에서 빼면 10~20배 줄어든다. `STATIC_TERRAIN_KV=0`으로 끄면 이전 경로다.

        정지 slot의 출력은 입력값을 그대로 통과시킨다. 손실이 unit slot만 보고
        rollout이 지형·임무를 마지막 관측값으로 채우므로 쓰이지 않는다.
        """
        batch_size, total_frames, total_token_slots, embedding_dim = model_input.shape
        static_index: torch.Tensor | None = None
        if STATIC_TERRAIN_KV and type_ids is not None:
            # 배치 안 slot layout이 같다는 것은 _validate_batch_layout이 보장한다.
            #
            # 임무는 query에 남긴다. mission feature에 time_remaining_ratio(매 프레임
            # 감소)와 completion_flag(전투 결과에 따라 바뀜)가 있어 정지 객체가 아니다.
            # slot이 하나뿐이라 query에 둬도 비용이 6%(강남역 144 -> 153)만 늘고,
            # 지형 140개를 빼는 효과에 비하면 무시할 수준이다.
            is_dynamic = (type_ids[0] == int(ObjectType.UNIT)) | (
                type_ids[0] == int(ObjectType.MISSION)
            )
            is_query = torch.ones(total_token_slots, dtype=torch.bool, device=model_input.device)
            is_query[:num_object_slots] = is_dynamic
            if not bool(is_query.all()):
                static_index = torch.nonzero(~is_query, as_tuple=False).flatten()
                query_index = torch.nonzero(is_query, as_tuple=False).flatten()

        if static_index is None:
            flat_input = model_input.reshape(batch_size, total_frames * total_token_slots, embedding_dim)
            attn_mask = self.temporal_attention_mask(
                total_frames=total_frames,
                total_token_slots=total_token_slots,
                device=flat_input.device,
            )
            flat_output = self.transformer(flat_input, attn_mask=attn_mask)
            return flat_output.reshape(batch_size, total_frames, total_token_slots, embedding_dim)

        num_query = int(query_index.numel())
        num_static = int(static_index.numel())
        query_input = model_input.index_select(2, query_index).reshape(
            batch_size, total_frames * num_query, embedding_dim
        )
        static_input = model_input.index_select(2, static_index).reshape(
            batch_size, total_frames * num_static, embedding_dim
        )
        attn_mask = self.temporal_attention_mask(
            total_frames=total_frames,
            total_token_slots=num_query,
            device=model_input.device,
            query_slots=num_query,
            static_slots=num_static,
        )
        query_output = self.transformer(
            query_input, attn_mask=attn_mask, static_context=static_input
        ).reshape(batch_size, total_frames, num_query, embedding_dim)

        output = model_input.clone()
        output[:, :, query_index] = query_output
        return output

    def _validate_predictor_inputs(
        self,
        history_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """masked predictor 입력 shape 계약을 확인하고 핵심 크기를 반환한다."""
        _expect_rank("history_tokens", history_tokens, 4)
        _expect_rank("type_ids", type_ids, 2)
        _expect_rank("team_ids", team_ids, 2)
        _expect_rank("action_tokens", action_tokens, 4)
        batch_size, history_frames, num_slots, embedding_dim = history_tokens.shape
        if history_frames != self.config.history_frames:
            raise ValueError(f"history_tokens 시간 길이는 {self.config.history_frames}이어야 한다")
        if embedding_dim != self.config.embedding_dim:
            raise ValueError(f"history_tokens embedding 차원은 {self.config.embedding_dim}이어야 한다")
        if type_ids.shape != (batch_size, num_slots):
            raise ValueError("type_ids shape는 (B, N)이어야 한다")
        if team_ids.shape != (batch_size, num_slots):
            raise ValueError("team_ids shape는 (B, N)이어야 한다")
        if action_tokens.shape[:2] != (batch_size, self.total_frames):
            raise ValueError("action_tokens shape 앞쪽은 (B, T_total)이어야 한다")
        if action_tokens.shape[-1] != embedding_dim:
            raise ValueError("action_tokens 마지막 차원은 history token embedding 차원과 같아야 한다")
        num_action_tokens = action_tokens.shape[2]
        if num_action_tokens <= 0:
            raise ValueError("action token 수는 1개 이상이어야 한다")
        return batch_size, history_frames, num_slots, embedding_dim, num_action_tokens

    def _prepare_input_from_mask(
        self,
        history_tokens: torch.Tensor,
        *,
        action_tokens: torch.Tensor,
        masked_slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """주어진 object mask로 실제 history token과 query token을 섞는다."""
        _expect_bool("masked_slot_mask", masked_slot_mask)
        batch_size, history_frames, num_slots, embedding_dim = history_tokens.shape
        if masked_slot_mask.shape != (batch_size, num_slots):
            raise ValueError("masked_slot_mask shape는 (B, N)이어야 한다")
        num_action_tokens = action_tokens.shape[2]
        anchors = history_tokens[:, 0]
        anchor_queries = self.anchor_projector(anchors)
        query_grid = self.mask_token.expand(batch_size, self.total_frames, num_slots, embedding_dim)
        query_grid = query_grid + self.time_pos_embedding.expand(batch_size, self.total_frames, num_slots, embedding_dim)
        query_grid = query_grid + anchor_queries.unsqueeze(1).expand(
            batch_size,
            self.total_frames,
            num_slots,
            embedding_dim,
        )

        model_input = query_grid.clone()

        # t=0은 모든 객체의 실제 token을 보여준다. 이것이 객체 identity anchor다.
        model_input[:, 0] = history_tokens[:, 0] + self.time_pos_embedding[:, 0]

        if history_frames > 1:
            history_pos = self.time_pos_embedding[:, 1:history_frames].expand(
                batch_size,
                history_frames - 1,
                num_slots,
                embedding_dim,
            )
            visible_history = (~masked_slot_mask).unsqueeze(1).unsqueeze(-1)
            actual_history = history_tokens[:, 1:] + history_pos
            model_input[:, 1:history_frames] = torch.where(
                visible_history,
                actual_history,
                model_input[:, 1:history_frames],
            )

        # action은 객체 query에 섞지 않고 별도 visible token으로 둔다. 미래 객체 query가
        # 이 token들을 attention으로 참조해야만 action-conditioned 예측이 된다.
        action_pos = self.time_pos_embedding.expand(
            batch_size,
            self.total_frames,
            num_action_tokens,
            embedding_dim,
        )
        action_input = action_tokens + action_pos
        return torch.cat([model_input, action_input], dim=2)

    def prepare_input(
        self,
        history_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """실제 history token과 C-JEPA query token을 섞은 입력 격자를 만든다."""
        self._validate_predictor_inputs(
            history_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            action_tokens=action_tokens,
        )

        masked_slot_mask, masked_indices = select_cjepa_masked_slots(
            type_ids=type_ids,
            team_ids=team_ids,
            maskable_type_ids=self.config.maskable_type_ids,
            num_masked_slots=self.config.num_masked_slots,
            mask_team_strategy=self.config.mask_team_strategy,
            mask_count_min=self.config.mask_count_min,
            mask_count_max=self.config.mask_count_max,
            rng=self._mask_rng,
        )
        model_input = self._prepare_input_from_mask(
            history_tokens,
            action_tokens=action_tokens,
            masked_slot_mask=masked_slot_mask,
        )
        return model_input, masked_indices, masked_slot_mask

    def prepare_input_with_masked_indices(
        self,
        history_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        action_tokens: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """평가용: 지정한 object slot index만 history t>=1에서 숨긴다."""
        batch_size, _, num_slots, _, _ = self._validate_predictor_inputs(
            history_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            action_tokens=action_tokens,
        )
        _expect_rank("masked_indices", masked_indices, 1)
        masked_indices = masked_indices.to(device=history_tokens.device, dtype=torch.long)
        if masked_indices.numel() > 0:
            if int(masked_indices.min().detach().cpu().item()) < 0:
                raise ValueError("masked_indices에 음수 slot index가 있다")
            if int(masked_indices.max().detach().cpu().item()) >= num_slots:
                raise ValueError("masked_indices에 slot 개수보다 큰 index가 있다")
        masked_slot_mask = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=history_tokens.device)
        masked_slot_mask[:, masked_indices] = True
        model_input = self._prepare_input_from_mask(
            history_tokens,
            action_tokens=action_tokens,
            masked_slot_mask=masked_slot_mask,
        )
        return model_input, masked_indices, masked_slot_mask

    def forward(
        self,
        history_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """masked history와 future token을 함께 예측한다."""
        model_input, masked_indices, masked_slot_mask = self.prepare_input(
            history_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            action_tokens=action_tokens,
        )
        num_object_slots = history_tokens.shape[2]
        output = self._run_transformer(
            model_input, type_ids=type_ids, num_object_slots=num_object_slots
        )
        output = self.output_projection(output)
        return output[:, :, :num_object_slots], masked_indices, masked_slot_mask

    def forward_with_masked_indices(
        self,
        history_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        action_tokens: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """평가용: 지정한 object slot을 숨긴 뒤 masked history/future를 예측한다."""
        model_input, masked_indices, masked_slot_mask = self.prepare_input_with_masked_indices(
            history_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            action_tokens=action_tokens,
            masked_indices=masked_indices,
        )
        num_object_slots = history_tokens.shape[2]
        output = self._run_transformer(
            model_input, type_ids=type_ids, num_object_slots=num_object_slots
        )
        output = self.output_projection(output)
        return output[:, :, :num_object_slots], masked_indices, masked_slot_mask

    @torch.no_grad()
    def inference(
        self,
        history_tokens: torch.Tensor,
        *,
        action_tokens: torch.Tensor,
        type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """planning 때 쓰는 비마스킹 future prediction 경로.

        `type_ids`(B, N)를 주면 정지 객체를 query에서 빼 attention 비용을 줄인다.
        """
        _expect_rank("history_tokens", history_tokens, 4)
        _expect_rank("action_tokens", action_tokens, 4)
        batch_size, history_frames, num_slots, embedding_dim = history_tokens.shape
        if history_frames != self.config.history_frames:
            raise ValueError(f"history_tokens 시간 길이는 {self.config.history_frames}이어야 한다")
        if embedding_dim != self.config.embedding_dim:
            raise ValueError(f"history_tokens embedding 차원은 {self.config.embedding_dim}이어야 한다")
        if action_tokens.shape[:2] != (batch_size, self.total_frames):
            raise ValueError("action_tokens shape 앞쪽은 (B, T_total)이어야 한다")
        if action_tokens.shape[-1] != embedding_dim:
            raise ValueError("action_tokens 마지막 차원은 history token embedding 차원과 같아야 한다")
        num_action_tokens = action_tokens.shape[2]
        if num_action_tokens <= 0:
            raise ValueError("action token 수는 1개 이상이어야 한다")

        anchors = history_tokens[:, 0]
        anchor_queries = self.anchor_projector(anchors)
        future_frames = self.config.pred_frames
        future_query = self.mask_token.expand(batch_size, future_frames, num_slots, embedding_dim)
        future_query = future_query + self.time_pos_embedding[:, history_frames:self.total_frames].expand(
            batch_size,
            future_frames,
            num_slots,
            embedding_dim,
        )
        future_query = future_query + anchor_queries.unsqueeze(1).expand(
            batch_size,
            future_frames,
            num_slots,
            embedding_dim,
        )
        history_input = history_tokens + self.time_pos_embedding[:, :history_frames].expand(
            batch_size,
            history_frames,
            num_slots,
            embedding_dim,
        )
        object_input = torch.cat([history_input, future_query], dim=1)
        action_pos = self.time_pos_embedding.expand(
            batch_size,
            self.total_frames,
            num_action_tokens,
            embedding_dim,
        )
        action_input = action_tokens + action_pos
        model_input = torch.cat([object_input, action_input], dim=2)
        output = self._run_transformer(
            model_input, type_ids=type_ids, num_object_slots=num_slots
        )
        output = self.output_projection(output)
        return output[:, history_frames:self.total_frames, :num_slots]


def _enforce_unit_physics(
    future_features: torch.Tensor,
    *,
    last_observed: torch.Tensor,
    type_ids: torch.Tensor,
) -> torch.Tensor:
    """예측된 unit 궤적에 이동 한계와 전사자 고정을 씌운다.

    월드모델은 6프레임을 한 번에 내놓을 뿐 프레임 간 연속성이나 이동 한계를 모른다.
    실측에서 프레임 간 이동이 중앙 2.5m인데 최대 24.9m로 한 스텝 한계 15m를 넘겼고,
    hp=0인 전사자가 6스텝에 44m를 움직였다.

    두 가지를 강제한다.
      1. 프레임 간 이동을 `MAX_MOVE_PER_STEP_UNITS`로 제한한다. 방향은 살리고 크기만
         줄이므로 예측이 가리키는 쪽은 유지된다.
      2. hp<=0이면 그 자리에 고정한다. 이미 죽어 있었으면 마지막 관측 위치에, 예측
         도중 죽으면 죽은 프레임 위치에 멈춘다.

    학습 손실은 `predict_cjepa_sequence`를 쓰므로 여기 변경이 gradient에 안 섞인다.
    planning과 화면 표시에만 적용된다.
    """
    _expect_rank("future_features", future_features, 4)
    _expect_rank("last_observed", last_observed, 3)
    is_unit = (type_ids[:, 0] == int(ObjectType.UNIT)).unsqueeze(-1)   # (B, N, 1)
    span = torch.tensor(
        [0.5 * (WORLD_X_MAX - WORLD_X_MIN), 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN)],
        dtype=future_features.dtype,
        device=future_features.device,
    )
    position_index = [UNIT_X_FEATURE_INDEX, UNIT_Y_FEATURE_INDEX]

    frames = []
    previous = last_observed[..., position_index]                      # (B, N, 2)
    alive = last_observed[..., UNIT_HP_FEATURE_INDEX] > 0.0            # (B, N)
    for step in range(future_features.shape[1]):
        frame = future_features[:, step]
        target = frame[..., position_index]
        # 이동 한계: 월드 단위로 재서 넘치면 방향을 유지한 채 줄인다.
        delta_world = (target - previous) * span
        distance = delta_world.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = (MAX_MOVE_PER_STEP_UNITS / distance).clamp(max=1.0)
        clamped = previous + delta_world * scale / span

        # 전사자 고정: 이번 프레임에 죽어 있거나 이미 죽었으면 직전 위치를 유지한다.
        alive = alive & (frame[..., UNIT_HP_FEATURE_INDEX] > 0.0)
        moved = torch.where(alive.unsqueeze(-1) & is_unit, clamped, previous)
        # unit이 아닌 slot은 손대지 않는다.
        position = torch.where(is_unit, moved, target)

        frame = frame.clone()
        frame[..., UNIT_X_FEATURE_INDEX] = position[..., 0]
        frame[..., UNIT_Y_FEATURE_INDEX] = position[..., 1]
        frames.append(frame)
        previous = position
    return torch.stack(frames, dim=1)


def _apply_position_residual(
    future_features: torch.Tensor,
    *,
    delta: torch.Tensor,
    last_observed: torch.Tensor,
    type_ids: torch.Tensor,
) -> torch.Tensor:
    """유닛 x/y를 "마지막 관측 위치 + 예측 변화량"으로 바꾼다.

    지형에 이미 쓰고 있는 논리를 유닛에도 적용하는 것이다. 다만 지형은 아예 정지라
    마지막 값을 그대로 쓰고, 유닛은 움직이므로 변화량만 예측하게 한다.

    `future_features`는 `(B, F, N, MAX_FEATURE_DIM)`, `delta`는 `(B, F, N, 2)`,
    `last_observed`는 `(B, N, MAX_FEATURE_DIM)`이다. unit이 아닌 slot은 손대지 않는다.
    """
    _expect_rank("future_features", future_features, 4)
    _expect_rank("delta", delta, 4)
    _expect_rank("last_observed", last_observed, 3)
    if delta.shape[-1] != 2:
        raise ValueError("delta 마지막 차원은 2(Δx, Δy)여야 한다")
    if delta.shape[:3] != future_features.shape[:3]:
        raise ValueError("delta와 future_features의 (B, F, N)이 같아야 한다")

    position_index = [UNIT_X_FEATURE_INDEX, UNIT_Y_FEATURE_INDEX]
    is_unit = (type_ids == int(ObjectType.UNIT)).unsqueeze(-1)
    anchor = last_observed[..., position_index].unsqueeze(1)          # (B, 1, N, 2)
    residual = anchor + delta
    absolute = future_features[..., position_index]
    position = torch.where(is_unit, residual, absolute)

    updated = future_features.clone()
    updated[..., UNIT_X_FEATURE_INDEX] = position[..., 0]
    updated[..., UNIT_Y_FEATURE_INDEX] = position[..., 1]
    return updated


def cjepa_prediction_loss(
    *,
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    masked_indices: torch.Tensor,
    history_frames: int,
    pred_frames: int,
    type_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """C-JEPA 학습 손실을 계산한다.

    DEVS slot은 entity id로 정렬되므로 Hungarian matching 없이 같은 index끼리
    비교한다. 손실은 숨겨진 history slot 복원과 future slot 예측을 분리해 낸다.

    `type_ids`(B, N)를 주면 future 손실을 **unit slot으로 제한한다.** 지형과 임무는
    정지 물체라 미래가 이미 알려져 있고, 예측 오차가 사실상 0이다. 그런데
    `F.mse_loss`가 전체 원소 수로 나누므로 이들이 평균에 들어가면 유닛 항의 gradient가
    `N_unit / N_전체`만큼 줄어든다. 실측 희석 배율:

        서울과기대(장애물 76) 10v10   4.8배
        성수역(장애물 224) 2v2       57.2배

    단순히 작아지는 게 아니라 **에피소드마다 맵과 팀 크기가 바뀌어 12배 범위로
    널뛴다.** 배치마다 유닛에 걸리는 실효 학습률이 달라진다는 뜻이다.

    지표도 같은 이유로 망가진다. `loss_future`는 지형 비율에 좌우되어 유닛 예측
    정확도를 나타내지 못한다. 검증 지표로 `loss_masked_history_state`가 잘 작동했던
    (신호/잡음 7.1 대 1.3) 이유가 이것이다 — 그쪽은 masked_indices로 유닛만 본다.

    하위 호환을 위해 `type_ids`가 없으면 이전처럼 전체 slot을 쓴다.
    """
    _expect_rank("pred_tokens", pred_tokens, 4)
    _expect_rank("target_tokens", target_tokens, 4)
    _expect_rank("masked_indices", masked_indices, 1)
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError("pred_tokens와 target_tokens shape가 같아야 한다")
    if pred_tokens.shape[1] != history_frames + pred_frames:
        raise ValueError("pred_tokens 시간 길이는 history_frames + pred_frames와 같아야 한다")

    if masked_indices.numel() > 0:
        masked_history_loss = F.mse_loss(
            pred_tokens[:, :history_frames, masked_indices],
            target_tokens[:, :history_frames, masked_indices].detach(),
        )
    else:
        masked_history_loss = pred_tokens.new_zeros(())

    future_pred = pred_tokens[:, history_frames:history_frames + pred_frames]
    future_target = target_tokens[:, history_frames:history_frames + pred_frames].detach()
    if type_ids is not None:
        _expect_rank("type_ids", type_ids, 2)
        if type_ids.shape[0] != pred_tokens.shape[0] or type_ids.shape[1] != pred_tokens.shape[2]:
            raise ValueError("type_ids shape는 (B, N)이어야 한다")
        # 임무도 예측 대상이라 포함한다 — time_remaining_ratio와 completion_flag가
        # 프레임마다 변한다. slot 1개라 희석은 무시할 수준이다.
        is_dynamic = (
            (type_ids == int(ObjectType.UNIT)) | (type_ids == int(ObjectType.MISSION))
        ).reshape(type_ids.shape[0], 1, type_ids.shape[1], 1)
        weight = is_dynamic.to(future_pred.dtype)
        total = weight.sum() * future_pred.shape[1] * future_pred.shape[-1]
        if float(total) <= 0.0:
            raise ValueError("unit/mission slot이 하나도 없어 future 손실을 낼 수 없다")
        future_loss = (((future_pred - future_target) ** 2) * weight).sum() / total
    else:
        future_loss = F.mse_loss(future_pred, future_target)
    return {
        "loss_masked_history": masked_history_loss,
        "loss_future": future_loss,
        "loss": masked_history_loss + future_loss,
    }


class JointActionConditioner(nn.Module):
    """유닛별 DEVS action을 latent token으로 인코딩한다."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.embedding_dim
        self.action_encoder = _mlp(ACTION_DIM, config.hidden_dim, dim, config.dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, dim),
        )
        self.action_token_norm = nn.LayerNorm(dim)

    def build_action_tokens(
        self,
        *,
        source_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> torch.Tensor:
        """joint action을 per-unit visible action token으로 변환한다.

        action token은 객체 slot과 분리되어 predictor slot 축 뒤에 붙는다. 각 token은
        DEVS 명령 임베딩과 그 명령을 실행하는 unit의 anchor latent를 함께 담는다.
        """
        _expect_rank("source_tokens", source_tokens, 3)
        _expect_rank("type_ids", type_ids, 2)
        _expect_rank("team_ids", team_ids, 2)
        _expect_rank("entity_ids", entity_ids, 2)
        _expect_rank("action_features", action_features, 3)
        _expect_rank("action_unit_ids", action_unit_ids, 2)
        _expect_rank("issued_mask", issued_mask, 2)
        _expect_bool("issued_mask", issued_mask)
        batch_size, num_slots, embedding_dim = source_tokens.shape
        if type_ids.shape != (batch_size, num_slots):
            raise ValueError("type_ids shape는 source_tokens의 (B, N)과 같아야 한다")
        if team_ids.shape != (batch_size, num_slots):
            raise ValueError("team_ids shape는 source_tokens의 (B, N)과 같아야 한다")
        if entity_ids.shape != (batch_size, num_slots):
            raise ValueError("entity_ids shape는 source_tokens의 (B, N)과 같아야 한다")
        if action_features.shape[:2] != action_unit_ids.shape:
            raise ValueError("action_features와 action_unit_ids의 batch/action 차원이 같아야 한다")
        if issued_mask.shape != action_unit_ids.shape:
            raise ValueError("issued_mask와 action_unit_ids shape가 같아야 한다")
        if action_features.shape[0] != batch_size or action_features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action_features shape는 (B, U, {ACTION_DIM})이어야 한다")
        _ensure_known_type_ids(type_ids)
        _team_embedding_indices(team_ids)

        action_tokens = self.action_encoder(action_features.float())
        source_by_action = source_tokens.new_zeros(action_tokens.shape)

        for batch_index in range(batch_size):
            seen_unit_ids: set[int] = set()
            for action_index in range(action_unit_ids.shape[1]):
                if not bool(issued_mask[batch_index, action_index].detach().cpu().item()):
                    continue
                unit_id = int(action_unit_ids[batch_index, action_index].detach().cpu().item())
                if unit_id in seen_unit_ids:
                    raise ValueError(f"같은 batch에서 unit {unit_id} action이 중복됐다")
                seen_unit_ids.add(unit_id)

                matched = torch.nonzero(
                    (entity_ids[batch_index] == unit_id)
                    & (type_ids[batch_index] == int(ObjectType.UNIT)),
                    as_tuple=False,
                ).flatten()
                if matched.numel() != 1:
                    raise ValueError(f"action 대상 unit {unit_id}에 해당하는 slot이 정확히 하나여야 한다")
                slot_index = int(matched[0].detach().cpu().item())
                if int(type_ids[batch_index, slot_index].detach().cpu().item()) != int(ObjectType.UNIT):
                    raise ValueError(f"action 대상 {unit_id}는 UNIT slot이어야 한다")
                team_id = int(team_ids[batch_index, slot_index].detach().cpu().item())
                if team_id not in (int(TeamId.BLUE), int(TeamId.RED)):
                    raise ValueError(f"action 대상 {unit_id}는 BLUE/RED unit이어야 한다")
                source_by_action[batch_index, action_index] = source_tokens[batch_index, slot_index]

        visible = issued_mask.unsqueeze(-1)
        action_tokens = self.action_token_norm(action_tokens + source_by_action)
        action_tokens = torch.where(visible, action_tokens, torch.zeros_like(action_tokens))
        if action_tokens.shape[-1] != embedding_dim:
            raise ValueError("action token embedding 차원이 source_tokens와 다르다")
        return action_tokens

    def build_action_grid(
        self,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> torch.Tensor:
        """legacy one-step dynamics용으로 action을 객체 slot 순서 grid에 맞춘다."""
        _expect_rank("type_ids", type_ids, 2)
        _expect_rank("team_ids", team_ids, 2)
        _expect_rank("entity_ids", entity_ids, 2)
        _expect_rank("action_features", action_features, 3)
        _expect_rank("action_unit_ids", action_unit_ids, 2)
        _expect_rank("issued_mask", issued_mask, 2)
        _expect_bool("issued_mask", issued_mask)
        batch_size, num_slots = type_ids.shape
        _ensure_known_type_ids(type_ids)
        _team_embedding_indices(team_ids)
        if team_ids.shape != (batch_size, num_slots):
            raise ValueError("team_ids shape는 (B, N)이어야 한다")
        if entity_ids.shape != (batch_size, num_slots):
            raise ValueError("entity_ids shape는 (B, N)이어야 한다")
        if action_features.shape[:2] != action_unit_ids.shape:
            raise ValueError("action_features와 action_unit_ids의 batch/action 차원이 같아야 한다")
        if issued_mask.shape != action_unit_ids.shape:
            raise ValueError("issued_mask와 action_unit_ids shape가 같아야 한다")
        if action_features.shape[0] != batch_size or action_features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action_features shape는 (B, U, {ACTION_DIM})이어야 한다")

        action_tokens = self.action_encoder(action_features.float())
        action_by_slot = action_tokens.new_zeros((batch_size, num_slots, action_tokens.shape[-1]))

        for batch_index in range(batch_size):
            seen_unit_ids: set[int] = set()
            for action_index in range(action_unit_ids.shape[1]):
                if not bool(issued_mask[batch_index, action_index].detach().cpu().item()):
                    continue
                unit_id = int(action_unit_ids[batch_index, action_index].detach().cpu().item())
                if unit_id in seen_unit_ids:
                    raise ValueError(f"같은 batch에서 unit {unit_id} action이 중복됐다")
                seen_unit_ids.add(unit_id)

                matched = torch.nonzero(entity_ids[batch_index] == unit_id, as_tuple=False).flatten()
                if matched.numel() != 1:
                    raise ValueError(f"action 대상 unit {unit_id}에 해당하는 slot이 정확히 하나여야 한다")
                slot_index = int(matched[0].detach().cpu().item())
                if int(type_ids[batch_index, slot_index].detach().cpu().item()) != int(ObjectType.UNIT):
                    raise ValueError(f"action 대상 {unit_id}는 UNIT slot이어야 한다")
                if int(team_ids[batch_index, slot_index].detach().cpu().item()) != int(TeamId.BLUE):
                    raise ValueError(f"action 대상 {unit_id}는 BLUE unit이어야 한다")
                action_by_slot[batch_index, slot_index] = action_tokens[batch_index, action_index]

        return action_by_slot

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> torch.Tensor:
        """legacy one-step dynamics에서만 각 유닛 action을 자신의 unit slot에 주입한다."""
        _expect_rank("tokens", tokens, 3)
        batch_size, num_slots, embedding_dim = tokens.shape
        action_by_slot = self.build_action_grid(
            type_ids=type_ids,
            team_ids=team_ids,
            entity_ids=entity_ids,
            action_features=action_features,
            action_unit_ids=action_unit_ids,
            issued_mask=issued_mask,
        )
        if action_by_slot.shape != (batch_size, num_slots, embedding_dim):
            raise ValueError("action_by_slot shape는 tokens shape와 같아야 한다")
        fused_delta = self.fusion(torch.cat([tokens, action_by_slot], dim=-1))
        return tokens + fused_delta


class ObjectStateDecoder(nn.Module):
    """예측 token을 다음 DEVS state feature로 복원한다."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.embedding_dim
        self.unit_decoder = _mlp(dim, config.hidden_dim, len(UNIT_FEATURE_NAMES), config.dropout)
        self.terrain_decoder = _mlp(dim, config.hidden_dim, len(TERRAIN_FEATURE_NAMES), config.dropout)
        self.mission_decoder = _mlp(dim, config.hidden_dim, len(MISSION_FEATURE_NAMES), config.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        type_ids: torch.Tensor,
        *,
        only_types: tuple[ObjectType, ...] | None = None,
    ) -> torch.Tensor:
        """타입별 decoder 출력만 padding feature의 앞쪽 차원에 채운다.

        `only_types`를 주면 그 타입만 디코딩하고 나머지 slot은 0으로 남긴다. 미래
        프레임에서 지형·임무를 건너뛰는 데 쓴다 — 정지 물체라 예측할 필요가 없고,
        호출부가 알고 있는 실제 값으로 채우는 편이 더 정확하다. 채우지 않고 0으로
        두면 안 된다: value head가 mission slot에서 목표 좌표를 읽는다.
        """
        _expect_rank("tokens", tokens, 3)
        _expect_rank("type_ids", type_ids, 2)
        if tokens.shape[:2] != type_ids.shape:
            raise ValueError("tokens와 type_ids의 batch, slot 차원이 같아야 한다")
        _ensure_known_type_ids(type_ids)

        batch_size, num_slots, _ = tokens.shape
        decoded = tokens.new_zeros((batch_size, num_slots, MAX_FEATURE_DIM))
        decoder_specs = (
            (ObjectType.UNIT, len(UNIT_FEATURE_NAMES), self.unit_decoder),
            (ObjectType.TERRAIN, len(TERRAIN_FEATURE_NAMES), self.terrain_decoder),
            (ObjectType.MISSION, len(MISSION_FEATURE_NAMES), self.mission_decoder),
        )
        for object_type, feature_dim, decoder in decoder_specs:
            if only_types is not None and object_type not in only_types:
                continue
            selected = type_ids == int(object_type)
            if torch.any(selected):
                decoded[selected, :feature_dim] = decoder(tokens[selected])
        return decoded


class SlotSelfStateDecoder(nn.Module):
    """slot latent 앞쪽 self_state_dim만 사용해 현재 자기 속성을 복원한다."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        dim = config.self_state_dim
        self.self_state_dim = dim
        self.unit_decoder = _mlp(dim, config.hidden_dim, len(UNIT_FEATURE_NAMES), config.dropout)
        self.terrain_decoder = _mlp(dim, config.hidden_dim, len(TERRAIN_FEATURE_NAMES), config.dropout)
        self.mission_decoder = _mlp(dim, config.hidden_dim, len(MISSION_FEATURE_NAMES), config.dropout)

    def forward(self, tokens: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """관계/전술 차원을 건드리지 않고 자기 상태 전용 latent slice만 읽는다."""
        _expect_rank("tokens", tokens, 3)
        _expect_rank("type_ids", type_ids, 2)
        if tokens.shape[:2] != type_ids.shape:
            raise ValueError("tokens와 type_ids의 batch, slot 차원이 같아야 한다")
        if tokens.shape[-1] < self.self_state_dim:
            raise ValueError("tokens 마지막 차원이 self_state_dim보다 작다")
        _ensure_known_type_ids(type_ids)

        self_tokens = tokens[..., : self.self_state_dim]
        batch_size, num_slots, _ = tokens.shape
        decoded = tokens.new_zeros((batch_size, num_slots, MAX_FEATURE_DIM))
        decoder_specs = (
            (ObjectType.UNIT, len(UNIT_FEATURE_NAMES), self.unit_decoder),
            (ObjectType.TERRAIN, len(TERRAIN_FEATURE_NAMES), self.terrain_decoder),
            (ObjectType.MISSION, len(MISSION_FEATURE_NAMES), self.mission_decoder),
        )
        for object_type, feature_dim, decoder in decoder_specs:
            selected = type_ids == int(object_type)
            if torch.any(selected):
                decoded[selected, :feature_dim] = decoder(self_tokens[selected])
        return decoded


class DEVSObjectCentricWorldModel(nn.Module):
    """DEVS 객체 슬롯용 action-conditioned JEPA 월드모델."""

    def __init__(self, config: ObjectSlotModelConfig):
        super().__init__()
        self.config = config
        dim = self.config.embedding_dim
        self.slot_encoder = (
            BlockObjectSlotEncoder(self.config) if PROPERTY_BLOCKS
            else TypedObjectSlotEncoder(self.config)
        )
        # action 전 context self-attention은 쓰지 않는다. slot_encoder는 객체별
        # raw DEVS feature를 같은 latent 공간으로 올리는 projection만 담당한다.
        # JEPA target용 EMA projection encoder. online encoder가 target을 함께
        # 공급하면 collapse 위험이 있으므로 target은 gradient가 흐르지 않는 EMA 사본이 만든다.
        self.target_slot_encoder = copy.deepcopy(self.slot_encoder)
        for parameter in self.target_slot_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_conditioner = JointActionConditioner(self.config)
        self.dynamics_predictor = ObjectSlotTransformer(
            embedding_dim=dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_predictor_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        # 유닛 위치 변화량(Δx, Δy) 전용 head. 마지막 층을 0으로 두어 학습 시작 시점의
        # 예측이 정확히 "마지막 관측 위치 그대로"가 되게 한다. 실측 정지 가정 오차
        # (RED t+1s 3.9m)에서 출발하므로, 현재 절대 좌표 예측(9.6m)보다 이미 낫다.
        # 블록을 쓰면 delta head도 위치 블록만 읽는다. 인코더·디코더와 같은 규약이라야
        # "위치는 이 블록에 산다"가 파이프라인 전체에서 일관된다.
        self._position_slice = block_layout(dim)["position"] if PROPERTY_BLOCKS else slice(0, dim)
        delta_input_dim = self._position_slice.stop - self._position_slice.start
        self.unit_position_delta = _mlp(delta_input_dim, self.config.hidden_dim, 2, self.config.dropout)
        final_delta_layer = self.unit_position_delta[-1]
        nn.init.zeros_(final_delta_layer.weight)
        nn.init.zeros_(final_delta_layer.bias)
        self.masked_predictor = CausalMaskedObjectPredictor(self.config)
        self.state_decoder = (
            BlockObjectStateDecoder(self.config) if PROPERTY_BLOCKS
            else ObjectStateDecoder(self.config)
        )
        self.self_state_decoder = SlotSelfStateDecoder(self.config)

    def encode_state(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """현재 DEVS state를 객체별 projection latent로 인코딩한다."""
        _expect_rank("alive_mask", alive_mask, 2)
        _expect_bool("alive_mask", alive_mask)
        if alive_mask.shape != type_ids.shape:
            raise ValueError("alive_mask shape는 type_ids shape와 같아야 한다")
        return self.slot_encoder(features, feature_mask, type_ids, team_ids)

    @torch.no_grad()
    def encode_state_target(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """EMA target projection encoder로 현재 state를 JEPA target token으로 인코딩한다."""
        _expect_rank("alive_mask", alive_mask, 2)
        _expect_bool("alive_mask", alive_mask)
        if alive_mask.shape != type_ids.shape:
            raise ValueError("alive_mask shape는 type_ids shape와 같아야 한다")
        return self.target_slot_encoder(features, feature_mask, type_ids, team_ids)

    @torch.no_grad()
    def update_target_encoders(self, momentum: float | None = None) -> None:
        """online encoder 파라미터를 EMA로 target encoder에 반영한다.

        학습 루프에서 optimizer step 직후마다 호출해야 한다. momentum이 None이면
        config.ema_momentum을 사용한다.
        """
        ema = self.config.ema_momentum if momentum is None else float(momentum)
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema momentum은 [0, 1) 범위여야 한다")
        pairs = ((self.slot_encoder, self.target_slot_encoder),)
        for online, target in pairs:
            for online_param, target_param in zip(online.parameters(), target.parameters(), strict=True):
                target_param.mul_(ema).add_(online_param.detach(), alpha=1.0 - ema)
            for online_buffer, target_buffer in zip(online.buffers(), target.buffers(), strict=True):
                target_buffer.copy_(online_buffer)

    def encode_state_sequence(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """시간축이 있는 DEVS state sequence를 token sequence로 인코딩한다."""
        return self._encode_state_sequence_with(
            self.encode_state,
            features=features,
            feature_mask=feature_mask,
            type_ids=type_ids,
            team_ids=team_ids,
            alive_mask=alive_mask,
        )

    @torch.no_grad()
    def encode_state_sequence_target(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """EMA target encoder로 state sequence를 JEPA target token으로 인코딩한다."""
        return self._encode_state_sequence_with(
            self.encode_state_target,
            features=features,
            feature_mask=feature_mask,
            type_ids=type_ids,
            team_ids=team_ids,
            alive_mask=alive_mask,
        )

    def _encode_state_sequence_with(
        self,
        encode_fn,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """지정한 encoder 경로로 state sequence를 인코딩하는 공통 구현."""
        _expect_rank("features", features, 4)
        _expect_rank("feature_mask", feature_mask, 4)
        _expect_rank("type_ids", type_ids, 3)
        _expect_rank("team_ids", team_ids, 3)
        _expect_rank("alive_mask", alive_mask, 3)
        _expect_bool("feature_mask", feature_mask)
        _expect_bool("alive_mask", alive_mask)
        if features.shape != feature_mask.shape:
            raise ValueError("features와 feature_mask shape가 같아야 한다")
        batch_size, num_frames, num_slots, feature_dim = features.shape
        expected_state_shape = (batch_size, num_frames, num_slots)
        if type_ids.shape != expected_state_shape:
            raise ValueError("type_ids shape는 (B, T, N)이어야 한다")
        if team_ids.shape != expected_state_shape:
            raise ValueError("team_ids shape는 (B, T, N)이어야 한다")
        if alive_mask.shape != expected_state_shape:
            raise ValueError("alive_mask shape는 (B, T, N)이어야 한다")
        if feature_dim != MAX_FEATURE_DIM:
            raise ValueError(f"features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")

        flat_tokens = encode_fn(
            features=features.reshape(batch_size * num_frames, num_slots, feature_dim),
            feature_mask=feature_mask.reshape(batch_size * num_frames, num_slots, feature_dim),
            type_ids=type_ids.reshape(batch_size * num_frames, num_slots),
            team_ids=team_ids.reshape(batch_size * num_frames, num_slots),
            alive_mask=alive_mask.reshape(batch_size * num_frames, num_slots),
        )
        return flat_tokens.reshape(batch_size, num_frames, num_slots, self.config.embedding_dim)

    @staticmethod
    def _expect_static_slot_metadata(name: str, value: torch.Tensor) -> None:
        """시간이 지나도 slot type/order가 고정되는지 확인한다."""
        _expect_rank(name, value, 3)
        reference = value[:, 0]
        for frame_index in range(1, value.shape[1]):
            if not torch.equal(value[:, frame_index], reference):
                raise ValueError(f"{name}는 시간축에서 변하면 안 된다: frame={frame_index}")

    def build_transition_action_tokens(
        self,
        *,
        source_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> torch.Tensor:
        """전이 action sequence를 C-JEPA predictor의 별도 action token으로 변환한다.

        state frame이 `T`개라면 action frame은 `T-1`개다. `action_features[:, k]`는
        `state[:, k] -> state[:, k+1]` 전이에 쓰인 DEVS 명령이며, action token은
        다음 state frame인 `k+1` 위치에서 visible condition으로 들어간다.
        """
        _expect_rank("source_tokens", source_tokens, 3)
        _expect_rank("type_ids", type_ids, 3)
        _expect_rank("team_ids", team_ids, 3)
        _expect_rank("entity_ids", entity_ids, 3)
        _expect_rank("action_features", action_features, 4)
        _expect_rank("action_unit_ids", action_unit_ids, 3)
        _expect_rank("issued_mask", issued_mask, 3)
        _expect_bool("issued_mask", issued_mask)
        self._expect_static_slot_metadata("type_ids", type_ids)
        self._expect_static_slot_metadata("team_ids", team_ids)
        self._expect_static_slot_metadata("entity_ids", entity_ids)

        batch_size, total_frames, num_slots = type_ids.shape
        if source_tokens.shape != (batch_size, num_slots, self.config.embedding_dim):
            raise ValueError("source_tokens shape는 (B, N, D)이어야 한다")
        if total_frames != self.config.history_frames + self.config.pred_frames:
            raise ValueError("type_ids 시간 길이는 history_frames + pred_frames와 같아야 한다")
        action_frames = total_frames - 1
        if action_features.shape[:2] != (batch_size, action_frames):
            raise ValueError("action_features shape는 (B, T-1, U, ACTION_DIM)이어야 한다")
        if action_features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action_features 마지막 차원은 {ACTION_DIM}이어야 한다")
        if action_unit_ids.shape != action_features.shape[:3]:
            raise ValueError("action_unit_ids shape는 (B, T-1, U)이어야 한다")
        if issued_mask.shape != action_unit_ids.shape:
            raise ValueError("issued_mask shape는 action_unit_ids shape와 같아야 한다")

        flat_source_tokens = source_tokens.unsqueeze(1).expand(
            batch_size,
            action_frames,
            num_slots,
            self.config.embedding_dim,
        ).reshape(batch_size * action_frames, num_slots, self.config.embedding_dim)
        flat_action_tokens = self.action_conditioner.build_action_tokens(
            source_tokens=flat_source_tokens,
            type_ids=type_ids[:, 1:].reshape(batch_size * action_frames, num_slots),
            team_ids=team_ids[:, 1:].reshape(batch_size * action_frames, num_slots),
            entity_ids=entity_ids[:, 1:].reshape(batch_size * action_frames, num_slots),
            action_features=action_features.reshape(batch_size * action_frames, action_features.shape[2], ACTION_DIM),
            action_unit_ids=action_unit_ids.reshape(batch_size * action_frames, action_unit_ids.shape[2]),
            issued_mask=issued_mask.reshape(batch_size * action_frames, issued_mask.shape[2]),
        )
        num_action_tokens = action_features.shape[2]
        action_tokens = flat_action_tokens.new_zeros(
            (batch_size, total_frames, num_action_tokens, self.config.embedding_dim)
        )
        action_tokens[:, 1:] = flat_action_tokens.reshape(
            batch_size,
            action_frames,
            num_action_tokens,
            self.config.embedding_dim,
        )
        return action_tokens

    def predict_cjepa_sequence(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """C-JEPA 방식으로 masked history와 future object token을 예측한다."""
        _expect_rank("features", features, 4)
        _expect_rank("feature_mask", feature_mask, 4)
        _expect_rank("type_ids", type_ids, 3)
        _expect_rank("entity_ids", entity_ids, 3)
        _expect_rank("team_ids", team_ids, 3)
        _expect_rank("alive_mask", alive_mask, 3)
        _expect_rank("action_features", action_features, 4)
        _expect_rank("action_unit_ids", action_unit_ids, 3)
        _expect_rank("issued_mask", issued_mask, 3)
        total_frames = self.config.history_frames + self.config.pred_frames
        if features.shape[1] != total_frames:
            raise ValueError(f"features 시간 길이는 {total_frames}이어야 한다")
        self._expect_static_slot_metadata("type_ids", type_ids)
        self._expect_static_slot_metadata("entity_ids", entity_ids)
        self._expect_static_slot_metadata("team_ids", team_ids)

        history_frames = self.config.history_frames
        # predictor 입력은 online encoder가 만든다. gradient는 이 경로로만 흐른다.
        history_tokens = self.encode_state_sequence(
            features=features[:, :history_frames],
            feature_mask=feature_mask[:, :history_frames],
            type_ids=type_ids[:, :history_frames],
            team_ids=team_ids[:, :history_frames],
            alive_mask=alive_mask[:, :history_frames],
        )
        # 손실 target은 EMA encoder가 만든다. collapse 방지의 핵심 분리다.
        target_tokens = self.encode_state_sequence_target(
            features=features,
            feature_mask=feature_mask,
            type_ids=type_ids,
            team_ids=team_ids,
            alive_mask=alive_mask,
        )
        action_tokens = self.build_transition_action_tokens(
            source_tokens=history_tokens[:, 0],
            type_ids=type_ids,
            team_ids=team_ids,
            entity_ids=entity_ids,
            action_features=action_features,
            action_unit_ids=action_unit_ids,
            issued_mask=issued_mask,
        )
        pred_tokens, masked_indices, masked_slot_mask = self.masked_predictor(
            history_tokens,
            type_ids=type_ids[:, 0],
            team_ids=team_ids[:, 0],
            action_tokens=action_tokens,
        )
        losses = cjepa_prediction_loss(
            pred_tokens=pred_tokens,
            target_tokens=target_tokens,
            masked_indices=masked_indices,
            history_frames=self.config.history_frames,
            pred_frames=self.config.pred_frames,
            type_ids=type_ids[:, 0],
        )
        pred_features = self.state_decoder(
            pred_tokens.reshape(pred_tokens.shape[0] * pred_tokens.shape[1], pred_tokens.shape[2], pred_tokens.shape[3]),
            type_ids[:, 0].unsqueeze(1).expand(type_ids.shape[0], total_frames, type_ids.shape[2]).reshape(
                type_ids.shape[0] * total_frames,
                type_ids.shape[2],
            ),
        ).reshape(features.shape[0], total_frames, features.shape[2], MAX_FEATURE_DIM)
        if POSITION_RESIDUAL:
            # 미래 프레임만 잔차로 바꾼다. history 구간 출력은 masked-state loss용이라
            # "마지막 관측 프레임"을 기준으로 삼을 수 없다.
            #
            # rollout_cjepa_future와 같은 식이어야 한다. 학습과 추론이 어긋나면 지금까지
            # 겪은 것과 같은 종류의 조용한 불일치가 된다.
            future_updated = _apply_position_residual(
                pred_features[:, history_frames:total_frames],
                delta=self.unit_position_delta(
                    pred_tokens[:, history_frames:total_frames, :, self._position_slice]
                ),
                last_observed=features[:, history_frames - 1],
                type_ids=type_ids[:, 0].unsqueeze(1).expand(
                    type_ids.shape[0], self.config.pred_frames, type_ids.shape[2]
                ),
            )
            pred_features = torch.cat(
                [pred_features[:, :history_frames], future_updated], dim=1
            )
        # 이 출력은 slot self-state loss 전용이다. 입력이 history_tokens뿐이라
        # loss를 단독으로 걸면 predictor/action 경로에는 gradient가 흐르지 않는다.
        history_self_features = self.self_state_decoder(
            history_tokens.reshape(
                history_tokens.shape[0] * history_tokens.shape[1],
                history_tokens.shape[2],
                history_tokens.shape[3],
            ),
            type_ids[:, :history_frames].reshape(type_ids.shape[0] * history_frames, type_ids.shape[2]),
        ).reshape(features.shape[0], history_frames, features.shape[2], MAX_FEATURE_DIM)
        return {
            "target_tokens": target_tokens,
            "history_tokens": history_tokens,
            "action_tokens": action_tokens,
            "pred_tokens": pred_tokens,
            "pred_features": pred_features,
            "history_self_features": history_self_features,
            "masked_indices": masked_indices,
            "masked_slot_mask": masked_slot_mask,
            **losses,
        }

    def rollout_cjepa_future(
        self,
        *,
        history_features: torch.Tensor,
        history_feature_mask: torch.Tensor,
        history_type_ids: torch.Tensor,
        history_entity_ids: torch.Tensor,
        history_team_ids: torch.Tensor,
        history_alive_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """CEM 후보 action sequence로 미래 DEVS state feature를 예측한다.

        이 함수는 target state를 받지 않는다. `history_features`는 관측된
        `history_frames`개 state이고, `action_features`는
        `history_frames + pred_frames - 1`개 전이 action이다.
        """
        _expect_rank("history_features", history_features, 4)
        _expect_rank("history_feature_mask", history_feature_mask, 4)
        _expect_rank("history_type_ids", history_type_ids, 3)
        _expect_rank("history_entity_ids", history_entity_ids, 3)
        _expect_rank("history_team_ids", history_team_ids, 3)
        _expect_rank("history_alive_mask", history_alive_mask, 3)
        _expect_rank("action_features", action_features, 4)
        _expect_rank("action_unit_ids", action_unit_ids, 3)
        _expect_rank("issued_mask", issued_mask, 3)
        _expect_bool("history_feature_mask", history_feature_mask)
        _expect_bool("history_alive_mask", history_alive_mask)
        _expect_bool("issued_mask", issued_mask)

        batch_size, history_frames, num_slots, feature_dim = history_features.shape
        if history_frames != self.config.history_frames:
            raise ValueError(f"history_features 시간 길이는 {self.config.history_frames}이어야 한다")
        if feature_dim != MAX_FEATURE_DIM:
            raise ValueError(f"history_features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")
        expected_history_shape = (batch_size, history_frames, num_slots)
        if history_type_ids.shape != expected_history_shape:
            raise ValueError("history_type_ids shape는 (B, T_hist, N)이어야 한다")
        if history_entity_ids.shape != expected_history_shape:
            raise ValueError("history_entity_ids shape는 (B, T_hist, N)이어야 한다")
        if history_team_ids.shape != expected_history_shape:
            raise ValueError("history_team_ids shape는 (B, T_hist, N)이어야 한다")
        if history_alive_mask.shape != expected_history_shape:
            raise ValueError("history_alive_mask shape는 (B, T_hist, N)이어야 한다")
        if history_feature_mask.shape != history_features.shape:
            raise ValueError("history_feature_mask shape는 history_features shape와 같아야 한다")

        self._expect_static_slot_metadata("history_type_ids", history_type_ids)
        self._expect_static_slot_metadata("history_entity_ids", history_entity_ids)
        self._expect_static_slot_metadata("history_team_ids", history_team_ids)

        total_frames = self.config.history_frames + self.config.pred_frames
        action_frames = total_frames - 1
        if action_features.shape[:2] != (batch_size, action_frames):
            raise ValueError("action_features shape는 (B, T_total-1, U, ACTION_DIM)이어야 한다")
        if action_features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action_features 마지막 차원은 {ACTION_DIM}이어야 한다")
        if action_unit_ids.shape != action_features.shape[:3]:
            raise ValueError("action_unit_ids shape는 (B, T_total-1, U)이어야 한다")
        if issued_mask.shape != action_unit_ids.shape:
            raise ValueError("issued_mask shape는 action_unit_ids shape와 같아야 한다")

        history_tokens = self.encode_state_sequence(
            features=history_features,
            feature_mask=history_feature_mask,
            type_ids=history_type_ids,
            team_ids=history_team_ids,
            alive_mask=history_alive_mask,
        )
        full_type_ids = history_type_ids[:, :1].expand(batch_size, total_frames, num_slots)
        full_entity_ids = history_entity_ids[:, :1].expand(batch_size, total_frames, num_slots)
        full_team_ids = history_team_ids[:, :1].expand(batch_size, total_frames, num_slots)
        action_tokens = self.build_transition_action_tokens(
            source_tokens=history_tokens[:, 0],
            type_ids=full_type_ids,
            team_ids=full_team_ids,
            entity_ids=full_entity_ids,
            action_features=action_features,
            action_unit_ids=action_unit_ids,
            issued_mask=issued_mask,
        )
        future_tokens = self.masked_predictor.inference(
            history_tokens,
            action_tokens=action_tokens,
            type_ids=history_type_ids[:, 0],
        )
        future_type_ids = full_type_ids[:, self.config.history_frames:total_frames]
        # 미래 프레임은 unit과 mission만 디코딩한다. 지형은 정지 물체라 미래가 이미
        # 알려져 있고, 예측하는 것보다 마지막 관측값을 쓰는 편이 정확하다. 임무는
        # time_remaining_ratio와 completion_flag가 변하므로 예측 대상으로 남긴다.
        # 지형을 0으로 두면 안 된다 — value head의 attention 문맥에 들어간다.
        future_features = self.state_decoder(
            future_tokens.reshape(
                batch_size * self.config.pred_frames,
                num_slots,
                self.config.embedding_dim,
            ),
            future_type_ids.reshape(batch_size * self.config.pred_frames, num_slots),
            only_types=(ObjectType.UNIT, ObjectType.MISSION),
        ).reshape(batch_size, self.config.pred_frames, num_slots, MAX_FEATURE_DIM)
        is_terrain = (future_type_ids == int(ObjectType.TERRAIN)).unsqueeze(-1)
        static_features = history_features[:, -1].unsqueeze(1).expand_as(future_features)
        future_features = torch.where(is_terrain, static_features, future_features)
        if POSITION_RESIDUAL:
            future_features = _apply_position_residual(
                future_features,
                delta=self.unit_position_delta(future_tokens[..., self._position_slice]),
                last_observed=history_features[:, -1],
                type_ids=future_type_ids,
            )
        if ENFORCE_ROLLOUT_PHYSICS:
            future_features = _enforce_unit_physics(
                future_features,
                last_observed=history_features[:, -1],
                type_ids=future_type_ids,
            )
        return {
            "history_tokens": history_tokens,
            "action_tokens": action_tokens,
            "future_tokens": future_tokens,
            "future_features": future_features,
            "future_type_ids": future_type_ids,
            "future_entity_ids": full_entity_ids[:, self.config.history_frames:total_frames],
            "future_team_ids": full_team_ids[:, self.config.history_frames:total_frames],
        }

    def predict_next_tokens(
        self,
        state_tokens: torch.Tensor,
        *,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        alive_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> torch.Tensor:
        """legacy one-step dynamics 경로다. C-JEPA rollout은 별도 action token을 쓴다."""
        conditioned = self.action_conditioner(
            state_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            entity_ids=entity_ids,
            action_features=action_features,
            action_unit_ids=action_unit_ids,
            issued_mask=issued_mask,
        )
        attention_allowed = build_object_attention_mask(type_ids, alive_mask)
        predicted_context = self.dynamics_predictor(conditioned, attention_allowed)
        return state_tokens + self.delta_head(predicted_context)

    def forward(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
        action_features: torch.Tensor,
        action_unit_ids: torch.Tensor,
        issued_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """한 step 월드모델 예측을 수행한다."""
        state_tokens = self.encode_state(
            features=features,
            feature_mask=feature_mask,
            type_ids=type_ids,
            team_ids=team_ids,
            alive_mask=alive_mask,
        )
        pred_tokens = self.predict_next_tokens(
            state_tokens,
            type_ids=type_ids,
            team_ids=team_ids,
            entity_ids=entity_ids,
            alive_mask=alive_mask,
            action_features=action_features,
            action_unit_ids=action_unit_ids,
            issued_mask=issued_mask,
        )
        pred_features = self.state_decoder(pred_tokens, type_ids)
        return {
            "state_tokens": state_tokens,
            "pred_tokens": pred_tokens,
            "pred_features": pred_features,
        }
