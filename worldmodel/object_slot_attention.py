"""DEVS 객체 slot을 action-conditioned predictor로 상호작용시키는 월드모델 모듈.

Le-WM의 `ViT encoder -> action encoder -> predictor` 흐름에서 ViT encoder를
DEVS typed object slot encoder로 교체한 구조다. 여기서는 이미지에서 slot을
발견하지 않는다. DEVS state가 이미 객체를 알고 있으므로, raw slot은 타입별
projection만 거치고 객체 간 상호작용은 action-conditioned predictor에서 학습한다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hackerthon.worldmodel.actions import ACTION_DIM
from hackerthon.worldmodel.slots import (
    MAX_FEATURE_DIM,
    MISSION_FEATURE_NAMES,
    TERRAIN_FEATURE_NAMES,
    UNIT_FEATURE_NAMES,
    ObjectType,
    TeamId,
)


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

    def forward(self, tokens: torch.Tensor, attention_allowed: torch.Tensor) -> torch.Tensor:
        """attention_allowed=True인 slot 쌍만 self-attention에 사용한다."""
        _expect_rank("tokens", tokens, 3)
        _expect_rank("attention_allowed", attention_allowed, 3)
        _expect_bool("attention_allowed", attention_allowed)
        batch_size, num_slots, _ = tokens.shape
        if attention_allowed.shape != (batch_size, num_slots, num_slots):
            raise ValueError("attention_allowed shape는 (B, N, N)이어야 한다")

        blocked = ~attention_allowed
        attn_mask = blocked.unsqueeze(1).expand(
            batch_size,
            self.num_heads,
            num_slots,
            num_slots,
        )
        attn_mask = attn_mask.reshape(batch_size * self.num_heads, num_slots, num_slots)

        attn_input = self.attn_norm(tokens)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
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

    def forward(self, tokens: torch.Tensor, attention_allowed: torch.Tensor) -> torch.Tensor:
        """객체 slot들의 관계 표현을 갱신한다."""
        for layer in self.layers:
            tokens = layer(tokens, attention_allowed)
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

    def forward(self, tokens: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        """시간축과 객체축을 펼친 token sequence에 causal attention을 적용한다."""
        _expect_rank("tokens", tokens, 3)
        _expect_rank("attn_mask", attn_mask, 2)
        _expect_bool("attn_mask", attn_mask)
        if attn_mask.shape != (tokens.shape[1], tokens.shape[1]):
            raise ValueError("attn_mask shape는 (L, L)이어야 한다")
        attn_input = self.attn_norm(tokens)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
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

    def forward(self, tokens: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        """mask token, 실제 history token, future query token을 causal하게 처리한다."""
        for layer in self.layers:
            tokens = layer(tokens, attn_mask=attn_mask)
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
    ) -> torch.Tensor:
        """query frame이 미래 frame token을 보지 못하게 하는 attention mask."""
        if total_frames <= 0:
            raise ValueError("total_frames는 0보다 커야 한다")
        if total_token_slots <= 0:
            raise ValueError("total_token_slots는 0보다 커야 한다")
        frame_ids = torch.arange(total_frames, device=device).repeat_interleave(total_token_slots)
        # PyTorch MultiheadAttention의 bool attn_mask는 True가 차단을 뜻한다.
        return frame_ids.unsqueeze(0) > frame_ids.unsqueeze(1)

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
        batch_size, total_frames, total_token_slots, embedding_dim = model_input.shape
        num_object_slots = history_tokens.shape[2]
        flat_input = model_input.reshape(batch_size, total_frames * total_token_slots, embedding_dim)
        attn_mask = self.temporal_attention_mask(
            total_frames=total_frames,
            total_token_slots=total_token_slots,
            device=flat_input.device,
        )
        flat_output = self.transformer(flat_input, attn_mask=attn_mask)
        output = flat_output.reshape(batch_size, total_frames, total_token_slots, embedding_dim)
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
        batch_size, total_frames, total_token_slots, embedding_dim = model_input.shape
        num_object_slots = history_tokens.shape[2]
        flat_input = model_input.reshape(batch_size, total_frames * total_token_slots, embedding_dim)
        attn_mask = self.temporal_attention_mask(
            total_frames=total_frames,
            total_token_slots=total_token_slots,
            device=flat_input.device,
        )
        flat_output = self.transformer(flat_input, attn_mask=attn_mask)
        output = flat_output.reshape(batch_size, total_frames, total_token_slots, embedding_dim)
        output = self.output_projection(output)
        return output[:, :, :num_object_slots], masked_indices, masked_slot_mask

    @torch.no_grad()
    def inference(self, history_tokens: torch.Tensor, *, action_tokens: torch.Tensor) -> torch.Tensor:
        """planning 때 쓰는 비마스킹 future prediction 경로."""
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
        total_token_slots = num_slots + num_action_tokens
        flat_input = model_input.reshape(batch_size, self.total_frames * total_token_slots, embedding_dim)
        attn_mask = self.temporal_attention_mask(
            total_frames=self.total_frames,
            total_token_slots=total_token_slots,
            device=flat_input.device,
        )
        flat_output = self.transformer(flat_input, attn_mask=attn_mask)
        output = flat_output.reshape(batch_size, self.total_frames, total_token_slots, embedding_dim)
        output = self.output_projection(output)
        return output[:, history_frames:self.total_frames, :num_slots]


def cjepa_prediction_loss(
    *,
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    masked_indices: torch.Tensor,
    history_frames: int,
    pred_frames: int,
) -> dict[str, torch.Tensor]:
    """C-JEPA 학습 손실을 계산한다.

    DEVS slot은 entity id로 정렬되므로 Hungarian matching 없이 같은 index끼리
    비교한다. 손실은 숨겨진 history slot 복원과 future slot 예측을 분리해 낸다.
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

    future_loss = F.mse_loss(
        pred_tokens[:, history_frames:history_frames + pred_frames],
        target_tokens[:, history_frames:history_frames + pred_frames].detach(),
    )
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

                matched = torch.nonzero(entity_ids[batch_index] == unit_id, as_tuple=False).flatten()
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

    def forward(self, tokens: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """타입별 decoder 출력만 padding feature의 앞쪽 차원에 채운다."""
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
        self.slot_encoder = TypedObjectSlotEncoder(self.config)
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
        self.masked_predictor = CausalMaskedObjectPredictor(self.config)
        self.state_decoder = ObjectStateDecoder(self.config)
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
        )
        pred_features = self.state_decoder(
            pred_tokens.reshape(pred_tokens.shape[0] * pred_tokens.shape[1], pred_tokens.shape[2], pred_tokens.shape[3]),
            type_ids[:, 0].unsqueeze(1).expand(type_ids.shape[0], total_frames, type_ids.shape[2]).reshape(
                type_ids.shape[0] * total_frames,
                type_ids.shape[2],
            ),
        ).reshape(features.shape[0], total_frames, features.shape[2], MAX_FEATURE_DIM)
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
        )
        future_type_ids = full_type_ids[:, self.config.history_frames:total_frames]
        future_features = self.state_decoder(
            future_tokens.reshape(
                batch_size * self.config.pred_frames,
                num_slots,
                self.config.embedding_dim,
            ),
            future_type_ids.reshape(batch_size * self.config.pred_frames, num_slots),
        ).reshape(batch_size, self.config.pred_frames, num_slots, MAX_FEATURE_DIM)
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
