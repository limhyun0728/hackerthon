"""상태 하나를 보고 그 에피소드가 어떻게 끝날지 예측하는 value head.

CEM은 6스텝 앞만 채점한다. 측정해 보면 그 구간에서는 RED이 아직 반응하지 않고
(DEVS에서도 t+4초까지 후보 간 위치 산포 0.00m), 6스텝 점수와 최종 결과의
순위상관이 초반 시점에서 0 근처다. 6스텝 끝 상태를 이 head로 평가해 그 너머를
접어 넣는 것이 목적이다.

**임무 결과 3-class만 쓰지 않는다.** 실측 분포가 TIMEOUT 81% / WIN 9.5% / LOSE 9.5%로
한쪽에 쏠려 있어 그것만으로는 후보를 갈라내지 못한다. 대신 연속값 보조 목표를 함께
낸다.

- outcome_logits : WIN / LOSE / TIMEOUT
- blue_hp_ratio  : 종료 시점 BLUE 체력 비 (0~1)
- red_hp_ratio   : 종료 시점 RED 체력 비 (0~1)
- objective_gap  : 종료 시점 목표까지 거리 (정규화)

planning에 쓰는 scalar V는 임무별로 이 출력들을 조합해 만든다(`scalar_value`).

라벨은 Monte Carlo다 — 상태가 속한 에피소드가 실제로 어떻게 끝났는지를 그대로
쓴다. 따라서 이 V는 "규칙 기반 BLUE로 계속 갔을 때의 가치"이지 최적 행동의 가치가
아니다. 후보 순위를 매기는 데는 쓸 수 있지만 절대값으로 해석하면 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import nn

from hackerthon.worldmodel.slots import ObjectType
from hackerthon.worldmodel.object_slot_attention import (
    STATIC_TERRAIN_KV,
    ObjectSlotModelConfig,
    ObjectSlotTransformer,
    TypedObjectSlotEncoder,
    build_object_attention_mask,
)

# 임무 종류. slots.MISSION_* 과 같은 값이다.
MISSION_DESTROY_AND_REACH = 0
MISSION_DESTROY_ALL = 1
MISSION_REACH_OBJECTIVE = 2
MISSION_HOLD_OBJECTIVE = 3

OUTCOME_WIN, OUTCOME_LOSE, OUTCOME_TIMEOUT = 0, 1, 2
OUTCOME_NAMES = ("WIN", "LOSE", "TIMEOUT")

# objective_gap 정규화 기준 = 관측거리(10유닛 = 100m).
# "적을 볼 수 있는 거리만큼 목표에서 떨어져 있으면 도달 못한 것"이라는 뜻이다.
# 좌표계 크기(맵 반폭 20)를 쓰면 실제 종료 거리가 p25~p75 = 2.3~4.0유닛에 몰려
# 있어 대부분 0.85~0.95로 압축되고 변별력이 죽는다.
OBJECTIVE_GAP_SCALE = 10.0

# slot feature 안의 index. 기하 특징을 직접 계산할 때 쓴다.
UNIT_HP_INDEX, UNIT_X_INDEX, UNIT_Y_INDEX = 1, 3, 4
MISSION_OBJECTIVE_X_INDEX, MISSION_OBJECTIVE_Y_INDEX = 1, 2
MISSION_TIME_REMAINING_INDEX = 3
# 경과 시간으로 나눌 때의 하한. t=0 근처에서 진척률이 발산하는 것을 막는다.
MIN_ELAPSED_RATIO = 0.05
# 기하 특징 : 목표까지 최소/평균거리, 적까지 최소거리, 생존비 3개,
#            적/아군 HP비, 경과 시간비, 시간당 피해율
NUM_GEOMETRY_FEATURES = 10
NUM_MISSION_TYPES = 4
# 기하 특징 x 임무 one-hot 외적. "이 임무에서 이 특징을 볼지"를 선형으로 표현한다.
#
# concat만으로는 부족했다. 임무 정보가 202차원 중 4개뿐인데 기하 특징과 **곱셈으로**
# 상호작용해야 하기 때문이다. 실측에서 임무별로 목표거리와 달성도의 상관 부호가
# 갈린다 — destroy_and_reach -0.237 / hold_objective -0.401 / reach_objective -0.295
# 인데 destroy_all만 +0.065다. 목표가 그 임무의 승리 조건에 없으니 당연하다.
#
# 데이터의 3/4이 "목표에 가까울수록 좋다"고 말하는데 destroy_all만 반대라, one-hot을
# 이어붙이기만 해서는 그 하나를 뒤집지 못했다. 규칙 780 에피소드로 재학습했더니
# 나머지 셋은 올랐지만(hold +0.118 -> +0.615) destroy_all은 +0.006 -> -0.256으로
# 부호가 뒤집혔다.
NUM_GEOMETRY_MISSION_FEATURES = NUM_GEOMETRY_FEATURES * NUM_MISSION_TYPES


@dataclass(frozen=True)
class ValueHeadConfig:
    """value head 구조 설정."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim/hidden_dim은 0보다 커야 한다")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers/num_heads는 0보다 커야 한다")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim은 num_heads로 나누어떨어져야 한다")


class ValueHead(nn.Module):
    """객체 slot을 팀별로 모아 에피소드 최종 결과를 예측한다.

    지형 slot 수가 맵마다 100~200개로 가변이라 padding 대신 attention mask와
    masked mean pooling으로 처리한다. 그래야 맵이 바뀌어도 같은 head를 쓴다.
    """

    def __init__(self, config: ValueHeadConfig):
        super().__init__()
        self.config = config
        encoder_config = ObjectSlotModelConfig(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_predictor_layers=1,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.slot_encoder = TypedObjectSlotEncoder(encoder_config)
        self.interaction = ObjectSlotTransformer(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        dim = config.embedding_dim
        # BLUE pool + RED pool + 전체 pool + 임무 one-hot(4) + 기하 특징 + 기하x임무
        self.trunk = nn.Sequential(
            nn.Linear(
                dim * 3 + NUM_MISSION_TYPES + NUM_GEOMETRY_FEATURES + NUM_GEOMETRY_MISSION_FEATURES,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.outcome_head = nn.Linear(config.hidden_dim, 3)
        # 주 목표. mission_completed()를 연속 완화한 임무 달성도(0~1)를 직접 낸다.
        # 이 값이 곧 planning용 V라 추론 시 손으로 조합할 게 없다.
        self.progress_head = nn.Linear(config.hidden_dim, 1)
        self.scalar_head = nn.Linear(config.hidden_dim, 3)  # blue_hp, red_hp, obj_gap

    @staticmethod
    def _geometry(
        features: torch.Tensor,
        type_ids: torch.Tensor,
        blue_mask: torch.Tensor,
        red_mask: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """목표·적까지의 거리, 생존비, 전투 진척률을 스칼라로 뽑는다.

        `blue_mask`/`red_mask`는 생존 여부를 안 씌운 전체 slot이다. 죽은 유닛도 slot에
        남으므로 그 개수가 곧 초기 병력이고, 그래야 "초기 대비 얼마나 깎았나"를 낼 수
        있다. 생존만 세는 값은 안에서 `alive_mask`로 따로 만든다.

        **시간당 피해율이 핵심이다.** 라벨은 에피소드 종료 시점의 격파 비율인데, 상태만
        보고는 "지금 이만큼 깎은 게 잘한 건가"를 알 수 없다. t=10에 20% 깎은 것과
        t=50에 20% 깎은 것은 전혀 다르다. 실측 순위상관:

            임무                  현재 적HP비   누적피해/경과
            destroy_all            -0.264       +0.403
            destroy_and_reach      -0.296       +0.459
            hold_objective         -0.096       +0.119
            reach_objective        -0.005       +0.032

        격파가 승리 조건에 든 두 임무에서 신호가 1.5배가 된다. reach 계열에서는 목표까지
        거리가 이미 그 역할을 하므로 효과가 없다. 남은 시간만 넣는 것은 소용없다
        (상관 -0.002 ~ +0.009) — 누적 피해와 함께여야 진척률이 된다.

        좌표를 주고 관계를 알아서 배우게 두면 안 되는 것도 같은 이유다. 목표 좌표는
        mission slot에, 유닛 좌표는 unit slot에 따로 있어 **차이를 계산해야** 하는데
        pooling이 평균을 내버려 상대 위치가 사라진다. 실제로 그렇게 학습한 V는 "목표에
        접근하면 달성도가 내려간다"고 예측했다 — 데이터의 실제 상관은 -0.36(가까울수록
        좋음)인데 부호를 반대로 배웠다.
        """
        batch_size = features.shape[0]
        device = features.device
        is_mission = type_ids == int(ObjectType.MISSION)
        # mission slot은 배치마다 하나다. 없으면 0으로 둔다.
        mission_index = torch.argmax(is_mission.to(torch.int64), dim=1)
        rows = torch.arange(batch_size, device=device)
        objective = features[
            rows, mission_index][:, [MISSION_OBJECTIVE_X_INDEX, MISSION_OBJECTIVE_Y_INDEX]
        ]

        position = features[..., [UNIT_X_INDEX, UNIT_Y_INDEX]]
        to_objective = (position - objective.unsqueeze(1)).norm(dim=-1)

        def summarize(mask: torch.Tensor, distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """mask가 True인 slot만의 (최소, 평균) 거리. 없으면 최대값 2.0으로 둔다."""
            large = torch.full_like(distance, 2.0)
            picked = torch.where(mask, distance, large)
            minimum = picked.min(dim=1).values
            weight = mask.to(distance.dtype)
            total = (distance * weight).sum(dim=1)
            count = weight.sum(dim=1)
            average = torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(total, 2.0))
            return minimum, average

        blue_alive = blue_mask & alive_mask
        red_alive = red_mask & alive_mask
        objective_min, objective_mean = summarize(blue_alive, to_objective)

        # BLUE-RED 최소 거리. 교전 임박도를 나타낸다.
        blue_position = torch.where(blue_alive.unsqueeze(-1), position, torch.full_like(position, 1e3))
        red_position = torch.where(red_alive.unsqueeze(-1), position, torch.full_like(position, -1e3))
        pairwise = torch.cdist(blue_position, red_position)
        contact_min = pairwise.min(dim=2).values.min(dim=1).values.clamp(max=2.0)
        has_pair = (blue_alive.any(dim=1) & red_alive.any(dim=1)).to(features.dtype)
        contact_min = contact_min * has_pair + 2.0 * (1.0 - has_pair)

        blue_count = blue_alive.sum(dim=1).to(features.dtype)
        red_count = red_alive.sum(dim=1).to(features.dtype)
        ratio = blue_count / (blue_count + red_count).clamp_min(1.0)

        # 초기 병력은 전체 slot 수다. 죽은 유닛도 slot에 hp=0으로 남는다.
        hp_ratio = features[..., UNIT_HP_INDEX]
        initial_blue = blue_mask.sum(dim=1).to(features.dtype).clamp_min(1.0)
        initial_red = red_mask.sum(dim=1).to(features.dtype).clamp_min(1.0)
        blue_hp_ratio = (hp_ratio * blue_mask.to(features.dtype)).sum(dim=1) / initial_blue
        red_hp_ratio = (hp_ratio * red_mask.to(features.dtype)).sum(dim=1) / initial_red

        time_remaining = features[rows, mission_index][:, MISSION_TIME_REMAINING_INDEX]
        elapsed = (1.0 - time_remaining).clamp(0.0, 1.0)
        damage_rate = (1.0 - red_hp_ratio) / elapsed.clamp_min(MIN_ELAPSED_RATIO)

        return torch.stack(
            [
                objective_min,
                objective_mean,
                contact_min,
                ratio,
                blue_count / 10.0,
                red_count / 10.0,
                blue_hp_ratio,
                red_hp_ratio,
                elapsed,
                damage_rate,
            ],
            dim=-1,
        )

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """mask가 True인 slot만 평균낸다. 하나도 없으면 0 벡터."""
        weight = mask.unsqueeze(-1).to(tokens.dtype)
        total = (tokens * weight).sum(dim=1)
        count = weight.sum(dim=1).clamp_min(1.0)
        return total / count

    def forward(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
        blue_mask: torch.Tensor,
        red_mask: torch.Tensor,
        mission_onehot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """상태 배치에서 최종 결과 예측을 낸다."""
        tokens = self.slot_encoder(features, feature_mask, type_ids, team_ids)
        allowed = build_object_attention_mask(type_ids, alive_mask)
        if STATIC_TERRAIN_KV:
            # 지형을 query에서 빼고 key/value로만 둔다. pooling이 유닛만 쓰므로 지형
            # 토큰의 출력은 어차피 안 쓰이는데, attention이 N^2이라 slot의 89%인 지형이
            # 비용의 대부분을 차지한다. 배치 안 slot layout이 같다고 보고 첫 항목으로
            # 나눈다 (predictor와 같은 가정).
            is_query = type_ids[0] != int(ObjectType.TERRAIN)
            if not bool(is_query.all()):
                query_index = torch.nonzero(is_query, as_tuple=False).flatten()
                static_index = torch.nonzero(~is_query, as_tuple=False).flatten()
                query_tokens = tokens.index_select(1, query_index)
                static_tokens = tokens.index_select(1, static_index)
                # key 축은 query slot 다음에 static slot이 오도록 다시 정렬한다.
                allowed = torch.cat(
                    [
                        allowed.index_select(1, query_index).index_select(2, query_index),
                        allowed.index_select(1, query_index).index_select(2, static_index),
                    ],
                    dim=2,
                )
                updated = self.interaction(
                    query_tokens, allowed, static_context=static_tokens
                )
                tokens = tokens.index_copy(1, query_index, updated)
            else:
                tokens = self.interaction(tokens, allowed)
        else:
            tokens = self.interaction(tokens, allowed)

        geometry = self._geometry(features, type_ids, blue_mask, red_mask, alive_mask)
        # 기하 x 임무 외적. 임무마다 어떤 기하 특징을 볼지 선형으로 고를 수 있게 한다.
        geometry_by_mission = (
            geometry.unsqueeze(-1) * mission_onehot.unsqueeze(1)
        ).flatten(start_dim=1)
        pooled = torch.cat(
            [
                self._masked_mean(tokens, blue_mask & alive_mask),
                self._masked_mean(tokens, red_mask & alive_mask),
                self._masked_mean(tokens, alive_mask),
                mission_onehot,
                geometry,
                geometry_by_mission,
            ],
            dim=-1,
        )
        hidden = self.trunk(pooled)
        scalars = torch.sigmoid(self.scalar_head(hidden))
        return {
            "progress": torch.sigmoid(self.progress_head(hidden)).squeeze(-1),
            "outcome_logits": self.outcome_head(hidden),
            "blue_hp_ratio": scalars[..., 0],
            "red_hp_ratio": scalars[..., 1],
            "objective_gap": scalars[..., 2],
        }


def scalar_value(prediction: dict[str, torch.Tensor], mission_type: int | None = None) -> torch.Tensor:
    """planning용 scalar V. 예측된 임무 달성도를 그대로 쓴다.

    달성도 자체가 임무별로 정의된 값(mission_completed의 연속 완화)이므로 여기서
    임무별 가중치를 다시 매길 필요가 없다. mission_type 인자는 호출부 호환을 위해
    받되 쓰지 않는다.
    """
    return prediction["progress"]


def save_value_head(path: Path, model: ValueHead) -> None:
    """checkpoint 계약: config와 state_dict를 함께 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"value_head_config": asdict(model.config), "model_state_dict": model.state_dict()},
        path,
    )


def load_value_head(path: Path, device: torch.device) -> ValueHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ValueHead(ValueHeadConfig(**dict(payload["value_head_config"]))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
