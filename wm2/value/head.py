"""wm2 value head (설계 10절).

V(상태) = "이 상태에서 이대로 계속 가면 임무 달성도가 얼마나 되나" (MC 라벨).
CEM 채점에서 horizon 너머를 담당한다 — horizon 안은 예측 보상이 dense하게 맡는다.

구 value_head.py의 값 치른 교훈을 계승한다:
- 기하 특징 × 임무 one-hot 외적 — 임무마다 목표거리의 부호가 다르므로(destroy_all은
  목표 무관) 곱셈 상호작용이 필요했다 (구 실측: concat만으로는 부호가 뒤집힘)
- 팀별 masked mean pooling — 맵·병력 수가 바뀌어도 같은 head를 쓴다
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..config import ModelConfig
from ..model.encoder import BlockSlotEncoder
from ..model.features import (
    MAX_HP,
    OBJECTIVE_RADIUS,
    TeamId,
    denorm_x,
    denorm_y,
)
from ..model.predictor import _Layer

NUM_MISSION_TYPES = 4
NUM_GEOMETRY = 8
# reach 성분의 자 길이 (유닛). 이 거리 밖은 원식이 음수 → clamp 0으로 뭉개진다.
#
# 구 시스템은 10(관측거리)이었다 — 에피소드 "끝 상태"만 채점했고 끝 상태 거리가
# 2.3~4유닛에 몰려 있어 근거리 확대가 옳았다. 우리는 매 6초, 모든 거리의 상태를
# 채점하므로 10이면 스폰~중반(11유닛 밖)에서 전 후보가 동점 0이 되어 CEM이
# 제비뽑기가 된다 (wm2_loop2에서 gain=+0.000 연발로 실측). 맵 폭 40으로 늘려
# 전 구간에서 "가까워지면 점수가 오르는" 단조성을 확보한다. 선형 = 모든 1유닛
# 등가(형태 prior 없음). 근거리 변별이 무뎌지는 대가는 hold_objective 지표로
# 감시하고, 필요가 실측되면 그때 비선형 자를 검토한다.
#
# 이 값이 바뀌면 V 라벨의 의미가 바뀐다 — V 재학습 필수.
OBJECTIVE_GAP_SCALE = 40.0


@dataclass(frozen=True)
class ValueConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0


def mission_progress(
    *,
    mission_type: int,
    blue_alive: int,
    red_hp_total: float,
    red_initial: int,
    objective_distance: float,
) -> float:
    """승리조건의 연속 완화 (구 value_online.mission_progress와 동일 의미)."""
    if blue_alive <= 0:
        return 0.0
    if not math.isfinite(objective_distance):
        objective_distance = OBJECTIVE_GAP_SCALE
    destroy = 1.0 - min(1.0, red_hp_total / (max(red_initial, 1) * MAX_HP))
    reach = 1.0 - min(1.0, max(0.0, objective_distance - OBJECTIVE_RADIUS) / OBJECTIVE_GAP_SCALE)
    if mission_type == 1:      # destroy_all
        return destroy
    if mission_type in (2, 3):  # reach / hold
        return reach
    return min(destroy, reach)  # destroy_and_reach: AND는 min (가중평균 금지 — 구 교훈)


def geometry_features(
    unit_features: torch.Tensor,   # (B, U, 10)
    team_ids: torch.Tensor,        # (U,)
    mission_features: torch.Tensor,  # (B, 5)
) -> torch.Tensor:
    """(B, NUM_GEOMETRY). 좌표는 정규화 특징에서 복원해 월드 단위로 계산한다."""
    b = unit_features.shape[0]
    blue = team_ids == int(TeamId.BLUE)
    red = team_ids == int(TeamId.RED)
    alive = unit_features[..., 7] > 0.5                       # (B, U)
    x = unit_features[..., 3] * 20.0                          # denorm_x와 동일 (span 40 → ×20)
    y = unit_features[..., 4] * 12.5 - 2.5                    # denorm_y: span 25, 중심 -2.5
    obj_x = mission_features[:, 1:2] * 20.0
    obj_y = mission_features[:, 2:3] * 12.5 - 2.5

    inf = torch.full_like(x, 1e6)
    obj_dist = torch.sqrt((x - obj_x) ** 2 + (y - obj_y) ** 2)
    blue_alive_mask = alive & blue.reshape(1, -1)
    red_alive_mask = alive & red.reshape(1, -1)
    obj_masked = torch.where(blue_alive_mask, obj_dist, inf)
    min_obj = obj_masked.min(dim=1).values.clamp(max=OBJECTIVE_GAP_SCALE * 2)

    # BLUE-RED 최근접 거리
    bx = torch.where(blue_alive_mask, x, inf)
    rx = torch.where(red_alive_mask, x, inf)
    cross = torch.sqrt(
        (x.unsqueeze(2) - x.unsqueeze(1)) ** 2 + (y.unsqueeze(2) - y.unsqueeze(1)) ** 2
    )  # (B, U, U)
    pair_mask = blue_alive_mask.unsqueeze(2) & red_alive_mask.unsqueeze(1)
    cross = torch.where(pair_mask, cross, torch.full_like(cross, 1e6))
    min_cross = cross.reshape(b, -1).min(dim=1).values.clamp(max=40.0)

    blue_total = blue.sum().clamp_min(1).float()
    red_total = red.sum().clamp_min(1).float()
    return torch.stack(
        [
            min_obj / OBJECTIVE_GAP_SCALE,
            min_cross / 10.0,
            blue_alive_mask.sum(dim=1).float() / blue_total,
            red_alive_mask.sum(dim=1).float() / red_total,
            (unit_features[..., 1] * blue.reshape(1, -1)).sum(dim=1) / blue_total,
            (unit_features[..., 1] * red.reshape(1, -1)).sum(dim=1) / red_total,
            mission_features[:, 3],   # time_remaining
            mission_features[:, 4],   # completion
        ],
        dim=-1,
    )


class WM2ValueHead(nn.Module):
    def __init__(self, config: ValueConfig | None = None):
        super().__init__()
        self.config = config or ValueConfig()
        c = self.config
        self.encoder = BlockSlotEncoder(c.embedding_dim, c.hidden_dim, c.dropout)
        self.layers = nn.ModuleList(
            _Layer(c.embedding_dim, c.hidden_dim, c.num_heads, c.dropout)
            for _ in range(c.num_layers)
        )
        pooled = c.embedding_dim * 3 + NUM_GEOMETRY + NUM_GEOMETRY * NUM_MISSION_TYPES
        self.output = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        unit_features: torch.Tensor,     # (B, U, 10)
        terrain_features: torch.Tensor,  # (B, T, 9)
        mission_features: torch.Tensor,  # (B, 5)
        team_ids: torch.Tensor,          # (U,)
    ) -> torch.Tensor:
        """(B,) progress ∈ [0, 1]."""
        b, num_units, _ = unit_features.shape
        units = self.encoder.encode_units(unit_features.unsqueeze(1), team_ids).squeeze(1)
        mission = self.encoder.encode_mission(mission_features.unsqueeze(1)).squeeze(1)
        terrain = self.encoder.encode_terrain(terrain_features)

        queries = torch.cat([units, mission.unsqueeze(1)], dim=1)      # (B, U+1, D)
        visible = torch.ones(
            b, queries.shape[1], queries.shape[1] + terrain.shape[1],
            dtype=torch.bool, device=queries.device,
        )
        for layer in self.layers:
            queries = layer(queries, terrain, visible)

        alive = unit_features[..., 7] > 0.5
        blue = (team_ids == int(TeamId.BLUE)).reshape(1, -1) & alive
        red = (team_ids == int(TeamId.RED)).reshape(1, -1) & alive

        def pool(mask: torch.Tensor) -> torch.Tensor:
            weight = mask.float().unsqueeze(-1)
            return (queries[:, :num_units] * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

        geo = geometry_features(unit_features, team_ids, mission_features)
        onehot = torch.eye(NUM_MISSION_TYPES, device=geo.device)[
            mission_features[:, 0].long().clamp(0, NUM_MISSION_TYPES - 1)
        ]
        crossed = (geo.unsqueeze(-1) * onehot.unsqueeze(1)).reshape(b, -1)
        pooled = torch.cat([pool(blue), pool(red), queries[:, num_units], geo, crossed], dim=-1)
        # 라벨은 15초 progress 증가분이라 음수(전멸·점유 상실)가 가능 — 선형 출력.
        return self.output(pooled).squeeze(-1)


def save_value_head(path: Path, model: WM2ValueHead) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)


def load_value_head(path: Path, device: torch.device) -> WM2ValueHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = WM2ValueHead(ValueConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model
