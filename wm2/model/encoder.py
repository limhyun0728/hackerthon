"""블록 인코더 (설계 3절).

토큰 = [position 16 | velocity 8 | state 16 | heading 8 | identity 16] concat.
- 위치 인코더는 타입 공용 — 유닛/건물/목표의 좌표가 embedding의 같은 축에 놓인다.
  입력은 (x, y, w, h) 4차원으로 통일하고 유닛·임무는 w=h=0.
- 블록별 LayerNorm. 전체 LN은 블록을 다시 섞으므로 금지 (구 시스템 실측).
- identity = type + team + 팀내 index 임베딩. (type, team, index)만의 함수 = 시간 불변
  이므로 predictor가 층마다 리셋할 수 있다.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_DIM, ObjectType, TeamId, block_layout

MAX_TEAM_SLOTS = 10  # 팀내 index 임베딩 폭 (팀 최대 10 규약)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class BlockSlotEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layout = block_layout(embedding_dim)
        self.embedding_dim = embedding_dim
        pos_dim = self.layout["position"].stop - self.layout["position"].start
        vel_dim = self.layout["velocity"].stop - self.layout["velocity"].start
        state_dim = self.layout["state"].stop - self.layout["state"].start
        heading_dim = self.layout["heading"].stop - self.layout["heading"].start
        id_dim = self.layout["identity"].stop - self.layout["identity"].start

        self.position_encoder = _mlp(4, hidden_dim, pos_dim, dropout)
        self.velocity_encoder = _mlp(2, hidden_dim, vel_dim, dropout)
        # 속도·방향이 없는 타입은 학습된 상수로 채운다. 0이면 그 축이 죽은 채
        # attention에 들어가 타입 구분이 흐려진다 (구 시스템의 heading_default 관례).
        self.velocity_default = nn.Parameter(torch.zeros(vel_dim))
        self.unit_state_encoder = _mlp(3, hidden_dim, state_dim, dropout)      # hp, ammo, alive
        self.terrain_state_encoder = _mlp(4, hidden_dim, state_dim, dropout)   # trav, cost, cover, los
        self.mission_state_encoder = _mlp(2, hidden_dim, state_dim, dropout)   # time_remaining, completion
        self.heading_encoder = _mlp(2, hidden_dim, heading_dim, dropout)
        self.heading_default = nn.Parameter(torch.zeros(heading_dim))

        self.type_embedding = nn.Embedding(len(ObjectType), id_dim)
        self.team_embedding = nn.Embedding(3, id_dim)                # NONE/BLUE/RED
        self.slot_index_embedding = nn.Embedding(MAX_TEAM_SLOTS, id_dim)

        self.block_norms = nn.ModuleDict(
            {name: nn.LayerNorm(sl.stop - sl.start) for name, sl in self.layout.items()}
        )

    # ── identity (리셋의 앵커) ──────────────────────────────────────────
    def unit_identity(self, team_ids: torch.Tensor) -> torch.Tensor:
        """(U,) team id → (U, id_dim). 팀내 index는 layout이 id 정렬이라 팀별 누적 순번."""
        team_index = torch.where(team_ids == int(TeamId.BLUE), 0, 1)  # BLUE=0, RED=1 임베딩 행
        # 팀내 순번: BLUE 구간과 RED 구간 각각 0..k
        slot_index = torch.zeros_like(team_ids)
        for team_value in (int(TeamId.BLUE), int(TeamId.RED)):
            selected = team_ids == team_value
            slot_index[selected] = torch.arange(int(selected.sum()), device=team_ids.device)
        return (
            self.type_embedding(torch.full_like(team_ids, int(ObjectType.UNIT)))
            + self.team_embedding(team_index + 1)   # 0=NONE, 1=BLUE, 2=RED
            + self.slot_index_embedding(slot_index.clamp(max=MAX_TEAM_SLOTS - 1))
        )

    def terrain_identity(self, count: int, device: torch.device) -> torch.Tensor:
        base = self.type_embedding(
            torch.full((count,), int(ObjectType.TERRAIN), dtype=torch.long, device=device)
        ) + self.team_embedding(torch.zeros(count, dtype=torch.long, device=device))
        return base

    def mission_identity(self, device: torch.device) -> torch.Tensor:
        return (
            self.type_embedding(torch.tensor([int(ObjectType.MISSION)], device=device))
            + self.team_embedding(torch.tensor([0], device=device))
        ).squeeze(0)

    # ── 토큰 조립 ───────────────────────────────────────────────────────
    def _assemble(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        state: torch.Tensor,
        heading: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        parts = {
            "position": position,
            "velocity": velocity,
            "state": state,
            "heading": heading,
            "identity": identity,
        }
        normed = [self.block_norms[name](parts[name]) for name in ("position", "velocity", "state", "heading", "identity")]
        return torch.cat(normed, dim=-1)

    def encode_units(self, unit_features: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        """(B, F, U, 10) → (B, F, U, D). UNIT_FEATURES 순서 전제."""
        b, f, u, _ = unit_features.shape
        xy = unit_features[..., 3:5]
        pos_in = torch.cat([xy, torch.zeros_like(xy)], dim=-1)          # (…, 4) w=h=0
        position = self.position_encoder(pos_in)
        velocity = self.velocity_encoder(unit_features[..., 8:10])
        state = self.unit_state_encoder(unit_features[..., [1, 2, 7]])
        heading = self.heading_encoder(unit_features[..., 5:7])
        identity = self.unit_identity(team_ids).reshape(1, 1, u, -1).expand(b, f, u, -1)
        return self._assemble(position, velocity, state, heading, identity)

    def encode_terrain(self, terrain_features: torch.Tensor) -> torch.Tensor:
        """(B, T, 9) → (B, T, D)."""
        b, t, _ = terrain_features.shape
        position = self.position_encoder(terrain_features[..., 1:5])    # x, y, w, h
        state = self.terrain_state_encoder(terrain_features[..., 5:9])
        velocity = self.velocity_default.reshape(1, 1, -1).expand(b, t, -1)
        heading = self.heading_default.reshape(1, 1, -1).expand(b, t, -1)
        identity = self.terrain_identity(t, terrain_features.device).reshape(1, t, -1).expand(b, t, -1)
        return self._assemble(position, velocity, state, heading, identity)

    def encode_mission(self, mission_features: torch.Tensor) -> torch.Tensor:
        """(B, F, 5) → (B, F, D)."""
        b, f, _ = mission_features.shape
        xy = mission_features[..., 1:3]
        position = self.position_encoder(torch.cat([xy, torch.zeros_like(xy)], dim=-1))
        state = self.mission_state_encoder(mission_features[..., 3:5])
        velocity = self.velocity_default.reshape(1, 1, -1).expand(b, f, -1)
        heading = self.heading_default.reshape(1, 1, -1).expand(b, f, -1)
        identity = self.mission_identity(mission_features.device).reshape(1, 1, -1).expand(b, f, -1)
        return self._assemble(position, velocity, state, heading, identity)


class ActionEncoder(nn.Module):
    """계획된 명령 → 액션 노드 (C-JEPA NodeEmbedder 대응). 마스킹 불가, 손실 없음."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = _mlp(ACTION_DIM, hidden_dim, embedding_dim, dropout)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, 8, U_blue, ACTION_DIM) → (B, 8, U_blue, D)."""
        return self.encoder(actions)
