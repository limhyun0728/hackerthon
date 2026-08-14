"""wm2 predictor (설계 4·5절).

토큰 배치:
  query 열:  유닛 9프레임 (a,h1,h2,f1..f6) + 임무 9프레임 → (B, 9·(U+1), D)
  KV-only:   지형 (정적, 프레임 복제 없음) + 액션 노드 (발행틱 a..f5)

가시성 (설계 5절 확정):
  상태 토큰끼리는 전방향 — 미래 상태 토큰은 ground truth가 아니라 자기 예측이므로
  참조는 누설이 아니라 상호 정제다. 창 안의 유일한 ground truth 미래 입력은 액션이고,
  액션 노드(발행틱 j)는 프레임 ≥ j+1 인 query에게만 보인다.
  사망 유닛의 history 토큰은 key에서 차단 (구 규약).

구조 규율:
  - 미래/마스킹 query = mask token + 시간 임베딩 + anchor(frame a 토큰) 투영
    (C-JEPA의 anchor_queries 대응)
  - identity 슬라이스는 층마다 인코더 값으로 리셋 (OA-WAM address reset)
  - 지형·액션은 층을 거쳐도 갱신되지 않는 정적 KV
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .encoder import ActionEncoder, BlockSlotEncoder
from .features import MAX_RED_SLOTS, NUM_ACTION_TYPES, block_layout

# 액션 특징 안의 표적 onehot 위치 ([issued|type4|move2|target10|turn2])
_TARGET_SLICE = slice(1 + NUM_ACTION_TYPES + 2, 1 + NUM_ACTION_TYPES + 2 + MAX_RED_SLOTS)

HISTORY_FRAMES = 3
PRED_FRAMES = 6
TOTAL_FRAMES = HISTORY_FRAMES + PRED_FRAMES   # 9
ACTION_TICKS = TOTAL_FRAMES - 1               # 8 (발행틱 a..f5)


class _Layer(nn.Module):
    """pre-LN attention + FFN. K/V는 (query 열 ∪ 정적 KV)."""

    def __init__(self, dim: int, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, static_kv: torch.Tensor, visible: torch.Tensor
    ) -> torch.Tensor:
        """queries (B,Q,D), static_kv (B,S,D), visible (B,Q,Q+S) bool."""
        b, q_len, dim = queries.shape
        kv = torch.cat([self.norm_kv(queries), self.norm_kv(static_kv)], dim=1)
        q = self.q_proj(self.norm_q(queries))
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        head_dim = dim // self.heads

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b, -1, self.heads, head_dim).transpose(1, 2)

        mask = visible.unsqueeze(1)  # (B, 1, Q, K) — True=허용
        attended = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(b, q_len, dim)
        queries = queries + self.dropout(self.out_proj(attended))
        return queries + self.ffn(queries)


class WM2Predictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.layout = block_layout(dim)
        self.encoder = BlockSlotEncoder(dim, config.hidden_dim, config.dropout)
        self.action_encoder = ActionEncoder(dim, config.hidden_dim, config.dropout)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.time_embedding = nn.Embedding(TOTAL_FRAMES, dim)
        self.anchor_proj = nn.Linear(dim, dim)
        # 액션 노드에 발신자(사수) identity를 얹는 투영. 이게 없으면 같은 틱·같은
        # 표적의 ENGAGE 토큰이 사수 불문 동일해져, 표적 피해 예측이 사수별 사거리를
        # 조회할 수 없다 — "전원 한계사거리 유지사격"에 73HP를 상상하던 귀속 실패의
        # 원인 (실측: 상상 73 vs DEVS 4.8). MOVE는 목적지 좌표로 자가결합돼 무사했다.
        id_dim = self.layout["identity"].stop - self.layout["identity"].start
        self.action_identity = nn.Linear(id_dim, dim)
        # 표적 주소: ENGAGE onehot을 표적 유닛의 identity 임베딩(같은 테이블)로 번역해
        # 액션 토큰에 더한다. onehot→MLP만으로는 유닛 토큰의 identity와 좌표계가 달라
        # 대응을 학습으로 번역해야 하는데, 결합은 구조로 보장한다 (사수 서명과 같은 원리,
        # 역할 구분을 위해 별도 투영).
        # bias 없음: 표적 onehot이 전부 0인 액션(MOVE/STOP/TURN)에는 정확히 0이
        # 더해진다 — 표적 주소는 ENGAGE 토큰에만 존재한다.
        self.target_identity = nn.Linear(id_dim, dim, bias=False)
        self.layers = nn.ModuleList(
            _Layer(dim, config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        )

    def forward(
        self,
        *,
        unit_features: torch.Tensor,     # (B, 9, U, 10) — [3:]는 레이블 재료라 입력에 안 쓴다
        terrain_features: torch.Tensor,  # (B, T, 9)
        mission_features: torch.Tensor,  # (B, 9, 5)
        actions: torch.Tensor,           # (B, 8, U_blue, ACTION_DIM)
        team_ids: torch.Tensor,          # (U,)
        masked_units: torch.Tensor,      # (B, U) bool — h1·h2에서 가릴 유닛
    ) -> dict[str, torch.Tensor]:
        b, _, num_units, _ = unit_features.shape
        device = unit_features.device
        dim = self.config.embedding_dim

        # ── 인코딩 ──────────────────────────────────────────────────────
        history_units = self.encoder.encode_units(unit_features[:, :HISTORY_FRAMES], team_ids)
        mission_history = self.encoder.encode_mission(mission_features[:, :HISTORY_FRAMES])
        terrain = self.encoder.encode_terrain(terrain_features)          # (B, T, D)
        action_nodes = self.action_encoder(actions)                       # (B, 8, Ub, D)

        time_emb = self.time_embedding(torch.arange(TOTAL_FRAMES, device=device))  # (9, D)

        # anchor query: frame a 토큰의 투영 (마스킹 이전 값 — a는 항상 공개라 순서 무관)
        anchor_q_units = self.anchor_proj(history_units[:, 0])            # (B, U, D)
        anchor_q_mission = self.anchor_proj(mission_history[:, 0])        # (B, D)

        # ── h1·h2 마스킹: mask token + 시간 + anchor 투영으로 교체 ──────
        masked = masked_units.unsqueeze(1).unsqueeze(-1)                  # (B, 1, U, 1)
        replacement = (
            self.mask_token.reshape(1, 1, 1, dim)
            + time_emb[1:HISTORY_FRAMES].reshape(1, HISTORY_FRAMES - 1, 1, dim)
            + anchor_q_units.unsqueeze(1)
        )
        history_units = torch.cat(
            [
                history_units[:, :1],
                torch.where(masked, replacement, history_units[:, 1:]),
            ],
            dim=1,
        )

        # ── 미래 query ──────────────────────────────────────────────────
        future_units = (
            self.mask_token.reshape(1, 1, 1, dim)
            + time_emb[HISTORY_FRAMES:].reshape(1, PRED_FRAMES, 1, dim)
            + anchor_q_units.unsqueeze(1)
        ).expand(b, PRED_FRAMES, num_units, dim)
        future_mission = (
            self.mask_token.reshape(1, 1, dim)
            + time_emb[HISTORY_FRAMES:].reshape(1, PRED_FRAMES, dim)
            + anchor_q_mission.unsqueeze(1)
        ).expand(b, PRED_FRAMES, dim)

        # 시간 임베딩을 상태 토큰에도 더한다 (프레임 구분)
        history_units = history_units + time_emb[:HISTORY_FRAMES].reshape(1, HISTORY_FRAMES, 1, dim)
        mission_history = mission_history + time_emb[:HISTORY_FRAMES].reshape(1, HISTORY_FRAMES, dim)

        unit_tokens = torch.cat([history_units, future_units], dim=1)     # (B, 9, U, D)
        mission_tokens = torch.cat([mission_history, future_mission], dim=1)  # (B, 9, D)

        # query 열: 프레임별 [유닛 U개, 임무 1개] → (B, 9*(U+1), D)
        per_frame = torch.cat([unit_tokens, mission_tokens.unsqueeze(2)], dim=2)
        queries = per_frame.reshape(b, TOTAL_FRAMES * (num_units + 1), dim)

        # 정적 KV: 지형 + 액션 노드(발행틱 시간 + 사수 서명 + 표적 주소)
        num_blue = actions.shape[2]
        all_identity = self.encoder.unit_identity(team_ids)               # (U, id_dim)
        shooter_identity = self.action_identity(
            all_identity[:num_blue]
        ).reshape(1, 1, num_blue, dim)
        red_identity = all_identity[num_blue:]                            # (R, id_dim) — layout은 blue 先
        target_onehot = actions[..., _TARGET_SLICE][..., : red_identity.shape[0]]
        target_address = self.target_identity(target_onehot @ red_identity)  # (B, 8, Ub, D)
        action_flat = (
            action_nodes
            + time_emb[:ACTION_TICKS].reshape(1, ACTION_TICKS, 1, dim)
            + shooter_identity
            + target_address
        ).reshape(b, -1, dim)
        static_kv = torch.cat([terrain, action_flat], dim=1)

        visible = self._visibility(
            unit_features=unit_features,
            masked_units=masked_units,
            num_terrain=terrain.shape[1],
            num_blue=actions.shape[2],
        )

        # identity 리셋 준비 — query 열과 같은 배치의 (B, Q, id_dim)
        id_slice = self.layout["identity"]
        unit_identity = self.encoder.unit_identity(team_ids)              # (U, id)
        mission_identity = self.encoder.mission_identity(device)          # (id,)
        per_frame_identity = torch.cat(
            [
                unit_identity.reshape(1, 1, num_units, -1).expand(b, TOTAL_FRAMES, num_units, -1),
                mission_identity.reshape(1, 1, 1, -1).expand(b, TOTAL_FRAMES, 1, -1),
            ],
            dim=2,
        ).reshape(b, TOTAL_FRAMES * (num_units + 1), -1)

        for layer in self.layers:
            queries = layer(queries, static_kv, visible)
            # OA-WAM address reset: identity는 시간 불변이므로 층마다 원본으로 되쓴다
            queries = torch.cat(
                [queries[..., : id_slice.start], per_frame_identity], dim=-1
            )

        per_frame = queries.reshape(b, TOTAL_FRAMES, num_units + 1, dim)
        return {
            "unit_tokens": per_frame[:, :, :num_units],    # (B, 9, U, D)
            "mission_tokens": per_frame[:, :, num_units],  # (B, 9, D)
        }

    def _visibility(
        self,
        *,
        unit_features: torch.Tensor,
        masked_units: torch.Tensor,
        num_terrain: int,
        num_blue: int,
    ) -> torch.Tensor:
        """(B, Q, Q + S) bool. True=참조 허용.

        - 상태↔상태: 전방향. 단 사망 유닛의 history 토큰은 key에서 차단.
          (마스킹된 유닛의 h 토큰은 실측이 아니므로 차단하지 않는다)
        - 지형: 전부 허용.
        - 액션(발행틱 j): query 프레임 ≥ j+1 만 허용.
        """
        b, _, num_units, _ = unit_features.shape
        device = unit_features.device
        q_per_frame = num_units + 1
        q_len = TOTAL_FRAMES * q_per_frame

        # 상태 key 차단: history 프레임에서 죽어 있는 유닛 (alive=특징 7)
        state_key_ok = torch.ones(b, TOTAL_FRAMES, q_per_frame, dtype=torch.bool, device=device)
        alive_history = unit_features[:, :HISTORY_FRAMES, :, 7] > 0.0     # (B, 3, U)
        blocked = (~alive_history) & (~masked_units.unsqueeze(1))
        state_key_ok[:, :HISTORY_FRAMES, :num_units] = ~blocked
        state_key_ok = state_key_ok.reshape(b, q_len)

        visible_state = state_key_ok.unsqueeze(1).expand(b, q_len, q_len).clone()

        visible_terrain = torch.ones(b, q_len, num_terrain, dtype=torch.bool, device=device)

        # 액션: key index = j*num_blue + u → 발행틱 j. query 프레임 인덱스 fq ≥ j+1.
        frame_of_query = (
            torch.arange(q_len, device=device) // q_per_frame
        ).reshape(1, q_len, 1)
        tick_of_action = (
            torch.arange(ACTION_TICKS, device=device)
            .repeat_interleave(num_blue)
            .reshape(1, 1, -1)
        )
        visible_action = (frame_of_query >= tick_of_action + 1).expand(b, q_len, -1)

        return torch.cat([visible_state, visible_terrain, visible_action], dim=2)
