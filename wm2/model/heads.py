"""출력 head (설계 8절). 전부 frame a 기준 누적 잔차 — 절대좌표 디코더는 없다.

레이블 프레임: h1..f6 (unit 토큰 프레임 index 1..8).
completion은 임무 미래 토큰 (프레임 index 3..8)에서 낸다.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


def _head(dim: int, hidden: int, out_dim: int, *, zero_init: bool) -> nn.Sequential:
    final = nn.Linear(hidden, out_dim)
    if zero_init:
        # 위치 Δ는 "안 움직였다"(0)에서 시작하는 게 안전한 기저다.
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), final)


class WM2Heads(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim, hidden = config.embedding_dim, config.hidden_dim
        self.dpos = _head(dim, hidden, 2, zero_init=True)
        self.ddmg = _head(dim, hidden, 1, zero_init=False)
        self.dammo = _head(dim, hidden, 1, zero_init=False)
        self.heading = _head(dim, hidden, 2, zero_init=False)
        self.completion = _head(dim, hidden, 1, zero_init=False)

    def forward(
        self, unit_tokens: torch.Tensor, mission_tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """unit_tokens (B,9,U,D), mission_tokens (B,9,D) → 레이블과 같은 shape."""
        label_tokens = unit_tokens[:, 1:]                    # (B, 8, U, D) h1..f6
        return {
            "dpos": self.dpos(label_tokens),                 # (B, 8, U, 2) 월드 단위
            "ddmg": self.ddmg(label_tokens).squeeze(-1),     # (B, 8, U)  /MAX_HP 스케일
            "dammo": self.dammo(label_tokens).squeeze(-1),   # (B, 8, U)
            "heading": self.heading(label_tokens),           # (B, 8, U, 2)
            "completion_logit": self.completion(mission_tokens[:, 3:]).squeeze(-1),  # (B, 6)
        }
