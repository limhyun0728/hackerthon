"""조립 (설계 9절): x̂_k = x(a) + Δ̂_k, 물리 클램프, 임무 슬롯 완성.

학습 gradient에는 안 섞인다 — 계획·평가·표시 전용 (구 _enforce_unit_physics 관례).
ENGAGE 게이트는 없다: "무리한 ENGAGE → 아무 일 없음"은 계획 명령 토큰으로 학습된
실행 의미론이 담당한다 (설계 6절).
"""

from __future__ import annotations

import torch

from .features import MAX_HP, MAX_MOVE_PER_STEP


def assemble_positions(
    anchor_xy: torch.Tensor,   # (B, U, 2) 월드 단위 — frame a 위치
    dpos: torch.Tensor,        # (B, F, U, 2) 예측 잔차 (월드 단위)
) -> torch.Tensor:
    """(B, F, U, 2) 절대 위치. 잔차가 빼는 프레임 = 여기가 더하는 프레임 (a)."""
    return anchor_xy.unsqueeze(1) + dpos


def assemble_hp(
    anchor_hp: torch.Tensor,   # (B, U) — frame a HP (0..MAX_HP)
    ddmg: torch.Tensor,        # (B, F, U) 예측 누적 피해 (/MAX_HP 스케일)
) -> torch.Tensor:
    """(B, F, U) 절대 HP. 피해는 음수가 될 수 없고 HP는 0 아래로 안 내려간다."""
    damage = ddmg.clamp_min(0.0) * MAX_HP
    return (anchor_hp.unsqueeze(1) - damage).clamp_min(0.0)


def clamp_physics(
    positions: torch.Tensor,   # (B, F, U, 2) 조립된 절대 위치
    anchor_xy: torch.Tensor,   # (B, U, 2)
    hp: torch.Tensor,          # (B, F, U) 조립된 HP
    anchor_alive: torch.Tensor,  # (B, U) bool
) -> torch.Tensor:
    """프레임 간 이동 ≤ MAX_MOVE_PER_STEP, 사망 시 그 자리 동결.

    방향은 살리고 크기만 줄인다. anchor에서 이미 죽어 있으면 anchor에 고정.
    """
    frames = []
    previous = anchor_xy
    alive = anchor_alive.clone()
    for step in range(positions.shape[1]):
        target = positions[:, step]
        delta = target - previous
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        scale = (MAX_MOVE_PER_STEP / distance).clamp(max=1.0)
        moved = previous + delta * scale
        alive = alive & (hp[:, step] > 0.0)
        current = torch.where(alive.unsqueeze(-1), moved, previous)
        frames.append(current)
        previous = current
    return torch.stack(frames, dim=1)
