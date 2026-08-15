"""후보 채점 (설계 10절 + 2026-08-15 레버 2: 생존 항).

score(후보) = [progress(ŝ₆) − progress(s₀)] + β·[H(ŝ₆) − H(s₀)] + λ·V(ŝ₆)

- progress 항: 상상된 상태에 승리 조건의 연속 완화를 그대로 계산 — 손가중치 없음
- 생존 항 (β=SURVIVAL_BETA): H = 아군 HP 합 / (아군 수·MAX_HP). 최종 판정이 생존+완료를
  요구하는데 progress는 완료만 봐서, 창 안의 출혈이 채점상 공짜였다 (100쌍×3런 실측:
  V 팔이 λ=0의 최종 승리 10을 2~4로 깎은 원인). λ=0 팔도 이 항은 받는다 — 목적함수
  자체의 수리이지 V의 기능이 아니다.
- V: ŝ₆부터 에피소드 끝까지의 γ-할인 잔여 보상(rtg) 예측 — 같은 δ 정의로 학습
- 모든 후보가 같은 s₀에서 출발하므로 s₀ 항들은 순위에 영향 없는 공통 상수
"""

from __future__ import annotations

import torch

from ..config import SURVIVAL_BETA
from ..model.features import (
    MAX_AMMO,
    MAX_HP,
    MISSION_DESTROY_ALL,
    MISSION_DESTROY_AND_REACH,
    OBJECTIVE_RADIUS,
    TeamId,
    norm_x,
    norm_y,
)
from ..value.head import OBJECTIVE_GAP_SCALE, WM2ValueHead


def progress_batch(
    positions: torch.Tensor,   # (B, U, 2) 월드 단위
    hp: torch.Tensor,          # (B, U) 0..MAX_HP
    team_ids: torch.Tensor,    # (U,)
    mission_type: int,
    objective: tuple[float, float],
) -> torch.Tensor:
    """(B,) — mission_progress의 배치 버전. 상상 상태에도 실측에도 같은 산술."""
    blue = (team_ids == int(TeamId.BLUE)).reshape(1, -1)
    red = (team_ids == int(TeamId.RED)).reshape(1, -1)
    alive = hp > 0.0

    red_total = (red.float() * MAX_HP).sum(dim=1).clamp_min(1.0)
    destroy = 1.0 - (hp * red.float()).sum(dim=1) / red_total

    obj = torch.tensor(objective, dtype=positions.dtype, device=positions.device)
    dist = (positions - obj.reshape(1, 1, 2)).norm(dim=-1)              # (B, U)
    blue_alive = alive & blue
    dist_masked = torch.where(blue_alive, dist, torch.full_like(dist, 1e6))
    min_dist = dist_masked.min(dim=1).values
    reach = 1.0 - ((min_dist - OBJECTIVE_RADIUS).clamp_min(0.0) / OBJECTIVE_GAP_SCALE).clamp(max=1.0)

    if mission_type == MISSION_DESTROY_ALL:
        progress = destroy
    elif mission_type == MISSION_DESTROY_AND_REACH:
        progress = torch.minimum(destroy, reach)   # AND는 min (구 교훈: 가중평균 금지)
    else:
        progress = reach

    any_blue_alive = blue_alive.any(dim=1)
    return torch.where(any_blue_alive, progress, torch.zeros_like(progress))


def value_input_from_assembled(
    *,
    positions: torch.Tensor,   # (C, F, U, 2) 물리 클램프 후
    hp: torch.Tensor,          # (C, F, U)
    ammo: torch.Tensor,        # (C, U) — ŝ₆ 시점
    heading: torch.Tensor,     # (C, U, 2) 예측 cos,sin (f6)
    team_ids: torch.Tensor,
    mission_type: int,
    objective: tuple[float, float],
    time_remaining: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ŝ₆를 value head 입력 (unit_features, mission_features)로 조립한다."""
    c, _, num_units, _ = positions.shape
    final_pos = positions[:, -1]
    final_hp = hp[:, -1]
    velocity = positions[:, -1] - positions[:, -2]
    alive = (final_hp > 0.0).float()
    xn = (final_pos[..., 0] - (-20.0)) / 40.0 * 2.0 - 1.0
    yn = (final_pos[..., 1] - (-15.0)) / 25.0 * 2.0 - 1.0
    unit_features = torch.stack(
        [
            team_ids.float().reshape(1, -1).expand(c, num_units),
            final_hp / MAX_HP,
            ammo / MAX_AMMO,
            xn,
            yn,
            heading[..., 0],
            heading[..., 1],
            alive,
            velocity[..., 0] * alive,
            velocity[..., 1] * alive,
        ],
        dim=-1,
    )
    completion = progress_batch(final_pos, final_hp, team_ids, mission_type, objective) >= 1.0
    mission_features = torch.stack(
        [
            torch.full((c,), float(mission_type), device=positions.device),
            torch.full((c,), norm_x(objective[0]), device=positions.device),
            torch.full((c,), norm_y(objective[1]), device=positions.device),
            torch.full((c,), max(0.0, time_remaining), device=positions.device),
            completion.float(),
        ],
        dim=-1,
    )
    return unit_features, mission_features


def score_candidates(
    *,
    positions: torch.Tensor,     # (C, 6, U, 2) 상상·클램프된 절대 위치 (f1..f6)
    hp: torch.Tensor,            # (C, 6, U)
    ammo_final: torch.Tensor,    # (C, U)
    heading_final: torch.Tensor, # (C, U, 2)
    current_positions: torch.Tensor,  # (U, 2) — s₀ = h2 실측
    current_hp: torch.Tensor,         # (U,)
    team_ids: torch.Tensor,
    terrain_features: torch.Tensor,   # (T, 9)
    mission_type: int,
    objective: tuple[float, float],
    time_remaining: float,
    value_head: WM2ValueHead | None,
    lam: float,
) -> dict[str, torch.Tensor]:
    c = positions.shape[0]
    progress_now = progress_batch(
        current_positions.unsqueeze(0), current_hp.unsqueeze(0),
        team_ids, mission_type, objective,
    )
    progress_end = progress_batch(positions[:, -1], hp[:, -1], team_ids, mission_type, objective)
    blue = (team_ids == int(TeamId.BLUE)).float()
    denom = (blue.sum() * MAX_HP).clamp_min(1.0)
    survival_now = (current_hp * blue).sum() / denom
    survival_end = (hp[:, -1] * blue.reshape(1, -1)).sum(dim=1) / denom
    gain = (progress_end - progress_now) + SURVIVAL_BETA * (survival_end - survival_now)

    value = torch.zeros_like(gain)
    if value_head is not None and lam != 0.0:
        unit_features, mission_features = value_input_from_assembled(
            positions=positions, hp=hp, ammo=ammo_final, heading=heading_final,
            team_ids=team_ids, mission_type=mission_type, objective=objective,
            time_remaining=time_remaining,
        )
        terrain = terrain_features.unsqueeze(0).expand(c, -1, -1)
        with torch.no_grad():
            value = value_head(
                unit_features=unit_features, terrain_features=terrain,
                mission_features=mission_features, team_ids=team_ids,
            )
    return {"score": gain + lam * value, "gain": gain, "value": value}
