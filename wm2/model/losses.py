"""손실 (설계 8절).

L = w_pos·L1(Δ위치) + w_dmg·MSE(Δ피해) + w_ammo·MSE(Δ탄약)
  + w_head·MSE(heading) + w_comp·BCE(completion)

적용 범위 (레이블 프레임 index 기준, 0..7 = h1..f6):
  - 미래 프레임 (2..7): 앵커 생존 유닛 전부
  - history 프레임 (0..1): **마스킹된** 앵커 생존 유닛만 — 가려진 h 복원 과제
mission 항은 completion 하나뿐. latent 손실은 config.latent_weight > 0일 때만 (기본 0).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..config import LossConfig

HISTORY_LABEL_FRAMES = 2   # 레이블 프레임 중 h1, h2


def residual_target_mask(
    pos_loss_mask: torch.Tensor,   # (B, U) bool — 앵커 생존
    masked_units: torch.Tensor,    # (B, U) bool
    label_frames: int,
) -> torch.Tensor:
    """(B, F_label, U) — 잔차 손실을 적용할 (프레임, 유닛)."""
    b, u = pos_loss_mask.shape
    mask = torch.zeros(b, label_frames, u, dtype=torch.bool, device=pos_loss_mask.device)
    mask[:, HISTORY_LABEL_FRAMES:] = pos_loss_mask.unsqueeze(1)
    mask[:, :HISTORY_LABEL_FRAMES] = (pos_loss_mask & masked_units).unsqueeze(1)
    return mask


def compute_losses(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    *,
    pos_loss_mask: torch.Tensor,
    masked_units: torch.Tensor,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    """outputs: heads.forward 결과. labels: dpos/ddmg/dammo/heading/completion 텐서."""
    selection = residual_target_mask(
        pos_loss_mask, masked_units, outputs["dpos"].shape[1]
    )
    count = selection.sum().clamp_min(1)

    def masked_mean(error: torch.Tensor) -> torch.Tensor:
        # error: (B, F, U) — selection 밖은 0으로 두고 선택 표본수로 나눈다
        return (error * selection).sum() / count

    loss_pos = masked_mean((outputs["dpos"] - labels["dpos"]).abs().sum(dim=-1))
    # 피해는 MSE(평균 학습) + 큰 가중치(config, run10 실측 근거) 조합이어야 한다.
    # 피해 라벨은 "절반은 0, 나머지는 크게"인 쏠림 분포(cf4_mid 실측: hold 평균
    # 10.0/중앙값 0.2HP)라 L1로 바꾸면 중앙값을 배워 기대 피해가 ~0으로 붕괴한다
    # (run10: hold 상상 1.9 vs 실측 5.9). 반대로 가중치가 작으면(구 8.0) 토큰별
    # 분리를 배울 압력이 부족해 무사격 계획에도 사격 평균이 번진다 (run8: approach
    # 상상 11 vs 라벨 평균 3.1). CEM 채점이 소비하는 건 기대 피해이므로 평균이 목표다.
    loss_dmg = masked_mean((outputs["ddmg"] - labels["ddmg"]).square())
    loss_ammo = masked_mean((outputs["dammo"] - labels["dammo"]).square())
    loss_heading = masked_mean((outputs["heading"] - labels["heading"]).square().sum(dim=-1))
    loss_completion = F.binary_cross_entropy_with_logits(
        outputs["completion_logit"], labels["completion"]
    )

    total = (
        config.position * loss_pos
        + config.damage * loss_dmg
        + config.ammo * loss_ammo
        + config.heading * loss_heading
        + config.completion * loss_completion
    )
    return {
        "loss": total,
        "loss_pos": loss_pos.detach(),
        "loss_dmg": loss_dmg.detach(),
        "loss_ammo": loss_ammo.detach(),
        "loss_heading": loss_heading.detach(),
        "loss_completion": loss_completion.detach(),
    }
