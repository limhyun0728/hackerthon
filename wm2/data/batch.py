"""window 배치 조립. 한 배치는 같은 episode(layout)의 window로만 만든다 (구 규약).

entity 구성이 다르면 슬롯 축이 안 맞아 배치가 성립하지 않는다 — 병력 2~10 무작위라
episode 간 조합이 사실상 안 겹치기 때문 (구 시스템 주석과 같은 이유).
"""

from __future__ import annotations

import numpy as np
import torch

from ..model.features import denorm_x, denorm_y, MAX_HP
from .windows import Window


def collate(
    windows: list[Window],
    masked_units: list[np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    layout = windows[0].layout
    for w in windows[1:]:
        if w.layout is not layout and w.layout.unit_ids != layout.unit_ids:
            raise ValueError("한 배치는 같은 episode layout의 window로만 만든다")

    def stack(name: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(w, name) for w in windows])).to(device)

    unit_features = stack("unit_features")
    b, _, num_units, _ = unit_features.shape

    masked = torch.zeros(b, num_units, dtype=torch.bool, device=device)
    for i, slots in enumerate(masked_units):
        if len(slots):
            masked[i, torch.as_tensor(slots, device=device, dtype=torch.long)] = True

    # anchor 절대값 (frame a): 조립·평가용
    anchor_x = torch.tensor(
        [[denorm_x(float(v)) for v in w.unit_features[0, :, 3]] for w in windows],
        dtype=torch.float32, device=device,
    )
    anchor_y = torch.tensor(
        [[denorm_y(float(v)) for v in w.unit_features[0, :, 4]] for w in windows],
        dtype=torch.float32, device=device,
    )

    return {
        "unit_features": unit_features,
        "terrain_features": torch.from_numpy(
            np.broadcast_to(layout.terrain_features, (b,) + layout.terrain_features.shape).copy()
        ).to(device),
        "mission_features": stack("mission_features"),
        "actions": stack("actions"),
        "team_ids": torch.as_tensor(np.asarray(layout.team_ids), device=device),
        "masked_units": masked,
        "labels": {
            "dpos": stack("dpos"),
            "ddmg": stack("ddmg"),
            "dammo": stack("dammo"),
            "heading": stack("heading"),
            "completion": stack("completion"),
        },
        "pos_loss_mask": stack("pos_loss_mask"),
        "anchor_xy": torch.stack([anchor_x, anchor_y], dim=-1),
        "anchor_hp": unit_features[:, 0, :, 1] * MAX_HP,
    }
