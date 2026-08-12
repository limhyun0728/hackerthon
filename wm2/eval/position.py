"""판정 게이트 1 (설계 12절): RED 미래 위치 오차 vs 정지 가정.

    python -m hackerthon.wm2.eval.position \
        --checkpoint output/wm2_run1/wm2_best.pt \
        --episode-dirs 'output/statickv_rule/episode_*' --device cuda:0

f1..f6 = 현재(h2) 기준 t+1..t+6. 게이트: **RED t+2(f2) < 9.8m**.
마스킹 없이(전량 관측) 잰다 — 계획 경로와 같은 조건.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]

from ..config import ModelConfig
from ..data.batch import collate
from ..data.episodes import load_episode
from ..data.windows import build_windows
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor


@torch.no_grad()
def evaluate(checkpoint_path: str, episode_dirs: list[str], device_name: str, *, every: int = 5) -> None:
    device = torch.device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WM2Predictor(ModelConfig()).to(device)
    heads = WM2Heads(ModelConfig()).to(device)
    model.load_state_dict(payload["model"])
    heads.load_state_dict(payload["heads"])
    model.eval(); heads.eval()

    dirs = sorted(
        {
            d
            for p in episode_dirs
            for d in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p)))
            if Path(d).is_dir()
        }
    )
    # 학습 분리와 같은 규칙(매 5번째 = 검증)만 평가한다
    dirs = [d for i, d in enumerate(dirs) if i % every == every - 1]

    error_sum = np.zeros(6); stationary_sum = np.zeros(6); count = 0
    for d in dirs:
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            continue
        windows = build_windows(episode, stride=3)
        if not windows:
            continue
        for start in range(0, len(windows), 32):
            chunk = windows[start : start + 32]
            batch = collate(chunk, [np.zeros(0, dtype=np.int64)] * len(chunk), device)
            out = model(
                unit_features=batch["unit_features"],
                terrain_features=batch["terrain_features"],
                mission_features=batch["mission_features"],
                actions=batch["actions"],
                team_ids=batch["team_ids"],
                masked_units=batch["masked_units"],
            )
            pred = heads(out["unit_tokens"], out["mission_tokens"])
            red = batch["team_ids"] == 1
            alive = batch["pos_loss_mask"][:, red]
            diff = pred["dpos"][:, 2:][:, :, red] - batch["labels"]["dpos"][:, 2:][:, :, red]
            err = diff.norm(dim=-1)
            # 정지가정 = "현재(h2)에서 안 움직인다". a 기준 잔차에서 h2 잔차를 빼 현재 기준으로.
            stat = (
                batch["labels"]["dpos"][:, 2:][:, :, red]
                - batch["labels"]["dpos"][:, 1:2][:, :, red]
            ).norm(dim=-1)
            mask = alive.unsqueeze(1).expand_as(err)
            for k in range(6):
                error_sum[k] += float(err[:, k][mask[:, k]].sum())
                stationary_sum[k] += float(stat[:, k][mask[:, k]].sum())
            count += int(mask[:, 0].sum())

    red_m = error_sum / max(1, count) * 10.0
    stat_m = stationary_sum / max(1, count) * 10.0
    print(f"episodes={len(dirs)} 표본={count}")
    print("horizon   모델(m)  정지가정(m)")
    for k in range(6):
        gate = "  ← 게이트(<9.8m)" if k == 1 else ""
        print(f"t+{k+1}      {red_m[k]:7.1f}  {stat_m[k]:9.1f}{gate}")
    verdict = "통과" if red_m[1] < 9.8 else "실패"
    print(f"게이트 1 (RED t+2 < 9.8m): {red_m[1]:.1f}m → {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episode_dirs, args.device)


if __name__ == "__main__":
    main()
