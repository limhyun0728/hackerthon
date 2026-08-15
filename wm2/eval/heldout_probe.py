"""홀드아웃 맵 평가 프로브 — RED 위치 오차 vs 정지가정, 맵별 분해 (평가 전용).

train_wm._load_all은 홀드아웃 에피소드가 학습에 섞이는 걸 막는다(옳은 가드).
이 스크립트는 반대 방향 — 그 맵들 위에서 WM의 예측 강건성을 재는 읽기 전용 평가다.
어떤 학습도 하지 않는다. "rule+CF+CEM 레시피가 미학습 지형에 강건한가"의 실측
(2026-08-15 플랫폼 준비 체크리스트 ①).

    python -m wm2.eval.heldout_probe \
        --checkpoint output/wm2_run13/wm2_best.pt \
        --episode-dirs 'output/heldout_rule/episode_*' 'output/heldout_rule2/episode_*' \
        --device cuda:1

참조 기준(학습 맵, validation_shared_v2에서 실측): run13 f2=2.3/7.4 f6=7.9/19.8.
홀드아웃 수치가 이 근방이면 강건, 정지가정 대비 우위가 사라지면 일반화 실패.
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import torch

from ..config import LossConfig, ModelConfig
from ..data.episodes import load_episode
from ..data.scenarios import _signature, heldout_signatures
from ..data.windows import build_windows
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor
from ..train import train_wm as tw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-windows", type=int, default=2500)
    args = parser.parse_args()

    from tqdm import tqdm

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WM2Predictor(ModelConfig()).to(device)
    model.load_state_dict(payload["model"]); model.eval()
    heads = WM2Heads(ModelConfig()).to(device)
    heads.load_state_dict(payload["heads"]); heads.eval()

    sigs = heldout_signatures()
    dirs = sorted({
        d for p in args.episode_dirs
        for d in (glob.glob(p) or glob.glob(str(tw._REPO_ROOT / p)))
        if Path(d).is_dir()
    })
    groups: dict[str, list] = defaultdict(list)
    for d in tqdm(dirs, desc="에피소드 로드", unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            continue
        name = sigs.get(_signature(episode.obstacles), "비홀드아웃")
        windows = build_windows(episode)
        if windows:
            groups[name].append(windows)

    loss_config = LossConfig()
    print(f"\n{args.checkpoint} — RED 위치 오차/정지가정(m)")
    print(f"{'맵':<12}{'ep':>4} | f1..f6")
    for name in sorted(groups):
        _, red_m, stat_m = tw.validate(
            model, heads, groups[name], device, loss_config, max_windows=args.max_windows
        )
        table = " ".join(f"f{k+1}={red_m[k]:.1f}/{stat_m[k]:.1f}" for k in range(6))
        print(f"{name:<12}{len(groups[name]):>4} | {table}")
    everything = [w for eps in groups.values() for w in eps]
    if len(groups) > 1 and everything:
        _, red_m, stat_m = tw.validate(
            model, heads, everything, device, loss_config, max_windows=args.max_windows
        )
        table = " ".join(f"f{k+1}={red_m[k]:.1f}/{stat_m[k]:.1f}" for k in range(6))
        print(f"{'전체':<12}{len(everything):>4} | {table}")


if __name__ == "__main__":
    main()
