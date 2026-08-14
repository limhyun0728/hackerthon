"""value head 학습 — 라벨 지평(label_steps) 가변판. 기존 train_value.py는 수정하지 않는다.

배경 (2026-08-13 진단): CEM이 교착에서 접근을 안 뽑는 이유는 접근의 진짜 보상(V 프리미엄
+0.018)과 제자리 사격의 팬텀 보상(피해 과대의 argmax 꼬리)이 박빙이기 때문. 15스텝 라벨은
"접근 → 교전 회수"의 회수분을 절반만 담는다. 지평을 늘려 진짜 마진 자체를 키운다.

    python -m wm2.value.train_value_h \
        --episode-dirs 'output/statickv_rule/episode_*' ... \
        --label-steps 30 --device cuda:0 --output checkpoints/wm2_value_h30_rule.pt

주의: train_value.build_samples(실측 모드)는 모듈 전역 LABEL_STEPS를 호출 시점에 읽으므로
여기서 전역을 바꿔 재사용한다. build_imagined_samples는 label_steps 인자로 직접 넘긴다
(기본값이 def 시점에 묶여 있어 전역 변경이 안 통한다).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import train_value as tv
from .head import WM2ValueHead, save_value_head


def load_samples_h(patterns, *, label_steps: int, imagined=None):
    """train_value.load_samples와 동일하되 라벨 지평을 명시한다."""
    from tqdm import tqdm
    from ..data.episodes import load_episode
    from ..data.scenarios import assert_not_heldout, heldout_signatures

    tv.LABEL_STEPS = label_steps   # build_samples는 전역을 런타임에 읽는다 (위 주석)
    signatures = heldout_signatures()
    result = []
    mode = f"상상 입력 생성(h={label_steps})" if imagined else f"에피소드 로드(h={label_steps})"
    for d in tqdm(tv._expand(patterns), desc=mode, unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError, NotADirectoryError):
            continue
        assert_not_heldout(episode, signatures)
        if imagined is not None:
            model, heads, device = imagined
            samples = tv.build_imagined_samples(
                episode, model, heads, device, label_steps=label_steps
            )
        else:
            samples = tv.build_samples(episode)
        if samples is not None:
            result.append(samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--validation-dirs", nargs="+", default=["output/validation_shared_v2/episode_*"])
    parser.add_argument("--label-steps", type=int, default=30, help="V 라벨 지평 (원본 15)")
    parser.add_argument(
        "--world-model-checkpoint", default=None,
        help="주면 상상 입력 모드 (train_value와 동일 의미)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-every", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    imagined = None
    if args.world_model_checkpoint:
        from ..config import ModelConfig
        from ..model.heads import WM2Heads
        from ..model.predictor import WM2Predictor

        payload = torch.load(args.world_model_checkpoint, map_location=device, weights_only=False)
        wm = WM2Predictor(ModelConfig()).to(device); wm.load_state_dict(payload["model"]); wm.eval()
        wm_heads = WM2Heads(ModelConfig()).to(device); wm_heads.load_state_dict(payload["heads"]); wm_heads.eval()
        imagined = (wm, wm_heads, device)
        print(f"상상 입력 모드: ŝ_(t+6) 입력, 라벨 = 실제 {args.label_steps}초 진행분")

    train_samples = load_samples_h(args.episode_dirs, label_steps=args.label_steps, imagined=imagined)
    val_samples = load_samples_h(args.validation_dirs, label_steps=args.label_steps, imagined=imagined)
    print(
        f"train episodes={len(train_samples)} samples={sum(len(s.labels) for s in train_samples)} | "
        f"val episodes={len(val_samples)} samples={sum(len(s.labels) for s in val_samples)} | "
        f"label_steps={args.label_steps}"
    )

    model = WM2ValueHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    weights = np.asarray([len(s.labels) for s in train_samples], dtype=np.float64)
    weights /= weights.sum()

    best_mae = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        s = train_samples[int(rng.choice(len(train_samples), p=weights))]
        index = rng.choice(len(s.labels), size=min(args.batch_size, len(s.labels)), replace=False)
        pred = tv._forward(model, s, index, device)
        target = torch.from_numpy(s.labels[index]).to(device)
        loss = (pred - target).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.val_every == 0:
            metrics = tv.evaluate(model, val_samples, device)
            signs = " ".join(
                f"m{m}={metrics.get(f'sign_m{m}', float('nan')):+.2f}" for m in range(4)
            )
            print(
                f"step={step} loss={float(loss):.5f} val_mae={metrics['mae']:.4f} "
                f"(기준선 {metrics['baseline_mae']:.4f}) rho={metrics['rho_within_episode']:.3f} "
                f"목표거리부호[{signs}] elapsed={time.time()-started:.0f}s",
                flush=True,
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                save_value_head(Path(args.output), model)
    print(f"done best_val_mae={best_mae:.4f} label_steps={args.label_steps} → {args.output}")


if __name__ == "__main__":
    main()
