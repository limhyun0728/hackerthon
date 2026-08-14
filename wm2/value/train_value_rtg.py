"""value head 학습 — return-to-go 라벨판. 기존 train_value.py는 수정하지 않는다.

배경 (2026-08-14 진단): 15스텝 델타 라벨의 두 병리가 loop4/5/6 완주로 실측 확정됐다.
① 결승선 배회 — 완주 상태는 잔여 델타 0이라 "지금 완주"(gain +0.1, V 0)와 "직전 배회"
(gain 0, V +0.1)가 λ=1에서 동점, λ>1에서 배회 우세 (λ 용량-반응 22>19>10, McNemar p≈.003).
② 접근 회수분 절단 — 접근→교전 회수가 15스텝 창 밖이면 라벨에서 잘려 프리미엄 +0.018.

라벨을 에피소드 끝까지의 할인 합으로 바꾼다:

    rtg[t] = (progress[t+1] − progress[t]) + γ·rtg[t+1],  rtg[last] = 0

완주가 엄밀 우위가 되고(미루면 γ 할인 손해), 창 밖 회수분도 감쇠만 될 뿐 전액 반영된다.
에피소드 종료는 게임 규칙상 실제 종말(60초 제한)이고 V 입력에 time_remaining이 있으므로
끝 근처의 작은 라벨은 절단 편향이 아니라 참값이다 — 기본은 전 구간 사용(--tail-exclude로
제외 가능). CEM 채점에서 V 가중은 벨만 정합상 γ^6(γ=0.97이면 ≈0.83)을 쓴다.

    # 실측 입력판 (프로브 대조용)
    python -m wm2.value.train_value_rtg \
        --episode-dirs 'output/statickv_rule/episode_*' 'output/blockfix/episode_*' \
                       'output/noblock/episode_*' 'output/blockfix2/episode_*' \
        --device cuda:0 --output checkpoints/wm2_value_rtg_real.pt

    # 상상 입력판 (실전용 — WM 재학습 시 재생성)
    ... --world-model-checkpoint output/wm2_run12/wm2_latest.pt \
        --output checkpoints/wm2_value_rtg_run12.pt

주의: loop3 등 CEM 자기생성 궤적은 넣지 말 것 — destroy 총 진행 이동이 rule의 1/3인
빈혈 데이터로 실측됨 (2026-08-14, value-head-status 메모).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from . import train_value as tv
from .head import WM2ValueHead, save_value_head


def _rtg_by_tick(episode, gamma: float) -> dict[int, float]:
    """진행도 return-to-go: rtg[t] = Σ_{k≥t} γ^{k−t}·(p[k+1]−p[k]), 끝에서 0."""
    ticks = sorted(episode.ticks)
    p = {t: tv.episode_progress(episode, t) for t in ticks}
    rtg = {ticks[-1]: 0.0}
    for a, b in zip(reversed(ticks[:-1]), reversed(ticks[1:])):
        rtg[a] = (p[b] - p[a]) + gamma * rtg[b]
    return rtg


def _label_ticks_real(episode) -> list[int]:
    """build_samples의 표본 시각을 그대로 재현한다 (train_value.py:74-76과 결합)."""
    last = episode.ticks[-1]
    return [
        t for t in range(tv.MIN_TICK, last + 1, tv.SAMPLE_STRIDE)
        if t in episode.frames and (t - 1) in episode.frames
    ]


def _label_ticks_imagined(episode) -> list[int]:
    """build_imagined_samples의 라벨 시각(t6)을 그대로 재현한다 (train_value.py:130-133)."""
    from ..data.windows import build_windows

    last = episode.ticks[-1]
    return [
        w.anchor_tick + 8
        for w in build_windows(episode)
        if w.anchor_tick % tv.SAMPLE_STRIDE == 0 and (w.anchor_tick + 8) <= last
    ]


def _relabel(samples: tv.EpisodeSamples, episode, *, gamma: float,
             imagined: bool, tail_exclude: int) -> tv.EpisodeSamples | None:
    ticks = _label_ticks_imagined(episode) if imagined else _label_ticks_real(episode)
    assert len(ticks) == len(samples.labels), (
        f"라벨 시각 재현 불일치: {len(ticks)} vs {len(samples.labels)} — "
        "train_value.py의 표본 인덱싱이 바뀌었는지 확인"
    )
    rtg = _rtg_by_tick(episode, gamma)
    labels = np.asarray([rtg[t] for t in ticks], dtype=np.float32)
    keep = np.asarray([episode.ticks[-1] - t >= tail_exclude for t in ticks])
    if not keep.any():
        return None
    return replace(
        samples,
        unit_features=samples.unit_features[keep],
        mission_features=samples.mission_features[keep],
        labels=labels[keep],
        objective_dist=samples.objective_dist[keep],
    )


def load_samples_rtg(patterns, *, gamma: float, imagined, tail_exclude: int):
    from tqdm import tqdm

    from ..data.episodes import load_episode
    from ..data.scenarios import assert_not_heldout, heldout_signatures

    signatures = heldout_signatures()
    result = []
    mode = "상상 입력 생성(rtg)" if imagined else "에피소드 로드(rtg)"
    for d in tqdm(tv._expand(patterns), desc=mode, unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError, NotADirectoryError):
            continue
        assert_not_heldout(episode, signatures)
        if imagined is not None:
            model, heads, device = imagined
            samples = tv.build_imagined_samples(episode, model, heads, device)
        else:
            samples = tv.build_samples(episode)
        if samples is None:
            continue
        samples = _relabel(
            samples, episode, gamma=gamma,
            imagined=imagined is not None, tail_exclude=tail_exclude,
        )
        if samples is not None:
            result.append(samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--validation-dirs", nargs="+", default=["output/validation_shared_v2/episode_*"])
    parser.add_argument("--gamma", type=float, default=0.97, help="틱당 할인율 (반감기 ~23틱)")
    parser.add_argument("--tail-exclude", type=int, default=0,
                        help="에피소드 끝에서 N틱 미만 남은 표본 제외 (기본 0 — 위 docstring 근거)")
    parser.add_argument("--world-model-checkpoint", default=None,
                        help="주면 상상 입력 모드 (train_value와 동일 의미)")
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
        print(f"상상 입력 모드: ŝ_(t+6) 입력, 라벨 = γ={args.gamma} return-to-go")

    train_samples = load_samples_rtg(
        args.episode_dirs, gamma=args.gamma, imagined=imagined, tail_exclude=args.tail_exclude
    )
    val_samples = load_samples_rtg(
        args.validation_dirs, gamma=args.gamma, imagined=imagined, tail_exclude=args.tail_exclude
    )
    print(
        f"train episodes={len(train_samples)} samples={sum(len(s.labels) for s in train_samples)} | "
        f"val episodes={len(val_samples)} samples={sum(len(s.labels) for s in val_samples)} | "
        f"γ={args.gamma} tail_exclude={args.tail_exclude}"
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
                f"step={step} loss={float(loss.detach()):.5f} val_mae={metrics['mae']:.4f} "
                f"(기준선 {metrics['baseline_mae']:.4f}) rho={metrics['rho_within_episode']:.3f} "
                f"목표거리부호[{signs}] elapsed={time.time()-started:.0f}s",
                flush=True,
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                save_value_head(Path(args.output), model)
    print(f"done best_val_mae={best_mae:.4f} γ={args.gamma} → {args.output}")


if __name__ == "__main__":
    main()
