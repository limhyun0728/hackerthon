"""value head 학습: 15-step progress 증가분 회귀.

표본 = 실측 에피소드의 상태 s_t (t ≥ 6, 2틱 간격),
라벨 = progress(min(t+K, 종료)) − progress(t). 시뮬 불필요, 로그만 읽는다.

    python -m wm2.value.train_value \
        --episode-dirs 'output/statickv_rule/episode_*' 'output/blockfix/episode_*' \
        --device cuda:0 --output checkpoints/wm2_value.pt

측정 계약 (Level 1·2):
- 홀드아웃(validation_shared_v2) MAE — 상수 예측 기준선과 비교
- 임무별 MAE
- 같은 에피소드 내 상태들의 V 순위 vs 실제 증가분 순위 (Spearman)
- V vs 목표거리 상관의 임무별 부호 (reach 계열 음수, destroy_all ~0)
"""

from __future__ import annotations

import argparse
import glob
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.episodes import Episode, load_episode, objective_distance
from ..data.scenarios import assert_not_heldout, heldout_signatures
from ..data.windows import _unit_vector, _mission_vector, build_layout
from ..model.features import MISSION_TYPE_BY_NAME
from .head import WM2ValueHead, mission_progress, save_value_head

_REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_STEPS = 15
SAMPLE_STRIDE = 2
MIN_TICK = 6


@dataclass
class EpisodeSamples:
    """같은 layout을 공유하는 한 에피소드의 표본 묶음."""

    unit_features: np.ndarray     # (S, U, 10)
    mission_features: np.ndarray  # (S, 5)
    terrain_features: np.ndarray  # (T, 9)
    team_ids: np.ndarray          # (U,)
    labels: np.ndarray            # (S,)
    objective_dist: np.ndarray    # (S,) — 부호 검사용
    mission_type: int


def episode_progress(episode: Episode, tick: int) -> float:
    frame = episode.frames[tick]
    red_hp = sum(frame[u].hp for u in episode.red_ids if u in frame)
    blue_alive = sum(1 for u in episode.blue_ids if u in frame and frame[u].hp > 0.0)
    return mission_progress(
        mission_type=episode.mission_type,
        blue_alive=blue_alive,
        red_hp_total=red_hp,
        red_initial=len(episode.red_ids),
        objective_distance=objective_distance(episode, tick),
    )


def build_samples(episode: Episode) -> EpisodeSamples | None:
    layout = build_layout(episode)
    ticks = episode.ticks
    last = ticks[-1]
    progress_by_tick = {t: episode_progress(episode, t) for t in ticks}

    units, missions, labels, obj_dists = [], [], [], []
    for t in range(MIN_TICK, last + 1, SAMPLE_STRIDE):
        if t not in episode.frames or (t - 1) not in episode.frames:
            continue
        frame, prev = episode.frames[t], episode.frames[t - 1]
        units.append(
            np.stack(
                [
                    _unit_vector(frame[uid], prev.get(uid, frame[uid]), int(layout.team_ids[ui]))
                    for ui, uid in enumerate(layout.unit_ids)
                ]
            )
        )
        missions.append(_mission_vector(episode, t, frame))
        end = min(t + LABEL_STEPS, last)
        labels.append(progress_by_tick[end] - progress_by_tick[t])
        obj_dists.append(objective_distance(episode, t))
    if not units:
        return None
    return EpisodeSamples(
        unit_features=np.stack(units).astype(np.float32),
        mission_features=np.stack(missions).astype(np.float32),
        terrain_features=layout.terrain_features,
        team_ids=np.asarray(layout.team_ids),
        labels=np.asarray(labels, dtype=np.float32),
        objective_dist=np.asarray(obj_dists, dtype=np.float32),
        mission_type=episode.mission_type,
    )


@torch.no_grad()
def build_imagined_samples(
    episode: Episode,
    model,
    heads,
    device: torch.device,
    *,
    label_steps: int = LABEL_STEPS,
    stride: int = SAMPLE_STRIDE,
) -> EpisodeSamples | None:
    """상상 입력 표본: 입력 = 실제 명령을 월드모델에 넣어 그린 ŝ_{t+6}, 라벨 = 로그의 실제
    progress(t+21) − progress(t+6).

    V의 실전 입력 분포(상상 상태)와 학습 분포를 일치시킨다 — 실측 상태로 학습한 V가
    상상 상태 위에서 순위를 뒤집었던 실측(분해: V↔연속실현 −0.31)의 처방.
    월드모델을 재학습하면 이 표본도 재생성해야 한다 (V는 모델 버전에 결합된다).
    """
    from ..data.batch import collate
    from ..data.windows import build_windows
    from ..model.features import MAX_AMMO
    from ..model.rollout import assemble_hp, assemble_positions, clamp_physics
    from ..plan.score import value_input_from_assembled

    layout = build_layout(episode)
    ticks = episode.ticks
    last = ticks[-1]
    progress_by_tick = {t: episode_progress(episode, t) for t in ticks}
    windows = [
        w for w in build_windows(episode)
        if w.anchor_tick % stride == 0 and (w.anchor_tick + 8) <= last
    ]
    if not windows:
        return None

    units_out, mission_out, labels, obj_dists = [], [], [], []
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
        raw = assemble_positions(batch["anchor_xy"], pred["dpos"][:, 2:])
        hp = assemble_hp(batch["anchor_hp"], pred["ddmg"][:, 2:])
        clamped = clamp_physics(raw, batch["anchor_xy"], hp, batch["anchor_hp"] > 0)
        ammo_anchor = batch["unit_features"][:, 0, :, 2] * MAX_AMMO
        ammo_final = (ammo_anchor - pred["dammo"][:, -1].clamp_min(0.0) * MAX_AMMO).clamp_min(0.0)
        for i, w in enumerate(chunk):
            t6 = w.anchor_tick + 8
            uf, mf = value_input_from_assembled(
                positions=clamped[i : i + 1],
                hp=hp[i : i + 1],
                ammo=ammo_final[i : i + 1],
                heading=pred["heading"][i : i + 1, -1],
                team_ids=batch["team_ids"],
                mission_type=layout.mission_type,
                objective=layout.objective,
                time_remaining=max(0.0, (episode.duration_sec - t6) / episode.duration_sec),
            )
            units_out.append(uf[0].cpu().numpy())
            mission_out.append(mf[0].cpu().numpy())
            end = min(t6 + label_steps, last)
            labels.append(progress_by_tick[end] - progress_by_tick[t6])
            obj_dists.append(objective_distance(episode, t6))
    return EpisodeSamples(
        unit_features=np.stack(units_out).astype(np.float32),
        mission_features=np.stack(mission_out).astype(np.float32),
        terrain_features=layout.terrain_features,
        team_ids=np.asarray(layout.team_ids),
        labels=np.asarray(labels, dtype=np.float32),
        objective_dist=np.asarray(obj_dists, dtype=np.float32),
        mission_type=episode.mission_type,
    )


def _expand(patterns: list[str]) -> list[str]:
    return sorted(
        {
            d
            for p in patterns
            for d in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p)))
            if Path(d).is_dir()
        }
    )


def load_samples(
    patterns: list[str],
    *,
    check_heldout: bool = True,
    imagined: tuple | None = None,   # (world_model, heads, device) — 상상 입력 모드
) -> list[EpisodeSamples]:
    from tqdm import tqdm

    signatures = heldout_signatures() if check_heldout else {}
    result = []
    mode = "상상 입력 생성" if imagined is not None else "에피소드 로드"
    for d in tqdm(_expand(patterns), desc=mode, unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError, NotADirectoryError):
            continue
        if check_heldout:
            assert_not_heldout(episode, signatures)
        if imagined is not None:
            model, heads, device = imagined
            samples = build_imagined_samples(episode, model, heads, device)
        else:
            samples = build_samples(episode)
        if samples is not None:
            result.append(samples)
    return result


def _forward(model, s: EpisodeSamples, index: np.ndarray, device) -> torch.Tensor:
    unit = torch.from_numpy(s.unit_features[index]).to(device)
    mission = torch.from_numpy(s.mission_features[index]).to(device)
    terrain = torch.from_numpy(
        np.broadcast_to(s.terrain_features, (len(index),) + s.terrain_features.shape).copy()
    ).to(device)
    team = torch.from_numpy(s.team_ids).to(device)
    return model(
        unit_features=unit, terrain_features=terrain,
        mission_features=mission, team_ids=team,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denominator = math.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")


@torch.no_grad()
def evaluate(model, val_samples: list[EpisodeSamples], device) -> dict:
    model.eval()
    abs_err = {m: [] for m in range(4)}
    rhos = []
    sign_data = {m: ([], []) for m in range(4)}   # (V, obj_dist)
    all_labels = []
    for s in val_samples:
        index = np.arange(len(s.labels))
        pred = _forward(model, s, index, device).cpu().numpy()
        err = np.abs(pred - s.labels)
        abs_err[s.mission_type] += err.tolist()
        all_labels += s.labels.tolist()
        rhos.append(_spearman(pred, s.labels))
        finite = np.isfinite(s.objective_dist)
        sign_data[s.mission_type][0].extend(pred[finite].tolist())
        sign_data[s.mission_type][1].extend(s.objective_dist[finite].tolist())
    model.train()
    labels = np.asarray(all_labels)
    baseline = np.abs(labels - labels.mean()).mean()   # 상수 예측 기준선
    result = {
        "mae": float(np.mean([e for v in abs_err.values() for e in v])),
        "baseline_mae": float(baseline),
        "rho_within_episode": float(np.nanmean(rhos)),
    }
    for m, errors in abs_err.items():
        if errors:
            result[f"mae_m{m}"] = float(np.mean(errors))
    for m, (v, d) in sign_data.items():
        if len(v) > 10:
            result[f"sign_m{m}"] = _spearman(np.asarray(v), np.asarray(d))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--validation-dirs", nargs="+", default=["output/validation_shared_v2/episode_*"])
    parser.add_argument(
        "--world-model-checkpoint", default=None,
        help="주면 상상 입력 모드: 입력 = 이 모델이 그린 ŝ_{t+6} (V의 실전 입력 분포와 일치)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/wm2_value.pt")
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
        print("상상 입력 모드: 입력 = 월드모델이 그린 ŝ_(t+6), 라벨 = 로그의 실제 15초 증가분")

    train_samples = load_samples(args.episode_dirs, imagined=imagined)
    val_samples = load_samples(args.validation_dirs, imagined=imagined)
    n_train = sum(len(s.labels) for s in train_samples)
    print(f"train episodes={len(train_samples)} samples={n_train} | "
          f"val episodes={len(val_samples)} samples={sum(len(s.labels) for s in val_samples)}")

    model = WM2ValueHead().to(device)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    weights = np.asarray([len(s.labels) for s in train_samples], dtype=np.float64)
    weights /= weights.sum()

    best_mae = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        s = train_samples[int(rng.choice(len(train_samples), p=weights))]
        index = rng.choice(len(s.labels), size=min(args.batch_size, len(s.labels)), replace=False)
        pred = _forward(model, s, index, device)
        target = torch.from_numpy(s.labels[index]).to(device)
        loss = (pred - target).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.val_every == 0:
            metrics = evaluate(model, val_samples, device)
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
    print(f"done best_val_mae={best_mae:.4f} → {args.output}")


if __name__ == "__main__":
    main()
