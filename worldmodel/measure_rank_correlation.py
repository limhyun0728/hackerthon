"""JEPA rollout 채점이 DEVS rollout 채점의 순위를 얼마나 재현하는지 측정한다.

hybrid prefilter(JEPA로 대량 후보 선별 → DEVS로 상위 K 검증)의 전제 조건을
검증하는 도구다. 같은 후보 집합에 대해 두 backend의 점수를 모두 계산해서
Spearman 순위 상관, top-K recall, decoder 복원 오차를 보고한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from hackerthon.worldmodel.actions import build_action_batch_from_v2_run
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    ObservedActionWindow,
    build_initial_distribution,
    rollout_with_world_model,
    sample_future_action_plans,
    score_future_features_torch,
)
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig
from hackerthon.worldmodel.slots import ObjectType, build_slot_batch_from_v2_run, load_v2_config

UNIT_HP_INDEX = 1
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4
X_SCALE = (WORLD_X_MAX - WORLD_X_MIN) * 0.5
Y_SCALE = (WORLD_Y_MAX - WORLD_Y_MIN) * 0.5


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DEVSObjectCentricWorldModel, ObjectSlotModelConfig]:
    """학습 checkpoint에서 모델과 설정을 복원한다."""
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(payload["model_config"])
    config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """두 점수 배열의 Spearman 순위 상관을 계산한다."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("spearman 입력은 같은 길이의 1차원 배열이어야 한다")
    rank_a = a.argsort().argsort().astype(np.float64)
    rank_b = b.argsort().argsort().astype(np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = math.sqrt(float((rank_a * rank_a).sum()) * float((rank_b * rank_b).sum()))
    if denom <= 0.0:
        return 0.0
    return float((rank_a * rank_b).sum() / denom)


def top_k_recall(true_scores: np.ndarray, proxy_scores: np.ndarray, *, k: int, keep: int) -> float:
    """DEVS 기준 top-k 후보 중 JEPA top-keep에 살아남는 비율."""
    true_top = set(np.argsort(true_scores)[::-1][:k].tolist())
    proxy_keep = set(np.argsort(proxy_scores)[::-1][:keep].tolist())
    return len(true_top & proxy_keep) / float(k)


def snapshot_rows_at(run_dir: Path, time_sec: float) -> list[dict[str, object]]:
    """soldier_log에서 특정 시점의 unit row를 수치형으로 읽는다."""
    rows: list[dict[str, object]] = []
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["time"]) != float(time_sec):
                continue
            rows.append(
                {
                    "time": float(row["time"]),
                    "id": int(row["id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "heading": float(row["heading"]),
                    "hp": float(row["hp"]),
                    "ammo": int(float(row["ammo"])),
                }
            )
    if not rows:
        raise ValueError(f"time={time_sec} unit row가 없다")
    return rows


def reconstruction_errors(
    jepa_features: torch.Tensor,
    devs_features: torch.Tensor,
    type_ids: np.ndarray,
) -> tuple[float, float]:
    """같은 후보에 대한 JEPA 예측과 DEVS 실제 미래의 unit feature 오차."""
    unit_indices = torch.as_tensor(
        np.flatnonzero(type_ids == int(ObjectType.UNIT)),
        device=jepa_features.device,
        dtype=torch.long,
    )
    jepa_units = jepa_features.index_select(2, unit_indices)
    devs_units = devs_features.index_select(2, unit_indices)
    dx = (jepa_units[..., UNIT_X_INDEX] - devs_units[..., UNIT_X_INDEX]) * X_SCALE
    dy = (jepa_units[..., UNIT_Y_INDEX] - devs_units[..., UNIT_Y_INDEX]) * Y_SCALE
    pos_error_m = float(torch.sqrt(dx * dx + dy * dy).mean())
    hp_error = float((jepa_units[..., UNIT_HP_INDEX] - devs_units[..., UNIT_HP_INDEX]).abs().mean())
    return pos_error_m, hp_error


def _times_with_full_context(run_dir: Path, history_frames: int) -> list[float]:
    """history state와 전이 command가 모두 존재하는 결정 시점 목록."""
    state_times: set[float] = set()
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state_times.add(float(row["time"]))
    command_times: set[float] = set()
    with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            command_times.add(float(row["time"]))
    valid: list[float] = []
    for time_sec in sorted(state_times):
        history = [time_sec - offset for offset in reversed(range(history_frames))]
        if all(t in state_times for t in history) and all(t in command_times for t in history[:-1]):
            valid.append(time_sec)
    return valid


def measure_tick(
    *,
    run_dir: Path,
    time_sec: float,
    model: DEVSObjectCentricWorldModel,
    model_config: ObjectSlotModelConfig,
    cem_config: CEMConfig,
    obstacles: list,
    duration_sec: float,
    device: torch.device,
    seed: int,
    top_k: int,
    keep: int,
) -> dict[str, float]:
    """한 결정 시점에서 두 backend 점수를 비교한다."""
    current_batch = build_slot_batch_from_v2_run(run_dir, time_sec)
    history_times = [time_sec - offset for offset in reversed(range(model_config.history_frames))]
    history_batches = tuple(build_slot_batch_from_v2_run(run_dir, t) for t in history_times)
    observed_batches = tuple(
        build_action_batch_from_v2_run(run_dir, command_time_sec=t, state_time_sec=t)
        for t in history_times[:-1]
    )
    observed = ObservedActionWindow(
        action_features=torch.stack(
            [torch.as_tensor(a.features, dtype=torch.float32, device=device) for a in observed_batches], dim=0
        ),
        action_unit_ids=torch.stack(
            [torch.as_tensor(a.unit_ids, dtype=torch.long, device=device) for a in observed_batches], dim=0
        ),
        issued_mask=torch.stack(
            [torch.as_tensor(a.issued_mask, dtype=torch.bool, device=device) for a in observed_batches], dim=0
        ),
    )

    distribution = build_initial_distribution(current_batch, cem_config, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    plans = sample_future_action_plans(
        distribution=distribution,
        current_batch=current_batch,
        config=cem_config,
        generator=generator,
        device=device,
    )

    snapshot = snapshot_from_slot_rows(
        unit_rows=snapshot_rows_at(run_dir, time_sec),
        obstacles=obstacles,
        base_time_sec=time_sec,
        episode_duration_sec=duration_sec,
    )
    devs_features = rollout_plans_with_devs(plans=plans, snapshot=snapshot, seed=seed, device=device)
    jepa_features = rollout_with_world_model(
        model=model,
        history_batches=history_batches,
        observed_actions=observed,
        future_plans=plans,
        device=device,
    )
    if not isinstance(jepa_features, torch.Tensor):
        jepa_features = torch.as_tensor(jepa_features, device=device)

    devs_scores = score_future_features_torch(current_batch=current_batch, future_features=devs_features)
    jepa_scores = score_future_features_torch(current_batch=current_batch, future_features=jepa_features)
    devs_np = devs_scores.detach().cpu().numpy()
    jepa_np = jepa_scores.detach().cpu().numpy()
    pos_error_m, hp_error = reconstruction_errors(jepa_features, devs_features, current_batch.type_ids)
    return {
        "time": float(time_sec),
        "spearman": spearman(devs_np, jepa_np),
        "recall": top_k_recall(devs_np, jepa_np, k=top_k, keep=keep),
        "pos_error_m": pos_error_m,
        "hp_error": hp_error,
    }


def main(argv: Iterable[str] | None = None) -> None:
    """실험 output root의 episode들에서 rank correlation을 측정한다."""
    parser = argparse.ArgumentParser(description="JEPA vs DEVS rollout 채점 순위 상관 측정")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--max-ticks", type=int, default=30)
    parser.add_argument("--tick-stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--keep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    model, model_config = load_model(args.checkpoint, device)
    cem_config = CEMConfig(
        num_candidates=args.candidates,
        num_elites=max(2, args.candidates // 8),
        num_iterations=1,
        future_horizon=model_config.pred_frames,
        seed=args.seed,
    )
    print(
        f"checkpoint={args.checkpoint} pred_frames={model_config.pred_frames} "
        f"candidates={args.candidates} top{args.top_k} recall@keep{args.keep}"
    )

    episode_dirs = sorted(d for d in args.output_root.iterdir() if d.is_dir())
    if not episode_dirs:
        raise ValueError(f"{args.output_root}에 episode 디렉터리가 없다")

    results: list[dict[str, float]] = []
    skipped = 0
    for run_dir in episode_dirs:
        if len(results) >= args.max_ticks:
            break
        config = load_v2_config(run_dir)
        obstacles = config["obstacles"]
        duration_sec = float(config["duration"])
        for time_sec in _times_with_full_context(run_dir, model_config.history_frames)[:: args.tick_stride]:
            if len(results) >= args.max_ticks:
                break
            if time_sec + model_config.pred_frames > duration_sec:
                continue
            try:
                tick = measure_tick(
                    run_dir=run_dir,
                    time_sec=time_sec,
                    model=model,
                    model_config=model_config,
                    cem_config=cem_config,
                    obstacles=obstacles,
                    duration_sec=duration_sec,
                    device=device,
                    seed=args.seed + len(results),
                    top_k=args.top_k,
                    keep=args.keep,
                )
            except ValueError:
                skipped += 1
                continue
            results.append(tick)
            print(
                f"{run_dir.name} t={tick['time']:>4.0f}  spearman={tick['spearman']:+.3f}  "
                f"recall={tick['recall']:.2f}  pos_err={tick['pos_error_m']:.2f}m  hp_err={tick['hp_error']:.3f}"
            )

    if not results:
        raise ValueError("측정 가능한 결정 시점이 없다")
    rhos = np.asarray([r["spearman"] for r in results])
    recalls = np.asarray([r["recall"] for r in results])
    print()
    print(f"ticks={len(rhos)} (skipped={skipped})")
    print(f"spearman: mean={rhos.mean():+.3f} median={float(np.median(rhos)):+.3f} min={rhos.min():+.3f}")
    print(f"top{args.top_k} recall@keep{args.keep}: mean={recalls.mean():.2f}")
    print(
        f"decoder 복원 오차: pos={np.mean([r['pos_error_m'] for r in results]):.2f}m "
        f"hp={np.mean([r['hp_error'] for r in results]):.3f} (ratio)"
    )
    print()
    print("해석: spearman>0.7 이면 prefilter 신뢰 가능, 0.4~0.7 이면 keep을 늘려 보수적으로,")
    print("      <0.4 이면 pred_frames 재학습/soft-kill 항 조정 전에 prefilter를 쓰면 안 된다.")


if __name__ == "__main__":
    main()
