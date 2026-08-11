"""Measure CEM backend timing and JEPA-vs-DEVS elite agreement.

This diagnostic keeps the candidate set fixed for rank metrics:
the same CEM candidates are rolled out once with DEVS and once with JEPA,
then scored by the same value head or heuristic score. Full DEVS CEM is
usually too slow at 300x30, so the script reports a measured one-iteration
cost and a linear full-selection estimate. JEPA full CEM can be run exactly.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hackerthon.combat_config import MAX_FIRE_RANGE  # noqa: E402
from hackerthon.terrain import has_los, path_pad_for_unit_radius, set_path_pad  # noqa: E402
from hackerthon.worldmodel.actions import ActionType, build_action_batch_from_v2_run  # noqa: E402
from hackerthon.worldmodel.cem_planner import (  # noqa: E402
    CEMConfig,
    CEMDistribution,
    ObservedActionWindow,
    build_initial_distribution,
    retarget_engage_from_prediction,
    rollout_with_world_model,
    sample_future_action_plans,
    score_future_features_torch,
    update_distribution,
)
from hackerthon.worldmodel.devs_rollout import (  # noqa: E402
    rollout_plans_with_devs,
    snapshot_from_slot_rows,
)
from hackerthon.worldmodel.object_slot_attention import (  # noqa: E402
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.slots import (  # noqa: E402
    ObjectType,
    TeamId,
    build_slot_batch_from_v2_run,
    load_v2_config,
    mission_type_from_config,
    objective_from_config,
)
from hackerthon.worldmodel.value_head import ValueHead, load_value_head  # noqa: E402
from hackerthon.worldmodel.value_scoring import make_value_score_fn  # noqa: E402


@dataclass(frozen=True)
class DecisionContext:
    run_dir: Path
    time_sec: float
    current_batch: object
    history_batches: tuple
    observed_actions: ObservedActionWindow
    snapshot: object
    obstacles: list
    mission_type: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter()
    value = fn()
    _sync(device)
    return value, time.perf_counter() - start


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DEVSObjectCentricWorldModel, ObjectSlotModelConfig]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = dict(payload["model_config"])
    if "maskable_type_ids" in config_dict:
        config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"missing model keys initialized randomly: {sorted(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"unexpected model keys ignored: {sorted(incompatible.unexpected_keys)}")
    model.eval()
    return model, config


def _snapshot_rows_at(run_dir: Path, time_sec: float) -> list[dict[str, object]]:
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
        raise ValueError(f"{run_dir} has no soldier rows at time={time_sec}")
    return rows


def _command_times(run_dir: Path) -> set[float]:
    times: set[float] = set()
    with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.add(float(row["time"]))
    return times


def _state_times(run_dir: Path) -> set[float]:
    times: set[float] = set()
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.add(float(row["time"]))
    return times


def _times_with_full_context(run_dir: Path, history_frames: int) -> list[float]:
    states = _state_times(run_dir)
    commands = _command_times(run_dir)
    out: list[float] = []
    for time_sec in sorted(states):
        history = [time_sec - offset for offset in reversed(range(history_frames))]
        if all(t in states for t in history) and all(t in commands for t in history[:-1]):
            out.append(float(time_sec))
    return out


def _choose_run_dir(output_root: Path, preferred_index: int) -> Path:
    if (output_root / "soldier_log.csv").exists():
        return output_root
    candidates = sorted(p for p in output_root.iterdir() if (p / "soldier_log.csv").exists())
    if not candidates:
        raise ValueError(f"{output_root} has no episode directories")
    index = min(max(0, preferred_index), len(candidates) - 1)
    return candidates[index]


def _build_context(
    *,
    run_dir: Path,
    time_sec: float | None,
    model_config: ObjectSlotModelConfig,
    device: torch.device,
) -> DecisionContext:
    config = load_v2_config(run_dir)
    real_map = config.get("real_map", {})
    if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
        set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

    valid_times = _times_with_full_context(run_dir, model_config.history_frames)
    if not valid_times:
        raise ValueError(f"{run_dir} has no decision time with full model context")
    if time_sec is None:
        picked_time = valid_times[min(3, len(valid_times) - 1)]
    else:
        picked_time = float(time_sec)
        if picked_time not in valid_times:
            nearest = min(valid_times, key=lambda t: abs(t - picked_time))
            print(f"requested time {picked_time} not valid; using nearest full-context time {nearest}")
            picked_time = nearest

    history_times = [picked_time - offset for offset in reversed(range(model_config.history_frames))]
    history_batches = tuple(build_slot_batch_from_v2_run(run_dir, t) for t in history_times)
    observed_batches = tuple(
        build_action_batch_from_v2_run(run_dir, command_time_sec=t, state_time_sec=t)
        for t in history_times[:-1]
    )
    observed = ObservedActionWindow(
        action_features=torch.stack(
            [torch.as_tensor(a.features, dtype=torch.float32, device=device) for a in observed_batches],
            dim=0,
        ),
        action_unit_ids=torch.stack(
            [torch.as_tensor(a.unit_ids, dtype=torch.long, device=device) for a in observed_batches],
            dim=0,
        ),
        issued_mask=torch.stack(
            [torch.as_tensor(a.issued_mask, dtype=torch.bool, device=device) for a in observed_batches],
            dim=0,
        ),
    )

    rows = _snapshot_rows_at(run_dir, picked_time)
    mission_type = mission_type_from_config(config)
    snapshot = snapshot_from_slot_rows(
        unit_rows=rows,
        obstacles=config["obstacles"],
        base_time_sec=picked_time,
        episode_duration_sec=float(config["duration"]),
        objective=objective_from_config(config),
        mission_type=mission_type,
    )
    return DecisionContext(
        run_dir=run_dir,
        time_sec=picked_time,
        current_batch=build_slot_batch_from_v2_run(run_dir, picked_time),
        history_batches=history_batches,
        observed_actions=observed,
        snapshot=snapshot,
        obstacles=config["obstacles"],
        mission_type=mission_type,
    )


def _normalize(values: torch.Tensor) -> torch.Tensor:
    total = values.sum()
    if float(total.detach().cpu().item()) <= 0.0:
        raise ValueError("probability vector has non-positive mass")
    return values / total


def _apply_current_engage_hard_mask(
    *,
    distribution: CEMDistribution,
    context: DecisionContext,
    device: torch.device,
) -> CEMDistribution:
    type_ids = torch.as_tensor(context.current_batch.type_ids, device=device).long()
    team_ids = torch.as_tensor(context.current_batch.team_ids, device=device).long()
    entity_ids = torch.as_tensor(context.current_batch.entity_ids, device=device).long()
    blue_indices = torch.nonzero(
        (type_ids == int(ObjectType.UNIT)) & (team_ids == int(TeamId.BLUE)),
        as_tuple=False,
    ).flatten()
    red_indices = torch.nonzero(
        (type_ids == int(ObjectType.UNIT)) & (team_ids == int(TeamId.RED)),
        as_tuple=False,
    ).flatten()
    rows = _snapshot_rows_at(context.run_dir, context.time_sec)
    row_by_id = {int(row["id"]): row for row in rows}

    action_probs = distribution.action_probs.clone()
    target_probs = distribution.target_probs.clone()
    for blue_pos, blue_slot in enumerate(blue_indices.tolist()):
        blue_id = int(entity_ids[blue_slot].detach().cpu().item())
        allowed = torch.zeros((len(red_indices),), dtype=torch.float32, device=device)
        shooter = row_by_id.get(blue_id)
        for red_pos, red_slot in enumerate(red_indices.tolist()):
            red_id = int(entity_ids[red_slot].detach().cpu().item())
            target = row_by_id.get(red_id)
            if shooter is None or target is None:
                continue
            if float(shooter["hp"]) <= 0.0 or int(shooter["ammo"]) <= 0:
                continue
            if float(target["hp"]) <= 0.0:
                continue
            distance = math.hypot(float(shooter["x"]) - float(target["x"]), float(shooter["y"]) - float(target["y"]))
            if distance > MAX_FIRE_RANGE:
                continue
            if not has_los(
                (float(shooter["x"]), float(shooter["y"])),
                (float(target["x"]), float(target["y"])),
                context.obstacles,
            ):
                continue
            allowed[red_pos] = 1.0
        if bool(torch.any(allowed > 0.0)):
            target_probs[0, blue_pos] = _normalize(allowed)
        else:
            action_probs[0, blue_pos, int(ActionType.ENGAGE)] = 0.0
            target_probs[0, blue_pos] = torch.full_like(
                target_probs[0, blue_pos],
                1.0 / float(target_probs.shape[-1]),
            )
        action_probs[0, blue_pos] = _normalize(action_probs[0, blue_pos])

    return CEMDistribution(
        action_probs=action_probs,
        move_mean=distribution.move_mean,
        move_std=distribution.move_std,
        turn_mean=distribution.turn_mean,
        turn_std=distribution.turn_std,
        target_probs=target_probs,
    )


def _jepa_rollout_fn(
    *,
    model: DEVSObjectCentricWorldModel,
    context: DecisionContext,
    device: torch.device,
    chunk_size: int,
):
    def rollout(plans):
        chunks = []
        for start in range(0, plans.action_features.shape[0], chunk_size):
            stop = min(start + chunk_size, plans.action_features.shape[0])
            index = torch.arange(start, stop, device=plans.action_features.device)
            chunks.append(
                rollout_with_world_model(
                    model=model,
                    history_batches=context.history_batches,
                    observed_actions=context.observed_actions,
                    future_plans=plans.take_candidates(index),
                    device=device,
                    chunk_size=chunk_size,
                )
            )
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    return rollout


def _devs_rollout_fn(*, context: DecisionContext, device: torch.device, seed: int):
    def rollout(plans):
        return rollout_plans_with_devs(
            plans=plans,
            snapshot=context.snapshot,
            seed=seed,
            device=device,
        )

    return rollout


def _score_fn(
    *,
    value_head: ValueHead | None,
    context: DecisionContext,
    device: torch.device,
):
    if value_head is None:
        def score(features):
            return score_future_features_torch(
                current_batch=context.current_batch,
                future_features=features,
            )

        return score
    return make_value_score_fn(
        value_head=value_head,
        current_batch=context.current_batch,
        mission_type=context.mission_type,
        device=device,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size < 2 or b.size != a.size:
        return float("nan"), float("nan")
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    denom = math.sqrt(float(np.dot(aa, aa)) * float(np.dot(bb, bb)))
    pearson = float(np.dot(aa, bb) / denom) if denom > 0.0 else 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= float(ra.mean())
    rb -= float(rb.mean())
    rdenom = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
    spearman = float(np.dot(ra, rb) / rdenom) if rdenom > 0.0 else 0.0
    return pearson, spearman


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:k]


def _rank_position(scores: np.ndarray, candidate: int) -> int:
    order = np.argsort(scores)[::-1]
    matches = np.nonzero(order == int(candidate))[0]
    if matches.size != 1:
        raise ValueError("candidate not found in score order")
    return int(matches[0]) + 1


def _measure_one_population(
    *,
    context: DecisionContext,
    model: DEVSObjectCentricWorldModel,
    value_head: ValueHead | None,
    config: CEMConfig,
    device: torch.device,
    chunk_size: int,
    seed: int,
    prefilter_ks: Sequence[int],
) -> dict[str, object]:
    distribution = _apply_current_engage_hard_mask(
        distribution=build_initial_distribution(context.current_batch, config, device=device),
        context=context,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    plans, sample_seconds = _elapsed(
        device,
        lambda: sample_future_action_plans(
            distribution=distribution,
            current_batch=context.current_batch,
            config=config,
            generator=generator,
            device=device,
        ),
    )
    score = _score_fn(value_head=value_head, context=context, device=device)
    devs_rollout = _devs_rollout_fn(context=context, device=device, seed=seed)
    jepa_rollout = _jepa_rollout_fn(model=model, context=context, device=device, chunk_size=chunk_size)

    devs_features, devs_seconds = _elapsed(device, lambda: devs_rollout(plans))
    devs_scores, devs_score_seconds = _elapsed(device, lambda: score(devs_features))
    jepa_features, jepa_seconds = _elapsed(device, lambda: jepa_rollout(plans))
    jepa_scores, jepa_score_seconds = _elapsed(device, lambda: score(jepa_features))

    devs_np = devs_scores.detach().cpu().numpy()
    jepa_np = jepa_scores.detach().cpu().numpy()
    pearson, spearman = _corr(devs_np, jepa_np)
    elite = config.num_elites
    devs_top = _top_indices(devs_np, elite)
    jepa_top = _top_indices(jepa_np, elite)
    overlap = len(set(devs_top.tolist()) & set(jepa_top.tolist()))
    devs_best = int(devs_top[0])
    jepa_best = int(jepa_top[0])
    prefilter_metrics: list[dict[str, float]] = []
    for raw_k in prefilter_ks:
        keep = int(raw_k)
        if keep < elite or keep > config.num_candidates:
            continue
        jepa_keep = _top_indices(jepa_np, keep)
        subset_order = np.argsort(devs_np[jepa_keep])[::-1][:elite]
        hybrid_top = jepa_keep[subset_order]
        hybrid_overlap = len(set(devs_top.tolist()) & set(hybrid_top.tolist()))
        hybrid_best = int(hybrid_top[0])
        prefilter_metrics.append(
            {
                "keep": float(keep),
                "overlap": float(hybrid_overlap),
                "recall": float(hybrid_overlap) / float(elite),
                "best_devs_rank": float(_rank_position(devs_np, hybrid_best)),
                "elite_devs_mean_score": float(devs_np[hybrid_top].mean()),
            }
        )

    return {
        "sample_seconds": sample_seconds,
        "devs_rollout_seconds": devs_seconds,
        "devs_score_seconds": devs_score_seconds,
        "jepa_rollout_seconds": jepa_seconds,
        "jepa_score_seconds": jepa_score_seconds,
        "pearson": pearson,
        "spearman": spearman,
        "elite_overlap": float(overlap),
        "elite_recall": float(overlap) / float(elite),
        "devs_best": float(devs_best),
        "jepa_best": float(jepa_best),
        "jepa_best_devs_rank": float(_rank_position(devs_np, jepa_best)),
        "devs_best_jepa_rank": float(_rank_position(jepa_np, devs_best)),
        "devs_top_mean_score": float(devs_np[devs_top].mean()),
        "jepa_top_devs_mean_score": float(devs_np[jepa_top].mean()),
        "devs_best_score": float(devs_np[devs_best]),
        "jepa_best_devs_score": float(devs_np[jepa_best]),
        "prefilter_metrics": prefilter_metrics,
    }


def _time_hybrid_full_cem(
    *,
    context: DecisionContext,
    config: CEMConfig,
    jepa_rollout_fn,
    devs_rollout_fn,
    score_fn,
    device: torch.device,
    prefilter_k: int,
) -> tuple[float, float, float]:
    if prefilter_k < config.num_elites or prefilter_k > config.num_candidates:
        raise ValueError("--hybrid-prefilter-k must be between elites and candidates")
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    distribution = _apply_current_engage_hard_mask(
        distribution=build_initial_distribution(context.current_batch, config, device=device),
        context=context,
        device=device,
    )

    def run():
        best_score = -float("inf")
        best_plan = sample_future_action_plans(
            distribution=distribution,
            current_batch=context.current_batch,
            config=config,
            generator=generator,
            device=device,
        ).take_candidates(torch.tensor([0], dtype=torch.long, device=device))
        current_distribution = distribution
        last_population_mean = float("nan")
        for iteration in range(config.num_iterations):
            iter_start = time.perf_counter()
            plans = sample_future_action_plans(
                distribution=current_distribution,
                current_batch=context.current_batch,
                config=config,
                generator=generator,
                device=device,
            )
            jepa_features = jepa_rollout_fn(plans)
            jepa_scores = score_fn(jepa_features)
            _, prefilter_indices = torch.topk(
                jepa_scores,
                k=prefilter_k,
                largest=True,
                sorted=False,
            )
            filtered_plans = plans.take_candidates(prefilter_indices)
            devs_features = devs_rollout_fn(filtered_plans)
            devs_scores = score_fn(devs_features)
            elite_scores, subset_elite_indices = torch.topk(
                devs_scores,
                k=config.num_elites,
                largest=True,
                sorted=True,
            )
            elite_indices = prefilter_indices.index_select(0, subset_elite_indices)
            iteration_best = float(elite_scores[0].detach().cpu().item())
            if iteration_best > best_score:
                best_score = iteration_best
                best_plan = plans.take_candidates(elite_indices[:1])
            last_population_mean = float(devs_scores.mean().detach().cpu().item())
            current_distribution = update_distribution(
                distribution=current_distribution,
                plans=plans,
                elite_indices=elite_indices,
                config=config,
            )
            _sync(device)
            print(
                f"hybrid_k{prefilter_k} iter {iteration + 1:02d}/{config.num_iterations} "
                f"elapsed={_fmt_seconds(time.perf_counter() - iter_start)} "
                f"best={iteration_best:.4f} prefilter_pop={last_population_mean:.4f}",
                flush=True,
            )
        if config.retarget_engage:
            retarget_start = time.perf_counter()
            best_plan = retarget_engage_from_prediction(
                plan=best_plan,
                future_features=devs_rollout_fn(best_plan),
                current_batch=context.current_batch,
                generator=generator,
                device=device,
            )
            del best_plan
            _sync(device)
            print(
                f"hybrid_k{prefilter_k} retarget elapsed={_fmt_seconds(time.perf_counter() - retarget_start)}",
                flush=True,
            )
        return best_score, last_population_mean

    (best_score, population), seconds = _elapsed(device, run)
    return seconds, float(best_score), float(population)


def _time_full_cem(
    *,
    label: str,
    context: DecisionContext,
    config: CEMConfig,
    rollout_fn,
    score_fn,
    device: torch.device,
) -> tuple[float, float, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    distribution = _apply_current_engage_hard_mask(
        distribution=build_initial_distribution(context.current_batch, config, device=device),
        context=context,
        device=device,
    )

    def run():
        best_score = -float("inf")
        best_plan = sample_future_action_plans(
            distribution=distribution,
            current_batch=context.current_batch,
            config=config,
            generator=generator,
            device=device,
        ).take_candidates(torch.tensor([0], dtype=torch.long, device=device))
        current_distribution = distribution
        last_population_mean = float("nan")
        for iteration in range(config.num_iterations):
            iter_start = time.perf_counter()
            plans = sample_future_action_plans(
                distribution=current_distribution,
                current_batch=context.current_batch,
                config=config,
                generator=generator,
                device=device,
            )
            features = rollout_fn(plans)
            scores = score_fn(features)
            elite_scores, elite_indices = torch.topk(scores, k=config.num_elites, largest=True, sorted=True)
            iteration_best = float(elite_scores[0].detach().cpu().item())
            if iteration_best > best_score:
                best_score = iteration_best
                best_plan = plans.take_candidates(elite_indices[:1])
            last_population_mean = float(scores.mean().detach().cpu().item())
            current_distribution = update_distribution(
                distribution=current_distribution,
                plans=plans,
                elite_indices=elite_indices,
                config=config,
            )
            _sync(device)
            print(
                f"{label} iter {iteration + 1:02d}/{config.num_iterations} "
                f"elapsed={_fmt_seconds(time.perf_counter() - iter_start)} "
                f"best={iteration_best:.4f} pop={last_population_mean:.4f}",
                flush=True,
            )
        if config.retarget_engage:
            retarget_start = time.perf_counter()
            best_plan = retarget_engage_from_prediction(
                plan=best_plan,
                future_features=rollout_fn(best_plan),
                current_batch=context.current_batch,
                generator=generator,
                device=device,
            )
            del best_plan
            _sync(device)
            print(
                f"{label} retarget elapsed={_fmt_seconds(time.perf_counter() - retarget_start)}",
                flush=True,
            )
        return best_score, last_population_mean

    (best_score, population), seconds = _elapsed(device, run)
    return seconds, float(best_score), float(population)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.2f}m"
    return f"{minutes / 60.0:.2f}h"


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEM backend timing and elite-rank diagnostic")
    parser.add_argument("--output-root", type=Path, default=Path("output/respos"))
    parser.add_argument("--run-index", type=int, default=2)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-head", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--skip-jepa-full", action="store_true")
    parser.add_argument("--skip-one-population", action="store_true")
    parser.add_argument("--run-devs-full", action="store_true")
    parser.add_argument(
        "--prefilter-ks",
        type=str,
        default="30,60,90,150",
        help="Comma-separated JEPA top-K values to re-rank by DEVS in one-population diagnostics.",
    )
    parser.add_argument("--run-hybrid-full", action="store_true")
    parser.add_argument("--hybrid-prefilter-k", type=int, default=90)
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_prefilter_ks(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.elites > args.candidates:
        raise ValueError("--elites cannot exceed --candidates")
    device = torch.device(args.device)
    model, model_config = _load_model(args.checkpoint, device)
    value_head = load_value_head(args.value_head, device) if args.value_head is not None else None
    run_dir = _choose_run_dir(args.output_root, args.run_index)
    context = _build_context(
        run_dir=run_dir,
        time_sec=args.time,
        model_config=model_config,
        device=device,
    )
    config = CEMConfig(
        num_candidates=args.candidates,
        num_elites=args.elites,
        num_iterations=args.iterations,
        future_horizon=model_config.pred_frames,
        seed=args.seed,
        min_action_probability=0.0,
    )

    print(
        "context "
        f"run={context.run_dir} time={context.time_sec:.1f} "
        f"horizon={config.future_horizon} candidates={config.num_candidates} "
        f"elites={config.num_elites} iterations={config.num_iterations}"
    )
    print(f"world_model={args.checkpoint}")
    print(f"value_head={args.value_head if args.value_head is not None else 'heuristic'}")

    full_devs_est = full_jepa_est = None
    if not args.skip_one_population:
        one = _measure_one_population(
            context=context,
            model=model,
            value_head=value_head,
            config=config,
            device=device,
            chunk_size=args.chunk_size,
            seed=args.seed,
            prefilter_ks=_parse_prefilter_ks(args.prefilter_ks),
        )

        per_devs = one["devs_rollout_seconds"] + one["devs_score_seconds"]
        per_jepa = one["jepa_rollout_seconds"] + one["jepa_score_seconds"]
        full_devs_est = config.num_iterations * per_devs + per_devs / float(config.num_candidates)
        full_jepa_est = config.num_iterations * per_jepa + per_jepa / float(config.num_candidates)

        print(f"\nOne CEM population (same {config.num_candidates} candidates, score includes value head)")
        print(
            f"  sample={_fmt_seconds(one['sample_seconds'])} "
            f"DEVS rollout+score={_fmt_seconds(per_devs)} "
            f"JEPA rollout+score={_fmt_seconds(per_jepa)} "
            f"speedup={per_devs / max(per_jepa, 1e-9):.1f}x"
        )
        print(
            f"  pearson={one['pearson']:+.3f} spearman={one['spearman']:+.3f} "
            f"top{config.num_elites} overlap={int(one['elite_overlap'])}/{config.num_elites} "
            f"({100.0 * one['elite_recall']:.1f}%)"
        )
        print(
            f"  JEPA best candidate is DEVS rank {int(one['jepa_best_devs_rank'])}/{config.num_candidates}; "
            f"DEVS best candidate is JEPA rank {int(one['devs_best_jepa_rank'])}/{config.num_candidates}"
        )
        print(
            f"  DEVS top{config.num_elites} mean score={one['devs_top_mean_score']:.4f}; "
            f"JEPA top{config.num_elites} mean under DEVS={one['jepa_top_devs_mean_score']:.4f}"
        )
        prefilter_metrics = one["prefilter_metrics"]
        if prefilter_metrics:
            print("  JEPA prefilter then DEVS re-rank:")
            for metric in prefilter_metrics:
                print(
                    f"    top{int(metric['keep'])}: "
                    f"DEVS top{config.num_elites} overlap={int(metric['overlap'])}/{config.num_elites} "
                    f"({100.0 * metric['recall']:.1f}%) "
                    f"best_devs_rank={int(metric['best_devs_rank'])}/{config.num_candidates} "
                    f"mean_score={metric['elite_devs_mean_score']:.4f}"
                )

        print(f"\nFull {config.num_candidates}x{config.num_iterations} CEM selection estimate")
        print(f"  DEVS backend estimate: {_fmt_seconds(full_devs_est)}")
        print(f"  JEPA backend linear estimate: {_fmt_seconds(full_jepa_est)}")
        print(f"  estimated selection speedup: {full_devs_est / max(full_jepa_est, 1e-9):.1f}x")

    if not args.skip_jepa_full:
        jepa_rollout = _jepa_rollout_fn(
            model=model,
            context=context,
            device=device,
            chunk_size=args.chunk_size,
        )
        score = _score_fn(value_head=value_head, context=context, device=device)
        seconds, best, population = _time_full_cem(
            label="jepa_full",
            context=context,
            config=config,
            rollout_fn=jepa_rollout,
            score_fn=score,
            device=device,
        )
        print(f"\nActual JEPA {config.num_candidates}x{config.num_iterations} CEM")
        print(
            f"  elapsed={_fmt_seconds(seconds)} "
            f"best_score={best:.4f} last_population_mean={population:.4f}"
        )
        if full_devs_est is not None:
            print(f"  DEVS/backend speedup vs actual JEPA: {full_devs_est / max(seconds, 1e-9):.1f}x")

    if args.run_devs_full:
        devs_rollout = _devs_rollout_fn(context=context, device=device, seed=args.seed)
        score = _score_fn(value_head=value_head, context=context, device=device)
        seconds, best, population = _time_full_cem(
            label="devs_full",
            context=context,
            config=config,
            rollout_fn=devs_rollout,
            score_fn=score,
            device=device,
        )
        print(f"\nActual DEVS {config.num_candidates}x{config.num_iterations} CEM")
        print(
            f"  elapsed={_fmt_seconds(seconds)} "
            f"best_score={best:.4f} last_population_mean={population:.4f}"
        )

    if args.run_hybrid_full:
        jepa_rollout = _jepa_rollout_fn(
            model=model,
            context=context,
            device=device,
            chunk_size=args.chunk_size,
        )
        devs_rollout = _devs_rollout_fn(context=context, device=device, seed=args.seed)
        score = _score_fn(value_head=value_head, context=context, device=device)
        seconds, best, population = _time_hybrid_full_cem(
            context=context,
            config=config,
            jepa_rollout_fn=jepa_rollout,
            devs_rollout_fn=devs_rollout,
            score_fn=score,
            device=device,
            prefilter_k=args.hybrid_prefilter_k,
        )
        print(f"\nActual hybrid JEPA-top{args.hybrid_prefilter_k} -> DEVS {config.num_candidates}x{config.num_iterations} CEM")
        print(
            f"  elapsed={_fmt_seconds(seconds)} "
            f"best_score={best:.4f} last_prefilter_population_mean={population:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
