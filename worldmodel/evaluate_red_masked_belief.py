"""RED 1-slot partial-observation belief 평가.

현재 checkpoint가 "보이지 않는 RED slot"을 belief처럼 복원할 수 있는지
진단한다. RED slot은 삭제하지 않고 identity/type/team metadata를 유지한 채,
history t>=1의 해당 RED state token만 mask token으로 바꿔 예측한다.

기본 target 조건은 "history anchor(t0)에서는 BLUE가 봤고, 현재 decision
tick에서는 BLUE 관측 밖인 살아있는 RED 1명"이다. 따라서 이 평가는 완전한
fog-of-war가 아니라 "last seen 이후 hidden RED 추적"에 가깝다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.combat_config import PERCEPTION_RANGE
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN, has_los
from hackerthon.worldmodel.actions import available_command_times, load_unit_rows_at_time
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    ObservedActionWindow,
    build_initial_distribution,
    combine_observed_and_future_actions,
    sample_future_action_plans,
    score_future_features_torch,
    stack_slot_history,
)
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.measure_rank_correlation import spearman, top_k_recall
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig
from hackerthon.worldmodel.slots import MAX_FEATURE_DIM, SlotBatch, available_times
from hackerthon.worldmodel.train_object_centric_jepa import TrainingWindow, collate_training_batch, load_training_window

UNIT_HP_INDEX = 1
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4
UNIT_HEADING_COS_INDEX = 5
UNIT_HEADING_SIN_INDEX = 6
UNIT_ALIVE_INDEX = 7
X_SCALE = (WORLD_X_MAX - WORLD_X_MIN) * 0.5
Y_SCALE = (WORLD_Y_MAX - WORLD_Y_MIN) * 0.5


@dataclass(frozen=True)
class BeliefMetrics:
    """한 masked RED target 평가 결과."""

    run_dir: Path
    start_time: float
    current_time: float
    target_id: int
    current_pos_error_m: float
    current_pos_error_last_seen_m: float
    current_hp_mae: float
    current_hp_mae_last_seen: float
    future_pos_error_m: float
    future_pos_error_last_seen_m: float
    future_hp_mae: float
    future_hp_mae_last_seen: float
    future_alive_accuracy: float
    rank_spearman: float | None = None
    topk_recall: float | None = None


def _normalize_angle_deg(angle: float) -> float:
    """각도 차이를 [-180, 180] 범위로 정규화한다."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _episode_dirs(path: Path) -> tuple[Path, ...]:
    """단일 episode dir 또는 episode root를 받아 episode dir 목록을 반환한다."""
    if (path / "soldier_log.csv").exists() and (path / "config.json").exists():
        return (path,)
    return tuple(sorted(child for child in path.iterdir() if (child / "soldier_log.csv").exists()))


def _load_config(run_dir: Path) -> dict[str, object]:
    """episode config.json을 읽는다."""
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _is_urban(run_dir: Path) -> bool:
    """obstacle이 하나 이상 있으면 urban/cover terrain으로 간주한다."""
    return bool(_load_config(run_dir).get("obstacles", []))


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[DEVSObjectCentricWorldModel, ObjectSlotModelConfig]:
    """checkpoint에서 월드모델을 복원한다."""
    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(payload["model_config"])
    if "maskable_type_ids" in config_dict:
        config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def _unit_alive(row: Mapping[str, object]) -> bool:
    """unit row가 생존 상태인지 반환한다."""
    return float(row["hp"]) > 0.0 and str(row.get("mode", "")).upper() != "DESTROYED"


def _visible_red_ids_from_blue(
    rows: Sequence[Mapping[str, object]],
    *,
    obstacles: Sequence[Sequence[float]],
    fov_deg: float = 120.0,
) -> set[int]:
    """BLUE 중 하나라도 FOV/거리/LOS로 볼 수 있는 RED id 집합."""
    blue_rows = [row for row in rows if int(row["id"]) < 200 and _unit_alive(row)]
    red_rows = [row for row in rows if int(row["id"]) >= 200 and _unit_alive(row)]
    visible: set[int] = set()
    for blue in blue_rows:
        sx = float(blue["x"])
        sy = float(blue["y"])
        heading = float(blue["heading"])
        for red in red_rows:
            ex = float(red["x"])
            ey = float(red["y"])
            dx = ex - sx
            dy = ey - sy
            distance = math.hypot(dx, dy)
            if distance > PERCEPTION_RANGE:
                continue
            relative_theta = _normalize_angle_deg(math.degrees(math.atan2(dy, dx)) - heading)
            if abs(relative_theta) > fov_deg / 2.0:
                continue
            if not has_los((sx, sy), (ex, ey), list(obstacles)):
                continue
            visible.add(int(red["id"]))
    return visible


def _alive_red_ids(rows: Sequence[Mapping[str, object]]) -> set[int]:
    """살아있는 RED id 집합."""
    return {int(row["id"]) for row in rows if int(row["id"]) >= 200 and _unit_alive(row)}


def _build_windows(
    run_dir: Path,
    *,
    history_frames: int,
    pred_frames: int,
    max_windows: int,
) -> tuple[TrainingWindow, ...]:
    """완전한 state/action window를 run_dir에서 찾는다."""
    total_frames = history_frames + pred_frames
    state_times = available_times(run_dir)
    command_times = set(available_command_times(run_dir))
    state_time_set = set(state_times)
    windows: list[TrainingWindow] = []
    for start_time in state_times:
        state_window = tuple(float(start_time + offset) for offset in range(total_frames))
        action_window = state_window[:-1]
        if not all(time_value in state_time_set for time_value in state_window):
            continue
        if not all(time_value in command_times for time_value in action_window):
            continue
        windows.append(TrainingWindow(run_dir=run_dir, state_times=state_window, action_times=action_window))
        if len(windows) >= max_windows:
            break
    return tuple(windows)


def _choose_hidden_red_target(
    window: TrainingWindow,
    *,
    obstacles: Sequence[Sequence[float]],
    history_frames: int,
    rng: np.random.Generator,
) -> int | None:
    """anchor에서 보였고 현재는 안 보이는 RED 1명을 선택한다."""
    anchor_rows = load_unit_rows_at_time(window.run_dir, window.state_times[0])
    current_rows = load_unit_rows_at_time(window.run_dir, window.state_times[history_frames - 1])
    anchor_visible = _visible_red_ids_from_blue(anchor_rows, obstacles=obstacles)
    current_visible = _visible_red_ids_from_blue(current_rows, obstacles=obstacles)
    current_alive = _alive_red_ids(current_rows)
    candidates = sorted((anchor_visible & current_alive) - current_visible)
    if not candidates:
        return None
    return int(rng.choice(candidates))


def _slot_index_for_entity(window: TrainingWindow, *, target_id: int) -> int:
    """window 첫 state의 entity_ids에서 target slot index를 찾는다."""
    loaded = load_training_window(window)
    entity_ids = loaded.states[0].entity_ids
    matches = np.flatnonzero(entity_ids == int(target_id))
    if matches.shape != (1,):
        raise ValueError(f"target_id={target_id} slot을 정확히 하나 찾지 못했다: {matches.tolist()}")
    return int(matches[0])


def _predict_with_explicit_mask(
    *,
    model: DEVSObjectCentricWorldModel,
    batch,
    masked_slot_index: int,
) -> torch.Tensor:
    """batch의 특정 slot을 history t>=1에서 숨긴 후 전체 frame feature를 예측한다."""
    history_frames = model.config.history_frames
    total_frames = model.config.history_frames + model.config.pred_frames
    history_tokens = model.encode_state_sequence(
        features=batch.features[:, :history_frames],
        feature_mask=batch.feature_mask[:, :history_frames],
        type_ids=batch.type_ids[:, :history_frames],
        team_ids=batch.team_ids[:, :history_frames],
        alive_mask=batch.alive_mask[:, :history_frames],
    )
    action_tokens = model.build_transition_action_tokens(
        source_tokens=history_tokens[:, 0],
        type_ids=batch.type_ids,
        team_ids=batch.team_ids,
        entity_ids=batch.entity_ids,
        action_features=batch.action_features,
        action_unit_ids=batch.action_unit_ids,
        issued_mask=batch.issued_mask,
    )
    pred_tokens, _, _ = model.masked_predictor.forward_with_masked_indices(
        history_tokens,
        type_ids=batch.type_ids[:, 0],
        team_ids=batch.team_ids[:, 0],
        action_tokens=action_tokens,
        masked_indices=torch.tensor([masked_slot_index], dtype=torch.long, device=batch.features.device),
    )
    pred_features = model.state_decoder(
        pred_tokens.reshape(pred_tokens.shape[0] * pred_tokens.shape[1], pred_tokens.shape[2], pred_tokens.shape[3]),
        batch.type_ids[:, 0]
        .unsqueeze(1)
        .expand(batch.type_ids.shape[0], total_frames, batch.type_ids.shape[2])
        .reshape(batch.type_ids.shape[0] * total_frames, batch.type_ids.shape[2]),
    ).reshape(batch.features.shape[0], total_frames, batch.features.shape[2], MAX_FEATURE_DIM)
    return pred_features


def _pos_error_m(pred: torch.Tensor, target: torch.Tensor) -> float:
    """unit x/y feature 오차를 월드 거리 단위로 변환한다."""
    dx = (pred[..., UNIT_X_INDEX] - target[..., UNIT_X_INDEX]) * X_SCALE
    dy = (pred[..., UNIT_Y_INDEX] - target[..., UNIT_Y_INDEX]) * Y_SCALE
    return float(torch.sqrt(dx * dx + dy * dy).mean().detach().cpu().item())


def _hp_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """hp_ratio MAE."""
    return float((pred[..., UNIT_HP_INDEX] - target[..., UNIT_HP_INDEX]).abs().mean().detach().cpu().item())


def _alive_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """alive feature를 0.5 threshold로 이산화한 accuracy."""
    pred_alive = pred[..., UNIT_ALIVE_INDEX] >= 0.5
    target_alive = target[..., UNIT_ALIVE_INDEX] >= 0.5
    return float((pred_alive == target_alive).float().mean().detach().cpu().item())


def _target_metrics(
    *,
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
    masked_slot_index: int,
    history_frames: int,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """masked target slot의 current/future 오차와 last-seen baseline 오차를 계산한다."""
    current_index = history_frames - 1
    pred_current = pred_features[:, current_index, masked_slot_index]
    target_current = target_features[:, current_index, masked_slot_index]
    pred_future = pred_features[:, history_frames:, masked_slot_index]
    target_future = target_features[:, history_frames:, masked_slot_index]

    last_seen = target_features[:, 0:1, masked_slot_index]
    last_seen_current = last_seen[:, 0]
    last_seen_future = last_seen.expand(target_features.shape[0], target_future.shape[1], MAX_FEATURE_DIM)

    return (
        _pos_error_m(pred_current, target_current),
        _pos_error_m(last_seen_current, target_current),
        _hp_mae(pred_current, target_current),
        _hp_mae(last_seen_current, target_current),
        _pos_error_m(pred_future, target_future),
        _pos_error_m(last_seen_future, target_future),
        _hp_mae(pred_future, target_future),
        _hp_mae(last_seen_future, target_future),
        _alive_accuracy(pred_future, target_future),
    )


def _snapshot_rows_at(run_dir: Path, time_sec: float) -> list[dict[str, object]]:
    """soldier_log에서 특정 시점 unit row를 수치형 dict로 읽는다."""
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


def _observed_action_window(loaded, *, history_frames: int, device: torch.device) -> ObservedActionWindow:
    """window 앞쪽 history transition action을 ObservedActionWindow로 변환한다."""
    observed_actions = loaded.actions[: history_frames - 1]
    return ObservedActionWindow(
        action_features=torch.stack(
            [torch.as_tensor(action.features, dtype=torch.float32, device=device) for action in observed_actions],
            dim=0,
        ),
        action_unit_ids=torch.stack(
            [torch.as_tensor(action.unit_ids, dtype=torch.long, device=device) for action in observed_actions],
            dim=0,
        ),
        issued_mask=torch.stack(
            [torch.as_tensor(action.issued_mask, dtype=torch.bool, device=device) for action in observed_actions],
            dim=0,
        ),
    )


def _rollout_masked_world_model(
    *,
    model: DEVSObjectCentricWorldModel,
    history_batches,
    observed_actions: ObservedActionWindow,
    future_plans,
    masked_slot_index: int,
    device: torch.device,
) -> torch.Tensor:
    """CEM 후보들을 masked RED history 조건으로 월드모델 rollout한다."""
    history = stack_slot_history(history_batches, device=device)
    candidates = future_plans.action_features.shape[0]
    action_features, action_unit_ids, issued_mask = combine_observed_and_future_actions(
        observed=observed_actions,
        future=future_plans,
    )
    with torch.no_grad():
        history_features = history["features"].unsqueeze(0).expand(candidates, *history["features"].shape)
        history_feature_mask = history["feature_mask"].unsqueeze(0).expand(candidates, *history["feature_mask"].shape)
        history_type_ids = history["type_ids"].unsqueeze(0).expand(candidates, *history["type_ids"].shape)
        history_entity_ids = history["entity_ids"].unsqueeze(0).expand(candidates, *history["entity_ids"].shape)
        history_team_ids = history["team_ids"].unsqueeze(0).expand(candidates, *history["team_ids"].shape)
        history_alive_mask = history["alive_mask"].unsqueeze(0).expand(candidates, *history["alive_mask"].shape)
        history_tokens = model.encode_state_sequence(
            features=history_features,
            feature_mask=history_feature_mask,
            type_ids=history_type_ids,
            team_ids=history_team_ids,
            alive_mask=history_alive_mask,
        )
        total_frames = model.config.history_frames + model.config.pred_frames
        full_type_ids = history_type_ids[:, :1].expand(candidates, total_frames, history_type_ids.shape[2])
        full_entity_ids = history_entity_ids[:, :1].expand(candidates, total_frames, history_entity_ids.shape[2])
        full_team_ids = history_team_ids[:, :1].expand(candidates, total_frames, history_team_ids.shape[2])
        action_tokens = model.build_transition_action_tokens(
            source_tokens=history_tokens[:, 0],
            type_ids=full_type_ids,
            team_ids=full_team_ids,
            entity_ids=full_entity_ids,
            action_features=action_features.to(device=device),
            action_unit_ids=action_unit_ids.to(device=device),
            issued_mask=issued_mask.to(device=device),
        )
        pred_tokens, _, _ = model.masked_predictor.forward_with_masked_indices(
            history_tokens,
            type_ids=history_type_ids[:, 0],
            team_ids=history_team_ids[:, 0],
            action_tokens=action_tokens,
            masked_indices=torch.tensor([masked_slot_index], dtype=torch.long, device=device),
        )
        future_tokens = pred_tokens[:, model.config.history_frames : total_frames]
        future_type_ids = full_type_ids[:, model.config.history_frames : total_frames]
        future_features = model.state_decoder(
            future_tokens.reshape(candidates * model.config.pred_frames, history_type_ids.shape[2], model.config.embedding_dim),
            future_type_ids.reshape(candidates * model.config.pred_frames, history_type_ids.shape[2]),
        ).reshape(candidates, model.config.pred_frames, history_type_ids.shape[2], MAX_FEATURE_DIM)
    return future_features


def _rank_metrics(
    *,
    model: DEVSObjectCentricWorldModel,
    loaded,
    current_index: int,
    belief_current_batch: SlotBatch,
    masked_slot_index: int,
    obstacles: Sequence[Sequence[float]],
    duration_sec: float,
    device: torch.device,
    candidates: int,
    seed: int,
    top_k: int,
    keep: int,
) -> tuple[float, float]:
    """같은 후보 집합에서 masked-WM score와 DEVS score 순위 일치도를 계산한다."""
    true_current_batch = loaded.states[current_index]
    observed = _observed_action_window(loaded, history_frames=model.config.history_frames, device=device)
    cem_config = CEMConfig(
        num_candidates=candidates,
        num_elites=max(2, candidates // 8),
        num_iterations=1,
        future_horizon=model.config.pred_frames,
        seed=seed,
    )
    distribution = build_initial_distribution(belief_current_batch, cem_config, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    plans = sample_future_action_plans(
        distribution=distribution,
        current_batch=belief_current_batch,
        config=cem_config,
        generator=generator,
        device=device,
    )
    snapshot = snapshot_from_slot_rows(
        unit_rows=_snapshot_rows_at(loaded.spec.run_dir, true_current_batch.time_sec),
        obstacles=obstacles,
        base_time_sec=true_current_batch.time_sec,
        episode_duration_sec=duration_sec,
    )
    devs_features = rollout_plans_with_devs(plans=plans, snapshot=snapshot, seed=seed, device=device)
    masked_features = _rollout_masked_world_model(
        model=model,
        history_batches=loaded.states[: model.config.history_frames],
        observed_actions=observed,
        future_plans=plans,
        masked_slot_index=masked_slot_index,
        device=device,
    )
    devs_scores = score_future_features_torch(current_batch=true_current_batch, future_features=devs_features)
    masked_scores = score_future_features_torch(current_batch=belief_current_batch, future_features=masked_features)
    devs_np = devs_scores.detach().cpu().numpy()
    masked_np = masked_scores.detach().cpu().numpy()
    return (
        spearman(devs_np, masked_np),
        top_k_recall(devs_np, masked_np, k=top_k, keep=keep),
    )


def _belief_current_batch(
    *,
    current_batch: SlotBatch,
    pred_features: torch.Tensor,
    current_index: int,
    masked_slot_index: int,
) -> SlotBatch:
    """masked 모델이 복원한 current RED feature로 current SlotBatch를 대체한다."""
    features = current_batch.features.copy()
    alive_mask = current_batch.alive_mask.copy()
    predicted_slot = pred_features[0, current_index, masked_slot_index].detach().cpu().numpy()
    features[masked_slot_index] = predicted_slot
    alive_mask[masked_slot_index] = bool(predicted_slot[UNIT_ALIVE_INDEX] >= 0.5)
    return replace(current_batch, features=features, alive_mask=alive_mask)


def _evaluate_window(
    *,
    model: DEVSObjectCentricWorldModel,
    window: TrainingWindow,
    target_id: int,
    obstacles: Sequence[Sequence[float]],
    duration_sec: float,
    device: torch.device,
    rank_candidates: int,
    seed: int,
    top_k: int,
    keep: int,
) -> BeliefMetrics:
    """단일 window/target에 대해 reconstruction과 optional rank metric을 계산한다."""
    loaded = load_training_window(window)
    masked_slot_index = _slot_index_for_entity(window, target_id=target_id)
    batch = collate_training_batch((loaded,), device=device)
    with torch.no_grad():
        pred_features = _predict_with_explicit_mask(
            model=model,
            batch=batch,
            masked_slot_index=masked_slot_index,
        )
    metric_values = _target_metrics(
        pred_features=pred_features,
        target_features=batch.features,
        masked_slot_index=masked_slot_index,
        history_frames=model.config.history_frames,
    )
    rank_spearman = None
    rank_recall = None
    if rank_candidates > 0:
        belief_current = _belief_current_batch(
            current_batch=loaded.states[model.config.history_frames - 1],
            pred_features=pred_features,
            current_index=model.config.history_frames - 1,
            masked_slot_index=masked_slot_index,
        )
        rank_spearman, rank_recall = _rank_metrics(
            model=model,
            loaded=loaded,
            current_index=model.config.history_frames - 1,
            belief_current_batch=belief_current,
            masked_slot_index=masked_slot_index,
            obstacles=obstacles,
            duration_sec=duration_sec,
            device=device,
            candidates=rank_candidates,
            seed=seed,
            top_k=top_k,
            keep=keep,
        )
    return BeliefMetrics(
        run_dir=window.run_dir,
        start_time=window.state_times[0],
        current_time=window.state_times[model.config.history_frames - 1],
        target_id=target_id,
        current_pos_error_m=metric_values[0],
        current_pos_error_last_seen_m=metric_values[1],
        current_hp_mae=metric_values[2],
        current_hp_mae_last_seen=metric_values[3],
        future_pos_error_m=metric_values[4],
        future_pos_error_last_seen_m=metric_values[5],
        future_hp_mae=metric_values[6],
        future_hp_mae_last_seen=metric_values[7],
        future_alive_accuracy=metric_values[8],
        rank_spearman=rank_spearman,
        topk_recall=rank_recall,
    )


def _write_csv(path: Path, rows: Sequence[BeliefMetrics]) -> None:
    """BeliefMetrics를 CSV로 저장한다."""
    if not rows:
        raise ValueError("저장할 row가 없다")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {name: getattr(row, name) for name in fieldnames}
            payload["run_dir"] = str(payload["run_dir"])
            writer.writerow(payload)


def _mean(values: Iterable[float | None]) -> float | None:
    """None을 제외한 평균."""
    selected = [float(value) for value in values if value is not None]
    if not selected:
        return None
    return float(np.mean(selected))


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="RED 1-slot masked belief 평가")
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-windows-per-episode", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-open", action="store_true", help="기본은 obstacle이 있는 urban episode만 평가")
    parser.add_argument("--rank-candidates", type=int, default=0, help="0이면 action ranking metric 생략")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--keep", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("output/red_masked_belief_eval.csv"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.max_episodes <= 0 or args.max_windows_per_episode <= 0 or args.max_samples <= 0:
        raise ValueError("max 관련 인자는 0보다 커야 한다")
    if args.rank_candidates < 0:
        raise ValueError("rank-candidates는 음수일 수 없다")
    if args.top_k <= 0 or args.keep <= 0:
        raise ValueError("top-k/keep은 0보다 커야 한다")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    model, config = _load_model(args.checkpoint, device=device)
    episode_dirs = [
        run_dir
        for run_dir in _episode_dirs(args.episode_root)
        if args.include_open or _is_urban(run_dir)
    ][: args.max_episodes]
    if not episode_dirs:
        raise ValueError("평가할 episode가 없다")

    results: list[BeliefMetrics] = []
    skipped_no_hidden = 0
    for run_dir in episode_dirs:
        if len(results) >= args.max_samples:
            break
        run_config = _load_config(run_dir)
        obstacles = run_config.get("obstacles", [])
        duration_sec = float(run_config.get("duration", 40.0))
        windows = _build_windows(
            run_dir,
            history_frames=config.history_frames,
            pred_frames=config.pred_frames,
            max_windows=args.max_windows_per_episode,
        )
        for window in windows:
            if len(results) >= args.max_samples:
                break
            target_id = _choose_hidden_red_target(
                window,
                obstacles=obstacles,
                history_frames=config.history_frames,
                rng=rng,
            )
            if target_id is None:
                skipped_no_hidden += 1
                continue
            try:
                metric = _evaluate_window(
                    model=model,
                    window=window,
                    target_id=target_id,
                    obstacles=obstacles,
                    duration_sec=duration_sec,
                    device=device,
                    rank_candidates=args.rank_candidates,
                    seed=args.seed + len(results),
                    top_k=args.top_k,
                    keep=args.keep,
                )
            except ValueError:
                continue
            results.append(metric)
            rank_text = ""
            if metric.rank_spearman is not None:
                rank_text = f" rank_rho={metric.rank_spearman:+.3f} recall={metric.topk_recall:.2f}"
            print(
                f"{run_dir.name} t={metric.current_time:>4.0f} R{metric.target_id} "
                f"cur_pos={metric.current_pos_error_m:.2f}m(last={metric.current_pos_error_last_seen_m:.2f}) "
                f"future_pos={metric.future_pos_error_m:.2f}m(last={metric.future_pos_error_last_seen_m:.2f}) "
                f"future_hp={metric.future_hp_mae:.3f}(last={metric.future_hp_mae_last_seen:.3f})"
                f"{rank_text}"
            )

    if not results:
        raise ValueError(f"평가 sample이 없다. hidden target skip={skipped_no_hidden}")
    _write_csv(args.out_csv, results)
    print()
    print(f"samples={len(results)} skipped_no_hidden={skipped_no_hidden}")
    print(
        "current pos: "
        f"wm={_mean(row.current_pos_error_m for row in results):.2f}m "
        f"last_seen={_mean(row.current_pos_error_last_seen_m for row in results):.2f}m"
    )
    print(
        "future pos: "
        f"wm={_mean(row.future_pos_error_m for row in results):.2f}m "
        f"last_seen={_mean(row.future_pos_error_last_seen_m for row in results):.2f}m"
    )
    print(
        "future hp: "
        f"wm={_mean(row.future_hp_mae for row in results):.3f} "
        f"last_seen={_mean(row.future_hp_mae_last_seen for row in results):.3f}"
    )
    print(f"future alive acc={_mean(row.future_alive_accuracy for row in results):.3f}")
    if args.rank_candidates > 0:
        print(
            f"rank spearman={_mean(row.rank_spearman for row in results):+.3f} "
            f"top{args.top_k} recall@{args.keep}={_mean(row.topk_recall for row in results):.2f}"
        )
    print(f"csv={args.out_csv}")


if __name__ == "__main__":
    main()
