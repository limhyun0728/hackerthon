"""DEVS episode를 새로 돌리고 종료 뒤 궤적 전체로 월드모델을 학습한다.

이 파일은 저장된 과거 로그를 학습 데이터로 쓰지 않는다. 각 episode에서
CEM commander가 직접 DEVS action을 고르고, episode가 끝난 뒤 그 episode에서
수집한 state/action trajectory만 학습 window로 변환한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from hackerthon.combat_config import MAX_FIRE_RANGE, PERCEPTION_RANGE  # noqa: E402

from pypdevs.DEVS import AtomicDEVS, CoupledDEVS  # noqa: E402
from pypdevs.infinity import INFINITY  # noqa: E402
from pypdevs.simulator import Simulator  # noqa: E402

from hackerthon.simulation_direct_commander_5v5 import CSVLoggerAtomic, RulePolicyAtomic  # noqa: E402
from hackerthon.red_policy import UrbanRedPolicy  # noqa: E402
from hackerthon.sim_units import LosSoldierAtomic, LosWorldAtomic  # noqa: E402
from hackerthon.terrain import DEFAULT_OBSTACLES, clamp_to_world, has_los, next_waypoint  # noqa: E402
from hackerthon.worldmodel.actions import (  # noqa: E402
    ActionBatch,
    ActionType,
    build_action_batch,
    unit_name,
)
from hackerthon.worldmodel.cem_planner import (  # noqa: E402
    CEMConfig,
    CEMDistribution,
    ObservedActionWindow,
    build_initial_distribution,
    optimize_cem,
    rollout_with_world_model,
    sample_future_action_plans,
    score_future_features_torch,
)
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows  # noqa: E402
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig  # noqa: E402
from hackerthon.worldmodel.policy_head import (  # noqa: E402
    MOVE_X_INDEX,
    MOVE_Y_INDEX,
    THETA_COS_INDEX,
    THETA_KNOWN_INDEX,
    THETA_SIN_INDEX,
    PolicyHead,
    PolicyHeadConfig,
    build_policy_guided_distribution,
    compute_policy_loss,
    load_policy,
    save_policy,
)
from hackerthon.worldmodel.slots import (  # noqa: E402
    CURRENT_OBJECTIVE,
    OBJECTIVE_RADIUS,
    ObjectType,
    SlotBatch,
    TeamId,
    build_slot_batch,
)
from hackerthon.worldmodel.train_object_centric_jepa import (  # noqa: E402
    EpisodeTrajectory,
    LoadedTrainingWindow,
    LossWeights,
    OptimizerConfig,
    TrainingWindow,
    build_training_windows,
    build_training_windows_from_episode,
    create_model_and_optimizer,
    evaluate_existing_model,
    load_training_windows,
    save_checkpoint,
    train_existing_model,
)


BLUE_POSITIONS = tuple((101 + index, -8.0, y, 0.0) for index, y in enumerate((-6.0, -3.0, 0.0, 3.0, 6.0)))
RED_POSITIONS = tuple((201 + index, 9.0, y, 180.0) for index, y in enumerate((-6.0, -3.0, 0.0, 3.0, 6.0)))
BLUE_IDS = tuple(position[0] for position in BLUE_POSITIONS)
RED_IDS = tuple(position[0] for position in RED_POSITIONS)
ALL_UNIT_IDS = BLUE_IDS + RED_IDS


def _load_obstacle_config(path: Path) -> tuple[dict[str, object], tuple[tuple[float, float, float, float], ...]]:
    """config.json에서 실제 장애물 rect를 읽는다."""
    config = json.loads(path.read_text(encoding="utf-8"))
    obstacles = tuple(
        tuple(float(value) for value in rect)
        for rect in config["obstacles"]
    )
    return config, obstacles


@dataclass(frozen=True)
class EpisodeResult:
    """한 DEVS episode 실행 결과."""

    episode_index: int
    run_dir: Path
    outcome: str
    trajectory: EpisodeTrajectory
    final_rows: tuple[dict[str, object], ...]
    rollout_windows: tuple = ()


@dataclass(frozen=True)
class WorldModelFutureActionBatch:
    """JEPA rollout에 넘길 BLUE 후보 action + RED 관측 action token."""

    action_features: torch.Tensor
    action_unit_ids: torch.Tensor
    issued_mask: torch.Tensor


def _policy_blue_red_indices(batch: SlotBatch) -> tuple[np.ndarray, np.ndarray]:
    """Policy head 학습에 쓸 BLUE/RED unit slot index를 얻는다."""
    unit = batch.type_ids == int(ObjectType.UNIT)
    blue = np.flatnonzero(unit & (batch.team_ids == int(TeamId.BLUE)))
    red = np.flatnonzero(unit & (batch.team_ids == int(TeamId.RED)))
    if blue.size == 0 or red.size == 0:
        raise ValueError("policy 학습용 BLUE/RED unit slot이 없다")
    return blue, red


def _policy_sample_from_state_action(state: SlotBatch, action: ActionBatch) -> dict[str, np.ndarray]:
    """한 tick의 실제 CEM 선택을 policy head supervised target으로 바꾼다."""
    blue, red = _policy_blue_red_indices(state)
    blue_ids = state.entity_ids[blue]
    red_ids = state.entity_ids[red]
    action_unit_ids = action.unit_ids.astype(np.int64)
    action_indices: list[int] = []
    for blue_id in blue_ids:
        matches = np.flatnonzero(action_unit_ids == int(blue_id))
        if matches.shape != (1,):
            raise ValueError(f"policy 학습 action에서 BLUE unit {int(blue_id)}를 정확히 하나 찾아야 한다")
        action_indices.append(int(matches[0]))
    # 월드모델 action에는 RED rule token도 들어가지만 policy head target은 BLUE 명령만 사용한다.
    blue_action_features = action.features[action_indices]
    blue_issued_mask = action.issued_mask[action_indices]
    blue_action_type_ids = action.action_type_ids[action_indices]
    blue_target_entity_ids = action.target_entity_ids[action_indices]

    target_index = np.full(blue_ids.shape, -1, dtype=np.int64)
    for index, target_id in enumerate(blue_target_entity_ids):
        target_id = int(target_id)
        if target_id < 0:
            continue
        matches = np.flatnonzero(red_ids == target_id)
        if matches.shape != (1,):
            raise ValueError(f"policy 학습 target {target_id}를 RED slot에서 찾을 수 없다")
        target_index[index] = int(matches[0])

    return {
        "features": state.features,
        "feature_mask": state.feature_mask,
        "type_ids": state.type_ids,
        "team_ids": state.team_ids,
        "alive_mask": state.alive_mask,
        "blue_indices": blue,
        "red_indices": red,
        "issued": blue_issued_mask,
        "action_type": blue_action_type_ids,
        "move_dest": blue_action_features[:, [MOVE_X_INDEX, MOVE_Y_INDEX]],
        "theta_known": blue_action_features[:, THETA_KNOWN_INDEX] >= 0.5,
        "theta_vec": blue_action_features[:, [THETA_COS_INDEX, THETA_SIN_INDEX]],
        "target_index": target_index,
    }


def _policy_samples_from_episode(trajectory: EpisodeTrajectory) -> list[dict[str, np.ndarray]]:
    """방금 끝난 episode trajectory 전체를 policy head 학습 샘플로 만든다."""
    state_by_time = {float(state.time_sec): state for state in trajectory.states}
    samples: list[dict[str, np.ndarray]] = []
    for action in trajectory.actions:
        action_time = float(action.time_sec)
        if action_time not in state_by_time:
            raise ValueError(f"policy 학습 state time={action_time}를 찾을 수 없다")
        samples.append(_policy_sample_from_state_action(state_by_time[action_time], action))
    if not samples:
        raise ValueError("policy 학습 샘플이 없다")
    return samples


def _collate_policy_samples(samples: Sequence[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    """동일한 slot layout의 policy 샘플들을 torch batch로 묶는다."""
    first = samples[0]
    for sample in samples[1:]:
        if not np.array_equal(sample["blue_indices"], first["blue_indices"]):
            raise ValueError("policy batch 안의 BLUE slot layout이 다르다")
        if not np.array_equal(sample["red_indices"], first["red_indices"]):
            raise ValueError("policy batch 안의 RED slot layout이 다르다")

    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.stack([sample[key] for sample in samples]), dtype=dtype, device=device)

    return {
        "features": stack("features", torch.float32),
        "feature_mask": stack("feature_mask", torch.bool),
        "type_ids": stack("type_ids", torch.long),
        "team_ids": stack("team_ids", torch.long),
        "alive_mask": stack("alive_mask", torch.bool),
        "blue_indices": torch.as_tensor(first["blue_indices"], dtype=torch.long, device=device),
        "red_indices": torch.as_tensor(first["red_indices"], dtype=torch.long, device=device),
        "issued": stack("issued", torch.bool),
        "action_type": stack("action_type", torch.long),
        "move_dest": stack("move_dest", torch.float32),
        "theta_known": stack("theta_known", torch.bool),
        "theta_vec": stack("theta_vec", torch.float32),
        "target_index": stack("target_index", torch.long),
    }


def _batch_index_groups(count: int, batch_size: int, *, generator: torch.Generator) -> list[list[int]]:
    """policy sample index를 shuffle한 뒤 batch 단위로 나눈다."""
    if count <= 0:
        raise ValueError("policy sample 수는 0보다 커야 한다")
    if batch_size <= 0:
        raise ValueError("policy batch_size는 0보다 커야 한다")
    order = torch.randperm(count, generator=generator).tolist()
    return [order[start:start + batch_size] for start in range(0, count, batch_size)]


def _train_policy_on_episode(
    *,
    policy: PolicyHead,
    optimizer: torch.optim.Optimizer,
    trajectory: EpisodeTrajectory,
    device: torch.device,
    epochs: int,
    batch_size: int,
    gradient_clip_norm: float,
    entropy_weight: float,
    seed: int,
    episode_index: int,
) -> None:
    """episode 종료 뒤 CEM이 실제 선택한 action으로 policy head를 즉시 학습한다."""
    if epochs <= 0:
        raise ValueError("policy 학습 epoch는 0보다 커야 한다")
    if gradient_clip_norm <= 0.0:
        raise ValueError("policy gradient clip norm은 0보다 커야 한다")
    if entropy_weight < 0.0:
        raise ValueError("policy entropy weight는 음수일 수 없다")
    samples = _policy_samples_from_episode(trajectory)
    trainable_parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    if not trainable_parameters:
        raise ValueError("policy head에 학습 가능한 parameter가 없다")

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed + episode_index * 1009)
    policy.to(device)
    for local_epoch in range(1, epochs + 1):
        policy.train()
        groups = _batch_index_groups(len(samples), batch_size, generator=shuffle_generator)
        epoch_values: dict[str, list[float]] = {
            "loss": [],
            "type_loss": [],
            "move_loss": [],
            "target_loss": [],
            "theta_loss": [],
            "entropy_loss": [],
            "type_entropy": [],
            "target_entropy": [],
        }
        for group in groups:
            batch = _collate_policy_samples([samples[index] for index in group], device)
            output = policy(
                features=batch["features"],
                feature_mask=batch["feature_mask"],
                type_ids=batch["type_ids"],
                team_ids=batch["team_ids"],
                alive_mask=batch["alive_mask"],
                blue_indices=batch["blue_indices"],
                red_indices=batch["red_indices"],
            )
            metrics = compute_policy_loss(output, batch, entropy_weight=entropy_weight)
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip_norm)
            optimizer.step()
            for name in epoch_values:
                epoch_values[name].append(float(metrics[name].detach().cpu().item()))
        mean = {name: sum(values) / len(values) for name, values in epoch_values.items()}
        print(
            "policy_epoch="
            f"{episode_index}:{local_epoch} loss={mean['loss']:.6f} "
            f"type={mean['type_loss']:.6f} move={mean['move_loss']:.6f} "
            f"target={mean['target_loss']:.6f} theta={mean['theta_loss']:.6f} "
            f"entropy={mean['entropy_loss']:.6f}"
        )
    policy.eval()


def _world_x_from_norm(x_norm: torch.Tensor) -> float:
    """정규화 x좌표를 DEVS 월드 좌표로 되돌린다."""
    from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN

    value = float(torch.as_tensor(x_norm).detach().cpu().item())
    return (value + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def _world_y_from_norm(y_norm: torch.Tensor) -> float:
    """정규화 y좌표를 DEVS 월드 좌표로 되돌린다."""
    from hackerthon.terrain import WORLD_Y_MAX, WORLD_Y_MIN

    value = float(torch.as_tensor(y_norm).detach().cpu().item())
    return (value + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def _status_to_row(status: Mapping[str, object]) -> dict[str, object]:
    """Soldier status 메시지를 slot/action builder가 읽는 row 형식으로 바꾼다."""
    target_id = status.get("target_id")
    return {
        "time": float(status["time"]),
        "id": int(status["id"]),
        "x": float(status["x"]),
        "y": float(status["y"]),
        "heading": float(status["heading"]),
        "hp": float(status["hp"]),
        "ammo": int(status["ammo"]),
        "mode": str(status["mode"]),
        "target_id": "" if target_id is None else target_id,
    }


def _slot_unit_rows(slot_batch: SlotBatch) -> list[dict[str, object]]:
    """SlotBatch에서 action command 변환에 필요한 unit row를 복원한다."""
    rows: list[dict[str, object]] = []
    for index, entity_id in enumerate(slot_batch.entity_ids):
        unit_id = int(entity_id)
        if unit_id <= 0:
            continue
        if unit_id < 100:
            continue
        hp_ratio = float(slot_batch.features[index, 1])
        ammo_ratio = float(slot_batch.features[index, 2])
        rows.append(
            {
                "time": float(slot_batch.time_sec),
                "id": unit_id,
                "x": _world_x_from_norm(slot_batch.features[index, 3]),
                "y": _world_y_from_norm(slot_batch.features[index, 4]),
                "heading": math.degrees(math.atan2(float(slot_batch.features[index, 6]), float(slot_batch.features[index, 5]))),
                "hp": hp_ratio * 100.0,
                "ammo": int(round(ammo_ratio * 30.0)),
                "mode": "IDLE",
                "target_id": "",
            }
        )
    return rows


def _alive_blue_ids_from_rows(rows: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
    """현재 state row에서 살아 있는 BLUE id를 고른다."""
    return tuple(
        sorted(
            int(row["id"])
            for row in rows
            if int(row["id"]) in BLUE_IDS and float(row["hp"]) > 0.0
        )
    )


def _all_red_destroyed(rows: Sequence[Mapping[str, object]]) -> bool:
    """현재 state에서 RED 전멸 여부를 계산한다."""
    red_rows = [row for row in rows if int(row["id"]) in RED_IDS]
    if not red_rows:
        raise ValueError("RED unit row가 없다")
    return all(float(row["hp"]) <= 0.0 for row in red_rows)


def _any_blue_alive(rows: Sequence[Mapping[str, object]]) -> bool:
    """현재 state에서 BLUE 생존 여부를 계산한다."""
    blue_rows = [row for row in rows if int(row["id"]) in BLUE_IDS]
    if not blue_rows:
        raise ValueError("BLUE unit row가 없다")
    return any(float(row["hp"]) > 0.0 for row in blue_rows)


def _objective_reached(rows: Sequence[Mapping[str, object]]) -> bool:
    """살아 있는 BLUE가 objective 반경에 들어왔는지 본다."""
    for row in rows:
        if int(row["id"]) not in BLUE_IDS:
            continue
        if float(row["hp"]) <= 0.0:
            continue
        distance = math.hypot(float(row["x"]) - CURRENT_OBJECTIVE[0], float(row["y"]) - CURRENT_OBJECTIVE[1])
        if distance <= OBJECTIVE_RADIUS:
            return True
    return False


def determine_outcome(rows: Sequence[Mapping[str, object]]) -> str:
    """episode 종료 state로 승패/시간초과를 판정한다."""
    if _all_red_destroyed(rows) and _objective_reached(rows):
        return "WIN"
    if not _any_blue_alive(rows):
        return "LOSE"
    return "TIMEOUT"


def _objective_distance(rows: Sequence[Mapping[str, object]]) -> float:
    """살아 있는 BLUE 중 objective에 가장 가까운 거리를 계산한다."""
    distances = [
        math.hypot(float(row["x"]) - CURRENT_OBJECTIVE[0], float(row["y"]) - CURRENT_OBJECTIVE[1])
        for row in rows
        if int(row["id"]) in BLUE_IDS and float(row["hp"]) > 0.0
    ]
    if not distances:
        return float("inf")
    return min(distances)


def _episode_summary(result: EpisodeResult) -> dict[str, object]:
    """긴 tick CSV 없이 학습 진행 판단에 필요한 episode 요약만 만든다."""
    blue_rows = [row for row in result.final_rows if int(row["id"]) in BLUE_IDS]
    red_rows = [row for row in result.final_rows if int(row["id"]) in RED_IDS]
    if not blue_rows or not red_rows:
        raise ValueError("episode summary를 만들 BLUE/RED row가 없다")
    final_time = max(float(row["time"]) for row in result.final_rows)
    return {
        "episode": int(result.episode_index),
        "run_dir": str(result.run_dir),
        "outcome": result.outcome,
        "final_time": final_time,
        "blue_alive": sum(float(row["hp"]) > 0.0 for row in blue_rows),
        "red_alive": sum(float(row["hp"]) > 0.0 for row in red_rows),
        "blue_hp": sum(max(float(row["hp"]), 0.0) for row in blue_rows),
        "red_hp": sum(max(float(row["hp"]), 0.0) for row in red_rows),
        "objective_distance": _objective_distance(result.final_rows),
        "states": len(result.trajectory.states),
        "actions": len(result.trajectory.actions),
        "rollout_windows": len(result.rollout_windows),
    }


def _append_episode_summary(output_root: Path, result: EpisodeResult) -> None:
    """장기 학습용으로 episode 최종 요약만 JSONL에 누적 저장한다."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "episode_summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_episode_summary(result), ensure_ascii=False) + "\n")


def _append_validation_summary(output_root: Path, row: Mapping[str, object]) -> None:
    """고정 validation 결과와 early-stopping 상태를 JSONL로 누적한다."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "validation_summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _scheduled_policy_prior_mix(
    *,
    episode_index: int,
    warmup_episodes: int,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    """policy가 켜진 뒤 CEM 기본 prior를 섞는 비율을 선형으로 낮춘다."""
    if decay_episodes <= 0:
        return float(end)
    # episode 1부터 세므로 warmup 직후 첫 policy 사용 구간은 start에 가깝게 둔다.
    active_episode = max(0, int(episode_index) - int(warmup_episodes) - 1)
    ratio = min(1.0, float(active_episode) / float(decay_episodes))
    return float(start + (end - start) * ratio)


def _scheduled_policy_entropy_weight(
    *,
    episode_index: int,
    warmup_episodes: int,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    """policy entropy 보너스를 warmup 이후 높은 값에서 낮은 값으로 선형 감소시킨다."""
    if decay_episodes <= 0:
        return float(end)
    active_episode = max(0, int(episode_index) - int(warmup_episodes) - 1)
    ratio = min(1.0, float(active_episode) / float(decay_episodes))
    return float(start + (end - start) * ratio)


def _distance_between_rows(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    """두 unit row 사이의 월드 거리."""
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _normalize_angle_deg(angle: float) -> float:
    """DEVS Soldier와 같은 방식으로 상대각을 [-180, 180)으로 정규화한다."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def _unit_entity_type(unit_id: int) -> str:
    """World observation에 들어가는 BLUE/RED entity type 이름."""
    if int(unit_id) in BLUE_IDS:
        return "soldier"
    if int(unit_id) in RED_IDS:
        return "enemy"
    raise ValueError(f"알 수 없는 unit id: {unit_id}")


def _unit_state_name(row: Mapping[str, object]) -> str:
    """row의 hp/mode를 World observation state 문자열로 바꾼다."""
    if float(row["hp"]) <= 0.0 or str(row.get("mode", "")).upper() == "DESTROYED":
        return "DESTROYED"
    return "ALIVE"


def _is_enemy_pair(observer_id: int, entity_id: int) -> bool:
    """observer 기준 enemy unit인지 판정한다."""
    if int(observer_id) in BLUE_IDS:
        return int(entity_id) in RED_IDS
    if int(observer_id) in RED_IDS:
        return int(entity_id) in BLUE_IDS
    raise ValueError(f"알 수 없는 observer id: {observer_id}")


def _visible_entities_for_unit(
    *,
    observer_row: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    obstacles: Sequence[Sequence[float]],
    fov_deg: float = 120.0,
) -> list[dict[str, object]]:
    """SlotBatch 복원 row에서 RED rule이 받을 visible_entities를 만든다."""
    observer_id = int(observer_row["id"])
    sx = float(observer_row["x"])
    sy = float(observer_row["y"])
    heading = float(observer_row["heading"])
    visible: list[dict[str, object]] = []
    for row in rows:
        entity_id = int(row["id"])
        if entity_id == observer_id:
            continue
        entity_state = _unit_state_name(row)
        if entity_state == "DESTROYED" and not _is_enemy_pair(observer_id, entity_id):
            continue
        ex = float(row["x"])
        ey = float(row["y"])
        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)
        if distance > PERCEPTION_RANGE:
            continue
        relative_theta = _normalize_angle_deg(math.degrees(math.atan2(dy, dx)) - heading)
        if abs(relative_theta) > fov_deg / 2.0:
            continue
        if not has_los((sx, sy), (ex, ey), obstacles):
            continue
        visible.append(
            {
                "id": entity_id,
                "type": _unit_entity_type(entity_id),
                "x": round(ex, 3),
                "y": round(ey, 3),
                "hp": float(row["hp"]),
                "heading": float(row["heading"]),
                "r": round(distance, 3),
                "theta": round(relative_theta, 2),
                "state": entity_state,
            }
        )
    visible.sort(key=lambda entity: float(entity["r"]))
    return visible


def _normalize_probability_vector(values: torch.Tensor) -> torch.Tensor:
    """0이 아닌 합을 가진 categorical 확률 벡터를 정규화한다."""
    if values.ndim != 1:
        raise ValueError("확률 벡터 rank는 1이어야 한다")
    if torch.any(values < 0.0):
        raise ValueError("확률 벡터에는 음수가 있으면 안 된다")
    total = values.sum()
    if float(total.detach().cpu().item()) <= 0.0:
        raise ValueError("확률 벡터 합은 0보다 커야 한다")
    return values / total


def _format_command_detail(command: Mapping[str, object]) -> str:
    """commands_log 확인용 detail 문자열을 만든다."""
    action = str(command["action"]).upper()
    if action == "MOVE":
        return f"({command['x']},{command['y']})"
    if action == "ENGAGE":
        return f"->{unit_name(int(command['target_id']))}"
    if action == "TURN":
        return str(command["theta"])
    return ""


class CEMCommanderAtomic(AtomicDEVS):
    """전체 DEVS state를 받아 CEM으로 BLUE unit별 action을 직접 고르는 commander."""

    def __init__(
        self,
        *,
        controlled_ids: Sequence[int],
        all_unit_ids: Sequence[int],
        obstacles: Sequence[Sequence[float]],
        duration_sec: float,
        run_dir: Path,
        model: DEVSObjectCentricWorldModel,
        model_config: ObjectSlotModelConfig,
        cem_config: CEMConfig,
        device: torch.device,
        rollout_backend: str = "jepa",
        episode_seed: int | None = None,
        rollout_replay_per_tick: int = 0,
        policy: PolicyHead | None = None,
        policy_prior_mix: float = 0.25,
        red_target_priority: str = "nearest",
        write_csv_logs: bool = True,
        dt: float = 1.0,
    ):
        super().__init__("CEMCommander")
        # 학습된 proposer. 있으면 CEM 초기 분포를 policy 출력으로 세운다.
        self.policy = policy
        if not 0.0 <= float(policy_prior_mix) <= 1.0:
            raise ValueError("policy_prior_mix는 [0, 1] 범위여야 한다")
        # 초반에는 기본 CEM prior를 더 섞어 사격/정지 탐색이 policy 편향에 눌리지 않게 한다.
        self.policy_prior_mix = float(policy_prior_mix)
        if red_target_priority not in ("nearest", "low_hp", "smart"):
            raise ValueError("red_target_priority는 nearest, low_hp 또는 smart여야 한다")
        self.red_target_priority = red_target_priority
        self.controlled_ids = tuple(int(unit_id) for unit_id in controlled_ids)
        self.all_unit_ids = tuple(int(unit_id) for unit_id in all_unit_ids)
        self.obstacles = tuple(tuple(float(value) for value in rect) for rect in obstacles)
        self.duration_sec = float(duration_sec)
        self.run_dir = Path(run_dir)
        self.model = model
        self.model_config = model_config
        self.cem_config = cem_config
        self.device = device
        if rollout_backend not in ("jepa", "devs"):
            raise ValueError("rollout_backend는 jepa 또는 devs여야 한다")
        self.rollout_backend = rollout_backend
        self.dt = float(dt)
        # jepa rollout은 한 번의 예측 길이가 pred_frames로 고정된다. devs rollout은
        # 실제 시뮬레이션이므로 horizon을 pred_frames와 독립적으로 늘릴 수 있다.
        if self.rollout_backend == "jepa" and self.cem_config.future_horizon != self.model_config.pred_frames:
            raise ValueError("jepa backend에서 CEM future_horizon은 model pred_frames와 같아야 한다")

        self.status_in = self.addInPort("status_in")
        self.red_command_in = self.addInPort("red_command_in")
        self.orders_out = {unit_id: self.addOutPort(f"orders_out_{unit_id}") for unit_id in self.controlled_ids}
        self.sigma = INFINITY
        self.state = "WAIT"
        self._rows_by_time: dict[float, dict[int, dict[str, object]]] = {}
        self._states_by_time: dict[float, SlotBatch] = {}
        self._command_rows_by_time: dict[float, list[dict[str, object]]] = {}
        self._blue_command_times: set[float] = set()
        self._latest_state_time: float | None = None
        self.write_csv_logs = bool(write_csv_logs)
        # episode마다 다른 CEM 샘플링/rollout 난수를 쓰기 위한 seed. 고정 cem seed만
        # 쓰면 (특히 devs backend에서) 모든 episode가 같은 trajectory로 반복된다.
        self.episode_seed = int(self.cem_config.seed if episode_seed is None else episode_seed)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.episode_seed)
        # DEVS rollout 결과를 학습 데이터로 회수할 tick당 후보 수 (0이면 비활성).
        self.rollout_replay_per_tick = int(rollout_replay_per_tick)
        self.rollout_windows: list[LoadedTrainingWindow] = []
        self._replay_generator = torch.Generator(device=self.device)
        self._replay_generator.manual_seed(self.episode_seed + 9999)

        self._command_log = None
        self._command_writer = None
        self._planner_log = None
        self._planner_writer = None
        if self.write_csv_logs:
            self._command_log = (self.run_dir / "commands_log.csv").open("w", newline="", encoding="utf-8")
            self._command_writer = csv.DictWriter(
                self._command_log,
                fieldnames=["time", "unit_id", "role", "action", "detail", "reason"],
            )
            self._command_writer.writeheader()
            self._planner_log = (self.run_dir / "planner_log.csv").open("w", newline="", encoding="utf-8")
            self._planner_writer = csv.DictWriter(
                self._planner_log,
                fieldnames=["time", "selector", "best_score", "population_mean", "commands"],
            )
            self._planner_writer.writeheader()

    def timeAdvance(self):
        return self.sigma

    def extTransition(self, inputs):
        if self.red_command_in in inputs:
            for command in inputs[self.red_command_in]:
                self._record_red_command(command)
        if self.status_in in inputs:
            for status in inputs[self.status_in]:
                row = _status_to_row(status)
                status_time = float(row["time"])
                unit_id = int(row["id"])
                self._rows_by_time.setdefault(status_time, {})[unit_id] = row
            self._record_complete_states()
            if self._latest_state_time is not None:
                self.sigma = 0.0
                self.state = "READY"
        return self.state

    def intTransition(self):
        self.sigma = INFINITY
        self.state = "WAIT"
        return self.state

    def outputFnc(self):
        if self._latest_state_time is None:
            return {}
        current_time = float(self._latest_state_time)
        if current_time in self._blue_command_times:
            return {}
        current_batch = self._states_by_time[current_time]
        current_rows = tuple(self._rows_by_time[current_time][unit_id] for unit_id in self.all_unit_ids)
        if not _alive_blue_ids_from_rows(current_rows):
            return {}

        plan, selector, best_score, population_mean = self._select_plan(current_batch)
        commands = self._commands_from_plan(plan, current_batch=current_batch, step_index=0)
        self._record_commands(current_time, commands, selector=selector, best_score=best_score, population_mean=population_mean)

        out = {}
        for command in commands:
            port = self.orders_out.get(int(command["unit_id"]))
            if port is not None:
                out[port] = [command]
        return out

    def close(self) -> None:
        """episode 종료 뒤 열린 log 파일을 닫는다."""
        if self._command_log is not None:
            self._command_log.flush()
            self._command_log.close()
        if self._planner_log is not None:
            self._planner_log.flush()
            self._planner_log.close()

    def trajectory(self, *, episode_id: str, outcome: str) -> EpisodeTrajectory:
        """수집된 state/action을 episode trajectory로 변환한다."""
        states = tuple(self._states_by_time[time_value] for time_value in sorted(self._states_by_time))
        actions: list[ActionBatch] = []
        for time_value in sorted(self._command_rows_by_time):
            if time_value not in self._rows_by_time:
                continue
            current_rows = tuple(self._rows_by_time[time_value][unit_id] for unit_id in self.all_unit_ids)
            actions.append(
                build_action_batch(
                    unit_ids=BLUE_IDS,
                    command_rows=self._command_rows_by_time[time_value],
                    current_unit_rows=current_rows,
                    time_sec=time_value,
                    allow_extra_commands=True,
                )
            )
        return EpisodeTrajectory(
            episode_id=episode_id,
            states=states,
            actions=tuple(actions),
            outcome=outcome,
        )

    def latest_rows(self) -> tuple[dict[str, object], ...]:
        """마지막 complete state의 unit row를 반환한다."""
        if self._latest_state_time is None:
            raise ValueError("complete state가 아직 없다")
        return tuple(self._rows_by_time[self._latest_state_time][unit_id] for unit_id in self.all_unit_ids)

    def _record_complete_states(self) -> None:
        """모든 unit status가 모인 tick을 SlotBatch로 기록한다."""
        for status_time in sorted(self._rows_by_time):
            if status_time in self._states_by_time:
                continue
            rows_by_unit = self._rows_by_time[status_time]
            if any(unit_id not in rows_by_unit for unit_id in self.all_unit_ids):
                continue
            rows = tuple(rows_by_unit[unit_id] for unit_id in self.all_unit_ids)
            self._states_by_time[status_time] = build_slot_batch(
                unit_rows=rows,
                obstacles=self.obstacles,
                time_sec=status_time,
                duration_sec=self.duration_sec,
            )
            self._latest_state_time = status_time

    def _select_plan(self, current_batch: SlotBatch):
        """history가 충분하면 CEM optimize, 부족하면 명시적 warmup sampler를 쓴다.

        devs backend는 현재 state 스냅샷만으로 rollout하므로 history 없이도
        첫 tick부터 CEM을 쓸 수 있다.
        """
        if self.rollout_backend == "devs" or self._has_model_history(current_batch):
            return self._select_with_cem(current_batch)
        distribution = self._apply_current_engage_hard_mask(
            build_initial_distribution(current_batch, self.cem_config, device=self.device),
            current_batch=current_batch,
        )
        plan = sample_future_action_plans(
            distribution=distribution,
            current_batch=current_batch,
            config=self.cem_config,
            generator=self._generator,
            device=self.device,
        ).take_candidates(torch.tensor([0], dtype=torch.long, device=self.device))
        return plan, "warmup", 0.0, 0.0

    def _has_model_history(self, current_batch: SlotBatch) -> bool:
        """현재 tick에서 월드모델 rollout에 필요한 history/action이 있는지 확인한다."""
        history_times = tuple(
            float(current_batch.time_sec - offset * self.dt)
            for offset in reversed(range(self.model_config.history_frames))
        )
        action_times = history_times[:-1]
        return (
            all(time_value in self._states_by_time for time_value in history_times)
            and all(time_value in self._command_rows_by_time for time_value in action_times)
            and all(time_value in self._blue_command_times for time_value in action_times)
            and all(time_value in self._rows_by_time for time_value in action_times)
        )

    def _jepa_rollout_fn(self, current_batch: SlotBatch):
        """학습 중인 월드모델로 후보 미래를 예측하는 rollout_fn을 만든다."""
        history_times = tuple(
            float(current_batch.time_sec - offset * self.dt)
            for offset in reversed(range(self.model_config.history_frames))
        )
        action_times = history_times[:-1]
        history_batches = tuple(self._states_by_time[time_value] for time_value in history_times)
        observed_batches = tuple(self._action_batch_for_time(time_value) for time_value in action_times)
        observed = ObservedActionWindow(
            action_features=torch.stack(
                [torch.as_tensor(action.features, dtype=torch.float32, device=self.device) for action in observed_batches],
                dim=0,
            ),
            action_unit_ids=torch.stack(
                [torch.as_tensor(action.unit_ids, dtype=torch.long, device=self.device) for action in observed_batches],
                dim=0,
            ),
            issued_mask=torch.stack(
                [torch.as_tensor(action.issued_mask, dtype=torch.bool, device=self.device) for action in observed_batches],
                dim=0,
            ),
        )

        def rollout_fn(plans):
            return rollout_with_world_model(
                model=self.model,
                history_batches=history_batches,
                observed_actions=observed,
                future_plans=self._blue_only_future_action_batch(plans, current_batch=current_batch),
                device=self.device,
            )

        return rollout_fn

    def _devs_rollout_fn(self, current_batch: SlotBatch):
        """실제 DEVS 단기 시뮬레이션으로 후보 미래를 계산하는 rollout_fn을 만든다."""
        current_time = float(current_batch.time_sec)
        rows = tuple(self._rows_by_time[current_time][unit_id] for unit_id in self.all_unit_ids)
        snapshot = snapshot_from_slot_rows(
            unit_rows=rows,
            obstacles=self.obstacles,
            base_time_sec=current_time,
            episode_duration_sec=self.duration_sec,
        )
        # tick마다 다른 seed, 같은 tick의 후보끼리는 같은 seed(common random numbers).
        # episode_seed를 섞어 episode 간에도 rollout 난수가 달라지게 한다.
        rollout_seed = self.episode_seed * 100003 + int(round(current_time))

        holder: dict = {}

        def rollout_fn(plans):
            features = rollout_plans_with_devs(
                plans=plans,
                snapshot=snapshot,
                seed=rollout_seed,
                device=self.device,
                red_target_priority=self.red_target_priority,
            )
            # 마지막 CEM iteration의 (후보, 실제 미래) 쌍을 학습 회수용으로 보관한다.
            holder["plans"] = plans
            holder["features"] = features
            return features

        return rollout_fn, holder

    def _future_slot_batch(self, current_batch: SlotBatch, frame_features: np.ndarray, time_sec: float) -> SlotBatch:
        """DEVS rollout 미래 frame feature를 SlotBatch로 감싼다."""
        is_unit = current_batch.type_ids == int(ObjectType.UNIT)
        alive = np.ones(current_batch.alive_mask.shape, dtype=bool)
        alive[is_unit] = frame_features[is_unit, 7] >= 0.5
        return SlotBatch(
            features=frame_features.astype(np.float32),
            feature_mask=current_batch.feature_mask.copy(),
            type_ids=current_batch.type_ids.copy(),
            entity_ids=current_batch.entity_ids.copy(),
            team_ids=current_batch.team_ids.copy(),
            alive_mask=alive,
            names=current_batch.names,
            time_sec=float(time_sec),
        )

    def _red_rule_action_batch(self, state_batch: SlotBatch, *, time_sec: float) -> ActionBatch:
        """현재/후보 state에서 RED rule을 평가해 RED action batch를 만든다."""
        rows = _slot_unit_rows(state_batch)
        row_by_id = {int(row["id"]): row for row in rows}
        commands: list[dict[str, object]] = []
        for red_id in RED_IDS:
            row = row_by_id[red_id]
            if float(row["hp"]) <= 0.0:
                continue
            observation = {
                "time": float(time_sec),
                "self": {
                    "id": int(red_id),
                    "x": round(float(row["x"]), 3),
                    "y": round(float(row["y"]), 3),
                    "hp": float(row["hp"]),
                    "ammo": int(row["ammo"]),
                    "heading": round(float(row["heading"]), 2),
                    "mode": str(row["mode"]),
                },
                "visible_entities": _visible_entities_for_unit(
                    observer_row=row,
                    rows=rows,
                    obstacles=self.obstacles,
                    fov_deg=120.0,
                ),
            }
            # 후보 rollout용 RED token은 후보 state에서 rule을 즉시 평가한 외생 반응이다.
            command = dict(
                UrbanRedPolicy(
                    target_type="soldier",
                    obstacles=self.obstacles,
                    target_priority=self.red_target_priority,
                ).decide(observation)
            )
            command["time"] = float(time_sec)
            command["duration_sec"] = float(command.get("duration_sec", 1.0))
            commands.append(command)
        return build_action_batch(
            unit_ids=RED_IDS,
            command_rows=commands,
            current_unit_rows=rows,
            time_sec=time_sec,
        )

    def _empty_red_action_batch(self, state_batch: SlotBatch, *, time_sec: float) -> ActionBatch:
        """RED command가 아직 출력되지 않은 tick의 빈 RED action batch를 만든다."""
        return build_action_batch(
            unit_ids=RED_IDS,
            command_rows=(),
            current_unit_rows=_slot_unit_rows(state_batch),
            time_sec=time_sec,
        )

    def _recorded_red_action_batch(self, time_sec: float) -> ActionBatch:
        """이미 실제 RED brain이 출력한 같은 tick RED command를 action batch로 만든다."""
        current_rows = tuple(self._rows_by_time[time_sec][unit_id] for unit_id in self.all_unit_ids)
        command_rows = [
            row
            for row in self._command_rows_by_time.get(float(time_sec), [])
            if int(row["unit_id"]) in RED_IDS
        ]
        return build_action_batch(
            unit_ids=RED_IDS,
            command_rows=command_rows,
            current_unit_rows=current_rows,
            time_sec=time_sec,
        )

    def _blue_only_future_action_batch(self, plans, *, current_batch: SlotBatch) -> WorldModelFutureActionBatch:
        """Inference용 월드모델에는 BLUE 후보 action token만 넘긴다."""
        del current_batch
        return WorldModelFutureActionBatch(
            action_features=plans.action_features,
            action_unit_ids=plans.action_unit_ids,
            issued_mask=plans.issued_mask,
        )

    def _future_action_batch(
        self,
        plans,
        candidate: int,
        step: int,
        *,
        time_sec: float,
    ) -> ActionBatch:
        """CEM BLUE 후보 action만 한 step ActionBatch로 만든다."""
        blue_unit_ids = plans.action_unit_ids[candidate, step].detach().cpu().numpy().astype(np.int64)
        return ActionBatch(
            features=plans.action_features[candidate, step].detach().cpu().numpy().astype(np.float32),
            issued_mask=plans.issued_mask[candidate, step].detach().cpu().numpy().astype(bool),
            unit_ids=blue_unit_ids,
            action_type_ids=plans.action_type_ids[candidate, step].detach().cpu().numpy().astype(np.int64),
            target_entity_ids=plans.target_entity_ids[candidate, step].detach().cpu().numpy().astype(np.int64),
            names=tuple(unit_name(int(unit_id)) for unit_id in blue_unit_ids),
            time_sec=float(time_sec),
        )

    def _collect_rollout_replay(self, current_batch: SlotBatch, holder: dict) -> None:
        """마지막 CEM iteration의 DEVS rollout 결과를 학습 window로 회수한다.

        점수 상위 절반 + 무작위 절반을 골라 (history 실측 + 후보 액션 + DEVS 실제
        미래)를 그대로 학습 window로 만든다. 실행되지 않은 반사실적 액션의 결과가
        학습 분포에 들어오는 것이 목적이다.
        """
        if self.rollout_replay_per_tick <= 0 or "plans" not in holder:
            return
        if self.cem_config.future_horizon != self.model_config.pred_frames:
            return
        if not self._has_model_history(current_batch):
            return
        current_time = float(current_batch.time_sec)
        history_times = tuple(
            float(current_time - offset * self.dt)
            for offset in reversed(range(self.model_config.history_frames))
        )
        history_states = tuple(self._states_by_time[time_value] for time_value in history_times)
        observed_actions = tuple(self._action_batch_for_time(time_value) for time_value in history_times[:-1])

        plans = holder["plans"]
        features = holder["features"]
        scores = score_future_features_torch(current_batch=current_batch, future_features=features)
        count = min(self.rollout_replay_per_tick, int(scores.shape[0]))
        top_count = max(1, count // 2)
        order = torch.argsort(scores, descending=True)
        chosen = order[:top_count].tolist()
        rest = order[top_count:]
        if rest.numel() > 0 and count > top_count:
            perm = torch.randperm(rest.numel(), generator=self._replay_generator, device=rest.device)
            chosen += rest[perm[: count - top_count]].tolist()

        horizon = self.cem_config.future_horizon
        features_np = features.detach().cpu().numpy()
        for candidate in chosen:
            future_states = tuple(
                self._future_slot_batch(
                    current_batch,
                    features_np[candidate, step],
                    current_time + (step + 1) * self.dt,
                )
                for step in range(horizon)
            )
            future_actions = tuple(
                self._future_action_batch(
                    plans,
                    candidate,
                    step,
                    time_sec=current_time + step * self.dt,
                )
                for step in range(horizon)
            )
            states = history_states + future_states
            actions = observed_actions + future_actions
            self.rollout_windows.append(
                LoadedTrainingWindow(
                    spec=TrainingWindow(
                        run_dir=self.run_dir,
                        state_times=tuple(float(state.time_sec) for state in states),
                        action_times=tuple(float(action.time_sec) for action in actions),
                    ),
                    states=states,
                    actions=actions,
                )
            )

    def _select_with_cem(self, current_batch: SlotBatch):
        """CEM으로 future action sequence를 고른다. 후보 평가는 rollout_backend가 정한다."""
        current_time = float(current_batch.time_sec)
        # optimize_cem은 config.seed로 내부 generator를 초기화한다. episode/tick을
        # 섞지 않으면 매 episode가 같은 후보를 샘플링해 같은 trajectory를 반복한다.
        cem_seed = self.episode_seed * 1009 + int(round(current_time / self.dt))
        cem_config = replace(self.cem_config, seed=cem_seed)
        if self.policy is not None:
            base_distribution = build_policy_guided_distribution(
                policy=self.policy,
                current_batch=current_batch,
                cem_config=cem_config,
                device=self.device,
                prior_mix=self.policy_prior_mix,
            )
        else:
            base_distribution = build_initial_distribution(current_batch, cem_config, device=self.device)
        distribution = self._apply_current_engage_hard_mask(base_distribution, current_batch=current_batch)
        rollout_holder: dict | None = None
        if self.rollout_backend == "devs":
            rollout_fn, rollout_holder = self._devs_rollout_fn(current_batch)
        else:
            rollout_fn = self._jepa_rollout_fn(current_batch)

        def score_fn(future_features):
            return score_future_features_torch(current_batch=current_batch, future_features=future_features)

        result = optimize_cem(
            current_batch=current_batch,
            initial_distribution=distribution,
            config=cem_config,
            rollout_fn=rollout_fn,
            score_fn=score_fn,
            device=self.device,
        )
        if rollout_holder is not None:
            self._collect_rollout_replay(current_batch, rollout_holder)
        population_mean = result.iteration_stats[-1].population_mean
        return result.best_plan, f"cem_{self.rollout_backend}", result.best_score, population_mean

    def _engage_allowed(
        self,
        *,
        shooter_id: int,
        target_id: int,
        row_by_id: Mapping[int, Mapping[str, object]],
    ) -> bool:
        """현재 DEVS state에서 ENGAGE가 실제 피해 가능성이 있는지 판정한다."""
        if shooter_id not in row_by_id:
            raise ValueError(f"shooter {shooter_id}가 현재 state에 없다")
        if target_id not in row_by_id:
            raise ValueError(f"target {target_id}가 현재 state에 없다")
        if shooter_id not in BLUE_IDS:
            raise ValueError(f"shooter {shooter_id}는 BLUE unit이어야 한다")
        if target_id not in RED_IDS:
            raise ValueError(f"target {target_id}는 RED unit이어야 한다")
        shooter = row_by_id[shooter_id]
        target = row_by_id[target_id]
        if float(shooter["hp"]) <= 0.0:
            return False
        if int(shooter["ammo"]) <= 0:
            return False
        if float(target["hp"]) <= 0.0:
            return False
        shooter_pos = (float(shooter["x"]), float(shooter["y"]))
        target_pos = (float(target["x"]), float(target["y"]))
        if _distance_between_rows(shooter, target) > MAX_FIRE_RANGE:
            return False
        return has_los(shooter_pos, target_pos, self.obstacles)

    def _current_engage_allowed_mask(self, current_batch: SlotBatch) -> torch.Tensor:
        """현재 state에서 BLUE x RED ENGAGE 가능 mask를 만든다."""
        rows = _slot_unit_rows(current_batch)
        row_by_id = {int(row["id"]): row for row in rows}
        mask = torch.zeros((len(BLUE_IDS), len(RED_IDS)), dtype=torch.bool, device=self.device)
        for blue_index, blue_id in enumerate(BLUE_IDS):
            for red_index, red_id in enumerate(RED_IDS):
                mask[blue_index, red_index] = self._engage_allowed(
                    shooter_id=blue_id,
                    target_id=red_id,
                    row_by_id=row_by_id,
                )
        return mask

    def _apply_current_engage_hard_mask(
        self,
        distribution: CEMDistribution,
        *,
        current_batch: SlotBatch,
    ) -> CEMDistribution:
        """CEM 첫 action에서 불가능한 ENGAGE target을 샘플링하지 않게 막는다."""
        allowed = self._current_engage_allowed_mask(current_batch)
        action_probs = distribution.action_probs.clone()
        target_probs = distribution.target_probs.clone()
        for unit_index in range(allowed.shape[0]):
            valid_targets = allowed[unit_index].float()
            if torch.any(allowed[unit_index]):
                target_probs[0, unit_index] = _normalize_probability_vector(valid_targets)
            else:
                action_probs[0, unit_index, int(ActionType.ENGAGE)] = 0.0
                target_probs[0, unit_index] = torch.full_like(
                    target_probs[0, unit_index],
                    1.0 / float(target_probs.shape[-1]),
                )
            action_probs[0, unit_index] = _normalize_probability_vector(action_probs[0, unit_index])
        return CEMDistribution(
            action_probs=action_probs,
            move_mean=distribution.move_mean,
            move_std=distribution.move_std,
            turn_mean=distribution.turn_mean,
            turn_std=distribution.turn_std,
            target_probs=target_probs,
        )

    def _action_batch_for_time(self, time_value: float) -> ActionBatch:
        """저장된 command row와 같은 tick state row로 ActionBatch를 만든다."""
        current_rows = tuple(self._rows_by_time[time_value][unit_id] for unit_id in self.all_unit_ids)
        return build_action_batch(
            unit_ids=BLUE_IDS,
            command_rows=self._command_rows_by_time[time_value],
            current_unit_rows=current_rows,
            time_sec=time_value,
            allow_extra_commands=True,
        )

    def _commands_from_plan(
        self,
        plan,
        *,
        current_batch: SlotBatch,
        step_index: int,
    ) -> list[dict[str, object]]:
        """CEM plan의 한 step을 Soldier가 받는 DEVS command dict로 변환한다."""
        rows = _slot_unit_rows(current_batch)
        row_by_id = {int(row["id"]): row for row in rows}
        commands: list[dict[str, object]] = []
        for unit_index in range(plan.action_features.shape[2]):
            unit_id = int(plan.action_unit_ids[0, step_index, unit_index].detach().cpu().item())
            if unit_id not in row_by_id:
                raise ValueError(f"CEM plan unit {unit_id}가 현재 state에 없다")
            if not bool(plan.issued_mask[0, step_index, unit_index].detach().cpu().item()):
                continue
            action_type_id = int(plan.action_type_ids[0, step_index, unit_index].detach().cpu().item())
            if action_type_id == 0:
                commands.append({"unit_id": unit_id, "action": "STOP", "duration_sec": 1.0, "reason": "cem stop"})
            elif action_type_id == 1:
                target_x = _world_x_from_norm(plan.move_xy_norm[0, step_index, unit_index, 0])
                target_y = _world_y_from_norm(plan.move_xy_norm[0, step_index, unit_index, 1])
                target_x, target_y = clamp_to_world(target_x, target_y)
                current_pos = (float(row_by_id[unit_id]["x"]), float(row_by_id[unit_id]["y"]))
                waypoint = next_waypoint(current_pos, (target_x, target_y), self.obstacles, max_step=1.5)
                if waypoint is None:
                    commands.append({"unit_id": unit_id, "action": "STOP", "duration_sec": 1.0, "reason": "cem move blocked"})
                else:
                    commands.append(
                        {
                            "unit_id": unit_id,
                            "action": "MOVE",
                            "x": round(float(waypoint[0]), 3),
                            "y": round(float(waypoint[1]), 3),
                            "duration_sec": 1.0,
                            "reason": "cem move",
                        }
                    )
            elif action_type_id == 2:
                target_id = int(plan.target_entity_ids[0, step_index, unit_index].detach().cpu().item())
                if self._engage_allowed(shooter_id=unit_id, target_id=target_id, row_by_id=row_by_id):
                    commands.append(
                        {
                            "unit_id": unit_id,
                            "action": "ENGAGE",
                            "target_id": target_id,
                            "duration_sec": 1.0,
                            "reason": "cem engage",
                        }
                    )
                else:
                    commands.append(
                        {
                            "unit_id": unit_id,
                            "action": "STOP",
                            "duration_sec": 1.0,
                            "reason": "cem engage blocked by los/range",
                        }
                    )
            elif action_type_id == 3:
                theta = float(plan.theta_radians[0, step_index, unit_index].detach().cpu().item())
                commands.append(
                    {
                        "unit_id": unit_id,
                        "action": "TURN",
                        "theta": round(math.degrees(theta), 2),
                        "duration_sec": 1.0,
                        "reason": "cem turn",
                    }
                )
            else:
                raise ValueError(f"DEVS action vocabulary에 없는 action_type_id: {action_type_id}")
        return commands

    def _record_commands(
        self,
        time_value: float,
        commands: Sequence[Mapping[str, object]],
        *,
        selector: str,
        best_score: float,
        population_mean: float,
    ) -> None:
        """선택된 command를 메모리와 CSV에 기록한다."""
        rows = [self._command_log_row(time_value, command, role="CEM") for command in commands]
        self._append_command_rows(time_value, rows)
        self._blue_command_times.add(float(time_value))
        if self._command_log is not None:
            self._command_log.flush()
        if self._planner_writer is not None:
            self._planner_writer.writerow(
                {
                    "time": float(time_value),
                    "selector": selector,
                    "best_score": float(best_score),
                    "population_mean": float(population_mean),
                    "commands": json.dumps(rows, ensure_ascii=False),
                }
            )
        if self._planner_log is not None:
            self._planner_log.flush()

    def _record_red_command(self, command: Mapping[str, object]) -> None:
        """RulePolicyAtomic이 실제 출력한 RED command를 action 로그에 추가한다."""
        time_value = float(command["time"])
        row = self._command_log_row(time_value, command, role="RED_RULE")
        self._append_command_rows(time_value, [row])

    def _command_log_row(
        self,
        time_value: float,
        command: Mapping[str, object],
        *,
        role: str,
    ) -> dict[str, object]:
        """DEVS command dict를 commands_log row로 변환한다."""
        return {
            "time": float(time_value),
            "unit_id": int(command["unit_id"]),
            "role": role,
            "action": str(command["action"]).upper(),
            "detail": _format_command_detail(command),
            "reason": str(command["reason"]),
        }

    def _append_command_rows(self, time_value: float, rows: Sequence[Mapping[str, object]]) -> None:
        """같은 tick의 BLUE/RED command row를 덮어쓰지 않고 누적한다."""
        time_key = float(time_value)
        stored_rows = self._command_rows_by_time.setdefault(time_key, [])
        existing_units = {int(row["unit_id"]) for row in stored_rows}
        for row in rows:
            unit_id = int(row["unit_id"])
            if unit_id in existing_units:
                raise ValueError(f"time={time_key}에 unit {unit_id} command가 이미 기록됐다")
            stored_row = dict(row)
            stored_rows.append(stored_row)
            existing_units.add(unit_id)
            if self._command_writer is not None:
                self._command_writer.writerow(stored_row)
        if self._command_log is not None:
            self._command_log.flush()


class CEMEpisodeBattleModel(CoupledDEVS):
    """CEM commander를 붙인 v2 5v5 DEVS battle model."""

    def __init__(
        self,
        *,
        obstacles: Sequence[Sequence[float]],
        run_dir: Path,
        seed: int,
        duration_sec: float,
        model: DEVSObjectCentricWorldModel,
        model_config: ObjectSlotModelConfig,
        cem_config: CEMConfig,
        device: torch.device,
        rollout_backend: str = "jepa",
        rollout_replay_per_tick: int = 0,
        policy: PolicyHead | None = None,
        policy_prior_mix: float = 0.25,
        red_target_priority: str = "nearest",
        write_csv_logs: bool = True,
    ):
        super().__init__("CEMEpisodeBattleModel")
        random.seed(seed)
        initial_entities = [
            {"id": unit_id, "type": "soldier", "x": x, "y": y, "heading": heading, "state": "ALIVE"}
            for unit_id, x, y, heading in BLUE_POSITIONS
        ] + [
            {"id": unit_id, "type": "enemy", "x": x, "y": y, "heading": heading, "state": "ALIVE"}
            for unit_id, x, y, heading in RED_POSITIONS
        ]

        self.world = self.addSubModel(
            LosWorldAtomic(
                initial_entities=initial_entities,
                obstacles=obstacles,
                # 실제 episode와 CEM 후보 DEVS rollout의 피해 규칙을 맞춘다.
                expected_damage=True,
            )
        )
        self.write_csv_logs = bool(write_csv_logs)
        self.logger = (
            self.addSubModel(CSVLoggerAtomic(os.path.join(run_dir, "soldier_log.csv")))
            if self.write_csv_logs
            else None
        )
        self.commander = self.addSubModel(
            CEMCommanderAtomic(
                controlled_ids=BLUE_IDS,
                all_unit_ids=ALL_UNIT_IDS,
                obstacles=obstacles,
                duration_sec=duration_sec,
                run_dir=run_dir,
                model=model,
                model_config=model_config,
                cem_config=cem_config,
                device=device,
                rollout_backend=rollout_backend,
                episode_seed=seed,
                rollout_replay_per_tick=rollout_replay_per_tick,
                policy=policy,
                policy_prior_mix=policy_prior_mix,
                red_target_priority=red_target_priority,
                write_csv_logs=self.write_csv_logs,
            )
        )

        for unit_id, x, y, heading in BLUE_POSITIONS:
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=f"Blue_{unit_id}",
                    soldier_id=unit_id,
                    initial_x=x,
                    initial_y=y,
                    initial_heading=heading,
                    max_move_per_step=1.5,
                    fov_deg=120.0,
                    obstacles=obstacles,
                )
            )
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            if self.logger is not None:
                self.connectPorts(soldier.status_out, self.logger.status_in)
            self.connectPorts(soldier.status_out, self.commander.status_in)
            self.connectPorts(self.commander.orders_out[unit_id], soldier.command_in)

        for unit_id, x, y, heading in RED_POSITIONS:
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=f"Red_{unit_id}",
                    soldier_id=unit_id,
                    initial_x=x,
                    initial_y=y,
                    initial_heading=heading,
                    fov_deg=120.0,
                    obstacles=obstacles,
                    turn_to_damage=True,
                )
            )
            brain = self.addSubModel(
                RulePolicyAtomic(
                    name=f"Red_Rule_{unit_id}",
                    policy=UrbanRedPolicy(
                        target_type="soldier",
                        obstacles=obstacles,
                        target_priority=red_target_priority,
                    ),
                    decision_delay=1.0,
                )
            )
            self.connectPorts(soldier.observation_out, brain.observation_in)
            self.connectPorts(brain.command_out, soldier.command_in)
            self.connectPorts(brain.command_out, self.commander.red_command_in)
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(self.world.spawn_out, soldier.spawn_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            if self.logger is not None:
                self.connectPorts(soldier.status_out, self.logger.status_in)
            self.connectPorts(soldier.status_out, self.commander.status_in)


def _write_episode_config(
    *,
    run_dir: Path,
    episode_index: int,
    seed: int,
    duration_sec: float,
    obstacles: Sequence[Sequence[float]],
    rollout_backend: str,
    red_target_priority: str,
    write_csv_logs: bool,
    model_config: ObjectSlotModelConfig,
    cem_config: CEMConfig,
    obstacle_config: Mapping[str, object] | None = None,
    obstacle_config_source: Path | None = None,
) -> None:
    """episode 재현에 필요한 config를 저장한다."""
    config = {
        "architecture": "episodic_cem_object_centric_jepa",
        "episode_index": int(episode_index),
        "seed": int(seed),
        "duration": float(duration_sec),
        "rollout_backend": rollout_backend,
        "red_target_priority": red_target_priority,
        "write_csv_logs": bool(write_csv_logs),
        "obstacles": [list(rect) for rect in obstacles],
        "objective": list(CURRENT_OBJECTIVE),
        "blue_ids": list(BLUE_IDS),
        "red_ids": list(RED_IDS),
        "model_config": {
            key: value for key, value in model_config.__dict__.items()
        },
        "cem_config": {
            key: value for key, value in cem_config.__dict__.items()
        },
    }
    if obstacle_config:
        if obstacle_config_source is not None:
            config["obstacle_config"] = str(obstacle_config_source)
        for key in ("building_polygons", "real_map"):
            if key in obstacle_config:
                config[key] = obstacle_config[key]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_episode(
    *,
    episode_index: int,
    output_root: Path,
    seed: int,
    duration_sec: float,
    model: DEVSObjectCentricWorldModel,
    model_config: ObjectSlotModelConfig,
    cem_config: CEMConfig,
    device: torch.device,
    obstacles: Sequence[Sequence[float]] = DEFAULT_OBSTACLES,
    rollout_backend: str = "jepa",
    rollout_replay_per_tick: int = 0,
    policy: PolicyHead | None = None,
    policy_prior_mix: float = 0.25,
    red_target_priority: str = "nearest",
    write_csv_logs: bool = True,
    obstacle_config: Mapping[str, object] | None = None,
    obstacle_config_source: Path | None = None,
) -> EpisodeResult:
    """DEVS episode를 새로 실행하고 CEM trajectory를 반환한다."""
    run_dir = output_root / f"episode_{episode_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_episode_config(
        run_dir=run_dir,
        episode_index=episode_index,
        seed=seed,
        duration_sec=duration_sec,
        obstacles=obstacles,
        rollout_backend=rollout_backend,
        red_target_priority=red_target_priority,
        write_csv_logs=write_csv_logs,
        model_config=model_config,
        cem_config=cem_config,
        obstacle_config=obstacle_config,
        obstacle_config_source=obstacle_config_source,
    )
    model.eval()
    battle = CEMEpisodeBattleModel(
        obstacles=obstacles,
        run_dir=run_dir,
        seed=seed,
        duration_sec=duration_sec,
        model=model,
        model_config=model_config,
        cem_config=cem_config,
        device=device,
        rollout_backend=rollout_backend,
        rollout_replay_per_tick=rollout_replay_per_tick,
        policy=policy,
        policy_prior_mix=policy_prior_mix,
        red_target_priority=red_target_priority,
        write_csv_logs=write_csv_logs,
    )
    simulator = Simulator(battle)
    simulator.setTerminationTime(duration_sec)
    simulator.simulate()
    latest_rows = battle.commander.latest_rows()
    outcome = determine_outcome(latest_rows)
    trajectory = battle.commander.trajectory(episode_id=run_dir.name, outcome=outcome)
    battle.commander.close()
    return EpisodeResult(
        episode_index=episode_index,
        run_dir=run_dir,
        outcome=outcome,
        trajectory=trajectory,
        final_rows=latest_rows,
        rollout_windows=tuple(battle.commander.rollout_windows),
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="DEVS episode CEM 실행 후 trajectory 단위 JEPA 학습")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--terrain",
        choices=("urban", "open", "mixed"),
        default="urban",
        help="urban은 DEFAULT_OBSTACLES 시가지, open은 장애물 없는 개활지, mixed는 episode마다 두 지형을 랜덤(50/50) 선택",
    )
    parser.add_argument(
        "--obstacle-config",
        type=Path,
        default=None,
        help="config.json의 obstacles를 실제 장애물 지형으로 사용한다. 지정하면 --terrain보다 우선한다.",
    )
    parser.add_argument(
        "--cem-rollout",
        choices=("devs", "jepa"),
        default="devs",
        help="CEM 후보 평가 방식. devs는 실제 DEVS 단기 rollout(정확, 느림), jepa는 월드모델 예측(빠름, 부정확할 수 있음)",
    )
    parser.add_argument(
        "--red-target-priority",
        choices=("nearest", "low_hp", "smart"),
        default="nearest",
        help="RED rule target 선택. nearest는 가장 가까운 BLUE, low_hp는 부상당한 BLUE, smart는 처치 가능성과 방어선을 함께 본다",
    )
    parser.add_argument(
        "--cem-horizon",
        type=int,
        default=None,
        help="CEM planning horizon(초). 생략하면 pred_frames를 쓴다. devs backend에서만 pred_frames와 다르게 줄 수 있다",
    )
    parser.add_argument(
        "--rollout-replay-per-tick",
        type=int,
        default=8,
        help="tick당 학습으로 회수할 DEVS rollout 후보 수 (상위 절반 + 무작위 절반, 0이면 비활성)",
    )
    parser.add_argument(
        "--devs-warmup-episodes",
        type=int,
        default=0,
        help="초기 N개 episode는 --cem-rollout 대신 DEVS rollout으로 CEM 후보를 평가한다",
    )
    parser.add_argument(
        "--devs-warmup-candidates",
        type=int,
        default=None,
        help="DEVS warmup 구간에서만 사용할 CEM 후보 수. 생략하면 --cem-candidates를 그대로 쓴다",
    )
    parser.add_argument(
        "--devs-refresh-period",
        type=int,
        default=0,
        help="warmup 이후 매 N episode마다 DEVS rollout로 CEM을 평가해 JEPA drift를 보정한다 (0이면 비활성)",
    )
    parser.add_argument(
        "--devs-refresh-candidates",
        type=int,
        default=None,
        help="DEVS refresh episode에서만 사용할 CEM 후보 수. 생략하면 warmup 후보 수를 따른다",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        default=None,
        help="JEPA 월드모델 warm-start checkpoint. 생략하면 매번 랜덤 초기화로 시작한다 "
        "(jepa backend로 채점할 계획이면 반드시 이전 run의 checkpoint를 지정해야 한다)",
    )
    parser.add_argument(
        "--world-model-resume-optimizer",
        action="store_true",
        help="world model optimizer state(momentum 등)도 함께 복원한다",
    )
    parser.add_argument(
        "--freeze-world-model",
        action="store_true",
        help="episode 종료 뒤 JEPA 월드모델 추가 학습을 건너뛰고 policy head만 학습한다",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=None,
        help="학습된 policy head checkpoint. 지정하면 CEM 초기 분포를 policy 제안으로 세운다",
    )
    parser.add_argument(
        "--policy-epochs-per-episode",
        type=int,
        default=None,
        help="episode 종료 뒤 policy head를 학습할 epoch 수. 생략하면 policy checkpoint 사용 시 1, 미사용 시 0",
    )
    parser.add_argument(
        "--policy-warmup-episodes",
        type=int,
        default=10,
        help="초기 N개 episode 동안 policy 제안/학습을 끄고 월드모델만 학습한다",
    )
    parser.add_argument("--policy-batch-size", type=int, default=64)
    parser.add_argument("--policy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--policy-prior-mix-start",
        type=float,
        default=0.75,
        help="policy 사용 초기 CEM 기본 prior 혼합 비율",
    )
    parser.add_argument(
        "--policy-prior-mix-end",
        type=float,
        default=0.25,
        help="prior mix schedule 종료 비율",
    )
    parser.add_argument(
        "--policy-prior-mix-decay-episodes",
        type=int,
        default=200,
        help="warmup 이후 prior mix를 start에서 end까지 줄이는 episode 수",
    )
    parser.add_argument(
        "--policy-entropy-weight",
        type=float,
        default=0.05,
        help="policy warmup 이후 시작 entropy 보너스 가중치",
    )
    parser.add_argument(
        "--policy-entropy-weight-end",
        type=float,
        default=0.01,
        help="entropy schedule 종료 가중치",
    )
    parser.add_argument(
        "--policy-entropy-decay-episodes",
        type=int,
        default=200,
        help="warmup 이후 entropy weight를 시작값에서 종료값까지 줄이는 episode 수",
    )
    parser.add_argument(
        "--policy-output-path",
        type=Path,
        default=None,
        help="온라인 학습된 policy head 저장 경로. 생략하면 world model checkpoint 이름에서 파생한다",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("output/episodic_cem_jepa"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--pred-frames", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-encoder-layers", type=int, default=0)
    parser.add_argument("--num-predictor-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-masked-slots", type=int, default=2)
    parser.add_argument("--self-state-dim", type=int, default=32)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--mask-team-strategy", choices=("random_team", "blue", "red", "all"), default="random_team")
    parser.add_argument("--mask-count-min", type=int, default=1)
    parser.add_argument("--mask-count-max", type=int, default=5)
    parser.add_argument("--blue-team-mask-probability", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--blue-team-mask-count-min", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--blue-team-mask-count-max", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--slot-self-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--cem-candidates", type=int, default=64)
    parser.add_argument("--cem-elites", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--train-epochs-per-episode", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/episodic_cem_jepa.pt"))
    parser.add_argument(
        "--best-checkpoint-path",
        type=Path,
        default=None,
        help="validation monitor가 개선될 때 저장할 best checkpoint. 생략하면 checkpoint 이름에 _best를 붙인다",
    )
    parser.add_argument(
        "--validation-run-dir",
        type=Path,
        action="append",
        default=[],
        help="학습에 사용하지 않는 고정 DEVS validation episode 디렉터리. 여러 번 지정 가능",
    )
    parser.add_argument("--validation-every", type=int, default=5, help="몇 episode마다 validation할지")
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-seed", type=int, default=1234, help="validation masking 고정 seed")
    parser.add_argument(
        "--validation-max-windows",
        type=int,
        default=512,
        help="validation에 사용할 최대 window 수. 0이면 전부 사용",
    )
    parser.add_argument(
        "--validation-monitor",
        choices=(
            "loss_total",
            "loss_latent",
            "loss_future",
            "loss_masked_history",
            "loss_future_state",
            "loss_masked_history_state",
            "loss_combat_state",
            "loss_damage_delta",
            "loss_slot_self_state",
        ),
        default="loss_total",
        help="best checkpoint와 early stopping에 사용할 validation metric",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="validation monitor가 개선되지 않아도 허용할 validation check 수. 0이면 조기 종료 비활성",
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stop-min-episodes",
        type=int,
        default=50,
        help="이 episode 수 전에는 patience를 채워도 조기 종료하지 않는다",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--disable-csv-logs",
        action="store_true",
        help="학습 중 soldier/commands/planner tick CSV를 쓰지 않고 episode_summary.jsonl만 남긴다",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """episode 실행과 종료 후 학습을 반복한다."""
    args = _parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes는 0보다 커야 한다")
    if args.duration <= 0.0:
        raise ValueError("duration은 0보다 커야 한다")
    if args.freeze_world_model and args.world_model_checkpoint is None:
        raise ValueError("--freeze-world-model은 --world-model-checkpoint와 함께 써야 한다")
    if args.validation_every <= 0:
        raise ValueError("--validation-every는 0보다 커야 한다")
    if args.validation_batch_size <= 0:
        raise ValueError("--validation-batch-size는 0보다 커야 한다")
    if args.validation_max_windows < 0:
        raise ValueError("--validation-max-windows는 음수일 수 없다")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience는 음수일 수 없다")
    if args.early_stop_min_delta < 0.0:
        raise ValueError("--early-stop-min-delta는 음수일 수 없다")
    if args.early_stop_min_episodes < 0:
        raise ValueError("--early-stop-min-episodes는 음수일 수 없다")
    if args.validation_run_dir and args.freeze_world_model:
        raise ValueError("고정 월드모델에서는 validation 기반 world-model early stopping을 사용할 수 없다")
    for validation_dir in args.validation_run_dir:
        if not validation_dir.exists():
            raise FileNotFoundError(f"validation run dir가 없다: {validation_dir}")
    obstacle_config = None
    configured_obstacles = None
    if args.obstacle_config is not None:
        if not args.obstacle_config.exists():
            raise FileNotFoundError(f"obstacle config가 없다: {args.obstacle_config}")
        obstacle_config, configured_obstacles = _load_obstacle_config(args.obstacle_config)

    device = torch.device(args.device)
    model_config = ObjectSlotModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_predictor_layers=args.num_predictor_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        history_frames=args.history_frames,
        pred_frames=args.pred_frames,
        num_masked_slots=args.num_masked_slots,
        self_state_dim=args.self_state_dim,
        mask_seed=args.mask_seed,
        mask_team_strategy=args.mask_team_strategy,
        mask_count_min=args.mask_count_min,
        mask_count_max=args.mask_count_max,
        blue_team_mask_probability=args.blue_team_mask_probability,
        blue_team_mask_count_min=args.blue_team_mask_count_min,
        blue_team_mask_count_max=args.blue_team_mask_count_max,
        ema_momentum=args.ema_momentum,
    )
    if args.cem_horizon is not None and args.cem_rollout == "jepa" and args.cem_horizon != args.pred_frames:
        raise ValueError("jepa backend에서는 --cem-horizon이 --pred-frames와 같아야 한다")
    cem_config = CEMConfig(
        num_candidates=args.cem_candidates,
        num_elites=args.cem_elites,
        num_iterations=args.cem_iterations,
        future_horizon=args.pred_frames if args.cem_horizon is None else args.cem_horizon,
        seed=args.seed,
        min_action_probability=0.0,
    )
    if args.devs_warmup_episodes < 0:
        raise ValueError("--devs-warmup-episodes는 음수일 수 없다")
    if args.devs_warmup_candidates is not None and args.devs_warmup_candidates <= 0:
        raise ValueError("--devs-warmup-candidates는 0보다 커야 한다")
    if args.devs_refresh_period < 0:
        raise ValueError("--devs-refresh-period는 음수일 수 없다")
    if args.devs_refresh_candidates is not None and args.devs_refresh_candidates <= 0:
        raise ValueError("--devs-refresh-candidates는 0보다 커야 한다")
    devs_warmup_cem_config = cem_config
    if args.devs_warmup_candidates is not None:
        # DEVS rollout은 정확하지만 비싸므로 warmup 구간 후보 수만 별도로 줄일 수 있다.
        devs_warmup_cem_config = replace(
            cem_config,
            num_candidates=args.devs_warmup_candidates,
            num_elites=min(cem_config.num_elites, args.devs_warmup_candidates),
        )
    devs_refresh_cem_config = devs_warmup_cem_config
    if args.devs_refresh_candidates is not None:
        # hard switch 이후에도 주기적으로 정확한 DEVS teacher를 넣어 JEPA 고착을 풀어준다.
        devs_refresh_cem_config = replace(
            cem_config,
            num_candidates=args.devs_refresh_candidates,
            num_elites=min(cem_config.num_elites, args.devs_refresh_candidates),
        )
    if args.devs_warmup_episodes > 0:
        print(
            "DEVS rollout warmup 활성: "
            f"episodes=1..{args.devs_warmup_episodes} "
            f"candidates={devs_warmup_cem_config.num_candidates} "
            f"elites={devs_warmup_cem_config.num_elites} "
            f"then={args.cem_rollout}"
        )
    if args.devs_refresh_period > 0:
        print(
            "DEVS rollout refresh 활성: "
            f"period={args.devs_refresh_period} "
            f"candidates={devs_refresh_cem_config.num_candidates} "
            f"elites={devs_refresh_cem_config.num_elites}"
        )
    optimizer_config = OptimizerConfig(
        epochs=args.train_epochs_per_episode,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint_path,
    )
    loss_weights = LossWeights(slot_self_state=args.slot_self_state_loss_weight)
    model, optimizer = create_model_and_optimizer(
        model_config=model_config,
        optimizer_config=optimizer_config,
    )

    resumed_completed_epochs = 0
    resumed_global_step = 0
    resumed_best_validation_value = math.inf
    resumed_bad_validation_checks = 0
    resumed_best_episode = 0
    if args.world_model_checkpoint is not None:
        # create_model_and_optimizer는 항상 랜덤 초기화 모델을 만든다. jepa
        # backend로 채점할 때 이전 run에서 학습된 표현을 이어받으려면 여기서
        # state_dict를 덮어써야 한다. 안 하면 학습 안 된 모델로 채점을 시작한다.
        payload = torch.load(args.world_model_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if args.world_model_resume_optimizer:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            resumed_completed_epochs = int(payload.get("epoch", 0))
            resumed_global_step = int(payload.get("global_step", 0))
            early_state = payload.get("early_stopping") or {}
            stored_monitor = early_state.get("monitor")
            if stored_monitor in (None, args.validation_monitor):
                stored_best = early_state.get("best_validation_value")
                if stored_best is not None:
                    resumed_best_validation_value = float(stored_best)
                resumed_bad_validation_checks = int(early_state.get("bad_validation_checks", 0))
                resumed_best_episode = int(early_state.get("best_episode") or 0)
            print(
                "world model optimizer resume: "
                f"epoch={resumed_completed_epochs} global_step={resumed_global_step} "
                f"best_val={resumed_best_validation_value}"
            )
        print(f"world model warm-start: {args.world_model_checkpoint}")
    if args.freeze_world_model:
        print("world model freeze 활성: episode 종료 후 JEPA 추가 학습을 건너뛰고 policy만 학습한다")

    best_checkpoint_path = (
        args.best_checkpoint_path
        if args.best_checkpoint_path is not None
        else args.checkpoint_path.with_name(f"{args.checkpoint_path.stem}_best{args.checkpoint_path.suffix}")
    )
    validation_windows: tuple[LoadedTrainingWindow, ...] = ()
    if args.validation_run_dir:
        if best_checkpoint_path.resolve() == args.checkpoint_path.resolve():
            raise ValueError("--best-checkpoint-path는 --checkpoint-path와 달라야 한다")
        validation_specs = build_training_windows(
            tuple(args.validation_run_dir),
            history_frames=model_config.history_frames,
            pred_frames=model_config.pred_frames,
            time_step=1.0,
        )
        if args.validation_max_windows > 0 and len(validation_specs) > args.validation_max_windows:
            validation_rng = np.random.default_rng(args.validation_seed)
            selected_indices = sorted(
                validation_rng.choice(
                    len(validation_specs),
                    size=args.validation_max_windows,
                    replace=False,
                ).tolist()
            )
            validation_specs = tuple(validation_specs[index] for index in selected_indices)
        validation_windows = load_training_windows(validation_specs)
        print(
            "validation 활성: "
            f"runs={len(args.validation_run_dir)} windows={len(validation_windows)} "
            f"every={args.validation_every} monitor={args.validation_monitor} "
            f"best={best_checkpoint_path} patience={args.early_stop_patience}"
        )
    else:
        print("validation 비활성: --validation-run-dir를 지정하면 best checkpoint와 early stopping이 켜진다")

    policy_epochs_per_episode = (
        (1 if args.policy_checkpoint is not None else 0)
        if args.policy_epochs_per_episode is None
        else args.policy_epochs_per_episode
    )
    if policy_epochs_per_episode < 0:
        raise ValueError("--policy-epochs-per-episode는 음수일 수 없다")
    if args.policy_warmup_episodes < 0:
        raise ValueError("--policy-warmup-episodes는 음수일 수 없다")
    if args.policy_batch_size <= 0:
        raise ValueError("--policy-batch-size는 0보다 커야 한다")
    if args.policy_learning_rate <= 0.0:
        raise ValueError("--policy-learning-rate는 0보다 커야 한다")
    if args.policy_weight_decay < 0.0:
        raise ValueError("--policy-weight-decay는 음수일 수 없다")
    if args.policy_gradient_clip_norm <= 0.0:
        raise ValueError("--policy-gradient-clip-norm은 0보다 커야 한다")
    if not 0.0 <= args.policy_prior_mix_start <= 1.0:
        raise ValueError("--policy-prior-mix-start는 [0, 1] 범위여야 한다")
    if not 0.0 <= args.policy_prior_mix_end <= 1.0:
        raise ValueError("--policy-prior-mix-end는 [0, 1] 범위여야 한다")
    if args.policy_prior_mix_start < args.policy_prior_mix_end:
        raise ValueError("--policy-prior-mix-start는 end보다 작을 수 없다")
    if args.policy_prior_mix_decay_episodes < 0:
        raise ValueError("--policy-prior-mix-decay-episodes는 음수일 수 없다")
    if args.policy_entropy_weight < 0.0:
        raise ValueError("--policy-entropy-weight는 음수일 수 없다")
    if args.policy_entropy_weight_end < 0.0:
        raise ValueError("--policy-entropy-weight-end는 음수일 수 없다")
    if args.policy_entropy_weight < args.policy_entropy_weight_end:
        raise ValueError("--policy-entropy-weight는 end보다 작을 수 없다")
    if args.policy_entropy_decay_episodes < 0:
        raise ValueError("--policy-entropy-decay-episodes는 음수일 수 없다")

    policy = None
    policy_optimizer = None
    policy_ready = False
    policy_output_path = args.policy_output_path
    if args.policy_checkpoint is not None:
        policy = load_policy(args.policy_checkpoint, device)
        policy_ready = True
        print(f"policy proposer 활성: {args.policy_checkpoint}")
    elif policy_epochs_per_episode > 0:
        torch.manual_seed(args.seed)
        policy = PolicyHead(PolicyHeadConfig()).to(device)
        print("policy head 온라인 학습 활성: 첫 episode는 CEM 기본 분포로 수집한다")
    if policy is not None and policy_epochs_per_episode > 0:
        policy_optimizer = torch.optim.AdamW(
            tuple(parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=args.policy_learning_rate,
            weight_decay=args.policy_weight_decay,
        )
        if policy_output_path is None:
            policy_output_path = args.checkpoint_path.with_name(f"{args.checkpoint_path.stem}_policy.pt")
        print(
            f"policy 온라인 학습 활성: epochs_per_episode={policy_epochs_per_episode} "
            f"save={policy_output_path}"
        )
        print(
            "policy schedule: "
            f"prior_mix {args.policy_prior_mix_start:.3f}->{args.policy_prior_mix_end:.3f} "
            f"/ {args.policy_prior_mix_decay_episodes}ep, "
            f"entropy {args.policy_entropy_weight:.4f}->{args.policy_entropy_weight_end:.4f} "
            f"/ {args.policy_entropy_decay_episodes}ep"
        )
    if args.policy_warmup_episodes > 0 and policy is not None:
        print(
            f"policy warmup 활성: 처음 {args.policy_warmup_episodes}개 episode는 "
            "policy 제안과 policy 학습을 모두 건너뛴다"
        )

    global_step = resumed_global_step
    completed_epochs = resumed_completed_epochs
    best_validation_value = resumed_best_validation_value
    bad_validation_checks = resumed_bad_validation_checks
    best_episode = resumed_best_episode
    early_stopped = False
    completed_episode_count = 0
    started = time.time()
    terrain_rng = random.Random(args.seed + 777)
    for episode_index in range(1, args.episodes + 1):
        completed_episode_count = episode_index
        if configured_obstacles is not None:
            episode_terrain = "obstacle_config"
            episode_obstacles = configured_obstacles
        elif args.terrain == "mixed":
            episode_terrain = "open" if terrain_rng.random() < 0.5 else "urban"
            episode_obstacles = () if episode_terrain == "open" else DEFAULT_OBSTACLES
        else:
            episode_terrain = args.terrain
            episode_obstacles = () if episode_terrain == "open" else DEFAULT_OBSTACLES
        in_devs_warmup = episode_index <= args.devs_warmup_episodes
        refresh_offset = episode_index - args.devs_warmup_episodes
        in_devs_refresh = (
            not in_devs_warmup
            and args.devs_refresh_period > 0
            and refresh_offset > 0
            and refresh_offset % args.devs_refresh_period == 0
        )
        # 초반과 refresh episode는 정확한 DEVS rollout으로 후보를 평가해 JEPA rollout의 고착을 풀어준다.
        effective_rollout_backend = "devs" if in_devs_warmup or in_devs_refresh else args.cem_rollout
        effective_cem_config = (
            devs_warmup_cem_config
            if in_devs_warmup
            else devs_refresh_cem_config
            if in_devs_refresh
            else cem_config
        )
        rollout_phase = (
            "devs_warmup"
            if in_devs_warmup
            else "devs_refresh"
            if in_devs_refresh
            else args.cem_rollout
        )
        policy_prior_mix = _scheduled_policy_prior_mix(
            episode_index=episode_index,
            warmup_episodes=args.policy_warmup_episodes,
            start=args.policy_prior_mix_start,
            end=args.policy_prior_mix_end,
            decay_episodes=args.policy_prior_mix_decay_episodes,
        )
        policy_entropy_weight = _scheduled_policy_entropy_weight(
            episode_index=episode_index,
            warmup_episodes=args.policy_warmup_episodes,
            start=args.policy_entropy_weight,
            end=args.policy_entropy_weight_end,
            decay_episodes=args.policy_entropy_decay_episodes,
        )
        # warmup 동안은 월드모델만 학습한다. policy checkpoint가 있어도 이 구간에서는
        # CEM 초기 분포에 개입시키지 않는다.
        policy_active = policy_ready and episode_index > args.policy_warmup_episodes
        result = run_episode(
            episode_index=episode_index,
            output_root=args.output_root,
            seed=args.seed + episode_index - 1,
            duration_sec=args.duration,
            model=model,
            model_config=model_config,
            cem_config=effective_cem_config,
            device=device,
            obstacles=episode_obstacles,
            rollout_backend=effective_rollout_backend,
            rollout_replay_per_tick=args.rollout_replay_per_tick,
            policy=policy if policy_active else None,
            policy_prior_mix=policy_prior_mix,
            red_target_priority=args.red_target_priority,
            write_csv_logs=not args.disable_csv_logs,
            obstacle_config=obstacle_config,
            obstacle_config_source=args.obstacle_config,
        )
        windows = build_training_windows_from_episode(
            result.trajectory,
            history_frames=model_config.history_frames,
            pred_frames=model_config.pred_frames,
            time_step=1.0,
        )
        policy_schedule_text = (
            f" policy_prior_mix={policy_prior_mix:.3f} policy_entropy={policy_entropy_weight:.4f}"
            if policy is not None and episode_index > args.policy_warmup_episodes
            else ""
        )
        rollout_text = (
            f" rollout={effective_rollout_backend} phase={rollout_phase} "
            f"candidates={effective_cem_config.num_candidates}"
        )
        print(
            f"episode={episode_index} terrain={episode_terrain} outcome={result.outcome} "
            f"states={len(result.trajectory.states)} actions={len(result.trajectory.actions)} "
            f"windows={len(windows)} rollout_windows={len(result.rollout_windows)} "
            f"run_dir={result.run_dir}{rollout_text}{policy_schedule_text}"
        )
        _append_episode_summary(args.output_root, result)
        train_metrics = ()
        if args.freeze_world_model:
            # policy head는 CEM이 실제 실행한 trajectory로 학습하지만, 이미 검증된
            # JEPA rollout 평가기는 고정해 policy 변화만 관찰한다.
            model.eval()
        else:
            train_metrics, global_step = train_existing_model(
                model=model,
                optimizer=optimizer,
                windows=windows + result.rollout_windows,
                model_config=model_config,
                optimizer_config=optimizer_config,
                loss_weights=loss_weights,
                start_epoch=completed_epochs,
                global_step=global_step,
            )
            completed_epochs += optimizer_config.epochs

            if validation_windows and episode_index % args.validation_every == 0:
                validation_metrics = evaluate_existing_model(
                    model=model,
                    windows=validation_windows,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    loss_weights=loss_weights,
                    batch_size=args.validation_batch_size,
                    validation_seed=args.validation_seed,
                )
                validation_value = float(getattr(validation_metrics, args.validation_monitor))
                improved = validation_value < best_validation_value - args.early_stop_min_delta
                if improved:
                    best_validation_value = validation_value
                    bad_validation_checks = 0
                    best_episode = episode_index
                    save_checkpoint(
                        path=best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        loss_weights=loss_weights,
                        epoch=completed_epochs,
                        global_step=global_step,
                        metrics=train_metrics,
                        episode=episode_index,
                        checkpoint_kind="best",
                        validation_metrics=validation_metrics,
                        validation_monitor=args.validation_monitor,
                        validation_value=validation_value,
                        best_validation_value=best_validation_value,
                        best_episode=best_episode,
                        bad_validation_checks=bad_validation_checks,
                    )
                else:
                    bad_validation_checks += 1

                # train_existing_model이 저장한 latest checkpoint를 validation/early
                # stopping 상태까지 포함한 payload로 다시 갱신한다.
                save_checkpoint(
                    path=args.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    loss_weights=loss_weights,
                    epoch=completed_epochs,
                    global_step=global_step,
                    metrics=train_metrics,
                    episode=episode_index,
                    checkpoint_kind="latest",
                    validation_metrics=validation_metrics,
                    validation_monitor=args.validation_monitor,
                    validation_value=validation_value,
                    best_validation_value=best_validation_value,
                    best_episode=best_episode,
                    bad_validation_checks=bad_validation_checks,
                )
                validation_row = {
                    "episode": episode_index,
                    "epoch": completed_epochs,
                    "global_step": global_step,
                    "monitor": args.validation_monitor,
                    "monitor_value": validation_value,
                    "improved": improved,
                    "best_value": best_validation_value,
                    "best_episode": best_episode,
                    "bad_validation_checks": bad_validation_checks,
                    **vars(validation_metrics),
                }
                _append_validation_summary(args.output_root, validation_row)
                print(
                    "validation_summary="
                    f"episode={episode_index} {args.validation_monitor}={validation_value:.6f} "
                    f"total={validation_metrics.loss_total:.6f} "
                    f"future_state={validation_metrics.loss_future_state:.6f} "
                    f"masked_state={validation_metrics.loss_masked_history_state:.6f} "
                    f"combat={validation_metrics.loss_combat_state:.6f} "
                    f"best={best_validation_value:.6f} improved={improved} "
                    f"bad_checks={bad_validation_checks}/{args.early_stop_patience}"
                )
                if (
                    args.early_stop_patience > 0
                    and episode_index >= args.early_stop_min_episodes
                    and bad_validation_checks >= args.early_stop_patience
                ):
                    early_stopped = True
                    print(
                        "early_stopping="
                        f"episode={episode_index} best_episode={best_episode} "
                        f"best_{args.validation_monitor}={best_validation_value:.6f}"
                    )
                    break
            elif validation_windows:
                # validation 사이 episode에서도 resume용 latest checkpoint에
                # 누적 best/patience 상태를 보존한다.
                save_checkpoint(
                    path=args.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    loss_weights=loss_weights,
                    epoch=completed_epochs,
                    global_step=global_step,
                    metrics=train_metrics,
                    episode=episode_index,
                    checkpoint_kind="latest",
                    validation_monitor=args.validation_monitor,
                    best_validation_value=(
                        None if math.isinf(best_validation_value) else best_validation_value
                    ),
                    best_episode=best_episode if best_episode > 0 else None,
                    bad_validation_checks=bad_validation_checks,
                )
        policy_train_enabled = episode_index > args.policy_warmup_episodes
        if policy is not None and policy_optimizer is not None and policy_epochs_per_episode > 0 and policy_train_enabled:
            _train_policy_on_episode(
                policy=policy,
                optimizer=policy_optimizer,
                trajectory=result.trajectory,
                device=device,
                epochs=policy_epochs_per_episode,
                batch_size=args.policy_batch_size,
                gradient_clip_norm=args.policy_gradient_clip_norm,
                entropy_weight=policy_entropy_weight,
                seed=args.seed,
                episode_index=episode_index,
            )
            policy_ready = True
            save_policy(policy_output_path, policy)
    elapsed = time.time() - started
    print(
        f"done episodes={completed_episode_count}/{args.episodes} global_step={global_step} "
        f"early_stopped={early_stopped} elapsed_s={elapsed:.1f}"
    )
    print(f"episode_summary={args.output_root / 'episode_summary.jsonl'}")
    print(f"checkpoint_latest={args.checkpoint_path}")
    if validation_windows:
        print(f"validation_summary={args.output_root / 'validation_summary.jsonl'}")
        print(f"checkpoint_best={best_checkpoint_path}")
        print(
            f"best_episode={best_episode} best_{args.validation_monitor}="
            f"{best_validation_value:.6f}"
        )
    if policy_output_path is not None:
        print(f"policy_checkpoint={policy_output_path}")


if __name__ == "__main__":
    main()
