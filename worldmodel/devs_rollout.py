"""CEM 후보 joint action sequence를 실제 DEVS로 짧게 rollout해 평가한다.

이 모듈은 학습 모델을 쓰지 않는다. 현재 episode의 state 스냅샷에서 후보마다
독립적인 단기 DEVS 시뮬레이션을 만들어 실행하고, 실제 미래 state feature를
반환한다. 후보 간 비교 분산을 줄이기 위해 모든 후보가 같은 random seed로
시작한다(common random numbers).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR.parent
for path in (str(PROJECT_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hackerthon.combat_config import MAX_FIRE_RANGE  # noqa: E402
from pypdevs.DEVS import AtomicDEVS, CoupledDEVS  # noqa: E402
from pypdevs.infinity import INFINITY  # noqa: E402
from pypdevs.simulator import Simulator  # noqa: E402

from hackerthon.red_policy import UrbanRedPolicy  # noqa: E402
from hackerthon.simulation_direct_commander_5v5 import RulePolicyAtomic  # noqa: E402
from hackerthon.sim_units import LosSoldierAtomic, LosWorldAtomic  # noqa: E402
from hackerthon.terrain import clamp_to_world, has_los, next_waypoint  # noqa: E402

from hackerthon.worldmodel.cem_planner import FutureActionPlanBatch  # noqa: E402
from hackerthon.worldmodel.slots import MAX_FEATURE_DIM, SlotBatch, build_slot_batch  # noqa: E402
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN  # noqa: E402


BLUE_ID_MAX = 199


@dataclass(frozen=True)
class RolloutSnapshot:
    """rollout 시작점이 되는 episode 중간 state."""

    unit_rows: tuple[dict[str, object], ...]
    obstacles: tuple[tuple[float, float, float, float], ...]
    base_time_sec: float
    episode_duration_sec: float

    def __post_init__(self) -> None:
        """스냅샷 계약을 즉시 검증한다."""
        if not self.unit_rows:
            raise ValueError("unit_rows가 비어 있다")
        if self.episode_duration_sec <= 0.0:
            raise ValueError("episode_duration_sec는 0보다 커야 한다")
        ids = [int(row["id"]) for row in self.unit_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("unit_rows에 중복 id가 있다")


def _denorm_x(x_norm: float) -> float:
    """[-1, 1] x좌표를 DEVS 월드 좌표로 되돌린다."""
    return (float(x_norm) + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def _denorm_y(y_norm: float) -> float:
    """[-1, 1] y좌표를 DEVS 월드 좌표로 되돌린다."""
    return (float(y_norm) + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def _status_to_row(status: Mapping[str, object]) -> dict[str, object]:
    """Soldier status 메시지를 slot builder가 읽는 row 형식으로 바꾼다."""
    return {
        "time": float(status["time"]),
        "id": int(status["id"]),
        "x": float(status["x"]),
        "y": float(status["y"]),
        "heading": float(status["heading"]),
        "hp": float(status["hp"]),
        "ammo": int(status["ammo"]),
    }


@dataclass(frozen=True)
class _CandidatePlan:
    """한 후보의 tick별 액션 (CPU 값)."""

    action_type_ids: np.ndarray  # (H, U)
    move_xy_norm: np.ndarray  # (H, U, 2)
    target_entity_ids: np.ndarray  # (H, U)
    theta_radians: np.ndarray  # (H, U)
    issued_mask: np.ndarray  # (H, U)
    unit_ids: np.ndarray  # (U,)


class ScriptedPlanCommanderAtomic(AtomicDEVS):
    """미리 정해진 CEM 후보 plan을 tick 순서대로 실행하는 commander.

    실제 CEMCommanderAtomic과 같은 message 계약을 쓰되, 탐색 없이 주어진
    plan의 step k 액션을 내부 시각 k에 그대로 내보낸다. MOVE waypoint와
    ENGAGE 가능 판정은 실제 commander와 같은 규칙으로 현재 위치 기준
    매 tick 다시 계산한다.
    """

    def __init__(
        self,
        *,
        plan: _CandidatePlan,
        simulated_ids: Sequence[int],
        obstacles: Sequence[Sequence[float]],
        horizon: int,
    ):
        super().__init__("ScriptedPlanCommander")
        self.plan = plan
        self.simulated_ids = tuple(int(unit_id) for unit_id in simulated_ids)
        self.obstacles = [tuple(rect) for rect in obstacles]
        self.horizon = int(horizon)
        self.status_in = self.addInPort("status_in")
        self.orders_out = {
            int(unit_id): self.addOutPort(f"orders_out_{int(unit_id)}")
            for unit_id in plan.unit_ids.tolist()
        }
        self.sigma = INFINITY
        self.state = "WAIT"
        self.frames: dict[float, dict[int, dict[str, object]]] = {}
        self._commanded_times: set[float] = set()

    def timeAdvance(self):
        return self.sigma

    def extTransition(self, inputs):
        if self.status_in in inputs:
            for status in inputs[self.status_in]:
                row = _status_to_row(status)
                self.frames.setdefault(float(row["time"]), {})[int(row["id"])] = row
        latest = self._latest_complete_time()
        if latest is not None and latest not in self._commanded_times and int(round(latest)) < self.horizon:
            self.sigma = 0.0
            self.state = "READY"
        return self.state

    def intTransition(self):
        self.sigma = INFINITY
        self.state = "WAIT"
        return self.state

    def outputFnc(self):
        latest = self._latest_complete_time()
        if latest is None or latest in self._commanded_times:
            return {}
        step_index = int(round(latest))
        if step_index >= self.horizon:
            return {}
        self._commanded_times.add(latest)
        row_by_id = self.frames[latest]
        out = {}
        for unit_index, unit_id in enumerate(self.plan.unit_ids.tolist()):
            unit_id = int(unit_id)
            if unit_id not in row_by_id:
                continue
            if not bool(self.plan.issued_mask[step_index, unit_index]):
                continue
            if float(row_by_id[unit_id]["hp"]) <= 0.0:
                continue
            command = self._command_for(step_index, unit_index, unit_id, row_by_id)
            port = self.orders_out.get(unit_id)
            if port is not None and command is not None:
                out[port] = [command]
        return out

    def _latest_complete_time(self) -> float | None:
        """시뮬레이션 중인 모든 유닛의 status가 모인 최신 시각을 찾는다."""
        complete = [
            time_value
            for time_value, rows in self.frames.items()
            if all(unit_id in rows for unit_id in self.simulated_ids)
        ]
        return max(complete) if complete else None

    def _command_for(
        self,
        step_index: int,
        unit_index: int,
        unit_id: int,
        row_by_id: Mapping[int, Mapping[str, object]],
    ) -> dict[str, object] | None:
        """plan의 한 액션을 실제 commander와 같은 규칙으로 command dict로 바꾼다."""
        action_type_id = int(self.plan.action_type_ids[step_index, unit_index])
        if action_type_id == 0:
            return {"unit_id": unit_id, "action": "STOP", "duration_sec": 1.0, "reason": "rollout stop"}
        if action_type_id == 1:
            target_x = _denorm_x(self.plan.move_xy_norm[step_index, unit_index, 0])
            target_y = _denorm_y(self.plan.move_xy_norm[step_index, unit_index, 1])
            target_x, target_y = clamp_to_world(target_x, target_y)
            current_pos = (float(row_by_id[unit_id]["x"]), float(row_by_id[unit_id]["y"]))
            waypoint = next_waypoint(current_pos, (target_x, target_y), self.obstacles, max_step=1.5)
            if waypoint is None:
                return {"unit_id": unit_id, "action": "STOP", "duration_sec": 1.0, "reason": "rollout move blocked"}
            return {
                "unit_id": unit_id,
                "action": "MOVE",
                "x": round(float(waypoint[0]), 3),
                "y": round(float(waypoint[1]), 3),
                "duration_sec": 1.0,
                "reason": "rollout move",
            }
        if action_type_id == 2:
            target_id = int(self.plan.target_entity_ids[step_index, unit_index])
            if self._engage_allowed(shooter_id=unit_id, target_id=target_id, row_by_id=row_by_id):
                return {
                    "unit_id": unit_id,
                    "action": "ENGAGE",
                    "target_id": target_id,
                    "duration_sec": 1.0,
                    "reason": "rollout engage",
                }
            return {"unit_id": unit_id, "action": "STOP", "duration_sec": 1.0, "reason": "rollout engage blocked"}
        if action_type_id == 3:
            theta = float(self.plan.theta_radians[step_index, unit_index])
            return {
                "unit_id": unit_id,
                "action": "TURN",
                "theta": round(math.degrees(theta), 2),
                "duration_sec": 1.0,
                "reason": "rollout turn",
            }
        raise ValueError(f"DEVS action vocabulary에 없는 action_type_id: {action_type_id}")

    def _engage_allowed(
        self,
        *,
        shooter_id: int,
        target_id: int,
        row_by_id: Mapping[int, Mapping[str, object]],
    ) -> bool:
        """실제 commander와 같은 hp/탄약/사거리/LOS 판정."""
        shooter = row_by_id.get(shooter_id)
        target = row_by_id.get(target_id)
        if shooter is None or target is None:
            return False
        if float(shooter["hp"]) <= 0.0 or int(shooter["ammo"]) <= 0:
            return False
        if float(target["hp"]) <= 0.0:
            return False
        shooter_pos = (float(shooter["x"]), float(shooter["y"]))
        target_pos = (float(target["x"]), float(target["y"]))
        distance = math.hypot(shooter_pos[0] - target_pos[0], shooter_pos[1] - target_pos[1])
        if distance > MAX_FIRE_RANGE:
            return False
        return has_los(shooter_pos, target_pos, self.obstacles)


class _RolloutBattleModel(CoupledDEVS):
    """스냅샷에서 시작하는 단기 rollout용 DEVS battle model."""

    def __init__(
        self,
        *,
        snapshot: RolloutSnapshot,
        plan: _CandidatePlan,
        horizon: int,
        red_target_priority: str = "nearest",
    ):
        super().__init__("RolloutBattleModel")
        alive_rows = [row for row in snapshot.unit_rows if float(row["hp"]) > 0.0]
        initial_entities = [
            {
                "id": int(row["id"]),
                "type": "soldier" if int(row["id"]) <= BLUE_ID_MAX else "enemy",
                "x": float(row["x"]),
                "y": float(row["y"]),
                "heading": float(row["heading"]),
                "hp": int(round(float(row["hp"]))),
                "ammo": int(row["ammo"]),
                "state": "ALIVE",
            }
            for row in alive_rows
        ]
        self.world = self.addSubModel(
            LosWorldAtomic(
                initial_entities=initial_entities,
                obstacles=snapshot.obstacles,
                expected_damage=True,
            )
        )
        simulated_ids = [int(row["id"]) for row in alive_rows]
        self.commander = self.addSubModel(
            ScriptedPlanCommanderAtomic(
                plan=plan,
                simulated_ids=simulated_ids,
                obstacles=snapshot.obstacles,
                horizon=horizon,
            )
        )

        for row in alive_rows:
            unit_id = int(row["id"])
            is_blue = unit_id <= BLUE_ID_MAX
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=("Blue_" if is_blue else "Red_") + str(unit_id),
                    soldier_id=unit_id,
                    initial_x=float(row["x"]),
                    initial_y=float(row["y"]),
                    initial_heading=float(row["heading"]),
                    hp=int(round(float(row["hp"]))),
                    ammo=int(row["ammo"]),
                    fov_deg=120.0,
                    obstacles=snapshot.obstacles,
                    **({"max_move_per_step": 1.5} if is_blue else {"turn_to_damage": True}),
                )
            )
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.commander.status_in)
            if is_blue:
                port = self.commander.orders_out.get(unit_id)
                if port is not None:
                    self.connectPorts(port, soldier.command_in)
            else:
                brain = self.addSubModel(
                    RulePolicyAtomic(
                        name=f"Red_Rule_{unit_id}",
                        policy=UrbanRedPolicy(
                            target_type="soldier",
                            obstacles=snapshot.obstacles,
                            target_priority=red_target_priority,
                        ),
                        decision_delay=1.0,
                    )
                )
                self.connectPorts(soldier.observation_out, brain.observation_in)
                self.connectPorts(brain.command_out, soldier.command_in)
                self.connectPorts(self.world.spawn_out, soldier.spawn_in)


def _plan_for_candidate(plans: FutureActionPlanBatch, candidate_index: int) -> _CandidatePlan:
    """torch plan batch에서 한 후보를 CPU numpy로 잘라낸다."""
    return _CandidatePlan(
        action_type_ids=plans.action_type_ids[candidate_index].detach().cpu().numpy(),
        move_xy_norm=plans.move_xy_norm[candidate_index].detach().cpu().numpy(),
        target_entity_ids=plans.target_entity_ids[candidate_index].detach().cpu().numpy(),
        theta_radians=plans.theta_radians[candidate_index].detach().cpu().numpy(),
        issued_mask=plans.issued_mask[candidate_index].detach().cpu().numpy(),
        unit_ids=plans.action_unit_ids[candidate_index, 0].detach().cpu().numpy(),
    )


def _frames_to_features(
    *,
    snapshot: RolloutSnapshot,
    frames: Mapping[float, Mapping[int, Mapping[str, object]]],
    horizon: int,
) -> np.ndarray:
    """rollout frame들을 (H, N, F) slot feature 배열로 변환한다."""
    dead_rows = {
        int(row["id"]): row for row in snapshot.unit_rows if float(row["hp"]) <= 0.0
    }
    all_ids = sorted(int(row["id"]) for row in snapshot.unit_rows)
    feature_frames: list[np.ndarray] = []
    last_known: dict[int, Mapping[str, object]] = {
        int(row["id"]): row for row in snapshot.unit_rows
    }
    for step in range(1, horizon + 1):
        frame_rows = frames.get(float(step), {})
        merged: list[dict[str, str]] = []
        for unit_id in all_ids:
            if unit_id in frame_rows:
                source = frame_rows[unit_id]
                last_known[unit_id] = source
            elif unit_id in dead_rows:
                source = dead_rows[unit_id]
            else:
                source = last_known[unit_id]
            merged.append({key: str(value) for key, value in source.items()})
        batch = build_slot_batch(
            unit_rows=merged,
            obstacles=snapshot.obstacles,
            time_sec=snapshot.base_time_sec + float(step),
            duration_sec=snapshot.episode_duration_sec,
        )
        feature_frames.append(batch.features)
    return np.stack(feature_frames, axis=0)


def rollout_plans_with_devs(
    *,
    plans: FutureActionPlanBatch,
    snapshot: RolloutSnapshot,
    seed: int,
    device: torch.device,
    red_target_priority: str = "nearest",
) -> torch.Tensor:
    """CEM 후보 전체를 실제 DEVS로 rollout해 미래 feature를 반환한다.

    반환 shape는 `(C, H, N, MAX_FEATURE_DIM)`이고 rollout_with_world_model과
    같은 계약이다. 모든 후보는 같은 seed로 시작해 후보 간 점수 차이가
    난수가 아니라 액션 차이에서 나오게 한다.

    rollout은 DEVS 코드가 쓰는 전역 random 모듈을 후보마다 재시드해야 하므로,
    진입 전에 전역 RNG 상태를 저장했다가 종료 시 복원한다. 이걸 빼먹으면
    본 에피소드의 난수 스트림이 rollout seed로 덮여 에피소드 간 다양성이
    통째로 사라진다.
    """
    candidates, horizon, _, _ = plans.action_features.shape
    features: list[np.ndarray] = []
    episode_rng_state = random.getstate()
    try:
        for candidate_index in range(candidates):
            plan = _plan_for_candidate(plans, candidate_index)
            random.seed(seed)
            battle = _RolloutBattleModel(
                snapshot=snapshot,
                plan=plan,
                horizon=horizon,
                red_target_priority=red_target_priority,
            )
            simulator = Simulator(battle)
            simulator.setTerminationTime(float(horizon) + 0.5)
            simulator.simulate()
            features.append(
                _frames_to_features(snapshot=snapshot, frames=battle.commander.frames, horizon=horizon)
            )
    finally:
        random.setstate(episode_rng_state)
    stacked = np.stack(features, axis=0).astype(np.float32)
    if stacked.shape != (candidates, horizon, stacked.shape[2], MAX_FEATURE_DIM):
        raise ValueError(f"rollout feature shape가 잘못됐다: {stacked.shape}")
    return torch.as_tensor(stacked, device=device)


def snapshot_from_slot_rows(
    *,
    unit_rows: Sequence[Mapping[str, object]],
    obstacles: Sequence[Sequence[float]],
    base_time_sec: float,
    episode_duration_sec: float,
) -> RolloutSnapshot:
    """commander가 들고 있는 현재 unit row로 rollout 스냅샷을 만든다."""
    return RolloutSnapshot(
        unit_rows=tuple(dict(row) for row in unit_rows),
        obstacles=tuple(
            (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])) for rect in obstacles
        ),
        base_time_sec=float(base_time_sec),
        episode_duration_sec=float(episode_duration_sec),
    )
