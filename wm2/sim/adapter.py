"""구 DEVS 시뮬레이터 격리 어댑터 — wm2에서 구코드를 import하는 유일한 파일 (설계 0절).

wm2 계약(numpy)을 구 계약(RolloutSnapshot / FutureActionPlanBatch)으로 변환해
`rollout_plans_with_devs`를 호출하고, 결과를 유닛 상태 numpy로 되돌린다.

구코드는 `hackerthon.` 절대 import를 쓰므로 리포 부모 디렉터리를 sys.path에 얹는다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

import torch  # noqa: E402

from hackerthon.worldmodel.actions import ACTION_DIM as OLD_ACTION_DIM  # noqa: E402
from hackerthon.worldmodel.cem_planner import FutureActionPlanBatch  # noqa: E402
from hackerthon.worldmodel.devs_rollout import (  # noqa: E402
    RolloutSnapshot,
    rollout_plans_with_devs,
)

from ..data.episodes import Episode  # noqa: E402

# 구 slot feature(unit) 배치: team, hp_ratio, ammo_ratio, x, y, cos, sin, alive
OLD_UNIT_HP, OLD_UNIT_AMMO = 1, 2
OLD_UNIT_X, OLD_UNIT_Y = 3, 4
OLD_UNIT_COS, OLD_UNIT_SIN = 5, 6


@dataclass(frozen=True)
class PlanSpec:
    """wm2 쪽 무작위 계획 (전부 numpy).

    action_type_ids: (C, H, U_blue) — 0 STOP / 1 MOVE / 2 ENGAGE / 3 TURN (구와 동일 어휘)
    move_xy_norm:    (C, H, U_blue, 2) [-1,1]
    target_ids:      (C, H, U_blue) ENGAGE 표적 entity id (아니면 0)
    theta_radians:   (C, H, U_blue)
    issued:          (C, H, U_blue) bool
    """

    action_type_ids: np.ndarray
    move_xy_norm: np.ndarray
    target_ids: np.ndarray
    theta_radians: np.ndarray
    issued: np.ndarray


def rollout(
    *,
    episode: Episode,
    base_tick: int,
    plan: PlanSpec,
    horizon: int = 6,
    seed: int = 0,
    red_target_priority: str = "nearest",
) -> np.ndarray:
    """base_tick 상태에서 계획들을 DEVS로 굴린다.

    반환: (C, H, U, 4+2+2) = 유닛별 (x, y, hp, ammo, heading_cos, heading_sin) —
    유닛 축은 episode 유닛 id 오름차순 (wm2 layout과 동일).
    """
    frame = episode.frames[base_tick]
    unit_ids = sorted(frame)
    unit_rows = tuple(
        {
            "id": uid,
            "x": frame[uid].x,
            "y": frame[uid].y,
            "heading": frame[uid].heading_deg,
            "hp": frame[uid].hp,
            "ammo": frame[uid].ammo,
        }
        for uid in unit_ids
    )
    snapshot = RolloutSnapshot(
        unit_rows=unit_rows,
        obstacles=episode.obstacles,
        base_time_sec=float(base_tick),
        episode_duration_sec=episode.duration_sec,
        objective=episode.objective,
        mission_type=episode.mission_type,
    )

    candidates, plan_horizon, num_blue = plan.action_type_ids.shape
    if plan_horizon != horizon:
        raise ValueError("plan horizon이 rollout horizon과 다르다")
    blue_ids = sorted(episode.blue_ids)
    if num_blue != len(blue_ids):
        raise ValueError("plan의 유닛 수가 BLUE 수와 다르다")

    as_t = torch.as_tensor
    shape3 = (candidates, horizon, num_blue)
    plans = FutureActionPlanBatch(
        action_features=torch.zeros(*shape3, OLD_ACTION_DIM),  # shape 검증용, rollout은 안 읽음
        action_unit_ids=as_t(np.asarray(blue_ids)).reshape(1, 1, -1).expand(*shape3).clone(),
        issued_mask=as_t(plan.issued.astype(bool)),
        action_type_ids=as_t(plan.action_type_ids.astype(np.int64)),
        target_entity_ids=as_t(plan.target_ids.astype(np.int64)),
        target_indices=torch.zeros(*shape3, dtype=torch.long),
        move_xy_norm=as_t(plan.move_xy_norm.astype(np.float32)),
        theta_radians=as_t(plan.theta_radians.astype(np.float32)),
        red_target_ids=as_t(np.asarray(sorted(episode.red_ids), dtype=np.int64)),
    )

    features = rollout_plans_with_devs(
        plans=plans,
        snapshot=snapshot,
        seed=seed,
        device=torch.device("cpu"),
        red_target_priority=red_target_priority,
    )
    # (C, H, N, F) — 앞 U개 슬롯이 id 정렬 유닛 (구 build_slot_batch 규약)
    array = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    units = array[:, :, : len(unit_ids), :]
    from ..model.features import denorm_x, denorm_y, MAX_HP, MAX_AMMO

    out = np.zeros((candidates, horizon, len(unit_ids), 6), dtype=np.float32)
    out[..., 0] = np.vectorize(denorm_x)(units[..., OLD_UNIT_X])
    out[..., 1] = np.vectorize(denorm_y)(units[..., OLD_UNIT_Y])
    out[..., 2] = units[..., OLD_UNIT_HP] * MAX_HP
    out[..., 3] = units[..., OLD_UNIT_AMMO] * MAX_AMMO
    out[..., 4] = units[..., OLD_UNIT_COS]
    out[..., 5] = units[..., OLD_UNIT_SIN]
    return out


# ── rule 연속 실행 (V 검증용 21초 실현 측정, eval/planning) ──────────────
import math as _math  # noqa: E402
import random as _random  # noqa: E402

from pypdevs.DEVS import AtomicDEVS, CoupledDEVS  # noqa: E402
from pypdevs.infinity import INFINITY  # noqa: E402
from pypdevs.simulator import Simulator  # noqa: E402

from hackerthon.red_policy import UrbanRedPolicy  # noqa: E402
from hackerthon.simulation_direct_commander_5v5 import RulePolicyAtomic  # noqa: E402
from hackerthon.sim_units import LosSoldierAtomic, LosWorldAtomic  # noqa: E402


class _FrameRecorder(AtomicDEVS):
    """soldier status를 시각별로 모으기만 하는 수동 관측자."""

    def __init__(self):
        super().__init__("FrameRecorder")
        self.status_in = self.addInPort("status_in")
        self.frames: dict[float, dict[int, dict]] = {}
        self.state = "RECORD"

    def extTransition(self, inputs):
        for row in inputs.get(self.status_in, ()):
            self.frames.setdefault(round(float(row["time"]), 2), {})[int(row["id"])] = row
        return self.state

    def timeAdvance(self):
        return INFINITY


class _RuleContinuationBattle(CoupledDEVS):
    """임의 상태에서 양 팀 모두 rule로 계속 싸우는 battle (구 _RolloutBattleModel 변형).

    BLUE도 RED와 같은 RulePolicyAtomic 파이프라인을 탄다 — V의 의미("rule로 계속
    갔을 때")에 대응하는 근사 실행이다.
    """

    def __init__(self, *, rows, obstacles, objective, blue_target_priority, red_target_priority, seed):
        super().__init__("RuleContinuationBattle")
        alive = [r for r in rows if float(r["hp"]) > 0.0]
        self.world = self.addSubModel(
            LosWorldAtomic(
                initial_entities=[
                    {
                        "id": int(r["id"]),
                        "type": "soldier" if int(r["id"]) < 200 else "enemy",
                        "x": float(r["x"]), "y": float(r["y"]),
                        "heading": float(r["heading"]),
                        "hp": int(round(float(r["hp"]))), "ammo": int(r["ammo"]),
                        "state": "ALIVE",
                    }
                    for r in alive
                ],
                obstacles=obstacles,
                expected_damage=True,
            )
        )
        self.recorder = self.addSubModel(_FrameRecorder())
        for r in alive:
            unit_id = int(r["id"])
            is_blue = unit_id < 200
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=("Blue_" if is_blue else "Red_") + str(unit_id),
                    soldier_id=unit_id,
                    initial_x=float(r["x"]), initial_y=float(r["y"]),
                    initial_heading=float(r["heading"]),
                    hp=int(round(float(r["hp"]))), ammo=int(r["ammo"]),
                    fov_deg=120.0, obstacles=obstacles,
                    **({} if is_blue else {"turn_to_damage": True}),
                )
            )
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.recorder.status_in)
            policy = UrbanRedPolicy(
                target_type="enemy" if is_blue else "soldier",
                obstacles=obstacles,
                target_priority=blue_target_priority if is_blue else red_target_priority,
                lane_seed=seed + (1 if is_blue else 0),
                assault_target=tuple(objective) if is_blue else None,
                max_step=1.0,
            )
            brain = self.addSubModel(
                RulePolicyAtomic(
                    name=("Blue_Rule_" if is_blue else "Red_Rule_") + str(unit_id),
                    policy=policy,
                    decision_delay=1.0,
                )
            )
            self.connectPorts(soldier.observation_out, brain.observation_in)
            self.connectPorts(brain.command_out, soldier.command_in)


def continue_with_rules(
    *,
    unit_states: np.ndarray,     # (U, 6) = x, y, hp, ammo, cos, sin — rollout 출력 그대로
    unit_ids: list[int],
    obstacles,
    objective: tuple[float, float],
    seconds: int = 15,
    seed: int = 0,
    blue_target_priority: str = "nearest",
    red_target_priority: str = "nearest",
) -> np.ndarray:
    """(U, 4) = seconds 뒤의 x, y, hp, ammo. 사망자는 그 자리에 동결."""
    rows = [
        {
            "id": uid,
            "x": float(unit_states[i, 0]), "y": float(unit_states[i, 1]),
            "heading": _math.degrees(_math.atan2(float(unit_states[i, 5]), float(unit_states[i, 4]))),
            "hp": float(unit_states[i, 2]), "ammo": float(unit_states[i, 3]),
        }
        for i, uid in enumerate(unit_ids)
    ]
    state = _random.getstate()
    try:
        _random.seed(seed)
        battle = _RuleContinuationBattle(
            rows=rows, obstacles=obstacles, objective=objective,
            blue_target_priority=blue_target_priority,
            red_target_priority=red_target_priority, seed=seed,
        )
        simulator = Simulator(battle)
        simulator.setTerminationTime(float(seconds) + 1.5)
        simulator.simulate()
        frames = battle.recorder.frames
    finally:
        _random.setstate(state)

    final = np.zeros((len(unit_ids), 4), dtype=np.float32)
    last_known = {int(r["id"]): r for r in rows}
    times = sorted(frames)
    for t in times:
        if t > seconds + 1e-6:
            break
        for uid, row in frames[t].items():
            last_known[uid] = row
    for i, uid in enumerate(unit_ids):
        r = last_known[uid]
        final[i] = (float(r["x"]), float(r["y"]), float(r["hp"]), float(r["ammo"]))
    return final
