"""CEM 지휘관이 6틱마다 재계획하는 전체 에피소드 (4a-1).

구코드 접점 — sim/ 패키지의 두 번째 파일 (adapter.py와 함께). RED는 에피소드 내내
연속된 RulePolicyAtomic(lane 상태 보존 — chained rollout과 달리 학습 분포와 같은
세계)이고, BLUE는 CEMCommanderAtomic이 계획·집행한다.

부산물로 V online 학습 짝을 수집한다: 매 재계획의 (상상 ŝ₆ 특징, 이후 실제 15초
progress 증가분). CEM 자신의 분포에서 나온 V 데이터 — 오프라인 replay 상상으로는
닫히지 않던 분포 갭(V_cont −0.21)을 닫는 경로다.
"""

from __future__ import annotations

import math
import random as _random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .adapter import _REPO_ROOT  # sys.path 부트스트랩 재사용  # noqa: F401

from pypdevs.DEVS import AtomicDEVS, CoupledDEVS  # noqa: E402
from pypdevs.infinity import INFINITY  # noqa: E402
from pypdevs.simulator import Simulator  # noqa: E402

from hackerthon.red_policy import UrbanRedPolicy  # noqa: E402
from hackerthon.simulation_direct_commander_5v5 import RulePolicyAtomic  # noqa: E402
from hackerthon.sim_units import LosSoldierAtomic, LosWorldAtomic  # noqa: E402
from hackerthon.terrain import has_los, largest_free_component, next_waypoint, snap_to_component  # noqa: E402

from ..config import CEMConfig
from ..data.episodes import UnitState
from ..data.windows import (
    ACTION_TICKS,
    EpisodeLayout,
    TOTAL_FRAMES,
    Window,
    _mission_vector,
    _parse_action,
    _unit_vector,
)
from ..model.features import (
    ACTION_ENGAGE,
    ACTION_MOVE,
    ACTION_STOP,
    ACTION_TURN,
    MAX_AMMO,
    MAX_FIRE_RANGE_UNITS,
    MAX_HP,
    MISSION_HOLD_OBJECTIVE,
    OBJECTIVE_RADIUS,
    TeamId,
    norm_x,
    norm_y,
)
from ..plan.cem import PlanCandidates, plan as cem_plan, _action_tokens
from ..model.rollout import assemble_hp, assemble_positions, clamp_physics
from ..plan.score import progress_batch, value_input_from_assembled

REPLAN_EVERY = 6
FIRST_PLAN_TICK = 3   # pre..h2 = 0..3 이 모여야 첫 window가 선다


@dataclass
class VPair:
    tick: int
    unit_features: np.ndarray     # (U, 10) — 상상 ŝ₆
    mission_features: np.ndarray  # (5,)
    predicted_value: float        # 수집 시점 V 예측 (실시간 건강 지표용)
    label: float | None = None    # 에피소드 종료 후 채움


@dataclass
class EpisodeResult:
    frames: dict[int, dict[int, UnitState]]
    raw_frames: dict[float, dict[int, dict]]   # soldier_log 형식 저장용 (mode/target 포함)
    executed_commands: list[dict]
    planned_commands: list[dict]
    v_pairs: list[VPair]
    outcome: str
    final_progress: float
    progress_by_tick: dict[int, float]


class _MissionView:
    """windows._mission_vector가 기대하는 최소 인터페이스."""

    def __init__(self, mission_type, objective, duration_sec, blue_ids, red_ids):
        self.mission_type = mission_type
        self.objective = objective
        self.duration_sec = duration_sec
        self.blue_ids = blue_ids
        self.red_ids = red_ids


class CEMCommanderAtomic(AtomicDEVS):
    def __init__(self, *, bridge: "PlannerBridge", blue_ids, all_ids, obstacles, duration):
        super().__init__("CEMCommander")
        self.bridge = bridge
        self.blue_ids = tuple(blue_ids)
        self.all_ids = tuple(all_ids)
        self.obstacles = [tuple(r) for r in obstacles]
        self.duration = float(duration)
        self.status_in = self.addInPort("status_in")
        self.orders_out = {uid: self.addOutPort(f"orders_out_{uid}") for uid in self.blue_ids}
        self.sigma = INFINITY
        self.state = "WAIT"
        self.frames: dict[float, dict[int, dict]] = {}
        self._commanded: set[float] = set()
        self._free_component = None

    def timeAdvance(self):
        return self.sigma

    def _latest_complete(self):
        complete = [
            t for t, rows in self.frames.items()
            if all(uid in rows for uid in self.all_ids)
        ]
        return max(complete) if complete else None

    def extTransition(self, inputs):
        if self.status_in in inputs:
            for status in inputs[self.status_in]:
                # mode/target_id는 soldier_log 형식 호환용으로 함께 보관한다
                row = {
                    k: status.get(k, "")
                    for k in ("time", "id", "x", "y", "heading", "hp", "ammo", "mode", "target_id")
                }
                self.frames.setdefault(round(float(row["time"]), 2), {})[int(row["id"])] = row
        latest = self._latest_complete()
        if latest is not None and latest not in self._commanded and latest < self.duration - 0.5:
            self.sigma = 0.0
            self.state = "READY"
        return self.state

    def intTransition(self):
        self.sigma = INFINITY
        self.state = "WAIT"
        return self.state

    def outputFnc(self):
        latest = self._latest_complete()
        if latest is None or latest in self._commanded:
            return {}
        self._commanded.add(latest)
        tick = int(round(latest))
        commands = self.bridge.commands_for_tick(tick, self.frames)
        out = {}
        for command in commands:
            port = self.orders_out.get(int(command["unit_id"]))
            if port is not None:
                out[port] = [command]
        return out

    def free_component(self):
        if self._free_component is None:
            self._free_component = largest_free_component(list(self.obstacles))
        return self._free_component


class PlannerBridge:
    """DEVS 밖의 계획·집행 로직. commander atomic이 tick마다 호출한다."""

    def __init__(
        self, *, model, heads, value_head, layout: EpisodeLayout, cem_config: CEMConfig,
        device, rng: np.random.Generator, lam: float, obstacles, duration: float,
        commander_ref: list, label: str = "",
    ):
        self.label = label
        self.model = model
        self.heads = heads
        self.value_head = value_head
        self.layout = layout
        self.cem_config = cem_config
        self.device = device
        self.rng = rng
        self.lam = lam
        self.obstacles = [tuple(r) for r in obstacles]
        self.duration = duration
        self.commander_ref = commander_ref
        self.mission_view = _MissionView(
            layout.mission_type, layout.objective, duration,
            tuple(layout.unit_ids[: layout.num_blue]), tuple(layout.unit_ids[layout.num_blue:]),
        )
        self.active_plan: PlanCandidates | None = None
        self.plan_start_tick = -1
        self.issued: dict[int, dict[int, dict]] = {}     # tick → uid → 실행 명령
        self.executed_log: list[dict] = []
        self.planned_log: list[dict] = []
        self.v_pairs: list[VPair] = []

    # ── window 구성 ────────────────────────────────────────────────────
    def _states(self, frames, tick) -> dict[int, UnitState]:
        rows = frames.get(float(tick), {})
        return {
            uid: UnitState(
                unit_id=uid, x=float(r["x"]), y=float(r["y"]),
                heading_deg=float(r["heading"]), hp=float(r["hp"]), ammo=float(r["ammo"]),
            )
            for uid, r in rows.items()
        }

    def _build_window(self, frames, tick) -> Window:
        layout = self.layout
        states = {t: self._states(frames, t) for t in range(tick - 3, tick + 1)}
        unit_features = np.zeros((TOTAL_FRAMES, layout.num_units, 10), dtype=np.float32)
        for fi in range(3):
            t = tick - 2 + fi
            for ui, uid in enumerate(layout.unit_ids):
                unit_features[fi, ui] = _unit_vector(
                    states[t][uid], states[t - 1][uid], int(layout.team_ids[ui])
                )
        mission_features = np.zeros((TOTAL_FRAMES, 5), dtype=np.float32)
        for fi in range(3):
            t = tick - 2 + fi
            mission_features[fi] = _mission_vector(self.mission_view, t, states[t])
        actions = np.zeros((ACTION_TICKS, layout.num_blue, 19), dtype=np.float32)
        red_ids = tuple(sorted(self.mission_view.red_ids))
        blue_slot = {uid: i for i, uid in enumerate(layout.unit_ids[: layout.num_blue])}
        for j, t in enumerate((tick - 2, tick - 1)):
            for uid, cmd in self.issued.get(t, {}).items():
                slot = blue_slot.get(uid)
                if slot is not None:
                    actions[j, slot] = _parse_action(
                        cmd["action"], cmd.get("detail", ""), red_ids
                    )
        zeros_label = np.zeros((TOTAL_FRAMES - 1, layout.num_units), dtype=np.float32)
        return Window(
            layout=layout, anchor_tick=tick - 2,
            unit_features=unit_features, mission_features=mission_features,
            dpos=np.zeros((TOTAL_FRAMES - 1, layout.num_units, 2), dtype=np.float32),
            ddmg=zeros_label, dammo=zeros_label,
            heading=np.zeros((TOTAL_FRAMES - 1, layout.num_units, 2), dtype=np.float32),
            completion=np.zeros(6, dtype=np.float32),
            pos_loss_mask=np.asarray(
                [states[tick][uid].hp > 0 for uid in layout.unit_ids], dtype=bool
            ),
            actions=actions,
        )

    # ── V 짝 수집: best 후보의 상상 ŝ₆ ─────────────────────────────────
    @torch.no_grad()
    def _collect_v_pair(self, window: Window, result, tick):
        best = PlanCandidates(
            result.candidates.action_type_ids[result.best_index : result.best_index + 1],
            result.candidates.move_xy[result.best_index : result.best_index + 1],
            result.candidates.target_slots[result.best_index : result.best_index + 1],
            result.candidates.theta[result.best_index : result.best_index + 1],
            result.candidates.issued[result.best_index : result.best_index + 1],
        )
        tokens = torch.from_numpy(_action_tokens(window, best)).to(self.device)
        unit_t = torch.from_numpy(window.unit_features).to(self.device).unsqueeze(0)
        out = self.model(
            unit_features=unit_t,
            terrain_features=torch.from_numpy(self.layout.terrain_features).to(self.device).unsqueeze(0),
            mission_features=torch.from_numpy(window.mission_features).to(self.device).unsqueeze(0),
            actions=tokens,
            team_ids=torch.from_numpy(np.asarray(self.layout.team_ids)).to(self.device),
            masked_units=torch.zeros(1, self.layout.num_units, dtype=torch.bool, device=self.device),
        )
        pred = self.heads(out["unit_tokens"], out["mission_tokens"])
        from ..model.features import denorm_x, denorm_y

        anchor = window.unit_features[0]
        anchor_xy = torch.tensor(
            [[denorm_x(v) for v in anchor[:, 3]], [denorm_y(v) for v in anchor[:, 4]]]
        ).T.float().unsqueeze(0).to(self.device)
        anchor_hp = torch.from_numpy((anchor[:, 1] * MAX_HP).astype(np.float32)).unsqueeze(0).to(self.device)
        raw = assemble_positions(anchor_xy, pred["dpos"][:, 2:])
        hp = assemble_hp(anchor_hp, pred["ddmg"][:, 2:])
        clamped = clamp_physics(raw, anchor_xy, hp, anchor_hp > 0)
        ammo_final = (
            torch.from_numpy((anchor[:, 2] * MAX_AMMO).astype(np.float32)).unsqueeze(0).to(self.device)
            - pred["dammo"][:, -1].clamp_min(0) * MAX_AMMO
        ).clamp_min(0)
        uf, mf = value_input_from_assembled(
            positions=clamped, hp=hp, ammo=ammo_final, heading=pred["heading"][:, -1],
            team_ids=torch.from_numpy(np.asarray(self.layout.team_ids)).to(self.device),
            mission_type=self.layout.mission_type, objective=self.layout.objective,
            time_remaining=max(0.0, (self.duration - (tick + 6)) / self.duration),
        )
        self.v_pairs.append(
            VPair(
                tick=tick,
                unit_features=uf[0].cpu().numpy(),
                mission_features=mf[0].cpu().numpy(),
                predicted_value=float(result.value[result.best_index]),
            )
        )

    # ── tick 처리 ──────────────────────────────────────────────────────
    def commands_for_tick(self, tick: int, frames) -> list[dict]:
        if tick < FIRST_PLAN_TICK:
            return []
        if (
            self.active_plan is None
            or tick - self.plan_start_tick >= REPLAN_EVERY
        ) and (tick - FIRST_PLAN_TICK) % REPLAN_EVERY == 0:
            window = self._build_window(frames, tick)
            started = time.time()
            result = cem_plan(
                window=window, model=self.model, heads=self.heads,
                value_head=self.value_head if self.lam != 0.0 else None,
                config=self.cem_config, device=self.device, rng=self.rng, lam=self.lam,
            )
            print(
                f"{self.label} tick={tick} 재계획 best={result.scores[result.best_index]:+.3f} "
                f"(gain {result.gain[result.best_index]:+.3f} V {result.value[result.best_index]:+.3f}) "
                f"{time.time()-started:.1f}s",
                flush=True,
            )
            self.active_plan = PlanCandidates(
                result.candidates.action_type_ids[result.best_index : result.best_index + 1],
                result.candidates.move_xy[result.best_index : result.best_index + 1],
                result.candidates.target_slots[result.best_index : result.best_index + 1],
                result.candidates.theta[result.best_index : result.best_index + 1],
                result.candidates.issued[result.best_index : result.best_index + 1],
            )
            self.plan_start_tick = tick
            self._collect_v_pair(window, result, tick)
        if self.active_plan is None:
            return []
        step = tick - self.plan_start_tick
        if step >= REPLAN_EVERY:
            return []
        return self._execute_step(step, tick, frames)

    def _execute_step(self, step: int, tick: int, frames) -> list[dict]:
        layout = self.layout
        states = self._states(frames, tick)
        red_ids = sorted(self.mission_view.red_ids)
        commands = []
        commander = self.commander_ref[0]
        for ui, uid in enumerate(layout.unit_ids[: layout.num_blue]):
            if not self.active_plan.issued[0, step, ui] or states[uid].hp <= 0:
                continue
            action = int(self.active_plan.action_type_ids[0, step, ui])
            me = (states[uid].x, states[uid].y)
            planned = {"tick": tick, "unit_id": uid, "step": step, "action": action}
            command = None
            if action == ACTION_STOP:
                command = {"unit_id": uid, "action": "STOP", "duration_sec": 1.0,
                           "detail": "", "reason": f"cem stop|step={step}"}
            elif action == ACTION_MOVE:
                target = tuple(self.active_plan.move_xy[0, step, ui])
                planned["target"] = target
                waypoint = next_waypoint(me, target, self.obstacles, max_step=1.0)
                if waypoint is None:
                    projected = snap_to_component(target, commander.free_component())
                    if projected is not None:
                        waypoint = next_waypoint(me, projected, self.obstacles, max_step=1.0)
                if waypoint is None:
                    command = {"unit_id": uid, "action": "STOP", "duration_sec": 1.0,
                               "detail": "", "reason": f"cem move blocked|plan=MOVE|step={step}"}
                else:
                    command = {"unit_id": uid, "action": "MOVE",
                               "x": round(float(waypoint[0]), 3), "y": round(float(waypoint[1]), 3),
                               "duration_sec": 1.0,
                               "detail": f"({round(float(waypoint[0]),3)},{round(float(waypoint[1]),3)})",
                               "reason": f"cem move|step={step}"}
            elif action == ACTION_ENGAGE:
                slot = int(self.active_plan.target_slots[0, step, ui])
                target_id = red_ids[slot] if 0 <= slot < len(red_ids) else -1
                planned["target_id"] = target_id
                chosen = self._feasible_target(uid, target_id, states)
                if chosen is not None:
                    tag = "" if chosen == target_id else " resampled"
                    command = {"unit_id": uid, "action": "ENGAGE", "target_id": chosen,
                               "duration_sec": 1.0, "detail": f"->R{chosen}",
                               "reason": f"cem engage{tag}|plan=ENGAGE|step={step}"}
                else:
                    command = {"unit_id": uid, "action": "STOP", "duration_sec": 1.0,
                               "detail": "", "reason": f"cem engage blocked|plan=ENGAGE|step={step}"}
            elif action == ACTION_TURN:
                theta = math.degrees(float(self.active_plan.theta[0, step, ui]))
                command = {"unit_id": uid, "action": "TURN", "theta": round(theta, 2),
                           "duration_sec": 1.0, "detail": f"{round(theta,2)}",
                           "reason": f"cem turn|step={step}"}
            if command is None:
                continue
            self.planned_log.append(planned)
            self.executed_log.append({"tick": tick, **command})
            self.issued.setdefault(tick, {})[uid] = command
            commands.append(command)
        return commands

    def _feasible_target(self, shooter_id, target_id, states) -> int | None:
        me = states[shooter_id]

        def ok(tid):
            t = states.get(tid)
            if t is None or t.hp <= 0:
                return False
            if math.hypot(t.x - me.x, t.y - me.y) > MAX_FIRE_RANGE_UNITS:
                return False
            return has_los((me.x, me.y), (t.x, t.y), self.obstacles)

        if ok(target_id):
            return target_id
        candidates = [
            tid for tid in self.mission_view.red_ids if ok(tid)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda tid: math.hypot(states[tid].x - me.x, states[tid].y - me.y))


def _progress(states: dict[int, UnitState], layout: EpisodeLayout) -> float:
    team = torch.from_numpy(np.asarray(layout.team_ids))
    ids = list(layout.unit_ids)
    pos = torch.tensor([[states[u].x, states[u].y] for u in ids]).float().unsqueeze(0)
    hp = torch.tensor([states[u].hp for u in ids]).float().unsqueeze(0)
    return float(progress_batch(pos, hp, team, layout.mission_type, layout.objective))


def run_cem_episode(
    *,
    scenario: dict,               # config.json 내용 (obstacles, objective, mission, ids, initial_positions)
    layout: EpisodeLayout,
    model, heads, value_head,
    cem_config: CEMConfig,
    device, lam: float, seed: int,
    duration: float = 60.0,
    label: str = "",
    survival_beta: float | None = None,   # None이면 config.SURVIVAL_BETA — β ablation용 오버라이드
) -> EpisodeResult:
    obstacles = [tuple(r) for r in scenario["obstacles"]]
    blue_ids = sorted(int(v) for v in scenario["blue_ids"])
    red_ids = sorted(int(v) for v in scenario["red_ids"])
    all_ids = blue_ids + red_ids
    # config의 initial_positions는 {"blue": [{"id","x","y","heading"}...], "red": [...]}
    spawn: dict[int, tuple[float, float, float]] = {}
    positions = scenario["initial_positions"]
    rows = (
        positions.get("blue", []) + positions.get("red", [])
        if isinstance(positions, dict)
        else positions
    )
    for row in rows:
        spawn[int(row["id"])] = (float(row["x"]), float(row["y"]), float(row.get("heading", 0.0)))

    rng = np.random.default_rng(seed)
    commander_ref: list = []
    bridge = PlannerBridge(
        model=model, heads=heads, value_head=value_head, layout=layout,
        cem_config=cem_config, device=device, rng=rng, lam=lam,
        obstacles=obstacles, duration=duration, commander_ref=commander_ref,
        label=label,
    )
    episode_started = time.time()
    print(
        f"{label} 시작: {len(blue_ids)}v{len(red_ids)} mission={layout.mission_type} "
        f"objective=({layout.objective[0]:.1f},{layout.objective[1]:.1f})",
        flush=True,
    )

    state = _random.getstate()
    try:
        _random.seed(seed)

        class _Battle(CoupledDEVS):
            def __init__(self):
                super().__init__("CEMLoopBattle")
                self.world = self.addSubModel(LosWorldAtomic(
                    initial_entities=[
                        {"id": uid, "type": "soldier" if uid < 200 else "enemy",
                         "x": spawn[uid][0], "y": spawn[uid][1], "heading": spawn[uid][2],
                         "hp": 100, "ammo": 30, "state": "ALIVE"}
                        for uid in all_ids
                    ],
                    obstacles=obstacles, expected_damage=True,
                ))
                self.commander = self.addSubModel(CEMCommanderAtomic(
                    bridge=bridge, blue_ids=blue_ids, all_ids=all_ids,
                    obstacles=obstacles, duration=duration,
                ))
                commander_ref.append(self.commander)
                assault = tuple(scenario["objective"]) if layout.mission_type == MISSION_HOLD_OBJECTIVE else None
                for uid in all_ids:
                    is_blue = uid < 200
                    soldier = self.addSubModel(LosSoldierAtomic(
                        name=("Blue_" if is_blue else "Red_") + str(uid), soldier_id=uid,
                        initial_x=spawn[uid][0], initial_y=spawn[uid][1],
                        initial_heading=spawn[uid][2], hp=100, ammo=30,
                        fov_deg=120.0, obstacles=obstacles,
                        **({} if is_blue else {"turn_to_damage": True}),
                    ))
                    self.connectPorts(self.world.world_out, soldier.world_in)
                    self.connectPorts(self.world.damage_out, soldier.damage_in)
                    self.connectPorts(soldier.status_out, self.world.status_in)
                    self.connectPorts(soldier.status_out, self.commander.status_in)
                    if is_blue:
                        port = self.commander.orders_out[uid]
                        self.connectPorts(port, soldier.command_in)
                    else:
                        brain = self.addSubModel(RulePolicyAtomic(
                            name=f"Red_Rule_{uid}",
                            policy=UrbanRedPolicy(
                                target_type="soldier", obstacles=obstacles,
                                target_priority="nearest", lane_seed=seed,
                                assault_target=assault, max_step=1.0,
                            ),
                            decision_delay=1.0,
                        ))
                        self.connectPorts(soldier.observation_out, brain.observation_in)
                        self.connectPorts(brain.command_out, soldier.command_in)

        battle = _Battle()
        simulator = Simulator(battle)
        simulator.setTerminationTime(duration + 0.5)
        simulator.simulate()
        raw_frames = battle.commander.frames
    finally:
        _random.setstate(state)

    frames: dict[int, dict[int, UnitState]] = {}
    for t, rows in raw_frames.items():
        tick = int(round(t))
        if abs(t - tick) > 1e-6:
            continue
        frames[tick] = {
            uid: UnitState(uid, float(r["x"]), float(r["y"]), float(r["heading"]),
                           float(r["hp"]), float(r["ammo"]))
            for uid, r in rows.items()
        }

    progress_by_tick = {t: _progress(states, layout) for t, states in sorted(frames.items())}
    last = max(frames)
    blue_alive_end = any(frames[last][u].hp > 0 for u in blue_ids if u in frames[last])
    # 승패는 최종 프레임 기준 — 구 시스템(evaluate_fixed_batch._combat_outcome)과 동일 관점.
    # 순간 터치(completed_ticks) 판정은 터치 후 전멸을 WIN으로 세는 부풀림이 있었다.
    outcome = "WIN" if progress_by_tick[last] >= 1.0 - 1e-6 else ("LOSE" if not blue_alive_end else "TIMEOUT")

    # V 짝 라벨: γ-할인 return-to-go, δ = Δprogress + β·Δ아군HP비율 —
    # train_value_rtg·score.py와 동일 정의/상수(config). 온라인 갱신이 rtg+생존 V를
    # 다른 의미의 라벨로 끌어가지 않도록 정합을 유지한다.
    from ..config import SURVIVAL_BETA, VALUE_GAMMA
    from ..model.features import MAX_HP

    beta = SURVIVAL_BETA if survival_beta is None else survival_beta
    hp_denom = max(len(blue_ids), 1) * MAX_HP
    h_by_tick = {
        t: sum(states[u].hp for u in blue_ids if u in states) / hp_denom
        for t, states in frames.items()
    }
    ticks_sorted = sorted(progress_by_tick)
    rtg = {ticks_sorted[-1]: 0.0}
    for a, b in zip(reversed(ticks_sorted[:-1]), reversed(ticks_sorted[1:])):
        rtg[a] = (
            (progress_by_tick[b] - progress_by_tick[a])
            + beta * (h_by_tick[b] - h_by_tick[a])
            + VALUE_GAMMA * rtg[b]
        )
    for pair in bridge.v_pairs:
        pair.label = rtg[min(pair.tick + 6, last)]

    print(
        f"{label} 종료: {outcome} progress={progress_by_tick[last]:.3f} "
        f"({time.time()-episode_started:.0f}s)",
        flush=True,
    )
    return EpisodeResult(
        frames=frames,
        raw_frames=raw_frames,
        executed_commands=bridge.executed_log,
        planned_commands=bridge.planned_log,
        v_pairs=bridge.v_pairs,
        outcome=outcome,
        final_progress=progress_by_tick[last],
        progress_by_tick=progress_by_tick,
    )
