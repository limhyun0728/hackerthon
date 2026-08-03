"""2계층(이벤트 기반 LLM 플래너 + 결정론 실행기) 5대5 장애물 시나리오.

기존 코드는 수정하지 않고 서브클래스/임포트로 재사용한다.
  python v2_two_layer/simulation_two_layer_5v5.py --planner scripted
  python v2_two_layer/simulation_two_layer_5v5.py --planner llm --model qwen2.5vl:72b
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pypdevs.DEVS import AtomicDEVS, CoupledDEVS  # noqa: E402
from pypdevs.infinity import INFINITY  # noqa: E402
from pypdevs.simulator import Simulator  # noqa: E402

from hackerthon.simulation_direct_commander_5v5 import CSVLoggerAtomic, RulePolicyAtomic  # noqa: E402

from hackerthon.executor import Executor  # noqa: E402
from hackerthon.planner import LLMPlanner, ScriptedPlanner  # noqa: E402
from hackerthon.red_policy import UrbanRedPolicy  # noqa: E402
from hackerthon.sim_units import LosSoldierAtomic, LosWorldAtomic  # noqa: E402
from hackerthon.terrain import DEFAULT_OBSTACLES  # noqa: E402


def _load_obstacle_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    obstacles = tuple(
        tuple(float(value) for value in rect)
        for rect in config["obstacles"]
    )
    return config, obstacles


class TwoLayerCommanderAtomic(AtomicDEVS):
    """매 틱 실행기를 돌리고, 이벤트가 있을 때만 플래너를 호출하는 지휘부."""

    def __init__(self, name, controlled_ids, planner, obstacles, run_dir, dt=1.0):
        super().__init__(name)
        self.controlled_ids = list(controlled_ids)
        self.planner = planner
        self.executor = Executor(controlled_ids, obstacles)
        self.dt = dt
        self.current_plan = None

        self.intel_in = self.addInPort("intel_in")
        self.orders_out = {uid: self.addOutPort(f"orders_out_{uid}") for uid in self.controlled_ids}
        self.intel_buffer = {}
        self.sigma = INFINITY
        self.command_time = 0.0
        self.state = "IDLE"

        self._cmd_log = open(os.path.join(run_dir, "commands_log.csv"), "w", newline="", encoding="utf-8")
        self._cmd_writer = csv.DictWriter(
            self._cmd_log, fieldnames=["time", "unit_id", "role", "action", "detail", "reason"]
        )
        self._cmd_writer.writeheader()
        self._plan_log = open(os.path.join(run_dir, "planner_log.csv"), "w", newline="", encoding="utf-8")
        self._plan_writer = csv.DictWriter(
            self._plan_log,
            fieldnames=["time", "events", "decision", "tactic", "assignments", "flank_side", "focus", "latency_s", "rationale"],
        )
        self._plan_writer.writeheader()
        self._event_log = open(os.path.join(run_dir, "events_log.csv"), "w", newline="", encoding="utf-8")
        self._event_writer = csv.DictWriter(self._event_log, fieldnames=["time", "type", "detail"])
        self._event_writer.writeheader()

    def timeAdvance(self):
        return self.sigma

    def extTransition(self, inputs):
        if self.sigma != INFINITY:
            self.sigma = max(0.0, self.sigma - self.elapsed)
        if self.intel_in in inputs:
            for obs in inputs[self.intel_in]:
                uid = obs.get("self", {}).get("id")
                if uid in self.controlled_ids:
                    self.intel_buffer[uid] = obs
        if self.intel_buffer and self.sigma == INFINITY:
            self.sigma = self.dt
        return self.state

    def intTransition(self):
        self.command_time += self.dt
        self.sigma = self.dt if self.intel_buffer else INFINITY
        return self.state

    def outputFnc(self):
        if not self.intel_buffer:
            return {}
        tick = round(self.command_time + self.dt, 3)

        # 사망 유닛을 먼저 제거하면 hp=0 전이를 볼 수 없어
        # FRIENDLY_KIA가 사라진다. 이벤트/신념 갱신은 전체 관측으로 한다.
        all_intel = dict(self.intel_buffer)
        self.executor.update_belief(all_intel, tick)
        events = self.executor.detect_events(all_intel, tick)
        for event in events:
            detail = json.dumps({k: v for k, v in event.items() if k != "type"}, ensure_ascii=False)
            self._event_writer.writerow({"time": tick, "type": event["type"], "detail": detail})
        self._event_log.flush()

        if not self.executor.friendlies:
            return {}

        if self.current_plan is None or events:
            context = self._planner_context(tick, events)
            decision = self.planner.decide(context)
            if decision["decision"] == "REVISE":
                self.current_plan = {
                    "tactic": decision["tactic"],
                    "assignments": decision["assignments"],
                    "flank_side": decision.get("flank_side"),
                    "focus_target_id": decision.get("focus_target_id"),
                }
                self.executor.set_plan(self.current_plan)
            self._plan_writer.writerow({
                "time": tick,
                "events": ";".join(e["type"] for e in events) or "INITIAL",
                "decision": decision["decision"],
                "tactic": self.current_plan["tactic"],
                "assignments": " | ".join(
                    f"{a['unit_id']}:{a['role']}" for a in self.current_plan["assignments"]
                ),
                "flank_side": self.executor.plan.get("flank_side"),
                "focus": self.executor.plan.get("focus_target"),
                "latency_s": decision.get("latency_s", 0.0),
                "rationale": decision.get("rationale", ""),
            })
            self._plan_log.flush()
            print(f"[Planner t={tick}] {decision['decision']} tactic={self.current_plan['tactic']} "
                  f"events={[e['type'] for e in events]}")

        commands = self.executor.commands(tick)
        roles = self.executor.plan.get("roles", {})
        for cmd in commands:
            uid = cmd["unit_id"]
            detail = (
                f"({cmd['x']},{cmd['y']})" if cmd["action"] == "MOVE"
                else (f"->R{cmd['target_id']}" if cmd["action"] == "ENGAGE" else "")
            )
            self._cmd_writer.writerow({
                "time": tick, "unit_id": uid, "role": roles.get(uid, "?"),
                "action": cmd["action"], "detail": detail, "reason": cmd["reason"],
            })
        self._cmd_log.flush()

        out = {}
        for cmd in commands:
            port = self.orders_out.get(cmd["unit_id"])
            if port is not None:
                out[port] = [cmd]
        return out

    def _planner_context(self, tick, events):
        roles = self.executor.plan.get("roles", {})
        blue_summary = [
            {
                "id": uid,
                "pos": [round(f["x"], 1), round(f["y"], 1)],
                "hp": f["hp"],
                "ammo": f["ammo"],
                "role": roles.get(uid, "-"),
            }
            for uid, f in sorted(self.executor.friendlies.items())
        ]
        red_summary = [
            {"id": eid, "pos": [round(e["x"], 1), round(e["y"], 1)], "hp": e["hp"], "last_seen": e["last_seen"]}
            for eid, e in sorted(self.executor.enemies.items())
        ]
        return {
            "tick": tick,
            "events": events,
            "living_blue": sorted(self.executor.friendlies),
            "known_enemies": {eid: e for eid, e in self.executor.enemies.items()},
            "destroyed_count": len(self.executor._known_dead),
            "blue_summary": blue_summary,
            "red_summary": red_summary,
            "current_plan": self.current_plan,
            "obstacles": [list(rect) for rect in self.executor.obstacles],
        }


class TwoLayerBattleModel(CoupledDEVS):
    def __init__(self, planner, obstacles, run_dir, seed=42, name="TwoLayerBattleModel"):
        super().__init__(name)
        random.seed(seed)

        # 양 편의 개방된 진입부에서 시작해 중앙 시가지로 진입한다.
        # 각 유닛을 서로 다른 동서 축에 배치해 여러 골목을 사용하게 한다.
        lanes = (-6.0, -3.0, 0.0, 3.0, 6.0)
        blue_positions = [(101 + i, -8.0, y, 0.0) for i, y in enumerate(lanes)]
        red_positions = [(201 + i, 9.0, y, 180.0) for i, y in enumerate(lanes)]

        initial_entities = [
            {"id": sid, "type": "soldier", "x": x, "y": y, "heading": h, "state": "ALIVE"}
            for sid, x, y, h in blue_positions
        ] + [
            {"id": sid, "type": "enemy", "x": x, "y": y, "heading": h, "state": "ALIVE"}
            for sid, x, y, h in red_positions
        ]

        self.world = self.addSubModel(LosWorldAtomic(initial_entities=initial_entities, obstacles=obstacles))
        self.logger = self.addSubModel(CSVLoggerAtomic(os.path.join(run_dir, "soldier_log.csv")))

        blue_ids = [sid for sid, *_ in blue_positions]
        commander = self.addSubModel(
            TwoLayerCommanderAtomic(
                name="Blue_TwoLayer_Commander",
                controlled_ids=blue_ids,
                planner=planner,
                obstacles=obstacles,
                run_dir=run_dir,
            )
        )

        for sid, x, y, h in blue_positions:
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=f"Blue_{sid}", soldier_id=sid, initial_x=x, initial_y=y,
                    initial_heading=h, max_move_per_step=1.5, fov_deg=120.0,
                    obstacles=obstacles,
                )
            )
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.logger.status_in)
            self.connectPorts(soldier.observation_out, commander.intel_in)
            self.connectPorts(commander.orders_out[sid], soldier.command_in)

        for sid, x, y, h in red_positions:
            soldier = self.addSubModel(
                LosSoldierAtomic(
                    name=f"Red_{sid}", soldier_id=sid, initial_x=x, initial_y=y,
                    initial_heading=h, fov_deg=120.0, obstacles=obstacles,
                    turn_to_damage=True,
                )
            )
            brain = self.addSubModel(
                RulePolicyAtomic(
                    name=f"Red_Rule_{sid}",
                    policy=UrbanRedPolicy(target_type="soldier", obstacles=obstacles),
                    decision_delay=1.0,
                )
            )
            self.connectPorts(soldier.observation_out, brain.observation_in)
            self.connectPorts(brain.command_out, soldier.command_in)
            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(self.world.spawn_out, soldier.spawn_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.logger.status_in)


def main():
    parser = argparse.ArgumentParser(description="2계층 지휘 구조 + 장애물 5대5")
    parser.add_argument("--planner", choices=["scripted", "llm"], default="scripted")
    parser.add_argument("--model", default="qwen2.5vl:72b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--terrain",
        choices=["urban", "open"],
        default="urban",
        help="urban은 DEFAULT_OBSTACLES 시가지, open은 장애물 없는 개활지 (CEM 실험과 동일 조건)",
    )
    parser.add_argument(
        "--obstacle-config",
        default=None,
        help="config.json의 obstacles를 실제 장애물 지형으로 사용한다. 지정하면 --terrain보다 우선한다.",
    )
    args = parser.parse_args()

    run_dir = os.path.join(
        "output", f"v2_{args.planner}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(run_dir, exist_ok=False)

    planner = (
        LLMPlanner(model=args.model, temperature=args.temperature)
        if args.planner == "llm"
        else ScriptedPlanner()
    )
    obstacle_config = {}
    if args.obstacle_config:
        obstacle_config, obstacles = _load_obstacle_config(args.obstacle_config)
    else:
        obstacles = () if args.terrain == "open" else DEFAULT_OBSTACLES

    config = {
        "architecture": "two_layer(event_planner + rule_executor)",
        "planner": planner.name,
        "obstacles": [list(rect) for rect in obstacles],
        "seed": args.seed,
        "duration": args.duration,
        "blue_fov_deg": 120.0,
        "red_fov_deg": 120.0,
        "red_policy": "urban search/pursue + obstacle routing + turn toward incoming fire",
        "bullet_model": "hitscan + LOS absorption (walls block move/sight/fire)",
    }
    if obstacle_config:
        config["obstacle_config"] = str(Path(args.obstacle_config))
        for key in ("building_polygons", "real_map"):
            if key in obstacle_config:
                config[key] = obstacle_config[key]
    Path(run_dir, "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[v2] planner={planner.name} run_dir={run_dir}")

    started = time.time()
    model = TwoLayerBattleModel(planner=planner, obstacles=obstacles, run_dir=run_dir, seed=args.seed)
    sim = Simulator(model)
    sim.setTerminationTime(args.duration)
    sim.simulate()
    print(f"[v2] done in {time.time() - started:.1f}s wall clock. run_dir={run_dir}")


if __name__ == "__main__":
    main()
