import csv
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
import random

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdevs.DEVS import CoupledDEVS, AtomicDEVS
from pypdevs.simulator import Simulator
from pypdevs.infinity import INFINITY

from hackerthon.Soldier import SoldierAtomic
from hackerthon.worldModel import WorldAtomic
from hackerthon.ruleAdapter import RuleSoldierPolicy
from hackerthon.commanderAdapter import CommanderPolicy
from hackerthon.commander_helpers import sequence_frame_to_actions


class CSVLoggerAtomic(AtomicDEVS):
    def __init__(self, filename):
        super().__init__("CSVLogger")
        self.status_in = self.addInPort("status_in")
        self.file = open(filename, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["time", "id", "x", "y", "heading", "hp", "ammo", "mode", "target_id"],
        )
        self.writer.writeheader()

    def extTransition(self, inputs):
        if self.status_in in inputs:
            for status in inputs[self.status_in]:
                self.writer.writerow(status)
                self.file.flush()
        return "LOGGING"

    def timeAdvance(self):
        return INFINITY

    def intTransition(self):
        return "LOGGING"


class CommanderAtomic(AtomicDEVS):
    def __init__(
        self,
        name="Commander",
        controlled_ids=None,
        policy=None,
        dt=1.0,
        run_dir=".",
    ):
        super().__init__(name)

        self.memory = deque(maxlen=20)
        self.prev_snapshot = None
        self.policy = policy or CommanderPolicy()
        self.dt = dt

        self.controlled_ids = controlled_ids or []
        self.intel_in = self.addInPort("intel_in")
        self.orders_out = {
            uid: self.addOutPort(f"orders_out_{uid}")
            for uid in self.controlled_ids
        }

        self.intel_buffer = {}
        self.state = "IDLE"
        self.sigma = INFINITY
        self.command_time = 0.0

        # 매 tick 새 판단에서 만들어진 단일 frame 명령을 실행한다.
        self.active_decision = None
        self.active_sequence = None
        self.active_frames = []
        self.active_frame_index = 0
        self.sequence_id = 0

        # FRIENDLY_KIA 이벤트로 확인된 사망 유닛 누적. stale intel 때문에 hp=0이 늦게 반영되므로
        # 이벤트 감지 즉시 여기에 추가해 patched_intel에서 제거한다.
        self._known_dead: set = set()

        # intel_buffer 위치는 1틱 stale → MOVE 명령 후 병사가 이미 그 위치에 있어서
        # 다음 틱 waypoint가 distance=0 → IDLE 반복.
        # 가장 최근 MOVE 명령의 목적지를 추적해 다음 waypoint 계산 때 anticipated 위치로 사용.
        self._anticipated_positions: dict = {}  # {uid: (x, y)}

        self.command_log_file = open(os.path.join(run_dir, "commander_sequence_log.csv"), "w", newline="", encoding="utf-8")
        self.command_writer = csv.DictWriter(
            self.command_log_file,
            fieldnames=[
                "time",
                "event",
                "intent",
                "sequence_id",
                "frame_index",
                "replan_reason",
                "map_image_path",
                "sequence_json",
                "commands_json",
                "events_json",
                "memory_json",
            ],
        )
        self.command_writer.writeheader()

        self.commands_log_file = open(os.path.join(run_dir, "commander_commands_log.csv"), "w", newline="", encoding="utf-8")
        self.commands_writer = csv.DictWriter(
            self.commands_log_file,
            fieldnames=["time", "phase", "unit_id", "command", "replan"],
        )
        self.commands_writer.writeheader()

        self.belief_log_file = open(os.path.join(run_dir, "commander_belief_log.csv"), "w", newline="", encoding="utf-8")
        self.belief_writer = csv.DictWriter(
            self.belief_log_file,
            fieldnames=[
                "time",
                "entity_id",
                "kind",
                "status",
                "x",
                "y",
                "bearing_deg",
                "bucket",
                "observed_this_tick",
                "last_seen_tick",
                "mode",
                "target_id",
            ],
        )
        self.belief_writer.writeheader()

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

    def outputFnc(self):
        """매 tick의 지휘 판단을 실행 명령으로 내보낸다."""
        if not self.intel_buffer:
            return {}

        sim_time = round(self.command_time + self.dt, 3)

        current_snapshot = self._make_snapshot(self.intel_buffer)
        events = self._detect_events(sim_time, current_snapshot)

        # FRIENDLY_KIA 이벤트에서 사망 유닛 즉시 등록
        for evt in events:
            if evt.get("type") == "FRIENDLY_KIA":
                detail = evt.get("detail", "")
                for part in detail.split():
                    if part.isdigit():
                        self._known_dead.add(int(part))
                        break

        memory_for_prompt = list(self.memory)
        if events:
            memory_for_prompt.append({
                "time": sim_time,
                "events": events,
            })

        patched_intel = self._patch_intel_with_anticipated(self.intel_buffer)
        # 사망 확인된 유닛은 intel에서 제거 → alive_ids / LLM이 명령 대상으로 포함하지 않음
        for dead_uid in self._known_dead:
            patched_intel.pop(dead_uid, None)

        # Stage 1·3·4는 매 tick 실행하고 Stage 2 갱신 여부는 policy가 판정한다.
        decision = self.policy.decide(
            patched_intel,
            memory_for_prompt,
            sim_time=sim_time,
            controlled_ids=self.controlled_ids,
        )
        self._activate_sequence(decision)
        self._log_belief_rows(decision["belief_rows"])

        plan_trigger = decision["plan"]["trigger"]
        replan_reason = ""
        if decision["tactical_plan_updated"]:
            if plan_trigger.startswith("FRIENDLY_KIA"):
                event_name = "KIA_REPLAN"
                replan_reason = plan_trigger
            elif plan_trigger.startswith("FRIENDLY_FIRST_DAMAGE"):
                event_name = "FIRST_DAMAGE_REPLAN"
                replan_reason = plan_trigger
            elif plan_trigger == "PLAN_INTERVAL":
                event_name = "SCHEDULED_PLAN_UPDATE"
            else:
                event_name = "INITIAL_DECISION"
            events.append({
                "time": sim_time,
                "type": event_name,
                "detail": plan_trigger,
            })
        else:
            event_name = "TICK_DECISION"

        frame_index = self.active_frame_index
        frame = self._current_frame()
        commands = sequence_frame_to_actions(
            patched_intel,
            frame,
            shooting_range=self.policy.shooting_range,
            max_move_per_step=self.policy.max_move_per_step,
            plan=decision.get("plan") if isinstance(decision, dict) else None,
            frame_tick=frame_index,
        )

        for cmd in commands:
            cmd["duration_sec"] = self.dt

        # anticipated_positions 갱신: MOVE 명령의 목적지가 다음 틱의 실제 위치
        for cmd in commands:
            uid = cmd.get("unit_id")
            if cmd.get("action") == "MOVE" and uid is not None:
                self._anticipated_positions[uid] = (
                    float(cmd.get("x", 0.0)),
                    float(cmd.get("y", 0.0)),
                )
            elif cmd.get("action") in ("ENGAGE", "STOP") and uid is not None:
                # 이동 없음 → anticipated를 현재 patched 위치로 유지
                info = (patched_intel.get(uid, {}) or {}).get("self", {}) or {}
                self._anticipated_positions[uid] = (
                    float(info.get("x", 0.0)),
                    float(info.get("y", 0.0)),
                )

        # 현재 활성 전술 이름을 최종 명령 로그의 phase로 기록한다.
        plan = decision["plan"]
        self._log_commands(
            sim_time,
            commands,
            replan=replan_reason,
            phase=plan["tactic"],
        )
        command_events = self._build_command_events(sim_time, commands, frame_index)
        events.extend(command_events)

        memory_record = {
            "time": sim_time,
            "events": events,
            "assessment": decision.get("assessment") if isinstance(decision, dict) else None,
            "plan": decision.get("plan") if isinstance(decision, dict) else None,
            "action_plan": decision.get("action_plan") if isinstance(decision, dict) else None,
            "marked_plan": decision.get("marked_plan") if isinstance(decision, dict) else None,
            "sequence": decision.get("sequence") if isinstance(decision, dict) else None,
            "sequence_id": self.sequence_id,
            "frame_index": frame_index,
            "commands": commands,
        }
        self.memory.append(memory_record)
        self.prev_snapshot = current_snapshot

        self._log_commander_event(
            sim_time=sim_time,
            event=event_name,
            commands=commands,
            events=events,
            decision=decision,
            frame_index=frame_index,
            replan_reason=replan_reason,
        )

        print("\n[Commander Commands]")
        print(f"event={event_name} | time={sim_time} | sequence_id={self.sequence_id} | frame={frame_index}")
        if replan_reason:
            print(f"replan_reason={replan_reason}")
        if decision["tactical_plan_updated"]:
            print("[Full Sequence]")
            for af in self.active_frames:
                tick = af.get("tick", "?")
                frame_cmds = []
                for c in (af.get("commands") or []):
                    uid = c.get("unit_id")
                    act = c.get("action")
                    tgt = c.get("target_id")
                    frame_cmds.append(f"{uid}:{act}{'→'+str(tgt) if tgt else ''}")
                print(f"  tick={tick}: {' | '.join(frame_cmds)}")
        print(f"[Frame {frame_index}] {json.dumps(commands, ensure_ascii=False)}")

        # Advance the sequence cursor after emitting the selected frame.
        self.active_frame_index += 1

        out_dict = {}
        for cmd in commands:
            uid = cmd.get("unit_id")
            if uid in self.orders_out:
                out_dict[self.orders_out[uid]] = [cmd]

        return out_dict

    def intTransition(self):
        self.command_time += self.dt
        self.sigma = self.dt if self.intel_buffer else INFINITY
        return self.state

    def _patch_intel_with_anticipated(self, intel_buffer: dict) -> dict:
        """anticipated_positions를 반영한 intel_buffer 복사본을 반환한다.

        intel_buffer 위치는 1틱 stale이라 이미 이동한 병사의 old 위치를 담고 있다.
        MOVE 명령 후 도달한 anticipated 위치를 cur_pos로 대체하면,
        sequence_frame_to_actions가 계산하는 waypoint가 그 다음 위치가 되어
        distance=0 IDLE 반복을 막는다.
        """
        if not self._anticipated_positions:
            return intel_buffer
        patched = {}
        for uid, obs in intel_buffer.items():
            if uid in self._anticipated_positions and obs and obs.get("self"):
                ax, ay = self._anticipated_positions[uid]
                obs_copy = dict(obs)
                obs_copy["self"] = dict(obs["self"])
                obs_copy["self"]["x"] = ax
                obs_copy["self"]["y"] = ay
                patched[uid] = obs_copy
            else:
                patched[uid] = obs
        return patched

    def _activate_sequence(self, decision):
        if not isinstance(decision, dict) or not decision.get("sequence"):
            raise ValueError("CommanderPolicy.decide must return a dict containing 'sequence'")

        sequence = decision["sequence"]
        frames = sequence.get("frames", [])
        if not frames:
            raise ValueError("CommanderPolicy returned an empty command sequence")

        self.sequence_id += 1
        self.active_decision = decision
        self.active_sequence = sequence
        self.active_frames = sorted(frames, key=lambda item: int(item.get("tick", 0)))
        self.active_frame_index = 0

    def _current_frame(self):
        if not self.active_frames:
            raise ValueError("No active command sequence frame available")
        index = min(self.active_frame_index, len(self.active_frames) - 1)
        return self.active_frames[index]

    def _log_commander_event(self, sim_time, event, commands, events, decision, frame_index, replan_reason):
        plan = decision.get("plan") if isinstance(decision, dict) else {}
        sequence = decision.get("sequence") if isinstance(decision, dict) else {}
        self.command_writer.writerow({
            "time": round(sim_time, 3),
            "event": event,
            "intent": (plan or {}).get("intent", ""),
            "sequence_id": self.sequence_id,
            "frame_index": frame_index,
            "replan_reason": replan_reason,
            "map_image_path": decision.get("map_image_path") if isinstance(decision, dict) else "",
            "sequence_json": json.dumps(sequence or {}, ensure_ascii=False),
            "commands_json": json.dumps(commands, ensure_ascii=False),
            "events_json": json.dumps(events, ensure_ascii=False),
            "memory_json": json.dumps(list(self.memory), ensure_ascii=False),
        })
        self.command_log_file.flush()

    def _log_commands(self, sim_time, commands, replan="", phase=""):
        """매 틱 실제 발령된 unit별 명령을 기록."""
        for cmd in commands:
            uid = cmd.get("unit_id")
            action = cmd.get("action", "STOP")
            target = cmd.get("target_id")
            command_str = f"{action}->{target}" if target is not None else action
            self.commands_writer.writerow({
                "time": round(sim_time, 3),
                "phase": phase,
                "unit_id": uid,
                "command": command_str,
                "replan": replan,
            })
        self.commands_log_file.flush()

    def _log_belief_rows(self, rows):
        # 수정: commanderMap.belief_rows()가 만든 행을 visualizer용 로그로 그대로 저장한다.
        for row in rows:
            self.belief_writer.writerow(row)
        self.belief_log_file.flush()

    def _make_snapshot(self, intel_buffer):
        friendly = {}
        visible_enemies = {}

        for uid, obs in intel_buffer.items():
            self_info = obs.get("self", {}) or {}
            friendly[int(uid)] = {
                "hp": int(self_info.get("hp", 0)),
                "ammo": int(self_info.get("ammo", 0)),
                "mode": self_info.get("mode", "UNKNOWN"),
                "x": self_info.get("x"),
                "y": self_info.get("y"),
            }

            for ent in obs.get("visible_entities", []) or []:
                if ent.get("type") != "enemy":
                    continue
                if ent.get("state") == "DESTROYED":
                    continue
                if float(ent.get("hp", 100) or 0) <= 0:
                    continue

                enemy_id = int(ent.get("id"))
                seen_by = visible_enemies.setdefault(enemy_id, {
                    "id": enemy_id,
                    "hp": ent.get("hp"),
                    "seen_by": set(),
                    "min_range": float(ent.get("r", 999.0)),
                })
                seen_by["seen_by"].add(int(uid))
                seen_by["min_range"] = min(
                    seen_by["min_range"],
                    float(ent.get("r", 999.0)),
                )

        for enemy in visible_enemies.values():
            enemy["seen_by"] = sorted(enemy["seen_by"])
            enemy["min_range"] = round(enemy["min_range"], 3)

        return {
            "friendly": friendly,
            "visible_enemies": visible_enemies,
        }

    def _detect_events(self, sim_time, current_snapshot):
        events = []
        previous = self.prev_snapshot

        current_enemies = current_snapshot.get("visible_enemies", {})
        current_friendlies = current_snapshot.get("friendly", {})

        if previous is None:
            for enemy_id, enemy in sorted(current_enemies.items()):
                events.append({
                    "time": sim_time,
                    "type": "INITIAL_CONTACT",
                    "detail": f"enemy {enemy_id} visible by units {enemy['seen_by']} at min_range={enemy['min_range']}",
                })
            return events

        previous_enemies = previous.get("visible_enemies", {})
        previous_friendlies = previous.get("friendly", {})

        for unit_id, current in sorted(current_friendlies.items()):
            prev = previous_friendlies.get(unit_id)
            if prev is None:
                continue

            prev_hp = int(prev.get("hp", 0))
            cur_hp = int(current.get("hp", 0))

            if prev_hp > 0 and cur_hp <= 0:
                events.append({
                    "time": sim_time,
                    "type": "FRIENDLY_KIA",
                    "detail": f"unit {unit_id} destroyed",
                })
            elif cur_hp < prev_hp:
                events.append({
                    "time": sim_time,
                    "type": "FRIENDLY_DAMAGED",
                    "detail": f"unit {unit_id} hp {prev_hp}->{cur_hp}",
                })

        for enemy_id, enemy in sorted(current_enemies.items()):
            if enemy_id not in previous_enemies:
                events.append({
                    "time": sim_time,
                    "type": "CONTACT_SPOTTED",
                    "detail": f"enemy {enemy_id} visible by units {enemy['seen_by']} at min_range={enemy['min_range']}",
                })

        for enemy_id in sorted(previous_enemies.keys() - current_enemies.keys()):
            events.append({
                "time": sim_time,
                "type": "CONTACT_LOST",
                "detail": f"enemy {enemy_id} no longer visible",
            })

        return events

    def _build_command_events(self, sim_time, commands, frame_index):
        events = []
        for cmd in commands:
            unit_id = cmd.get("unit_id", "UNKNOWN")
            action = cmd.get("action", "UNKNOWN")
            target_id = cmd.get("target_id")

            if target_id is None:
                detail = f"sequence_frame={frame_index} unit {unit_id} ordered {action}"
            else:
                detail = f"sequence_frame={frame_index} unit {unit_id} ordered {action} target {target_id}"

            events.append({
                "time": sim_time,
                "type": "COMMAND_ISSUED",
                "detail": detail,
            })

        return events


class RulePolicyAtomic(AtomicDEVS):
    def __init__(self, name="RulePolicyAtomic", policy=None, decision_delay=1.0):
        super().__init__(name)
        self.policy = policy
        self.decision_delay = decision_delay
        self.observation_in = self.addInPort("observation_in")
        self.command_out = self.addOutPort("command_out")
        self.pending_command = None
        self.state = "WAIT"

    def extTransition(self, inputs):
        if self.observation_in in inputs:
            observation = inputs[self.observation_in][-1]
            command = dict(self.policy.decide(observation))
            # RED rule 명령도 action token으로 학습해야 하므로 실제 출력 tick을 명령에 박아둔다.
            command["time"] = round(float(observation["time"]) + float(self.decision_delay), 2)
            command["duration_sec"] = float(command.get("duration_sec", 1.0))
            self.pending_command = command
            self.state = "READY"
        return self.state

    def timeAdvance(self):
        if self.state == "READY":
            return self.decision_delay
        return INFINITY

    def outputFnc(self):
        if self.pending_command is None:
            return {}
        return {self.command_out: [self.pending_command]}

    def intTransition(self):
        self.pending_command = None
        self.state = "WAIT"
        return self.state


class CommanderBattleModel(CoupledDEVS):
    def __init__(self, name="CommanderBattleModel"):
        super().__init__(name)
        random.seed(42)

        self.run_dir = os.path.join("output", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"[Run] output dir: {self.run_dir}")

        def make_random_line_positions(start_id, count, x_min, x_max, y_min, y_max, heading):
            return [
                (
                    start_id + i,
                    round(random.uniform(x_min, x_max), 2),
                    round(random.uniform(y_min, y_max), 2),
                    heading,
                )
                for i in range(count)
            ]
        
        def make_linear_line_positions_blue(start_id, count, x_min, x_max, y_min, y_max, heading):
            positions = []
            
            # count가 1개일 때는 시작점에 배치, 그 이상일 때는 균등 분할
            x_step = (x_max - x_min) / (count - 1) if count > 1 else 0
            y_step = (y_max - y_min) / (count - 1) if count > 1 else 0
            
            for i in range(count):
                x = x_min + (i * x_step)
                y = y_min + (i * y_step)
                
                positions.append((
                    start_id + i,
                    round(x, 2),
                    round(y, 2),
                    heading
                ))
                
            return positions

        blue_positions = make_linear_line_positions_blue(
            start_id=101,
            count=5,
            x_min=-5.0,
            x_max=-3.0,
            y_min=-5.0,
            y_max=5.0,
            heading=0.0,
        )

        wave1_red_positions = make_random_line_positions(
            start_id=201,
            count=5,
            x_min=5.0,
            x_max=8.0,
            y_min=-5.0,
            y_max=5.0,
            heading=180.0,
        )

        initial_entities = []

        for sid, x, y, heading in blue_positions:
            initial_entities.append({
                "id": sid,
                "type": "soldier",
                "x": x,
                "y": y,
                "heading": heading,
                "state": "ALIVE",
            })

        for sid, x, y, heading in wave1_red_positions:
            initial_entities.append({
                "id": sid,
                "type": "enemy",
                "x": x,
                "y": y,
                "heading": heading,
                "state": "ALIVE",
            })

        self.world = self.addSubModel(
            WorldAtomic(initial_entities=initial_entities)
        )
        self.logger = self.addSubModel(
            CSVLoggerAtomic(os.path.join(self.run_dir, "soldier_commander_log.csv"))
        )

        blue_ids = [sid for sid, _, _, _ in blue_positions]

        blue_commander = self.addSubModel(
            CommanderAtomic(
                name="Blue_Commander",
                controlled_ids=blue_ids,
                policy=CommanderPolicy(
                    io_dir=os.path.join(self.run_dir, "llm_io"),
                    map_dir=os.path.join(self.run_dir, "maps"),
                    run_log_dir=self.run_dir,
                ),
                dt=1.0,
                run_dir=self.run_dir,
            )
        )

        for b_id, x, y, heading in blue_positions:
            soldier = self.addSubModel(
                SoldierAtomic(
                    name=f"Blue_Soldier_{b_id}",
                    soldier_id=b_id,
                    initial_x=x,
                    initial_y=y,
                    initial_heading=heading,
                    max_move_per_step=blue_commander.policy.max_move_per_step,
                    fov_deg=120.0,
                )
            )

            self.connectPorts(self.world.world_out, soldier.world_in)
            self.connectPorts(self.world.damage_out, soldier.damage_in)
            self.connectPorts(soldier.status_out, self.world.status_in)
            self.connectPorts(soldier.status_out, self.logger.status_in)
            self.connectPorts(soldier.observation_out, blue_commander.intel_in)
            self.connectPorts(blue_commander.orders_out[b_id], soldier.command_in)

        for r_id, x, y, heading in wave1_red_positions:
            self._add_red_soldier(r_id, x, y, heading, active=True)

    def _add_red_soldier(self, r_id, x, y, heading, active=True):
        soldier = self.addSubModel(
            SoldierAtomic(
                name=f"Red_Soldier_{r_id}",
                soldier_id=r_id,
                initial_x=x,
                initial_y=y,
                initial_heading=heading,
                fov_deg=120.0,
                active=active,
            )
        )

        brain = self.addSubModel(
            RulePolicyAtomic(
                name=f"Red_Rule_{r_id}",
                policy=RuleSoldierPolicy(target_type="soldier"),
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


if __name__ == "__main__":
    print("지휘관 체제(C2) 좌우 대치 시나리오 시작: Blue 5명 vs Red 5명")
    model = CommanderBattleModel()
    sim = Simulator(model)
    sim.setTerminationTime(60)
    sim.simulate()
    print(f"시뮬레이션 완료. 결과 디렉토리: {model.run_dir}")
    print(f"  soldier_commander_log.csv, commander_sequence_log.csv, commander_commands_log.csv, commander_belief_log.csv")
    print(f"  llm_io/  maps/")
