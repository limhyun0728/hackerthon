import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from hackerthon.combat_config import PERCEPTION_RANGE, SHOOTING_RANGE


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180) % 360 - 180


@dataclass
class SoldierState:
    soldier_id: int = 101
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    hp: int = 100
    ammo: int = 30
    mode: str = "IDLE"
    target_id: Optional[int] = None
    active: bool = True
    visible_entities: List[Dict[str, Any]] = field(default_factory=list)
    # commander sequence에서 내려온 현재 명령과 남은 실행 시간을 저장한다.
    active_command: Optional[Dict[str, Any]] = None
    command_time_remaining: float = 0.0


class SoldierAtomic(AtomicDEVS):
    def __init__(
        self,
        name: str = "SoldierAtomic",
        soldier_id: int = 101,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_heading: float = 0.0,
        hp: int = 100,
        ammo: int = 30,
        perception_range: float = PERCEPTION_RANGE,
        shooting_range: float = SHOOTING_RANGE,
        fov_deg: float = 360.0,
        max_move_per_step: float = 1.0,
        active: bool = True,
    ):
        super().__init__(name)
        self.state = SoldierState(
            soldier_id=soldier_id,
            x=initial_x,
            y=initial_y,
            heading=initial_heading,
            hp=hp,
            ammo=ammo,
            active=active,
            mode="IDLE" if active else "INACTIVE",
        )
        self.perception_range = perception_range
        self.shooting_range = shooting_range
        self.fov_deg = fov_deg
        self.max_move_per_step = max_move_per_step
        self.current_time = 0.0

        self.world_in = self.addInPort("world_in")
        self.command_in = self.addInPort("command_in")
        self.damage_in = self.addInPort("damage_in")
        self.spawn_in = self.addInPort("spawn_in")

        self.observation_out = self.addOutPort("observation_out")
        self.status_out = self.addOutPort("status_out")

    def timeAdvance(self):
        if not self.state.active:
            return INFINITY
        return 1.0

    def extTransition(self, inputs):
        self.current_time += self.elapsed

        if self.spawn_in in inputs:
            self._handle_spawn(inputs[self.spawn_in])

        if not self.state.active:
            return self.state

        if self.damage_in in inputs:
            self._handle_damage(inputs[self.damage_in])

        if self.world_in in inputs:
            entities = self._parse_world_inputs(inputs[self.world_in])
            self.state.visible_entities = self._perceive(entities)

        if self.command_in in inputs:
            self._handle_command(inputs[self.command_in])

        if self.state.hp <= 0:
            self.state.hp = 0
            self.state.mode = "DESTROYED"
            self.state.target_id = None
            # 수정: 외부 전이에서 사망한 경우에도 남은 시퀀스 명령을 모두 폐기한다.
            self.state.active_command = None
            self.state.command_time_remaining = 0.0

        return self.state

    def intTransition(self):
        self.current_time += 1.0

        if not self.state.active:
            return self.state

        if self.state.hp <= 0:
            self.state.hp = 0
            self.state.mode = "DESTROYED"
            self.state.target_id = None
            # 수정: 사망한 병사는 남은 시퀀스 명령을 모두 폐기한다.
            self.state.active_command = None
            self.state.command_time_remaining = 0.0
            return self.state

        # 수정: commander의 5초 시퀀스 명령은 한 tick만 실행하지 않고 남은 시간 동안 계속 실행한다.
        if self.state.active_command is not None and self.state.command_time_remaining > 0.0:
            cmd = self.state.active_command
            action = str(cmd.get("action", "STOP")).upper()

            if action == "MOVE":
                self._move(cmd)
            elif action == "ENGAGE":
                self._engage(cmd)
            elif action == "TURN":
                # 수정: TURN은 상대각을 반복 적용하면 계속 회전하므로, 자세 유지 상태만 표시한다.
                self.state.mode = "TURN"
                self.state.target_id = None
            else:
                self.state.mode = "IDLE"
                self.state.target_id = None

            # 수정: 이번 internal tick에서 1초 분량을 실행했으므로 남은 지속 시간을 차감한다.
            self.state.command_time_remaining = max(0.0, self.state.command_time_remaining - 1.0)

            if self.state.command_time_remaining > 0.0 and self.state.mode != "IDLE":
                return self.state

            # 수정: 시퀀스 명령 시간이 끝나면 다음 명령을 기다리도록 실행 명령을 비운다.
            self.state.active_command = None

        if self.state.mode in ["MOVE", "TURN", "ENGAGE"]:
            self.state.mode = "IDLE"
            self.state.target_id = None

        return self.state

    def outputFnc(self):
        if not self.state.active:
            return {}

        observation = {
            "time": round(self.current_time, 2),
            "self": {
                "id": self.state.soldier_id,
                "x": round(self.state.x, 3),
                "y": round(self.state.y, 3),
                "hp": self.state.hp,
                "ammo": self.state.ammo,
                "heading": round(self.state.heading, 2),
                "mode": self.state.mode,
            },
            "visible_entities": self.state.visible_entities,
        }

        status = {
            "time": round(self.current_time, 2),
            "id": self.state.soldier_id,
            "x": round(self.state.x, 3),
            "y": round(self.state.y, 3),
            "heading": round(self.state.heading, 2),
            "hp": self.state.hp,
            "ammo": self.state.ammo,
            "mode": self.state.mode,
            "target_id": self.state.target_id,
        }

        return {
            self.observation_out: [observation],
            self.status_out: [status],
        }

    def _handle_spawn(self, messages):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("id") != self.state.soldier_id:
                continue

            self.state.x = float(msg.get("x", self.state.x))
            self.state.y = float(msg.get("y", self.state.y))
            self.state.heading = float(msg.get("heading", self.state.heading))
            self.state.hp = int(msg.get("hp", self.state.hp))
            self.state.ammo = int(msg.get("ammo", self.state.ammo))
            self.state.mode = "IDLE"
            self.state.target_id = None
            self.state.visible_entities = []
            self.state.active = True
            # 수정: 재스폰 시 이전 시퀀스 명령이 이어지지 않도록 초기화한다.
            self.state.active_command = None
            self.state.command_time_remaining = 0.0

    def _parse_world_inputs(self, messages):
        entities = []
        for msg in messages:
            if isinstance(msg, dict) and "entities" in msg:
                entities.extend(msg["entities"])
            elif isinstance(msg, dict):
                entities.append(msg)
            elif isinstance(msg, list):
                entities.extend(msg)
        return entities

    def _handle_damage(self, messages):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("target_id") != self.state.soldier_id:
                continue

            # damage가 int면 hp도 int로 유지되고(실제 episode), 기대피해 rollout의
            # float damage는 소수 그대로 반영돼야 하므로 int 절단을 하지 않는다.
            self.state.hp -= msg.get("damage", 0)
            if self.state.hp <= 0:
                self.state.hp = 0
                self.state.mode = "DESTROYED"
                self.state.target_id = None
                # 수정: 피해 처리 중 사망하면 진행 중인 시퀀스 명령을 중단한다.
                self.state.active_command = None
                self.state.command_time_remaining = 0.0

    def _handle_command(self, messages):
        if self.state.mode == "DESTROYED":
            return

        for cmd in messages:
            if not isinstance(cmd, dict):
                continue
            if cmd.get("unit_id", self.state.soldier_id) != self.state.soldier_id:
                continue

            # 수정: 새 명령을 현재 실행 명령으로 저장해서 intTransition에서도 같은 시퀀스를 이어간다.
            self.state.active_command = dict(cmd)
            try:
                duration_sec = float(cmd.get("duration_sec", 1.0))
            except (TypeError, ValueError):
                duration_sec = 1.0
            # 수정: 명령 수신 즉시 한 번 실행하므로, internal tick에서 실행할 남은 시간만 저장한다.
            self.state.command_time_remaining = max(0.0, duration_sec - 1.0)

            action = str(cmd.get("action", "STOP")).upper()

            if action == "MOVE":
                self._move(cmd)
            elif action == "TURN":
                self._turn_relative(cmd)
            elif action == "ENGAGE":
                self._engage(cmd)
            else:
                self.state.mode = "IDLE"
                self.state.target_id = None

    def _is_enemy_target(self, target_id: int) -> bool:
        self_id = self.state.soldier_id

        if 100 <= self_id < 200:
            return 200 <= target_id < 300

        if 200 <= self_id < 300:
            return 100 <= target_id < 200

        return False

    def _move(self, cmd: Dict[str, Any]):
        # New commander format: absolute waypoint coordinates.
        # Legacy rule policies may still send r/theta, so keep backward compatibility.
        if cmd.get("x") is not None and cmd.get("y") is not None:
            self._move_to_waypoint(cmd)
        else:
            self._move_relative(cmd)

    def _move_to_waypoint(self, cmd: Dict[str, Any]):
        try:
            target_x = float(cmd.get("x"))
            target_y = float(cmd.get("y"))
        except (TypeError, ValueError):
            self.state.mode = "IDLE"
            self.state.target_id = None
            return

        dx = target_x - self.state.x
        dy = target_y - self.state.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance <= 0.05:
            self.state.mode = "IDLE"
            self.state.target_id = None
            return

        # 절대 waypoint로 이동할 때 이동 방향을 자동으로 바라본다.
        self.state.heading = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        move_r = min(distance, self.max_move_per_step)
        self.state.x += move_r * dx / distance
        self.state.y += move_r * dy / distance
        self.state.mode = "MOVE"
        self.state.target_id = None

    def _move_relative(self, cmd: Dict[str, Any]):
        # Legacy format for rule-based policies: r + theta relative to current heading.
        move_r = max(0.0, min(float(cmd.get("r", 0.0)), self.max_move_per_step))
        move_theta = float(cmd.get("theta", 0.0))
        absolute_theta = normalize_angle_deg(self.state.heading + move_theta)

        self.state.x += move_r * math.cos(math.radians(absolute_theta))
        self.state.y += move_r * math.sin(math.radians(absolute_theta))
        # 상대 이동 명령도 실제 이동 방향으로 heading을 자동 갱신한다.
        self.state.heading = absolute_theta
        self.state.mode = "MOVE"
        self.state.target_id = None

    def _turn_relative(self, cmd: Dict[str, Any]):
        delta_theta = float(cmd.get("theta", 0.0))
        self.state.heading = normalize_angle_deg(self.state.heading + delta_theta)
        self.state.mode = "TURN"
        self.state.target_id = None

    def _engage(self, cmd: Dict[str, Any]):
        target_id = cmd.get("target_id")

        if target_id is None:
            self.state.mode = "IDLE"
            self.state.target_id = None
            return

        target_id = int(target_id)

        if not self._is_enemy_target(target_id):
            self.state.mode = "IDLE"
            self.state.target_id = None
            return

        if self.state.ammo <= 0:
            self.state.mode = "IDLE"
            self.state.target_id = None
            return

        # 직접 관측한 표적을 교전할 때 표적 방향을 자동으로 바라본다.
        target = next(
            (
                entity
                for entity in self.state.visible_entities
                if int(entity.get("id", -1)) == target_id
                and entity.get("state") != "DESTROYED"
                and float(entity.get("hp", 1)) > 0.0
            ),
            None,
        )
        if target is not None:
            self.state.heading = normalize_angle_deg(
                self.state.heading + float(target.get("theta", 0.0))
            )

        self.state.ammo -= 1
        self.state.target_id = target_id
        self.state.mode = "ENGAGE"

    def _perceive(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        visible = []
        sx, sy, heading = self.state.x, self.state.y, self.state.heading

        for entity in entities:
            if entity.get("id") == self.state.soldier_id:
                continue
            if "x" not in entity or "y" not in entity:
                continue

            is_destroyed = (
                entity.get("state") == "DESTROYED"
                or float(entity.get("hp", 1)) <= 0.0
            )
            if is_destroyed:
                # 적 사망을 지휘관에게 확인시킬 수 있도록 파괴된 적만 관측에 남긴다.
                # 파괴된 아군은 기존과 같이 주변 유닛의 visible_entities에서 제외한다.
                try:
                    entity_id = int(entity.get("id"))
                except (TypeError, ValueError):
                    continue
                if not self._is_enemy_target(entity_id):
                    continue

            dx = float(entity["x"]) - sx
            dy = float(entity["y"]) - sy
            r = math.sqrt(dx**2 + dy**2)
            if r > self.perception_range:
                continue

            absolute_angle = math.degrees(math.atan2(dy, dx))
            relative_theta = normalize_angle_deg(absolute_angle - heading)
            if abs(relative_theta) > self.fov_deg / 2:
                continue

            visible.append({
                "id": entity.get("id"),
                "type": entity.get("type", "unknown"),
                # 관측 순간 World가 제공한 절대 좌표를 그대로 보존한다.
                # 이후 MOVE/ENGAGE로 관측자의 위치·heading이 바뀌어도 이 좌표는 변하지 않는다.
                "x": round(float(entity["x"]), 3),
                "y": round(float(entity["y"]), 3),
                "hp": entity.get("hp", None),
                "heading": entity.get("heading", None),
                "r": round(r, 3),
                "theta": round(relative_theta, 2),
                "state": entity.get("state", "UNKNOWN"),
            })

        visible.sort(key=lambda e: e["r"])
        return visible
