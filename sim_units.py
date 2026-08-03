"""기존 Soldier/World에 LOS(장애물 차폐)만 얹은 서브클래스.

원본 파일은 수정하지 않는다. 규칙:
- 관측: 벽 뒤의 개체는 보이지 않는다.
- 사격: 사수→표적 선분이 벽과 교차하면 탄이 흡수되어 피해가 없다.
둘 다 terrain.has_los 하나를 쓴다.
"""
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.combat_config import damage_for_distance, hit_probability  # noqa: E402
from hackerthon.Soldier import SoldierAtomic  # noqa: E402
from hackerthon.worldModel import WorldAtomic  # noqa: E402

from hackerthon.terrain import has_los  # noqa: E402


class LosSoldierAtomic(SoldierAtomic):
    """장애물이 시야를 가리는 Soldier.

    ``turn_to_damage``가 켜진 유닛은 피격 메시지의 사수 좌표를 이용해
    피격 방향으로 즉시 회전한다. 회전 반응은 이전 tick의 명령보다
    우선하며 이동은 발생하지 않는다.
    """

    def __init__(self, *args, obstacles=None, turn_to_damage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = obstacles or []
        self.turn_to_damage = bool(turn_to_damage)
        self._pending_damage_heading = None

    def extTransition(self, inputs):
        self._pending_damage_heading = None
        state = super().extTransition(inputs)
        if (
            self.turn_to_damage
            and self._pending_damage_heading is not None
            and self.state.hp > 0
        ):
            self.state.heading = self._pending_damage_heading
            self.state.mode = "TURN"
            self.state.target_id = None
            self.state.active_command = None
            self.state.command_time_remaining = 0.0
        return state

    def _handle_damage(self, messages):
        if self.turn_to_damage:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if msg.get("target_id") != self.state.soldier_id:
                    continue
                if msg.get("source_x") is None or msg.get("source_y") is None:
                    continue
                dx = float(msg["source_x"]) - self.state.x
                dy = float(msg["source_y"]) - self.state.y
                if abs(dx) + abs(dy) > 1e-9:
                    self._pending_damage_heading = (math.degrees(math.atan2(dy, dx)) + 180.0) % 360.0 - 180.0
        super()._handle_damage(messages)

    def _perceive(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        visible = super()._perceive(entities)
        me = (self.state.x, self.state.y)
        return [
            entity
            for entity in visible
            if has_los(me, (float(entity["x"]), float(entity["y"])), self.obstacles)
        ]


class LosWorldAtomic(WorldAtomic):
    """장애물이 탄을 흡수하는 World. 피해 판정 직전에 LOS를 검사한다."""

    def __init__(self, *args, obstacles=None, expected_damage=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacles = obstacles or []
        # rollout 평가용 모드: 명중 주사위 대신 기대 피해를 결정론적으로 적용한다.
        # 실제 episode는 False(Bernoulli 명중)를 유지해야 한다.
        self.expected_damage = bool(expected_damage)

    def extTransition(self, inputs):
        if self.status_in in inputs:
            all_statuses = list(inputs[self.status_in])

            for status in all_statuses:
                self._update_entity_from_status(status)

            for status in all_statuses:
                if status.get("mode") != "ENGAGE" or status.get("target_id") is None:
                    continue

                target = next(
                    (e for e in self.entities if e.get("id") == status["target_id"]),
                    None,
                )
                if target is None or target.get("state") == "DESTROYED":
                    continue
                if target.get("hp") is not None and float(target.get("hp", 0)) <= 0:
                    continue

                shooter_pos = (float(status.get("x", 0.0)), float(status.get("y", 0.0)))
                target_pos = (float(target.get("x", 0.0)), float(target.get("y", 0.0)))

                # 장애물 차폐: 탄은 벽에 흡수된다.
                if not has_los(shooter_pos, target_pos, self.obstacles):
                    continue

                distance = math.hypot(
                    target_pos[0] - shooter_pos[0],
                    target_pos[1] - shooter_pos[1],
                )
                if self.expected_damage:
                    damage = hit_probability(distance) * float(damage_for_distance(distance))
                    if damage <= 0.0:
                        continue
                    target["hp"] = max(0.0, float(target.get("hp", 100)) - damage)
                else:
                    if random.random() > hit_probability(distance):
                        continue
                    damage = damage_for_distance(distance)
                    if damage <= 0:
                        continue
                    target["hp"] = max(0, int(target.get("hp", 100)) - damage)
                if target["hp"] <= 0:
                    target["state"] = "DESTROYED"
                self.pending_damages.append({
                    "target_id": status["target_id"],
                    "damage": damage,
                    "source_id": status.get("id"),
                    "source_x": shooter_pos[0],
                    "source_y": shooter_pos[1],
                })

        return self.state
