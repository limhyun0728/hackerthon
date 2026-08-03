"""시가지 시나리오의 Red 섬멸 규칙 정책.

Red는 제자리에서 반격만 하는 표적이 아니다. Blue 섬멸을 임무로 삼고
시가지를 탐색하며, 보이는 표적이 사거리 밖이면 장애물을 우회해 추격한다.
표적을 놓치면 마지막 관측 지점까지 이동한 뒤 다시 탐색을 시작한다.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from hackerthon.combat_config import DAMAGE_PER_HIT, EFFECTIVE_FIRE_RANGE, MAX_FIRE_RANGE, damage_for_distance, hit_probability
from hackerthon.terrain import next_waypoint


SEARCH_LANES: Tuple[Tuple[float, float], ...] = (
    (-7.5, -6.0),
    (-7.5, -3.0),
    (-7.5, 0.0),
    (-7.5, 3.0),
    (-7.5, 6.0),
)

OPEN_SCREEN_LANES: Tuple[Tuple[float, float], ...] = (
    (4.0, -6.0),
    (4.0, -3.0),
    (4.0, 0.0),
    (4.0, 3.0),
    (4.0, 6.0),
)


class UrbanRedPolicy:
    """120도 FOV로 탐색·추격·사격하는 상태형 Red 정책."""

    def __init__(self, target_type: str = "soldier", obstacles=None, target_priority: str = "nearest"):
        if target_priority not in ("nearest", "low_hp", "smart"):
            raise ValueError("target_priority는 nearest, low_hp 또는 smart여야 한다")
        self.target_type = target_type
        self.obstacles = list(obstacles or [])
        self.target_priority = target_priority
        self.last_seen: Optional[Tuple[float, float]] = None
        self.search_index: Optional[int] = None

    @staticmethod
    def _stop(unit_id: int, reason: str) -> Dict[str, Any]:
        return {"unit_id": unit_id, "action": "STOP", "reason": reason}

    def _move(self, unit_id: int, me: Tuple[float, float], goal: Tuple[float, float], reason: str) -> Dict[str, Any]:
        waypoint = next_waypoint(me, goal, self.obstacles, max_step=1.0)
        if waypoint is None:
            return {"unit_id": unit_id, "action": "TURN", "theta": 45.0, "reason": "route blocked: scan"}
        return {
            "unit_id": unit_id,
            "action": "MOVE",
            "x": round(waypoint[0], 3),
            "y": round(waypoint[1], 3),
            "reason": reason,
        }

    def _target_key(self, entity: Dict[str, Any]) -> Tuple[float, float]:
        """RED target 우선순위를 정한다. low_hp는 부상당한 BLUE를 먼저 압박한다."""
        distance = float(entity.get("r", 999.0))
        hp = float(entity.get("hp", 1.0))
        if self.target_priority == "low_hp":
            return hp, distance
        return distance, hp

    @staticmethod
    def _smart_fire_key(entity: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
        """사격 가능한 후보 중 즉시 처치와 기대 처치 시간을 함께 본다."""
        distance = float(entity.get("r", 999.0))
        hp = max(0.0, float(entity.get("hp", 100.0)))
        expected_damage = hit_probability(distance) * float(damage_for_distance(distance))
        if expected_damage <= 0.0:
            expected_shots = float("inf")
        else:
            expected_shots = hp / expected_damage
        # 처치 직전 표적은 거리보다 먼저 마무리한다. 그 외에는 7u 안의 안정 사격을
        # 우선하고, 같은 band에서는 기대 처치 시간이 짧은 표적에 집중한다.
        kill_now_rank = 0.0 if hp <= float(DAMAGE_PER_HIT) else 1.0
        range_band = 0.0 if distance <= EFFECTIVE_FIRE_RANGE else 1.0
        return kill_now_rank, range_band, expected_shots, distance, int(entity.get("id", 999))

    @staticmethod
    def _smart_pursuit_key(entity: Dict[str, Any]) -> Tuple[float, float, int]:
        """추격 단계에서는 멀리 있는 부상병 집착보다 접촉 유지가 우선이다."""
        distance = float(entity.get("r", 999.0))
        hp = float(entity.get("hp", 100.0))
        return distance, hp, int(entity.get("id", 999))

    def _search_lanes(self) -> Tuple[Tuple[float, float], ...]:
        """open smart RED는 목표 후방을 버리지 않고 중앙 차단선을 유지한다."""
        if self.target_priority == "smart" and not self.obstacles:
            return OPEN_SCREEN_LANES
        return SEARCH_LANES

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self_info = observation.get("self", {})
        unit_id = int(self_info.get("id", 0))
        hp = int(self_info.get("hp", 0))
        ammo = int(self_info.get("ammo", 0))
        me = (float(self_info.get("x", 0.0)), float(self_info.get("y", 0.0)))

        if hp <= 0:
            return self._stop(unit_id, "destroyed")

        enemies: List[Dict[str, Any]] = [
            entity
            for entity in observation.get("visible_entities", [])
            if entity.get("type") == self.target_type
            and entity.get("state") != "DESTROYED"
            and float(entity.get("hp", 1)) > 0
        ]

        if enemies:
            if ammo <= 0:
                return self._stop(unit_id, "out of ammo")

            if self.target_priority == "smart":
                fireable = [
                    entity
                    for entity in enemies
                    if float(entity.get("r", 999.0)) <= MAX_FIRE_RANGE
                    and hit_probability(float(entity.get("r", 999.0))) > 0.0
                ]
                if fireable:
                    target = min(fireable, key=self._smart_fire_key)
                    target_pos = (float(target["x"]), float(target["y"]))
                    self.last_seen = target_pos
                    return {
                        "unit_id": unit_id,
                        "action": "ENGAGE",
                        "target_id": int(target["id"]),
                        "reason": f"smart focus B{int(target['id'])}",
                    }
                target = min(enemies, key=self._smart_pursuit_key)
            else:
                # 기본은 가장 가까운 즉시 위협, 변형 실험에서는 낮은 HP 표적을 우선한다.
                target = min(enemies, key=self._target_key)

            target_pos = (float(target["x"]), float(target["y"]))
            self.last_seen = target_pos
            distance = float(target.get("r", math.dist(me, target_pos)))

            if distance <= EFFECTIVE_FIRE_RANGE:
                return {
                    "unit_id": unit_id,
                    "action": "ENGAGE",
                    "target_id": int(target["id"]),
                    "reason": f"destroy visible B{int(target['id'])}",
                }
            return self._move(unit_id, me, target_pos, f"pursue visible B{int(target['id'])}")

        if self.last_seen is not None:
            if math.dist(me, self.last_seen) > 0.75:
                return self._move(unit_id, me, self.last_seen, "pursue last seen Blue position")
            self.last_seen = None
            return {"unit_id": unit_id, "action": "TURN", "theta": 45.0, "reason": "search at last contact"}

        # 첫 탐색 축을 유닛별로 분산하고, 도착할 때마다 다음 골목으로 순환한다.
        lanes = self._search_lanes()
        if self.search_index is None:
            self.search_index = max(0, unit_id - 201) % len(lanes)
        goal = lanes[self.search_index]
        if math.dist(me, goal) <= 0.75:
            if self.target_priority == "smart" and not self.obstacles:
                return self._stop(unit_id, "hold open screen for Blue approach")
            self.search_index = (self.search_index + 1) % len(lanes)
            goal = lanes[self.search_index]
        return self._move(unit_id, me, goal, "search city lane for Blue")
