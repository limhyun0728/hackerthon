"""시가지 시나리오의 Red 섬멸 규칙 정책.

Red는 제자리에서 반격만 하는 표적이 아니다. Blue 섬멸을 임무로 삼고
시가지를 탐색하며, 보이는 표적이 사거리 밖이면 장애물을 우회해 추격한다.
표적을 놓치면 마지막 관측 지점까지 이동한 뒤 다시 탐색을 시작한다.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from hackerthon.combat_config import DAMAGE_PER_HIT, EFFECTIVE_FIRE_RANGE, MAX_FIRE_RANGE, damage_for_distance, hit_probability
from hackerthon.terrain import (
    WORLD_X_MAX,
    WORLD_X_MIN,
    cell_of,
    component_points,
    largest_free_component,
    next_waypoint,
    ring_positions,
    snap_to_component,
)


# 기본 시가지(URBAN_OBSTACLES) 전용 폴백. 실측 건물맵에서는 이 좌표들이 건물
# 내부에 떨어져 유닛이 영구 정지하므로, obstacles가 있으면 맵에서 직접 뽑는다.
FALLBACK_SEARCH_LANES: Tuple[Tuple[float, float], ...] = (
    (-7.5, -6.0),
    (-7.5, -3.0),
    (-7.5, 0.0),
    (-7.5, 3.0),
    (-7.5, 6.0),
)
SEARCH_LANES = FALLBACK_SEARCH_LANES

# 맵에서 유도할 때 뽑을 탐색 지점 수와, 지점 간 최소 간격(유닛).
SEARCH_LANE_COUNT = 5
SEARCH_LANE_MIN_SEPARATION = 3.0

# 거점 공격 시 거점에서 이만큼 떨어진 지점들로 접근한다.
ASSAULT_APPROACH_RADIUS = 4.0

OPEN_SCREEN_LANES: Tuple[Tuple[float, float], ...] = (
    (4.0, -6.0),
    (4.0, -3.0),
    (4.0, 0.0),
    (4.0, 3.0),
    (4.0, 6.0),
)


# 같은 지형에서 매번 flood fill을 돌리지 않도록 lane 결과를 캐시한다.
# rollout 평가는 tick마다 policy를 새로 만들기 때문에 캐시가 없으면 매우 느리다.
_LANE_CACHE: Dict[Tuple, Tuple[Tuple[float, float], ...]] = {}


_ASSAULT_CACHE: Dict[Tuple, Tuple[Tuple[float, float], ...]] = {}


def derive_assault_points(
    obstacles,
    target: Tuple[float, float],
    *,
    count: int = SEARCH_LANE_COUNT,
    radius: float = ASSAULT_APPROACH_RADIUS,
) -> Tuple[Tuple[float, float], ...]:
    """거점(target) 주위의 도달 가능한 접근 지점을 방위별로 뽑는다.

    전원이 같은 한 점으로 몰리면 궤적이 단조로워지고 학습 데이터의 다양성이
    떨어진다. 사방의 접근로를 나눠 맡게 해서 포위 기동 형태가 나오게 한다.
    """
    rects = tuple(tuple(float(value) for value in rect) for rect in obstacles)
    key = (rects, (float(target[0]), float(target[1])), int(count), float(radius))
    cached = _ASSAULT_CACHE.get(key)
    if cached is not None:
        return cached

    component = largest_free_component(list(rects))
    reachable = set(component)
    approaches = [
        point
        for point in ring_positions(target, radius, n=max(8, count * 4))
        if cell_of(point) in reachable
    ]
    if not approaches:
        # 거점 주위가 전부 막혀 있으면 거점 자체로 향한다.
        snapped = snap_to_component(target, component)
        approaches = [snapped if snapped is not None else target]

    # 방위가 고르게 섞이도록 원주 순서에서 일정 간격으로 고른다.
    step = max(1, len(approaches) // count)
    picked = tuple(approaches[(index * step) % len(approaches)] for index in range(count))
    _ASSAULT_CACHE[key] = picked
    return picked


def derive_search_lanes(
    obstacles,
    *,
    seed: int = 0,
    count: int = SEARCH_LANE_COUNT,
    min_separation: float = SEARCH_LANE_MIN_SEPARATION,
) -> Tuple[Tuple[float, float], ...]:
    """RED 진영 쪽 빈 셀에서 탐색 지점을 무작위로 뽑는다.

    최대 이동 가능 영역에서만 고르므로 도달 가능성이 생성 시점에 보장된다.
    맵이 바뀌어도 건물 안에 목표가 찍히지 않는다.
    """
    rects = tuple(tuple(float(value) for value in rect) for rect in obstacles)
    key = (rects, int(seed), int(count), float(min_separation))
    cached = _LANE_CACHE.get(key)
    if cached is not None:
        return cached

    points = component_points(largest_free_component(list(rects)))
    if not points:
        raise ValueError("이동 가능한 자유 공간이 없다: 지형이나 PATH_PAD를 확인해라")

    center_x = (WORLD_X_MIN + WORLD_X_MAX) / 2.0
    # RED는 자기 진영(월드 중앙 기준 +x)을 순찰한다. BLUE가 objective로 전진하면
    # 어차피 이 구역을 통과하므로 교전은 계속 생긴다.
    candidates = [point for point in points if point[0] >= center_x] or points

    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen: List[Tuple[float, float]] = []
    for point in candidates:
        if all(math.dist(point, picked) >= min_separation for picked in chosen):
            chosen.append(point)
            if len(chosen) >= count:
                break
    if not chosen:
        chosen = candidates[:count]

    lanes = tuple(chosen)
    _LANE_CACHE[key] = lanes
    return lanes


class UrbanRedPolicy:
    """120도 FOV로 탐색·추격·사격하는 상태형 Red 정책."""

    def __init__(
        self,
        target_type: str = "soldier",
        obstacles=None,
        target_priority: str = "nearest",
        lane_seed: int = 0,
        assault_target: Optional[Tuple[float, float]] = None,
    ):
        if target_priority not in ("nearest", "low_hp", "smart"):
            raise ValueError("target_priority는 nearest, low_hp 또는 smart여야 한다")
        self.target_type = target_type
        self.obstacles = list(obstacles or [])
        self.target_priority = target_priority
        self.lane_seed = int(lane_seed)
        # 값이 있으면 순찰 대신 그 거점을 공격한다. BLUE의 거점 방어 임무에서만 켠다.
        self.assault_target = (
            (float(assault_target[0]), float(assault_target[1]))
            if assault_target is not None
            else None
        )
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
        if self.assault_target is not None:
            if not self.obstacles:
                return (self.assault_target,)
            return derive_assault_points(self.obstacles, self.assault_target)
        if self.target_priority == "smart" and not self.obstacles:
            return OPEN_SCREEN_LANES
        if not self.obstacles:
            return FALLBACK_SEARCH_LANES
        return derive_search_lanes(self.obstacles, seed=self.lane_seed)

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
            # id 대역(101~/201~)과 무관하게 유닛마다 다른 lane을 잡게 한다.
            # 예전엔 201을 빼서 BLUE가 이 정책을 쓰면 전원 0번 lane으로 몰렸다.
            self.search_index = int(unit_id) % len(lanes)
        goal = lanes[self.search_index]
        if math.dist(me, goal) <= 0.75:
            if self.target_priority == "smart" and not self.obstacles:
                return self._stop(unit_id, "hold open screen for Blue approach")
            self.search_index = (self.search_index + 1) % len(lanes)
            goal = lanes[self.search_index]

        # 도착해야만 다음 lane으로 넘어가므로, 경로가 없는 lane을 잡으면 영구 고착된다.
        # 막힌 lane은 그 자리에서 건너뛰고 갈 수 있는 lane을 찾는다.
        for _ in range(len(lanes)):
            waypoint = next_waypoint(me, goal, self.obstacles, max_step=1.0)
            if waypoint is not None:
                return {
                    "unit_id": unit_id,
                    "action": "MOVE",
                    "x": round(waypoint[0], 3),
                    "y": round(waypoint[1], 3),
                    "reason": (
                        "assault objective" if self.assault_target is not None
                        else "search city lane for Blue"
                    ),
                }
            self.search_index = (self.search_index + 1) % len(lanes)
            goal = lanes[self.search_index]
        return {"unit_id": unit_id, "action": "TURN", "theta": 45.0, "reason": "all lanes unreachable: scan"}
