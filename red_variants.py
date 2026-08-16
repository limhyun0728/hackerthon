"""RED 교리 변형 정책들 — 월드모델의 상대 정책 민감도 실험용 (2026-08-15).

run13은 순찰형 rule RED의 데이터로 학습됐다. 상대 교리가 바뀌었을 때 예측이 얼마나
유지되는지를 재기 위해, 행동 양상이 정반대인 변형을 제공한다. UrbanRedPolicy와 같은
decide(observation) 인터페이스이고 생성자 kwargs는 호환을 위해 받되 대부분 무시한다.

- AmbushRedPolicy: 전원 제자리 매복 — 유효사거리 안에 표적이 오면 사격, 아니면 정지.
  정지가정 기준선이 완벽해지는 극단이라 "WM이 상대 습관을 학습했다"의 리트머스.
- KiteRedPolicy: 쏘고 빠지기 — 4u 안이면 이탈 기동, 4~7u 밴드에서 사격, 밖이면 접근.
  학습 데이터에 후퇴 분기가 전혀 없으므로 이동 방향이 정면으로 반대가 되는 변형.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from hackerthon.combat_config import EFFECTIVE_FIRE_RANGE
from hackerthon.terrain import next_waypoint


def _alive_enemies(observation: Dict[str, Any], target_type: str) -> List[Dict[str, Any]]:
    return [
        e for e in observation.get("visible_entities", [])
        if e.get("type") == target_type
        and e.get("state") != "DESTROYED"
        and float(e.get("hp", 1)) > 0
    ]


class AmbushRedPolicy:
    """제자리 매복: 이동 없음, 유효사거리 안 최근접 표적에만 사격."""

    def __init__(self, target_type: str = "soldier", obstacles=None, **_ignored):
        self.target_type = target_type
        self.obstacles = list(obstacles or [])
        # UrbanRedPolicy 인터페이스 호환 (rollout 상태 승계 코드가 읽는다)
        self.last_seen = None
        self.search_index = None

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self_info = observation.get("self", {})
        unit_id = int(self_info.get("id", 0))
        if int(self_info.get("hp", 0)) <= 0:
            return {"unit_id": unit_id, "action": "STOP", "reason": "destroyed"}
        enemies = _alive_enemies(observation, self.target_type)
        if enemies and int(self_info.get("ammo", 0)) > 0:
            target = min(enemies, key=lambda e: float(e.get("r", 999.0)))
            if float(target.get("r", 999.0)) <= EFFECTIVE_FIRE_RANGE:
                return {"unit_id": unit_id, "action": "ENGAGE",
                        "target_id": int(target["id"]), "reason": "ambush fire"}
        return {"unit_id": unit_id, "action": "STOP", "reason": "ambush hold"}


class KiteRedPolicy:
    """쏘고 빠지기: 4u 안이면 이탈, 4~7u에서 사격, 밖이면 접근. 후퇴 분기가 핵심."""

    KEEP_MIN = 4.0
    KEEP_MAX = float(EFFECTIVE_FIRE_RANGE)

    def __init__(self, target_type: str = "soldier", obstacles=None, max_step: float = 1.0,
                 **_ignored):
        self.target_type = target_type
        self.obstacles = list(obstacles or [])
        self.max_step = float(max_step)
        self.last_seen = None
        self.search_index = None

    def _move_toward(self, unit_id: int, me, goal, reason: str) -> Dict[str, Any]:
        waypoint = next_waypoint(me, goal, self.obstacles, max_step=self.max_step)
        if waypoint is None:
            return {"unit_id": unit_id, "action": "TURN", "theta": 45.0, "reason": "blocked: scan"}
        return {"unit_id": unit_id, "action": "MOVE",
                "x": round(waypoint[0], 3), "y": round(waypoint[1], 3), "reason": reason}

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self_info = observation.get("self", {})
        unit_id = int(self_info.get("id", 0))
        me = (float(self_info.get("x", 0.0)), float(self_info.get("y", 0.0)))
        if int(self_info.get("hp", 0)) <= 0:
            return {"unit_id": unit_id, "action": "STOP", "reason": "destroyed"}
        enemies = _alive_enemies(observation, self.target_type)
        if enemies:
            target = min(enemies, key=lambda e: float(e.get("r", 999.0)))
            tpos = (float(target["x"]), float(target["y"]))
            self.last_seen = tpos
            r = float(target.get("r", math.dist(me, tpos)))
            if r < self.KEEP_MIN:
                # 이탈: 표적 반대 방향으로 한 걸음 (경로 탐색으로 건물 우회)
                away = (me[0] + (me[0] - tpos[0]) / max(r, 0.1) * 3.0,
                        me[1] + (me[1] - tpos[1]) / max(r, 0.1) * 3.0)
                return self._move_toward(unit_id, me, away, "kite back")
            if r <= self.KEEP_MAX and int(self_info.get("ammo", 0)) > 0:
                return {"unit_id": unit_id, "action": "ENGAGE",
                        "target_id": int(target["id"]), "reason": "kite fire"}
            return self._move_toward(unit_id, me, tpos, "kite close")
        if self.last_seen is not None and math.dist(me, self.last_seen) > 0.75:
            return self._move_toward(unit_id, me, self.last_seen, "pursue last seen")
        return {"unit_id": unit_id, "action": "STOP", "reason": "kite hold"}


def build_red_policy(priority, *, obstacles, lane_seed=0, assault_target=None,
                     target_type: str = "soldier"):
    """우선순위 문자열(nearest/low_hp/smart) 또는 교리 변형(assault/ambush/kite)을
    정책 객체로 푼다 — episodic 생성기와 devs_rollout이 공유하는 단일 진입점."""
    from hackerthon.red_policy import UrbanRedPolicy

    if priority == "ambush":
        return AmbushRedPolicy(target_type=target_type, obstacles=obstacles)
    if priority == "kite":
        return KiteRedPolicy(target_type=target_type, obstacles=obstacles)
    if priority == "assault":
        return UrbanRedPolicy(
            target_type=target_type, obstacles=obstacles, target_priority="nearest",
            lane_seed=lane_seed,
            assault_target=assault_target if assault_target is not None else RED_ASSAULT_FALLBACK,
        )
    return UrbanRedPolicy(
        target_type=target_type, obstacles=obstacles, target_priority=priority,
        lane_seed=lane_seed, assault_target=assault_target,
    )


RED_ASSAULT_FALLBACK = (-10.0, -2.5)   # BLUE 진영 중앙 — assault 변형의 기본 공세 목표
