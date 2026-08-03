from typing import Any, Dict

from hackerthon.combat_config import EFFECTIVE_FIRE_RANGE

# Flank threat threshold: |theta| > this angle is treated as a flanking unit
_FLANK_THETA_DEG = 50.0

class RuleSoldierPolicy:
    """동쪽 방어 위치를 지키며 유효 사거리 안의 Blue만 교전하는 Red 경계 AI.

    Red는 적을 관측하지 못하거나 관측 표적이 유효 사거리 밖에 있으면 현재
    위치를 유지한다. 유효 사거리 안의 적만 ENGAGE하며 Soldier가 표적 방향으로
    heading을 자동 갱신한다.

    행동 우선순위:
      1. 적 미관측: 현재 방어 위치를 유지한다.
      2. 적 관측: 측면 위협을 우선하고, 없으면 가장 가까운 적을 고른다.
      3. 선택 표적이 7보다 멀면 현재 방어 위치를 유지한다.
      4. 선택 표적이 7 이내면 교전한다.

    Note: heading 갱신은 이 정책이 아니라 Soldier의 명령 실행부가 담당한다.
    """

    def __init__(self, target_type: str = "soldier"):
        self.target_type = target_type

    def decide(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self_info = observation.get("self", {})
        unit_id   = self_info.get("id", 0)
        hp        = int(self_info.get("hp", 0))
        ammo      = int(self_info.get("ammo", 0))
        visible   = observation.get("visible_entities", [])

        if hp <= 0:
            return {"unit_id": unit_id, "action": "STOP"}

        enemies = [
            e for e in visible
            if e.get("type") == self.target_type
            and e.get("state") != "DESTROYED"
            and float(e.get("hp", 1)) > 0
        ]

        # 적이 보이지 않으면 동쪽의 현재 방어 위치를 유지한다.
        if not enemies:
            return {"unit_id": unit_id, "action": "STOP"}

        # ── Select target ─────────────────────────────────────────────────────
        # Flanking units (|theta| > threshold) are visible but pose an asymmetric
        # threat — prioritise them so Red reacts to flank attacks.
        flank_threats = [
            e for e in enemies
            if abs(float(e.get("theta", 0.0))) > _FLANK_THETA_DEG
        ]
        if flank_threats:
            target = min(flank_threats, key=lambda e: float(e.get("r", 999.0)))
        else:
            target = min(enemies, key=lambda e: float(e.get("r", 999.0)))

        dist  = float(target.get("r", 999.0))

        if ammo <= 0:
            return {"unit_id": unit_id, "action": "STOP"}

        # 7 이내에서만 교전하고 실제 명중·피해는 World의 거리별 확률에 맡긴다.
        if dist <= EFFECTIVE_FIRE_RANGE:
            return {
                "unit_id":   unit_id,
                "action":    "ENGAGE",
                "target_id": target["id"],
            }

        # 적을 봤지만 7보다 멀면 추격하지 않고 방어 위치를 유지한다.
        return {"unit_id": unit_id, "action": "STOP"}


def _move_toward(unit_id: int, enemy: Dict[str, Any]) -> Dict[str, Any]:
    """Retained for backward compatibility."""
    return {
        "unit_id": unit_id,
        "action": "MOVE",
        "r": 999.0,
        "theta": float(enemy.get("theta", 0.0)),
    }
