"""고수준 플래너: 이벤트가 올라올 때만 전술·역할을 판단한다.

- LLMPlanner: 텍스트 belief 요약을 주고 CONTINUE(계획 유지) / REVISE(새 계획)를 받는다.
- ScriptedPlanner: LLM 없이 같은 계약을 구현한 규칙 플래너 (스모크·베이스라인용).
LLM이 실패하면 ScriptedPlanner로 폴백하므로 런은 죽지 않는다.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.commander_tactics import TACTIC_LIBRARY  # noqa: E402

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["CONTINUE", "REVISE"]},
        "tactic": {"type": "string", "enum": sorted(TACTIC_LIBRARY)},
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "integer"},
                    "role": {"type": "string"},
                },
                "required": ["unit_id", "role"],
            },
        },
        "flank_side": {"type": "string", "enum": ["N", "S", "AUTO"]},
        "focus_target_id": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "tactic", "assignments", "flank_side", "focus_target_id", "rationale"],
}


def _validate_plan(plan: Dict[str, Any], living_blue: List[int]) -> None:
    tactic = plan["tactic"]
    if tactic not in TACTIC_LIBRARY:
        raise ValueError(f"unknown tactic {tactic}")
    allowed = set(TACTIC_LIBRARY[tactic]["roles"])
    assigned = {int(a["unit_id"]): str(a["role"]) for a in plan["assignments"]}
    if sorted(assigned) != sorted(living_blue):
        raise ValueError(f"assignments must cover living units {living_blue}, got {sorted(assigned)}")
    bad = {r for r in assigned.values() if r not in allowed}
    if bad:
        raise ValueError(f"roles {bad} not allowed for {tactic}")
    missing = set(TACTIC_LIBRARY[tactic]["required_roles"]) - set(assigned.values())
    if missing:
        raise ValueError(f"required roles missing for {tactic}: {missing}")


def _normalize(plan: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(plan)
    if out.get("flank_side") == "AUTO":
        out["flank_side"] = None
    if int(out.get("focus_target_id", -1)) < 0:
        out["focus_target_id"] = None
    return out


class ScriptedPlanner:
    """규칙 기반 베이스라인. LLM과 동일한 계약(CONTINUE/REVISE)을 구현한다."""

    name = "scripted"

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        living = context["living_blue"]
        enemies = context["known_enemies"]
        destroyed = int(context.get("destroyed_count", 0))
        current = context.get("current_plan")
        events = {e["type"] for e in context.get("events", [])}

        # 전원 탄약 고갈이면 공세가 불가능하다: 집결 방어로 전환한다.
        ammo_by_unit = {b["id"]: b["ammo"] for b in context.get("blue_summary", [])}
        if ammo_by_unit and all(a <= 0 for a in ammo_by_unit.values()):
            if current and current["tactic"] == "REGROUP":
                return {"decision": "CONTINUE", "rationale": "out of ammo, holding rally"}
            anchor = min(living)
            return _normalize({
                "decision": "REVISE",
                "tactic": "REGROUP",
                "assignments": [
                    {"unit_id": u, "role": "ANCHOR" if u == anchor else "REGROUPER"}
                    for u in living
                ],
                "flank_side": "AUTO",
                "focus_target_id": -1,
                "rationale": "all units out of ammo: rally and hold",
            })

        # 1기로는 FIXER/FLANKER를 동시에 구성할 수 없다.
        # LLM 폴백이 불가능한 기존 계획을 CONTINUE하지 않게 한다.
        if len(living) == 1:
            uid = living[0]
            tactic = "FOCUS_FIRE" if enemies else "REGROUP"
            role = "SHOOTER" if enemies else "ANCHOR"
            focus = min(enemies, key=lambda eid: enemies[eid]["hp"]) if enemies else -1
            if (
                current
                and current.get("tactic") == tactic
                and current.get("assignments") == [{"unit_id": uid, "role": role}]
            ):
                return {"decision": "CONTINUE", "rationale": "single survivor plan still valid"}
            return _normalize({
                "decision": "REVISE",
                "tactic": tactic,
                "assignments": [{"unit_id": uid, "role": role}],
                "flank_side": "AUTO",
                "focus_target_id": focus,
                "rationale": "single survivor cannot form a multi-role maneuver",
            })

        # "본 적이 1"과 "남은 적이 1"을 구분한다: 격파 실적이 쌓인 뒤에만 잔적으로 간주.
        last_enemy = len(enemies) == 1 and destroyed >= 2
        # 각개격파가 정체되면(고립 표적 소진) 화력 집중으로 격상한다.
        stalled = "STALLED" in events and current is not None and current["tactic"] != "FOCUS_FIRE"
        needs_new = current is None or "FRIENDLY_KIA" in events
        if current and last_enemy and current["tactic"] != "FOCUS_FIRE":
            needs_new = True
        if stalled:
            needs_new = True
        if not needs_new:
            return {"decision": "CONTINUE", "rationale": "plan still valid"}

        if (last_enemy or stalled) and enemies:
            weakest = min(enemies, key=lambda e: enemies[e]["hp"])
            assignments = [{"unit_id": u, "role": "SHOOTER"} for u in living]
            return _normalize({
                "decision": "REVISE",
                "tactic": "FOCUS_FIRE",
                "assignments": assignments,
                "flank_side": "AUTO",
                "focus_target_id": weakest,
                "rationale": "isolated targets exhausted or last enemy: concentrate fire on weakest",
            })

        ordered = sorted(living)
        roles = {}
        for i, uid in enumerate(ordered):
            if i < max(1, len(ordered) // 2 - 1) + 1:
                roles[uid] = "FIXER"
            elif i >= len(ordered) - 2 and len(ordered) >= 3:
                roles[uid] = "FLANKER"
            else:
                roles[uid] = "SUPPORT"
        if "FLANKER" not in roles.values():
            roles[ordered[-1]] = "FLANKER"
        return _normalize({
            "decision": "REVISE",
            "tactic": "FIX_AND_FLANK",
            "assignments": [{"unit_id": u, "role": r} for u, r in roles.items()],
            "flank_side": "AUTO",
            "focus_target_id": -1,
            "rationale": "fix from standoff, envelop the isolated edge",
        })


class LLMPlanner:
    """이벤트 시에만 호출되는 텍스트 전용 LLM 플래너."""

    def __init__(self, model: str, temperature: float = 0.2, retries: int = 3):
        self.model = model
        self.temperature = float(temperature)
        self.retries = int(retries)
        self.fallback = ScriptedPlanner()
        self.name = f"llm:{model}"

    def _prompt(self, context: Dict[str, Any]) -> str:
        lines = [
            "You are the Blue force commander. Decide whether to keep or revise the current tactical plan.",
            "A deterministic executor handles all movement, pathing, and range control.",
            "You only choose: tactic, per-unit roles, flank side (or AUTO), and focus target (or -1).",
            "",
            f"Mission: destroy all Red forces, then reach the eastern objective at (10, 0).",
            f"Obstacles (impassable walls, block sight and bullets): {context['obstacles']}",
            f"Blue units: {json.dumps(context['blue_summary'])}",
            f"Known Red units: {json.dumps(context['red_summary'])}",
            "Known Red units are only partial observations. Unknown Red units may still exist;",
            "never treat the length of this list as the total enemy force size.",
            f"Current plan: {json.dumps(context.get('current_plan')) if context.get('current_plan') else 'NONE (produce initial plan, decision must be REVISE)'}",
            f"New events this tick: {json.dumps(context['events'])}",
            "",
            "Tactic library:",
        ]
        for tactic, definition in TACTIC_LIBRARY.items():
            roles = ", ".join(
                f"{role}: {description}"
                for role, description in definition["roles"].items()
            )
            lines.append(f"- {tactic}: {definition['summary']} Roles: [{roles}]")
        lines += [
            "",
            "Rules of thumb: Red searches for Blue, pursues through obstacle-aware routes,",
            "turns toward incoming fire, and engages visible Blue units inside range 7;",
            "Blue can fire out to 10. The executor makes FIXERs fight in the 4.5-6.8 band",
            "and only that role withdraws when an enemy is danger-close inside 2.5.",
            "FLANKERs attack isolated edge enemies, while INFILTRATORs minimize LOS exposure.",
            "When hard cover is available against an advancing enemy, COVER_AND_FIRE lets Blue",
            "move behind cover, peek to fire, and recover before the next shot.",
            "Prefer COVER_AND_FIRE when Red is advancing, Blue has taken damage, or open-ground",
            "standoff/flanking has stalled near available buildings or barriers.",
            "Choose DIVERSION_AND_FLANK when one covered element can deliberately draw Red attention",
            "while multiple INFILTRATORs use low-exposure obstacle routes to attack a flank.",
            "If the current plan is still making progress, answer CONTINUE.",
            "Assign a role to every living Blue unit when you REVISE. Return JSON.",
            "Do not choose a tactic if the living units cannot cover all of its required distinct roles;",
            "with one surviving unit use FOCUS_FIRE (if an enemy is known) or REGROUP.",
        ]
        return "\n".join(lines)

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import httpx
        import ollama

        living = context["living_blue"]
        last_error = ""
        for attempt in range(1, self.retries + 1):
            prompt = self._prompt(context)
            if last_error:
                prompt += f"\nPrevious output was invalid: {last_error[:300]}. Return corrected JSON."
            try:
                started = time.time()
                response = ollama.Client(timeout=600.0).chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return one valid JSON object matching the schema."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": self.temperature, "num_ctx": 8192},
                    format=PLAN_SCHEMA,
                    keep_alive="30m",
                )
                plan = json.loads(response["message"]["content"])
                plan["latency_s"] = round(time.time() - started, 1)
                if plan["decision"] == "CONTINUE":
                    if context.get("current_plan") is None:
                        raise ValueError("initial call must REVISE")
                    return plan
                _validate_plan(plan, living)
                return _normalize(plan)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                print(f"[planner NETWORK RETRY] {attempt}/{self.retries}: {exc!r}")
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = str(exc)
                print(f"[planner RETRY] {attempt}/{self.retries}: {last_error}")

        print("[planner] LLM failed, falling back to scripted planner")
        fallback_plan = self.fallback.decide(context)
        fallback_plan["rationale"] = "(fallback) " + fallback_plan.get("rationale", "")
        return fallback_plan
