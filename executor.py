"""역할별 결정론 실행기: LLM 계획(전술·역할)을 매 틱 기하로 집행한다.

원칙:
- 목표는 좌표가 아니라 함수다. 적 belief가 바뀌면 사격 위치·우회 목표가 따라 움직인다.
- FIXER는 4.5~6.8 교전 밴드에서 실제로 사격하며, 2.5 이하에서만 개별 후퇴한다.
- 엄폐 사격은 3.5~5.5에서 peek/fire/recover를 반복한다.
- INFILTRATOR는 적 사선 노출이 가장 적은 장애물 경로로 우회한다.
- 실행기는 LLM을 호출하지 않는다. 의미 있는 변화만 이벤트로 올린다.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from hackerthon.terrain import astar_path, clamp_to_world, has_los, next_waypoint, point_blocked, ring_positions

# 사격·응사 기하 (combat_config과 일치)
MAX_FIRE = 10.0
RED_RETURN = 7.0
STANDOFF_LO = 4.5                   # FIXER 교전 밴드 하한
STANDOFF_BAND = 5.5                 # FIXER 목표 반경
STANDOFF_HI = 6.8                   # FIXER 교전 밴드 상한
DANGER_CLOSE = 2.5                  # 엄폐 없이 이 안쪽으로 들어오면 해당 역할만 후퇴
WITHDRAW_SAFE = RED_RETURN + 0.3    # 탄약 고갈 유닛은 Red 응사권 밖까지 이탈
FLANK_ATTACK_R = 3.5                # FLANKER 근접 타격 반경 (명중 0.65)
FOCUS_ATTACK_R = 5.5                # FOCUS_FIRE 사격 반경 (명중 0.35)
ARRIVE_EPS = 0.6
PRIOR_ENEMY_CENTER = (6.5, 0.0)     # 임무 첩보: 적은 동쪽 어딘가
SCOUT_SEARCH_X = 8.0                 # 건물 밖 동쪽 진입로

VALID_ROLES = {
    "FIXER", "FLANKER", "SUPPORT", "SHOOTER", "SECURITY",
    "SCOUT", "RESERVE", "ANCHOR", "REGROUPER",
    "COVER_SHOOTER", "COVER_SUPPORT", "DECOY", "INFILTRATOR", "OVERWATCH",
}

COVER_PAD = 0.65
COVER_PEEK_MAX_STEP = 2.6
COVER_FIRE_LO = 3.5
COVER_FIRE_HI = 5.5
COVER_FIRE_IDEAL = 4.5


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class Executor:
    def __init__(self, controlled_ids: List[int], obstacles, max_step: float = 1.5):
        self.controlled_ids = list(controlled_ids)
        self.obstacles = obstacles
        self.max_step = float(max_step)

        self.plan: Dict[str, Any] = {}
        self.enemies: Dict[int, Dict[str, Any]] = {}   # 살아 있는 것으로 믿는 Red
        self.friendlies: Dict[int, Dict[str, Any]] = {}
        self._anticipated: Dict[int, Tuple[float, float]] = {}
        self._prev_hp: Dict[int, float] = {}
        self._prev_enemy_hp: Dict[int, float] = {}
        self._known_dead: set = set()
        self._contacted: set = set()
        self._damaged_once: set = set()
        self._flank_arrived: set = set()
        self._flank_arrived_now: set = set()
        self._reserve_point: Optional[Tuple[float, float]] = None
        self._scout_phase: Dict[int, str] = {}
        self._stall_window: List[bool] = []
        self._cover_state: Dict[int, Dict[str, Any]] = {}

    # ── belief ───────────────────────────────────────────────────────────────
    def update_belief(self, intel_buffer: Dict[int, Any], tick: int) -> None:
        self.friendlies = {}
        for uid, obs in intel_buffer.items():
            info = obs.get("self", {})
            if float(info.get("hp", 0)) <= 0:
                continue
            x, y = float(info["x"]), float(info["y"])
            if uid in self._anticipated:
                x, y = self._anticipated[uid]
            self.friendlies[int(uid)] = {
                "x": x, "y": y,
                "hp": float(info["hp"]), "ammo": int(info["ammo"]),
                "heading": float(info.get("heading", 0.0)),
                "visible": obs.get("visible_entities", []) or [],
            }
            for ent in obs.get("visible_entities", []) or []:
                if ent.get("type") != "enemy":
                    continue
                eid = int(ent["id"])
                dead = ent.get("state") == "DESTROYED" or float(ent.get("hp", 1) or 0) <= 0
                if dead:
                    self._known_dead.add(eid)
                    self.enemies.pop(eid, None)
                elif eid not in self._known_dead:
                    self.enemies[eid] = {
                        "x": float(ent["x"]), "y": float(ent["y"]),
                        "hp": float(ent.get("hp") or 100), "last_seen": tick,
                    }
        if self._reserve_point is None and self.friendlies:
            xs = [f["x"] for f in self.friendlies.values()]
            ys = [f["y"] for f in self.friendlies.values()]
            self._reserve_point = (sum(xs) / len(xs), sum(ys) / len(ys))

    # ── 이벤트 ───────────────────────────────────────────────────────────────
    def detect_events(self, intel_buffer: Dict[int, Any], tick: int) -> List[Dict[str, Any]]:
        events = []
        cur_hp = {
            int(uid): float(obs["self"]["hp"])
            for uid, obs in intel_buffer.items()
        }
        for uid, hp in cur_hp.items():
            prev = self._prev_hp.get(uid)
            if prev is not None and prev > 0 and hp <= 0:
                events.append({"type": "FRIENDLY_KIA", "unit": uid})
            elif prev is not None and hp < prev:
                # 첫 피격만 알리면 85→10 HP 같은 위기를 플래너가 놓친다.
                # 첫 피격과 전투력 임계(50/25) 하락을 따로 이벤트화한다.
                if prev > 25 >= hp:
                    events.append({"type": "FRIENDLY_CRITICAL", "unit": uid, "hp": hp, "threshold": 25})
                elif prev > 50 >= hp:
                    events.append({"type": "FRIENDLY_CRITICAL", "unit": uid, "hp": hp, "threshold": 50})
                elif uid not in self._damaged_once:
                    events.append({"type": "FRIENDLY_DAMAGED", "unit": uid, "hp": hp})
                self._damaged_once.add(uid)
        for eid in sorted(self.enemies):
            if eid not in self._contacted:
                self._contacted.add(eid)
                events.append({"type": "CONTACT_NEW", "enemy": eid})
        for eid in sorted(self._known_dead):
            if self._prev_enemy_hp.get(eid, 0) > 0:
                events.append({"type": "ENEMY_KIA", "enemy": eid})
        # 직전 틱 실행 중에 감지된 도착을 이번 틱 이벤트로 올린다.
        for uid in sorted(self._flank_arrived_now):
            events.append({"type": "FLANK_IN_POSITION", "unit": uid})
        self._flank_arrived_now = set()

        # 정체: 8틱 동안 적 hp 변화도 아군 이동도 없으면 한 번 알린다.
        moved = any(
            _dist((f["x"], f["y"]), self._anticipated.get(uid, (f["x"], f["y"]))) > 1e-6
            for uid, f in self.friendlies.items()
        )
        enemy_hp_changed = any(
            self._prev_enemy_hp.get(eid) != e["hp"] for eid, e in self.enemies.items()
        )
        self._stall_window.append(moved or enemy_hp_changed or bool(events))
        if len(self._stall_window) >= 8:
            if not any(self._stall_window[-8:]):
                events.append({"type": "STALLED", "ticks": 8})
                self._stall_window.clear()
            else:
                self._stall_window = self._stall_window[-8:]

        self._prev_hp = cur_hp
        self._prev_enemy_hp = {eid: e["hp"] for eid, e in self.enemies.items()}
        self._prev_enemy_hp.update({eid: 0 for eid in self._known_dead})
        return events

    # ── 계획 ─────────────────────────────────────────────────────────────────
    def set_plan(self, plan: Dict[str, Any]) -> None:
        old_tactic = self.plan.get("tactic")
        old_roles = self.plan.get("roles", {})
        old_flank_side = self.plan.get("flank_side")
        roles = {int(a["unit_id"]): str(a["role"]) for a in plan["assignments"]}
        bad = {r for r in roles.values() if r not in VALID_ROLES}
        if bad:
            raise ValueError(f"unknown roles from planner: {bad}")
        new_flank_side = plan.get("flank_side") or self._auto_flank_side()
        self.plan = {
            "tactic": plan["tactic"],
            "roles": roles,
            "flank_side": new_flank_side,
            "focus_target": plan.get("focus_target_id"),
        }
        flank_roles = {"FLANKER", "INFILTRATOR"}
        old_flankers = {uid for uid, role in old_roles.items() if role in flank_roles}
        new_flankers = {uid for uid, role in roles.items() if role in flank_roles}
        if (
            old_tactic != plan["tactic"]
            or old_flank_side != new_flank_side
            or old_flankers != new_flankers
        ):
            self._flank_arrived.clear()
        cover_roles = {"COVER_SHOOTER", "COVER_SUPPORT", "DECOY", "OVERWATCH"}
        if old_tactic != plan["tactic"]:
            self._cover_state.clear()
        else:
            self._cover_state = {
                uid: state for uid, state in self._cover_state.items()
                if roles.get(uid) in cover_roles and old_roles.get(uid) == roles.get(uid)
            }

    def _auto_flank_side(self) -> str:
        """가장자리 표적이 더 고립된 쪽(각개격파에 유리한 쪽)을 고른다."""
        live = list(self.enemies.values())
        if not live:
            return "N"
        def isolation(side: str) -> float:
            edge = max(live, key=lambda e: e["y"]) if side == "N" else min(live, key=lambda e: e["y"])
            others = [e for e in live if e is not edge]
            if not others:
                return 999.0
            return min(_dist((edge["x"], edge["y"]), (o["x"], o["y"])) for o in others)
        return "N" if isolation("N") >= isolation("S") else "S"

    # ── 틱 실행 ──────────────────────────────────────────────────────────────
    def commands(self, tick: int) -> List[Dict[str, Any]]:
        out = []
        roles = self.plan.get("roles", {})
        for uid in sorted(self.friendlies):
            role = roles.get(uid, "SECURITY")
            if self.friendlies[uid]["ammo"] <= 0:
                # 탄약이 없는 유닛은 어떤 역할이든 응사 범위 밖으로 이탈한다.
                cmd = self._run_withdraw(uid)
            else:
                handler = getattr(self, f"_run_{role.lower()}", self._run_security)
                cmd = handler(uid)
            cmd.setdefault("reason", role)
            cmd["unit_id"] = uid
            cmd["duration_sec"] = 1.0
            out.append(cmd)

        for cmd in out:
            uid = cmd["unit_id"]
            if cmd["action"] == "MOVE":
                self._anticipated[uid] = (cmd["x"], cmd["y"])
            else:
                f = self.friendlies[uid]
                self._anticipated[uid] = (f["x"], f["y"])
        return out

    # ── 공통 도구 ────────────────────────────────────────────────────────────
    def _pos(self, uid: int) -> Tuple[float, float]:
        f = self.friendlies[uid]
        return (f["x"], f["y"])

    def _visible_enemies(self, uid: int) -> List[Dict[str, Any]]:
        return [
            e for e in self.friendlies[uid]["visible"]
            if e.get("type") == "enemy"
            and e.get("state") != "DESTROYED"
            and float(e.get("hp", 1) or 0) > 0
        ]

    def _engage(self, uid: int, target_id: int, why: str) -> Dict[str, Any]:
        return {"action": "ENGAGE", "target_id": int(target_id), "reason": why}

    def _move_step(self, uid: int, goal: Tuple[float, float], why: str) -> Dict[str, Any]:
        me = self._pos(uid)
        goal = clamp_to_world(*goal)
        step = next_waypoint(me, goal, self.obstacles, self.max_step)
        if step is None or _dist(me, step) < 0.05:
            return {"action": "STOP", "reason": f"{why} (no path/arrived)"}
        return {"action": "MOVE", "x": round(step[0], 3), "y": round(step[1], 3), "reason": why}

    def _turn_toward(self, uid: int, point: Tuple[float, float], why: str) -> Dict[str, Any]:
        me = self._pos(uid)
        desired = math.degrees(math.atan2(point[1] - me[1], point[0] - me[0]))
        heading = float(self.friendlies[uid].get("heading", 0.0))
        delta = (desired - heading + 180.0) % 360.0 - 180.0
        if abs(delta) < 1.0:
            return {"action": "STOP", "reason": f"{why}: facing target"}
        return {"action": "TURN", "theta": round(delta, 2), "reason": why}

    def _nearest_enemy(self, point: Tuple[float, float]) -> Optional[Tuple[int, Dict[str, Any]]]:
        if not self.enemies:
            return None
        eid = min(self.enemies, key=lambda e: _dist(point, (self.enemies[e]["x"], self.enemies[e]["y"])))
        return eid, self.enemies[eid]

    def _band_position(self, uid: int) -> Optional[Tuple[float, float]]:
        """가장 가까운 적 기준 4.5~6.8 교전 밴드의 LOS 사격 위치."""
        near = self._nearest_enemy(self._pos(uid))
        if near is None:
            return None
        _, enemy = near
        epos = (enemy["x"], enemy["y"])
        candidates = [
            p for p in ring_positions(epos, STANDOFF_BAND)
            if not point_blocked(p, self.obstacles, pad=0.5) and has_los(p, epos, self.obstacles)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: _dist(self._pos(uid), p))

    def _hold_or_shoot(self, uid: int, why: str, finish_hp: Optional[float] = None) -> Dict[str, Any]:
        """밴드 규칙 사격. finish_hp가 있으면 그 이하 표적만 쏜다.

        적이 2.5 이하까지 들어온 경우에만 해당 역할이 개별 후퇴하고,
        그 밖의 6.8 이내 가시 표적은 즉시 사격한다.
        """
        me = self._pos(uid)
        visible = self._visible_enemies(uid)
        near = self._nearest_enemy(me)
        if near is not None:
            _, enemy = near
            d = _dist(me, (enemy["x"], enemy["y"]))
            if d < DANGER_CLOSE:
                if d < 1e-6:
                    away = (me[0] - 2.0, me[1])
                else:
                    away = (me[0] + (me[0] - enemy["x"]) / d * 2.0, me[1] + (me[1] - enemy["y"]) / d * 2.0)
                return self._move_step(uid, away, f"{why}: danger-close role withdrawal")
        shootable = [e for e in visible if float(e["r"]) <= STANDOFF_HI]
        if finish_hp is not None:
            shootable = [e for e in shootable if float(e.get("hp") or 100) <= finish_hp]
        if shootable and self.friendlies[uid]["ammo"] > 0:
            target = min(shootable, key=lambda e: (float(e.get("hp") or 100), float(e["r"])))
            return self._engage(uid, int(target["id"]), f"{why}: fire from standoff")
        return {"action": "STOP", "reason": f"{why}: observing, holding fire"}

    def _cover_perimeter_points(self) -> List[Tuple[float, float]]:
        """엄폐물 주변의 보행 가능한 모서리/변 후보를 만든다."""
        points: List[Tuple[float, float]] = []
        for xmin, ymin, xmax, ymax in self.obstacles:
            xs = (xmin - COVER_PAD, (xmin + xmax) / 2.0, xmax + COVER_PAD)
            ys = (ymin - COVER_PAD, (ymin + ymax) / 2.0, ymax + COVER_PAD)
            points.extend((x, ymin - COVER_PAD) for x in xs)
            points.extend((x, ymax + COVER_PAD) for x in xs)
            points.extend((xmin - COVER_PAD, y) for y in ys[1:-1])
            points.extend((xmax + COVER_PAD, y) for y in ys[1:-1])
        unique = []
        seen = set()
        for point in points:
            rounded = (round(point[0], 3), round(point[1], 3))
            if rounded in seen or point_blocked(rounded, self.obstacles, pad=0.2):
                continue
            seen.add(rounded)
            unique.append(rounded)
        return unique

    def _cover_pair(self, uid: int, target_id: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """표적에게 숨겨지는 hide와, 가까운 사격 가능 peek 쌍을 고른다."""
        enemy = self.enemies.get(target_id)
        if enemy is None:
            return None
        me = self._pos(uid)
        epos = (enemy["x"], enemy["y"])
        points = self._cover_perimeter_points()
        hidden = [p for p in points if not has_los(epos, p, self.obstacles)]
        peeks = [
            p for p in points
            if has_los(epos, p, self.obstacles)
            and COVER_FIRE_LO <= _dist(epos, p) <= COVER_FIRE_HI
        ]
        reserved = [
            state["hide"] for other, state in self._cover_state.items()
            if other != uid and state.get("hide") is not None
        ]
        pairs = []
        for hide in hidden:
            if next_waypoint(me, hide, self.obstacles, self.max_step) is None:
                continue
            for peek in peeks:
                transition = _dist(hide, peek)
                if transition > COVER_PEEK_MAX_STEP:
                    continue
                other_exposure = sum(
                    1 for eid, other in self.enemies.items()
                    if eid != target_id
                    and _dist(peek, (other["x"], other["y"])) <= RED_RETURN
                    and has_los(peek, (other["x"], other["y"]), self.obstacles)
                )
                crowding = sum(max(0.0, 2.5 - _dist(hide, used)) for used in reserved)
                firing_distance = abs(_dist(epos, peek) - COVER_FIRE_IDEAL)
                score = (other_exposure, crowding, firing_distance, _dist(me, hide), transition)
                pairs.append((score, hide, peek))
        if not pairs:
            return None
        _, hide, peek = min(pairs, key=lambda item: item[0])
        return hide, peek

    def _run_cover_unit(self, uid: int, label: str) -> Dict[str, Any]:
        if not self.enemies:
            return self._move_step(uid, PRIOR_ENEMY_CENTER, f"{label}: advance cover-to-cover")

        focus = self.plan.get("focus_target")
        if focus not in self.enemies:
            near = self._nearest_enemy(self._pos(uid))
            focus = near[0] if near is not None else None
        if focus is None:
            return {"action": "STOP", "reason": f"{label}: no target belief"}

        enemy = self.enemies[focus]
        epos = (enemy["x"], enemy["y"])
        state = self._cover_state.get(uid)
        moved_target = state is not None and _dist(state.get("enemy_pos", epos), epos) > 2.5
        if state is None or state.get("target") != focus or moved_target:
            pair = self._cover_pair(uid, focus)
            if pair is None:
                return self._run_shooter(uid)
            hide, peek = pair
            state = {
                "target": focus, "enemy_pos": epos, "hide": hide, "peek": peek,
                "phase": "TO_COVER", "last_hp": self.friendlies[uid]["hp"],
            }
            self._cover_state[uid] = state

        hp = self.friendlies[uid]["hp"]
        if hp < state.get("last_hp", hp):
            state["phase"] = "RECOVER"
        state["last_hp"] = hp
        me = self._pos(uid)

        if state["phase"] == "TO_COVER":
            if _dist(me, state["hide"]) > ARRIVE_EPS:
                return self._move_step(uid, state["hide"], f"{label}: move behind hard cover")
            state["phase"] = "PEEK"
            return {"action": "STOP", "reason": f"{label}: protected behind cover"}

        if state["phase"] == "RECOVER":
            if _dist(me, state["hide"]) > ARRIVE_EPS:
                return self._move_step(uid, state["hide"], f"{label}: recover behind cover")
            state["phase"] = "PEEK"
            return {"action": "STOP", "reason": f"{label}: recovered; prepare next peek"}

        if _dist(me, state["peek"]) > ARRIVE_EPS:
            return self._move_step(uid, state["peek"], f"{label}: peek around cover")

        visible = {int(entity["id"]): entity for entity in self._visible_enemies(uid)}
        if focus in visible and float(visible[focus]["r"]) <= MAX_FIRE:
            state["phase"] = "RECOVER"
            return self._engage(uid, focus, f"{label}: fire from cover then recover")
        return self._turn_toward(uid, epos, f"{label}: acquire target from peek")

    # ── 역할 행동 ────────────────────────────────────────────────────────────
    def _run_fixer(self, uid: int) -> Dict[str, Any]:
        me = self._pos(uid)
        band = self._band_position(uid)
        if band is None:
            return self._move_step(uid, PRIOR_ENEMY_CENTER, "FIXER: advance to believed enemy area")
        if _dist(me, band) > ARRIVE_EPS * 2 and not self._visible_enemies(uid):
            return self._move_step(uid, band, "FIXER: move to standoff band")
        near = self._nearest_enemy(me)
        if near is not None and _dist(me, (near[1]["x"], near[1]["y"])) > STANDOFF_HI:
            return self._move_step(uid, band, "FIXER: close to band")
        # FIXER는 단순 관측자가 아니라 실제 사격으로 적을 고정한다.
        return self._hold_or_shoot(uid, "FIXER")

    def _run_cover_shooter(self, uid: int) -> Dict[str, Any]:
        return self._run_cover_unit(uid, "COVER_SHOOTER")

    def _run_cover_support(self, uid: int) -> Dict[str, Any]:
        return self._run_cover_unit(uid, "COVER_SUPPORT")

    def _run_decoy(self, uid: int) -> Dict[str, Any]:
        """엄폐에서 노출→사격→복귀를 반복해 적의 시선과 사격을 끌어당긴다."""
        near = self._nearest_enemy(self._pos(uid))
        if near is not None:
            _, enemy = near
            me = self._pos(uid)
            epos = (enemy["x"], enemy["y"])
            distance = _dist(me, epos)
            if distance < DANGER_CLOSE:
                state = self._cover_state.get(uid)
                if state is not None and state.get("hide") is not None:
                    state["phase"] = "RECOVER"
                    if _dist(me, state["hide"]) > ARRIVE_EPS:
                        return self._move_step(uid, state["hide"], "DECOY: danger-close recover behind cover")
                    return {"action": "STOP", "reason": "DECOY: protected from danger-close threat"}
                if distance < 1e-6:
                    away = (me[0] - 2.0, me[1])
                else:
                    away = (
                        me[0] + (me[0] - epos[0]) / distance * 2.0,
                        me[1] + (me[1] - epos[1]) / distance * 2.0,
                    )
                return self._move_step(uid, away, "DECOY: danger-close role withdrawal")
        return self._run_cover_unit(uid, "DECOY")

    def _run_overwatch(self, uid: int) -> Dict[str, Any]:
        """다른 엄폐물에서 decoy/infiltrator를 위협하는 표적을 사격한다."""
        return self._run_cover_unit(uid, "OVERWATCH")

    def _path_exposure(self, path: List[Tuple[float, float]]) -> Tuple[int, int]:
        """경로 중 적의 7u 응사 범위와 LOS에 드러나는 최대/누적 횟수."""
        samples: List[Tuple[float, float]] = []
        for start, end in zip(path, path[1:]):
            length = _dist(start, end)
            steps = max(1, int(math.ceil(length)))
            for index in range(steps):
                alpha = index / steps
                samples.append((
                    start[0] + (end[0] - start[0]) * alpha,
                    start[1] + (end[1] - start[1]) * alpha,
                ))
        exposures = [
            sum(
                1 for enemy in self.enemies.values()
                if _dist(point, (enemy["x"], enemy["y"])) <= RED_RETURN
                and has_los(point, (enemy["x"], enemy["y"]), self.obstacles)
            )
            for point in samples
        ]
        return (max(exposures, default=0), sum(exposures))

    def _run_infiltrator(self, uid: int) -> Dict[str, Any]:
        """적 응사 범위/LOS 노출이 가장 적은 장애물 경로로 측후방을 우회한다."""
        if not self.enemies:
            return self._move_step(uid, PRIOR_ENEMY_CENTER, "INFILTRATOR: advance toward contact area")
        side = self.plan.get("flank_side", "N")
        live = list(self.enemies.items())
        target_id, target = (max if side == "N" else min)(live, key=lambda item: item[1]["y"])
        epos = (target["x"], target["y"])
        me = self._pos(uid)

        candidates = [
            point for radius in (FLANK_ATTACK_R, 4.5)
            for point in ring_positions(epos, radius)
            if not point_blocked(point, self.obstacles, pad=0.5)
            and has_los(point, epos, self.obstacles)
            and ((point[1] >= epos[1]) if side == "N" else (point[1] <= epos[1]))
        ]
        scored = []
        for point in candidates:
            path = astar_path(me, point, self.obstacles)
            if not path:
                continue
            max_exposure, total_exposure = self._path_exposure(path)
            path_length = sum(_dist(a, b) for a, b in zip(path, path[1:]))
            scored.append(((max_exposure, total_exposure, path_length), point))
        if not scored:
            return self._run_flanker(uid)

        scored.sort(key=lambda item: item[0])
        infiltrators = sorted(
            unit for unit, role in self.plan.get("roles", {}).items()
            if role == "INFILTRATOR" and unit in self.friendlies
        )
        rank = infiltrators.index(uid) if uid in infiltrators else 0
        goal = scored[min(rank, len(scored) - 1)][1]
        if _dist(me, goal) > ARRIVE_EPS:
            return self._move_step(uid, goal, f"INFILTRATOR: low-exposure {side} route to R{target_id}")

        if uid not in self._flank_arrived:
            self._flank_arrived.add(uid)
            self._flank_arrived_now.add(uid)
        visible = {int(entity["id"]): entity for entity in self._visible_enemies(uid)}
        if target_id in visible:
            return self._engage(uid, target_id, f"INFILTRATOR: attack R{target_id} from flank")
        return self._turn_toward(uid, epos, "INFILTRATOR: acquire flank target")

    def _run_flanker(self, uid: int) -> Dict[str, Any]:
        me = self._pos(uid)
        side = self.plan.get("flank_side", "N")
        live = list(self.enemies.items())
        if not live:
            return self._move_step(uid, PRIOR_ENEMY_CENTER, "FLANKER: advance to believed enemy area")

        edge_id, edge = (max if side == "N" else min)(live, key=lambda kv: kv[1]["y"])
        epos = (edge["x"], edge["y"])
        others = [(e["x"], e["y"]) for eid, e in live if eid != edge_id]

        def isolated(p):
            return all(_dist(p, o) >= STANDOFF_LO for o in others)

        # 가장 유리한 링부터: 3.5(0.65) → 5.0 → 6.5(0.35). 고립 위치가 있는
        # 가장 가까운 링에서 표적만 응사 가능한 1(2)대1 교전을 만든다.
        candidates: List[Tuple[float, float]] = []
        attack_r = FLANK_ATTACK_R
        for ring in (FLANK_ATTACK_R, 5.0, 6.5):
            candidates = [
                p for p in ring_positions(epos, ring)
                if not point_blocked(p, self.obstacles, pad=0.5)
                and has_los(p, epos, self.obstacles)
                and isolated(p)
            ]
            if candidates:
                attack_r = ring
                break
        if not candidates:
            return self._hold_or_shoot(uid, "FLANKER: no isolated approach")

        goal = min(candidates, key=lambda p: _dist(me, p))
        d_target = _dist(me, epos)
        if _dist(me, goal) <= ARRIVE_EPS or (d_target <= attack_r + 0.5 and isolated(me)):
            if uid not in self._flank_arrived:
                self._flank_arrived.add(uid)
                self._flank_arrived_now.add(uid)
            visible = {int(e["id"]): e for e in self._visible_enemies(uid)}
            if edge_id in visible and self.friendlies[uid]["ammo"] > 0:
                return self._engage(uid, edge_id, f"FLANKER: destroy isolated R{edge_id}")
            return {"action": "STOP", "reason": "FLANKER: in position, waiting LOS"}
        return self._move_step(uid, goal, f"FLANKER: envelop via {side} to R{edge_id}")

    def _run_support(self, uid: int) -> Dict[str, Any]:
        roles = self.plan.get("roles", {})
        element = [u for u, r in roles.items() if r == "FLANKER" and u in self.friendlies]
        if not element:
            element = [u for u, r in roles.items() if r == "FIXER" and u in self.friendlies]
        if element:
            xs = [self.friendlies[u]["x"] for u in element]
            ys = [self.friendlies[u]["y"] for u in element]
            goal = (sum(xs) / len(xs) - 2.5, sum(ys) / len(ys))
        else:
            goal = self._reserve_point or PRIOR_ENEMY_CENTER
        me = self._pos(uid)
        visible = [e for e in self._visible_enemies(uid) if float(e["r"]) <= STANDOFF_HI]
        if visible and self.friendlies[uid]["ammo"] > 0:
            return self._hold_or_shoot(uid, "SUPPORT")
        if _dist(me, goal) > 1.2:
            return self._move_step(uid, goal, "SUPPORT: trail the element")
        return {"action": "STOP", "reason": "SUPPORT: in trail position"}

    def _run_shooter(self, uid: int) -> Dict[str, Any]:
        focus = self.plan.get("focus_target")
        if focus not in self.enemies:
            near = self._nearest_enemy(self._pos(uid))
            if near is None:
                return self._move_step(uid, PRIOR_ENEMY_CENTER, "SHOOTER: advance")
            focus = min(self.enemies, key=lambda e: self.enemies[e]["hp"])
            self.plan["focus_target"] = focus
        enemy = self.enemies[focus]
        epos = (enemy["x"], enemy["y"])
        me = self._pos(uid)
        visible = {int(e["id"]): e for e in self._visible_enemies(uid)}
        if focus in visible and _dist(me, epos) <= FOCUS_ATTACK_R + 0.5 and self.friendlies[uid]["ammo"] > 0:
            return self._engage(uid, focus, f"SHOOTER: focus R{focus}")
        candidates = [
            p for p in ring_positions(epos, FOCUS_ATTACK_R)
            if not point_blocked(p, self.obstacles, pad=0.5) and has_los(p, epos, self.obstacles)
        ]
        if not candidates:
            return self._hold_or_shoot(uid, "SHOOTER: no LOS ring")
        others = [
            (e["x"], e["y"]) for eid, e in self.enemies.items() if eid != focus
        ]

        def exposure(p):
            """표적 외 다른 적 몇 명의 응사 범위(7) 안에 서게 되는가."""
            return sum(1 for o in others if _dist(p, o) <= RED_RETURN + 0.05)

        goal = min(candidates, key=lambda p: (exposure(p), _dist(me, p)))
        return self._move_step(uid, goal, f"SHOOTER: close on R{focus}")

    def _run_security(self, uid: int) -> Dict[str, Any]:
        me = self._pos(uid)
        anchor = self._reserve_point or me
        visible = [e for e in self._visible_enemies(uid) if float(e["r"]) <= STANDOFF_HI]
        if visible and self.friendlies[uid]["ammo"] > 0:
            return self._hold_or_shoot(uid, "SECURITY", finish_hp=40.0)
        if _dist(me, anchor) > 3.0:
            return self._move_step(uid, anchor, "SECURITY: hold near reserve")
        return {"action": "STOP", "reason": "SECURITY: holding"}

    def _run_scout(self, uid: int) -> Dict[str, Any]:
        phase = self._scout_phase.get(uid, "OUT")
        me = self._pos(uid)
        if phase == "OUT" and self.enemies:
            self._scout_phase[uid] = "BACK"
            phase = "BACK"
        if phase == "OUT":
            return self._move_step(uid, (SCOUT_SEARCH_X, me[1]), "SCOUT: search east")
        goal = self._reserve_point or me
        if _dist(me, goal) > 1.5:
            return self._move_step(uid, goal, "SCOUT: return to reserve")
        return {"action": "STOP", "reason": "SCOUT: back with reserve"}

    def _run_reserve(self, uid: int) -> Dict[str, Any]:
        return self._run_security(uid)

    def _run_anchor(self, uid: int) -> Dict[str, Any]:
        visible = [e for e in self._visible_enemies(uid) if float(e["r"]) <= STANDOFF_HI]
        if visible and self.friendlies[uid]["ammo"] > 0:
            return self._hold_or_shoot(uid, "ANCHOR", finish_hp=40.0)
        return {"action": "STOP", "reason": "ANCHOR: rally point"}

    def _run_withdraw(self, uid: int) -> Dict[str, Any]:
        """탄약 고갈 유닛: 모든 알려진 적의 응사 범위 밖으로 물러난다."""
        me = self._pos(uid)
        threats = [(e["x"], e["y"]) for e in self.enemies.values()]
        if not threats or all(_dist(me, t) >= WITHDRAW_SAFE for t in threats):
            return {"action": "STOP", "reason": "WITHDRAW: out of ammo, safe"}
        goal = self._reserve_point or me
        return self._move_step(uid, goal, "WITHDRAW: out of ammo")

    def _run_regrouper(self, uid: int) -> Dict[str, Any]:
        roles = self.plan.get("roles", {})
        anchors = [u for u, r in roles.items() if r == "ANCHOR" and u in self.friendlies]
        goal = self._pos(anchors[0]) if anchors else (self._reserve_point or self._pos(uid))
        if _dist(self._pos(uid), goal) > 2.0:
            return self._move_step(uid, goal, "REGROUPER: rejoin anchor")
        return {"action": "STOP", "reason": "REGROUPER: regrouped"}
