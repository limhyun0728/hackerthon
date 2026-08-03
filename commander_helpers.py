import math
import random
from typing import Any, Dict, List, Tuple

from hackerthon.combat_config import MAX_FIRE_RANGE
from hackerthon.commander_tactics import TACTIC_LIBRARY


# 지휘관 지도와 Stage 4 grid mark가 공유하는 월드 좌표 범위.
GRID_COLUMNS = "ABCDEFGHIJ"
GRID_ROWS = 10
GRID_X_MIN = -20.0
GRID_X_MAX = 20.0
GRID_Y_MIN = -15.0
GRID_Y_MAX = 10.0


def alive_ids(intel_buffer: Dict[int, Any]) -> List[int]:
    """현재 교신 중이고 살아 있는 Blue 유닛 ID를 반환한다."""
    return sorted(
        int(uid)
        for uid, observation in intel_buffer.items()
        if float(observation["self"]["hp"]) > 0.0
    )


def is_live_enemy(entity: Dict[str, Any]) -> bool:
    """관측 개체가 살아 있는 Red인지 판정한다."""
    return (
        entity["type"] == "enemy"
        and entity["state"] != "DESTROYED"
        and float(entity["hp"]) > 0.0
    )


def visible_enemies_for_unit(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """공유 belief가 아니라 해당 유닛이 직접 관측한 살아 있는 적을 반환한다."""
    enemies = [
        entity
        for entity in observation["visible_entities"]
        if is_live_enemy(entity)
    ]
    return sorted(enemies, key=lambda entity: float(entity["r"]))


def build_allowed_target_marks(
    intel_buffer: Dict[int, Any],
    max_fire_range: float = MAX_FIRE_RANGE,
) -> Dict[int, List[str]]:
    """직접 관측·탄약·최대 사거리를 만족하는 유닛별 Red mark를 만든다."""
    allowed_targets: Dict[int, List[str]] = {}
    for uid in alive_ids(intel_buffer):
        observation = intel_buffer[uid]
        if int(observation["self"]["ammo"]) <= 0:
            allowed_targets[uid] = []
            continue

        allowed_targets[uid] = [
            f"R{int(enemy['id'])}"
            for enemy in visible_enemies_for_unit(observation)
            if float(enemy["r"]) <= float(max_fire_range)
        ]
    return allowed_targets


def validate_situation_assessment(
    intel_buffer: Dict[int, Any],
    assessment: Dict[str, Any],
) -> None:
    """Stage 1의 아군 생존 수와 부상자 목록을 현재 관측과 대조한다."""
    living = alive_ids(intel_buffer)
    wounded = sorted(
        uid
        for uid in living
        if float(intel_buffer[uid]["self"]["hp"]) < 100.0
    )
    errors = []
    if int(assessment["friendly_alive"]) != len(living):
        errors.append(
            f"friendly_alive must be {len(living)}, got {assessment['friendly_alive']}"
        )
    if sorted(int(uid) for uid in assessment["friendly_wounded"]) != wounded:
        errors.append(
            f"friendly_wounded must be {wounded}, got {assessment['friendly_wounded']}"
        )
    if errors:
        raise ValueError("Invalid Stage 1 assessment:\n- " + "\n- ".join(errors))


def validate_tactical_plan(
    intel_buffer: Dict[int, Any],
    tactical_plan: Dict[str, Any],
) -> None:
    """Stage 2의 유닛 배정 범위와 전술별 역할 구성을 검사한다."""
    living = alive_ids(intel_buffer)
    assignments = tactical_plan["assignments"]
    assigned_ids = sorted(int(item["unit_id"]) for item in assignments)
    tactic = tactical_plan["tactic"]
    allowed_roles = set(TACTIC_LIBRARY[tactic]["roles"])
    selected_roles = {item["role"] for item in assignments}
    errors = []

    if assigned_ids != living:
        errors.append(
            f"assignments must cover living ids exactly: living={living}, assigned={assigned_ids}"
        )

    invalid_roles = sorted(selected_roles - allowed_roles)
    if invalid_roles:
        errors.append(f"roles not allowed for {tactic}: {invalid_roles}")

    missing_roles = sorted(
        set(TACTIC_LIBRARY[tactic]["required_roles"]) - selected_roles
    )
    if missing_roles:
        errors.append(f"required roles missing for {tactic}: {missing_roles}")

    if errors:
        raise ValueError("Invalid Stage 2 tactical plan:\n- " + "\n- ".join(errors))


def validate_tick_action_plan(
    intel_buffer: Dict[int, Any],
    action_plan: Dict[str, Any],
    allowed_targets: Dict[int, List[str]],
) -> None:
    """Stage 3의 생존 유닛 전체 배정과 ENGAGE 가능 여부를 검사한다."""
    living = alive_ids(intel_buffer)
    actions = action_plan["actions"]
    action_ids = sorted(int(item["unit_id"]) for item in actions)
    errors = []

    if action_ids != living:
        errors.append(
            f"actions must cover living ids exactly: living={living}, assigned={action_ids}"
        )

    for item in actions:
        uid = int(item["unit_id"])
        if uid not in allowed_targets:
            errors.append(f"unit {uid}: not a living command unit")
            continue
        if item["action"] == "ENGAGE" and not allowed_targets[uid]:
            errors.append(f"unit {uid}: ENGAGE requires at least one allowed target")

    if errors:
        raise ValueError("Invalid Stage 3 action plan:\n- " + "\n- ".join(errors))


def all_grid_marks() -> List[str]:
    """Stage 4에서 사용할 A1~J10 전체 mark를 반환한다."""
    return [
        f"{column}{row}"
        for column in GRID_COLUMNS
        for row in range(1, GRID_ROWS + 1)
    ]


def grid_cell_bounds(mark: str) -> Tuple[float, float, float, float]:
    """A1~J10 mark를 월드 좌표 경계로 변환한다."""
    mark = str(mark).strip().upper()
    column = mark[:1]
    row_text = mark[1:]
    if (
        column not in GRID_COLUMNS
        or not row_text.isdigit()
        or not 1 <= int(row_text) <= GRID_ROWS
    ):
        raise ValueError(f"Invalid grid mark: {mark}")

    column_index = GRID_COLUMNS.index(column)
    row_index = int(row_text) - 1
    cell_width = (GRID_X_MAX - GRID_X_MIN) / len(GRID_COLUMNS)
    cell_height = (GRID_Y_MAX - GRID_Y_MIN) / GRID_ROWS
    x_min = GRID_X_MIN + column_index * cell_width
    y_min = GRID_Y_MIN + row_index * cell_height
    return x_min, x_min + cell_width, y_min, y_min + cell_height


def grid_mark_for_point(x: float, y: float) -> str:
    """현재 월드 좌표가 속한 A1~J10 cell을 반환한다."""
    x = float(x)
    y = float(y)
    if not GRID_X_MIN <= x <= GRID_X_MAX or not GRID_Y_MIN <= y <= GRID_Y_MAX:
        raise ValueError(f"Point outside action grid: x={x}, y={y}")

    cell_width = (GRID_X_MAX - GRID_X_MIN) / len(GRID_COLUMNS)
    cell_height = (GRID_Y_MAX - GRID_Y_MIN) / GRID_ROWS
    column_index = min(int((x - GRID_X_MIN) / cell_width), len(GRID_COLUMNS) - 1)
    row_index = min(int((y - GRID_Y_MIN) / cell_height), GRID_ROWS - 1)
    return f"{GRID_COLUMNS[column_index]}{row_index + 1}"


def build_current_grid_marks(intel_buffer: Dict[int, Any]) -> Dict[int, str]:
    """생존 유닛별 현재 cell을 계산해 Stage 4의 금지 mark로 사용한다."""
    return {
        uid: grid_mark_for_point(
            float(intel_buffer[uid]["self"]["x"]),
            float(intel_buffer[uid]["self"]["y"]),
        )
        for uid in alive_ids(intel_buffer)
    }


def sample_grid_point(mark: str) -> Tuple[float, float]:
    """새 MOVE cell을 처음 선택했을 때 목표점을 균등 샘플링한다."""
    x_min, x_max, y_min, y_max = grid_cell_bounds(mark)
    return random.uniform(x_min, x_max), random.uniform(y_min, y_max)


def validate_marked_action_plan(
    action_plan: Dict[str, Any],
    marked_plan: Dict[str, Any],
    allowed_targets: Dict[int, List[str]],
    current_grid_marks: Dict[int, str],
) -> None:
    """Stage 4 mark가 Stage 3 행동과 허용 목록에 맞는지 검사한다."""
    action_by_unit = {
        int(item["unit_id"]): item["action"]
        for item in action_plan["actions"]
    }
    selections = marked_plan["selections"]
    selection_ids = sorted(int(item["unit_id"]) for item in selections)
    errors = []

    if selection_ids != sorted(action_by_unit):
        errors.append(
            "selections must cover action units exactly: "
            f"expected={sorted(action_by_unit)}, selected={selection_ids}"
        )

    valid_grid_marks = set(all_grid_marks())
    for item in selections:
        uid = int(item["unit_id"])
        mark = str(item["mark"]).upper()
        if uid not in action_by_unit:
            errors.append(f"unit {uid}: not present in the Stage 3 action plan")
            continue
        action = action_by_unit[uid]
        if action == "ENGAGE" and mark not in allowed_targets[uid]:
            errors.append(
                f"unit {uid}: target mark {mark} not in {allowed_targets[uid]}"
            )
        elif action == "MOVE":
            if mark not in valid_grid_marks:
                errors.append(f"unit {uid}: MOVE requires A1~J10, got {mark}")
            elif mark == current_grid_marks[uid]:
                errors.append(f"unit {uid}: MOVE cannot select current cell {mark}")
        elif action == "HOLD" and mark != "HOLD":
            errors.append(f"unit {uid}: HOLD requires mark=HOLD, got {mark}")

    if errors:
        raise ValueError("Invalid Stage 4 marked plan:\n- " + "\n- ".join(errors))


def _clamp_waypoint(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    max_move_per_step: float,
) -> Tuple[float, float]:
    """목표점 방향으로 한 tick 최대 이동 거리만큼의 waypoint를 계산한다."""
    dx = target_x - current_x
    dy = target_y - current_y
    distance = math.hypot(dx, dy)
    if distance <= max_move_per_step:
        return round(target_x, 6), round(target_y, 6)
    scale = max_move_per_step / distance
    return round(current_x + dx * scale, 6), round(current_y + dy * scale, 6)


def build_command_sequence(
    intel_buffer: Dict[int, Any],
    action_plan: Dict[str, Any],
    marked_plan: Dict[str, Any],
    allowed_targets: Dict[int, List[str]],
    current_grid_marks: Dict[int, str],
    move_goal_cache: Dict[int, Dict[str, Any]],
    max_move_per_step: float,
) -> Dict[str, Any]:
    """Stage 3 행동과 Stage 4 mark를 Stage 5 실행 명령으로 변환한다."""
    validate_tick_action_plan(intel_buffer, action_plan, allowed_targets)
    validate_marked_action_plan(
        action_plan,
        marked_plan,
        allowed_targets,
        current_grid_marks,
    )

    action_by_unit = {
        int(item["unit_id"]): item["action"]
        for item in action_plan["actions"]
    }
    selection_by_unit = {
        int(item["unit_id"]): item
        for item in marked_plan["selections"]
    }
    living = alive_ids(intel_buffer)
    for cached_uid in list(move_goal_cache):
        if cached_uid not in living:
            del move_goal_cache[cached_uid]

    commands = []
    for uid in living:
        action = action_by_unit[uid]
        selection = selection_by_unit[uid]
        mark = str(selection["mark"]).upper()
        reason = str(selection["reason"])

        if action == "ENGAGE":
            move_goal_cache.pop(uid, None)
            commands.append({
                "unit_id": uid,
                "action": "ENGAGE",
                "target_id": int(mark[1:]),
                "reason": reason,
            })
        elif action == "MOVE":
            current_x = float(intel_buffer[uid]["self"]["x"])
            current_y = float(intel_buffer[uid]["self"]["y"])
            cached_goal = move_goal_cache.get(uid)
            if cached_goal is not None and cached_goal["mark"] == mark:
                sampled_x = float(cached_goal["x"])
                sampled_y = float(cached_goal["y"])
            else:
                sampled_x, sampled_y = sample_grid_point(mark)
                move_goal_cache[uid] = {
                    "mark": mark,
                    "x": sampled_x,
                    "y": sampled_y,
                }
            move_x, move_y = _clamp_waypoint(
                current_x,
                current_y,
                sampled_x,
                sampled_y,
                max_move_per_step,
            )
            commands.append({
                "unit_id": uid,
                "action": "MOVE",
                "x": move_x,
                "y": move_y,
                "reason": (
                    f"{reason}; cell={mark}; "
                    f"goal=({sampled_x:.3f},{sampled_y:.3f})"
                ),
            })
        else:
            move_goal_cache.pop(uid, None)
            commands.append({
                "unit_id": uid,
                "action": "STOP",
                "reason": reason,
            })

    return {
        "rationale": marked_plan["rationale"],
        "sequence_duration_sec": 1.0,
        "frames": [{"tick": 0, "commands": commands}],
    }


def sequence_frame_to_actions(
    intel_buffer: Dict[int, Any],
    frame: Dict[str, Any],
    shooting_range: float,
    max_move_per_step: float,
    plan: Dict[str, Any] | None = None,
    frame_tick: int = 0,
) -> List[Dict[str, Any]]:
    """Stage 5 명령을 검증해 시뮬레이터 입력 형식으로 전달한다."""
    del shooting_range, plan, frame_tick
    living = alive_ids(intel_buffer)
    by_unit = {
        int(command["unit_id"]): command
        for command in frame["commands"]
    }
    if sorted(by_unit) != living:
        raise ValueError(
            f"frame must command every living unit: living={living}, assigned={sorted(by_unit)}"
        )

    executable = []
    for uid in living:
        command = by_unit[uid]
        action = command["action"]
        output = {
            "unit_id": uid,
            "action": action,
            "duration_sec": 1.0,
            "reason": command["reason"],
        }

        if action == "MOVE":
            current_x = float(intel_buffer[uid]["self"]["x"])
            current_y = float(intel_buffer[uid]["self"]["y"])
            target_x = float(command["x"])
            target_y = float(command["y"])
            if math.hypot(target_x - current_x, target_y - current_y) > max_move_per_step + 1e-6:
                raise ValueError(f"unit {uid}: MOVE exceeds max_move_per_step")
            output.update({"x": target_x, "y": target_y})
        elif action == "ENGAGE":
            output["target_id"] = int(command["target_id"])
        elif action != "STOP":
            raise ValueError(f"unit {uid}: invalid action={action}")

        executable.append(output)

    return executable
