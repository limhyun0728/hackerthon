"""DEVS 명령을 유닛별 action slot 벡터로 변환한다.

여기서 joint action은 모든 유닛이 같은 행동을 한다는 뜻이 아니라,
같은 tick에 각 유닛에게 내려간 DEVS 명령들의 묶음이다. 현재
Soldier가 실제 처리하는 `MOVE`, `ENGAGE`, `TURN`, `STOP`만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.worldmodel.slots import TeamId, _norm_x, _norm_y, _team_from_unit_id


class ActionType(IntEnum):
    """현재 DEVS Soldier가 처리하는 명령 유형."""

    STOP = 0
    MOVE = 1
    ENGAGE = 2
    TURN = 3


ACTION_FEATURE_NAMES = (
    "action_type",
    "has_destination",
    "destination_x_norm",
    "destination_y_norm",
    "has_target",
    "target_team",
    "target_x_norm",
    "target_y_norm",
    "has_theta",
    "theta_cos",
    "theta_sin",
)

ACTION_DIM = len(ACTION_FEATURE_NAMES)
NO_COMMAND_TYPE_ID = -1
NO_TARGET_ENTITY_ID = -1
MOVE_DETAIL_RE = re.compile(
    r"^\(\s*(?P<x>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<y>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$"
)
ENGAGE_DETAIL_RE = re.compile(r"^->(?P<prefix>[BR])(?P<id>\d+)$")


@dataclass(frozen=True)
class ActionBatch:
    """한 command tick의 유닛별 action 묶음."""

    features: np.ndarray
    issued_mask: np.ndarray
    unit_ids: np.ndarray
    action_type_ids: np.ndarray
    target_entity_ids: np.ndarray
    names: tuple[str, ...]
    time_sec: float

    def __post_init__(self) -> None:
        """shape 불일치와 잘못된 action id를 즉시 드러낸다."""
        n_units = self.features.shape[0]
        if self.features.shape != (n_units, ACTION_DIM):
            raise ValueError(f"features shape는 ({n_units}, {ACTION_DIM})이어야 한다")
        for name, value in (
            ("issued_mask", self.issued_mask),
            ("unit_ids", self.unit_ids),
            ("action_type_ids", self.action_type_ids),
            ("target_entity_ids", self.target_entity_ids),
        ):
            if value.shape != (n_units,):
                raise ValueError(f"{name} shape는 ({n_units},)이어야 한다")
        if len(self.names) != n_units:
            raise ValueError("names 길이는 action row 개수와 같아야 한다")
        valid_ids = {int(action_type) for action_type in ActionType}
        for action_type_id in self.action_type_ids[self.issued_mask]:
            if int(action_type_id) not in valid_ids:
                raise ValueError(f"알 수 없는 action_type_id: {action_type_id}")

    def features_for_issued(self) -> np.ndarray:
        """실제로 명령이 내려간 유닛의 action feature만 반환한다."""
        return self.features[self.issued_mask]


def _blank_feature() -> np.ndarray:
    """명령이 없는 유닛의 빈 action feature를 만든다."""
    return np.zeros(ACTION_DIM, dtype=np.float32)


def unit_name(unit_id: int) -> str:
    """unit id를 action token 이름으로 바꾼다."""
    team = _team_from_unit_id(int(unit_id))
    if team == TeamId.BLUE:
        return f"B{int(unit_id)}"
    if team == TeamId.RED:
        return f"R{int(unit_id)}"
    raise ValueError(f"BLUE/RED에 속하지 않는 unit id: {unit_id}")


def _theta_pair(degrees: float) -> tuple[float, float]:
    """상대 회전각을 cos/sin 쌍으로 변환한다."""
    radians = math.radians(float(degrees))
    return math.cos(radians), math.sin(radians)


def _state_by_unit_id(unit_rows: Sequence[Mapping[str, str]]) -> dict[int, Mapping[str, str]]:
    """현재 state row를 unit id로 조회할 수 있게 만든다."""
    out: dict[int, Mapping[str, str]] = {}
    for row in unit_rows:
        unit_id = int(row["id"])
        if unit_id in out:
            raise ValueError(f"중복 unit state row: {unit_id}")
        out[unit_id] = row
    return out


def _parse_move_destination(command: Mapping[str, object]) -> tuple[float, float]:
    """DEVS MOVE 명령의 절대 목적지를 읽는다."""
    if "x" in command and "y" in command and command["x"] != "" and command["y"] != "":
        return float(command["x"]), float(command["y"])
    detail = str(command["detail"])
    match = MOVE_DETAIL_RE.match(detail)
    if match is None:
        raise ValueError(f"MOVE detail에서 목적지를 읽을 수 없다: {detail!r}")
    return float(match.group("x")), float(match.group("y"))


def _parse_engage_target(command: Mapping[str, object]) -> int:
    """DEVS ENGAGE 명령의 target id를 읽는다."""
    if "target_id" in command and command["target_id"] not in ("", None):
        return int(command["target_id"])
    detail = str(command["detail"])
    match = ENGAGE_DETAIL_RE.match(detail)
    if match is None:
        raise ValueError(f"ENGAGE detail에서 표적을 읽을 수 없다: {detail!r}")
    return int(match.group("id"))


def _parse_turn_theta(command: Mapping[str, object]) -> tuple[bool, float]:
    """DEVS TURN 명령의 theta를 읽는다.

    온라인 CEM 명령은 theta 필드를 직접 갖고, CSV/요약 row는 detail에
    각도 문자열을 남긴다. 둘 다 없을 때만 theta_unknown으로 둔다.
    """
    if "theta" in command and command["theta"] not in ("", None):
        return True, float(command["theta"])
    if "detail" in command and command["detail"] not in ("", None):
        return True, float(command["detail"])
    return False, 0.0


def action_feature_from_command(
    command: Mapping[str, object],
    *,
    current_unit_rows: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, ActionType, int]:
    """DEVS command dict 또는 commands_log row를 action feature로 변환한다."""
    state_by_id = _state_by_unit_id(current_unit_rows)
    action_name = str(command["action"]).upper()
    feature = _blank_feature()

    if action_name == "STOP":
        feature[0] = float(ActionType.STOP)
        return feature, ActionType.STOP, NO_TARGET_ENTITY_ID

    if action_name == "MOVE":
        x, y = _parse_move_destination(command)
        feature[0] = float(ActionType.MOVE)
        feature[1] = 1.0
        feature[2] = _norm_x(x)
        feature[3] = _norm_y(y)
        return feature, ActionType.MOVE, NO_TARGET_ENTITY_ID

    if action_name == "ENGAGE":
        target_id = _parse_engage_target(command)
        if target_id not in state_by_id:
            raise ValueError(f"현재 state에 ENGAGE target {target_id}가 없다")
        target_row = state_by_id[target_id]
        target_team = _team_from_unit_id(target_id)
        feature[0] = float(ActionType.ENGAGE)
        feature[4] = 1.0
        feature[5] = float(target_team)
        feature[6] = _norm_x(float(target_row["x"]))
        feature[7] = _norm_y(float(target_row["y"]))
        return feature, ActionType.ENGAGE, target_id

    if action_name == "TURN":
        theta_known, theta = _parse_turn_theta(command)
        theta_cos, theta_sin = _theta_pair(theta)
        feature[0] = float(ActionType.TURN)
        feature[8] = float(theta_known)
        feature[9] = theta_cos if theta_known else 0.0
        feature[10] = theta_sin if theta_known else 0.0
        return feature, ActionType.TURN, NO_TARGET_ENTITY_ID

    raise ValueError(f"DEVS action vocabulary에 없는 명령: {action_name!r}")


def build_action_batch(
    *,
    blue_unit_ids: Sequence[int] | None = None,
    unit_ids: Sequence[int] | None = None,
    command_rows: Sequence[Mapping[str, object]],
    current_unit_rows: Sequence[Mapping[str, str]],
    time_sec: float,
    allow_extra_commands: bool = False,
) -> ActionBatch:
    """지정 unit id 순서에 맞춰 한 tick의 action batch를 만든다."""
    if unit_ids is None:
        if blue_unit_ids is None:
            raise ValueError("unit_ids 또는 blue_unit_ids 중 하나는 필요하다")
        ordered_unit_ids = tuple(int(unit_id) for unit_id in blue_unit_ids)
    else:
        if blue_unit_ids is not None:
            raise ValueError("unit_ids와 blue_unit_ids를 동시에 넘길 수 없다")
        ordered_unit_ids = tuple(int(unit_id) for unit_id in unit_ids)
    if len(set(ordered_unit_ids)) != len(ordered_unit_ids):
        raise ValueError(f"action batch unit id가 중복됐다: {ordered_unit_ids}")

    command_by_unit: dict[int, Mapping[str, object]] = {}
    for row in command_rows:
        unit_id = int(row["unit_id"])
        if unit_id in command_by_unit:
            raise ValueError(f"같은 tick에 unit {unit_id} 명령이 중복됐다")
        command_by_unit[unit_id] = row

    features: list[np.ndarray] = []
    issued_mask: list[bool] = []
    action_type_ids: list[int] = []
    target_entity_ids: list[int] = []
    names: list[str] = []

    for unit_id in ordered_unit_ids:
        names.append(unit_name(int(unit_id)))
        command = command_by_unit.get(int(unit_id))
        if command is None:
            features.append(_blank_feature())
            issued_mask.append(False)
            action_type_ids.append(NO_COMMAND_TYPE_ID)
            target_entity_ids.append(NO_TARGET_ENTITY_ID)
            continue

        feature, action_type, target_id = action_feature_from_command(
            command,
            current_unit_rows=current_unit_rows,
        )
        features.append(feature)
        issued_mask.append(True)
        action_type_ids.append(int(action_type))
        target_entity_ids.append(target_id)

    extra_units = sorted(set(command_by_unit) - set(ordered_unit_ids))
    if extra_units and not allow_extra_commands:
        raise ValueError(f"unit_ids에 없는 명령 대상이 있다: {extra_units}")

    return ActionBatch(
        features=np.stack(features, axis=0),
        issued_mask=np.asarray(issued_mask, dtype=bool),
        unit_ids=np.asarray(ordered_unit_ids, dtype=np.int64),
        action_type_ids=np.asarray(action_type_ids, dtype=np.int64),
        target_entity_ids=np.asarray(target_entity_ids, dtype=np.int64),
        names=tuple(names),
        time_sec=float(time_sec),
    )


def load_command_rows_at_time(run_dir: Path, time_sec: float) -> list[dict[str, str]]:
    """commands_log.csv에서 지정 command tick의 명령 행을 읽는다."""
    log_path = run_dir / "commands_log.csv"
    rows: list[dict[str, str]] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if float(row["time"]) == float(time_sec):
                rows.append(row)
    if not rows:
        raise ValueError(f"{log_path}에 time={time_sec} 명령이 없다")
    return rows


def load_unit_rows_at_time(run_dir: Path, time_sec: float) -> list[dict[str, str]]:
    """soldier_log.csv에서 지정 state tick의 유닛 상태 행을 읽는다."""
    log_path = run_dir / "soldier_log.csv"
    rows: list[dict[str, str]] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if float(row["time"]) == float(time_sec):
                rows.append(row)
    if not rows:
        raise ValueError(f"{log_path}에 time={time_sec} 상태가 없다")
    return rows


def blue_unit_ids_from_rows(unit_rows: Sequence[Mapping[str, str]]) -> list[int]:
    """state row에서 아군 unit id를 정렬해 추출한다."""
    return sorted(int(row["id"]) for row in unit_rows if _team_from_unit_id(int(row["id"])) == TeamId.BLUE)


def build_action_batch_from_v2_run(
    run_dir: Path,
    *,
    command_time_sec: float,
    state_time_sec: float,
) -> ActionBatch:
    """v2 run 로그에서 command tick의 action batch를 만든다."""
    current_unit_rows = load_unit_rows_at_time(run_dir, state_time_sec)
    command_rows = load_command_rows_at_time(run_dir, command_time_sec)
    blue_unit_ids = blue_unit_ids_from_rows(current_unit_rows)
    return build_action_batch(
        blue_unit_ids=blue_unit_ids,
        command_rows=command_rows,
        current_unit_rows=current_unit_rows,
        time_sec=command_time_sec,
        allow_extra_commands=True,
    )


def available_command_times(run_dir: Path) -> list[float]:
    """commands_log.csv에 존재하는 command tick 목록을 반환한다."""
    log_path = run_dir / "commands_log.csv"
    times: set[float] = set()
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.add(float(row["time"]))
    return sorted(times)


def _format_action_line(batch: ActionBatch, index: int) -> str:
    """CLI 확인용으로 action 하나를 짧게 문자열화한다."""
    if not batch.issued_mask[index]:
        return f"{index:02d} {batch.names[index]} NO_COMMAND"
    action_type = ActionType(int(batch.action_type_ids[index]))
    pairs = ", ".join(
        f"{name}={value:.3f}"
        for name, value in zip(ACTION_FEATURE_NAMES, batch.features[index])
    )
    target_id = int(batch.target_entity_ids[index])
    target = "" if target_id == NO_TARGET_ENTITY_ID else f" target={target_id}"
    return f"{index:02d} {batch.names[index]} {action_type.name}{target}: {pairs}"


def main(argv: Iterable[str] | None = None) -> None:
    """특정 run/time을 action batch로 변환해 shape와 앞부분을 출력한다."""
    parser = argparse.ArgumentParser(description="v2 DEVS 명령 로그를 action slot 벡터로 변환")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--command-time", type=float, required=True)
    parser.add_argument("--state-time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    batch = build_action_batch_from_v2_run(
        args.run_dir,
        command_time_sec=args.command_time,
        state_time_sec=args.state_time,
    )
    print(f"command_time={batch.time_sec}")
    print(f"features_shape={batch.features.shape}")
    print(f"issued_mask={batch.issued_mask.tolist()}")
    print(f"action_type_ids={batch.action_type_ids.tolist()}")
    print(f"target_entity_ids={batch.target_entity_ids.tolist()}")
    for index in range(min(args.limit, batch.features.shape[0])):
        print(_format_action_line(batch, index))


if __name__ == "__main__":
    main()
