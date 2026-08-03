"""DEVS 상태를 객체 중심 slot 벡터로 변환한다.

이 모듈은 학습 모델이 아니라 첫 번째 입력 계층이다. 현재 v2 시나리오의
구조화된 DEVS/log 상태를 유닛, 지형, 임무 slot으로 정규화하고, 이후
Object Transformer가 사용할 수 있는 고정 폭 벡터 묶음으로 만든다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN


class ObjectType(IntEnum):
    """slot이 나타내는 객체 유형."""

    UNIT = 0
    TERRAIN = 1
    MISSION = 2


class TeamId(IntEnum):
    """유닛 피아 구분값."""

    NONE = -1
    BLUE = 0
    RED = 1


# 현재 v2 시나리오의 고정 계약이다. config에 objective가 없으므로 코드의
# 임무 정의와 같은 값을 명시 상수로 둔다.
CURRENT_OBJECTIVE = (10.0, 0.0)
MISSION_DESTROY_AND_REACH = 0
OBJECTIVE_RADIUS = 1.0
MAX_HP = 100.0
MAX_AMMO = 30.0

UNIT_FEATURE_NAMES = (
    "team",
    "hp_ratio",
    "ammo_ratio",
    "x_norm",
    "y_norm",
    "heading_cos",
    "heading_sin",
    "alive",
)

TERRAIN_FEATURE_NAMES = (
    "terrain_type",
    "x_norm",
    "y_norm",
    "width_norm",
    "height_norm",
    "traversability",
    "movement_cost",
    "cover_value",
    "los_block",
)

MISSION_FEATURE_NAMES = (
    "mission_type",
    "objective_x_norm",
    "objective_y_norm",
    "time_remaining_ratio",
    "completion_flag",
)

FEATURE_NAMES_BY_TYPE = {
    ObjectType.UNIT: UNIT_FEATURE_NAMES,
    ObjectType.TERRAIN: TERRAIN_FEATURE_NAMES,
    ObjectType.MISSION: MISSION_FEATURE_NAMES,
}

MAX_FEATURE_DIM = max(len(names) for names in FEATURE_NAMES_BY_TYPE.values())


@dataclass(frozen=True)
class SlotBatch:
    """동일 시점의 모든 객체 slot 묶음."""

    features: np.ndarray
    feature_mask: np.ndarray
    type_ids: np.ndarray
    entity_ids: np.ndarray
    team_ids: np.ndarray
    alive_mask: np.ndarray
    names: tuple[str, ...]
    time_sec: float

    def __post_init__(self) -> None:
        """shape 불일치를 즉시 드러내기 위한 검증."""
        n_slots = self.features.shape[0]
        if self.features.shape != self.feature_mask.shape:
            raise ValueError("features와 feature_mask의 shape가 같아야 한다")
        for name, value in (
            ("type_ids", self.type_ids),
            ("entity_ids", self.entity_ids),
            ("team_ids", self.team_ids),
            ("alive_mask", self.alive_mask),
        ):
            if value.shape != (n_slots,):
                raise ValueError(f"{name} shape는 ({n_slots},)이어야 한다")
        if len(self.names) != n_slots:
            raise ValueError("names 길이는 slot 개수와 같아야 한다")

    def indices_of(self, object_type: ObjectType) -> np.ndarray:
        """지정한 객체 유형에 해당하는 slot index를 반환한다."""
        return np.flatnonzero(self.type_ids == int(object_type))

    def compact_features(self, index: int) -> np.ndarray:
        """padding을 제거한 단일 slot feature를 반환한다."""
        return self.features[index, self.feature_mask[index]]

    def features_of(self, object_type: ObjectType) -> np.ndarray:
        """지정한 객체 유형의 feature matrix를 padding 없이 반환한다."""
        indices = self.indices_of(object_type)
        feature_dim = len(FEATURE_NAMES_BY_TYPE[object_type])
        return self.features[indices, :feature_dim]


def _norm_x(x: float) -> float:
    """월드 x좌표를 [-1, 1] 범위로 정규화한다."""
    return 2.0 * (float(x) - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) - 1.0


def _norm_y(y: float) -> float:
    """월드 y좌표를 [-1, 1] 범위로 정규화한다."""
    return 2.0 * (float(y) - WORLD_Y_MIN) / (WORLD_Y_MAX - WORLD_Y_MIN) - 1.0


def _norm_width(width: float) -> float:
    """지형 폭을 월드 x폭으로 나눠 정규화한다."""
    return float(width) / (WORLD_X_MAX - WORLD_X_MIN)


def _norm_height(height: float) -> float:
    """지형 높이를 월드 y폭으로 나눠 정규화한다."""
    return float(height) / (WORLD_Y_MAX - WORLD_Y_MIN)


def _heading_pair(degrees: float) -> tuple[float, float]:
    """heading 각도를 cos/sin 쌍으로 변환한다."""
    radians = math.radians(float(degrees))
    return math.cos(radians), math.sin(radians)


def _team_from_unit_id(unit_id: int) -> TeamId:
    """현재 시나리오의 id 규칙으로 피아를 판정한다."""
    return TeamId.BLUE if int(unit_id) < 200 else TeamId.RED


def _padded(vector: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """타입별 feature를 공통 폭으로 padding한다."""
    if len(vector) > MAX_FEATURE_DIM:
        raise ValueError("slot feature 길이가 MAX_FEATURE_DIM보다 크다")
    features = np.zeros(MAX_FEATURE_DIM, dtype=np.float32)
    mask = np.zeros(MAX_FEATURE_DIM, dtype=bool)
    features[: len(vector)] = np.asarray(vector, dtype=np.float32)
    mask[: len(vector)] = True
    return features, mask


def unit_feature_from_row(row: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, int, TeamId, bool]:
    """soldier_log 한 행을 unit slot feature로 변환한다."""
    unit_id = int(row["id"])
    team = _team_from_unit_id(unit_id)
    hp = float(row["hp"])
    ammo = float(row["ammo"])
    alive = hp > 0.0
    heading_cos, heading_sin = _heading_pair(float(row["heading"]))

    vector = (
        float(team),
        hp / MAX_HP,
        ammo / MAX_AMMO,
        _norm_x(float(row["x"])),
        _norm_y(float(row["y"])),
        heading_cos,
        heading_sin,
        float(alive),
    )
    features, mask = _padded(vector)
    return features, mask, unit_id, team, alive


def terrain_feature_from_rect(index: int, rect: Sequence[float]) -> tuple[np.ndarray, np.ndarray, int]:
    """AABB 장애물을 terrain slot feature로 변환한다."""
    if len(rect) != 4:
        raise ValueError("terrain rect는 xmin, ymin, xmax, ymax 네 값이어야 한다")
    xmin, ymin, xmax, ymax = map(float, rect)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("terrain rect의 xmax/ymax는 xmin/ymin보다 커야 한다")

    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + width / 2.0
    center_y = ymin + height / 2.0

    # 현재 장애물은 이동과 LOS를 모두 차단하는 완전 엄폐물이다.
    vector = (
        1.0,  # terrain_type=1: AABB obstacle
        _norm_x(center_x),
        _norm_y(center_y),
        _norm_width(width),
        _norm_height(height),
        0.0,  # traversability
        1.0,  # movement_cost
        1.0,  # cover_value
        1.0,  # los_block
    )
    features, mask = _padded(vector)
    return features, mask, int(index)


def mission_feature(
    *,
    time_sec: float,
    duration_sec: float,
    unit_rows: Sequence[Mapping[str, str]],
    objective: tuple[float, float] = CURRENT_OBJECTIVE,
) -> tuple[np.ndarray, np.ndarray]:
    """현재 임무 상태를 mission slot feature로 변환한다."""
    if duration_sec <= 0.0:
        raise ValueError("duration_sec는 0보다 커야 한다")

    red_alive = any(int(row["id"]) >= 200 and float(row["hp"]) > 0.0 for row in unit_rows)
    blue_positions = [
        (float(row["x"]), float(row["y"]))
        for row in unit_rows
        if int(row["id"]) < 200 and float(row["hp"]) > 0.0
    ]
    objective_reached = any(
        math.hypot(x - objective[0], y - objective[1]) <= OBJECTIVE_RADIUS
        for x, y in blue_positions
    )
    completion_flag = (not red_alive) and objective_reached
    time_remaining_ratio = max(0.0, (float(duration_sec) - float(time_sec)) / float(duration_sec))

    vector = (
        float(MISSION_DESTROY_AND_REACH),
        _norm_x(objective[0]),
        _norm_y(objective[1]),
        time_remaining_ratio,
        float(completion_flag),
    )
    return _padded(vector)


def build_slot_batch(
    *,
    unit_rows: Sequence[Mapping[str, str]],
    obstacles: Sequence[Sequence[float]],
    time_sec: float,
    duration_sec: float,
    objective: tuple[float, float] = CURRENT_OBJECTIVE,
) -> SlotBatch:
    """유닛, 지형, 임무 상태를 하나의 slot batch로 묶는다."""
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    type_ids: list[int] = []
    entity_ids: list[int] = []
    team_ids: list[int] = []
    alive_mask: list[bool] = []
    names: list[str] = []

    for row in sorted(unit_rows, key=lambda item: int(item["id"])):
        unit_features, unit_mask, unit_id, team, alive = unit_feature_from_row(row)
        features.append(unit_features)
        masks.append(unit_mask)
        type_ids.append(int(ObjectType.UNIT))
        entity_ids.append(unit_id)
        team_ids.append(int(team))
        alive_mask.append(alive)
        names.append(("B" if team == TeamId.BLUE else "R") + str(unit_id))

    for index, rect in enumerate(obstacles, start=1):
        terrain_features, terrain_mask, terrain_id = terrain_feature_from_rect(index, rect)
        features.append(terrain_features)
        masks.append(terrain_mask)
        type_ids.append(int(ObjectType.TERRAIN))
        entity_ids.append(terrain_id)
        team_ids.append(int(TeamId.NONE))
        alive_mask.append(True)
        names.append(f"T{terrain_id:03d}")

    mission_features, mission_mask = mission_feature(
        time_sec=time_sec,
        duration_sec=duration_sec,
        unit_rows=unit_rows,
        objective=objective,
    )
    features.append(mission_features)
    masks.append(mission_mask)
    type_ids.append(int(ObjectType.MISSION))
    entity_ids.append(0)
    team_ids.append(int(TeamId.NONE))
    alive_mask.append(True)
    names.append("MISSION")

    return SlotBatch(
        features=np.stack(features, axis=0),
        feature_mask=np.stack(masks, axis=0),
        type_ids=np.asarray(type_ids, dtype=np.int64),
        entity_ids=np.asarray(entity_ids, dtype=np.int64),
        team_ids=np.asarray(team_ids, dtype=np.int64),
        alive_mask=np.asarray(alive_mask, dtype=bool),
        names=tuple(names),
        time_sec=float(time_sec),
    )


def load_v2_config(run_dir: Path) -> Mapping[str, object]:
    """v2 run 디렉터리의 config를 읽는다."""
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_unit_rows_at_time(run_dir: Path, time_sec: float) -> list[dict[str, str]]:
    """soldier_log.csv에서 지정 시점의 유닛 상태 행을 읽는다."""
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


def build_slot_batch_from_v2_run(run_dir: Path, time_sec: float) -> SlotBatch:
    """v2 run 로그에서 같은 시나리오의 slot batch를 만든다."""
    config = load_v2_config(run_dir)
    unit_rows = load_unit_rows_at_time(run_dir, time_sec)
    obstacles = config["obstacles"]
    duration_sec = float(config["duration"])
    if not isinstance(obstacles, list):
        raise TypeError("config['obstacles']는 list여야 한다")
    return build_slot_batch(
        unit_rows=unit_rows,
        obstacles=obstacles,
        time_sec=time_sec,
        duration_sec=duration_sec,
    )


def available_times(run_dir: Path) -> list[float]:
    """soldier_log.csv에 존재하는 시점 목록을 반환한다."""
    log_path = run_dir / "soldier_log.csv"
    times: set[float] = set()
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            times.add(float(row["time"]))
    return sorted(times)


def _format_slot_line(batch: SlotBatch, index: int) -> str:
    """CLI 확인용으로 slot 하나를 짧게 문자열화한다."""
    object_type = ObjectType(int(batch.type_ids[index]))
    feature_names = FEATURE_NAMES_BY_TYPE[object_type]
    values = batch.compact_features(index)
    pairs = ", ".join(f"{name}={value:.3f}" for name, value in zip(feature_names, values))
    return f"{index:02d} {batch.names[index]} {object_type.name}: {pairs}"


def main(argv: Iterable[str] | None = None) -> None:
    """특정 run/time을 slot batch로 변환해 shape와 앞부분을 출력한다."""
    parser = argparse.ArgumentParser(description="v2 DEVS 로그를 object slot 벡터로 변환")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)

    batch = build_slot_batch_from_v2_run(args.run_dir, args.time)
    print(f"time={batch.time_sec}")
    print(f"features_shape={batch.features.shape}")
    print(f"type_ids={batch.type_ids.tolist()}")
    print(f"names={list(batch.names)}")
    for index in range(min(args.limit, batch.features.shape[0])):
        print(_format_slot_line(batch, index))


if __name__ == "__main__":
    main()
