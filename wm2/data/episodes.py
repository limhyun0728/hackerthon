"""에피소드 디렉터리 → 프레임. 구 코드 import 없이 CSV/JSON을 직접 읽는다 (설계 15절).

읽는 파일 (구 시스템이 기록한 그대로):
  soldier_log.csv   time,id,x,y,heading,hp,ammo,mode,target_id  — 1초 간격 전 유닛 상태
  commands_log.csv  time,unit_id,role,action,detail,reason      — 발행된 명령
  config.json       mission_type, objective, obstacles, blue_ids, red_ids, duration ...

주의: 구 rule 에피소드는 계획=실행이라 commands_log를 그대로 계획 명령으로 쓸 수 있다.
CEM 에피소드는 강등된 ENGAGE의 계획 표적이 로그에 없으므로 (설계 14절) 1~2단계에서는
rule 에피소드만 학습에 넣는다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..model.features import MISSION_TYPE_BY_NAME


@dataclass(frozen=True)
class UnitState:
    unit_id: int
    x: float
    y: float
    heading_deg: float
    hp: float
    ammo: float


@dataclass(frozen=True)
class Command:
    time: float
    unit_id: int
    role: str          # "CEM" | "blue_rule" | "RED" 등 — 기록된 값 그대로
    action: str        # STOP | MOVE | ENGAGE | TURN
    detail: str
    reason: str


@dataclass(frozen=True)
class Episode:
    run_dir: str
    mission_type: int
    objective: tuple[float, float]
    duration_sec: float
    blue_ids: tuple[int, ...]
    red_ids: tuple[int, ...]
    obstacles: tuple[tuple[float, float, float, float], ...]   # (xmin, ymin, xmax, ymax)
    # tick(정수 초) → unit_id → 상태. 1초 간격 프레임만 인정한다.
    frames: dict[int, dict[int, UnitState]]
    # tick → 그 tick에 발행된 명령들
    commands: dict[int, tuple[Command, ...]]

    @property
    def ticks(self) -> list[int]:
        return sorted(self.frames)


def _read_soldier_log(path: Path) -> dict[int, dict[int, UnitState]]:
    frames: dict[int, dict[int, UnitState]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            time_value = float(row["time"])
            tick = round(time_value)
            if abs(time_value - tick) > 1e-6:
                continue  # 1초 격자 밖 기록(중간 이벤트)은 프레임이 아니다
            state = UnitState(
                unit_id=int(row["id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                heading_deg=float(row["heading"]),
                hp=float(row["hp"]),
                ammo=float(row["ammo"]),
            )
            frames.setdefault(tick, {})[state.unit_id] = state
    return frames


def _read_commands(path: Path) -> dict[int, tuple[Command, ...]]:
    by_tick: dict[int, list[Command]] = {}
    if not path.exists():
        return {}
    with path.open() as f:
        for row in csv.DictReader(f):
            time_value = float(row["time"])
            tick = round(time_value)
            command = Command(
                time=time_value,
                unit_id=int(row["unit_id"]),
                role=str(row.get("role", "")),
                action=str(row.get("action", "")).upper(),
                detail=str(row.get("detail", "")),
                reason=str(row.get("reason", "")),
            )
            by_tick.setdefault(tick, []).append(command)
    return {tick: tuple(commands) for tick, commands in by_tick.items()}


def load_episode(run_dir: str | Path) -> Episode:
    """에피소드 디렉터리 하나를 읽는다. 필수 파일이 없으면 그대로 예외를 낸다."""
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text())

    mission_raw = config.get("mission_type", "destroy_and_reach")
    mission_type = (
        int(mission_raw)
        if isinstance(mission_raw, int)
        else MISSION_TYPE_BY_NAME[str(mission_raw)]
    )
    objective_raw = config.get("objective") or (10.0, 0.0)
    objective = (float(objective_raw[0]), float(objective_raw[1]))

    obstacles = tuple(
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        for r in config.get("obstacles", ())
    )

    frames = _read_soldier_log(root / "soldier_log.csv")
    if not frames:
        raise ValueError(f"{root}: soldier_log.csv에 1초 격자 프레임이 없다")

    blue_ids = tuple(int(v) for v in config.get("blue_ids", ()))
    red_ids = tuple(int(v) for v in config.get("red_ids", ()))
    if not blue_ids or not red_ids:
        # config에 없으면 관례(1xx=BLUE, 2xx=RED)로 복원한다
        first = frames[min(frames)]
        blue_ids = tuple(sorted(uid for uid in first if uid < 200))
        red_ids = tuple(sorted(uid for uid in first if uid >= 200))

    return Episode(
        run_dir=str(root),
        mission_type=mission_type,
        objective=objective,
        duration_sec=float(config.get("duration", 60.0)),
        blue_ids=blue_ids,
        red_ids=red_ids,
        obstacles=obstacles,
        frames=frames,
        commands=_read_commands(root / "commands_log.csv"),
    )


def objective_distance(episode: Episode, tick: int) -> float:
    """해당 tick에 생존 BLUE 중 목표까지 최단 거리. 전멸이면 inf."""
    rows = episode.frames.get(tick, {})
    distances = [
        math.hypot(s.x - episode.objective[0], s.y - episode.objective[1])
        for uid, s in rows.items()
        if uid in episode.blue_ids and s.hp > 0.0
    ]
    return min(distances) if distances else float("inf")
