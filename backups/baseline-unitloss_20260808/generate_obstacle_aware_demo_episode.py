"""Generate a NAVER demo episode whose units stay outside real buildings."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.real_building_obstacles import _point_in_polygon
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN, point_blocked


GRID_RES = 0.5
DEFAULT_CLEARANCE_UNITS = 0.08


UnitPlan = tuple[int, str, tuple[tuple[float, float], ...]]


def _load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _write_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _obstacles(config: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    return [
        (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
        for rect in config.get("obstacles", []) or []
        if len(rect) == 4
    ]


def _building_polygons(config: Mapping[str, Any]) -> list[tuple[tuple[float, float], ...]]:
    polygons = []
    for building in config.get("building_polygons", []) or []:
        points = building.get("points", []) if isinstance(building, Mapping) else building
        polygon = tuple((float(point[0]), float(point[1])) for point in points if len(point) >= 2)
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _inside_display_building(point: tuple[float, float], polygons: Sequence[tuple[tuple[float, float], ...]]) -> bool:
    return any(_point_in_polygon(point[0], point[1], polygon) for polygon in polygons)


def _segment_hits_building(
    start: tuple[float, float],
    end: tuple[float, float],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    sample_step: float = 0.12,
) -> bool:
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(1, math.ceil(distance / sample_step))
    for index in range(steps + 1):
        ratio = index / steps
        point = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        if _inside_display_building(point, polygons):
            return True
    return False


def _free(
    point: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    clearance: float,
) -> bool:
    x, y = point
    if x < WORLD_X_MIN or x > WORLD_X_MAX or y < WORLD_Y_MIN or y > WORLD_Y_MAX:
        return False
    if point_blocked(point, list(obstacles), pad=clearance):
        return False
    return not _inside_display_building(point, polygons)


def _nearest_free(
    desired: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    clearance: float,
) -> tuple[float, float]:
    candidates = []
    max_radius = 30
    for ix in range(-max_radius, max_radius + 1):
        for iy in range(-max_radius, max_radius + 1):
            x = round((desired[0] + ix * GRID_RES) / GRID_RES) * GRID_RES
            y = round((desired[1] + iy * GRID_RES) / GRID_RES) * GRID_RES
            distance = math.hypot(x - desired[0], y - desired[1])
            candidates.append((distance, x, y))
    for _, x, y in sorted(candidates):
        point = (x, y)
        if _free(point, obstacles, polygons, clearance=clearance):
            return point
    raise ValueError(f"walkable point를 찾지 못했다: {desired}")


def _cell(point: tuple[float, float]) -> tuple[int, int]:
    return (round(point[0] / GRID_RES), round(point[1] / GRID_RES))


def _point(cell: tuple[int, int]) -> tuple[float, float]:
    return (cell[0] * GRID_RES, cell[1] * GRID_RES)


def _path(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    clearance: float,
) -> list[tuple[float, float]]:
    start = _nearest_free(start, obstacles, polygons, clearance=clearance)
    goal = _nearest_free(goal, obstacles, polygons, clearance=clearance)
    start_cell = _cell(start)
    goal_cell = _cell(goal)
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
    g_score = {start_cell: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    found = False
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            found = True
            break
        current_point = _point(current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                neighbor_point = _point(neighbor)
                if not _free(neighbor_point, obstacles, polygons, clearance=clearance):
                    continue
                if dx and dy:
                    side_a = _point((current[0] + dx, current[1]))
                    side_b = _point((current[0], current[1] + dy))
                    if not (
                        _free(side_a, obstacles, polygons, clearance=clearance)
                        and _free(side_b, obstacles, polygons, clearance=clearance)
                    ):
                        continue
                if _segment_hits_building(current_point, neighbor_point, polygons):
                    continue
                cost = g_score[current] + math.hypot(dx, dy)
                if cost < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = cost
                    came[neighbor] = current
                    heapq.heappush(open_heap, (cost + heuristic(neighbor), neighbor))

    if not found:
        raise ValueError(f"path를 찾지 못했다: {start} -> {goal}")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came[cells[-1]])
    cells.reverse()
    return [_point(cell) for cell in cells]


def _route(
    waypoints: Sequence[tuple[float, float]],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    clearance: float,
) -> list[tuple[float, float]]:
    route: list[tuple[float, float]] = []
    for start, goal in zip(waypoints, waypoints[1:]):
        segment = _path(start, goal, obstacles, polygons, clearance=clearance)
        if route:
            route.extend(segment[1:])
        else:
            route.extend(segment)
    return route


def _heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _unit_rows(
    unit_id: int,
    route: Sequence[tuple[float, float]],
    *,
    start_time: float,
    speed: float,
) -> list[dict[str, Any]]:
    rows = []
    elapsed = start_time
    hp = 100
    ammo = 12 if unit_id < 200 else 9
    for index, point in enumerate(route):
        if index < len(route) - 1:
            heading = _heading(point, route[index + 1])
            mode = "MOVE"
        else:
            heading = _heading(route[index - 1], point) if index else 0.0
            mode = "IDLE"
        if index:
            previous = route[index - 1]
            elapsed += math.hypot(point[0] - previous[0], point[1] - previous[1]) / speed
        rows.append(
            {
                "time": round(elapsed, 2),
                "id": unit_id,
                "x": round(point[0], 3),
                "y": round(point[1], 3),
                "heading": round(heading, 2),
                "hp": hp,
                "ammo": ammo,
                "mode": mode,
                "target_id": "",
            }
        )
    return rows


def _state_at(rows: Sequence[dict[str, Any]], time_sec: float) -> dict[str, Any]:
    if time_sec <= rows[0]["time"]:
        return rows[0]
    if time_sec >= rows[-1]["time"]:
        return rows[-1]
    for previous, current in zip(rows, rows[1:]):
        if previous["time"] <= time_sec <= current["time"]:
            span = current["time"] - previous["time"] or 1.0
            ratio = (time_sec - previous["time"]) / span
            return {
                **previous,
                "time": time_sec,
                "x": previous["x"] + (current["x"] - previous["x"]) * ratio,
                "y": previous["y"] + (current["y"] - previous["y"]) * ratio,
            }
    return rows[-1]


def _next_route_point(rows: Sequence[dict[str, Any]], time_sec: float) -> tuple[float, float]:
    for row in rows:
        if row["time"] > time_sec + 0.05:
            return (float(row["x"]), float(row["y"]))
    last = rows[-1]
    return (float(last["x"]), float(last["y"]))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_rows(
    rows_by_unit: Mapping[int, Sequence[dict[str, Any]]],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    clearance: float,
) -> None:
    for unit_id, rows in rows_by_unit.items():
        for row in rows:
            point = (float(row["x"]), float(row["y"]))
            if not _free(point, obstacles, polygons, clearance=clearance):
                raise ValueError(f"unit {unit_id}가 obstacle 안에 있다: {point}")
        for previous, current in zip(rows, rows[1:]):
            start = (float(previous["x"]), float(previous["y"]))
            end = (float(current["x"]), float(current["y"]))
            if _segment_hits_building(start, end, polygons):
                raise ValueError(f"unit {unit_id} 경로가 building polygon을 지난다: {start} -> {end}")


def generate(run_dir: Path, *, clearance: float, speed: float) -> None:
    config = _load_config(run_dir)
    obstacles = _obstacles(config)
    polygons = _building_polygons(config)
    objective = _nearest_free(tuple(config.get("objective", [11.0, 1.5])), obstacles, polygons, clearance=clearance)
    config["objective"] = [round(objective[0], 3), round(objective[1], 3)]

    plans: list[UnitPlan] = [
        (101, "ASSAULT", ((-18.0, -8.0), (-14.0, -8.0), (-10.0, -6.0), (-6.0, -6.0), (-2.0, -6.0), (2.0, -5.5), (6.0, -4.5))),
        (102, "SUPPORT", ((-18.0, -3.0), (-15.0, -3.0), (-11.5, -2.0), (-8.0, -2.0), (-4.0, -1.5), (1.0, -1.5), (6.5, -1.0))),
        (103, "FLANKER", ((-18.0, 6.5), (-14.0, 6.0), (-10.0, 5.5), (-6.0, 5.5), (-2.0, 6.0), (3.0, 5.5), (7.5, 4.5))),
        (201, "BLOCKER", ((12.0, -4.0), (8.5, -4.5), (6.0, -4.0))),
        (202, "CENTER", ((18.0, -1.5), (14.0, -1.5), (10.0, -1.5), (7.0, -1.0))),
        (203, "OVERWATCH", ((18.0, 6.5), (14.0, 6.0), (10.0, 4.0), (7.5, 4.5))),
    ]

    rows_by_unit: dict[int, list[dict[str, Any]]] = {}
    roles: dict[int, str] = {}
    for unit_id, role, waypoints in plans:
        route = _route(waypoints, obstacles, polygons, clearance=clearance)
        rows_by_unit[unit_id] = _unit_rows(unit_id, route, start_time=0.0, speed=speed)
        roles[unit_id] = role

    final_time = max(rows[-1]["time"] for rows in rows_by_unit.values())
    for unit_id, rows in rows_by_unit.items():
        last = rows[-1]
        if last["time"] < final_time:
            rows.append({**last, "time": round(final_time, 2), "mode": "IDLE"})

    _validate_rows(rows_by_unit, obstacles, polygons, clearance=clearance)

    soldier_rows = sorted(
        (row for rows in rows_by_unit.values() for row in rows),
        key=lambda row: (row["time"], row["id"]),
    )
    _write_csv(
        run_dir / "soldier_log.csv",
        ("time", "id", "x", "y", "heading", "hp", "ammo", "mode", "target_id"),
        soldier_rows,
    )

    planner_rows = []
    tick = 0
    while tick <= math.ceil(final_time):
        planner_rows.append(
            {
                "time": tick,
                "selector": "obstacle_aware_demo",
                "best_score": round(12.0 + tick * 1.2, 2),
                "population_mean": round(8.0 + tick * 0.7, 2),
                "commands": "walkable route",
            }
        )
        tick += 5
    _write_csv(
        run_dir / "planner_log.csv",
        ("time", "selector", "best_score", "population_mean", "commands"),
        planner_rows,
    )

    command_rows = []
    for tick in range(0, math.ceil(final_time) + 1, 5):
        for unit_id in (101, 102, 103):
            rows = rows_by_unit[unit_id]
            state = _state_at(rows, tick)
            next_point = _next_route_point(rows, tick)
            action = "HOLD" if next_point == (float(rows[-1]["x"]), float(rows[-1]["y"])) else "MOVE"
            command_rows.append(
                {
                    "time": tick,
                    "unit_id": unit_id,
                    "role": roles[unit_id],
                    "action": action,
                    "detail": f"({next_point[0]:.1f},{next_point[1]:.1f})",
                    "reason": "avoid real building obstacles",
                }
            )
    _write_csv(
        run_dir / "commands_log.csv",
        ("time", "unit_id", "role", "action", "detail", "reason"),
        command_rows,
    )
    _write_config(run_dir, config)
    print(f"saved obstacle-aware demo logs in {run_dir} ({len(soldier_rows)} soldier rows, t_max={final_time:.1f}s)")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate obstacle-aware NAVER demo logs.")
    parser.add_argument("--run-dir", type=Path, default=Path("hackerthon/output/naver_dynamic_demo_run"))
    parser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE_UNITS)
    parser.add_argument("--speed", type=float, default=1.2)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    generate(args.run_dir, clearance=args.clearance, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
