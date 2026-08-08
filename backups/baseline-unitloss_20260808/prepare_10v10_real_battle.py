"""Prepare a real-building 10v10 battle config and initial deployment image."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from hackerthon.real_building_obstacles import _point_in_polygon
from hackerthon.rendering_common import draw_terrain, terrain_xy_values, unit_radius_units
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN, point_blocked


GRID_RES = 0.5
DEFAULT_CLEARANCE_UNITS = 0.12
DEFAULT_MIN_SPACING_UNITS = 0.75
DEFAULT_OBJECTIVE = (15.0, 8.0)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _inside_building(point: tuple[float, float], polygons: Sequence[tuple[tuple[float, float], ...]]) -> bool:
    return any(_point_in_polygon(point[0], point[1], polygon) for polygon in polygons)


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
    return not _inside_building(point, polygons)


def _snap(value: float) -> float:
    return round(round(float(value) / GRID_RES) * GRID_RES, 3)


def _nearest_free_spaced(
    desired: tuple[float, float],
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    *,
    occupied: Sequence[tuple[float, float]],
    clearance: float,
    min_spacing: float,
) -> tuple[float, float]:
    candidates: list[tuple[float, float, float]] = []
    max_radius_cells = 80
    for ix in range(-max_radius_cells, max_radius_cells + 1):
        for iy in range(-max_radius_cells, max_radius_cells + 1):
            x = _snap(desired[0] + ix * GRID_RES)
            y = _snap(desired[1] + iy * GRID_RES)
            distance = math.hypot(x - desired[0], y - desired[1])
            candidates.append((distance, x, y))

    for _, x, y in sorted(candidates):
        point = (x, y)
        if not _free(point, obstacles, polygons, clearance=clearance):
            continue
        if any(math.hypot(x - ox, y - oy) < min_spacing for ox, oy in occupied):
            continue
        return point
    raise ValueError(f"free deployment point를 찾지 못했다: {desired}")


def _formation_desires(team: str, team_size: int) -> list[tuple[float, float]]:
    if team_size <= 0:
        raise ValueError("team size는 0보다 커야 한다")
    columns = max(1, min(3, math.ceil(team_size / 5)))
    rows = math.ceil(team_size / columns)
    y_min = WORLD_Y_MIN + 2.0
    y_max = WORLD_Y_MAX - 2.0
    x_start = WORLD_X_MIN + 1.5 if team == "blue" else WORLD_X_MAX - 1.5
    x_step = 1.8 if team == "blue" else -1.8
    desires = []
    for index in range(team_size):
        column = index // rows
        row = index % rows
        y = 0.5 * (y_min + y_max) if rows == 1 else y_min + (y_max - y_min) * row / (rows - 1)
        x = x_start + column * x_step
        desires.append((x, y))
    return desires


def _deploy_team(
    *,
    team: str,
    team_size: int,
    obstacles: Sequence[tuple[float, float, float, float]],
    polygons: Sequence[tuple[tuple[float, float], ...]],
    occupied: list[tuple[float, float]],
    clearance: float,
    min_spacing: float,
) -> list[dict[str, float | int]]:
    base_id = 101 if team == "blue" else 201
    heading = 0.0 if team == "blue" else 180.0
    deployed = []
    for index, desired in enumerate(_formation_desires(team, team_size)):
        x, y = _nearest_free_spaced(
            desired,
            obstacles,
            polygons,
            occupied=occupied,
            clearance=clearance,
            min_spacing=min_spacing,
        )
        occupied.append((x, y))
        deployed.append({"id": base_id + index, "x": x, "y": y, "heading": heading})
    return deployed


def _prepare_config(
    *,
    config: dict[str, Any],
    team_size: int,
    objective: tuple[float, float],
    world_model_checkpoint: str,
    clearance: float,
    min_spacing: float,
) -> dict[str, Any]:
    obstacles = _obstacles(config)
    polygons = _building_polygons(config)
    occupied: list[tuple[float, float]] = []
    blue = _deploy_team(
        team="blue",
        team_size=team_size,
        obstacles=obstacles,
        polygons=polygons,
        occupied=occupied,
        clearance=clearance,
        min_spacing=min_spacing,
    )
    red = _deploy_team(
        team="red",
        team_size=team_size,
        obstacles=obstacles,
        polygons=polygons,
        occupied=occupied,
        clearance=clearance,
        min_spacing=min_spacing,
    )
    objective_point = _nearest_free_spaced(
        objective,
        obstacles,
        polygons,
        occupied=occupied,
        clearance=clearance,
        min_spacing=0.0,
    )

    prepared = dict(config)
    prepared["objective"] = [round(objective_point[0], 3), round(objective_point[1], 3)]
    prepared["team_size"] = int(team_size)
    prepared["blue_ids"] = [int(row["id"]) for row in blue]
    prepared["red_ids"] = [int(row["id"]) for row in red]
    prepared["initial_positions"] = {"blue": blue, "red": red}
    prepared["mission"] = {
        "name": "destroy_red_then_reach_objective",
        "objective": prepared["objective"],
        "kept_from_existing_world_model_flow": True,
    }
    prepared["world_model_checkpoint"] = world_model_checkpoint
    prepared["deployment"] = {
        "clearance_units": float(clearance),
        "min_spacing_units": float(min_spacing),
        "placement": "nearest free grid cell outside building polygons and AABB strips",
    }
    return prepared


def _render_initial(config: Mapping[str, Any], out_path: Path) -> None:
    positions = config["initial_positions"]
    blue = positions["blue"]
    red = positions["red"]
    objective = tuple(float(value) for value in config["objective"])
    unit_radius = unit_radius_units(config)

    terrain_x, terrain_y = terrain_xy_values(config)
    unit_x = [float(row["x"]) for row in blue + red]
    unit_y = [float(row["y"]) for row in blue + red]
    xs = terrain_x + unit_x + [objective[0], WORLD_X_MIN, WORLD_X_MAX]
    ys = terrain_y + unit_y + [objective[1], WORLD_Y_MIN, WORLD_Y_MAX]

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_title("Gangnam Real Buildings | 10v10 Initial Deployment")
    ax.set_xlabel("simulation x")
    ax.set_ylabel("simulation y")
    draw_terrain(ax, config, zorder=3)
    ax.plot([objective[0]], [objective[1]], marker="*", ms=15, color="#6a1b9a", zorder=8)
    ax.annotate("OBJ", objective, xytext=(7, 7), textcoords="offset points", color="#6a1b9a", fontsize=9, weight="bold")

    for row in blue + red:
        unit_id = int(row["id"])
        x = float(row["x"])
        y = float(row["y"])
        heading = float(row["heading"])
        color = "#1d4ed8" if unit_id < 200 else "#c62828"
        marker = "o" if unit_id < 200 else "s"
        ax.add_patch(Wedge((x, y), 2.2, heading - 35.0, heading + 35.0, fc=color, ec="none", alpha=0.045, zorder=1))
        ax.add_patch(Circle((x, y), unit_radius, fc=color, ec="#111827", lw=0.5, zorder=6))
        ax.plot([x], [y], marker=marker, ms=3.2, color=color, zorder=7)
        ax.annotate(
            ("B" if unit_id < 200 else "R") + str(unit_id),
            (x, y),
            xytext=(4, -8),
            textcoords="offset points",
            fontsize=7,
            color=color,
            weight="bold",
            zorder=8,
        )

    fig.text(
        0.5,
        0.02,
        "gray polygons: real buildings | AABB strips are stored for world-model terrain slots | units are placed outside obstacles",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach 10v10 initial positions to a real-building config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, default=None, help="생략하면 --config를 덮어쓴다.")
    parser.add_argument("--out-image", type=Path, default=None)
    parser.add_argument("--team-size", type=int, default=10)
    parser.add_argument("--objective-x", type=float, default=DEFAULT_OBJECTIVE[0])
    parser.add_argument("--objective-y", type=float, default=DEFAULT_OBJECTIVE[1])
    parser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE_UNITS)
    parser.add_argument("--min-spacing", type=float, default=DEFAULT_MIN_SPACING_UNITS)
    parser.add_argument("--world-model-checkpoint", default="checkpoints/devs_mixed_probe.pt")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.team_size <= 0:
        raise ValueError("--team-size는 0보다 커야 한다")
    config = _load_config(args.config)
    out_config = args.out_config or args.config
    out_image = args.out_image or out_config.with_name("initial_10v10.png")
    prepared = _prepare_config(
        config=config,
        team_size=args.team_size,
        objective=(args.objective_x, args.objective_y),
        world_model_checkpoint=args.world_model_checkpoint,
        clearance=args.clearance,
        min_spacing=args.min_spacing,
    )
    _write_config(out_config, prepared)
    _render_initial(prepared, out_image)
    print(
        "prepared "
        f"{args.team_size}v{args.team_size} config={out_config} "
        f"objective={prepared['objective']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
