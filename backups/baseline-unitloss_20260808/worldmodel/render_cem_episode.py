"""CEM episode 로그를 MP4 예시 영상으로 렌더링한다.

기존 v2 renderer는 tactic 로그를 기대한다. episodic CEM 결과는 selector,
best_score, population_mean, 직접 command 로그를 쓰므로 별도 renderer로
전장 이동, 사격선, CEM 점수 변화를 함께 보여준다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from hackerthon.rendering_common import draw_terrain, terrain_xy_values, unit_radius_units


SUBFRAMES = 5
FPS = 10
BLUE_COLOR = "#1f77b4"
RED_COLOR = "#c62828"
MOVE_COLOR = "#2e7d32"
FIRE_COLOR = "#ff8f00"


def _load_unit_rows(run_dir: Path) -> dict[int, dict[float, tuple[float, float, float, int, str, int | None]]]:
    """soldier_log.csv를 unit id와 time 기준으로 읽는다."""
    rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]] = defaultdict(dict)
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_sec = float(row["time"])
            unit_id = int(row["id"])
            target_id = int(row["target_id"]) if row["target_id"] not in ("", "None") else None
            rows[unit_id][time_sec] = (
                float(row["x"]),
                float(row["y"]),
                float(row["heading"]),
                int(float(row["hp"])),
                row["mode"],
                target_id,
            )
    if not rows:
        raise ValueError("soldier_log.csv에 unit row가 없다")
    return rows


def _load_planner_rows(run_dir: Path) -> list[tuple[float, str, float, float]]:
    """planner_log.csv에서 CEM selector와 점수를 읽는다."""
    rows: list[tuple[float, str, float, float]] = []
    with (run_dir / "planner_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    float(row["time"]),
                    row["selector"],
                    float(row["best_score"]),
                    float(row["population_mean"]),
                )
            )
    if not rows:
        raise ValueError("planner_log.csv에 CEM row가 없다")
    return rows


def _load_command_rows(run_dir: Path) -> dict[tuple[float, int], dict[str, str]]:
    """commands_log.csv를 tick/unit별 command로 읽는다."""
    commands: dict[tuple[float, int], dict[str, str]] = {}
    with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (float(row["time"]), int(row["unit_id"]))
            if key in commands:
                raise ValueError(f"같은 tick/unit command가 중복됐다: {key}")
            commands[key] = row
    if not commands:
        raise ValueError("commands_log.csv에 command row가 없다")
    return commands


def _state_at(
    rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]],
    unit_id: int,
    time_sec: float,
) -> tuple[float, float, float, int, str, int | None]:
    """두 정수 tick 사이 위치와 heading을 선형 보간한다."""
    times = sorted(rows[unit_id])
    clamped = min(max(time_sec, times[0]), times[-1])
    lower = max(time_value for time_value in times if time_value <= clamped)
    upper = min(time_value for time_value in times if time_value >= clamped)
    x0, y0, heading0, hp0, mode, target_id = rows[unit_id][lower]
    if upper == lower:
        return x0, y0, heading0, hp0, mode, target_id
    x1, y1, heading1, *_ = rows[unit_id][upper]
    alpha = (clamped - lower) / (upper - lower)
    heading_delta = (heading1 - heading0 + 180.0) % 360.0 - 180.0
    heading = (heading0 + alpha * heading_delta + 180.0) % 360.0 - 180.0
    return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0), heading, hp0, mode, target_id


def _planner_at(rows: list[tuple[float, str, float, float]], time_sec: float) -> tuple[str, float, float]:
    """현재 시간 이전의 최신 CEM score row를 반환한다."""
    current = rows[0]
    for row in rows:
        if row[0] <= time_sec:
            current = row
    _, selector, best_score, population_mean = current
    return selector, best_score, population_mean


def _command_at(commands: dict[tuple[float, int], dict[str, str]], unit_id: int, time_sec: float) -> dict[str, str] | None:
    """현재 frame이 속한 tick의 command를 찾는다."""
    tick = float(int(time_sec))
    return commands.get((tick, unit_id))


def _draw_move_arrow(ax, command: dict[str, str], x: float, y: float) -> FancyArrowPatch | None:
    """MOVE command의 목표 방향을 짧은 화살표로 그린다."""
    if command["action"] != "MOVE":
        return None
    detail = command["detail"].strip()
    if not (detail.startswith("(") and detail.endswith(")")):
        raise ValueError(f"MOVE detail 형식이 맞지 않는다: {detail}")
    raw_x, raw_y = detail[1:-1].split(",")
    target_x = float(raw_x)
    target_y = float(raw_y)
    arrow = FancyArrowPatch(
        (x, y),
        (target_x, target_y),
        arrowstyle="->",
        mutation_scale=10,
        lw=0.9,
        color=MOVE_COLOR,
        alpha=0.55,
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def render_episode(run_dir: Path, out_path: Path) -> None:
    """단일 CEM episode를 MP4로 렌더링한다."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    objective = tuple(config.get("objective", [10.0, 0.0]))
    unit_radius = unit_radius_units(config)
    unit_rows = _load_unit_rows(run_dir)
    planner_rows = _load_planner_rows(run_dir)
    commands = _load_command_rows(run_dir)
    blue_ids = sorted(unit_id for unit_id in unit_rows if unit_id < 200)
    red_ids = sorted(unit_id for unit_id in unit_rows if unit_id >= 200)
    if not blue_ids:
        raise ValueError("BLUE unit이 없다")
    if not red_ids:
        raise ValueError("RED unit이 없다")

    max_time = int(max(time_value for unit_data in unit_rows.values() for time_value in unit_data))
    all_x = [value[0] for unit_data in unit_rows.values() for value in unit_data.values()]
    all_y = [value[1] for unit_data in unit_rows.values() for value in unit_data.values()]
    terrain_x, terrain_y = terrain_xy_values(config)

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    ax.set_xlim(min([-10.0] + all_x + terrain_x) - 1.0, max([12.0] + all_x + terrain_x) + 1.0)
    ax.set_ylim(min([-10.0] + all_y + terrain_y) - 1.0, max([9.0] + all_y + terrain_y) + 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linewidth=0.5)
    title = ax.set_title("")
    fig.text(
        0.5,
        0.02,
        "episodic CEM-JEPA | gray: obstacle | blue/red: units | green arrows: CEM MOVE | orange dashed: ENGAGE",
        ha="center",
        fontsize=8,
        color="0.35",
    )

    draw_terrain(ax, config)
    ax.plot([objective[0]], [objective[1]], marker="*", ms=14, color="#6a1b9a", zorder=7)
    ax.annotate("OBJ", objective, xytext=(7, 7), textcoords="offset points", color="#6a1b9a", fontsize=8)

    artists = {}
    for unit_id in blue_ids + red_ids:
        color = BLUE_COLOR if unit_id < 200 else RED_COLOR
        marker = "o" if unit_id < 200 else "s"
        fov = Wedge((0, 0), 10.0, -60.0, 60.0, fc=color, ec="none", alpha=0.025, zorder=0)
        ax.add_patch(fov)
        body = Circle((0, 0), unit_radius, fc=color, ec="#111827", alpha=0.9, lw=0.5, zorder=5)
        ax.add_patch(body)
        engage_ring = None
        if unit_id >= 200:
            engage_ring = Circle((0, 0), 7.0, fc="none", ec=RED_COLOR, alpha=0.14, lw=0.5, ls=":", zorder=1)
            ax.add_patch(engage_ring)
        (dot,) = ax.plot([], [], marker, ms=2.2, color=color, zorder=6)
        (trail,) = ax.plot([], [], "-", lw=1.2, color=color, alpha=0.45, zorder=3)
        (fire,) = ax.plot([], [], "--", lw=0.9, color=FIRE_COLOR, alpha=0.9, zorder=4)
        label = ax.annotate(
            "",
            (0, 0),
            xytext=(5, -9),
            textcoords="offset points",
            fontsize=7,
            color="#0d3b66" if unit_id < 200 else RED_COLOR,
            zorder=7,
        )
        artists[unit_id] = (fov, engage_ring, body, dot, trail, fire, label)

    trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
    move_arrows: list[FancyArrowPatch] = []
    frame_count = (max_time + 1) * SUBFRAMES

    def draw(frame_index: int):
        time_sec = frame_index / SUBFRAMES
        selector, best_score, population_mean = _planner_at(planner_rows, time_sec)
        title.set_text(
            f"{run_dir.name} | t={time_sec:.1f} selector={selector} "
            f"best={best_score:.2f} population={population_mean:.2f}"
        )
        for arrow in move_arrows:
            arrow.remove()
        move_arrows.clear()

        out = []
        for unit_id in blue_ids + red_ids:
            x, y, heading, hp, mode, target_id = _state_at(unit_rows, unit_id, time_sec)
            fov, engage_ring, body, dot, trail, fire, label = artists[unit_id]
            fov.set_center((x, y))
            fov.set_theta1(heading - 60.0)
            fov.set_theta2(heading + 60.0)
            body.set_center((x, y))
            if engage_ring is not None:
                engage_ring.set_center((x, y))
            trails[unit_id].append((x, y))
            trail.set_data(*zip(*trails[unit_id]))
            dot.set_data([x], [y])
            is_dead = hp <= 0
            fov.set_visible(not is_dead)
            body.set_visible(not is_dead)
            if engage_ring is not None:
                engage_ring.set_visible(not is_dead)
            if is_dead:
                dot.set_marker("x")
                dot.set_color("0.45")
                label.set_color("0.45")
            label.set_text(("B" if unit_id < 200 else "R") + str(unit_id) + (f" hp{hp}" if hp < 100 else ""))
            label.xy = (x, y)

            if not is_dead and mode == "ENGAGE" and target_id in unit_rows:
                tx, ty, *_ = _state_at(unit_rows, target_id, time_sec)
                fire.set_data([x, tx], [y, ty])
            else:
                fire.set_data([], [])

            command = _command_at(commands, unit_id, time_sec)
            if command is not None and unit_id < 200 and not is_dead:
                arrow = _draw_move_arrow(ax, command, x, y)
                if arrow is not None:
                    move_arrows.append(arrow)
                    out.append(arrow)
            out += [fov, body, dot, trail, fire, label]
            if engage_ring is not None:
                out.append(engage_ring)
        return out

    anim = animation.FuncAnimation(fig, draw, frames=frame_count, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    print(f"saved {out_path} ({frame_count} frames, {frame_count / FPS:.1f}s)")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="CEM episode result renderer")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_path", type=Path)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    render_episode(args.run_dir, args.out_path)


if __name__ == "__main__":
    main()
