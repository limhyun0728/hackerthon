"""CEM episode milestone들을 같은 시간축으로 비교 렌더링한다.

한 시나리오에서 학습이 진행될수록 BLUE 이동 경로가 어떻게 바뀌는지 보기 위해
여러 episode를 격자로 배치하고 동일한 simulation time을 동시에 재생한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.rendering_common import draw_terrain, terrain_xy_values, unit_radius_units
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from hackerthon.worldmodel.episodic_cem_training_loop import determine_outcome
from hackerthon.worldmodel.evaluator import evaluate_v2_run_segment
from hackerthon.worldmodel.render_cem_episode import (
    BLUE_COLOR,
    FIRE_COLOR,
    MOVE_COLOR,
    RED_COLOR,
    SUBFRAMES,
    _command_at,
    _draw_move_arrow,
    _load_command_rows,
    _load_unit_rows,
    _state_at,
)


@dataclass(frozen=True)
class EpisodePanel:
    """격자 한 칸에 들어갈 episode 데이터."""

    number: int
    run_dir: Path
    config: dict
    unit_rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]]
    commands: dict[tuple[float, int], dict[str, str]]
    objective: tuple[float, float]
    score: float
    outcome: str
    objective_reached: bool
    final_objective_distance: float
    command_counts: Counter[str]


def _episode_number(path: Path) -> int:
    """폴더명에서 episode 번호를 읽는다."""
    match = re.search(r"episode_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"episode 폴더명이 아니다: {path.name}")
    return int(match.group(1))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV 로그를 dict row 목록으로 읽는다."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_rows(soldier_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """각 unit의 마지막 state row를 반환한다."""
    latest: dict[int, tuple[float, dict[str, str]]] = {}
    for row in soldier_rows:
        unit_id = int(row["id"])
        time_sec = float(row["time"])
        current = latest.get(unit_id)
        if current is None or time_sec >= current[0]:
            latest[unit_id] = (time_sec, row)
    if not latest:
        raise ValueError("soldier_log.csv에 unit row가 없다")
    return [row for _, row in latest.values()]


def _objective_status(rows: Sequence[dict[str, str]], objective: tuple[float, float]) -> tuple[bool, float]:
    """생존 BLUE가 objective에 도달했는지와 최종 최소 거리를 계산한다."""
    reached = False
    min_distance = math.inf
    for row in rows:
        unit_id = int(row["id"])
        if unit_id >= 200:
            continue
        distance = math.hypot(float(row["x"]) - objective[0], float(row["y"]) - objective[1])
        min_distance = min(min_distance, distance)
        if float(row["hp"]) > 0.0 and distance <= 1.0:
            reached = True
    if math.isinf(min_distance):
        raise ValueError("BLUE unit row가 없다")
    return reached, min_distance


def _episode_dirs(root_dir: Path) -> dict[int, Path]:
    """root 아래 episode 폴더를 번호로 색인한다."""
    dirs = {_episode_number(path): path for path in root_dir.glob("episode_*")}
    if not dirs:
        raise ValueError(f"episode 폴더가 없다: {root_dir}")
    return dirs


def _load_panel(root_dir: Path, episode_number: int) -> EpisodePanel:
    """한 episode의 렌더링과 요약 정보를 준비한다."""
    directories = _episode_dirs(root_dir)
    if episode_number not in directories:
        raise ValueError(f"요청한 episode가 없다: {episode_number}")
    run_dir = directories[episode_number]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    soldier_rows = _read_csv_rows(run_dir / "soldier_log.csv")
    command_rows = _read_csv_rows(run_dir / "commands_log.csv")
    last_rows = _latest_rows(soldier_rows)
    objective = tuple(float(value) for value in config["objective"])
    reached, final_distance = _objective_status(last_rows, objective)
    max_time = max(float(row["time"]) for row in soldier_rows)
    result = evaluate_v2_run_segment(run_dir, start_time=0.0, end_time=max_time)
    return EpisodePanel(
        number=episode_number,
        run_dir=run_dir,
        config=config,
        unit_rows=_load_unit_rows(run_dir),
        commands=_load_command_rows(run_dir),
        objective=objective,
        score=result.score,
        outcome=determine_outcome(last_rows),
        objective_reached=reached,
        final_objective_distance=final_distance,
        command_counts=Counter(row["action"] for row in command_rows),
    )


def _axis_limits(panels: Sequence[EpisodePanel]) -> tuple[float, float, float, float]:
    """선택된 episode들을 모두 담는 공통 축 범위를 계산한다."""
    xs = [WORLD_X_MIN, WORLD_X_MAX]
    ys = [WORLD_Y_MIN, WORLD_Y_MAX]
    for panel in panels:
        xs.append(panel.objective[0])
        ys.append(panel.objective[1])
        terrain_x, terrain_y = terrain_xy_values(panel.config)
        xs.extend(terrain_x)
        ys.extend(terrain_y)
        for unit_data in panel.unit_rows.values():
            for x, y, *_ in unit_data.values():
                xs.append(x)
                ys.append(y)
    return min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5


def _panel_title(panel: EpisodePanel) -> str:
    """각 subplot 제목에 들어갈 요약 문자열을 만든다."""
    reached = "OBJ" if panel.objective_reached else "NO-OBJ"
    return (
        f"EP {panel.number:04d} | {panel.outcome} | {reached}\n"
        f"score {panel.score:.1f} dist {panel.final_objective_distance:.1f} "
        f"M{panel.command_counts['MOVE']} E{panel.command_counts['ENGAGE']}"
    )


def render_progression(
    root_dir: Path,
    episode_numbers: Sequence[int],
    out_path: Path,
    columns: int,
    subframes: int,
    fps: int,
) -> None:
    """여러 episode를 같은 simulation time으로 나란히 MP4 렌더링한다."""
    if not episode_numbers:
        raise ValueError("episode 번호를 최소 하나 이상 지정해야 한다")
    if columns <= 0:
        raise ValueError("columns는 1 이상이어야 한다")
    if subframes <= 0:
        raise ValueError("subframes는 1 이상이어야 한다")
    if fps <= 0:
        raise ValueError("fps는 1 이상이어야 한다")

    panels = [_load_panel(root_dir, episode_number) for episode_number in episode_numbers]
    x_min, x_max, y_min, y_max = _axis_limits(panels)
    max_time = int(
        max(time_value for panel in panels for unit_data in panel.unit_rows.values() for time_value in unit_data)
    )
    frame_count = (max_time + 1) * subframes
    rows = math.ceil(len(panels) / columns)

    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.3 * rows), dpi=120, squeeze=False)
    fig.suptitle("CEM-JEPA Training Progression | same scenario, synchronized simulation time", fontsize=12)

    panel_artists = []
    for index, panel in enumerate(panels):
        ax = axes[index // columns][index % columns]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18, linewidth=0.4)
        ax.set_title(_panel_title(panel))
        draw_terrain(ax, panel.config)
        ax.plot([panel.objective[0]], [panel.objective[1]], marker="*", ms=10, color="#6a1b9a", zorder=7)

        blue_ids = sorted(unit_id for unit_id in panel.unit_rows if unit_id < 200)
        red_ids = sorted(unit_id for unit_id in panel.unit_rows if unit_id >= 200)
        if not blue_ids or not red_ids:
            raise ValueError(f"BLUE/RED unit이 모두 있어야 한다: {panel.run_dir}")

        unit_artists = {}
        trails = {unit_id: [] for unit_id in blue_ids + red_ids}
        unit_radius = unit_radius_units(panel.config)
        for unit_id in blue_ids + red_ids:
            color = BLUE_COLOR if unit_id < 200 else RED_COLOR
            marker = "o" if unit_id < 200 else "s"
            body = Circle((0, 0), unit_radius, fc=color, ec="#111827", alpha=0.9, lw=0.5, zorder=5)
            ax.add_patch(body)
            (dot,) = ax.plot([], [], marker, ms=1.8, color=color, zorder=6)
            (trail,) = ax.plot([], [], "-", lw=1.0, color=color, alpha=0.45, zorder=3)
            (fire,) = ax.plot([], [], "--", lw=0.7, color=FIRE_COLOR, alpha=0.85, zorder=4)
            label = ax.annotate(
                "",
                (0, 0),
                xytext=(4, -7),
                textcoords="offset points",
                fontsize=5.5,
                color="#0d3b66" if unit_id < 200 else RED_COLOR,
                zorder=7,
            )
            unit_artists[unit_id] = (body, dot, trail, fire, label)
        timer = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=7)
        panel_artists.append((panel, ax, blue_ids, red_ids, unit_artists, trails, [], timer))

    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_axis_off()

    def draw(frame_index: int):
        time_sec = frame_index / subframes
        drawn = []
        for panel, ax, blue_ids, red_ids, unit_artists, trails, move_arrows, timer in panel_artists:
            timer.set_text(f"t={time_sec:.1f}s")
            drawn.append(timer)
            for arrow in move_arrows:
                arrow.remove()
            move_arrows.clear()

            for unit_id in blue_ids + red_ids:
                x, y, _, hp, mode, target_id = _state_at(panel.unit_rows, unit_id, time_sec)
                body, dot, trail, fire, label = unit_artists[unit_id]
                body.set_center((x, y))
                trails[unit_id].append((x, y))
                trail.set_data(*zip(*trails[unit_id]))
                dot.set_data([x], [y])
                if hp <= 0:
                    body.set_visible(False)
                    dot.set_marker("x")
                    dot.set_color("0.45")
                    label.set_color("0.45")
                else:
                    body.set_visible(True)
                label.set_text(("B" if unit_id < 200 else "R") + str(unit_id))
                label.xy = (x, y)

                if hp > 0 and mode == "ENGAGE" and target_id in panel.unit_rows:
                    tx, ty, *_ = _state_at(panel.unit_rows, target_id, time_sec)
                    fire.set_data([x, tx], [y, ty])
                else:
                    fire.set_data([], [])

                command = _command_at(panel.commands, unit_id, time_sec)
                if command is not None and unit_id < 200 and hp > 0:
                    arrow = _draw_move_arrow(ax, command, x, y)
                    if arrow is not None:
                        arrow.set_color(MOVE_COLOR)
                        arrow.set_alpha(0.35)
                        move_arrows.append(arrow)
                        drawn.append(arrow)
                drawn.extend([body, dot, trail, fire, label])
        return drawn

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, draw, frames=frame_count, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2800))
    plt.close(fig)
    print(f"saved {out_path} ({frame_count} frames, {frame_count / fps:.1f}s)")


def _parse_episode_numbers(value: str) -> list[int]:
    """쉼표로 구분된 episode 번호 문자열을 정수 목록으로 바꾼다."""
    numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not numbers:
        raise ValueError("episode 번호를 읽을 수 없다")
    return numbers


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="CEM episode progression renderer")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--episodes", type=_parse_episode_numbers, default=[1, 100, 200, 300, 400, 500])
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--subframes", type=int, default=SUBFRAMES)
    parser.add_argument("--fps", type=int, default=SUBFRAMES)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    render_progression(args.root_dir, args.episodes, args.out_path, args.columns, args.subframes, args.fps)


if __name__ == "__main__":
    main()
