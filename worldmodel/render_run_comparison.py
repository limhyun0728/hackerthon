"""여러 DEVS run을 같은 시간축 side-by-side 영상으로 렌더링한다.

VLM/Ollama 전술 run과 CEM/JEPA run은 planner_log 컬럼이 다르지만,
soldier_log/commands_log/config 계약은 거의 같다. 이 도구는 유닛 궤적,
사격선, 장애물, objective를 공통 형식으로 그려 전술 움직임을 비교한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Wedge

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.rendering_common import draw_terrain, terrain_xy_values, unit_radius_units
from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN


SUBFRAMES = 5
FPS = 10
BLUE_COLOR = "#1f77b4"
RED_COLOR = "#c62828"
FIRE_COLOR = "#ff8f00"


@dataclass(frozen=True)
class RunSpec:
    """비교 영상 한 칸에 들어갈 run 지정."""

    label: str
    run_dir: Path


@dataclass
class RunPanel:
    """렌더링에 필요한 run 데이터."""

    label: str
    run_dir: Path
    config: dict
    rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]]
    commands: dict[tuple[float, int], dict[str, str]]
    planner_rows: list[tuple[float, str]]
    summary: str


def _parse_run_spec(value: str) -> RunSpec:
    """LABEL=RUN_DIR 인자를 읽는다."""
    if "=" not in value:
        raise ValueError("--run은 LABEL=RUN_DIR 형식이어야 한다")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("run label은 비어 있을 수 없다")
    run_dir = Path(raw_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir가 없다: {run_dir}")
    return RunSpec(label=label, run_dir=run_dir)


def _load_unit_rows(run_dir: Path) -> dict[int, dict[float, tuple[float, float, float, int, str, int | None]]]:
    """soldier_log.csv를 unit/time 색인으로 읽는다."""
    rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]] = defaultdict(dict)
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target_id = int(row["target_id"]) if row["target_id"] not in ("", "None") else None
            rows[int(row["id"])][float(row["time"])] = (
                float(row["x"]),
                float(row["y"]),
                float(row["heading"]),
                int(float(row["hp"])),
                str(row["mode"]),
                target_id,
            )
    if not rows:
        raise ValueError(f"{run_dir}/soldier_log.csv에 unit row가 없다")
    return rows


def _load_commands(run_dir: Path) -> dict[tuple[float, int], dict[str, str]]:
    """commands_log.csv를 tick/unit 색인으로 읽는다."""
    path = run_dir / "commands_log.csv"
    commands: dict[tuple[float, int], dict[str, str]] = {}
    if not path.exists():
        return commands
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            commands[(float(row["time"]), int(row["unit_id"]))] = row
    return commands


def _load_planner_rows(run_dir: Path) -> list[tuple[float, str]]:
    """planner_log.csv에서 화면 상단에 표시할 planner 상태를 읽는다."""
    path = run_dir / "planner_log.csv"
    if not path.exists():
        return []
    rows: list[tuple[float, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_sec = float(row["time"])
            if "tactic" in row:
                text = f"{row.get('tactic', '')} {row.get('decision', '')}".strip()
            elif "selector" in row:
                text = f"{row.get('selector', '')} best={float(row.get('best_score', 0.0)):.1f}"
            else:
                text = ""
            rows.append((time_sec, text))
    return rows


def _state_at(
    rows: dict[int, dict[float, tuple[float, float, float, int, str, int | None]]],
    unit_id: int,
    time_sec: float,
) -> tuple[float, float, float, int, str, int | None]:
    """두 tick 사이 상태를 선형 보간해서 반환한다."""
    times = sorted(rows[unit_id])
    clamped = min(max(time_sec, times[0]), times[-1])
    lower = max(value for value in times if value <= clamped)
    upper = min(value for value in times if value >= clamped)
    x0, y0, heading0, hp0, mode, target_id = rows[unit_id][lower]
    if upper == lower:
        return x0, y0, heading0, hp0, mode, target_id
    x1, y1, heading1, *_ = rows[unit_id][upper]
    alpha = (clamped - lower) / (upper - lower)
    heading_delta = (heading1 - heading0 + 180.0) % 360.0 - 180.0
    heading = (heading0 + alpha * heading_delta + 180.0) % 360.0 - 180.0
    return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0), heading, hp0, mode, target_id


def _planner_text(panel: RunPanel, time_sec: float) -> str:
    """현재 시점 이전의 최신 planner 상태 문자열."""
    current = ""
    for row_time, text in panel.planner_rows:
        if row_time <= time_sec:
            current = text
    return current


def _latest_rows(panel: RunPanel) -> list[tuple[int, tuple[float, float, float, int, str, int | None]]]:
    """각 unit의 마지막 상태를 반환한다."""
    result = []
    for unit_id, unit_rows in panel.rows.items():
        last_time = max(unit_rows)
        result.append((unit_id, unit_rows[last_time]))
    return sorted(result)


def _panel_summary(panel: RunPanel) -> str:
    """최종 상태 요약 문자열을 만든다."""
    latest = _latest_rows(panel)
    blue = [(unit_id, state) for unit_id, state in latest if unit_id < 200]
    red = [(unit_id, state) for unit_id, state in latest if unit_id >= 200]
    blue_alive = sum(state[3] > 0 for _, state in blue)
    red_alive = sum(state[3] > 0 for _, state in red)
    blue_hp = sum(state[3] for _, state in blue)
    red_hp = sum(state[3] for _, state in red)
    final_time = max(time for unit_rows in panel.rows.values() for time in unit_rows)
    if red_alive == 0:
        outcome = "COMBAT_WIN"
    elif blue_alive == 0:
        outcome = "LOSE"
    else:
        outcome = "TIMEOUT"
    return f"{outcome} t={final_time:.0f} B{blue_alive}/HP{blue_hp} R{red_alive}/HP{red_hp}"


def _load_panel(spec: RunSpec) -> RunPanel:
    """run spec에서 panel 데이터를 구성한다."""
    config = json.loads((spec.run_dir / "config.json").read_text(encoding="utf-8"))
    panel = RunPanel(
        label=spec.label,
        run_dir=spec.run_dir,
        config=config,
        rows=_load_unit_rows(spec.run_dir),
        commands=_load_commands(spec.run_dir),
        planner_rows=_load_planner_rows(spec.run_dir),
        summary="",
    )
    panel.summary = _panel_summary(panel)
    return panel


def _axis_limits(panels: Sequence[RunPanel]) -> tuple[float, float, float, float]:
    """모든 panel을 담는 공통 축 범위를 계산한다."""
    xs = [WORLD_X_MIN, WORLD_X_MAX, 10.0]
    ys = [WORLD_Y_MIN, WORLD_Y_MAX, 0.0]
    for panel in panels:
        for unit_rows in panel.rows.values():
            for x, y, *_ in unit_rows.values():
                xs.append(x)
                ys.append(y)
        terrain_x, terrain_y = terrain_xy_values(panel.config)
        xs.extend(terrain_x)
        ys.extend(terrain_y)
        objective = panel.config.get("objective", [10.0, 0.0])
        xs.append(float(objective[0]))
        ys.append(float(objective[1]))
    return min(xs) - 0.7, max(xs) + 0.7, min(ys) - 0.7, max(ys) + 0.7


def render_comparison(
    *,
    specs: Sequence[RunSpec],
    out_path: Path,
    fps: int,
    subframes: int,
) -> None:
    """여러 run을 나란히 비교 영상으로 저장한다."""
    if len(specs) < 2:
        raise ValueError("비교하려면 run이 최소 2개 필요하다")
    if fps <= 0 or subframes <= 0:
        raise ValueError("fps/subframes는 0보다 커야 한다")
    panels = [_load_panel(spec) for spec in specs]
    x_min, x_max, y_min, y_max = _axis_limits(panels)
    max_time = int(max(time for panel in panels for unit_rows in panel.rows.values() for time in unit_rows))
    frame_count = (max_time + 1) * subframes

    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), dpi=120, squeeze=False)
    fig.suptitle("Ollama/VLM tactics vs CEM-JEPA behavior | synchronized simulation time", fontsize=12)
    panel_states = []
    for index, panel in enumerate(panels):
        ax = axes[0][index]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18, linewidth=0.4)
        ax.set_title(f"{panel.label}\n{panel.summary}")
        draw_terrain(ax, panel.config)
        objective = panel.config.get("objective", [10.0, 0.0])
        ax.plot([float(objective[0])], [float(objective[1])], marker="*", ms=11, color="#6a1b9a", zorder=7)

        unit_artists = {}
        trails = defaultdict(list)
        blue_ids = sorted(unit_id for unit_id in panel.rows if unit_id < 200)
        red_ids = sorted(unit_id for unit_id in panel.rows if unit_id >= 200)
        unit_radius = unit_radius_units(panel.config)
        for unit_id in blue_ids + red_ids:
            color = BLUE_COLOR if unit_id < 200 else RED_COLOR
            marker = "o" if unit_id < 200 else "s"
            fov = Wedge((0, 0), 10.0, -60.0, 60.0, fc=color, ec="none", alpha=0.025, zorder=0)
            ax.add_patch(fov)
            body = Circle((0, 0), unit_radius, fc=color, ec="#111827", alpha=0.9, lw=0.5, zorder=5)
            ax.add_patch(body)
            ring = None
            if unit_id >= 200:
                ring = Circle((0, 0), 7.0, fc="none", ec=RED_COLOR, alpha=0.16, lw=0.5, ls=":", zorder=1)
                ax.add_patch(ring)
            (dot,) = ax.plot([], [], marker, ms=2.0, color=color, zorder=6)
            (trail,) = ax.plot([], [], "-", lw=1.0, color=color, alpha=0.45, zorder=3)
            (fire,) = ax.plot([], [], "--", lw=0.8, color=FIRE_COLOR, alpha=0.9, zorder=4)
            label = ax.annotate(
                "",
                (0, 0),
                xytext=(4, -8),
                textcoords="offset points",
                fontsize=6,
                color="#0d3b66" if unit_id < 200 else RED_COLOR,
                zorder=7,
            )
            unit_artists[unit_id] = (fov, ring, body, dot, trail, fire, label)
        timer = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=7)
        planner = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5)
        panel_states.append((panel, blue_ids, red_ids, unit_artists, trails, timer, planner))

    def draw(frame_index: int):
        time_sec = frame_index / subframes
        drawn = []
        for panel, blue_ids, red_ids, unit_artists, trails, timer, planner in panel_states:
            timer.set_text(f"t={time_sec:.1f}s")
            planner.set_text(_planner_text(panel, time_sec)[:56])
            drawn.extend([timer, planner])
            for unit_id in blue_ids + red_ids:
                x, y, heading, hp, mode, target_id = _state_at(panel.rows, unit_id, time_sec)
                fov, ring, body, dot, trail, fire, label = unit_artists[unit_id]
                fov.set_center((x, y))
                fov.set_theta1(heading - 60.0)
                fov.set_theta2(heading + 60.0)
                body.set_center((x, y))
                if ring is not None:
                    ring.set_center((x, y))
                trails[unit_id].append((x, y))
                trail.set_data(*zip(*trails[unit_id]))
                dot.set_data([x], [y])
                dead = hp <= 0
                dot.set_marker("x" if dead else ("o" if unit_id < 200 else "s"))
                dot.set_color("0.45" if dead else (BLUE_COLOR if unit_id < 200 else RED_COLOR))
                fov.set_visible(not dead)
                body.set_visible(not dead)
                if ring is not None:
                    ring.set_visible(not dead)
                label.set_color("0.45" if dead else ("#0d3b66" if unit_id < 200 else RED_COLOR))
                label.set_text(("B" if unit_id < 200 else "R") + str(unit_id) + (f" hp{hp}" if 0 < hp < 100 else ""))
                label.xy = (x, y)
                if not dead and mode == "ENGAGE" and target_id in panel.rows:
                    tx, ty, *_ = _state_at(panel.rows, target_id, time_sec)
                    fire.set_data([x, tx], [y, ty])
                else:
                    fire.set_data([], [])
                drawn.extend([fov, body, dot, trail, fire, label])
                if ring is not None:
                    drawn.append(ring)
        return drawn

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, draw, frames=frame_count, blit=False)
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=3200))
    plt.close(fig)
    print(f"saved {out_path} ({frame_count} frames, {frame_count / fps:.1f}s)")


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="여러 DEVS run side-by-side 비교 영상 렌더링")
    parser.add_argument("--run", action="append", type=_parse_run_spec, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--subframes", type=int, default=SUBFRAMES)
    args = parser.parse_args(list(argv) if argv is not None else None)
    render_comparison(specs=args.run, out_path=args.out, fps=args.fps, subframes=args.subframes)


if __name__ == "__main__":
    main()
