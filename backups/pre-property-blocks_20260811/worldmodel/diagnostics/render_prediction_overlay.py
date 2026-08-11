"""실제 RED과 월드모델이 예측한 RED을 겹쳐 그린 mp4를 만든다.

예측은 주황색이다. 불확실도 원은 그리지 않는다 — 여기서 보여주려는 것은 "얼마나
모르는가"가 아니라 "예측이 실제와 얼마나 맞는가"이므로, 두 점 사이의 거리 자체가
그 정보다.

사용법:
    python worldmodel/diagnostics/render_prediction_overlay.py <run_dir> [out.mp4]
"""

from __future__ import annotations

import collections
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

BLUE_MAX_ID = 200
METERS_PER_UNIT = 10.0
TRUE_BLUE = "#1f6feb"
TRUE_RED = "#da3633"
PREDICTED = "#fb8500"


def _read_states(run_dir: Path) -> dict[float, list[dict]]:
    by_time: dict[float, list[dict]] = collections.defaultdict(list)
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_time[float(row["time"])].append(row)
    return by_time


def _read_predictions(run_dir: Path) -> dict[float, list[dict]]:
    path = run_dir / "prediction_log.csv"
    if not path.exists():
        raise SystemExit(f"{path}가 없다. make_prediction_log.py를 먼저 돌려라")
    by_time: dict[float, list[dict]] = collections.defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_time[float(row["time"])].append(row)
    return by_time


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("사용법: render_prediction_overlay.py <run_dir> [out.mp4] [map_config.json]")
    run_dir = Path(argv[0])
    out_path = Path(argv[1]) if len(argv) > 1 else run_dir / "prediction_overlay.mp4"

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    # 학습 루프가 남기는 에피소드 config에는 building_polygons가 없다. 원본 맵에서 가져온다.
    map_config = Path(argv[2]) if len(argv) > 2 else None
    if map_config and map_config.exists():
        source = json.loads(map_config.read_text(encoding="utf-8"))
        config["building_polygons"] = source.get("building_polygons") or []
        config.setdefault("real_map", source.get("real_map", {}))
    states = _read_states(run_dir)
    predictions = _read_predictions(run_dir)
    times = sorted(states)

    figure, axes = plt.subplots(figsize=(11, 8), dpi=110)
    axes.set_aspect("equal")
    axes.set_facecolor("#0d1117")
    figure.patch.set_facecolor("#0d1117")

    # VWorld 건물 폴리곤. 에피소드 config에는 없으므로 --map-config로 원본 맵을 받는다.
    # AABB로 그리면 건물이 실제보다 커 보이고 골목이 사라져 기동이 안 읽힌다.
    polygons = config.get("building_polygons") or []
    if polygons:
        for polygon in polygons:
            points = polygon.get("points") if isinstance(polygon, dict) else polygon
            if not points:
                continue
            axes.fill([p[0] for p in points], [p[1] for p in points],
                      facecolor="#2d333b", edgecolor="#454c56", linewidth=0.7, zorder=1)
    else:
        for x0, y0, x1, y1 in config.get("obstacles", []):
            axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#30363d", zorder=1))

    objective = config.get("objective")
    if objective:
        axes.plot(objective[0], objective[1], marker="*", markersize=26,
                  color="#a371f7", zorder=6, linestyle="none")

    xs = [float(r["x"]) for rows in states.values() for r in rows]
    ys = [float(r["y"]) for rows in states.values() for r in rows]
    margin = 2.0
    axes.set_xlim(min(xs) - margin, max(xs) + margin)
    axes.set_ylim(min(ys) - margin, max(ys) + margin)
    axes.set_xticks([]); axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    blue_dots, = axes.plot([], [], "o", color=TRUE_BLUE, markersize=11, zorder=5, linestyle="none")
    red_dots, = axes.plot([], [], "o", color=TRUE_RED, markersize=11, zorder=5, linestyle="none")
    dead_dots, = axes.plot([], [], "x", color="#6e7681", markersize=9, zorder=4,
                           linestyle="none", markeredgewidth=2)
    predicted_dots, = axes.plot([], [], "o", color=PREDICTED, markersize=11, zorder=5,
                                linestyle="none", markerfacecolor="none", markeredgewidth=2.5)
    links = [axes.plot([], [], "-", color=PREDICTED, linewidth=1.2, alpha=0.55, zorder=3)[0]
             for _ in range(40)]
    title = axes.text(0.015, 0.975, "", transform=axes.transAxes, color="#e6edf3",
                      fontsize=13, va="top", family="monospace")
    legend = axes.text(0.015, 0.045, "", transform=axes.transAxes, fontsize=11, va="bottom",
                       family="monospace", color="#8b949e")

    def draw(index: int):
        time_sec = times[index]
        rows = states[time_sec]
        alive_blue = [(float(r["x"]), float(r["y"])) for r in rows
                      if int(r["id"]) < BLUE_MAX_ID and float(r["hp"]) > 0]
        alive_red = [(float(r["x"]), float(r["y"])) for r in rows
                     if int(r["id"]) >= BLUE_MAX_ID and float(r["hp"]) > 0]
        dead = [(float(r["x"]), float(r["y"])) for r in rows if float(r["hp"]) <= 0]
        blue_dots.set_data([p[0] for p in alive_blue], [p[1] for p in alive_blue])
        red_dots.set_data([p[0] for p in alive_red], [p[1] for p in alive_red])
        dead_dots.set_data([p[0] for p in dead], [p[1] for p in dead])

        truth = {int(r["id"]): r for r in rows}
        predicted_points, errors = [], []
        for link in links:
            link.set_data([], [])
        for slot, row in enumerate(predictions.get(time_sec, [])):
            unit_id = int(row["id"])
            actual = truth.get(unit_id)
            if actual is None or float(actual["hp"]) <= 0:
                continue          # 전사한 적은 예측을 겹쳐 봐야 의미가 없다
            px, py = float(row["x"]), float(row["y"])
            predicted_points.append((px, py))
            ax_, ay_ = float(actual["x"]), float(actual["y"])
            errors.append(math.hypot(px - ax_, py - ay_) * METERS_PER_UNIT)
            if slot < len(links):
                links[slot].set_data([ax_, px], [ay_, py])
        predicted_dots.set_data([p[0] for p in predicted_points], [p[1] for p in predicted_points])

        mean_error = sum(errors) / len(errors) if errors else 0.0
        # 서버에 한글 폰트가 없어 라벨은 영문으로 둔다.
        title.set_text(
            f"t={time_sec:5.1f}s   BLUE {len(alive_blue)}  RED {len(alive_red)}"
            f"   pred error {mean_error:4.1f}m"
        )
        legend.set_text("BLUE actual    RED actual    RED predicted (open)    x = dead")
        return [blue_dots, red_dots, dead_dots, predicted_dots, title, legend, *links]

    anim = animation.FuncAnimation(figure, draw, frames=len(times), interval=250, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=animation.FFMpegWriter(fps=4, bitrate=2400))
    plt.close(figure)

    total = [
        math.hypot(float(p["x"]) - float(t["x"]), float(p["y"]) - float(t["y"])) * METERS_PER_UNIT
        for time_sec, rows in predictions.items()
        for p in rows
        for t in [next((r for r in states.get(time_sec, []) if int(r["id"]) == int(p["id"])), None)]
        if t is not None and float(t["hp"]) > 0
    ]
    print(f"{out_path}  ({len(times)}프레임)")
    if total:
        print(f"  RED 예측 오차 평균 {sum(total)/len(total):.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
