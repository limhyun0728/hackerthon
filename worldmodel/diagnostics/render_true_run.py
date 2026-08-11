"""플랫폼 세션의 확정 구간을 DEVS 실제 결과로만 이어 붙여 mp4로 만든다.

전개안 미리보기는 월드모델 예측이라 BLUE까지 예측값이고, 실제보다 1.79배 많이
움직인다(실측: 프레임당 실제 4.1m 대 예측 7.4m, 정지 비율 58% 대 12%). 여기서는
`/api/select`가 돌려주는 `true_path`만 쓴다. 그건 선택된 계획 하나를 DEVS로 굴린
결과라 위치도 사격도 전사도 전부 정확하다.

사용법:
    python worldmodel/diagnostics/render_true_run.py <sel_1.json ...> --out run.mp4 \\
        --map-config output/maps/gangnam/config.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

BLUE_MAX_ID = 200
METERS_PER_UNIT = 10.0
BLUE_COLOR = "#1f6feb"
RED_COLOR = "#da3633"
FIRE_COLOR = "#f0883e"


def main() -> int:
    parser = argparse.ArgumentParser(description="DEVS 확정 구간만으로 실행 영상")
    parser.add_argument("selections", nargs="+", type=Path, help="/api/select 응답 json들")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-config", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    config = json.loads(args.map_config.read_text(encoding="utf-8"))
    frames: list[dict] = []
    for path in args.selections:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for frame in payload.get("true_path") or []:
            frames.append(frame)
    if not frames:
        raise SystemExit("true_path가 비어 있다. 플랫폼이 DEVS 결과를 안 내려줬다")

    figure, axes = plt.subplots(figsize=(11, 8), dpi=110)
    axes.set_aspect("equal")
    axes.set_facecolor("#0d1117")
    figure.patch.set_facecolor("#0d1117")

    for polygon in config.get("building_polygons") or []:
        points = polygon.get("points") if isinstance(polygon, dict) else polygon
        if points:
            axes.fill([p[0] for p in points], [p[1] for p in points],
                      facecolor="#2d333b", edgecolor="#454c56", linewidth=0.7, zorder=1)
    if not config.get("building_polygons"):
        for x0, y0, x1, y1 in config.get("obstacles", []):
            axes.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#30363d", zorder=1))

    xs = [u["x"] for f in frames for u in f["units"]]
    ys = [u["y"] for f in frames for u in f["units"]]
    axes.set_xlim(min(xs) - 2, max(xs) + 2)
    axes.set_ylim(min(ys) - 2, max(ys) + 2)
    axes.set_xticks([]); axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    blue_dots, = axes.plot([], [], "o", color=BLUE_COLOR, markersize=12, zorder=5, linestyle="none")
    red_dots, = axes.plot([], [], "o", color=RED_COLOR, markersize=12, zorder=5, linestyle="none")
    dead_dots, = axes.plot([], [], "x", color="#6e7681", markersize=10, zorder=4,
                           linestyle="none", markeredgewidth=2.5)
    fire_lines = [axes.plot([], [], "-", color=FIRE_COLOR, linewidth=2.0, alpha=0.9, zorder=6)[0]
                  for _ in range(30)]
    labels = [axes.text(0, 0, "", color="#c9d1d9", fontsize=8, zorder=7,
                        family="monospace", visible=False) for _ in range(40)]
    title = axes.text(0.015, 0.975, "", transform=axes.transAxes, color="#e6edf3",
                      fontsize=13, va="top", family="monospace")
    note = axes.text(0.015, 0.04, "DEVS ground truth  (no prediction)", transform=axes.transAxes,
                     color="#8b949e", fontsize=11, va="bottom", family="monospace")

    def draw(index: int):
        frame = frames[index]
        units = frame["units"]
        alive_blue = [(u["x"], u["y"]) for u in units if u["id"] < BLUE_MAX_ID and u["hp"] > 0]
        alive_red = [(u["x"], u["y"]) for u in units if u["id"] >= BLUE_MAX_ID and u["hp"] > 0]
        dead = [(u["x"], u["y"]) for u in units if u["hp"] <= 0]
        blue_dots.set_data([p[0] for p in alive_blue], [p[1] for p in alive_blue])
        red_dots.set_data([p[0] for p in alive_red], [p[1] for p in alive_red])
        dead_dots.set_data([p[0] for p in dead], [p[1] for p in dead])

        for label in labels:
            label.set_visible(False)
        for slot, unit in enumerate(units[: len(labels)]):
            label = labels[slot]
            side = "B" if unit["id"] < BLUE_MAX_ID else "R"
            state = "KIA" if unit["hp"] <= 0 else f"{int(unit['hp'])}"
            label.set_position((unit["x"] + 0.35, unit["y"] + 0.35))
            label.set_text(f"{side}{unit['id'] % 100} {state}")
            label.set_color("#6e7681" if unit["hp"] <= 0 else "#c9d1d9")
            label.set_visible(True)

        position = {u["id"]: (u["x"], u["y"]) for u in units}
        for line in fire_lines:
            line.set_data([], [])
        for slot, pair in enumerate((frame.get("fire") or [])[: len(fire_lines)]):
            shooter, target = position.get(pair[0]), position.get(pair[1])
            if shooter and target:
                fire_lines[slot].set_data([shooter[0], target[0]], [shooter[1], target[1]])

        fires = len(frame.get("fire") or [])
        title.set_text(
            f"t={index + 1:>2}s   BLUE {len(alive_blue)}  RED {len(alive_red)}   fire {fires}"
        )
        return [blue_dots, red_dots, dead_dots, title, note, *fire_lines, *labels]

    anim = animation.FuncAnimation(figure, draw, frames=len(frames), interval=1000 // args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.out), writer=animation.FFMpegWriter(fps=args.fps, bitrate=2600))
    plt.close(figure)

    first, last = frames[0]["units"], frames[-1]["units"]
    def alive(units, red):
        return sum(1 for u in units if (u["id"] >= BLUE_MAX_ID) == red and u["hp"] > 0)
    moved = []
    start = {u["id"]: u for u in first}
    for unit in last:
        if unit["id"] in start:
            base = start[unit["id"]]
            moved.append(math.hypot(unit["x"] - base["x"], unit["y"] - base["y"]) * METERS_PER_UNIT)
    print(f"{args.out}  ({len(frames)}프레임)")
    print(f"  BLUE {alive(first, False)}->{alive(last, False)}   RED {alive(first, True)}->{alive(last, True)}")
    print(f"  평균 이동 {sum(moved)/max(len(moved),1):.0f}m   총 사격 {sum(len(f.get('fire') or []) for f in frames)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
