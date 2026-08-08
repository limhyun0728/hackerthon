"""현재 v2 시가지 배치를 정적 PNG로 렌더링한다."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.terrain import DEFAULT_OBSTACLES


out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/v2_urban_cover_scenario.png")
out.parent.mkdir(parents=True, exist_ok=True)

lanes = (-6.0, -3.0, 0.0, 3.0, 6.0)
blue = [(101 + i, -8.0, y, 0.0) for i, y in enumerate(lanes)]
red = [(201 + i, 9.0, y, 180.0) for i, y in enumerate(lanes)]

fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
ax.set_xlim(-10, 12)
ax.set_ylim(-9, 9)
ax.set_aspect("equal")
ax.grid(alpha=0.18, linewidth=0.6)
ax.set_title("v2 urban cover scenario | both teams FOV 120° / perception 10u")
ax.set_xlabel("east-west street axis")
ax.set_ylabel("north-south street axis")

for index, (xmin, ymin, xmax, ymax) in enumerate(DEFAULT_OBSTACLES, 1):
    width, height = xmax - xmin, ymax - ymin
    small_cover = width * height < 2.5
    ax.add_patch(Rectangle(
        (xmin, ymin), width, height,
        fc="#687b8c" if small_cover else "#8b8b8b",
        ec="#35424b" if small_cover else "#444444",
        lw=1.0,
        zorder=3,
    ))
    ax.text(
        (xmin + xmax) / 2, (ymin + ymax) / 2,
        "cover" if small_cover else f"B{index}",
        ha="center", va="center", fontsize=7, color="white", zorder=4,
    )

for uid, x, y, heading in blue + red:
    color = "#1976d2" if uid < 200 else "#c62828"
    marker = "o" if uid < 200 else "s"
    ax.add_patch(Wedge((x, y), 10.0, heading - 60, heading + 60, fc=color, ec="none", alpha=0.035, zorder=0))
    ax.plot(x, y, marker=marker, ms=8, color=color, zorder=6)
    ax.text(x, y + 0.35, str(uid), ha="center", fontsize=8, color=color, weight="bold", zorder=7)

ax.plot(10.0, 0.0, marker="*", ms=15, color="#f9a825", mec="#8d6e00", zorder=7)
ax.text(10.0, -0.55, "Blue objective (10, 0)", ha="center", fontsize=8, color="#6d4c00")
fig.text(
    0.5, 0.025,
    "dark gray: small buildings | blue-gray: vehicle/barrier cover | all obstacles block movement, sight, and fire",
    ha="center", fontsize=9, color="0.3",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out)
print(f"saved {out}")
