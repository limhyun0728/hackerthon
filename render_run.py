"""v2 런 로그를 애니메이션 mp4로 렌더링한다.

사용법: python v2_two_layer/render_run.py <run_dir> [out.mp4]
건물(회색), 120도 FOV, Red 교전거리, 궤적, 사격선(주황),
현재 전술(상단), 이벤트 텍스트를 그린다.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Wedge

from hackerthon.rendering_common import draw_terrain, terrain_xy_values, unit_radius_units

run_dir = Path(sys.argv[1])
out_path = sys.argv[2] if len(sys.argv) > 2 else str(run_dir / "battle.mp4")

SUBFRAMES = 5
FPS = 10

config = json.loads((run_dir / "config.json").read_text())
UNIT_RADIUS_UNITS = unit_radius_units(config)

rows = defaultdict(dict)
with open(run_dir / "soldier_log.csv") as f:
    for r in csv.DictReader(f):
        t = float(r["time"])
        uid = int(r["id"])
        tgt = int(r["target_id"]) if r["target_id"] not in ("", "None") else None
        rows[uid][t] = (
            float(r["x"]), float(r["y"]), float(r["heading"]),
            int(r["hp"]), r["mode"], tgt,
        )

BLUE = sorted(u for u in rows if u < 200)
RED = sorted(u for u in rows if u >= 200)
T_MAX = int(max(t for u in rows for t in rows[u]))

tactic_by_time = []
with open(run_dir / "planner_log.csv") as f:
    for r in csv.DictReader(f):
        tactic_by_time.append((float(r["time"]), r["tactic"], r["decision"], r["events"]))

roles_by_time = []
with open(run_dir / "commands_log.csv") as f:
    for r in csv.DictReader(f):
        roles_by_time.append((float(r["time"]), int(r["unit_id"]), r["role"]))
role_lookup = {}
for t, uid, role in roles_by_time:
    role_lookup[(t, uid)] = role


def state(uid, t):
    ts = sorted(rows[uid])
    t = min(max(t, ts[0]), ts[-1])
    lo = max(x for x in ts if x <= t)
    hi = min(x for x in ts if x >= t)
    x0, y0, heading0, hp0, mode, tgt = rows[uid][lo]
    if hi == lo:
        return x0, y0, heading0, hp0, mode, tgt
    x1, y1, heading1, *_ = rows[uid][hi]
    a = (t - lo) / (hi - lo)
    heading_delta = (heading1 - heading0 + 180.0) % 360.0 - 180.0
    heading = (heading0 + a * heading_delta + 180.0) % 360.0 - 180.0
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0), heading, hp0, mode, tgt


def tactic_at(t):
    current = ("", "", "")
    for pt, tactic, decision, events in tactic_by_time:
        if pt <= t:
            current = (tactic, decision, events)
    return current


def role_at(uid, t):
    ticks = [pt for (pt, pu) in role_lookup if pu == uid and pt <= t]
    if not ticks:
        return "-"
    return role_lookup[(max(ticks), uid)]


plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(10, 7), dpi=110)
all_x = [value[0] for unit_rows in rows.values() for value in unit_rows.values()]
all_y = [value[1] for unit_rows in rows.values() for value in unit_rows.values()]
terrain_x, terrain_y = terrain_xy_values(config)
ax.set_xlim(min([-10.0] + all_x + terrain_x) - 1.0, max([12.0] + all_x + terrain_x) + 1.0)
ax.set_ylim(min([-10.0] + all_y + terrain_y) - 1.0, max([9.0] + all_y + terrain_y) + 1.0)
ax.set_aspect("equal")
ax.grid(alpha=0.2, linewidth=0.5)
title = ax.set_title("")
fig.text(
    0.5, 0.02,
    "two-layer urban combat | gray: building | faint sector: 120° FOV / 10u | dotted circle: Red engage 7u",
    ha="center", fontsize=8, color="0.35",
)

draw_terrain(ax, config)

BLUE_C, RED_C = "#1f77b4", "#c62828"

artists = {}
for uid in BLUE + RED:
    color = BLUE_C if uid < 200 else RED_C
    marker = "o" if uid < 200 else "s"
    fov = Wedge((0, 0), 10.0, -60.0, 60.0, fc=color, ec="none", alpha=0.025, zorder=0)
    ax.add_patch(fov)
    body = Circle((0, 0), UNIT_RADIUS_UNITS, fc=color, ec="#111827", alpha=0.9, lw=0.5, zorder=5)
    ax.add_patch(body)
    engage_ring = None
    if uid >= 200:
        engage_ring = Circle((0, 0), 7.0, fc="none", ec=RED_C, alpha=0.16, lw=0.5, ls=":", zorder=1)
        ax.add_patch(engage_ring)
    (dot,) = ax.plot([], [], marker, ms=2.2, color=color, zorder=6)
    (trail,) = ax.plot([], [], "-", lw=1.2, color=color, alpha=0.45)
    (fire,) = ax.plot([], [], "--", lw=0.9, color="#ff8f00", alpha=0.9)
    label = ax.annotate("", (0, 0), xytext=(5, -9), textcoords="offset points",
                        fontsize=7, color="#0d3b66" if uid < 200 else RED_C, zorder=6)
    artists[uid] = (fov, engage_ring, body, dot, trail, fire, label)

trails = defaultdict(list)
N_FRAMES = (T_MAX + 1) * SUBFRAMES


def draw(frame):
    t = frame / SUBFRAMES
    tactic, decision, events = tactic_at(t)
    title.set_text(f"t={t:.1f}   tactic={tactic}   last planner: {decision} ({events})")
    out = []
    for uid in BLUE + RED:
        x, y, heading, hp, mode, tgt = state(uid, t)
        fov, engage_ring, body, dot, trail, fire, label = artists[uid]
        fov.set_center((x, y))
        fov.set_theta1(heading - 60.0)
        fov.set_theta2(heading + 60.0)
        body.set_center((x, y))
        if engage_ring is not None:
            engage_ring.set_center((x, y))
        trails[uid].append((x, y))
        trail.set_data(*zip(*trails[uid]))
        dot.set_data([x], [y])
        dead = hp <= 0
        if dead:
            dot.set_marker("x")
            dot.set_color("0.45")
            label.set_color("0.45")
            fov.set_visible(False)
            body.set_visible(False)
            if engage_ring is not None:
                engage_ring.set_visible(False)
        else:
            body.set_visible(True)
        if uid < 200:
            role = role_at(uid, t)
            label.set_text(f"B{uid} {role}" + (f" hp{hp}" if 0 < hp < 100 else ""))
        else:
            label.set_text(f"R{uid}" + (f" hp{hp}" if 0 < hp < 100 else ""))
        label.xy = (x, y)
        if not dead and mode == "ENGAGE" and tgt in rows:
            tx, ty, *_ = state(tgt, t)
            fire.set_data([x, tx], [y, ty])
        else:
            fire.set_data([], [])
        out += [fov, body, dot, trail, fire, label]
        if engage_ring is not None:
            out.append(engage_ring)
    return out


anim = animation.FuncAnimation(fig, draw, frames=N_FRAMES, blit=False)
anim.save(out_path, writer=animation.FFMpegWriter(fps=FPS, bitrate=2400))
print(f"saved {out_path} ({N_FRAMES} frames, {N_FRAMES / FPS:.0f}s)")
