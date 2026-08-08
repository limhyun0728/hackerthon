import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge
import numpy as np
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.combat_config import PERCEPTION_RANGE


def draw_belief_frame(frame_time, group, ax):
    # 수정: commanderMap.belief_rows() 로그를 재생하는 지휘관 시점 렌더러.
    ax.clear()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-15, 10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f"Commander Belief Map  t={frame_time:.2f} sec")
    ax.set_aspect('equal')

    enemy_xy = {}
    for _, row in group.iterrows():
        if str(row["kind"]) != "enemy":
            continue
        eid = int(row["entity_id"])
        status = str(row["status"])
        x, y = float(row["x"]), float(row["y"])
        if "bearing" in status:
            wb = float(row["bearing_deg"])
            rad = {"near": 1.5, "mid": 3.0, "far": 6.0}.get(str(row["bucket"]), 3.0)
            enemy_xy[eid] = (x + rad * np.cos(np.radians(wb)), y + rad * np.sin(np.radians(wb)))
        else:
            enemy_xy[eid] = (x, y)

    for _, row in group.iterrows():
        eid = int(row["entity_id"])
        kind = str(row["kind"])
        status = str(row["status"])
        x, y = float(row["x"]), float(row["y"])

        if kind == "friendly":
            if status == "in_contact":
                ax.add_patch(Circle((x, y), PERCEPTION_RANGE, fill=False, ls=":", lw=1.4, ec="blue", alpha=0.34))
                ax.plot(x, y, "o", color="blue", markersize=12, markeredgecolor="black")
                ax.text(x, y + 0.7, f"B{eid}", color="blue", fontsize=8, ha="center")

                if str(row.get("mode", "")) == "ENGAGE" and pd.notna(row.get("target_id")) and str(row.get("target_id")) != "":
                    tid = int(float(row["target_id"]))
                    if tid in enemy_xy:
                        tx, ty = enemy_xy[tid]
                        ax.plot([x, tx], [y, ty], color="orange", lw=1.8)
                        ax.plot(x, y, "o", color="none", markersize=15, markeredgecolor="orange", markeredgewidth=2)
            else:
                ax.plot(x, y, "s", color="gray", markersize=12, markeredgecolor="black")
                ax.text(x, y + 0.7, f"B{eid}", color="gray", fontsize=8, ha="center")
                ax.text(x, y - 1.0, "STOP/OOC", color="gray", fontsize=6, ha="center")
        else:
            if status == "dead":
                ax.plot(x, y, "X", color="black", markersize=13)
                ax.text(x, y - 1.0, f"R{eid} KIA", color="black", fontsize=7, ha="center")
            elif "bearing" in status:
                wb = float(row["bearing_deg"])
                bucket = str(row["bucket"])
                rad = {"near": 1.5, "mid": 3.0, "far": 6.0}.get(bucket, 3.0)
                tx = x + rad * np.cos(np.radians(wb))
                ty = y + rad * np.sin(np.radians(wb))
                ax.annotate("", xy=(tx, ty), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5, linestyle="--"))
                ax.text(tx, ty + 0.5, f"R{eid} ~{bucket}", color="red", fontsize=7, ha="center")
            else:
                alpha = 1.0 if bool(row.get("observed_this_tick", True)) else 0.45
                ax.plot(x, y, "s", color="red", markersize=12, markeredgecolor="black", alpha=alpha)
                ax.text(x, y + 0.7, f"R{eid}", color="red", fontsize=8, ha="center", alpha=alpha)


def draw_frame(frame_time, group, ax):
    ax.clear()

    ax.set_xlim(-20, 20)
    ax.set_ylim(-15, 10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f"Simulation Time: {frame_time:.2f} sec")
    ax.set_aspect('equal')

    entities = {row['id']: row for _, row in group.iterrows()}

    for _, row in group.iterrows():
        eid = int(row['id'])
        x, y = float(row['x']), float(row['y'])
        heading = float(row['heading'])
        hp = float(row['hp'])
        mode = str(row['mode'])

        is_dead = hp <= 0 or mode == 'DESTROYED'

        if 100 <= eid < 200:
            color = 'blue'
            marker = 'X' if is_dead else 'o'
            label = f"B{eid}({hp})"
        elif 200 <= eid < 300:
            color = 'red'
            marker = 'X' if is_dead else 's'
            label = f"R{eid}({hp})"
        else:
            color = 'gray'
            marker = 's'
            label = f"C{eid}"
            is_dead = False

        if is_dead:
            ax.plot(x, y, marker=marker, color='black', markersize=10)
            ax.text(x, y - 1.0, "KIA", color='black', fontsize=8, ha='center')
        else:
            ax.plot(x, y, marker=marker, color=color, markersize=10, markeredgecolor='black')
            ax.text(x, y + 0.8, label, color=color, fontsize=8, ha='center')

            dx = np.cos(np.radians(heading)) * 1.5
            dy = np.sin(np.radians(heading)) * 1.5
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc=color, ec=color)

            fov_angle = 120.0
            theta1 = heading - fov_angle / 2
            theta2 = heading + fov_angle / 2
            wedge = Wedge((x, y), PERCEPTION_RANGE, theta1, theta2, color=color, alpha=0.07)
            ax.add_patch(wedge)

        if mode == 'ENGAGE' and not is_dead:
            target_id = row.get('target_id')
            if pd.notna(target_id):
                target_id = int(float(target_id))
                if target_id in entities:
                    tx = float(entities[target_id]['x'])
                    ty = float(entities[target_id]['y'])

                    ax.plot([x, tx], [y, ty], color='red', linestyle=':', linewidth=2)
                    ax.plot(tx, ty, marker='*', color='yellow', markersize=8)


def main():
    parser = argparse.ArgumentParser(description="M&S Battle Visualizer")
    parser.add_argument('--log', type=str, default=None, help='CSV log file path')
    parser.add_argument('--view', type=str, default='god', choices=['god', 'commander'])
    parser.add_argument('--speed', type=int, default=500, help='Animation speed in ms')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--save', type=str, default=None, help='Save animation path: .mp4 or .gif')
    args = parser.parse_args()

    if args.live and args.save:
        raise ValueError("--live와 --save는 동시에 사용할 수 없습니다. 저장은 replay 모드에서만 지원합니다.")

    # 수정: commander view는 지휘관 belief 로그를 기본 입력으로 사용한다.
    if args.log is None:
        args.log = "commander_belief_log.csv" if args.view == "commander" else "soldier_commander_log.csv"
    frame_fn = draw_belief_frame if args.view == "commander" else draw_frame

    print(f"[{args.log}] 파일을 읽어 시각화를 시작합니다...")

    try:
        df = pd.read_csv(args.log)
    except FileNotFoundError:
        print(f"에러: {args.log} 파일을 찾을 수 없습니다. 시뮬레이션을 먼저 실행해주세요.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    def load_latest_frame():
        if not os.path.exists(args.log):
            return None, None

        latest_df = pd.read_csv(args.log)
        if latest_df.empty:
            return None, None

        latest_time = latest_df["time"].max()
        latest_group = latest_df[latest_df["time"] == latest_time]
        return latest_time, latest_group

    if args.live:
        def update(_):
            frame_time, group = load_latest_frame()
            if group is None:
                return
            frame_fn(frame_time, group, ax)

        ani = animation.FuncAnimation(
            fig,
            update,
            interval=args.speed,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()

    else:
        grouped = dict(tuple(df.groupby('time')))
        times = sorted(grouped.keys())

        def update(frame):
            frame_time = times[frame]
            frame_fn(frame_time, grouped[frame_time], ax)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=args.speed,
            repeat=False,
        )

        plt.tight_layout()

        if args.save:
            ext = os.path.splitext(args.save)[1].lower()
            fps = max(1, int(round(1000 / args.speed)))

            if ext == ".mp4":
                writer = animation.FFMpegWriter(fps=fps)
            elif ext == ".gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                raise ValueError("--save는 .mp4 또는 .gif 확장자만 지원합니다.")

            ani.save(args.save, writer=writer, dpi=150)
            print(f"[저장 완료] {args.save}")

        else:
            plt.show()


if __name__ == "__main__":
    main()
