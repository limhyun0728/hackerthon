"""실행된 MOVE가 적·목표 쪽으로 가는지 잰다.

commands_log.csv의 MOVE detail은 그 tick에 실제로 밟은 waypoint다(계획 목적지가
아니라 A* 한 걸음). 그래서 "한 걸음이 거리를 줄였는지"를 직접 잰다.

  A 가설 (샘플링)  목적지가 애초에 적/목표 쪽이 아니다 -> 걸음이 거리를 못 줄인다
  B 가설 (채점)    목적지는 맞는데 그런 후보가 안 뽑힌다 -> 걸음은 줄이는데 드물다

규칙 정책과 나란히 놓고 봐야 의미가 있다.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
from pathlib import Path

import numpy as np

BLUE_MAX_ID = 200


def analyze(root: Path, limit: int) -> dict:
    toward_enemy: list[float] = []
    toward_objective: list[float] = []
    step_length: list[float] = []
    first_last_enemy: list[tuple[float, float]] = []
    moves = 0
    commands = 0

    runs = sorted(p for p in root.glob("episode_*") if p.is_dir())[:limit]
    for run in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        objective = (float(config["objective"][0]), float(config["objective"][1]))

        by_time: dict[float, dict[int, dict]] = collections.defaultdict(dict)
        with (run / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_time[float(row["time"])][int(row["id"])] = row

        log = run / "commands_log.csv"
        if not log.exists():
            continue
        with log.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                commands += 1
                if row["action"] != "MOVE":
                    continue
                time_value = float(row["time"])
                unit_id = int(row["unit_id"])
                state = by_time.get(time_value)
                if not state or unit_id not in state:
                    continue
                try:
                    waypoint = ast.literal_eval(row["detail"])
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(waypoint, tuple) or len(waypoint) != 2:
                    continue

                unit = state[unit_id]
                position = np.array([float(unit["x"]), float(unit["y"])])
                target = np.array([float(waypoint[0]), float(waypoint[1])])
                reds = [
                    np.array([float(r["x"]), float(r["y"])])
                    for rid, r in state.items()
                    if rid >= BLUE_MAX_ID and float(r["hp"]) > 0.0
                ]
                moves += 1
                step_length.append(float(np.linalg.norm(target - position)))

                if reds:
                    before = min(float(np.linalg.norm(position - r)) for r in reds)
                    after = min(float(np.linalg.norm(target - r)) for r in reds)
                    toward_enemy.append(before - after)          # 양수 = 가까워짐
                goal = np.array(objective)
                toward_objective.append(
                    float(np.linalg.norm(position - goal)) - float(np.linalg.norm(target - goal))
                )

        # 에피소드 전체로 적과의 거리가 좁혀졌는지
        times = sorted(by_time)
        if len(times) >= 2:
            def nearest(time_value: float) -> float:
                state = by_time[time_value]
                blues = [r for i, r in state.items() if i < BLUE_MAX_ID and float(r["hp"]) > 0]
                reds = [r for i, r in state.items() if i >= BLUE_MAX_ID and float(r["hp"]) > 0]
                if not blues or not reds:
                    return float("nan")
                return min(
                    ((float(b["x"]) - float(r["x"])) ** 2 + (float(b["y"]) - float(r["y"])) ** 2) ** 0.5
                    for b in blues
                    for r in reds
                )
            start, end = nearest(times[0]), nearest(times[-1])
            if np.isfinite(start) and np.isfinite(end):
                first_last_enemy.append((start, end))

    enemy = np.array(toward_enemy)
    objective_delta = np.array(toward_objective)
    pairs = np.array(first_last_enemy) if first_last_enemy else np.zeros((0, 2))
    return {
        "runs": len(runs),
        "commands": commands,
        "move_share": moves / commands if commands else 0.0,
        "step_mean": float(np.mean(step_length)) if step_length else 0.0,
        "enemy_mean": float(np.mean(enemy)) if enemy.size else float("nan"),
        "enemy_toward": float((enemy > 0).mean()) if enemy.size else float("nan"),
        "objective_mean": float(np.mean(objective_delta)) if objective_delta.size else float("nan"),
        "objective_toward": float((objective_delta > 0).mean()) if objective_delta.size else float("nan"),
        "start_gap": float(pairs[:, 0].mean()) if pairs.size else float("nan"),
        "end_gap": float(pairs[:, 1].mean()) if pairs.size else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MOVE가 적/목표 쪽으로 가는지 측정")
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    results = {root.name: analyze(root, args.limit) for root in args.roots}
    width = max(16, max(len(n) for n in results) + 2)
    print(f"{'':30}" + "".join(f"{n:>{width}}" for n in results))

    rows = (
        ("에피소드", "runs", "{:d}"),
        ("MOVE 비율", "move_share", "{:.1%}"),
        ("한 걸음 길이(m)", "step_mean", "{:.1f}"),
        ("", None, None),
        ("적에게 가까워진 걸음", "enemy_toward", "{:.1%}"),
        ("걸음당 적과 거리 변화(m)", "enemy_mean", "{:+.2f}"),
        ("", None, None),
        ("목표에 가까워진 걸음", "objective_toward", "{:.1%}"),
        ("걸음당 목표 거리 변화(m)", "objective_mean", "{:+.2f}"),
        ("", None, None),
        ("적과 거리 시작(m)", "start_gap", "{:.0f}"),
        ("적과 거리 종료(m)", "end_gap", "{:.0f}"),
    )
    for label, key, fmt in rows:
        if key is None:
            print()
            continue
        cells = []
        for name in results:
            value = results[name][key]
            if key in ("step_mean", "enemy_mean", "objective_mean", "start_gap", "end_gap"):
                value = value * 10
            cells.append(fmt.format(value))
        print(f"  {label:<28}" + "".join(f"{c:>{width}}" for c in cells))

    print("\n  '적에게 가까워진 걸음'이 50% 근처면 방향이 무작위 -> A 가설(샘플링)")
    print("  뚜렷이 50%를 넘는데 거리가 안 좁혀지면 -> 다른 원인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
