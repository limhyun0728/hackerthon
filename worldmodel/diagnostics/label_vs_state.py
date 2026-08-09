"""Monte Carlo 라벨이 상태 정보를 담고 있는지 임무별로 잰다.

train_value_head는 에피소드 종료 상태로 라벨 하나를 만들고, 그 에피소드의 **모든
시점 상태**에 같은 값을 붙인다. 그러면 초반 상태와 후반 상태가 같은 라벨을 받는다.

destroy_all은 달성도가 적 잔여 HP의 단조 함수라 이 문제가 특히 크다. 실제로 V의
순위상관이 +0.006(신호 없음)에서 데이터를 4배 늘린 뒤 -0.256(부호 반대)이 됐다.

여기서는 각 시점 상태의 관측 가능한 양이 그 에피소드 라벨과 얼마나 맞는지 본다.
  현재 적 HP 비   vs  라벨
  현재 목표 거리   vs  라벨
상관이 0 근처면 라벨이 그 상태를 구분하지 못한다는 뜻이다.

사용법:
    python worldmodel/diagnostics/label_vs_state.py output/statickv_rule output/unitloss_rule
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hackerthon.worldmodel.value_head import OBJECTIVE_GAP_SCALE
from hackerthon.worldmodel.slots import (
    MAX_HP,
    MISSION_DESTROY_ALL,
    MISSION_HOLD_OBJECTIVE,
    MISSION_REACH_OBJECTIVE,
    OBJECTIVE_RADIUS,
    mission_type_from_config,
    objective_from_config,
)

BLUE_MAX_ID = 200


def episode_label(entry: dict, config: dict) -> float:
    """train_value_head._labels_from_summary와 같은 계산."""
    initial = config.get("initial_positions", {})
    red_initial = max(len(initial.get("red", [])), 1)
    red_hp = float(entry.get("red_hp", 0.0))
    distance = float(entry.get("objective_distance", OBJECTIVE_GAP_SCALE))
    if not math.isfinite(distance):
        distance = OBJECTIVE_GAP_SCALE
    destroy = 1.0 - min(1.0, red_hp / (red_initial * MAX_HP))
    reach = 1.0 - min(1.0, max(0.0, distance - OBJECTIVE_RADIUS) / OBJECTIVE_GAP_SCALE)
    mission = mission_type_from_config(config)
    if int(entry.get("blue_alive", 0)) <= 0:
        return 0.0
    if mission == MISSION_DESTROY_ALL:
        return destroy
    if mission in (MISSION_REACH_OBJECTIVE, MISSION_HOLD_OBJECTIVE):
        return reach
    return min(destroy, reach)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean()
    br -= br.mean()
    denominator = math.sqrt(float((ar**2).sum() * (br**2).sum()))
    return float((ar * br).sum() / denominator) if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="MC 라벨이 상태를 구분하는지 측정")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--limit", type=int, default=400, help="root당 최대 에피소드")
    args = parser.parse_args()

    rows: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for root in args.roots:
        summary = root / "episode_summary.jsonl"
        if not summary.exists():
            continue
        for line in summary.read_text(encoding="utf-8").splitlines()[: args.limit]:
            if not line.strip():
                continue
            entry = json.loads(line)
            run = Path(entry["run_dir"])
            config_path = run / "config.json"
            log_path = run / "soldier_log.csv"
            if not config_path.exists() or not log_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            label = episode_label(entry, config)
            mission = config["mission_type"]
            objective = objective_from_config(config)

            by_time: dict[float, list[dict]] = collections.defaultdict(list)
            with log_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    by_time[float(row["time"])].append(row)
            times = sorted(by_time)
            if len(times) < 4:
                continue
            red_initial = max(sum(1 for r in by_time[times[0]] if int(r["id"]) >= BLUE_MAX_ID), 1)

            for time_value in times[:-1][:: args.stride]:
                state = by_time[time_value]
                blues = [r for r in state if int(r["id"]) < BLUE_MAX_ID and float(r["hp"]) > 0]
                if not blues:
                    continue
                red_hp = sum(float(r["hp"]) for r in state if int(r["id"]) >= BLUE_MAX_ID)
                rows[mission]["red_hp_ratio"].append(red_hp / (red_initial * MAX_HP))
                rows[mission]["objective_distance"].append(
                    min(
                        math.hypot(float(b["x"]) - objective[0], float(b["y"]) - objective[1])
                        for b in blues
                    )
                )
                rows[mission]["time_ratio"].append(time_value / max(times[-1], 1e-6))
                rows[mission]["label"].append(label)

    if not rows:
        raise SystemExit("표본이 없다")

    print("에피소드 라벨(종료 상태 하나)이 각 시점 상태와 얼마나 맞는지\n")
    print(f"{'임무':<20}{'n':>7}{'현재 적HP비':>14}{'목표까지 거리':>15}{'경과 시간비':>14}")
    print(f"{'':20}{'':7}{'(음수 정상)':>14}{'(음수 정상)':>15}{'(양수 정상)':>14}")
    for mission in sorted(rows):
        row = rows[mission]
        label = np.array(row["label"])
        cells = []
        for key in ("red_hp_ratio", "objective_distance", "time_ratio"):
            cells.append(f"{spearman(np.array(row[key]), label):>+14.3f}")
        print(f"  {mission:<18}{len(label):>7}" + "".join(cells))

    print("\n  현재 적HP비 : 적을 많이 깎았으면 라벨(달성도)이 높아야 하므로 음수가 정상")
    print("  목표까지 거리: 가까울수록 달성도가 높아야 하므로 음수가 정상")
    print("  경과 시간비  : 라벨이 그 에피소드 최종값 하나라 시점과 무관해야 정상(0 근처)")
    print("\n  상관이 0 근처면 라벨이 그 상태를 구분하지 못한다는 뜻이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
