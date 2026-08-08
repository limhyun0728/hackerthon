"""차단된 ENGAGE의 실패 사유를 분해한다.

재실행이 필요 없다. step 1~5의 ENGAGE 표적은 살아있는지 여부와 무관하게 RED 전체에
대해 **균등 추출**되므로(cem_planner.py:711, 781), 차단 사건을 관측했을 때 각 사유의
확률은 그 시점 state에서 세면 정확히 나온다.

  P(사유 | 차단됨) = (그 사유에 해당하는 RED 수) / (무효인 RED 수)

핵심 질문은 따로 있다: 차단 시점에 **유효한 표적이 하나라도 있었나**.
  없었다면 -> 유닛이 너무 멀다. MOVE 목적지를 고쳐야 한다 (안 B).
  있었다면 -> 표적 추출이 나빴다. 생존 필터 + stickiness로 고친다.
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hackerthon.combat_config import EFFECTIVE_FIRE_RANGE, MAX_FIRE_RANGE
from hackerthon.terrain import has_los

DEFAULT_ROOT = Path("output/_diag")
BLUE_MAX_ID = 200


def classify(shooter, target, obstacles) -> str:
    """_engage_allowed와 같은 순서로 판정한다."""
    if float(target["hp"]) <= 0.0:
        return "표적 전사"
    dx = float(shooter["x"]) - float(target["x"])
    dy = float(shooter["y"]) - float(target["y"])
    if (dx * dx + dy * dy) ** 0.5 > MAX_FIRE_RANGE:
        return "사거리 밖"
    if not has_los(
        (float(shooter["x"]), float(shooter["y"])),
        (float(target["x"]), float(target["y"])),
        obstacles,
    ):
        return "LOS 차단"
    return "유효"


def main(argv: list[str] | None = None) -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    cause_weight: dict[str, float] = collections.defaultdict(float)
    blocked_events = 0
    hopeless = 0            # 유효 표적이 하나도 없던 사건
    valid_counts: list[int] = []
    nearest_alive: list[float] = []
    hopeless_nearest: list[float] = []   # 유효 표적이 없던 사건
    lucky_nearest: list[float] = []      # 유효 표적이 있었는데 놓친 사건
    by_step: dict[int, list[int]] = collections.defaultdict(list)

    for run_dir in sorted(p for p in root.glob("episode_*") if p.is_dir()):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]

        rows_by_time: dict[float, dict[int, dict]] = collections.defaultdict(dict)
        with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows_by_time[float(row["time"])][int(row["id"])] = row

        with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reason = row["reason"]
                if "cem engage blocked" not in reason:
                    continue
                time_value = float(row["time"])
                shooter_id = int(row["unit_id"])
                state = rows_by_time.get(time_value)
                if not state or shooter_id not in state:
                    continue
                shooter = state[shooter_id]
                reds = [r for rid, r in state.items() if rid >= BLUE_MAX_ID]
                if not reds:
                    continue

                step = int(reason.split("step=")[1].split("|")[0])
                labels = [classify(shooter, red, obstacles) for red in reds]
                invalid = [label for label in labels if label != "유효"]
                if not invalid:
                    continue
                blocked_events += 1
                num_valid = len(labels) - len(invalid)
                valid_counts.append(num_valid)
                by_step[step].append(num_valid)
                if num_valid == 0:
                    hopeless += 1
                for label in invalid:
                    cause_weight[label] += 1.0 / len(invalid)

                alive = [
                    ((float(shooter["x"]) - float(r["x"])) ** 2
                     + (float(shooter["y"]) - float(r["y"])) ** 2) ** 0.5
                    for r in reds if float(r["hp"]) > 0.0
                ]
                if alive:
                    nearest_alive.append(min(alive))
                    (hopeless_nearest if num_valid == 0 else lucky_nearest).append(min(alive))

    if not blocked_events:
        raise SystemExit("차단 사건이 없다")

    print(f"차단된 ENGAGE {blocked_events}건\n")
    print("사유 분해 (균등 표적 추출 가정, 정확한 기대값)")
    for label, weight in sorted(cause_weight.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<12}{weight / blocked_events:>7.1%}")

    print(f"\n유효 표적이 하나도 없던 경우   {hopeless / blocked_events:>7.1%}"
          f"   ({hopeless}/{blocked_events})")
    print(f"유효 표적이 있었는데 놓친 경우 {1 - hopeless / blocked_events:>7.1%}")

    mean_valid = sum(valid_counts) / len(valid_counts)
    print(f"\n차단 시점 평균 유효 표적 수    {mean_valid:.2f}개")
    def report(label: str, values: list[float]) -> None:
        if not values:
            return
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        in_effective = sum(1 for v in values if v <= EFFECTIVE_FIRE_RANGE) / len(values)
        in_max = sum(1 for v in values if v <= MAX_FIRE_RANGE) / len(values)
        print(f"  {label:<22} 중앙 {median * 10:>4.0f}m   "
              f"유효사거리 내 {in_effective:>5.1%}   최대사거리 내 {in_max:>5.1%}   n={len(values)}")

    print(f"\n가장 가까운 생존 적까지 거리 "
          f"(유효 {EFFECTIVE_FIRE_RANGE * 10:.0f}m, 최대 {MAX_FIRE_RANGE * 10:.0f}m)")
    report("전체", nearest_alive)
    report("유효 표적 없던 경우", hopeless_nearest)
    report("유효 표적 있던 경우", lucky_nearest)

    print("\n스텝별 (차단 시점에 유효 표적이 있었던 비율)")
    for step in sorted(by_step):
        counts = by_step[step]
        had = sum(1 for c in counts if c > 0)
        print(f"  step {step}   {had / len(counts):>6.1%}   n={len(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
