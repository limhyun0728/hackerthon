"""계획한 액션과 실제 실행된 액션을 비교한다.

commands_log.csv의 reason에 `plan=<종류>|step=<k>`가 박혀 있고, action 열이 실제
실행된 액션이다. 둘이 갈리는 지점이 계획의 실행 불가능성을 드러낸다.

사용법:
    python plan_vs_exec.py output/_diag output/_smoke_retarget
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

ACTIONS = ("STOP", "MOVE", "ENGAGE", "TURN")


def analyze(root: Path) -> dict:
    planned: collections.Counter = collections.Counter()
    executed: collections.Counter = collections.Counter()
    transitions: collections.Counter = collections.Counter()
    engage_by_step: dict[int, list[int]] = collections.defaultdict(list)
    total = 0

    for run_dir in sorted(p for p in root.glob("episode_*") if p.is_dir()):
        log = run_dir / "commands_log.csv"
        if not log.exists():
            continue
        with log.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reason = row["reason"]
                if "plan=" not in reason:
                    continue
                plan = reason.split("plan=")[1].split("|")[0]
                actual = row["action"]
                total += 1
                planned[plan] += 1
                executed[actual] += 1
                if plan != actual:
                    transitions[(plan, actual)] += 1
                if "step=" in reason:
                    step = int(reason.split("step=")[1].split("|")[0])
                    if plan == "ENGAGE":
                        engage_by_step[step].append(1 if actual == "ENGAGE" else 0)
    return {
        "total": total,
        "planned": planned,
        "executed": executed,
        "transitions": transitions,
        "engage_by_step": engage_by_step,
    }


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv]
    results = {root.name: analyze(root) for root in roots}
    results = {name: r for name, r in results.items() if r["total"]}
    if not results:
        raise SystemExit("분석할 command가 없다")

    names = list(results)
    width = max(14, max(len(n) for n in names) + 2)

    print(f"{'':10}" + "".join(f"{n:>{width}}" for n in names))
    print(f"{'명령 수':10}" + "".join(f"{results[n]['total']:>{width}}" for n in names))
    print()
    print("계획 -> 실행 비율")
    for action in ACTIONS:
        cells = []
        for name in names:
            r = results[name]
            plan_share = r["planned"][action] / r["total"]
            exec_share = r["executed"][action] / r["total"]
            cells.append(f"{plan_share:>6.1%} ->{exec_share:>6.1%}")
        print(f"  {action:<8}" + "".join(f"{c:>{width}}" for c in cells))

    print("\n계획과 다르게 실행된 경우")
    keys = sorted({k for r in results.values() for k in r["transitions"]})
    for key in keys:
        cells = [f"{results[n]['transitions'][key] / results[n]['total']:>6.1%}" for n in names]
        print(f"  {key[0]} -> {key[1]:<8}" + "".join(f"{c:>{width}}" for c in cells))

    print("\nENGAGE 계획 대비 실행률 (스텝별)")
    steps = sorted({s for r in results.values() for s in r["engage_by_step"]})
    for step in steps:
        cells = []
        for name in names:
            values = results[name]["engage_by_step"].get(step, [])
            cells.append(f"{sum(values) / len(values):>6.1%} n={len(values):<4}" if values else "-")
        print(f"  step {step}  " + "".join(f"{c:>{width}}" for c in cells))

    print("\n  전체   ", end="")
    for name in names:
        values = [v for vs in results[name]["engage_by_step"].values() for v in vs]
        cell = f"{sum(values) / len(values):>6.1%} n={len(values):<4}" if values else "-"
        print(f"{cell:>{width}}", end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
