"""사격 기회의 천장을 잰다.

모든 (생존 BLUE, tick) 쌍에서 **쏠 수 있는 적이 하나라도 있었는지** 센다. 이 비율이
어떤 표적 추출로도 넘을 수 없는 실행 ENGAGE 비율의 상한이다.

대조군과 재추출이 둘 다 실행 ENGAGE 23%에서 멈췄다. 그 값이 이 천장과 같으면
병목은 표적 선택이 아니라 위치(접근)다.
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

BLUE_MAX_ID = 200


def analyze(root: Path) -> dict:
    opportunities = 0
    unit_ticks = 0
    nearest: list[float] = []
    valid_counts: list[int] = []

    for run in sorted(p for p in root.glob("episode_*") if p.is_dir()):
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
        by_time: dict[float, list[dict]] = collections.defaultdict(list)
        with (run / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_time[float(row["time"])].append(row)

        for rows in by_time.values():
            blues = [r for r in rows if int(r["id"]) < BLUE_MAX_ID and float(r["hp"]) > 0]
            reds = [r for r in rows if int(r["id"]) >= BLUE_MAX_ID and float(r["hp"]) > 0]
            for blue in blues:
                unit_ticks += 1
                if not reds:
                    continue
                position = (float(blue["x"]), float(blue["y"]))
                distances = [
                    ((position[0] - float(r["x"])) ** 2 + (position[1] - float(r["y"])) ** 2) ** 0.5
                    for r in reds
                ]
                nearest.append(min(distances))
                valid = sum(
                    1
                    for red, distance in zip(reds, distances)
                    if distance <= MAX_FIRE_RANGE
                    and has_los(position, (float(red["x"]), float(red["y"])), obstacles)
                )
                valid_counts.append(valid)
                if valid > 0:
                    opportunities += 1

    return {
        "unit_ticks": unit_ticks,
        "opportunity_share": opportunities / unit_ticks if unit_ticks else 0.0,
        "nearest_median": sorted(nearest)[len(nearest) // 2] if nearest else float("nan"),
        "in_effective": (
            sum(1 for d in nearest if d <= EFFECTIVE_FIRE_RANGE) / len(nearest) if nearest else 0.0
        ),
        "mean_valid": sum(valid_counts) / len(valid_counts) if valid_counts else 0.0,
    }


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv]
    results = {root.name: analyze(root) for root in roots}
    width = max(16, max(len(n) for n in results) + 2)

    print(f"{'':26}" + "".join(f"{n:>{width}}" for n in results))
    rows = (
        ("생존 BLUE x tick 수", "unit_ticks", "{:>d}"),
        ("쏠 수 있는 적이 있던 비율", "opportunity_share", "{:>.1%}"),
        ("평균 유효 표적 수", "mean_valid", "{:>.2f}"),
        ("가장 가까운 적 중앙(m)", "nearest_median", "{:>.0f}"),
        ("유효사거리 내 비율", "in_effective", "{:>.1%}"),
    )
    for label, key, fmt in rows:
        cells = []
        for name in results:
            value = results[name][key]
            cells.append(fmt.format(value * 10 if key == "nearest_median" else value))
        print(f"  {label:<24}" + "".join(f"{c:>{width}}" for c in cells))

    print(f"\n  (유효사거리 {EFFECTIVE_FIRE_RANGE * 10:.0f}m, 최대사거리 {MAX_FIRE_RANGE * 10:.0f}m)")
    print("  '쏠 수 있는 적이 있던 비율'이 실행 ENGAGE 비율의 상한이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
