"""value head가 옳은 관계를 배웠는지 개입 실험으로 잰다.

MAE만 보면 "평균적으로 얼마나 맞나"는 알아도 **부호가 맞는지**는 모른다. CEM은 후보
간 상대 순위로 액션을 고르므로, 부호가 뒤집히면 MAE가 좋아도 정반대 액션을 고른다.

실제로 그런 일이 있었다. 기하 특징을 넣기 전 V는 "목표에 접근하면 달성도가 내려간다"고
예측했다 — 데이터의 실제 상관은 -0.36(가까울수록 좋음)인데 부호를 반대로 배웠고,
그 결과 CEM이 목표를 회피하는 후보를 골라 규칙 정책에 24대 2로 졌다.

원인은 구조였다. 적 HP는 unit slot feature에 직접 있어 pooling으로 그대로 전달되지만,
목표까지 거리는 mission slot과 unit slot의 **차이를 계산해야** 하는데 pooling이 평균을
내버려 상대 위치가 소실된다.

사용법:
    python worldmodel/measure_value_sensitivity.py \\
        --checkpoint checkpoints/value_head_geo.pt --device cuda:2
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import path_pad_for_unit_radius, set_path_pad
from hackerthon.worldmodel.slots import (
    ObjectType,
    TeamId,
    build_slot_batch,
    mission_type_from_config,
    objective_from_config,
)
from hackerthon.worldmodel.value_head import load_value_head

# 개입 강도. HP는 절반으로, 위치는 목표까지 거리의 70%를 좁힌다.
HP_SCALE = 0.5
APPROACH_RATIO = 0.7
# 개입 시점. episode 중반이라야 아직 전개 여지가 남아 있다.
SAMPLE_TICK_INDEX = 12


def _predict(model, rows, config, time_sec, device) -> float:
    batch = build_slot_batch(
        unit_rows=[
            {key: str(row[key]) for key in ("id", "x", "y", "heading", "hp", "ammo")}
            for row in rows
        ],
        obstacles=config["obstacles"],
        time_sec=time_sec,
        duration_sec=float(config["duration"]),
        objective=objective_from_config(config),
        mission_type=mission_type_from_config(config),
    )
    type_ids = torch.as_tensor(batch.type_ids, device=device).long()
    team_ids = torch.as_tensor(batch.team_ids, device=device).long()
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    unit = type_ids == int(ObjectType.UNIT)
    with torch.no_grad():
        prediction = model(
            features=features.unsqueeze(0),
            feature_mask=torch.as_tensor(batch.feature_mask, device=device).unsqueeze(0),
            type_ids=type_ids.unsqueeze(0),
            team_ids=team_ids.unsqueeze(0),
            alive_mask=((features[:, 1] > 0.0) & unit).unsqueeze(0),
            blue_mask=(unit & (team_ids == int(TeamId.BLUE))).unsqueeze(0),
            red_mask=(unit & (team_ids == int(TeamId.RED))).unsqueeze(0),
            mission_onehot=torch.eye(4, device=device)[
                mission_type_from_config(config)
            ].unsqueeze(0),
        )
    return float(prediction["progress"])


def measure(*, checkpoint: Path, roots: list[Path], device: torch.device, limit: int):
    model = load_value_head(checkpoint, device)
    result: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for root in roots:
        summary = root / "episode_summary.jsonl"
        if not summary.exists():
            continue
        for line in summary.read_text(encoding="utf-8").splitlines()[:limit]:
            if not line.strip():
                continue
            run_dir = Path(json.loads(line)["run_dir"])
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            real_map = config.get("real_map") or {}
            if real_map.get("unit_radius_units"):
                set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

            by_time: dict[float, list[dict[str, str]]] = {}
            with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    by_time.setdefault(float(row["time"]), []).append(row)
            times = sorted(by_time)
            if len(times) <= SAMPLE_TICK_INDEX:
                continue
            time_sec = times[SAMPLE_TICK_INDEX]
            base = [dict(row) for row in by_time[time_sec]]
            alive_blue = [r for r in base if int(r["id"]) < 200 and float(r["hp"]) > 0]
            alive_red = [r for r in base if int(r["id"]) >= 200 and float(r["hp"]) > 0]
            if not alive_blue or not alive_red:
                continue

            mission = config["mission_type"]
            baseline = _predict(model, base, config, time_sec, device)

            variant = [dict(r) for r in base]
            for row in variant:
                if int(row["id"]) >= 200:
                    row["hp"] = str(float(row["hp"]) * HP_SCALE)
            result[mission]["red_hp"].append(_predict(model, variant, config, time_sec, device) - baseline)

            variant = [dict(r) for r in base]
            for row in variant:
                if int(row["id"]) < 200:
                    row["hp"] = str(float(row["hp"]) * HP_SCALE)
            result[mission]["blue_hp"].append(_predict(model, variant, config, time_sec, device) - baseline)

            objective_x, objective_y = objective_from_config(config)
            variant = [dict(r) for r in base]
            for row in variant:
                if int(row["id"]) < 200:
                    row["x"] = str(float(row["x"]) + APPROACH_RATIO * (objective_x - float(row["x"])))
                    row["y"] = str(float(row["y"]) + APPROACH_RATIO * (objective_y - float(row["y"])))
            result[mission]["approach"].append(_predict(model, variant, config, time_sec, device) - baseline)
    return result


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="value head 민감도 개입 실험")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=[Path("output/rule_baseline10"), Path("output/multimap_p6_r2")],
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--limit", type=int, default=40, help="root당 최대 episode 수")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = measure(
        checkpoint=args.checkpoint,
        roots=list(args.roots),
        device=torch.device(args.device),
        limit=args.limit,
    )
    if not result:
        raise ValueError("측정할 episode가 없다")

    print(f"checkpoint={args.checkpoint.name}   개입 시점 t=index {SAMPLE_TICK_INDEX}\n")
    print(f"{'임무':<20}{'n':>4}{'적HP 절반':>12}{'아군HP 절반':>13}{'목표 접근':>12}")
    print(f"{'':20}{'':4}{'(+ 정상)':>12}{'(- 정상)':>13}{'(+ 정상)':>12}")

    totals = collections.defaultdict(list)
    for mission in sorted(result):
        row = result[mission]
        for key in ("red_hp", "blue_hp", "approach"):
            totals[key].extend(row[key])
        marks = []
        for key, want_positive in (("red_hp", True), ("blue_hp", False), ("approach", True)):
            value = float(np.mean(row[key])) if row[key] else float("nan")
            ok = value > 0.01 if want_positive else value < -0.01
            marks.append(f"{value:>+11.4f}{'' if ok else ' !'}")
        print(f"{mission:<20}{len(row['red_hp']):>4}" + "".join(marks))

    print()
    for key, label, want_positive in (
        ("red_hp", "적 HP 절반", True),
        ("blue_hp", "아군 HP 절반", False),
        ("approach", "목표 접근", True),
    ):
        values = np.array(totals[key])
        ok = values.mean() > 0.01 if want_positive else values.mean() < -0.01
        print(
            f"  전체 {label:<14} 평균 {values.mean():+.4f}  중앙 {np.median(values):+.4f}"
            f"   {'OK' if ok else '<-- 부호 오류'}"
        )
    print("\n! 표시는 그 임무에서 부호가 기대와 반대라는 뜻이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
