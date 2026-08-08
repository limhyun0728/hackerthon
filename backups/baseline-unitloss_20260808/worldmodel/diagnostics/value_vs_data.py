"""V의 예측을 실제 데이터와 직접 맞대본다. 개입도 순간이동도 하지 않는다.

measure_value_sensitivity.py는 유닛을 목표 쪽으로 당겨 넣는데, 지형을 보지 않아
건물 안에 박힌 상태가 만들어질 수 있고 그런 상태는 학습 분포 밖이다. 또 4개 임무
전부에 "목표에 다가가면 점수가 올라야 한다"를 기대하는데, destroy_all은 목표와
무관하므로 그 기대 자체가 틀렸다.

여기서는 실제 에피소드의 실제 상태만 쓴다.
  x축 = 그 시점 BLUE의 목표까지 평균거리
  y축 = (실측) 그 에피소드의 최종 임무 달성도 / (예측) V가 매긴 점수
둘의 상관 부호가 같으면 V는 데이터를 옳게 배운 것이다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hackerthon.terrain import path_pad_for_unit_radius, set_path_pad
from hackerthon.worldmodel.slots import (
    ObjectType,
    TeamId,
    build_slot_batch,
    mission_type_from_config,
    objective_from_config,
)
from hackerthon.worldmodel.value_head import load_value_head
from hackerthon.worldmodel.value_online import mission_progress

MISSION_NAMES = ("destroy_and_reach", "destroy_all", "reach_objective", "hold_objective")


def predict(model, rows, config, time_sec, device) -> float:
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
            mission_onehot=torch.eye(4, device=device)[mission_type_from_config(config)].unsqueeze(0),
        )
    return float(prediction["progress"])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean()
    br -= br.mean()
    denominator = np.sqrt((ar**2).sum() * (br**2).sum())
    return float((ar * br).sum() / denominator) if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="V 예측과 실측 달성도를 실제 상태로 비교")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("output/rule_baseline10")])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--stride", type=int, default=4, help="몇 tick마다 표본을 뽑을지")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_value_head(args.checkpoint, device)
    per_mission: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for root in args.roots:
        summary = root / "episode_summary.jsonl"
        if not summary.exists():
            continue
        for line in summary.read_text(encoding="utf-8").splitlines()[: args.limit]:
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

            by_time: dict[float, list[dict]] = collections.defaultdict(list)
            with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    by_time[float(row["time"])].append(row)
            times = sorted(by_time)
            if len(times) < 8:
                continue

            objective = objective_from_config(config)
            mission = config["mission_type"]
            mission_id = mission_type_from_config(config)
            red_initial = sum(1 for r in by_time[times[0]] if int(r["id"]) >= 200)

            final_rows = by_time[times[-1]]
            final_blue = [r for r in final_rows if int(r["id"]) < 200 and float(r["hp"]) > 0]
            final_progress = mission_progress(
                mission_type=mission_id,
                blue_alive=len(final_blue),
                red_hp=sum(float(r["hp"]) for r in final_rows if int(r["id"]) >= 200),
                red_initial=red_initial,
                objective_distance=(
                    min(
                        ((float(b["x"]) - objective[0]) ** 2 + (float(b["y"]) - objective[1]) ** 2) ** 0.5
                        for b in final_blue
                    )
                    if final_blue
                    else float("inf")
                ),
            )

            for time_sec in times[:-1][:: args.stride]:
                rows = by_time[time_sec]
                blues = [r for r in rows if int(r["id"]) < 200 and float(r["hp"]) > 0]
                if not blues:
                    continue
                distance = float(
                    np.mean([
                        ((float(b["x"]) - objective[0]) ** 2 + (float(b["y"]) - objective[1]) ** 2) ** 0.5
                        for b in blues
                    ])
                )
                per_mission[mission]["distance"].append(distance)
                per_mission[mission]["actual"].append(final_progress)
                per_mission[mission]["predicted"].append(
                    predict(model, rows, config, time_sec, device)
                )

    if not per_mission:
        raise SystemExit("표본이 없다")

    print(f"checkpoint={args.checkpoint.name}   실제 상태만 사용(순간이동 없음)\n")
    print("목표까지 거리 vs 달성도의 순위상관 (음수 = 가까울수록 좋음)")
    print(f"{'임무':<20}{'n':>6}{'실측 데이터':>14}{'V 예측':>12}{'부호 일치':>12}")
    all_distance, all_actual, all_predicted = [], [], []
    for mission in sorted(per_mission):
        row = per_mission[mission]
        distance = np.array(row["distance"])
        actual = np.array(row["actual"])
        predicted = np.array(row["predicted"])
        all_distance.extend(distance); all_actual.extend(actual); all_predicted.extend(predicted)
        data_correlation = spearman(distance, actual)
        model_correlation = spearman(distance, predicted)
        agree = "예" if np.sign(data_correlation) == np.sign(model_correlation) else "아니오"
        print(f"{mission:<20}{len(distance):>6}{data_correlation:>+14.3f}{model_correlation:>+12.3f}{agree:>12}")

    distance = np.array(all_distance)
    print(f"\n{'전체':<20}{len(distance):>6}"
          f"{spearman(distance, np.array(all_actual)):>+14.3f}"
          f"{spearman(distance, np.array(all_predicted)):>+12.3f}")

    print("\nV 예측과 실측 달성도의 순위상관 (높을수록 V가 잘 맞춤)")
    for mission in sorted(per_mission):
        row = per_mission[mission]
        print(f"  {mission:<20}{spearman(np.array(row['predicted']), np.array(row['actual'])):>+8.3f}")
    print(f"  {'전체':<20}{spearman(np.array(all_predicted), np.array(all_actual)):>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
