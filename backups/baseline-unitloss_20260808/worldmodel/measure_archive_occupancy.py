"""MAP-Elites 축이 후보를 실제로 갈라내는지 측정한다.

셀 대부분이 비면 아카이브가 지휘관에게 보여줄 대안을 못 만든다. 인터페이스를
만들기 전에 축이 후보 다양성을 잡아내는지부터 확인한다.

축은 6스텝 rollout에서 계산 가능한 태세 지표를 쓴다.
- 교전태세 : ENGAGE 명령 수 / (생존 BLUE 수 x horizon)
- 부대 대형 : horizon 끝 시점 생존 BLUE 유닛 간 평균 쌍거리(유닛)

두 축 모두 팀 크기와 무관하게 정규화되지만, 대형은 유닛이 2명 이상이어야
정의되므로 1인 부대는 대형 축을 접는다.

사용법:
    python worldmodel/measure_archive_occupancy.py <episode_root> --candidates 200
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import path_pad_for_unit_radius, set_path_pad
from hackerthon.worldmodel.actions import ActionType
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    build_initial_distribution,
    sample_future_action_plans,
    score_future_features_torch,
)
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.slots import (
    ObjectType,
    TeamId,
    available_times,
    build_slot_batch_from_v2_run,
    load_v2_config,
    mission_type_from_config,
    objective_from_config,
)


UNIT_HP_INDEX = 1
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4

# 절대 단위 격자. 시나리오가 바뀌어도 셀 의미가 유지되도록 고정한다.
ENGAGE_EDGES = (0.0, 0.001, 0.15, 0.35, 0.60)
SPREAD_EDGES = (0.0, 2.0, 5.0, 10.0, 15.0)

ENGAGE_LABELS = ("순수기동", "산발사격", "교전", "적극교전", "전력사격")
SPREAD_LABELS = ("밀집", "근접", "분진", "산개", "광역분산")

# 교리 어휘로 읽는 셀 이름 (교전태세 x 대형)
TACTIC_NAMES = {
    (0, 0): "REGROUP",
    (0, 1): "REGROUP",
    (0, 2): "SEARCH_AND_BACK",
    (0, 3): "SEARCH_AND_BACK",
    (0, 4): "SEARCH_AND_BACK",
    (3, 0): "FOCUS_FIRE",
    (4, 0): "FOCUS_FIRE",
    (3, 1): "COVER_AND_FIRE",
    (4, 1): "COVER_AND_FIRE",
    (3, 3): "FIX_AND_FLANK",
    (3, 4): "DIVERSION_AND_FLANK",
    (4, 3): "FIX_AND_FLANK",
    (4, 4): "DIVERSION_AND_FLANK",
}


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    """값이 속한 구간 index. 양끝은 클램프한다."""
    for index in range(len(edges) - 1, -1, -1):
        if value >= edges[index]:
            return index
    return 0


def _engage_posture(plan, candidate: int, alive_blue: int, horizon: int) -> float:
    """후보 plan의 ENGAGE 비율. 유닛 수와 horizon으로 정규화한다."""
    if alive_blue <= 0 or horizon <= 0:
        return 0.0
    types = plan.action_type_ids[candidate].detach().cpu().numpy()
    issued = plan.issued_mask[candidate].detach().cpu().numpy()
    engage = int(((types == int(ActionType.ENGAGE)) & issued).sum())
    return float(engage) / float(alive_blue * horizon)


def _formation_spread(features: np.ndarray, blue_indices: np.ndarray) -> float:
    """horizon 끝 시점 생존 BLUE 유닛 간 평균 쌍거리."""
    final = features[-1, blue_indices]
    alive = final[:, UNIT_HP_INDEX] > 0.01
    positions = final[alive][:, [UNIT_X_INDEX, UNIT_Y_INDEX]]
    if positions.shape[0] < 2:
        return float("nan")
    # 정규화 좌표를 월드 단위로 되돌린다.
    xs = (positions[:, 0] + 1.0) * 0.5 * 40.0 - 20.0
    ys = (positions[:, 1] + 1.0) * 0.5 * 25.0 - 15.0
    pts = np.stack([xs, ys], axis=-1)
    dists = [
        float(np.hypot(*(pts[i] - pts[j])))
        for i, j in itertools.combinations(range(pts.shape[0]), 2)
    ]
    return float(np.mean(dists))


def measure_run(
    run_dir: Path,
    *,
    horizon: int,
    candidates: int,
    decision_points: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """에피소드 하나의 결정 시점마다 후보를 뽑아 셀 좌표를 계산한다."""
    config = load_v2_config(run_dir)
    real_map = config.get("real_map", {})
    if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
        set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

    times = list(available_times(run_dir))
    if not times:
        return []
    picked = sorted(set(np.linspace(0, len(times) - 1, num=min(decision_points, len(times)), dtype=int).tolist()))

    rows_out: list[dict[str, object]] = []
    for index in picked:
        time_sec = times[index]
        current_batch = build_slot_batch_from_v2_run(run_dir, time_sec)
        unit_rows: list[dict[str, object]] = []
        with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if float(row["time"]) != float(time_sec):
                    continue
                unit_rows.append(
                    {
                        "time": float(row["time"]),
                        "id": int(row["id"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "heading": float(row["heading"]),
                        "hp": float(row["hp"]),
                        "ammo": int(float(row["ammo"])),
                    }
                )
        alive_blue = sum(1 for r in unit_rows if int(r["id"]) < 200 and float(r["hp"]) > 0.0)
        alive_red = sum(1 for r in unit_rows if int(r["id"]) >= 200 and float(r["hp"]) > 0.0)
        if alive_blue == 0 or alive_red == 0:
            continue

        cem_config = CEMConfig(
            num_candidates=candidates,
            num_elites=max(2, candidates // 8),
            num_iterations=1,
            future_horizon=horizon,
            seed=seed + index,
            min_action_probability=0.0,
        )
        distribution = build_initial_distribution(current_batch, cem_config, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + index)
        plans = sample_future_action_plans(
            distribution=distribution,
            current_batch=current_batch,
            config=cem_config,
            generator=generator,
            device=device,
        )
        snapshot = snapshot_from_slot_rows(
            unit_rows=unit_rows,
            obstacles=config["obstacles"],
            base_time_sec=time_sec,
            episode_duration_sec=float(config["duration"]),
            objective=objective_from_config(config),
            mission_type=mission_type_from_config(config),
        )
        features = rollout_plans_with_devs(
            plans=plans, snapshot=snapshot, seed=seed + index, device=device
        )
        scores = score_future_features_torch(
            current_batch=current_batch, future_features=features
        ).detach().cpu().numpy()
        features_np = features.detach().cpu().numpy()
        blue_indices = np.flatnonzero(
            (current_batch.type_ids == int(ObjectType.UNIT))
            & (current_batch.team_ids == int(TeamId.BLUE))
        )

        for candidate in range(features_np.shape[0]):
            engage = _engage_posture(plans, candidate, alive_blue, horizon)
            spread = _formation_spread(features_np[candidate], blue_indices)
            rows_out.append(
                {
                    "run_dir": run_dir.name,
                    "time": float(time_sec),
                    "blue_alive": alive_blue,
                    "red_alive": alive_red,
                    "engage_posture": engage,
                    "formation_spread": spread,
                    "engage_bin": _bin_index(engage, ENGAGE_EDGES),
                    "spread_bin": -1 if np.isnan(spread) else _bin_index(spread, SPREAD_EDGES),
                    "score": float(scores[candidate]),
                }
            )
    return rows_out


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAP-Elites 축 점유율 측정")
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--decision-points", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-csv", type=Path, default=Path("output/archive_occupancy.csv"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device(args.device)
    root = args.episode_root
    run_dirs = (
        [root]
        if (root / "soldier_log.csv").exists()
        else sorted(p for p in root.iterdir() if (p / "soldier_log.csv").exists())
    )[: args.max_episodes]
    if not run_dirs:
        raise ValueError(f"{root} 아래에 episode가 없다")

    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        print(f"=== {run_dir.name}")
        results = measure_run(
            run_dir,
            horizon=args.horizon,
            candidates=args.candidates,
            decision_points=args.decision_points,
            seed=args.seed,
            device=device,
        )
        print(f"    후보 {len(results)}개")
        rows.extend(results)

    if not rows:
        raise ValueError("측정된 후보가 없다")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grid = np.zeros((len(ENGAGE_EDGES), len(SPREAD_EDGES)), dtype=int)
    undefined = 0
    for row in rows:
        if int(row["spread_bin"]) < 0:
            undefined += 1
            continue
        grid[int(row["engage_bin"]), int(row["spread_bin"])] += 1

    print(f"\n후보 {len(rows)}개 (대형 미정의 {undefined}개 = 1인 부대)")
    header = "교전태세 / 대형"
    print("\n" + f"{header:<14}" + "".join(f"{label:>10}" for label in SPREAD_LABELS))
    for e_index, e_label in enumerate(ENGAGE_LABELS):
        cells = "".join(f"{grid[e_index, s_index]:>10}" for s_index in range(len(SPREAD_LABELS)))
        print(f"{e_label:<14}{cells}")

    filled = int((grid > 0).sum())
    total = grid.size
    print(f"\n점유 셀 {filled}/{total} ({100 * filled / total:.0f}%)")
    print(f"최대 셀 집중도 {100 * grid.max() / max(grid.sum(), 1):.0f}%")
    print("\n교리 셀 매핑 확인:")
    for (e_index, s_index), name in sorted(TACTIC_NAMES.items()):
        count = grid[e_index, s_index]
        if count:
            print(f"  {ENGAGE_LABELS[e_index]:<6} x {SPREAD_LABELS[s_index]:<6} -> {name:<20} {count}개")
    print(f"\ncsv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
