"""6스텝 채점이 장기 결과와 같은 방향을 가리키는지 측정한다.

CEM은 6스텝 앞만 보고 후보를 고른다. 그 점수가 장기 결과와 무관하다면 계획
근거 자체가 없는 것이므로, 평가함수를 고치기 전에 이것부터 확인해야 한다.

같은 후보 plan을 긴 horizon으로 한 번만 DEVS rollout한 뒤, 앞 6프레임만으로
매긴 점수와 전체 프레임으로 매긴 점수의 순위 상관을 본다. 두 점수가 같은
rollout에서 나오므로 월드모델 예측 오차는 개입하지 않는다. 순수하게 "짧게 보는
것이 길게 보는 것을 대변하는가"만 재는 셈이다.

사용법:
    python worldmodel/measure_horizon_alignment.py output/multimap_redmask \
        --short-horizon 6 --long-horizon 40 --candidates 48
"""

from __future__ import annotations

import argparse
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관. 표본이 상수면 정의되지 않으므로 nan을 반환한다."""
    if a.size < 3:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    if ar.std() == 0.0 or br.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _top_k_overlap(short_scores: np.ndarray, long_scores: np.ndarray, k: int) -> float:
    """짧은 점수 상위 k개가 긴 점수 상위 k개와 얼마나 겹치는지."""
    k = min(k, short_scores.size)
    if k <= 0:
        return float("nan")
    top_short = set(np.argsort(-short_scores)[:k].tolist())
    top_long = set(np.argsort(-long_scores)[:k].tolist())
    return len(top_short & top_long) / float(k)


def _team_hp(features: np.ndarray, type_ids: np.ndarray, team_ids: np.ndarray, team: TeamId) -> np.ndarray:
    """후보별 팀 HP 합. features는 (C, T, N, F)."""
    unit = (type_ids == int(ObjectType.UNIT)) & (team_ids == int(team))
    indices = np.flatnonzero(unit)
    return np.clip(features[:, -1, indices, UNIT_HP_INDEX], 0.0, 1.0).sum(axis=-1)


def _snapshot_rows_at(run_dir: Path, time_sec: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["time"]) != float(time_sec):
                continue
            rows.append(
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
    return rows


def measure_run(
    run_dir: Path,
    *,
    short_horizon: int,
    long_horizon: int,
    candidates: int,
    decision_points: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """에피소드 하나에서 결정 시점별 상관을 계산한다."""
    config = load_v2_config(run_dir)
    real_map = config.get("real_map", {})
    if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
        set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

    objective = objective_from_config(config)
    mission_type = mission_type_from_config(config)
    duration_sec = float(config["duration"])
    # rollout은 snapshot에서 새로 시뮬레이션하므로 로그 길이를 넘어가도 된다.
    # 예전엔 t + long_horizon <= 로그끝 으로 걸러서 측정 시점이 거의 다 날아갔다.
    times = list(available_times(run_dir))
    if not times:
        return []
    picked = np.linspace(0, len(times) - 1, num=min(decision_points, len(times)), dtype=int)

    out: list[dict[str, object]] = []
    for index in sorted(set(picked.tolist())):
        time_sec = times[index]
        current_batch = build_slot_batch_from_v2_run(run_dir, time_sec)
        rows = _snapshot_rows_at(run_dir, time_sec)
        alive_blue = sum(
            1 for row in rows if int(row["id"]) < 200 and float(row["hp"]) > 0.0
        )
        alive_red = sum(
            1 for row in rows if int(row["id"]) >= 200 and float(row["hp"]) > 0.0
        )
        if alive_blue == 0 or alive_red == 0:
            continue

        cem_config = CEMConfig(
            num_candidates=candidates,
            num_elites=max(2, candidates // 8),
            num_iterations=1,
            future_horizon=long_horizon,
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
            unit_rows=rows,
            obstacles=config["obstacles"],
            base_time_sec=time_sec,
            episode_duration_sec=duration_sec,
            objective=objective,
            mission_type=mission_type,
        )
        features = rollout_plans_with_devs(
            plans=plans, snapshot=snapshot, seed=seed + index, device=device
        )

        short_scores = score_future_features_torch(
            current_batch=current_batch, future_features=features[:, :short_horizon]
        ).detach().cpu().numpy()
        long_scores = score_future_features_torch(
            current_batch=current_batch, future_features=features
        ).detach().cpu().numpy()

        features_np = features.detach().cpu().numpy()
        red_hp = _team_hp(features_np, current_batch.type_ids, current_batch.team_ids, TeamId.RED)
        blue_hp = _team_hp(features_np, current_batch.type_ids, current_batch.team_ids, TeamId.BLUE)

        out.append(
            {
                "run_dir": run_dir.name,
                "time": float(time_sec),
                "blue_alive": alive_blue,
                "red_alive": alive_red,
                "mission": config.get("mission_type", "?"),
                "spearman_short_vs_long": _spearman(short_scores, long_scores),
                "top8_overlap": _top_k_overlap(short_scores, long_scores, 8),
                # 최종 상태의 실제 전력을 직접 본 상관. 점수 정의와 무관한 확인.
                "spearman_short_vs_red_hp": _spearman(short_scores, -red_hp),
                "spearman_short_vs_blue_hp": _spearman(short_scores, blue_hp),
            }
        )
    return out


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6스텝 채점과 장기 결과의 방향성 측정")
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--short-horizon", type=int, default=6)
    parser.add_argument("--long-horizon", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--decision-points", type=int, default=4, help="에피소드당 측정 시점 수")
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-csv", type=Path, default=Path("output/horizon_alignment.csv"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.long_horizon <= args.short_horizon:
        raise ValueError("--long-horizon은 --short-horizon보다 커야 한다")
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
            short_horizon=args.short_horizon,
            long_horizon=args.long_horizon,
            candidates=args.candidates,
            decision_points=args.decision_points,
            seed=args.seed,
            device=device,
        )
        for row in results:
            print(
                f"  t={row['time']:5.1f} {row['blue_alive']}v{row['red_alive']:<2} "
                f"spearman(short,long)={row['spearman_short_vs_long']:+.3f} "
                f"top8={row['top8_overlap']:.2f}"
            )
        rows.extend(results)

    if not rows:
        raise ValueError("측정된 결정 시점이 없다")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        values = [float(r[key]) for r in rows if not np.isnan(float(r[key]))]
        return float(np.mean(values)) if values else float("nan")

    print(f"\n측정 시점 {len(rows)}개  (short={args.short_horizon}, long={args.long_horizon}, 후보={args.candidates})")
    print(f"  Spearman(6스텝 점수, 장기 점수)   = {mean('spearman_short_vs_long'):+.3f}")
    print(f"  top-8 겹침 비율                   = {mean('top8_overlap'):.2f}")
    print(f"  Spearman(6스텝 점수, 최종 RED 격파) = {mean('spearman_short_vs_red_hp'):+.3f}")
    print(f"  Spearman(6스텝 점수, 최종 BLUE 생존) = {mean('spearman_short_vs_blue_hp'):+.3f}")
    print(f"  csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
