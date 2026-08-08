"""불확실 원이 실제 적을 실제로 담고 있는지 통계로 잰다.

지휘관 플랫폼은 미관측 RED를 월드모델 예측 위치에 찍고, 그 둘레에
반경 = 미관측 경과시간 x 최대 이동속도 인 원을 그린다. 그 원이 "적은 이 안에
있다"고 주장하는 셈인데, 주장이 맞는 비율(coverage)을 재지 않으면 원은
장식일 뿐이다.

두 가지 중심을 비교한다.
- 예측 중심 : 지금 화면이 그리는 방식. 기하학적 보증이 없다.
- 관측 중심 : 마지막 관측 위치가 중심. 속도 상한이 지켜지면 100%여야 한다.

진실은 DEVS 전개, 예측은 월드모델 전개다. 같은 plan을 두 백엔드에 넣어
같은 조건에서 대조한다.

사용법:
    python worldmodel/measure_belief_coverage.py \
        --checkpoint checkpoints/devs_mixed_probe.pt --scenarios 40 --device cuda:2
"""

from __future__ import annotations

import argparse
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
    ObservedActionWindow,
    build_initial_distribution,
    rollout_with_world_model,
    sample_future_action_plans,
)
from hackerthon.worldmodel.actions import ACTION_DIM
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.object_slot_attention import (
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.slots import ObjectType, build_slot_batch

# 팀별 실측 최대 이동속도 (유닛/초). 로그 p99/max 기준이며 원 반경의 근거다.
BLUE_MAX_SPEED = 1.5
RED_MAX_SPEED = 1.0
METERS_PER_UNIT = 10.0

# slot feature 안의 위치/체력 index. 정규화 좌표를 월드 좌표로 되돌릴 때 쓴다.
HP_INDEX, X_INDEX, Y_INDEX = 1, 3, 4
WORLD_W, WORLD_H = 40.0, 25.0
WORLD_X_MIN, WORLD_Y_MIN = -20.0, -15.0

# 관측 판정. commander_platform._observed_red_ids 와 같은 규칙(거리 + LOS)이다.
PERCEPTION_RANGE_UNITS = 10.0

# 한 번에 GPU에 올리는 후보 수. 16이면 10v10 시나리오에서 15GB를 넘긴다.
CHUNK_SIZE = 8


def _denormalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """정규화 좌표를 월드 좌표(유닛)로 되돌린다."""
    xs = (features[..., X_INDEX] + 1.0) * 0.5 * WORLD_W + WORLD_X_MIN
    ys = (features[..., Y_INDEX] + 1.0) * 0.5 * WORLD_H + WORLD_Y_MIN
    return xs, ys


def _has_los(start, end, rects) -> bool:
    """선분이 어느 사각형도 통과하지 않으면 시선이 트인 것으로 본다."""
    x0, y0 = start
    x1, y1 = end
    for rx0, ry0, rx1, ry1 in rects:
        # 선분-사각형 교차 : slab 방법
        dx, dy = x1 - x0, y1 - y0
        t_min, t_max = 0.0, 1.0
        blocked = True
        for pos, delta, low, high in ((x0, dx, rx0, rx1), (y0, dy, ry0, ry1)):
            if abs(delta) < 1e-12:
                if pos < low or pos > high:
                    blocked = False
                    break
                continue
            t0, t1 = (low - pos) / delta, (high - pos) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            t_min, t_max = max(t_min, t0), min(t_max, t1)
            if t_min > t_max:
                blocked = False
                break
        if blocked:
            return False
    return True


def _inside_any(x: float, y: float, rects) -> bool:
    """점이 어느 건물 사각형 안에 있는지."""
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rects)


def _penetration_depth(x: float, y: float, rects) -> float:
    """건물 안에 있을 때 가장 가까운 벽까지의 거리(유닛). 밖이면 0.

    벽에 살짝 걸친 것은 예측 오차로 읽히지만, 건물 한복판은 화면에서 바로
    "관통"으로 보인다. 위반 비율만으로는 이 둘을 구분할 수 없다.
    """
    best = 0.0
    for x0, y0, x1, y1 in rects:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        best = max(best, min(x - x0, x1 - x, y - y0, y1 - y))
    return best


def _load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config_dict = dict(payload["model_config"])
    config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def _sample_layout(config: dict, rng: np.random.Generator) -> list[dict[str, object]]:
    """자유 공간에서 진영을 나눠 무작위 부대를 배치한다."""
    from hackerthon.commander_platform import _initial_rows

    blue = int(rng.integers(2, 8))
    red = int(rng.integers(blue, 11))
    return _initial_rows(config, blue_count=blue, red_count=red, rng=rng)


def measure(
    *,
    map_configs: list[Path],
    checkpoint: Path,
    scenarios: int,
    candidates: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model, model_config = _load_model(checkpoint, device)
    print(f"모델 {checkpoint.name} history={model_config.history_frames} pred={model_config.pred_frames}")

    # step, team 별로 (원 안에 든 횟수, 전체) 를 모은다.
    keys = ("pred_red", "seen_red", "pred_blue", "seen_blue", "esc_red", "esc_blue")
    hit = {k: np.zeros(horizon) for k in keys}
    total = {k: np.zeros(horizon) for k in keys}
    errors = {"red": [[] for _ in range(horizon)], "blue": [[] for _ in range(horizon)]}
    # 물리 위반 : 예측이 건물 안이거나 한 스텝 이동 상한을 넘는 비율.
    # 화면 재생에서 벽 통과/순간이동으로 바로 보이는 항목이다.
    physics = {
        name: {"inside": 0.0, "jump": 0.0, "frames": 0.0, "steps": 0.0, "max_step": 0.0,
               "depths": []}
        for name in ("model", "devs")
    }
    unobserved_total = np.zeros(horizon)
    unobserved_hit = np.zeros(horizon)

    rng = np.random.default_rng(seed)
    done = 0
    while done < scenarios:
        config_path = map_configs[done % len(map_configs)]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        real_map = config.get("real_map", {})
        if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
            set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))
        rects = [tuple(float(v) for v in rect) for rect in config["obstacles"]]

        try:
            rows = _sample_layout(config, rng)
        except ValueError:
            done += 1
            continue

        str_rows = [
            {k: str(row[k]) for k in ("id", "x", "y", "heading", "hp", "ammo")} for row in rows
        ]
        objective = (float(rows[-1]["x"]), float(rows[-1]["y"]))
        batch = build_slot_batch(
            unit_rows=str_rows,
            obstacles=config["obstacles"],
            time_sec=0.0,
            duration_sec=60.0,
            objective=objective,
            mission_type=1,
        )
        cem_config = CEMConfig(
            num_candidates=candidates,
            num_elites=max(2, candidates // 8),
            num_iterations=1,
            future_horizon=horizon,
            seed=seed + done,
            min_action_probability=0.0,
        )
        distribution = build_initial_distribution(batch, cem_config, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + done)
        plans = sample_future_action_plans(
            distribution=distribution,
            current_batch=batch,
            config=cem_config,
            generator=generator,
            device=device,
        )

        need = int(model_config.history_frames)
        num_units = int(plans.action_unit_ids.shape[2])
        observed_window = ObservedActionWindow(
            action_features=torch.zeros(
                (need - 1, num_units, ACTION_DIM), dtype=torch.float32, device=device
            ),
            action_unit_ids=plans.action_unit_ids[0, 0]
            .unsqueeze(0)
            .expand(need - 1, num_units)
            .contiguous()
            .to(device),
            issued_mask=torch.zeros((need - 1, num_units), dtype=torch.bool, device=device),
        )
        chunks = []
        # 10v10 + 지형 슬롯 200개면 어텐션이 O(N^2)라 청크를 작게 잡아야 한다.
        for start in range(0, candidates, CHUNK_SIZE):
            index = torch.arange(start, min(start + CHUNK_SIZE, candidates), device=device)
            chunks.append(
                rollout_with_world_model(
                    model=model,
                    history_batches=tuple([batch] * need),
                    observed_actions=observed_window,
                    future_plans=plans.take_candidates(index),
                    device=device,
                )
            )
        predicted = torch.cat(chunks, 0).detach().cpu().numpy()

        snapshot = snapshot_from_slot_rows(
            unit_rows=rows,
            obstacles=config["obstacles"],
            base_time_sec=0.0,
            episode_duration_sec=60.0,
            objective=objective,
            mission_type=1,
        )
        truth = rollout_plans_with_devs(
            plans=plans, snapshot=snapshot, seed=seed + done, device=device
        ).detach().cpu().numpy()

        unit_index = np.flatnonzero(batch.type_ids == int(ObjectType.UNIT))
        entity_ids = np.array([int(batch.entity_ids[i]) for i in unit_index])
        is_red = entity_ids >= 200
        start_pos = {int(row["id"]): (float(row["x"]), float(row["y"])) for row in rows}
        cx = np.array([start_pos[i][0] for i in entity_ids])
        cy = np.array([start_pos[i][1] for i in entity_ids])

        px, py = _denormalize(predicted[:, :, unit_index])
        tx, ty = _denormalize(truth[:, :, unit_index])
        alive = truth[:, :, unit_index, HP_INDEX] > 0.01

        for step in range(horizon):
            radius_red = (step + 1) * RED_MAX_SPEED
            radius_blue = (step + 1) * BLUE_MAX_SPEED
            # 예측 중심 원 : |진실 - 예측| <= 반경 인가
            err = np.hypot(px[:, step] - tx[:, step], py[:, step] - ty[:, step])
            # 관측 중심 원 : |진실 - 마지막 관측| <= 반경 인가
            drift = np.hypot(tx[:, step] - cx[None, :], ty[:, step] - cy[None, :])
            for team, sel, radius, tag in (
                ("red", is_red, radius_red, "red"),
                ("blue", ~is_red, radius_blue, "blue"),
            ):
                mask = alive[:, step] & sel[None, :]
                if not mask.any():
                    continue
                hit[f"pred_{tag}"][step] += float((err[mask] <= radius).sum())
                total[f"pred_{tag}"][step] += float(mask.sum())
                hit[f"seen_{tag}"][step] += float((drift[mask] <= radius).sum())
                total[f"seen_{tag}"][step] += float(mask.sum())
                # 예측 자체가 도달가능 원을 벗어나는 비율. 원을 관측 위치로 옮기면
                # 이 비율만큼 예측 점이 자기 원 밖에 찍힌다.
                escape = np.hypot(px[:, step] - cx[None, :], py[:, step] - cy[None, :])
                hit[f"esc_{tag}"][step] += float((escape[mask] > radius).sum())
                total[f"esc_{tag}"][step] += float(mask.sum())
                errors[tag][step].append(err[mask])

            # 실제로 주황으로 그려지는 경우만 따로 : 미관측 RED
            for cand in range(alive.shape[0]):
                blue_pts = [
                    (tx[cand, step, i], ty[cand, step, i])
                    for i in range(len(entity_ids))
                    if not is_red[i] and alive[cand, step, i]
                ]
                for i in range(len(entity_ids)):
                    if not is_red[i] or not alive[cand, step, i]:
                        continue
                    target = (tx[cand, step, i], ty[cand, step, i])
                    seen = any(
                        np.hypot(target[0] - b[0], target[1] - b[1]) <= PERCEPTION_RANGE_UNITS
                        and _has_los(b, target, rects)
                        for b in blue_pts
                    )
                    if seen:
                        continue
                    unobserved_total[step] += 1.0
                    if err[cand, i] <= radius_red:
                        unobserved_hit[step] += 1.0

        for name, xs_, ys_, alive_ in (
            ("model", px, py, truth[:, :, unit_index, HP_INDEX] > 0.01),
            ("devs", tx, ty, alive),
        ):
            bucket = physics[name]
            bucket["frames"] += float(alive_.sum())
            for cand, step, unit in zip(*np.nonzero(alive_)):
                depth = _penetration_depth(xs_[cand, step, unit], ys_[cand, step, unit], rects)
                if depth > 0.0:
                    bucket["inside"] += 1.0
                    bucket["depths"].append(depth)
            caps = np.where(is_red, RED_MAX_SPEED, BLUE_MAX_SPEED)[None, None, :]
            moved = np.hypot(np.diff(xs_, axis=1), np.diff(ys_, axis=1))
            moving = alive_[:, 1:]
            bucket["steps"] += float(moving.sum())
            bucket["jump"] += float(((moved > caps + 0.02) & moving).sum())
            if moving.any():
                bucket["max_step"] = max(bucket["max_step"], float(moved[moving].max()))

        done += 1
        if done % 5 == 0:
            print(f"  시나리오 {done}/{scenarios}")

    return {
        "hit": hit,
        "total": total,
        "errors": errors,
        "unobserved_hit": unobserved_hit,
        "unobserved_total": unobserved_total,
        "physics": physics,
    }


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="불확실 원의 실제 포함률 측정")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/devs_mixed_probe.pt"))
    parser.add_argument("--maps-root", type=Path, default=Path("output/maps"))
    parser.add_argument(
        "--maps",
        nargs="+",
        default=None,
        help="쓸 맵 이름. 생략하면 maps-root 아래 전부. 홀드아웃 맵만 재려면 여기 지정한다",
    )
    parser.add_argument("--scenarios", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda:2")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    map_configs = sorted(args.maps_root.glob("*/config.json"))
    if args.maps:
        wanted = set(args.maps)
        map_configs = [p for p in map_configs if p.parent.name in wanted]
        missing = wanted - {p.parent.name for p in map_configs}
        if missing:
            raise ValueError(f"맵을 못 찾았다: {sorted(missing)}")
    if not map_configs:
        raise ValueError(f"{args.maps_root} 아래에 맵이 없다")
    print(f"맵 {len(map_configs)}개: {', '.join(p.parent.name for p in map_configs)}")

    result = measure(
        map_configs=map_configs,
        checkpoint=args.checkpoint,
        scenarios=args.scenarios,
        candidates=args.candidates,
        horizon=args.horizon,
        seed=args.seed,
        device=torch.device(args.device),
    )
    hit, total, errors = result["hit"], result["total"], result["errors"]

    print(f"\n{'='*74}")
    print("원 안에 실제 적이 있을 확률 (반경 = 경과초 x 최대속도)")
    print(f"{'='*74}")
    header = "".join(f"{'t+' + str(s + 1) + 's':>9}" for s in range(args.horizon))
    print(f"{'':<26}{header}")
    for label, key in (
        ("RED  예측 중심 (현재)", "pred_red"),
        ("RED  관측 중심 (대안)", "seen_red"),
        ("BLUE 예측 중심", "pred_blue"),
        ("BLUE 관측 중심", "seen_blue"),
    ):
        cells = "".join(
            f"{100 * hit[key][s] / max(total[key][s], 1):>8.1f}%" for s in range(args.horizon)
        )
        print(f"{label:<26}{cells}")
    print(f"\n표본 (RED) : " + "".join(f"{int(total['pred_red'][s]):>9}" for s in range(args.horizon)))

    print(f"\n{'-' * 74}")
    print("예측이 도달가능 원(관측 중심)을 벗어나는 비율 — 원을 옮길 때의 걸림돌")
    for label, key in (("RED", "esc_red"), ("BLUE", "esc_blue")):
        cells = "".join(
            f"{100 * hit[key][s] / max(total[key][s], 1):>8.1f}%" for s in range(args.horizon)
        )
        print(f"{label:<26}{cells}")

    unobs_total, unobs_hit = result["unobserved_total"], result["unobserved_hit"]
    if unobs_total.sum() > 0:
        print(f"\n{'실제 주황으로 그려지는 미관측 RED 만':<26}")
        cells = "".join(
            f"{100 * unobs_hit[s] / max(unobs_total[s], 1):>8.1f}%" for s in range(args.horizon)
        )
        print(f"{'  예측 중심 포함률':<26}{cells}")
        print(f"{'  표본':<26}" + "".join(f"{int(unobs_total[s]):>9}" for s in range(args.horizon)))

    print(f"\n{'='*74}")
    print("물리 위반 — 화면 재생에서 벽 통과/순간이동으로 보이는 항목")
    print(f"{'='*74}")
    for name, bucket in result["physics"].items():
        inside = 100 * bucket["inside"] / max(bucket["frames"], 1)
        jump = 100 * bucket["jump"] / max(bucket["steps"], 1)
        print(
            f"  {name:<6} 건물 안 {inside:>5.2f}%  이동상한 초과 {jump:>5.2f}%  "
            f"최대 스텝 {bucket['max_step'] * METERS_PER_UNIT:>5.1f}m/s"
        )
        if bucket["depths"]:
            depths = np.array(bucket["depths"]) * METERS_PER_UNIT
            deep = 100.0 * float((depths > 5.0).mean())
            print(
                f"         관통 깊이(벽까지) p50 {np.percentile(depths, 50):>4.1f}m  "
                f"p90 {np.percentile(depths, 90):>4.1f}m  max {depths.max():>5.1f}m  "
                f"| 5m 초과 {deep:>4.1f}%"
            )

    print(f"\n{'='*74}")
    print(f"예측 오차 분포 (m)")
    print(f"{'='*74}")
    for tag, cap in (("red", RED_MAX_SPEED), ("blue", BLUE_MAX_SPEED)):
        for step in range(args.horizon):
            if not errors[tag][step]:
                continue
            values = np.concatenate(errors[tag][step]) * METERS_PER_UNIT
            radius = (step + 1) * cap * METERS_PER_UNIT
            print(
                f"  {tag.upper():<5} t+{step + 1}s  평균 {values.mean():>6.1f}  "
                f"p50 {np.percentile(values, 50):>6.1f}  p95 {np.percentile(values, 95):>6.1f}  "
                f"| 원 반경 {radius:>5.0f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
