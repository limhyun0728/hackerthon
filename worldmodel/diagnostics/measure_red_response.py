"""BLUE 액션 대안마다 RED 예측이 달라지는지 잰다.

아카이브가 "이 전개안은 적을 잡고 저 안은 못 잡는다"를 보여주려면, 월드모델이
BLUE 행동에 따라 RED의 미래를 다르게 예측해야 한다. 그런데 plan에는 BLUE 액션만
들어가고 RED는 액션 조건이 없다. 모델이 RED를 BLUE와 무관한 배경으로 취급하면
후보 간 RED 차이는 0이 되고, 화면의 격파 수 차이는 전부 허구가 된다.

같은 상태에서 후보 plan을 여러 개 뽑아 **후보 축의 표준편차**를 잰다. DEVS로도 같이
굴려 "실제로는 얼마나 달라지는가"를 기준선으로 둔다.

    반응비 = 모델의 후보간 RED 분산 / DEVS의 후보간 RED 분산

1에 가까우면 상호작용을 잡은 것이고, 0에 가까우면 RED를 배경으로 두는 것이다.
BLUE 분산도 같이 내는데, 이건 액션이 직접 주어지므로 둘 다 커야 정상이다 —
BLUE마저 작으면 액션 조건화 자체가 안 먹는다는 뜻이다.

사용법:
    python worldmodel/diagnostics/measure_red_response.py \\
        --checkpoint checkpoints/cjepa_respos_ep0200.pt --scenarios 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hackerthon.worldmodel.cem_planner import (  # noqa: E402
    CEMConfig,
    ObservedActionWindow,
    build_initial_distribution,
    rollout_with_world_model,
    sample_future_action_plans,
)
from hackerthon.worldmodel.actions import ACTION_DIM  # noqa: E402
from hackerthon.worldmodel.devs_rollout import (  # noqa: E402
    rollout_plans_with_devs,
    snapshot_from_slot_rows,
)
from hackerthon.worldmodel.object_slot_attention import (  # noqa: E402
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.slots import ObjectType, build_slot_batch  # noqa: E402

METERS_PER_UNIT = 10.0
UNIT_X_INDEX, UNIT_Y_INDEX = 3, 4
UNIT_HP_INDEX = 1
MAX_HP = 100.0
CHUNK_SIZE = 16


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    config_dict = dict(payload["model_config"])
    config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    missing = model.load_state_dict(payload["model_state_dict"], strict=False).missing_keys
    if missing:
        print(f"  초기값 사용(checkpoint에 없음): {sorted(missing)}")
    model.eval()
    return model, config


def _denorm(values: np.ndarray, span: float, low: float) -> np.ndarray:
    return (values + 1.0) * 0.5 * span + low


def measure(
    *,
    map_configs: list[Path],
    checkpoint: Path,
    scenarios: int,
    candidates: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> None:
    from hackerthon.commander_platform import _initial_rows
    from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN

    span_x, span_y = WORLD_X_MAX - WORLD_X_MIN, WORLD_Y_MAX - WORLD_Y_MIN
    model, config = _load_model(checkpoint, device)
    print(f"모델 {checkpoint.name} history={config.history_frames} pred={config.pred_frames}")

    maps = [json.loads(p.read_text(encoding="utf-8")) for p in map_configs]
    rng = np.random.default_rng(seed)
    need = int(config.history_frames)

    keys = ("red", "blue", "red_hp", "blue_hp", "red_hp_mean", "blue_hp_mean")
    model_std = {k: [[] for _ in range(horizon)] for k in keys}
    devs_std = {k: [[] for _ in range(horizon)] for k in keys}

    done = 0
    while done < scenarios:
        cfg = maps[int(rng.integers(0, len(maps)))]
        blue = int(rng.integers(2, 8))
        red = int(rng.integers(blue, 11))
        try:
            rows = _initial_rows(cfg, blue_count=blue, red_count=red, rng=rng)
        except ValueError:
            continue
        points = [(r["x"], r["y"]) for r in rows]
        objective = points[int(rng.integers(0, len(points)))]
        batch = build_slot_batch(
            unit_rows=[{k: str(v) for k, v in r.items()} for r in rows],
            obstacles=cfg["obstacles"],
            time_sec=0.0,
            duration_sec=60.0,
            objective=objective,
            mission_type=1,
        )
        cem_config = CEMConfig(
            num_candidates=candidates,
            num_elites=max(2, candidates // 4),
            num_iterations=1,
            future_horizon=horizon,
            seed=seed + done,
            min_action_probability=0.0,
        )
        distribution = build_initial_distribution(batch, cem_config, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + done)
        plans = sample_future_action_plans(
            distribution=distribution, current_batch=batch, config=cem_config,
            generator=generator, device=device,
        )
        num_units = int(plans.action_unit_ids.shape[2])
        observed = ObservedActionWindow(
            action_features=torch.zeros((need - 1, num_units, ACTION_DIM), dtype=torch.float32, device=device),
            action_unit_ids=plans.action_unit_ids[0, 0].unsqueeze(0).expand(need - 1, num_units).contiguous().to(device),
            issued_mask=torch.zeros((need - 1, num_units), dtype=torch.bool, device=device),
        )
        chunks = []
        for start in range(0, candidates, CHUNK_SIZE):
            index = torch.arange(start, min(start + CHUNK_SIZE, candidates), device=device)
            chunks.append(
                rollout_with_world_model(
                    model=model, history_batches=tuple([batch] * need),
                    observed_actions=observed, future_plans=plans.take_candidates(index), device=device,
                )
            )
        predicted = torch.cat(chunks, 0).detach().cpu().numpy()

        snapshot = snapshot_from_slot_rows(
            unit_rows=rows, obstacles=cfg["obstacles"], base_time_sec=0.0,
            episode_duration_sec=60.0, objective=objective, mission_type=1,
        )
        truth = rollout_plans_with_devs(
            plans=plans, snapshot=snapshot, seed=seed + done, device=device
        ).detach().cpu().numpy()

        unit_index = np.flatnonzero(np.asarray(batch.type_ids) == int(ObjectType.UNIT))
        entity = np.array([int(batch.entity_ids[i]) for i in unit_index])
        is_red = entity >= 200

        for source, store in ((predicted, model_std), (truth, devs_std)):
            xs = _denorm(source[:, :, unit_index, UNIT_X_INDEX], span_x, WORLD_X_MIN) * METERS_PER_UNIT
            ys = _denorm(source[:, :, unit_index, UNIT_Y_INDEX], span_y, WORLD_Y_MIN) * METERS_PER_UNIT
            hp = np.clip(source[:, :, unit_index, UNIT_HP_INDEX], 0.0, 1.0) * MAX_HP
            for step in range(horizon):
                # 후보 축(axis=0)의 표준편차 = "액션 대안에 따라 얼마나 달라지는가"
                spread = np.hypot(xs[:, step].std(axis=0), ys[:, step].std(axis=0))
                store["red"][step].append(float(spread[is_red].mean()))
                store["blue"][step].append(float(spread[~is_red].mean()))
                # HP는 팀 합계로 본다. 지휘관이 보는 것은 "이 안이 적을 얼마나 깎나"다.
                red_total = hp[:, step][:, is_red].sum(axis=1)
                blue_total = hp[:, step][:, ~is_red].sum(axis=1)
                store["red_hp"][step].append(float(red_total.std()))
                store["blue_hp"][step].append(float(blue_total.std()))
                store["red_hp_mean"][step].append(float(red_total.mean()))
                store["blue_hp_mean"][step].append(float(blue_total.mean()))
        done += 1
        if done % 5 == 0:
            print(f"  시나리오 {done}/{scenarios}")

    print("\n" + "=" * 74)
    print("액션 대안 간 위치 표준편차 (m) — 후보가 다르면 위치가 얼마나 달라지나")
    print("=" * 74)
    print(f"{'':>7}{'RED 모델':>10}{'RED DEVS':>10}   {'BLUE 모델':>11}{'BLUE DEVS':>11}")
    for step in range(horizon):
        rm, rd = np.mean(model_std["red"][step]), np.mean(devs_std["red"][step])
        bm, bd = np.mean(model_std["blue"][step]), np.mean(devs_std["blue"][step])
        print(f"  t+{step+1}s{rm:>10.2f}{rd:>10.2f}{bm:>11.2f}{bd:>11.2f}")

    print("\n" + "=" * 74)
    print("액션 대안 간 팀 HP 합계 표준편차 — 후보를 고를 신호가 여기 있는가")
    print("=" * 74)
    print(f"{'':>7}{'RED 모델':>10}{'RED DEVS':>10}{'반응비':>9}   {'BLUE 모델':>11}{'BLUE DEVS':>11}"
          f"   {'RED HP평균(DEVS)':>18}")
    for step in range(horizon):
        rm, rd = np.mean(model_std["red_hp"][step]), np.mean(devs_std["red_hp"][step])
        bm, bd = np.mean(model_std["blue_hp"][step]), np.mean(devs_std["blue_hp"][step])
        rmean = np.mean(devs_std["red_hp_mean"][step])
        print(f"  t+{step+1}s{rm:>10.2f}{rd:>10.2f}{(rm/rd if rd > 1e-9 else 0):>9.2f}"
              f"{bm:>11.2f}{bd:>11.2f}{rmean:>18.1f}")
    print("\n반응비 1에 가까우면 BLUE 행동에 따른 RED 변화를 잡은 것,")
    print("0에 가까우면 RED를 BLUE와 무관한 배경으로 예측하는 것이다.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="액션 대안 간 RED 예측 변화 측정")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, default=Path("output/maps"))
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args(argv)

    map_configs = sorted(p / "config.json" for p in args.maps_root.iterdir() if (p / "config.json").exists())
    if not map_configs:
        raise ValueError(f"{args.maps_root} 아래에 맵이 없다")
    print(f"맵 {len(map_configs)}개: {', '.join(p.parent.name for p in map_configs)}")
    measure(
        map_configs=map_configs, checkpoint=args.checkpoint, scenarios=args.scenarios,
        candidates=args.candidates, horizon=args.horizon, seed=args.seed,
        device=torch.device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
