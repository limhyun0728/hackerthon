"""에피소드의 각 결심 시점에서 월드모델이 예측한 RED 위치를 뽑아 CSV로 남긴다.

실제 RED과 예측 RED을 화면에 겹쳐 보기 위한 것이다. 결심 주기(pred_frames)마다
그 시점 상태로 예측을 한 번 돌리고, 이후 k스텝의 예측 위치를 t0+k 시각에 붙인다.
즉 화면의 매 순간에는 "가장 최근 결심에서 이만큼 앞을 내다본 값"이 표시된다.

사용법:
    python worldmodel/diagnostics/make_prediction_log.py \\
        --run-dir output/statickv_rule/episode_0007_... \\
        --checkpoint checkpoints/cjepa_statickv.pt --device cuda:2
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
from hackerthon.worldmodel.actions import ACTION_DIM, ActionType
from hackerthon.worldmodel.object_slot_attention import (
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.slots import (
    ObjectType,
    TeamId,
    build_slot_batch,
    mission_type_from_config,
    objective_from_config,
)

BLUE_MAX_ID = 200
UNIT_HP_INDEX, UNIT_X_INDEX, UNIT_Y_INDEX = 1, 3, 4
# cem_planner의 action feature 배치와 같다.
MOVE_X_INDEX, MOVE_Y_INDEX = 2, 3
TARGET_TEAM_INDEX, TARGET_X_INDEX, TARGET_Y_INDEX = 5, 6, 7


def _world_bounds() -> tuple[float, float, float, float]:
    from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN

    return WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX


def _denorm(x_norm: float, y_norm: float) -> tuple[float, float]:
    x0, x1, y0, y1 = _world_bounds()
    return (x_norm + 1.0) * 0.5 * (x1 - x0) + x0, (y_norm + 1.0) * 0.5 * (y1 - y0) + y0


def _actions_at(commands: dict[float, list[dict]], time_sec: float, unit_ids, device):
    """그 tick에 실제로 나간 BLUE 명령을 action feature로 만든다.

    예측을 실제와 같은 조건에서 재현하려면 그때 내려진 명령을 그대로 넣어야 한다.
    """
    rows = {int(r["unit_id"]): r for r in commands.get(time_sec, [])}
    features = torch.zeros((len(unit_ids), ACTION_DIM), dtype=torch.float32, device=device)
    issued = torch.zeros((len(unit_ids),), dtype=torch.bool, device=device)
    x0, x1, y0, y1 = _world_bounds()
    for index, unit_id in enumerate(unit_ids):
        row = rows.get(int(unit_id))
        if row is None:
            continue
        issued[index] = True
        action = str(row.get("action", "STOP"))
        detail = str(row.get("detail", ""))
        if action == "MOVE" and detail.startswith("("):
            try:
                tx, ty = (float(v) for v in detail.strip("()").split(","))
            except ValueError:
                continue
            features[index, 0] = float(ActionType.MOVE)
            features[index, 1] = 1.0
            features[index, MOVE_X_INDEX] = (tx - x0) / (x1 - x0) * 2.0 - 1.0
            features[index, MOVE_Y_INDEX] = (ty - y0) / (y1 - y0) * 2.0 - 1.0
        elif action == "ENGAGE":
            features[index, 0] = float(ActionType.ENGAGE)
            features[index, 4] = 1.0
            features[index, TARGET_TEAM_INDEX] = float(TeamId.RED)
        elif action == "TURN":
            features[index, 0] = float(ActionType.TURN)
    return features, issued


def main() -> int:
    parser = argparse.ArgumentParser(description="RED 예측 위치 로그 생성")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ObjectSlotModelConfig(**{
        **dict(payload["model_config"]),
        "maskable_type_ids": tuple(payload["model_config"]["maskable_type_ids"]),
    })
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    run_config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    real_map = run_config.get("real_map") or {}
    if real_map.get("unit_radius_units"):
        set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

    by_time: dict[float, list[dict]] = collections.defaultdict(list)
    with (args.run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_time[float(row["time"])].append(row)
    commands: dict[float, list[dict]] = collections.defaultdict(list)
    command_path = args.run_dir / "commands_log.csv"
    if command_path.exists():
        with command_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                commands[float(row["time"])].append(row)

    times = sorted(by_time)
    history_frames, pred_frames = config.history_frames, config.pred_frames
    objective = objective_from_config(run_config)
    mission = mission_type_from_config(run_config)

    def batch_at(time_sec: float):
        return build_slot_batch(
            unit_rows=[
                {key: str(r[key]) for key in ("id", "x", "y", "heading", "hp", "ammo")}
                for r in by_time[time_sec]
            ],
            obstacles=run_config["obstacles"],
            time_sec=time_sec,
            duration_sec=float(run_config["duration"]),
            objective=objective,
            mission_type=mission,
        )

    out_path = args.out or (args.run_dir / "prediction_log.csv")
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "id", "x", "y", "hp", "from_time"])
        # 결심 주기마다 한 번 예측한다.
        for start in range(0, len(times) - 1, pred_frames):
            t0 = times[start]
            history_times = [times[max(0, start - offset)] for offset in reversed(range(history_frames))]
            batches = [batch_at(t) for t in history_times]
            reference = batches[-1]
            unit_ids = [
                int(reference.entity_ids[i])
                for i in range(len(reference.type_ids))
                if int(reference.type_ids[i]) == int(ObjectType.UNIT)
                and int(reference.entity_ids[i]) < BLUE_MAX_ID
            ]
            stack = lambda key, dtype: torch.stack(  # noqa: E731
                [torch.as_tensor(getattr(b, key), device=device).to(dtype) for b in batches], dim=0
            ).unsqueeze(0)
            action_features, issued = [], []
            for offset in range(history_frames + pred_frames - 1):
                index = start - (history_frames - 1) + offset
                time_sec = times[index] if 0 <= index < len(times) else t0
                feature, mask = _actions_at(commands, time_sec, unit_ids, device)
                action_features.append(feature)
                issued.append(mask)
            with torch.no_grad():
                output = model.rollout_cjepa_future(
                    history_features=stack("features", torch.float32),
                    history_feature_mask=stack("feature_mask", torch.bool),
                    history_type_ids=stack("type_ids", torch.long),
                    history_entity_ids=stack("entity_ids", torch.long),
                    history_team_ids=stack("team_ids", torch.long),
                    history_alive_mask=stack("alive_mask", torch.bool),
                    action_features=torch.stack(action_features, dim=0).unsqueeze(0),
                    action_unit_ids=torch.tensor(unit_ids, device=device)
                    .reshape(1, 1, -1)
                    .expand(1, history_frames + pred_frames - 1, len(unit_ids))
                    .contiguous(),
                    issued_mask=torch.stack(issued, dim=0).unsqueeze(0),
                )
            future = output["future_features"][0].detach().cpu().numpy()
            red_slots = [
                i
                for i in range(len(reference.type_ids))
                if int(reference.type_ids[i]) == int(ObjectType.UNIT)
                and int(reference.entity_ids[i]) >= BLUE_MAX_ID
            ]
            for step in range(pred_frames):
                index = start + step + 1
                if index >= len(times):
                    break
                for slot in red_slots:
                    x, y = _denorm(float(future[step, slot, UNIT_X_INDEX]), float(future[step, slot, UNIT_Y_INDEX]))
                    writer.writerow([
                        f"{times[index]:.1f}",
                        int(reference.entity_ids[slot]),
                        f"{x:.4f}",
                        f"{y:.4f}",
                        f"{float(future[step, slot, UNIT_HP_INDEX]) * 100.0:.1f}",
                        f"{t0:.1f}",
                    ])
                    written += 1

    print(f"{out_path}  ({written}행)")

    # 실측과의 오차를 같이 알려 준다.
    truth = {(float(r["time"]), int(r["id"])): (float(r["x"]), float(r["y"]))
             for rows in by_time.values() for r in rows if int(r["id"]) >= BLUE_MAX_ID}
    errors = []
    with out_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (float(row["time"]), int(row["id"]))
            if key in truth:
                tx, ty = truth[key]
                errors.append(np.hypot(float(row["x"]) - tx, float(row["y"]) - ty))
    if errors:
        errors = np.array(errors) * 10.0
        print(f"  RED 예측 오차  평균 {errors.mean():.1f}m  중앙 {np.median(errors):.1f}m  최대 {errors.max():.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
