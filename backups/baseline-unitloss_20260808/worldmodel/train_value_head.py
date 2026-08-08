"""에피소드 로그에서 value head를 학습한다.

라벨은 Monte Carlo다. 각 상태에 그 상태가 속한 에피소드의 실제 종료 결과를 붙인다.
따라서 학습된 V는 "규칙 기반 BLUE로 계속 갔을 때의 가치"다.

**분할은 에피소드 단위로 한다.** 같은 에피소드의 인접 시점은 상태가 거의 같고 라벨은
완전히 같으므로, 시점 단위로 섞어 나누면 검증셋에 학습 정보가 새어 최종 성능이
실제보다 좋게 나온다.

사용법:
    python worldmodel/train_value_head.py output/multimap_p6_replay output/multimap_p6_r2 \\
        --out checkpoints/value_head.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import path_pad_for_unit_radius, set_path_pad
from hackerthon.worldmodel.slots import (
    MAX_HP,
    ObjectType,
    TeamId,
    build_slot_batch,
    mission_type_from_config,
    objective_from_config,
)
from hackerthon.worldmodel.slots import OBJECTIVE_RADIUS
from hackerthon.worldmodel.value_head import (
    MISSION_DESTROY_ALL,
    MISSION_HOLD_OBJECTIVE,
    MISSION_REACH_OBJECTIVE,
    OBJECTIVE_GAP_SCALE,
    OUTCOME_LOSE,
    OUTCOME_NAMES,
    OUTCOME_TIMEOUT,
    OUTCOME_WIN,
    ValueHead,
    ValueHeadConfig,
    load_value_head,
    save_value_head,
    scalar_value,
)

OUTCOME_BY_NAME = {"WIN": OUTCOME_WIN, "LOSE": OUTCOME_LOSE, "TIMEOUT": OUTCOME_TIMEOUT}

# 보조 손실 가중치. 0이면 달성도 회귀만 학습한다. --aux-weight로 덮어쓴다.
AUX_WEIGHT = 0.3


def _episode_rows(run_dir: Path) -> dict[float, list[dict[str, str]]]:
    """soldier_log를 시각별로 묶는다."""
    by_time: dict[float, list[dict[str, str]]] = {}
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_time.setdefault(float(row["time"]), []).append(row)
    return by_time


def _labels_from_summary(entry: dict, config: dict) -> dict[str, float]:
    """에피소드 종료 상태에서 임무 달성도와 보조 목표를 만든다.

    달성도는 mission_completed()의 불리언 조건을 그대로 연속 완화한 값이다.
    임의 가중치를 쓰지 않는다 — destroy/reach의 정의와 둘을 잇는 min이 모두
    원래 승리 조건에서 나온다.
    """
    initial = config.get("initial_positions", {})
    blue_initial = max(len(initial.get("blue", [])), 1)
    red_initial = max(len(initial.get("red", [])), 1)

    blue_hp = float(entry.get("blue_hp", 0.0))
    red_hp = float(entry.get("red_hp", 0.0))
    # BLUE가 전멸하면 "생존 BLUE 중 목표까지 최소 거리"가 정의되지 않아 inf가 온다.
    # 그대로 두면 보조 라벨에 nan이 번지므로 여기서 척도 상한으로 눌러둔다.
    distance = float(entry.get("objective_distance", OBJECTIVE_GAP_SCALE))
    if not math.isfinite(distance):
        distance = OBJECTIVE_GAP_SCALE

    # not red_alive 의 완화 : 적을 얼마나 깎았나
    destroy = 1.0 - min(1.0, red_hp / (red_initial * MAX_HP))
    # objective_reached 의 완화 : 반경 안이면 정확히 1.0
    reach = 1.0 - min(1.0, max(0.0, distance - OBJECTIVE_RADIUS) / OBJECTIVE_GAP_SCALE)

    mission = mission_type_from_config(config)
    if int(entry.get("blue_alive", 0)) <= 0:
        progress = 0.0                      # 전멸은 원 조건에서도 무조건 실패
    elif mission == MISSION_DESTROY_ALL:
        progress = destroy
    elif mission in (MISSION_REACH_OBJECTIVE, MISSION_HOLD_OBJECTIVE):
        progress = reach
    else:                                    # DESTROY_AND_REACH : AND 조건 -> min
        progress = min(destroy, reach)

    return {
        "progress": progress,
        "outcome": float(OUTCOME_BY_NAME.get(str(entry.get("outcome", "TIMEOUT")), OUTCOME_TIMEOUT)),
        "blue_hp_ratio": min(1.0, blue_hp / (blue_initial * MAX_HP)),
        "red_hp_ratio": min(1.0, red_hp / (red_initial * MAX_HP)),
        "objective_gap": min(1.0, distance / OBJECTIVE_GAP_SCALE),
    }


def build_dataset(
    roots: list[Path], *, tick_stride: int, max_episodes: int | None
) -> list[dict]:
    """에피소드마다 여러 시점의 상태와 최종 라벨을 짝지어 모은다."""
    samples: list[dict] = []
    for root in roots:
        summary_path = root / "episode_summary.jsonl"
        if not summary_path.exists():
            print(f"  건너뜀 (episode_summary.jsonl 없음): {root}")
            continue
        entries = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_episodes:
            entries = entries[:max_episodes]
        for entry in entries:
            run_dir = Path(entry["run_dir"])
            if not (run_dir / "soldier_log.csv").exists():
                continue
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            real_map = config.get("real_map", {})
            if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
                set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))
            labels = _labels_from_summary(entry, config)
            mission = mission_type_from_config(config)
            objective = objective_from_config(config)
            duration = float(config.get("duration", 60.0))

            by_time = _episode_rows(run_dir)
            times = sorted(by_time)[::max(1, tick_stride)]
            for time_sec in times:
                rows = by_time[time_sec]
                if not any(float(r["hp"]) > 0 for r in rows):
                    continue
                samples.append(
                    {
                        "rows": rows,
                        "obstacles": config["obstacles"],
                        "time_sec": time_sec,
                        "duration": duration,
                        "objective": objective,
                        "mission": mission,
                        "episode_key": str(run_dir),
                        **labels,
                    }
                )
    return samples


def _to_batch(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """샘플 하나를 value head 입력으로 만든다."""
    batch = build_slot_batch(
        unit_rows=[{k: str(r[k]) for k in ("id", "x", "y", "heading", "hp", "ammo")} for r in sample["rows"]],
        obstacles=sample["obstacles"],
        time_sec=sample["time_sec"],
        duration_sec=sample["duration"],
        objective=sample["objective"],
        mission_type=sample["mission"],
    )
    unit = torch.as_tensor(batch.type_ids == int(ObjectType.UNIT), device=device)
    team = torch.as_tensor(batch.team_ids, device=device)
    alive = torch.as_tensor(batch.features[:, 1] > 0.0, device=device)
    return {
        "features": torch.as_tensor(batch.features, dtype=torch.float32, device=device).unsqueeze(0),
        "feature_mask": torch.as_tensor(batch.feature_mask, device=device).unsqueeze(0),
        "type_ids": torch.as_tensor(batch.type_ids, dtype=torch.long, device=device).unsqueeze(0),
        "team_ids": team.long().unsqueeze(0),
        "alive_mask": (alive & unit).unsqueeze(0),
        "blue_mask": (unit & (team == int(TeamId.BLUE))).unsqueeze(0),
        "red_mask": (unit & (team == int(TeamId.RED))).unsqueeze(0),
        "mission_onehot": torch.eye(4, device=device)[sample["mission"]].unsqueeze(0),
    }


def _run_epoch(
    model: ValueHead,
    samples: list[dict],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
) -> dict[str, float]:
    """한 epoch. optimizer가 None이면 평가만 한다."""
    model.train(optimizer is not None)
    cross_entropy = nn.CrossEntropyLoss()
    order = np.random.permutation(len(samples)) if optimizer else np.arange(len(samples))
    totals = {"loss": 0.0, "outcome": 0.0, "scalar": 0.0, "correct": 0.0,
              "progress": 0.0, "prog_abs": 0.0, "n": 0.0}

    for start in range(0, len(order), batch_size):
        chunk = [samples[int(i)] for i in order[start : start + batch_size]]
        # 슬롯 수가 샘플마다 달라 배치로 못 묶는다. 누적 후 한 번에 갱신한다.
        losses = []
        with torch.set_grad_enabled(optimizer is not None):
            for sample in chunk:
                inputs = _to_batch(sample, device)
                prediction = model(**inputs)
                target = torch.tensor([int(sample["outcome"])], device=device)
                outcome_loss = cross_entropy(prediction["outcome_logits"], target)
                # 주 손실은 달성도 회귀. 나머지는 표현을 잡아주는 보조라 가중치를 낮춘다.
                progress_loss = (
                    prediction["progress"]
                    - torch.tensor([sample["progress"]], dtype=torch.float32, device=device)
                ).pow(2).mean()
                scalar_loss = sum(
                    (prediction[key] - torch.tensor([sample[key]], dtype=torch.float32, device=device)).pow(2).mean()
                    for key in ("blue_hp_ratio", "red_hp_ratio", "objective_gap")
                )
                losses.append(
                    progress_loss + AUX_WEIGHT * outcome_loss + AUX_WEIGHT * scalar_loss
                )
                totals["progress"] += float(progress_loss)
                totals["prog_abs"] += float((prediction["progress"] - torch.tensor([sample["progress"]], device=device)).abs().mean())
                totals["outcome"] += float(outcome_loss)
                totals["scalar"] += float(scalar_loss)
                totals["correct"] += float(int(prediction["outcome_logits"].argmax()) == int(sample["outcome"]))
                totals["n"] += 1.0
            loss = torch.stack(losses).mean()
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals["loss"] += float(loss) * len(chunk)

    n = max(totals["n"], 1.0)
    return {
        "loss": totals["loss"] / n,
        "outcome": totals["outcome"] / n,
        "scalar": totals["scalar"] / n,
        "accuracy": totals["correct"] / n,
        "progress_mse": totals["progress"] / n,
        "progress_mae": totals["prog_abs"] / n,
    }


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="value head 학습")
    parser.add_argument("roots", type=Path, nargs="+", help="episode_summary.jsonl이 있는 출력 폴더들")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/value_head.pt"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tick-stride", type=int, default=6, help="에피소드 안에서 몇 초마다 표본을 뽑을지")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT,
                        help="보조 손실(outcome/hp/gap) 가중치. 0이면 달성도만 학습")
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="주면 학습하지 않고 이 checkpoint를 roots 전체에서 평가만 한다. "
        "학습에 안 쓴 맵으로 홀드아웃 판정할 때 쓴다",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    global AUX_WEIGHT
    AUX_WEIGHT = float(args.aux_weight)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"데이터 수집: {[str(r) for r in args.roots]}")
    samples = build_dataset(args.roots, tick_stride=args.tick_stride, max_episodes=args.max_episodes)
    if not samples:
        raise ValueError("표본이 없다")

    # 에피소드 단위 분할. 같은 에피소드의 인접 시점은 라벨이 같아 시점 단위로 나누면 샌다.
    episodes = sorted({s["episode_key"] for s in samples})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    cut = max(1, int(len(episodes) * (1.0 - args.val_ratio)))
    train_keys = set(episodes[:cut])
    train = [s for s in samples if s["episode_key"] in train_keys]
    validation = [s for s in samples if s["episode_key"] not in train_keys]

    counts = np.bincount([int(s["outcome"]) for s in samples], minlength=3)
    print(f"  표본 {len(samples)}개 / 에피소드 {len(episodes)}개")
    print(f"  학습 {len(train)} / 검증 {len(validation)}  (에피소드 {cut}/{len(episodes) - cut})")
    print(f"  결과 분포 " + "  ".join(f"{OUTCOME_NAMES[i]} {counts[i]}" for i in range(3)))
    print(f"  다수결 기준선 정확도 {counts.max() / counts.sum():.3f}")

    if args.eval_checkpoint is not None:
        model = load_value_head(args.eval_checkpoint, device)
        metrics = _run_epoch(model, samples, device, None, args.batch_size)
        # 기준선은 "이 데이터의 평균을 늘 답하기". 모델이 이걸 못 이기면 의미 없다.
        values = np.array([s["progress"] for s in samples])
        baseline = float(np.abs(values - values.mean()).mean())
        print(f"\n=== 평가 전용 : {args.eval_checkpoint.name} ===")
        print(f"  표본 {len(samples)}개 / 에피소드 {len(episodes)}개")
        print(f"  달성도 평균 {values.mean():.4f}  표준편차 {values.std():.4f}")
        print(f"  MAE {metrics['progress_mae']:.4f}   MSE {metrics['progress_mse']:.4f}")
        print(f"  기준선 MAE {baseline:.4f}  ->  {100 * (baseline - metrics['progress_mae']) / baseline:+.1f}%")
        return 0

    model = ValueHead(ValueHeadConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    print(f"  파라미터 {sum(p.numel() for p in model.parameters()):,}")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train, device, optimizer, args.batch_size)
        val_metrics = _run_epoch(model, validation, device, None, args.batch_size)
        mark = ""
        if val_metrics["progress_mae"] < best:
            best = val_metrics["progress_mae"]
            save_value_head(args.out, model)
            mark = " *best"
        print(
            f"epoch {epoch:>3}  train MAE={train_metrics['progress_mae']:.4f}"
            f"  |  val MAE={val_metrics['progress_mae']:.4f}"
            f" MSE={val_metrics['progress_mse']:.4f} acc={val_metrics['accuracy']:.3f}{mark}"
        )

    baseline = float(np.mean([abs(s["progress"] - np.mean([t["progress"] for t in train])) for s in validation]))
    print(f"\ncheckpoint={args.out}  best_val_progress_MAE={best:.4f}")
    print(f"평균값 예측 기준선 MAE={baseline:.4f}  ->  {'개선' if best < baseline else '학습 실패'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
