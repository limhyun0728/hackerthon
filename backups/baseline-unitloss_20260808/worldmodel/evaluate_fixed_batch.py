"""여러 checkpoint를 같은 DEVS seed batch에서 재평가한다.

이 스크립트는 학습을 하지 않는다. 비교하려는 checkpoint들을 같은 terrain,
duration, seed 목록, CEM 설정으로 각각 다시 실행하고 summary.csv를 만든다.
모델 자체의 action 선택 능력을 비교하려면 rollout backend는 jepa를 써야 한다.
devs backend는 모델을 최종 행동 선택에 쓰지 않으므로 CEM/evaluator baseline 비교용이다.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR.parent
for path in (str(PROJECT_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hackerthon.terrain import DEFAULT_OBSTACLES
from hackerthon.worldmodel.cem_planner import CEMConfig
from hackerthon.worldmodel.episodic_cem_training_loop import run_episode
from hackerthon.worldmodel.evaluator import evaluate_v2_run_segment
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig


@dataclass(frozen=True)
class CheckpointSpec:
    """평가할 checkpoint 하나."""

    label: str
    path: Path


def _parse_checkpoint(value: str) -> CheckpointSpec:
    """LABEL=PATH 형식의 checkpoint 인자를 읽는다."""
    if "=" not in value:
        raise ValueError("--checkpoint는 LABEL=PATH 형식이어야 한다")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("checkpoint label은 비어 있을 수 없다")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"checkpoint label은 파일명에 안전한 문자만 써야 한다: {label}")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint가 없다: {path}")
    return CheckpointSpec(label=label, path=path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    """쉼표와 범위를 섞은 seed 문자열을 정수 tuple로 바꾼다."""
    seeds: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"seed range의 끝이 시작보다 작다: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("seed를 하나 이상 지정해야 한다")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed 목록에 중복이 있다")
    return tuple(seeds)


def _load_checkpoint_model(
    spec: CheckpointSpec,
    *,
    device: torch.device,
) -> tuple[DEVSObjectCentricWorldModel, ObjectSlotModelConfig]:
    """checkpoint에서 모델과 config를 복원한다."""
    payload = torch.load(spec.path, map_location=device)
    config_dict = dict(payload["model_config"])
    if "maskable_type_ids" in config_dict:
        config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
    config = ObjectSlotModelConfig(**config_dict)
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def _latest_unit_rows(run_dir: Path) -> tuple[dict[str, str], ...]:
    """soldier_log.csv에서 각 유닛의 마지막 row를 읽는다."""
    latest: dict[int, tuple[float, dict[str, str]]] = {}
    with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unit_id = int(row["id"])
            time_sec = float(row["time"])
            current = latest.get(unit_id)
            if current is None or time_sec >= current[0]:
                latest[unit_id] = (time_sec, row)
    if not latest:
        raise ValueError(f"{run_dir}/soldier_log.csv에 row가 없다")
    return tuple(row for _, row in sorted(latest.values(), key=lambda item: int(item[1]["id"])))


def _objective_distance(rows: Sequence[dict[str, str]], objective: tuple[float, float]) -> float:
    """가장 가까운 BLUE와 objective 사이의 거리를 계산한다."""
    distances = [
        math.hypot(float(row["x"]) - objective[0], float(row["y"]) - objective[1])
        for row in rows
        if int(row["id"]) < 200
    ]
    if not distances:
        raise ValueError("BLUE row가 없다")
    return float(min(distances))


def _combat_outcome(rows: Sequence[dict[str, str]], objective: tuple[float, float]) -> str:
    """전투 기준 승패와 mission 기준 승리를 분리해 판정한다."""
    blue_rows = [row for row in rows if int(row["id"]) < 200]
    red_rows = [row for row in rows if int(row["id"]) >= 200]
    if not blue_rows or not red_rows:
        raise ValueError("BLUE/RED row가 모두 필요하다")
    blue_alive = any(float(row["hp"]) > 0.0 for row in blue_rows)
    red_alive = any(float(row["hp"]) > 0.0 for row in red_rows)
    objective_reached = any(
        float(row["hp"]) > 0.0
        and math.hypot(float(row["x"]) - objective[0], float(row["y"]) - objective[1]) <= 1.0
        for row in blue_rows
    )
    if not red_alive and objective_reached:
        return "MISSION_WIN"
    if not red_alive:
        return "COMBAT_WIN"
    if not blue_alive:
        return "LOSE"
    return "UNRESOLVED"


def _summary_row(
    *,
    label: str,
    checkpoint: Path,
    seed: int,
    episode_index: int,
    run_dir: Path,
    mission_outcome: str,
    objective: tuple[float, float],
) -> dict[str, object]:
    """한 평가 episode의 summary row를 만든다."""
    rows = _latest_unit_rows(run_dir)
    blue_rows = [row for row in rows if int(row["id"]) < 200]
    red_rows = [row for row in rows if int(row["id"]) >= 200]
    final_time = max(float(row["time"]) for row in rows)
    score = evaluate_v2_run_segment(run_dir, start_time=0.0, end_time=final_time).score
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "episode_index": episode_index,
        "mission_outcome": mission_outcome,
        "combat_outcome": _combat_outcome(rows, objective),
        "score": round(float(score), 6),
        "final_time": round(final_time, 3),
        "blue_alive": sum(float(row["hp"]) > 0.0 for row in blue_rows),
        "red_alive": sum(float(row["hp"]) > 0.0 for row in red_rows),
        "blue_hp": round(sum(float(row["hp"]) for row in blue_rows), 3),
        "red_hp": round(sum(float(row["hp"]) for row in red_rows), 3),
        "objective_distance": round(_objective_distance(rows, objective), 6),
        "run_dir": str(run_dir),
    }


def _write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """summary.csv를 저장한다."""
    if not rows:
        raise ValueError("저장할 summary row가 없다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_label_summary(rows: Sequence[dict[str, object]]) -> None:
    """label별 요약을 터미널에 출력한다."""
    labels = sorted({str(row["label"]) for row in rows})
    for label in labels:
        selected = [row for row in rows if row["label"] == label]
        combat_wins = sum(row["combat_outcome"] in ("COMBAT_WIN", "MISSION_WIN") for row in selected)
        mission_wins = sum(row["mission_outcome"] == "WIN" for row in selected)
        losses = sum(row["combat_outcome"] == "LOSE" for row in selected)
        avg_score = sum(float(row["score"]) for row in selected) / float(len(selected))
        avg_blue_hp = sum(float(row["blue_hp"]) for row in selected) / float(len(selected))
        avg_red_hp = sum(float(row["red_hp"]) for row in selected) / float(len(selected))
        print(
            f"{label}: n={len(selected)} combat_win={combat_wins} mission_win={mission_wins} "
            f"lose={losses} avg_score={avg_score:.2f} avg_blue_hp={avg_blue_hp:.1f} avg_red_hp={avg_red_hp:.1f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    """고정 seed batch 평가 entrypoint."""
    parser = argparse.ArgumentParser(description="checkpoint들을 같은 DEVS seed batch에서 재평가")
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, required=True, help="예: 42-49 또는 42,43,44")
    parser.add_argument("--output-root", type=Path, default=Path("output/fixed_batch_eval"))
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--terrain", choices=("open", "urban"), default="open")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--rollout-backend", choices=("jepa", "devs"), default="jepa")
    parser.add_argument("--cem-candidates", type=int, default=64)
    parser.add_argument("--cem-elites", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--cem-horizon", type=int, default=None)
    parser.add_argument("--cem-seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.duration <= 0.0:
        raise ValueError("duration은 0보다 커야 한다")
    if args.cem_candidates <= 0 or args.cem_elites <= 0 or args.cem_iterations <= 0:
        raise ValueError("CEM 수치 설정은 0보다 커야 한다")
    if args.cem_elites > args.cem_candidates:
        raise ValueError("cem_elites는 cem_candidates보다 클 수 없다")

    device = torch.device(args.device)
    obstacles = () if args.terrain == "open" else DEFAULT_OBSTACLES
    objective = (10.0, 0.0)
    summary_rows: list[dict[str, object]] = []

    for spec in args.checkpoint:
        model, model_config = _load_checkpoint_model(spec, device=device)
        horizon = model_config.pred_frames if args.cem_horizon is None else int(args.cem_horizon)
        if args.rollout_backend == "jepa" and horizon != model_config.pred_frames:
            raise ValueError(f"{spec.label}: jepa backend에서는 cem_horizon이 pred_frames와 같아야 한다")
        cem_config = CEMConfig(
            num_candidates=args.cem_candidates,
            num_elites=args.cem_elites,
            num_iterations=args.cem_iterations,
            future_horizon=horizon,
            seed=args.cem_seed,
            min_action_probability=0.0,
        )
        label_root = args.output_root / spec.label
        print(f"label={spec.label} checkpoint={spec.path} pred_frames={model_config.pred_frames}")
        for episode_index, seed in enumerate(args.seeds, start=1):
            result = run_episode(
                episode_index=episode_index,
                output_root=label_root,
                seed=seed,
                duration_sec=args.duration,
                model=model,
                model_config=model_config,
                cem_config=cem_config,
                device=device,
                obstacles=obstacles,
                rollout_backend=args.rollout_backend,
            )
            row = _summary_row(
                label=spec.label,
                checkpoint=spec.path,
                seed=seed,
                episode_index=episode_index,
                run_dir=result.run_dir,
                mission_outcome=result.outcome,
                objective=objective,
            )
            summary_rows.append(row)
            print(
                f"  seed={seed} combat={row['combat_outcome']} mission={row['mission_outcome']} "
                f"score={float(row['score']):.2f} BHP={float(row['blue_hp']):.0f} RHP={float(row['red_hp']):.0f}"
            )

    summary_path = args.summary_path or (args.output_root / "summary.csv")
    _write_summary(summary_path, summary_rows)
    print()
    _print_label_summary(summary_rows)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
