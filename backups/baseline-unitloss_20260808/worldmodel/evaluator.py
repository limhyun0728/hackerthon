"""DEVS state feature를 CEM 점수로 평가한다.

이 모듈은 학습 모델이 아니다. 월드모델이 예측한 객체 feature sequence를
DEVS 의미 공간에서 점수화해서 CEM이 더 좋은 action sequence를 고르게 한다.
현재 버전은 노출/불필요 교전 항목을 제외하고, 피해·목표 접근·빠른 성공·탄약
보존만 계산한다.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from hackerthon.worldmodel.slots import (
    MAX_FEATURE_DIM,
    OBJECTIVE_RADIUS,
    ObjectType,
    TeamId,
    build_slot_batch_from_v2_run,
)


UNIT_HP_INDEX = 1
UNIT_AMMO_INDEX = 2
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4
UNIT_ALIVE_INDEX = 7
MISSION_OBJECTIVE_X_INDEX = 1
MISSION_OBJECTIVE_Y_INDEX = 2
VALID_RATIO_MIN = 0.0
VALID_RATIO_MAX = 1.0
NO_ALIVE_BLUE_DISTANCE = math.hypot(WORLD_X_MAX - WORLD_X_MIN, WORLD_Y_MAX - WORLD_Y_MIN)


BLUE_MAX_STEP_PER_SEC = 1.5


@dataclass(frozen=True)
class EvaluatorWeights:
    """CEM 후보 점수의 가중치.

    피해 항은 HP-ratio 합(유닛수로 나누지 않음)이고, 전진 항은 horizon 내
    최대 이동량으로 정규화한 [-1, 1] 값이다. 적이 살아 있는 동안에는
    objective_progress, 적 전멸 후에는 objective_progress_cleared가 전진
    가중치로 쓰인다. 임무가 "적 전멸 후 objective 도달" 순서이므로 채점도
    같은 phase 구조를 갖는다.
    """

    enemy_damage: float = 6.0
    enemy_hp_mass_reduction: float = 4.0
    enemy_kia: float = 8.0
    enemy_finish: float = 4.0
    blue_damage: float = 4.0
    objective_progress: float = 1.0
    objective_progress_cleared: float = 5.0
    early_success: float = 5.0
    ammo_used: float = 0.2


@dataclass(frozen=True)
class EvaluationResult:
    """한 action sequence 후보에 대한 평가 결과."""

    score: float
    enemy_damage: float
    enemy_hp_mass_reduction: float
    enemy_kia: float
    enemy_finish: float
    blue_damage: float
    objective_progress: float
    early_success: float
    ammo_used: float
    success_frame: int
    final_objective_distance: float
    red_cleared: bool


def _expect_rank(name: str, value: np.ndarray, rank: int) -> None:
    """numpy 배열 rank 계약을 즉시 검증한다."""
    if value.ndim != rank:
        raise ValueError(f"{name} rank는 {rank}이어야 한다: shape={value.shape}")


def _expect_finite(name: str, value: np.ndarray) -> None:
    """NaN/Inf가 점수에 섞이지 않게 한다."""
    if not np.isfinite(value).all():
        raise ValueError(f"{name}에는 NaN 또는 Inf가 있으면 안 된다")


def _denorm_x(x_norm: np.ndarray | float) -> np.ndarray | float:
    """[-1, 1] x좌표를 DEVS 월드 x좌표로 되돌린다."""
    return (np.asarray(x_norm) + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def _denorm_y(y_norm: np.ndarray | float) -> np.ndarray | float:
    """[-1, 1] y좌표를 DEVS 월드 y좌표로 되돌린다."""
    return (np.asarray(y_norm) + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def _unit_indices(type_ids: np.ndarray) -> np.ndarray:
    """unit slot index를 반환한다."""
    return np.flatnonzero(type_ids == int(ObjectType.UNIT))


def _team_unit_indices(type_ids: np.ndarray, team_ids: np.ndarray, team_id: TeamId) -> np.ndarray:
    """특정 팀의 unit slot index를 반환한다."""
    return np.flatnonzero((type_ids == int(ObjectType.UNIT)) & (team_ids == int(team_id)))


def _mission_index(type_ids: np.ndarray) -> int:
    """mission slot이 정확히 하나인지 확인하고 index를 반환한다."""
    indices = np.flatnonzero(type_ids == int(ObjectType.MISSION))
    if indices.shape != (1,):
        raise ValueError(f"mission slot은 정확히 하나여야 한다: count={indices.size}")
    return int(indices[0])


def _semantic_ratio(values: np.ndarray) -> np.ndarray:
    """예측 feature를 DEVS ratio 의미 범위 [0, 1]로 투영한다."""
    return np.clip(values.astype(np.float32), VALID_RATIO_MIN, VALID_RATIO_MAX)


def _team_hp_sum(features: np.ndarray, unit_indices: np.ndarray) -> float:
    """팀 HP ratio 합을 계산한다."""
    return float(_semantic_ratio(features[unit_indices, UNIT_HP_INDEX]).sum())


def _team_ammo_sum(features: np.ndarray, unit_indices: np.ndarray) -> float:
    """팀 탄약 ratio 합을 계산한다."""
    return float(_semantic_ratio(features[unit_indices, UNIT_AMMO_INDEX]).sum())


def _objective_xy(features: np.ndarray, type_ids: np.ndarray) -> tuple[float, float]:
    """mission slot에서 objective 월드 좌표를 읽는다."""
    mission_index = _mission_index(type_ids)
    x = float(_denorm_x(features[mission_index, MISSION_OBJECTIVE_X_INDEX]))
    y = float(_denorm_y(features[mission_index, MISSION_OBJECTIVE_Y_INDEX]))
    return x, y


def _blue_positions(features: np.ndarray, blue_indices: np.ndarray) -> np.ndarray:
    """살아 있는 아군의 월드 좌표를 반환한다."""
    alive = _semantic_ratio(features[blue_indices, UNIT_ALIVE_INDEX]) >= 0.5
    alive_indices = blue_indices[alive]
    if alive_indices.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = _denorm_x(features[alive_indices, UNIT_X_INDEX])
    y = _denorm_y(features[alive_indices, UNIT_Y_INDEX])
    return np.stack([x, y], axis=-1).astype(np.float32)


def _objective_distance(features: np.ndarray, type_ids: np.ndarray, blue_indices: np.ndarray) -> float:
    """가장 가까운 생존 아군과 objective 사이의 거리를 계산한다."""
    objective = np.asarray(_objective_xy(features, type_ids), dtype=np.float32)
    positions = _blue_positions(features, blue_indices)
    if positions.shape[0] == 0:
        return float(NO_ALIVE_BLUE_DISTANCE)
    distances = np.linalg.norm(positions - objective.reshape(1, 2), axis=-1)
    return float(distances.min())


def _objective_reached(features: np.ndarray, type_ids: np.ndarray, blue_indices: np.ndarray) -> bool:
    """생존 아군 중 하나가 objective radius 안에 들어왔는지 확인한다."""
    return _objective_distance(features, type_ids, blue_indices) <= OBJECTIVE_RADIUS


def _red_eliminated(features: np.ndarray, red_indices: np.ndarray) -> bool:
    """적군이 모두 무력화됐는지 확인한다."""
    hp = _semantic_ratio(features[red_indices, UNIT_HP_INDEX])
    alive = _semantic_ratio(features[red_indices, UNIT_ALIVE_INDEX])
    return bool(np.all((hp <= 0.01) | (alive < 0.5)))


def _success(features: np.ndarray, type_ids: np.ndarray, blue_indices: np.ndarray, red_indices: np.ndarray) -> bool:
    """현재 MVP 임무 성공 조건을 계산한다."""
    return _red_eliminated(features, red_indices) and _objective_reached(features, type_ids, blue_indices)


def _positive_delta(start_value: float, end_value: float) -> float:
    """좋아진 방향의 변화량만 점수에 반영한다."""
    return max(0.0, float(start_value) - float(end_value))


def evaluate_future_sequence(
    *,
    start_features: np.ndarray,
    future_features: np.ndarray,
    type_ids: np.ndarray,
    team_ids: np.ndarray,
    weights: EvaluatorWeights = EvaluatorWeights(),
) -> EvaluationResult:
    """현재 state와 예측 future state sequence로 action 후보 점수를 계산한다."""
    _expect_rank("start_features", start_features, 2)
    _expect_rank("future_features", future_features, 3)
    _expect_rank("type_ids", type_ids, 1)
    _expect_rank("team_ids", team_ids, 1)
    _expect_finite("start_features", start_features)
    _expect_finite("future_features", future_features)
    if start_features.shape[1] != MAX_FEATURE_DIM:
        raise ValueError(f"start_features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")
    if future_features.shape[2] != MAX_FEATURE_DIM:
        raise ValueError(f"future_features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")
    if start_features.shape[0] != future_features.shape[1]:
        raise ValueError("start_features와 future_features의 slot 개수가 같아야 한다")
    if type_ids.shape != (start_features.shape[0],):
        raise ValueError("type_ids shape는 slot 개수와 같아야 한다")
    if team_ids.shape != (start_features.shape[0],):
        raise ValueError("team_ids shape는 slot 개수와 같아야 한다")
    if future_features.shape[0] <= 0:
        raise ValueError("future_features에는 최소 한 개 이상의 미래 frame이 있어야 한다")

    unit_indices = _unit_indices(type_ids)
    if unit_indices.size == 0:
        raise ValueError("unit slot이 하나도 없다")
    blue_indices = _team_unit_indices(type_ids, team_ids, TeamId.BLUE)
    red_indices = _team_unit_indices(type_ids, team_ids, TeamId.RED)
    if blue_indices.size == 0:
        raise ValueError("BLUE unit slot이 하나도 없다")
    if red_indices.size == 0:
        raise ValueError("RED unit slot이 하나도 없다")

    final_features = future_features[-1]
    start_enemy_hp = _team_hp_sum(start_features, red_indices)
    final_enemy_hp = _team_hp_sum(final_features, red_indices)
    start_blue_hp = _team_hp_sum(start_features, blue_indices)
    final_blue_hp = _team_hp_sum(final_features, blue_indices)
    start_blue_ammo = _team_ammo_sum(start_features, blue_indices)
    final_blue_ammo = _team_ammo_sum(final_features, blue_indices)

    enemy_damage = _positive_delta(start_enemy_hp, final_enemy_hp)
    blue_damage = _positive_delta(start_blue_hp, final_blue_hp)
    ammo_used = _positive_delta(start_blue_ammo, final_blue_ammo) / float(blue_indices.size)

    start_red_hp_each = _semantic_ratio(start_features[red_indices, UNIT_HP_INDEX])
    start_red_alive_each = _semantic_ratio(start_features[red_indices, UNIT_ALIVE_INDEX]) >= 0.5
    final_red_hp_each = _semantic_ratio(final_features[red_indices, UNIT_HP_INDEX])
    enemy_kia = float(np.sum((start_red_hp_each > 0.01) & (final_red_hp_each <= 0.01)))
    future_red_hp_each = _semantic_ratio(future_features[:, red_indices, UNIT_HP_INDEX])
    future_red_alive_each = _semantic_ratio(future_features[:, red_indices, UNIT_ALIVE_INDEX]) >= 0.5
    # 특정 표적을 명시하지 않고, 살아 있는 RED 전투력의 HP mass가 horizon 안에서
    # 빨리 낮아질수록 보상한다. KIA의 0/1 희소 신호를 보완하는 누적 신호다.
    start_red_hp_mass = float((start_red_hp_each * start_red_alive_each.astype(np.float32)).sum())
    future_red_hp_mass = (future_red_hp_each * future_red_alive_each.astype(np.float32)).sum(axis=-1)
    enemy_hp_mass_reduction = float(np.maximum(start_red_hp_mass - future_red_hp_mass, 0.0).mean())
    # 총 피해량만 보상하면 여러 적을 고르게 깎아도 점수가 높다. HP가 낮은 적을
    # 더 낮게 밀수록 추가 보상을 줘 CEM이 마무리 사격을 선호하게 만든다.
    start_finish_pressure = np.square(1.0 - start_red_hp_each)
    final_finish_pressure = np.square(1.0 - final_red_hp_each)
    enemy_finish = float(np.maximum(final_finish_pressure - start_finish_pressure, 0.0).sum())
    red_cleared = bool(np.all(final_red_hp_each <= 0.01))

    start_distance = _objective_distance(start_features, type_ids, blue_indices)
    final_distance = _objective_distance(final_features, type_ids, blue_indices)
    max_progress = float(future_features.shape[0]) * BLUE_MAX_STEP_PER_SEC
    objective_progress = (start_distance - final_distance) / max_progress
    progress_weight = weights.objective_progress_cleared if red_cleared else weights.objective_progress

    success_frame = -1
    for frame_index, frame_features in enumerate(future_features):
        if _success(frame_features, type_ids, blue_indices, red_indices):
            success_frame = frame_index
            break
    if success_frame >= 0:
        early_success = float(future_features.shape[0] - success_frame) / float(future_features.shape[0])
    else:
        early_success = 0.0

    score = (
        weights.enemy_damage * enemy_damage
        + weights.enemy_hp_mass_reduction * enemy_hp_mass_reduction
        + weights.enemy_kia * enemy_kia
        + weights.enemy_finish * enemy_finish
        - weights.blue_damage * blue_damage
        + progress_weight * objective_progress
        + weights.early_success * early_success
        - weights.ammo_used * ammo_used
    )
    return EvaluationResult(
        score=float(score),
        enemy_damage=float(enemy_damage),
        enemy_hp_mass_reduction=float(enemy_hp_mass_reduction),
        enemy_kia=float(enemy_kia),
        enemy_finish=float(enemy_finish),
        blue_damage=float(blue_damage),
        objective_progress=float(objective_progress),
        early_success=float(early_success),
        ammo_used=float(ammo_used),
        success_frame=int(success_frame),
        final_objective_distance=float(final_distance),
        red_cleared=red_cleared,
    )


def evaluate_future_batch(
    *,
    start_features: np.ndarray,
    future_features: np.ndarray,
    type_ids: np.ndarray,
    team_ids: np.ndarray,
    weights: EvaluatorWeights = EvaluatorWeights(),
) -> tuple[np.ndarray, tuple[EvaluationResult, ...]]:
    """여러 CEM 후보 future sequence를 한 번에 평가한다."""
    _expect_rank("start_features", start_features, 2)
    _expect_rank("future_features", future_features, 4)
    if future_features.shape[0] <= 0:
        raise ValueError("future_features batch가 비어 있다")
    results = tuple(
        evaluate_future_sequence(
            start_features=start_features,
            future_features=future_features[index],
            type_ids=type_ids,
            team_ids=team_ids,
            weights=weights,
        )
        for index in range(future_features.shape[0])
    )
    scores = np.asarray([result.score for result in results], dtype=np.float32)
    return scores, results


def evaluate_v2_run_segment(run_dir: Path, *, start_time: float, end_time: float) -> EvaluationResult:
    """v2 로그의 실제 state 구간을 evaluator로 점수화한다."""
    if end_time <= start_time:
        raise ValueError("end_time은 start_time보다 커야 한다")
    start_batch = build_slot_batch_from_v2_run(run_dir, start_time)
    future_batches = [
        build_slot_batch_from_v2_run(run_dir, float(time_sec))
        for time_sec in np.arange(start_time + 1.0, end_time + 1.0, 1.0)
    ]
    if not future_batches:
        raise ValueError("평가할 미래 frame이 없다")
    for batch in future_batches:
        if not np.array_equal(start_batch.type_ids, batch.type_ids):
            raise ValueError("평가 구간에서 type_ids가 변하면 안 된다")
        if not np.array_equal(start_batch.team_ids, batch.team_ids):
            raise ValueError("평가 구간에서 team_ids가 변하면 안 된다")
    future_features = np.stack([batch.features for batch in future_batches], axis=0)
    return evaluate_future_sequence(
        start_features=start_batch.features,
        future_features=future_features,
        type_ids=start_batch.type_ids,
        team_ids=start_batch.team_ids,
    )


def _format_result(result: EvaluationResult) -> str:
    """CLI 출력용 결과 문자열을 만든다."""
    return (
        f"score={result.score:.6f}\n"
        f"enemy_damage={result.enemy_damage:.6f}\n"
        f"enemy_hp_mass_reduction={result.enemy_hp_mass_reduction:.6f}\n"
        f"enemy_kia={result.enemy_kia:.0f}\n"
        f"enemy_finish={result.enemy_finish:.6f}\n"
        f"red_cleared={result.red_cleared}\n"
        f"blue_damage={result.blue_damage:.6f}\n"
        f"objective_progress={result.objective_progress:.6f}\n"
        f"early_success={result.early_success:.6f}\n"
        f"ammo_used={result.ammo_used:.6f}\n"
        f"success_frame={result.success_frame}\n"
        f"final_objective_distance={result.final_objective_distance:.6f}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    """v2 run 구간을 evaluator로 점수화한다."""
    parser = argparse.ArgumentParser(description="DEVS object feature sequence evaluator")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = evaluate_v2_run_segment(
        args.run_dir,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(_format_result(result))


if __name__ == "__main__":
    main()
