"""Torch 기반 CEM으로 DEVS joint action sequence를 탐색한다.

CEM은 후보 action sequence를 샘플링하고, 월드모델 rollout 결과를 evaluator로
점수화한 뒤 elite 후보로 분포를 갱신한다. 여기서는 DEVS action vocabulary
`STOP`, `MOVE`, `ENGAGE`, `TURN`만 사용한다.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.terrain import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from hackerthon.worldmodel.actions import ACTION_DIM, ActionType, NO_COMMAND_TYPE_ID, NO_TARGET_ENTITY_ID
from hackerthon.worldmodel.evaluator import BLUE_MAX_STEP_PER_SEC, EvaluatorWeights
from hackerthon.worldmodel.slots import (
    MAX_FEATURE_DIM,
    OBJECTIVE_RADIUS,
    ObjectType,
    SlotBatch,
    TeamId,
    build_slot_batch_from_v2_run,
)


MOVE_X_INDEX = 2
MOVE_Y_INDEX = 3
TARGET_TEAM_INDEX = 5
TARGET_X_INDEX = 6
TARGET_Y_INDEX = 7
THETA_COS_INDEX = 9
THETA_SIN_INDEX = 10
UNIT_HP_INDEX = 1
UNIT_AMMO_INDEX = 2
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4
UNIT_ALIVE_INDEX = 7
MISSION_OBJECTIVE_X_INDEX = 1
MISSION_OBJECTIVE_Y_INDEX = 2
ACTION_TYPE_COUNT = len(ActionType)
DEFAULT_ACTION_PRIOR = (0.20, 0.50, 0.25, 0.05)
NO_ALIVE_BLUE_DISTANCE = math.hypot(WORLD_X_MAX - WORLD_X_MIN, WORLD_Y_MAX - WORLD_Y_MIN)


@dataclass(frozen=True)
class CEMConfig:
    """CEM 탐색 설정."""

    num_candidates: int = 128
    num_elites: int = 16
    num_iterations: int = 4
    future_horizon: int = 1
    seed: int = 42
    smoothing: float = 0.35
    initial_move_std: float = 0.45
    min_move_std: float = 0.05
    initial_turn_std: float = math.pi
    min_turn_std: float = 0.05
    min_action_probability: float = 0.02
    action_prior: tuple[float, float, float, float] = DEFAULT_ACTION_PRIOR

    def __post_init__(self) -> None:
        """CEM 설정값을 즉시 검증한다."""
        if self.num_candidates <= 0:
            raise ValueError("num_candidates는 0보다 커야 한다")
        if self.num_elites <= 0:
            raise ValueError("num_elites는 0보다 커야 한다")
        if self.num_elites > self.num_candidates:
            raise ValueError("num_elites는 num_candidates보다 클 수 없다")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations는 0보다 커야 한다")
        if self.future_horizon <= 0:
            raise ValueError("future_horizon은 0보다 커야 한다")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing은 [0, 1) 범위여야 한다")
        if self.initial_move_std <= 0.0:
            raise ValueError("initial_move_std는 0보다 커야 한다")
        if self.min_move_std <= 0.0:
            raise ValueError("min_move_std는 0보다 커야 한다")
        if self.initial_turn_std <= 0.0:
            raise ValueError("initial_turn_std는 0보다 커야 한다")
        if self.min_turn_std <= 0.0:
            raise ValueError("min_turn_std는 0보다 커야 한다")
        if self.min_action_probability < 0.0:
            raise ValueError("min_action_probability는 음수일 수 없다")
        if len(self.action_prior) != ACTION_TYPE_COUNT:
            raise ValueError(f"action_prior 길이는 {ACTION_TYPE_COUNT}이어야 한다")


@dataclass(frozen=True)
class CEMDistribution:
    """CEM이 갱신하는 torch action 분포."""

    action_probs: torch.Tensor
    move_mean: torch.Tensor
    move_std: torch.Tensor
    turn_mean: torch.Tensor
    turn_std: torch.Tensor
    target_probs: torch.Tensor

    def __post_init__(self) -> None:
        """분포 shape와 확률값을 검증한다."""
        _expect_rank("action_probs", self.action_probs, 3)
        _expect_rank("move_mean", self.move_mean, 3)
        _expect_rank("move_std", self.move_std, 3)
        _expect_rank("turn_mean", self.turn_mean, 2)
        _expect_rank("turn_std", self.turn_std, 2)
        _expect_rank("target_probs", self.target_probs, 3)
        horizon, num_units, action_count = self.action_probs.shape
        if action_count != ACTION_TYPE_COUNT:
            raise ValueError(f"action_probs 마지막 차원은 {ACTION_TYPE_COUNT}이어야 한다")
        if self.move_mean.shape != (horizon, num_units, 2):
            raise ValueError("move_mean shape는 (H, U, 2)이어야 한다")
        if self.move_std.shape != (horizon, num_units, 2):
            raise ValueError("move_std shape는 (H, U, 2)이어야 한다")
        if self.turn_mean.shape != (horizon, num_units):
            raise ValueError("turn_mean shape는 (H, U)이어야 한다")
        if self.turn_std.shape != (horizon, num_units):
            raise ValueError("turn_std shape는 (H, U)이어야 한다")
        if self.target_probs.shape[:2] != (horizon, num_units):
            raise ValueError("target_probs 앞 두 차원은 (H, U)이어야 한다")
        _expect_probability_rows("action_probs", self.action_probs)
        _expect_probability_rows("target_probs", self.target_probs)
        _expect_positive("move_std", self.move_std)
        _expect_positive("turn_std", self.turn_std)


@dataclass(frozen=True)
class FutureActionPlanBatch:
    """CEM이 샘플링한 미래 joint action 후보 묶음."""

    action_features: torch.Tensor
    action_unit_ids: torch.Tensor
    issued_mask: torch.Tensor
    action_type_ids: torch.Tensor
    target_entity_ids: torch.Tensor
    target_indices: torch.Tensor
    move_xy_norm: torch.Tensor
    theta_radians: torch.Tensor
    red_target_ids: torch.Tensor

    def __post_init__(self) -> None:
        """후보 action batch의 shape 계약을 검증한다."""
        _expect_rank("action_features", self.action_features, 4)
        _expect_rank("action_unit_ids", self.action_unit_ids, 3)
        _expect_rank("issued_mask", self.issued_mask, 3)
        _expect_rank("action_type_ids", self.action_type_ids, 3)
        _expect_rank("target_entity_ids", self.target_entity_ids, 3)
        _expect_rank("target_indices", self.target_indices, 3)
        _expect_rank("move_xy_norm", self.move_xy_norm, 4)
        _expect_rank("theta_radians", self.theta_radians, 3)
        _expect_rank("red_target_ids", self.red_target_ids, 1)
        candidates, horizon, num_units, action_dim = self.action_features.shape
        common = (candidates, horizon, num_units)
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_features 마지막 차원은 {ACTION_DIM}이어야 한다")
        if self.action_unit_ids.shape != common:
            raise ValueError("action_unit_ids shape는 (C, H, U)이어야 한다")
        if self.issued_mask.shape != common:
            raise ValueError("issued_mask shape는 (C, H, U)이어야 한다")
        if self.action_type_ids.shape != common:
            raise ValueError("action_type_ids shape는 (C, H, U)이어야 한다")
        if self.target_entity_ids.shape != common:
            raise ValueError("target_entity_ids shape는 (C, H, U)이어야 한다")
        if self.target_indices.shape != common:
            raise ValueError("target_indices shape는 (C, H, U)이어야 한다")
        if self.move_xy_norm.shape != common + (2,):
            raise ValueError("move_xy_norm shape는 (C, H, U, 2)이어야 한다")
        if self.theta_radians.shape != common:
            raise ValueError("theta_radians shape는 (C, H, U)이어야 한다")
        if self.red_target_ids.numel() <= 0:
            raise ValueError("red_target_ids가 비어 있으면 ENGAGE 후보를 만들 수 없다")
        _expect_bool("issued_mask", self.issued_mask)
        _expect_finite("action_features", self.action_features)
        _expect_finite("move_xy_norm", self.move_xy_norm)
        _expect_finite("theta_radians", self.theta_radians)

    def take_candidates(self, indices: torch.Tensor) -> "FutureActionPlanBatch":
        """일부 candidate만 잘라 같은 구조로 반환한다."""
        _expect_rank("indices", indices, 1)
        indices = indices.to(device=self.action_features.device, dtype=torch.long)
        return FutureActionPlanBatch(
            action_features=self.action_features.index_select(0, indices),
            action_unit_ids=self.action_unit_ids.index_select(0, indices),
            issued_mask=self.issued_mask.index_select(0, indices),
            action_type_ids=self.action_type_ids.index_select(0, indices),
            target_entity_ids=self.target_entity_ids.index_select(0, indices),
            target_indices=self.target_indices.index_select(0, indices),
            move_xy_norm=self.move_xy_norm.index_select(0, indices),
            theta_radians=self.theta_radians.index_select(0, indices),
            red_target_ids=self.red_target_ids,
        )


@dataclass(frozen=True)
class ObservedActionWindow:
    """월드모델 history 구간에 해당하는 관측 전이 action."""

    action_features: torch.Tensor
    action_unit_ids: torch.Tensor
    issued_mask: torch.Tensor

    def __post_init__(self) -> None:
        """history action window shape를 검증한다."""
        _expect_rank("action_features", self.action_features, 3)
        _expect_rank("action_unit_ids", self.action_unit_ids, 2)
        _expect_rank("issued_mask", self.issued_mask, 2)
        if self.action_features.shape[:2] != self.action_unit_ids.shape:
            raise ValueError("action_features와 action_unit_ids의 시간/유닛 차원이 같아야 한다")
        if self.issued_mask.shape != self.action_unit_ids.shape:
            raise ValueError("issued_mask shape는 action_unit_ids shape와 같아야 한다")
        if self.action_features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action_features 마지막 차원은 {ACTION_DIM}이어야 한다")
        _expect_bool("issued_mask", self.issued_mask)
        _expect_finite("action_features", self.action_features)


@dataclass(frozen=True)
class CEMIterationStats:
    """한 CEM iteration의 요약값."""

    iteration: int
    best_score: float
    elite_mean: float
    population_mean: float


@dataclass(frozen=True)
class CEMResult:
    """CEM 최적화 결과."""

    best_score: float
    best_plan: FutureActionPlanBatch
    final_distribution: CEMDistribution
    iteration_stats: tuple[CEMIterationStats, ...]


def _expect_rank(name: str, value: torch.Tensor, rank: int) -> None:
    """torch tensor rank 계약을 즉시 검증한다."""
    if value.ndim != rank:
        raise ValueError(f"{name} rank는 {rank}이어야 한다: shape={tuple(value.shape)}")


def _expect_bool(name: str, value: torch.Tensor) -> None:
    """mask tensor는 bool dtype만 허용한다."""
    if value.dtype != torch.bool:
        raise TypeError(f"{name} dtype은 torch.bool이어야 한다: dtype={value.dtype}")


def _expect_finite(name: str, value: torch.Tensor) -> None:
    """NaN/Inf가 분포와 점수에 섞이지 않게 한다."""
    if not torch.isfinite(value).all():
        raise ValueError(f"{name}에는 NaN 또는 Inf가 있으면 안 된다")


def _expect_positive(name: str, value: torch.Tensor) -> None:
    """표준편차 등 양수 tensor를 검증한다."""
    _expect_finite(name, value)
    if torch.any(value <= 0.0):
        raise ValueError(f"{name}는 모두 0보다 커야 한다")


def _expect_probability_rows(name: str, value: torch.Tensor) -> None:
    """마지막 차원의 categorical 확률 합이 1인지 검증한다."""
    _expect_finite(name, value)
    if torch.any(value < 0.0):
        raise ValueError(f"{name}에는 음수 확률이 있으면 안 된다")
    if not torch.allclose(value.sum(dim=-1), torch.ones_like(value.sum(dim=-1)), atol=1e-5):
        raise ValueError(f"{name}의 마지막 차원 합은 1이어야 한다")


def _normalized_probability(values: torch.Tensor) -> torch.Tensor:
    """명시적으로 양수 합을 가진 확률 벡터를 정규화한다."""
    _expect_rank("values", values, 1)
    _expect_finite("values", values)
    if torch.any(values < 0.0):
        raise ValueError("확률 값에는 음수가 있으면 안 된다")
    total = values.sum()
    if float(total.detach().cpu().item()) <= 0.0:
        raise ValueError("확률 합은 0보다 커야 한다")
    return (values / total).float()


def _clip_move_xy(values: torch.Tensor) -> torch.Tensor:
    """정규화 좌표를 [-1, 1] 범위로 제한한다."""
    return torch.clamp(values, -1.0, 1.0).float()


def _wrap_radians(values: torch.Tensor) -> torch.Tensor:
    """각도를 [-pi, pi] 범위로 접는다."""
    return torch.remainder(values + math.pi, 2.0 * math.pi).sub(math.pi).float()


def _batch_tensor(batch: SlotBatch, name: str, *, device: torch.device) -> torch.Tensor:
    """SlotBatch의 numpy 배열을 지정 device tensor로 변환한다."""
    value = getattr(batch, name)
    return torch.as_tensor(value, device=device)


def _team_slot_indices(batch: SlotBatch, team_id: TeamId, *, device: torch.device) -> torch.Tensor:
    """특정 팀 unit slot index를 반환한다."""
    type_ids = _batch_tensor(batch, "type_ids", device=device)
    team_ids = _batch_tensor(batch, "team_ids", device=device)
    indices = torch.nonzero((type_ids == int(ObjectType.UNIT)) & (team_ids == int(team_id)), as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(f"{team_id.name} unit slot이 하나도 없다")
    return indices.long()


def _mission_index(batch: SlotBatch, *, device: torch.device) -> int:
    """mission slot index를 반환한다."""
    type_ids = _batch_tensor(batch, "type_ids", device=device)
    indices = torch.nonzero(type_ids == int(ObjectType.MISSION), as_tuple=False).flatten()
    if indices.shape != (1,):
        raise ValueError(f"mission slot은 정확히 하나여야 한다: count={indices.numel()}")
    return int(indices[0].detach().cpu().item())


def blue_unit_ids(batch: SlotBatch, *, device: torch.device) -> torch.Tensor:
    """SlotBatch에서 CEM 대상 BLUE unit id를 slot 순서대로 반환한다."""
    indices = _team_slot_indices(batch, TeamId.BLUE, device=device)
    entity_ids = _batch_tensor(batch, "entity_ids", device=device).long()
    return entity_ids.index_select(0, indices)


def red_target_table(batch: SlotBatch, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """ENGAGE 후보에 사용할 RED target id와 현재 좌표를 반환한다."""
    indices = _team_slot_indices(batch, TeamId.RED, device=device)
    entity_ids = _batch_tensor(batch, "entity_ids", device=device).long()
    features = _batch_tensor(batch, "features", device=device).float()
    target_ids = entity_ids.index_select(0, indices)
    target_xy = features.index_select(0, indices)[:, [UNIT_X_INDEX, UNIT_Y_INDEX]]
    return target_ids, _clip_move_xy(target_xy)


def alive_blue_mask(batch: SlotBatch, *, device: torch.device) -> torch.Tensor:
    """현재 state에서 명령을 받을 수 있는 BLUE unit mask를 반환한다."""
    indices = _team_slot_indices(batch, TeamId.BLUE, device=device)
    features = _batch_tensor(batch, "features", device=device).float()
    return features.index_select(0, indices)[:, UNIT_ALIVE_INDEX] >= 0.5


def build_initial_distribution(current_batch: SlotBatch, config: CEMConfig, *, device: torch.device) -> CEMDistribution:
    """현재 DEVS state로 CEM 초기 분포를 만든다."""
    blue_indices = _team_slot_indices(current_batch, TeamId.BLUE, device=device)
    _, red_xy = red_target_table(current_batch, device=device)
    features = _batch_tensor(current_batch, "features", device=device).float()
    horizon = config.future_horizon
    num_blue = int(blue_indices.numel())
    num_red = int(red_xy.shape[0])

    prior = _normalized_probability(torch.tensor(config.action_prior, dtype=torch.float32, device=device))
    action_probs = prior.reshape(1, 1, ACTION_TYPE_COUNT).expand(horizon, num_blue, ACTION_TYPE_COUNT).clone()

    mission_index = _mission_index(current_batch, device=device)
    objective_xy = features[mission_index, [MISSION_OBJECTIVE_X_INDEX, MISSION_OBJECTIVE_Y_INDEX]]
    current_xy = features.index_select(0, blue_indices)[:, [UNIT_X_INDEX, UNIT_Y_INDEX]]
    move_mean = torch.zeros((horizon, num_blue, 2), dtype=torch.float32, device=device)
    for step_index in range(horizon):
        alpha = float(step_index + 1) / float(horizon)
        move_mean[step_index] = current_xy + alpha * (objective_xy.reshape(1, 2) - current_xy)
    move_mean = _clip_move_xy(move_mean)
    move_std = torch.full((horizon, num_blue, 2), config.initial_move_std, dtype=torch.float32, device=device)

    turn_mean = torch.zeros((horizon, num_blue), dtype=torch.float32, device=device)
    turn_std = torch.full((horizon, num_blue), config.initial_turn_std, dtype=torch.float32, device=device)
    target_probs = torch.full((horizon, num_blue, num_red), 1.0 / float(num_red), dtype=torch.float32, device=device)
    return CEMDistribution(
        action_probs=action_probs,
        move_mean=move_mean,
        move_std=move_std,
        turn_mean=turn_mean,
        turn_std=turn_std,
        target_probs=target_probs,
    )


def _sample_categorical(
    probabilities: torch.Tensor,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """categorical 분포에서 index를 샘플링한다."""
    _expect_rank("probabilities", probabilities, 2)
    return torch.multinomial(probabilities, sample_count, replacement=True, generator=generator).transpose(0, 1).long()


def sample_future_action_plans(
    *,
    distribution: CEMDistribution,
    current_batch: SlotBatch,
    config: CEMConfig,
    generator: torch.Generator,
    device: torch.device,
) -> FutureActionPlanBatch:
    """현재 CEM 분포에서 미래 joint action 후보들을 샘플링한다."""
    unit_ids = blue_unit_ids(current_batch, device=device)
    issued_unit_mask = alive_blue_mask(current_batch, device=device)
    red_ids, red_xy = red_target_table(current_batch, device=device)
    horizon, num_units, _ = distribution.action_probs.shape
    if horizon != config.future_horizon:
        raise ValueError("distribution horizon과 config.future_horizon이 같아야 한다")
    if num_units != unit_ids.numel():
        raise ValueError("distribution unit 수와 current_batch BLUE unit 수가 같아야 한다")

    candidates = config.num_candidates
    action_features = torch.zeros((candidates, horizon, num_units, ACTION_DIM), dtype=torch.float32, device=device)
    action_unit_ids = unit_ids.reshape(1, 1, num_units).expand(candidates, horizon, num_units).clone()
    issued_mask = issued_unit_mask.reshape(1, 1, num_units).expand(candidates, horizon, num_units).clone()
    action_type_ids = torch.full((candidates, horizon, num_units), NO_COMMAND_TYPE_ID, dtype=torch.long, device=device)
    target_entity_ids = torch.full((candidates, horizon, num_units), NO_TARGET_ENTITY_ID, dtype=torch.long, device=device)

    move_xy_norm = _clip_move_xy(
        torch.normal(
            mean=distribution.move_mean.reshape(1, horizon, num_units, 2).expand(candidates, horizon, num_units, 2),
            std=distribution.move_std.reshape(1, horizon, num_units, 2).expand(candidates, horizon, num_units, 2),
            generator=generator,
        )
    )
    theta_radians = _wrap_radians(
        torch.normal(
            mean=distribution.turn_mean.reshape(1, horizon, num_units).expand(candidates, horizon, num_units),
            std=distribution.turn_std.reshape(1, horizon, num_units).expand(candidates, horizon, num_units),
            generator=generator,
        )
    )

    flat_action_probs = distribution.action_probs.reshape(horizon * num_units, ACTION_TYPE_COUNT)
    sampled_actions = _sample_categorical(flat_action_probs, sample_count=candidates, generator=generator)
    action_type_ids = sampled_actions.reshape(candidates, horizon, num_units)
    flat_target_probs = distribution.target_probs.reshape(horizon * num_units, distribution.target_probs.shape[-1])
    sampled_targets = _sample_categorical(flat_target_probs, sample_count=candidates, generator=generator)
    target_indices = sampled_targets.reshape(candidates, horizon, num_units)

    action_features[..., 0] = action_type_ids.float()
    move_mask = issued_mask & (action_type_ids == int(ActionType.MOVE))
    action_features[..., 1][move_mask] = 1.0
    action_features[..., MOVE_X_INDEX][move_mask] = move_xy_norm[..., 0][move_mask]
    action_features[..., MOVE_Y_INDEX][move_mask] = move_xy_norm[..., 1][move_mask]

    engage_mask = issued_mask & (action_type_ids == int(ActionType.ENGAGE))
    selected_red_xy = red_xy.index_select(0, target_indices.reshape(-1)).reshape(candidates, horizon, num_units, 2)
    selected_red_ids = red_ids.index_select(0, target_indices.reshape(-1)).reshape(candidates, horizon, num_units)
    action_features[..., 4][engage_mask] = 1.0
    action_features[..., TARGET_TEAM_INDEX][engage_mask] = float(TeamId.RED)
    action_features[..., TARGET_X_INDEX][engage_mask] = selected_red_xy[..., 0][engage_mask]
    action_features[..., TARGET_Y_INDEX][engage_mask] = selected_red_xy[..., 1][engage_mask]
    target_entity_ids[engage_mask] = selected_red_ids[engage_mask]

    turn_mask = issued_mask & (action_type_ids == int(ActionType.TURN))
    action_features[..., 8][turn_mask] = 1.0
    action_features[..., THETA_COS_INDEX][turn_mask] = torch.cos(theta_radians)[turn_mask]
    action_features[..., THETA_SIN_INDEX][turn_mask] = torch.sin(theta_radians)[turn_mask]

    action_features[~issued_mask] = 0.0
    action_type_ids[~issued_mask] = NO_COMMAND_TYPE_ID
    target_entity_ids[~issued_mask] = NO_TARGET_ENTITY_ID
    return FutureActionPlanBatch(
        action_features=action_features,
        action_unit_ids=action_unit_ids,
        issued_mask=issued_mask,
        action_type_ids=action_type_ids,
        target_entity_ids=target_entity_ids,
        target_indices=target_indices,
        move_xy_norm=move_xy_norm,
        theta_radians=theta_radians,
        red_target_ids=red_ids,
    )


def update_distribution(
    *,
    distribution: CEMDistribution,
    plans: FutureActionPlanBatch,
    elite_indices: torch.Tensor,
    config: CEMConfig,
) -> CEMDistribution:
    """elite 후보로 CEM 분포를 갱신한다."""
    _expect_rank("elite_indices", elite_indices, 1)
    if elite_indices.numel() != config.num_elites:
        raise ValueError("elite_indices 길이는 config.num_elites와 같아야 한다")
    elite_indices = elite_indices.to(device=plans.action_features.device, dtype=torch.long)

    elite_actions = plans.action_type_ids.index_select(0, elite_indices)
    elite_targets = plans.target_indices.index_select(0, elite_indices)
    elite_moves = plans.move_xy_norm.index_select(0, elite_indices)
    elite_theta = plans.theta_radians.index_select(0, elite_indices)
    elite_issued = plans.issued_mask.index_select(0, elite_indices)

    action_probs = distribution.action_probs.clone()
    target_probs = distribution.target_probs.clone()
    horizon, num_units, _ = action_probs.shape
    for step_index in range(horizon):
        for unit_index in range(num_units):
            valid = elite_issued[:, step_index, unit_index]
            if torch.any(valid):
                counts = torch.bincount(
                    elite_actions[valid, step_index, unit_index],
                    minlength=ACTION_TYPE_COUNT,
                ).float()
                empirical_action = _normalized_probability(counts)
                empirical_action = torch.clamp(empirical_action, min=config.min_action_probability)
                empirical_action = _normalized_probability(empirical_action)
                action_probs[step_index, unit_index] = (
                    config.smoothing * distribution.action_probs[step_index, unit_index]
                    + (1.0 - config.smoothing) * empirical_action
                )
                action_probs[step_index, unit_index] = _normalized_probability(action_probs[step_index, unit_index])

                target_counts = torch.bincount(
                    elite_targets[:, step_index, unit_index],
                    minlength=distribution.target_probs.shape[-1],
                ).float()
                empirical_target = _normalized_probability(target_counts)
                target_probs[step_index, unit_index] = (
                    config.smoothing * distribution.target_probs[step_index, unit_index]
                    + (1.0 - config.smoothing) * empirical_target
                )
                target_probs[step_index, unit_index] = _normalized_probability(target_probs[step_index, unit_index])

    move_mean_emp = elite_moves.mean(dim=0)
    move_std_emp = torch.clamp(elite_moves.std(dim=0, unbiased=False), min=config.min_move_std)
    move_mean = _clip_move_xy(config.smoothing * distribution.move_mean + (1.0 - config.smoothing) * move_mean_emp)
    move_std = torch.clamp(
        config.smoothing * distribution.move_std + (1.0 - config.smoothing) * move_std_emp,
        min=config.min_move_std,
    ).float()

    theta_vector = torch.polar(torch.ones_like(elite_theta), elite_theta)
    theta_mean_emp = torch.angle(theta_vector.mean(dim=0)).float()
    resultant = torch.abs(theta_vector.mean(dim=0)).float()
    theta_std_emp = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.clamp(resultant, min=1e-6, max=1.0)), min=0.0))
    theta_std_emp = torch.clamp(theta_std_emp, min=config.min_turn_std)
    turn_mean = _wrap_radians(config.smoothing * distribution.turn_mean + (1.0 - config.smoothing) * theta_mean_emp)
    turn_std = torch.clamp(
        config.smoothing * distribution.turn_std + (1.0 - config.smoothing) * theta_std_emp,
        min=config.min_turn_std,
    ).float()

    return CEMDistribution(
        action_probs=action_probs.float(),
        move_mean=move_mean,
        move_std=move_std,
        turn_mean=turn_mean,
        turn_std=turn_std,
        target_probs=target_probs.float(),
    )


def _denorm_x(x_norm: torch.Tensor) -> torch.Tensor:
    """[-1, 1] x좌표를 DEVS 월드 x좌표로 되돌린다."""
    return (x_norm + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def _denorm_y(y_norm: torch.Tensor) -> torch.Tensor:
    """[-1, 1] y좌표를 DEVS 월드 y좌표로 되돌린다."""
    return (y_norm + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def _objective_xy_from_start(start_features: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
    """현재 mission slot에서 objective 월드 좌표를 읽는다."""
    mission = torch.nonzero(type_ids == int(ObjectType.MISSION), as_tuple=False).flatten()
    if mission.shape != (1,):
        raise ValueError(f"mission slot은 정확히 하나여야 한다: count={mission.numel()}")
    mission_features = start_features[int(mission[0].detach().cpu().item())]
    return torch.stack(
        [
            _denorm_x(mission_features[MISSION_OBJECTIVE_X_INDEX]),
            _denorm_y(mission_features[MISSION_OBJECTIVE_Y_INDEX]),
        ],
        dim=0,
    )


def _objective_distance(
    features: torch.Tensor,
    *,
    objective_xy: torch.Tensor,
    blue_indices: torch.Tensor,
) -> torch.Tensor:
    """각 후보/frame에서 가장 가까운 생존 아군의 objective 거리를 계산한다."""
    blue = features.index_select(-2, blue_indices)
    alive = torch.clamp(blue[..., UNIT_ALIVE_INDEX], 0.0, 1.0) >= 0.5
    x = _denorm_x(blue[..., UNIT_X_INDEX])
    y = _denorm_y(blue[..., UNIT_Y_INDEX])
    positions = torch.stack([x, y], dim=-1)
    distances = torch.linalg.norm(positions - objective_xy.reshape(*((1,) * (positions.ndim - 1)), 2), dim=-1)
    distances = torch.where(alive, distances, torch.full_like(distances, float("inf")))
    min_distance = distances.min(dim=-1).values
    return torch.where(
        torch.isfinite(min_distance),
        min_distance,
        torch.full_like(min_distance, NO_ALIVE_BLUE_DISTANCE),
    )


def score_future_features_torch(
    *,
    current_batch: SlotBatch,
    future_features: torch.Tensor,
    weights: EvaluatorWeights = EvaluatorWeights(),
) -> torch.Tensor:
    """월드모델 future feature batch를 torch evaluator 점수로 변환한다."""
    _expect_rank("future_features", future_features, 4)
    _expect_finite("future_features", future_features)
    device = future_features.device
    start_features = _batch_tensor(current_batch, "features", device=device).float()
    type_ids = _batch_tensor(current_batch, "type_ids", device=device).long()
    team_ids = _batch_tensor(current_batch, "team_ids", device=device).long()
    if future_features.shape[2] != start_features.shape[0]:
        raise ValueError("future_features slot 수가 current_batch와 같아야 한다")
    if future_features.shape[3] != MAX_FEATURE_DIM:
        raise ValueError(f"future_features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")

    blue_indices = torch.nonzero((type_ids == int(ObjectType.UNIT)) & (team_ids == int(TeamId.BLUE)), as_tuple=False).flatten()
    red_indices = torch.nonzero((type_ids == int(ObjectType.UNIT)) & (team_ids == int(TeamId.RED)), as_tuple=False).flatten()
    if blue_indices.numel() == 0:
        raise ValueError("BLUE unit slot이 하나도 없다")
    if red_indices.numel() == 0:
        raise ValueError("RED unit slot이 하나도 없다")

    final_features = future_features[:, -1]
    start_red_hp = torch.clamp(start_features.index_select(0, red_indices)[:, UNIT_HP_INDEX], 0.0, 1.0).sum()
    final_red_hp = torch.clamp(final_features.index_select(1, red_indices)[..., UNIT_HP_INDEX], 0.0, 1.0).sum(dim=-1)
    start_blue_hp = torch.clamp(start_features.index_select(0, blue_indices)[:, UNIT_HP_INDEX], 0.0, 1.0).sum()
    final_blue_hp = torch.clamp(final_features.index_select(1, blue_indices)[..., UNIT_HP_INDEX], 0.0, 1.0).sum(dim=-1)
    start_blue_ammo = torch.clamp(start_features.index_select(0, blue_indices)[:, UNIT_AMMO_INDEX], 0.0, 1.0).sum()
    final_blue_ammo = torch.clamp(final_features.index_select(1, blue_indices)[..., UNIT_AMMO_INDEX], 0.0, 1.0).sum(dim=-1)

    enemy_damage = torch.clamp(start_red_hp - final_red_hp, min=0.0)
    blue_damage = torch.clamp(start_blue_hp - final_blue_hp, min=0.0)
    ammo_used = torch.clamp(start_blue_ammo - final_blue_ammo, min=0.0) / float(blue_indices.numel())

    start_red_hp_each = torch.clamp(start_features.index_select(0, red_indices)[:, UNIT_HP_INDEX], 0.0, 1.0)
    start_red_alive_each = torch.clamp(start_features.index_select(0, red_indices)[:, UNIT_ALIVE_INDEX], 0.0, 1.0) >= 0.5
    final_red_hp_each = torch.clamp(final_features.index_select(1, red_indices)[..., UNIT_HP_INDEX], 0.0, 1.0)
    enemy_kia = ((start_red_hp_each.unsqueeze(0) > 0.01) & (final_red_hp_each <= 0.01)).float().sum(dim=-1)
    future_red_hp = torch.clamp(future_features.index_select(2, red_indices)[..., UNIT_HP_INDEX], 0.0, 1.0)
    future_red_alive = torch.clamp(future_features.index_select(2, red_indices)[..., UNIT_ALIVE_INDEX], 0.0, 1.0)
    # 특정 표적 집중을 직접 보상하지 않고, 살아 있는 RED HP mass가 horizon 안에서
    # 빨리 줄어드는 후보를 선호하게 만든다. KIA의 희소 신호를 보완하는 누적 보상이다.
    start_red_hp_mass = (start_red_hp_each * start_red_alive_each.float()).sum()
    future_red_hp_mass = (future_red_hp * (future_red_alive >= 0.5).float()).sum(dim=-1)
    enemy_hp_mass_reduction = torch.clamp(start_red_hp_mass - future_red_hp_mass, min=0.0).mean(dim=-1)
    # 총 피해량 보상은 분산 사격도 높게 평가한다. 낮은 HP 적을 더 낮게 만드는
    # 연속 보상을 더해 HP 0 예측 전에도 마무리 사격 방향을 CEM이 볼 수 있게 한다.
    start_finish_pressure = torch.square(1.0 - start_red_hp_each).unsqueeze(0)
    final_finish_pressure = torch.square(1.0 - final_red_hp_each)
    enemy_finish = torch.clamp(final_finish_pressure - start_finish_pressure, min=0.0).sum(dim=-1)
    red_cleared_final = torch.all(final_red_hp_each <= 0.01, dim=-1)

    objective_xy = _objective_xy_from_start(start_features, type_ids)
    start_distance = _objective_distance(
        start_features.reshape(1, start_features.shape[0], start_features.shape[1]),
        objective_xy=objective_xy,
        blue_indices=blue_indices,
    )[0]
    final_distance = _objective_distance(final_features, objective_xy=objective_xy, blue_indices=blue_indices)
    max_progress = float(future_features.shape[1]) * BLUE_MAX_STEP_PER_SEC
    objective_progress = (start_distance - final_distance) / max_progress
    progress_weight = torch.where(
        red_cleared_final,
        torch.full_like(objective_progress, weights.objective_progress_cleared),
        torch.full_like(objective_progress, weights.objective_progress),
    )

    red_eliminated = torch.all((future_red_hp <= 0.01) | (future_red_alive < 0.5), dim=-1)
    distances = _objective_distance(future_features, objective_xy=objective_xy, blue_indices=blue_indices)
    objective_reached = distances <= OBJECTIVE_RADIUS
    success = red_eliminated & objective_reached
    has_success = torch.any(success, dim=-1)
    first_success = torch.argmax(success.float(), dim=-1)
    early_success = torch.where(
        has_success,
        (float(future_features.shape[1]) - first_success.float()) / float(future_features.shape[1]),
        torch.zeros_like(first_success, dtype=torch.float32),
    )

    return (
        weights.enemy_damage * enemy_damage
        + weights.enemy_hp_mass_reduction * enemy_hp_mass_reduction
        + weights.enemy_kia * enemy_kia
        + weights.enemy_finish * enemy_finish
        - weights.blue_damage * blue_damage
        + progress_weight * objective_progress
        + weights.early_success * early_success
        - weights.ammo_used * ammo_used
    ).float()


def optimize_cem(
    *,
    current_batch: SlotBatch,
    initial_distribution: CEMDistribution,
    config: CEMConfig,
    rollout_fn: Callable[[FutureActionPlanBatch], torch.Tensor],
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> CEMResult:
    """CEM으로 최고 점수의 future action plan을 찾는다."""
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    distribution = initial_distribution
    best_score = -float("inf")
    best_plan = sample_future_action_plans(
        distribution=distribution,
        current_batch=current_batch,
        config=config,
        generator=generator,
        device=device,
    ).take_candidates(torch.tensor([0], dtype=torch.long, device=device))
    stats: list[CEMIterationStats] = []

    for iteration in range(config.num_iterations):
        plans = sample_future_action_plans(
            distribution=distribution,
            current_batch=current_batch,
            config=config,
            generator=generator,
            device=device,
        )
        future_features = rollout_fn(plans)
        _expect_rank("future_features", future_features, 4)
        if future_features.shape[0] != config.num_candidates:
            raise ValueError("rollout_fn이 반환한 candidate 수가 config.num_candidates와 같아야 한다")
        scores = score_fn(future_features)
        _expect_rank("scores", scores, 1)
        _expect_finite("scores", scores)
        if scores.shape != (config.num_candidates,):
            raise ValueError("score_fn 반환 shape는 (num_candidates,)이어야 한다")

        elite_scores, elite_indices = torch.topk(scores, k=config.num_elites, largest=True, sorted=True)
        iteration_best_score = float(elite_scores[0].detach().cpu().item())
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_plan = plans.take_candidates(elite_indices[:1])

        stats.append(
            CEMIterationStats(
                iteration=iteration,
                best_score=iteration_best_score,
                elite_mean=float(elite_scores.mean().detach().cpu().item()),
                population_mean=float(scores.mean().detach().cpu().item()),
            )
        )
        distribution = update_distribution(
            distribution=distribution,
            plans=plans,
            elite_indices=elite_indices,
            config=config,
        )

    return CEMResult(
        best_score=float(best_score),
        best_plan=best_plan,
        final_distribution=distribution,
        iteration_stats=tuple(stats),
    )


def stack_slot_history(batches: tuple[SlotBatch, ...], *, device: torch.device) -> dict[str, torch.Tensor]:
    """SlotBatch history를 월드모델 입력용 torch tensor로 쌓는다."""
    if not batches:
        raise ValueError("history SlotBatch가 비어 있다")
    reference = batches[0]
    for index, batch in enumerate(batches[1:], start=1):
        if not torch.equal(torch.as_tensor(reference.type_ids), torch.as_tensor(batch.type_ids)):
            raise ValueError(f"type_ids가 history frame {index}에서 변했다")
        if not torch.equal(torch.as_tensor(reference.entity_ids), torch.as_tensor(batch.entity_ids)):
            raise ValueError(f"entity_ids가 history frame {index}에서 변했다")
        if not torch.equal(torch.as_tensor(reference.team_ids), torch.as_tensor(batch.team_ids)):
            raise ValueError(f"team_ids가 history frame {index}에서 변했다")
    return {
        "features": torch.stack([_batch_tensor(batch, "features", device=device).float() for batch in batches], dim=0),
        "feature_mask": torch.stack([_batch_tensor(batch, "feature_mask", device=device).bool() for batch in batches], dim=0),
        "type_ids": torch.stack([_batch_tensor(batch, "type_ids", device=device).long() for batch in batches], dim=0),
        "entity_ids": torch.stack([_batch_tensor(batch, "entity_ids", device=device).long() for batch in batches], dim=0),
        "team_ids": torch.stack([_batch_tensor(batch, "team_ids", device=device).long() for batch in batches], dim=0),
        "alive_mask": torch.stack([_batch_tensor(batch, "alive_mask", device=device).bool() for batch in batches], dim=0),
    }


def build_history_from_v2_run(run_dir: Path, *, end_time: float, history_frames: int) -> tuple[SlotBatch, ...]:
    """v2 run 로그에서 끝 시점 기준 history SlotBatch를 읽는다."""
    if history_frames <= 0:
        raise ValueError("history_frames는 0보다 커야 한다")
    start_time = float(end_time) - float(history_frames - 1)
    if start_time < 0.0:
        raise ValueError("history window 시작 시간이 0보다 작을 수 없다")
    return tuple(build_slot_batch_from_v2_run(run_dir, start_time + float(offset)) for offset in range(history_frames))


def combine_observed_and_future_actions(
    *,
    observed: ObservedActionWindow,
    future: FutureActionPlanBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """관측 history action과 CEM 미래 action을 월드모델 입력 순서로 결합한다."""
    candidates = future.action_features.shape[0]
    observed_features = observed.action_features.unsqueeze(0).expand(candidates, *observed.action_features.shape)
    observed_unit_ids = observed.action_unit_ids.unsqueeze(0).expand(candidates, *observed.action_unit_ids.shape)
    observed_issued = observed.issued_mask.unsqueeze(0).expand(candidates, *observed.issued_mask.shape)
    full_features = torch.cat([observed_features, future.action_features], dim=1).float()
    full_unit_ids = torch.cat([observed_unit_ids, future.action_unit_ids], dim=1).long()
    full_issued = torch.cat([observed_issued, future.issued_mask], dim=1).bool()
    return full_features, full_unit_ids, full_issued


def rollout_with_world_model(
    *,
    model: object,
    history_batches: tuple[SlotBatch, ...],
    observed_actions: ObservedActionWindow,
    future_plans: FutureActionPlanBatch,
    device: torch.device,
) -> torch.Tensor:
    """월드모델로 CEM 후보들의 future feature를 예측한다."""
    history = stack_slot_history(history_batches, device=device)
    candidates = future_plans.action_features.shape[0]
    action_features, action_unit_ids, issued_mask = combine_observed_and_future_actions(
        observed=observed_actions,
        future=future_plans,
    )

    model.eval()
    with torch.no_grad():
        output = model.rollout_cjepa_future(
            history_features=history["features"].unsqueeze(0).expand(candidates, *history["features"].shape),
            history_feature_mask=history["feature_mask"].unsqueeze(0).expand(candidates, *history["feature_mask"].shape),
            history_type_ids=history["type_ids"].unsqueeze(0).expand(candidates, *history["type_ids"].shape),
            history_entity_ids=history["entity_ids"].unsqueeze(0).expand(candidates, *history["entity_ids"].shape),
            history_team_ids=history["team_ids"].unsqueeze(0).expand(candidates, *history["team_ids"].shape),
            history_alive_mask=history["alive_mask"].unsqueeze(0).expand(candidates, *history["alive_mask"].shape),
            action_features=action_features.to(device=device),
            action_unit_ids=action_unit_ids.to(device=device),
            issued_mask=issued_mask.to(device=device),
        )
    return output["future_features"]


def _format_plan_line(plan: FutureActionPlanBatch, candidate_index: int, step_index: int) -> str:
    """CLI 확인용으로 한 step joint action을 문자열화한다."""
    parts: list[str] = []
    for unit_index in range(plan.action_features.shape[2]):
        unit_id = int(plan.action_unit_ids[candidate_index, step_index, unit_index].detach().cpu().item())
        if not bool(plan.issued_mask[candidate_index, step_index, unit_index].detach().cpu().item()):
            parts.append(f"B{unit_id}:NO_COMMAND")
            continue
        action_type = ActionType(int(plan.action_type_ids[candidate_index, step_index, unit_index].detach().cpu().item()))
        if action_type == ActionType.MOVE:
            x, y = plan.move_xy_norm[candidate_index, step_index, unit_index].detach().cpu().tolist()
            parts.append(f"B{unit_id}:MOVE({x:.3f},{y:.3f})")
        elif action_type == ActionType.ENGAGE:
            target_id = int(plan.target_entity_ids[candidate_index, step_index, unit_index].detach().cpu().item())
            parts.append(f"B{unit_id}:ENGAGE(R{target_id})")
        elif action_type == ActionType.TURN:
            theta = float(plan.theta_radians[candidate_index, step_index, unit_index].detach().cpu().item())
            parts.append(f"B{unit_id}:TURN({theta:.3f})")
        elif action_type == ActionType.STOP:
            parts.append(f"B{unit_id}:STOP")
        else:
            raise ValueError(f"DEVS action vocabulary에 없는 action_type_id: {int(action_type)}")
    return " | ".join(parts)


def main(argv: Iterable[str] | None = None) -> None:
    """현재 state에서 torch CEM 초기 후보를 샘플링해 확인한다."""
    parser = argparse.ArgumentParser(description="Torch DEVS CEM future action sampler")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--state-time", type=float, required=True)
    parser.add_argument("--future-horizon", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--num-elites", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--save-pt", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    config = CEMConfig(
        num_candidates=args.num_candidates,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        future_horizon=args.future_horizon,
        seed=args.seed,
    )
    current_batch = build_slot_batch_from_v2_run(args.run_dir, args.state_time)
    distribution = build_initial_distribution(current_batch, config, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    plans = sample_future_action_plans(
        distribution=distribution,
        current_batch=current_batch,
        config=config,
        generator=generator,
        device=device,
    )

    print(f"device={device}")
    print(f"action_features_shape={tuple(plans.action_features.shape)}")
    print(f"action_unit_ids={plans.action_unit_ids[0, 0].detach().cpu().tolist()}")
    print(f"issued_mask={plans.issued_mask[0, 0].detach().cpu().tolist()}")
    for candidate_index in range(min(args.limit, plans.action_features.shape[0])):
        print(f"candidate={candidate_index}")
        for step_index in range(plans.action_features.shape[1]):
            print(f"  step={step_index}: {_format_plan_line(plans, candidate_index, step_index)}")

    if args.save_pt is not None:
        torch.save(
            {
                "action_features": plans.action_features.detach().cpu(),
                "action_unit_ids": plans.action_unit_ids.detach().cpu(),
                "issued_mask": plans.issued_mask.detach().cpu(),
                "action_type_ids": plans.action_type_ids.detach().cpu(),
                "target_entity_ids": plans.target_entity_ids.detach().cpu(),
            },
            args.save_pt,
        )


if __name__ == "__main__":
    main()
