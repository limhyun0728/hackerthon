"""DEVS episode trajectory로 object-centric C-JEPA 월드모델을 학습한다.

학습 입력은 방금 끝난 시뮬레이션 episode의 연속 state와 DEVS action 전이이다.
파일 로그 reader는 검증 도구로만 남겨두고, 실제 반복 학습 loop는 메모리에
수집한 episode trajectory를 바로 window로 바꿔 사용한다. outcome head는 쓰지
않고, action-conditioned predictor가 다음 state 표현과 feature를 맞추도록
학습한다.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.worldmodel.actions import ACTION_DIM, ActionBatch, available_command_times, build_action_batch_from_v2_run
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig
from hackerthon.worldmodel.slots import (
    MAX_FEATURE_DIM,
    ObjectType,
    SlotBatch,
    TeamId,
    available_times,
    build_slot_batch_from_v2_run,
)

UNIT_HP_INDEX = 1
UNIT_ALIVE_INDEX = 7


@dataclass(frozen=True)
class TrainingWindow:
    """하나의 연속 DEVS 학습 구간."""

    run_dir: Path
    state_times: tuple[float, ...]
    action_times: tuple[float, ...]

    def __post_init__(self) -> None:
        """state/action 전이 개수를 검증한다."""
        if len(self.state_times) <= 1:
            raise ValueError("state_times는 최소 2개 이상이어야 한다")
        if len(self.action_times) != len(self.state_times) - 1:
            raise ValueError("action_times 길이는 state_times 길이보다 정확히 1 작아야 한다")


@dataclass(frozen=True)
class LoadedTrainingWindow:
    """메모리에 올라간 DEVS 학습 구간."""

    spec: TrainingWindow
    states: tuple[SlotBatch, ...]
    actions: tuple[ActionBatch, ...]

    def __post_init__(self) -> None:
        """로드된 state/action 묶음의 shape와 시간축을 검증한다."""
        if len(self.states) != len(self.spec.state_times):
            raise ValueError("로드된 state 수가 TrainingWindow와 다르다")
        if len(self.actions) != len(self.spec.action_times):
            raise ValueError("로드된 action 수가 TrainingWindow와 다르다")
        for expected_time, state in zip(self.spec.state_times, self.states, strict=True):
            if float(state.time_sec) != float(expected_time):
                raise ValueError(f"state time이 window와 다르다: {state.time_sec} != {expected_time}")
        for expected_time, action in zip(self.spec.action_times, self.actions, strict=True):
            if float(action.time_sec) != float(expected_time):
                raise ValueError(f"action time이 window와 다르다: {action.time_sec} != {expected_time}")
        _validate_state_sequence(self.states)
        _validate_action_sequence(self.actions)


@dataclass(frozen=True)
class EpisodeTrajectory:
    """하나의 DEVS episode가 실제로 만든 state/action 궤적."""

    episode_id: str
    states: tuple[SlotBatch, ...]
    actions: tuple[ActionBatch, ...]
    outcome: str

    def __post_init__(self) -> None:
        """episode 궤적의 최소 조건과 시간 순서를 검증한다."""
        if not self.episode_id:
            raise ValueError("episode_id는 비어 있을 수 없다")
        if len(self.states) <= 1:
            raise ValueError("episode trajectory에는 state가 최소 2개 있어야 한다")
        if not self.actions:
            raise ValueError("episode trajectory에는 action이 최소 1개 있어야 한다")
        state_times = [float(state.time_sec) for state in self.states]
        action_times = [float(action.time_sec) for action in self.actions]
        if state_times != sorted(state_times):
            raise ValueError("episode state time은 오름차순이어야 한다")
        if action_times != sorted(action_times):
            raise ValueError("episode action time은 오름차순이어야 한다")
        _validate_state_sequence(self.states)
        _validate_action_sequence(self.actions)


@dataclass(frozen=True)
class TrainingBatch:
    """모델 forward에 바로 넣는 torch tensor batch."""

    features: torch.Tensor
    feature_mask: torch.Tensor
    type_ids: torch.Tensor
    entity_ids: torch.Tensor
    team_ids: torch.Tensor
    alive_mask: torch.Tensor
    action_features: torch.Tensor
    action_unit_ids: torch.Tensor
    issued_mask: torch.Tensor


@dataclass(frozen=True)
class LossWeights:
    """latent JEPA 손실과 DEVS state 복원 손실의 비중."""

    latent: float = 1.0
    future_state: float = 1.0
    masked_history_state: float = 0.25
    combat_state: float = 4.0
    damage_delta: float = 8.0
    slot_self_state: float = 1.0

    def __post_init__(self) -> None:
        """손실 가중치가 음수가 아닌지 확인한다."""
        for name, value in (
            ("latent", self.latent),
            ("future_state", self.future_state),
            ("masked_history_state", self.masked_history_state),
            ("combat_state", self.combat_state),
            ("damage_delta", self.damage_delta),
            ("slot_self_state", self.slot_self_state),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} loss weight는 음수일 수 없다")
        if (
            self.latent
            + self.future_state
            + self.masked_history_state
            + self.combat_state
            + self.damage_delta
            + self.slot_self_state
            <= 0.0
        ):
            raise ValueError("loss weight 합은 0보다 커야 한다")


@dataclass(frozen=True)
class OptimizerConfig:
    """학습 루프 설정."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cuda:0"
    log_every: int = 1
    checkpoint_path: Path = Path("checkpoints/devs_object_centric_jepa.pt")

    def __post_init__(self) -> None:
        """optimizer 관련 수치 계약을 검증한다."""
        if self.epochs <= 0:
            raise ValueError("epochs는 0보다 커야 한다")
        if self.batch_size <= 0:
            raise ValueError("batch_size는 0보다 커야 한다")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate는 0보다 커야 한다")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay는 음수일 수 없다")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm은 0보다 커야 한다")
        if self.log_every <= 0:
            raise ValueError("log_every는 0보다 커야 한다")


@dataclass(frozen=True)
class StepMetrics:
    """한 optimizer step의 손실 요약."""

    loss_total: float
    loss_latent: float
    loss_future: float
    loss_masked_history: float
    loss_future_state: float
    loss_masked_history_state: float
    loss_combat_state: float
    loss_damage_delta: float
    loss_slot_self_state: float


def _validate_state_sequence(states: tuple[SlotBatch, ...]) -> None:
    """같은 window 안에서 slot layout이 고정되는지 확인한다."""
    if not states:
        raise ValueError("states가 비어 있다")
    reference = states[0]
    for index, state in enumerate(states):
        if state.features.shape != reference.features.shape:
            raise ValueError(f"state {index} features shape가 첫 state와 다르다")
        if state.feature_mask.shape != reference.feature_mask.shape:
            raise ValueError(f"state {index} feature_mask shape가 첫 state와 다르다")
        for name in ("type_ids", "entity_ids", "team_ids"):
            if not np.array_equal(getattr(state, name), getattr(reference, name)):
                raise ValueError(f"{name}는 window 안에서 변하면 안 된다: state_index={index}")


def _validate_action_sequence(actions: tuple[ActionBatch, ...]) -> None:
    """같은 window 안에서 action 대상 unit 순서가 고정되는지 확인한다."""
    if not actions:
        raise ValueError("actions가 비어 있다")
    reference = actions[0]
    for index, action in enumerate(actions):
        if action.features.shape != reference.features.shape:
            raise ValueError(f"action {index} features shape가 첫 action과 다르다")
        if action.features.shape[-1] != ACTION_DIM:
            raise ValueError(f"action feature 마지막 차원은 {ACTION_DIM}이어야 한다")
        if not np.array_equal(action.unit_ids, reference.unit_ids):
            raise ValueError(f"action unit_ids 순서는 window 안에서 변하면 안 된다: action_index={index}")


def _validate_batch_layout(windows: tuple[LoadedTrainingWindow, ...]) -> None:
    """batch 안의 모든 window가 같은 scenario layout인지 확인한다."""
    if not windows:
        raise ValueError("batch window가 비어 있다")
    reference_state = windows[0].states[0]
    reference_action = windows[0].actions[0]
    for index, window in enumerate(windows):
        state = window.states[0]
        action = window.actions[0]
        if state.features.shape != reference_state.features.shape:
            raise ValueError(f"batch window {index}의 slot feature shape가 기준과 다르다")
        if not np.array_equal(state.type_ids, reference_state.type_ids):
            raise ValueError(f"batch window {index}의 type_ids가 기준과 다르다")
        if not np.array_equal(state.entity_ids, reference_state.entity_ids):
            raise ValueError(f"batch window {index}의 entity_ids가 기준과 다르다")
        if not np.array_equal(state.team_ids, reference_state.team_ids):
            raise ValueError(f"batch window {index}의 team_ids가 기준과 다르다")
        if action.features.shape != reference_action.features.shape:
            raise ValueError(f"batch window {index}의 action feature shape가 기준과 다르다")
        if not np.array_equal(action.unit_ids, reference_action.unit_ids):
            raise ValueError(f"batch window {index}의 action unit_ids가 기준과 다르다")


def build_training_windows(
    run_dirs: tuple[Path, ...],
    *,
    history_frames: int,
    pred_frames: int,
    time_step: float,
) -> tuple[TrainingWindow, ...]:
    """run 로그에서 완전한 연속 state/action 학습 window를 찾는다."""
    if history_frames <= 0:
        raise ValueError("history_frames는 0보다 커야 한다")
    if pred_frames <= 0:
        raise ValueError("pred_frames는 0보다 커야 한다")
    if time_step <= 0.0:
        raise ValueError("time_step은 0보다 커야 한다")
    total_frames = history_frames + pred_frames
    windows: list[TrainingWindow] = []
    for run_dir in run_dirs:
        state_times = available_times(run_dir)
        command_times = available_command_times(run_dir)
        state_time_set = set(state_times)
        command_time_set = set(command_times)
        for start_time in state_times:
            window_state_times = tuple(float(start_time + offset * time_step) for offset in range(total_frames))
            window_action_times = tuple(window_state_times[:-1])
            state_complete = all(time_value in state_time_set for time_value in window_state_times)
            action_complete = all(time_value in command_time_set for time_value in window_action_times)
            if state_complete and action_complete:
                windows.append(
                    TrainingWindow(
                        run_dir=run_dir,
                        state_times=window_state_times,
                        action_times=window_action_times,
                    )
                )
    if not windows:
        raise ValueError("완전한 state/action 학습 window를 찾지 못했다")
    return tuple(windows)


def build_training_windows_from_episode(
    trajectory: EpisodeTrajectory,
    *,
    history_frames: int,
    pred_frames: int,
    time_step: float,
) -> tuple[LoadedTrainingWindow, ...]:
    """메모리의 episode trajectory에서 연속 학습 window를 만든다."""
    if history_frames <= 0:
        raise ValueError("history_frames는 0보다 커야 한다")
    if pred_frames <= 0:
        raise ValueError("pred_frames는 0보다 커야 한다")
    if time_step <= 0.0:
        raise ValueError("time_step은 0보다 커야 한다")
    total_frames = history_frames + pred_frames
    state_by_time = {float(state.time_sec): state for state in trajectory.states}
    action_by_time = {float(action.time_sec): action for action in trajectory.actions}
    if len(state_by_time) != len(trajectory.states):
        raise ValueError("episode state time이 중복됐다")
    if len(action_by_time) != len(trajectory.actions):
        raise ValueError("episode action time이 중복됐다")

    loaded: list[LoadedTrainingWindow] = []
    for start_time in sorted(state_by_time):
        state_times = tuple(float(start_time + offset * time_step) for offset in range(total_frames))
        action_times = tuple(state_times[:-1])
        if not all(time_value in state_by_time for time_value in state_times):
            continue
        if not all(time_value in action_by_time for time_value in action_times):
            continue
        states = tuple(state_by_time[time_value] for time_value in state_times)
        actions = tuple(action_by_time[time_value] for time_value in action_times)
        loaded.append(
            LoadedTrainingWindow(
                spec=TrainingWindow(
                    run_dir=Path(trajectory.episode_id),
                    state_times=state_times,
                    action_times=action_times,
                ),
                states=states,
                actions=actions,
            )
        )
    if not loaded:
        raise ValueError(f"episode {trajectory.episode_id}에서 학습 window를 만들지 못했다")
    return tuple(loaded)


def load_training_window(window: TrainingWindow) -> LoadedTrainingWindow:
    """TrainingWindow spec이 가리키는 DEVS 로그를 실제 batch 객체로 읽는다."""
    states = tuple(build_slot_batch_from_v2_run(window.run_dir, time_sec) for time_sec in window.state_times)
    actions = tuple(
        build_action_batch_from_v2_run(
            window.run_dir,
            command_time_sec=time_sec,
            state_time_sec=time_sec,
        )
        for time_sec in window.action_times
    )
    return LoadedTrainingWindow(spec=window, states=states, actions=actions)


def load_training_windows(windows: tuple[TrainingWindow, ...]) -> tuple[LoadedTrainingWindow, ...]:
    """학습에 사용할 모든 window를 메모리에 올린다."""
    if not windows:
        raise ValueError("로드할 TrainingWindow가 없다")
    return tuple(load_training_window(window) for window in windows)


def collate_training_batch(windows: tuple[LoadedTrainingWindow, ...], *, device: torch.device) -> TrainingBatch:
    """LoadedTrainingWindow 묶음을 torch tensor batch로 변환한다."""
    _validate_batch_layout(windows)
    features = torch.as_tensor(
        np.stack([np.stack([state.features for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.float32,
        device=device,
    )
    feature_mask = torch.as_tensor(
        np.stack([np.stack([state.feature_mask for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.bool,
        device=device,
    )
    type_ids = torch.as_tensor(
        np.stack([np.stack([state.type_ids for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.long,
        device=device,
    )
    entity_ids = torch.as_tensor(
        np.stack([np.stack([state.entity_ids for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.long,
        device=device,
    )
    team_ids = torch.as_tensor(
        np.stack([np.stack([state.team_ids for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.long,
        device=device,
    )
    alive_mask = torch.as_tensor(
        np.stack([np.stack([state.alive_mask for state in window.states], axis=0) for window in windows], axis=0),
        dtype=torch.bool,
        device=device,
    )
    action_features = torch.as_tensor(
        np.stack([np.stack([action.features for action in window.actions], axis=0) for window in windows], axis=0),
        dtype=torch.float32,
        device=device,
    )
    action_unit_ids = torch.as_tensor(
        np.stack([np.stack([action.unit_ids for action in window.actions], axis=0) for window in windows], axis=0),
        dtype=torch.long,
        device=device,
    )
    issued_mask = torch.as_tensor(
        np.stack([np.stack([action.issued_mask for action in window.actions], axis=0) for window in windows], axis=0),
        dtype=torch.bool,
        device=device,
    )
    if features.shape[-1] != MAX_FEATURE_DIM:
        raise ValueError(f"features 마지막 차원은 {MAX_FEATURE_DIM}이어야 한다")
    return TrainingBatch(
        features=features,
        feature_mask=feature_mask,
        type_ids=type_ids,
        entity_ids=entity_ids,
        team_ids=team_ids,
        alive_mask=alive_mask,
        action_features=action_features,
        action_unit_ids=action_unit_ids,
        issued_mask=issued_mask,
    )


def _masked_feature_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """padding이 아닌 실제 DEVS feature만 MSE에 사용한다."""
    if pred.shape != target.shape:
        raise ValueError("pred와 target shape가 같아야 한다")
    if mask.shape != pred.shape:
        raise ValueError("feature mask shape는 pred shape와 같아야 한다")
    if mask.dtype != torch.bool:
        raise TypeError("feature mask dtype은 torch.bool이어야 한다")
    valid_count = mask.float().sum()
    if float(valid_count.detach().cpu().item()) <= 0.0:
        raise ValueError("MSE에 사용할 유효 feature가 없다")
    squared = (pred - target).pow(2)
    return squared[mask].mean()


def _combat_unit_weights(type_ids: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
    """전투 의미가 큰 RED HP 변화를 더 크게 보는 unit 가중치."""
    unit = type_ids == int(ObjectType.UNIT)
    red = unit & (team_ids == int(TeamId.RED))
    blue = unit & (team_ids == int(TeamId.BLUE))
    return red.float() * 3.0 + blue.float()


def _combat_state_loss(
    *,
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    feature_mask: torch.Tensor,
    type_ids: torch.Tensor,
    team_ids: torch.Tensor,
) -> torch.Tensor:
    """미래 unit HP/alive feature를 별도 손실로 크게 맞춘다."""
    if pred_future.shape != target_future.shape:
        raise ValueError("combat_state pred/target shape가 같아야 한다")
    if feature_mask.shape != pred_future.shape:
        raise ValueError("combat_state feature_mask shape가 pred와 같아야 한다")
    weights = _combat_unit_weights(type_ids, team_ids)
    hp_valid = feature_mask[..., UNIT_HP_INDEX] & (weights > 0.0)
    alive_valid = feature_mask[..., UNIT_ALIVE_INDEX] & (weights > 0.0)
    hp_error = (pred_future[..., UNIT_HP_INDEX] - target_future[..., UNIT_HP_INDEX]).pow(2)
    alive_error = (pred_future[..., UNIT_ALIVE_INDEX] - target_future[..., UNIT_ALIVE_INDEX]).pow(2)
    weighted_error = hp_error * hp_valid.float() * weights + alive_error * alive_valid.float() * weights
    denom = (hp_valid.float() * weights).sum() + (alive_valid.float() * weights).sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return pred_future.new_zeros(())
    return weighted_error.sum() / denom


def _damage_delta_loss(
    *,
    pred_future: torch.Tensor,
    target_future: torch.Tensor,
    start_features: torch.Tensor,
    type_ids: torch.Tensor,
    team_ids: torch.Tensor,
) -> torch.Tensor:
    """마지막 관측 HP 대비 미래 HP 감소량 자체를 맞춰 피해/킬 신호를 latent에 남긴다."""
    if pred_future.shape != target_future.shape:
        raise ValueError("damage_delta pred/target shape가 같아야 한다")
    if start_features.shape != pred_future[:, 0].shape:
        raise ValueError("start_features shape가 미래 frame shape와 맞아야 한다")
    weights = _combat_unit_weights(type_ids, team_ids)
    start_hp = torch.clamp(start_features[..., UNIT_HP_INDEX], 0.0, 1.0).unsqueeze(1)
    pred_hp = torch.clamp(pred_future[..., UNIT_HP_INDEX], 0.0, 1.0)
    target_hp = torch.clamp(target_future[..., UNIT_HP_INDEX], 0.0, 1.0)
    pred_damage = torch.clamp(start_hp - pred_hp, min=0.0)
    target_damage = torch.clamp(start_hp - target_hp, min=0.0)
    # 실제 피해가 있거나 모델이 피해를 잘못 상상한 위치만 별도로 강하게 본다.
    event = ((target_damage > 1e-4) | (pred_damage > 1e-4)) & (weights > 0.0)
    if not bool(event.any()):
        return pred_future.new_zeros(())
    error = (pred_damage - target_damage).pow(2) * event.float() * weights
    denom = (event.float() * weights).sum()
    return error.sum() / denom


def compute_training_losses(
    *,
    model_output: dict[str, torch.Tensor],
    batch: TrainingBatch,
    model_config: ObjectSlotModelConfig,
    weights: LossWeights,
) -> tuple[torch.Tensor, StepMetrics]:
    """C-JEPA latent 손실과 DEVS feature 손실을 합친다."""
    history_frames = model_config.history_frames
    total_frames = model_config.history_frames + model_config.pred_frames
    pred_features = model_output["pred_features"]
    if pred_features.shape != batch.features.shape:
        raise ValueError("pred_features shape는 입력 features shape와 같아야 한다")
    if batch.features.shape[1] != total_frames:
        raise ValueError("batch 시간 길이는 history_frames + pred_frames와 같아야 한다")

    future_slice = slice(history_frames, total_frames)
    history_self_features = model_output["history_self_features"]
    if history_self_features.shape != batch.features[:, :history_frames].shape:
        raise ValueError("history_self_features shape는 history features와 같아야 한다")
    slot_self_state_loss = _masked_feature_mse(
        history_self_features,
        batch.features[:, :history_frames],
        batch.feature_mask[:, :history_frames],
    )
    future_state_loss = _masked_feature_mse(
        pred_features[:, future_slice],
        batch.features[:, future_slice],
        batch.feature_mask[:, future_slice],
    )
    combat_state_loss = _combat_state_loss(
        pred_future=pred_features[:, future_slice],
        target_future=batch.features[:, future_slice],
        feature_mask=batch.feature_mask[:, future_slice],
        type_ids=batch.type_ids[:, future_slice],
        team_ids=batch.team_ids[:, future_slice],
    )
    damage_delta_loss = _damage_delta_loss(
        pred_future=pred_features[:, future_slice],
        target_future=batch.features[:, future_slice],
        start_features=batch.features[:, history_frames - 1],
        type_ids=batch.type_ids[:, future_slice],
        team_ids=batch.team_ids[:, future_slice],
    )

    masked_indices = model_output["masked_indices"]
    if masked_indices.ndim != 1:
        raise ValueError("masked_indices rank는 1이어야 한다")
    if masked_indices.numel() > 0:
        masked_history_state_loss = _masked_feature_mse(
            pred_features[:, :history_frames, masked_indices],
            batch.features[:, :history_frames, masked_indices],
            batch.feature_mask[:, :history_frames, masked_indices],
        )
    else:
        masked_history_state_loss = pred_features.new_zeros(())

    latent_loss = model_output["loss"]
    loss_total = (
        weights.latent * latent_loss
        + weights.future_state * future_state_loss
        + weights.masked_history_state * masked_history_state_loss
        + weights.combat_state * combat_state_loss
        + weights.damage_delta * damage_delta_loss
        + weights.slot_self_state * slot_self_state_loss
    )
    metrics = StepMetrics(
        loss_total=float(loss_total.detach().cpu().item()),
        loss_latent=float(latent_loss.detach().cpu().item()),
        loss_future=float(model_output["loss_future"].detach().cpu().item()),
        loss_masked_history=float(model_output["loss_masked_history"].detach().cpu().item()),
        loss_future_state=float(future_state_loss.detach().cpu().item()),
        loss_masked_history_state=float(masked_history_state_loss.detach().cpu().item()),
        loss_combat_state=float(combat_state_loss.detach().cpu().item()),
        loss_damage_delta=float(damage_delta_loss.detach().cpu().item()),
        loss_slot_self_state=float(slot_self_state_loss.detach().cpu().item()),
    )
    return loss_total, metrics


def _batch_index_groups(num_items: int, batch_size: int, *, generator: torch.Generator) -> tuple[tuple[int, ...], ...]:
    """한 epoch에서 사용할 shuffle된 batch index 묶음을 만든다."""
    if num_items <= 0:
        raise ValueError("num_items는 0보다 커야 한다")
    if batch_size <= 0:
        raise ValueError("batch_size는 0보다 커야 한다")
    order = torch.randperm(num_items, generator=generator).tolist()
    groups = [tuple(order[start:start + batch_size]) for start in range(0, num_items, batch_size)]
    return tuple(groups)


def _mean_metrics(metrics: tuple[StepMetrics, ...]) -> StepMetrics:
    """epoch 안의 step metric 평균을 계산한다."""
    if not metrics:
        raise ValueError("평균낼 metric이 없다")
    count = float(len(metrics))
    return StepMetrics(
        loss_total=sum(item.loss_total for item in metrics) / count,
        loss_latent=sum(item.loss_latent for item in metrics) / count,
        loss_future=sum(item.loss_future for item in metrics) / count,
        loss_masked_history=sum(item.loss_masked_history for item in metrics) / count,
        loss_future_state=sum(item.loss_future_state for item in metrics) / count,
        loss_masked_history_state=sum(item.loss_masked_history_state for item in metrics) / count,
        loss_combat_state=sum(item.loss_combat_state for item in metrics) / count,
        loss_damage_delta=sum(item.loss_damage_delta for item in metrics) / count,
        loss_slot_self_state=sum(item.loss_slot_self_state for item in metrics) / count,
    )


def save_checkpoint(
    *,
    path: Path,
    model: DEVSObjectCentricWorldModel,
    optimizer: torch.optim.Optimizer,
    model_config: ObjectSlotModelConfig,
    optimizer_config: OptimizerConfig,
    loss_weights: LossWeights,
    epoch: int,
    global_step: int,
    metrics: tuple[StepMetrics, ...],
    episode: int | None = None,
    checkpoint_kind: str = "latest",
    validation_metrics: StepMetrics | None = None,
    validation_monitor: str | None = None,
    validation_value: float | None = None,
    best_validation_value: float | None = None,
    best_episode: int | None = None,
    bad_validation_checks: int = 0,
) -> None:
    """학습 재개와 CEM 평가에 필요한 checkpoint를 저장한다.

    ``latest``와 ``best`` 모두 같은 payload 계약을 쓴다. validation/early
    stopping 정보는 없는 경우 None으로 저장해 기존 checkpoint loader와의
    호환성을 유지한다.
    """
    if epoch <= 0:
        raise ValueError("checkpoint epoch는 0보다 커야 한다")
    if global_step <= 0:
        raise ValueError("checkpoint global_step은 0보다 커야 한다")
    if episode is not None and episode <= 0:
        raise ValueError("checkpoint episode는 0보다 커야 한다")
    if bad_validation_checks < 0:
        raise ValueError("bad_validation_checks는 음수일 수 없다")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "optimizer_config": {
                **asdict(optimizer_config),
                "checkpoint_path": str(optimizer_config.checkpoint_path),
            },
            "loss_weights": asdict(loss_weights),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": [asdict(item) for item in metrics],
            "episode": None if episode is None else int(episode),
            "checkpoint_kind": str(checkpoint_kind),
            "validation_metrics": None if validation_metrics is None else asdict(validation_metrics),
            "early_stopping": {
                "monitor": validation_monitor,
                "validation_value": None if validation_value is None else float(validation_value),
                "best_validation_value": (
                    None if best_validation_value is None else float(best_validation_value)
                ),
                "best_episode": None if best_episode is None else int(best_episode),
                "bad_validation_checks": int(bad_validation_checks),
            },
        },
        path,
    )


def _weighted_mean_metrics(
    weighted_metrics: tuple[tuple[StepMetrics, int], ...],
) -> StepMetrics:
    """서로 다른 마지막 batch 크기를 반영해 validation metric을 평균낸다."""
    if not weighted_metrics:
        raise ValueError("평균낼 validation metric이 없다")
    total_weight = sum(weight for _, weight in weighted_metrics)
    if total_weight <= 0:
        raise ValueError("validation metric weight 합은 0보다 커야 한다")

    def mean(name: str) -> float:
        return sum(getattr(metric, name) * weight for metric, weight in weighted_metrics) / float(total_weight)

    return StepMetrics(
        loss_total=mean("loss_total"),
        loss_latent=mean("loss_latent"),
        loss_future=mean("loss_future"),
        loss_masked_history=mean("loss_masked_history"),
        loss_future_state=mean("loss_future_state"),
        loss_masked_history_state=mean("loss_masked_history_state"),
        loss_combat_state=mean("loss_combat_state"),
        loss_damage_delta=mean("loss_damage_delta"),
        loss_slot_self_state=mean("loss_slot_self_state"),
    )


@torch.no_grad()
def evaluate_existing_model(
    *,
    model: DEVSObjectCentricWorldModel,
    windows: tuple[LoadedTrainingWindow, ...],
    model_config: ObjectSlotModelConfig,
    optimizer_config: OptimizerConfig,
    loss_weights: LossWeights,
    batch_size: int,
    validation_seed: int,
) -> StepMetrics:
    """고정 validation window에서 학습과 동일한 loss를 계산한다.

    검증 호출이 학습 중 mask RNG 진행 순서를 바꾸면 이후 학습 자체가 달라진다.
    따라서 validation 전용 RNG를 임시로 주입하고 종료 뒤 원래 RNG 객체를
    복원한다. model/EMA weight, optimizer, 학습 RNG에는 어떤 변경도 가하지 않는다.
    """
    if not windows:
        raise ValueError("validation window가 없다")
    if batch_size <= 0:
        raise ValueError("validation batch_size는 0보다 커야 한다")
    device = torch.device(optimizer_config.device)
    model.to(device)
    was_training = model.training
    predictor = getattr(model, "masked_predictor", None)
    original_mask_rng = getattr(predictor, "_mask_rng", None)
    if original_mask_rng is None:
        raise AttributeError("masked predictor에 validation용으로 교체할 _mask_rng가 없다")

    # 매 validation check마다 같은 mask sequence를 재현한다.
    predictor._mask_rng = np.random.default_rng(int(validation_seed))
    weighted_metrics: list[tuple[StepMetrics, int]] = []
    try:
        model.eval()
        # open/urban처럼 terrain slot 수가 다른 validation run도 함께 받을 수
        # 있도록 동일 layout끼리 묶은 뒤 batch를 만든다.
        layout_groups: dict[tuple[object, ...], list[LoadedTrainingWindow]] = {}
        for window in windows:
            state = window.states[0]
            action = window.actions[0]
            signature = (
                state.features.shape,
                state.type_ids.tobytes(),
                state.entity_ids.tobytes(),
                state.team_ids.tobytes(),
                action.features.shape,
                action.unit_ids.tobytes(),
            )
            layout_groups.setdefault(signature, []).append(window)

        for grouped_windows in layout_groups.values():
            for start in range(0, len(grouped_windows), batch_size):
                batch_windows = tuple(grouped_windows[start:start + batch_size])
                batch = collate_training_batch(batch_windows, device=device)
                output = model.predict_cjepa_sequence(
                    features=batch.features,
                    feature_mask=batch.feature_mask,
                    type_ids=batch.type_ids,
                    entity_ids=batch.entity_ids,
                    team_ids=batch.team_ids,
                    alive_mask=batch.alive_mask,
                    action_features=batch.action_features,
                    action_unit_ids=batch.action_unit_ids,
                    issued_mask=batch.issued_mask,
                )
                _, metrics = compute_training_losses(
                    model_output=output,
                    batch=batch,
                    model_config=model_config,
                    weights=loss_weights,
                )
                weighted_metrics.append((metrics, len(batch_windows)))
    finally:
        predictor._mask_rng = original_mask_rng
        model.train(was_training)
    return _weighted_mean_metrics(tuple(weighted_metrics))


def create_model_and_optimizer(
    *,
    model_config: ObjectSlotModelConfig,
    optimizer_config: OptimizerConfig,
) -> tuple[DEVSObjectCentricWorldModel, torch.optim.Optimizer]:
    """episode를跨서 유지할 월드모델과 optimizer를 만든다."""
    torch.manual_seed(optimizer_config.seed)
    device = torch.device(optimizer_config.device)
    model = DEVSObjectCentricWorldModel(model_config).to(device)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_parameters:
        raise ValueError("학습 가능한 model parameter가 없다")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    return model, optimizer


def train_existing_model(
    *,
    model: DEVSObjectCentricWorldModel,
    optimizer: torch.optim.Optimizer,
    windows: tuple[LoadedTrainingWindow, ...],
    model_config: ObjectSlotModelConfig,
    optimizer_config: OptimizerConfig,
    loss_weights: LossWeights,
    start_epoch: int = 0,
    global_step: int = 0,
) -> tuple[tuple[StepMetrics, ...], int]:
    """이미 유지 중인 월드모델을 새 episode window로 추가 학습한다."""
    if not windows:
        raise ValueError("학습 window가 없다")
    if start_epoch < 0:
        raise ValueError("start_epoch는 음수일 수 없다")
    if global_step < 0:
        raise ValueError("global_step은 음수일 수 없다")
    device = torch.device(optimizer_config.device)
    model.to(device)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_parameters:
        raise ValueError("학습 가능한 model parameter가 없다")
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(optimizer_config.seed + int(start_epoch))
    epoch_metrics: list[StepMetrics] = []

    for local_epoch in range(1, optimizer_config.epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        step_metrics: list[StepMetrics] = []
        groups = _batch_index_groups(len(windows), optimizer_config.batch_size, generator=shuffle_generator)
        for group in groups:
            batch_windows = tuple(windows[index] for index in group)
            batch = collate_training_batch(batch_windows, device=device)
            output = model.predict_cjepa_sequence(
                features=batch.features,
                feature_mask=batch.feature_mask,
                type_ids=batch.type_ids,
                entity_ids=batch.entity_ids,
                team_ids=batch.team_ids,
                alive_mask=batch.alive_mask,
                action_features=batch.action_features,
                action_unit_ids=batch.action_unit_ids,
                issued_mask=batch.issued_mask,
            )
            loss, metrics = compute_training_losses(
                model_output=output,
                batch=batch,
                model_config=model_config,
                weights=loss_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, optimizer_config.gradient_clip_norm)
            optimizer.step()
            model.update_target_encoders(model_config.ema_momentum)
            global_step += 1
            step_metrics.append(metrics)
            if global_step % optimizer_config.log_every == 0:
                print(
                    "step="
                    f"{global_step} epoch={epoch} loss={metrics.loss_total:.6f} "
                    f"latent={metrics.loss_latent:.6f} future_state={metrics.loss_future_state:.6f} "
                    f"masked_state={metrics.loss_masked_history_state:.6f} "
                    f"combat_state={metrics.loss_combat_state:.6f} damage_delta={metrics.loss_damage_delta:.6f} "
                    f"slot_self={metrics.loss_slot_self_state:.6f}"
                )
        mean_metrics = _mean_metrics(tuple(step_metrics))
        epoch_metrics.append(mean_metrics)
        print(
            "epoch_summary="
            f"{epoch} loss={mean_metrics.loss_total:.6f} latent={mean_metrics.loss_latent:.6f} "
            f"future={mean_metrics.loss_future:.6f} masked={mean_metrics.loss_masked_history:.6f} "
            f"future_state={mean_metrics.loss_future_state:.6f} "
            f"combat_state={mean_metrics.loss_combat_state:.6f} damage_delta={mean_metrics.loss_damage_delta:.6f} "
            f"slot_self={mean_metrics.loss_slot_self_state:.6f}"
        )
        save_checkpoint(
            path=optimizer_config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            optimizer_config=optimizer_config,
            loss_weights=loss_weights,
            epoch=epoch,
            global_step=global_step,
            metrics=tuple(epoch_metrics),
        )
    return tuple(epoch_metrics), global_step


def train(
    *,
    windows: tuple[LoadedTrainingWindow, ...],
    model_config: ObjectSlotModelConfig,
    optimizer_config: OptimizerConfig,
    loss_weights: LossWeights,
) -> tuple[DEVSObjectCentricWorldModel, tuple[StepMetrics, ...]]:
    """학습 window 묶음으로 새 object-centric C-JEPA 월드모델을 학습한다."""
    if not windows:
        raise ValueError("학습 window가 없다")
    model, optimizer = create_model_and_optimizer(
        model_config=model_config,
        optimizer_config=optimizer_config,
    )
    metrics, _ = train_existing_model(
        model=model,
        optimizer=optimizer,
        windows=windows,
        model_config=model_config,
        optimizer_config=optimizer_config,
        loss_weights=loss_weights,
    )
    return model, metrics


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="DEVS object-centric C-JEPA 월드모델 학습")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--pred-frames", type=int, default=2)
    parser.add_argument("--time-step", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/devs_object_centric_jepa.pt"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-encoder-layers", type=int, default=0)
    parser.add_argument("--num-predictor-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-masked-slots", type=int, default=2)
    parser.add_argument("--self-state-dim", type=int, default=32)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--mask-team-strategy", choices=("random_team", "blue", "red", "all"), default="random_team")
    parser.add_argument("--mask-count-min", type=int, default=1)
    parser.add_argument("--mask-count-max", type=int, default=5)
    parser.add_argument("--blue-team-mask-probability", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--blue-team-mask-count-min", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--blue-team-mask-count-max", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--future-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--masked-history-state-loss-weight", type=float, default=0.25)
    parser.add_argument("--combat-state-loss-weight", type=float, default=4.0)
    parser.add_argument("--damage-delta-loss-weight", type=float, default=8.0)
    parser.add_argument("--slot-self-state-loss-weight", type=float, default=1.0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("max_windows는 0보다 커야 한다")

    model_config = ObjectSlotModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_predictor_layers=args.num_predictor_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        history_frames=args.history_frames,
        pred_frames=args.pred_frames,
        num_masked_slots=args.num_masked_slots,
        self_state_dim=args.self_state_dim,
        mask_seed=args.mask_seed,
        mask_team_strategy=args.mask_team_strategy,
        mask_count_min=args.mask_count_min,
        mask_count_max=args.mask_count_max,
        blue_team_mask_probability=args.blue_team_mask_probability,
        blue_team_mask_count_min=args.blue_team_mask_count_min,
        blue_team_mask_count_max=args.blue_team_mask_count_max,
        ema_momentum=args.ema_momentum,
    )
    optimizer_config = OptimizerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint_path,
    )
    loss_weights = LossWeights(
        latent=args.latent_loss_weight,
        future_state=args.future_state_loss_weight,
        masked_history_state=args.masked_history_state_loss_weight,
        combat_state=args.combat_state_loss_weight,
        damage_delta=args.damage_delta_loss_weight,
        slot_self_state=args.slot_self_state_loss_weight,
    )

    window_specs = build_training_windows(
        tuple(args.run_dirs),
        history_frames=model_config.history_frames,
        pred_frames=model_config.pred_frames,
        time_step=args.time_step,
    )
    if args.max_windows is not None:
        window_specs = window_specs[:args.max_windows]
    if not window_specs:
        raise ValueError("max_windows 적용 뒤 학습 window가 없다")
    loaded_windows = load_training_windows(window_specs)
    total_steps = math.ceil(len(loaded_windows) / optimizer_config.batch_size) * optimizer_config.epochs
    print(
        f"windows={len(loaded_windows)} total_steps={total_steps} "
        f"history_frames={model_config.history_frames} pred_frames={model_config.pred_frames}"
    )
    train(
        windows=loaded_windows,
        model_config=model_config,
        optimizer_config=optimizer_config,
        loss_weights=loss_weights,
    )
    print(f"checkpoint={optimizer_config.checkpoint_path}")


if __name__ == "__main__":
    main()
