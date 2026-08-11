"""episode 루프 안에서 value head를 온라인 학습한다.

월드모델과 **같은 데이터를 공유하되 손실은 따로**다. 월드모델은 (상태, 액션, 다음
상태)로 dynamics를 배우고, V는 (상태, 임무) 하나로 그 episode의 최종 임무 달성도를
배운다.

버퍼를 두는 이유: 한 episode의 표본들은 라벨이 전부 같다(그 episode의 결과 하나).
그것만으로 여러 epoch를 돌리면 매번 같은 답을 외운다. 월드모델은 window마다 라벨이
달라 이 문제가 없다. 최근 N episode를 모아 라벨 다양성을 확보한다.

표본은 t >= horizon 인 상태에서만 뽑는다. 플랫폼에서 V가 보는 것은 6스텝 지평선의
끝 상태(t = 6, 12, 18 ...)이므로, 그보다 이른 상태는 추론에서 나타나지 않는다.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hackerthon.worldmodel.slots import (
    MAX_HP,
    ObjectType,
    SlotBatch,
    TeamId,
)
from hackerthon.worldmodel.value_head import (
    MISSION_DESTROY_ALL,
    MISSION_HOLD_OBJECTIVE,
    MISSION_REACH_OBJECTIVE,
    OBJECTIVE_GAP_SCALE,
    ValueHead,
)

# 임무 종류 개수. value head의 mission one-hot 차원과 같아야 한다.
NUM_MISSION_TYPES = 4
# objective_reached 판정 반경. slots.OBJECTIVE_RADIUS와 같은 값이다.
OBJECTIVE_RADIUS = 1.0


@dataclass
class ValueSample:
    """상태 하나와 그 episode의 최종 달성도."""

    batch: SlotBatch
    mission_type: int
    progress: float


def mission_progress(
    *,
    mission_type: int,
    blue_alive: int,
    red_hp: float,
    red_initial: int,
    objective_distance: float,
) -> float:
    """mission_completed()의 불리언 조건을 연속 완화한 임무 달성도.

    임의 가중치를 쓰지 않는다 — destroy/reach의 정의와 둘을 잇는 min이 모두 원래
    승리 조건에서 나온다. AND 조건에 가중 평균을 쓰면 "적은 다 잡았지만 목표에 못
    갔다"가 0.5로 나와 실패를 절반의 성공으로 오독한다.
    """
    if blue_alive <= 0:
        return 0.0  # 전멸은 원 조건에서도 무조건 실패
    if not math.isfinite(objective_distance):
        objective_distance = OBJECTIVE_GAP_SCALE

    destroy = 1.0 - min(1.0, red_hp / (max(red_initial, 1) * MAX_HP))
    reach = 1.0 - min(
        1.0, max(0.0, objective_distance - OBJECTIVE_RADIUS) / OBJECTIVE_GAP_SCALE
    )

    if mission_type == MISSION_DESTROY_ALL:
        return destroy
    if mission_type in (MISSION_REACH_OBJECTIVE, MISSION_HOLD_OBJECTIVE):
        return reach
    return min(destroy, reach)  # DESTROY_AND_REACH : AND 조건


class OnlineValueTrainer:
    """episode마다 표본을 모아 value head를 조금씩 학습한다."""

    def __init__(
        self,
        *,
        value_head: ValueHead,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs_per_episode: int = 3,
        batch_size: int = 32,
        samples_per_episode: int = 10,
        buffer_episodes: int = 30,
        min_episodes: int = 10,
        seed: int = 17,
    ) -> None:
        if epochs_per_episode <= 0 or batch_size <= 0:
            raise ValueError("epochs_per_episode와 batch_size는 0보다 커야 한다")
        if samples_per_episode <= 0:
            raise ValueError("samples_per_episode는 0보다 커야 한다")
        if buffer_episodes <= 0 or min_episodes <= 0:
            raise ValueError("buffer_episodes와 min_episodes는 0보다 커야 한다")
        if min_episodes > buffer_episodes:
            raise ValueError("min_episodes는 buffer_episodes보다 클 수 없다")

        self.value_head = value_head
        self.device = device
        self.optimizer = torch.optim.AdamW(
            value_head.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.epochs_per_episode = int(epochs_per_episode)
        self.batch_size = int(batch_size)
        self.samples_per_episode = int(samples_per_episode)
        self.min_episodes = int(min_episodes)
        # episode 단위로 밀어내야 라벨 다양성이 유지된다. 표본 단위로 자르면 한
        # episode가 통째로 남거나 통째로 빠지는 대신 일부만 남아 편향된다.
        self.buffer: deque[list[ValueSample]] = deque(maxlen=int(buffer_episodes))
        self.rng = np.random.default_rng(seed)

    @property
    def ready(self) -> bool:
        """학습을 시작할 만큼 표본이 쌓였는지. 채점 여부와는 무관하다 —
        V는 규칙 기반 데이터로 이미 학습된 상태에서 warm-start하므로 처음부터
        CEM 채점에 쓴다."""
        return len(self.buffer) >= self.min_episodes

    def add_episode(
        self,
        *,
        states_by_time: dict[float, SlotBatch],
        mission_type: int,
        progress: float,
        horizon: int,
    ) -> int:
        """끝난 episode에서 표본을 뽑아 버퍼에 넣는다. 넣은 개수를 돌려준다."""
        times = sorted(time for time in states_by_time if time >= float(horizon))
        if not times:
            return 0
        count = min(self.samples_per_episode, len(times))
        picked = self.rng.choice(len(times), size=count, replace=False)
        samples = [
            ValueSample(
                batch=states_by_time[times[int(index)]],
                mission_type=int(mission_type),
                progress=float(progress),
            )
            for index in picked
        ]
        self.buffer.append(samples)
        return len(samples)

    def _inputs(self, sample: ValueSample) -> dict[str, torch.Tensor]:
        batch = sample.batch
        type_ids = torch.as_tensor(batch.type_ids, device=self.device).long()
        team_ids = torch.as_tensor(batch.team_ids, device=self.device).long()
        features = torch.as_tensor(batch.features, dtype=torch.float32, device=self.device)
        unit = type_ids == int(ObjectType.UNIT)
        alive = (features[:, 1] > 0.0) & unit
        return {
            "features": features.unsqueeze(0),
            "feature_mask": torch.as_tensor(batch.feature_mask, device=self.device).unsqueeze(0),
            "type_ids": type_ids.unsqueeze(0),
            "team_ids": team_ids.unsqueeze(0),
            "alive_mask": alive.unsqueeze(0),
            "blue_mask": (unit & (team_ids == int(TeamId.BLUE))).unsqueeze(0),
            "red_mask": (unit & (team_ids == int(TeamId.RED))).unsqueeze(0),
            "mission_onehot": torch.eye(NUM_MISSION_TYPES, device=self.device)[
                sample.mission_type
            ].unsqueeze(0),
        }

    def train(self) -> dict[str, float]:
        """버퍼에서 무작위로 뽑아 몇 epoch 학습한다."""
        flat = [sample for episode in self.buffer for sample in episode]
        if len(flat) < self.batch_size or not self.ready:
            return {"samples": float(len(flat)), "mae": float("nan"), "steps": 0.0}

        self.value_head.train()
        total_abs = 0.0
        seen = 0
        steps = 0
        for _ in range(self.epochs_per_episode):
            order = self.rng.permutation(len(flat))
            for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
                chunk = [flat[int(i)] for i in order[start : start + self.batch_size]]
                # 슬롯 수가 표본마다 달라 텐서로 못 묶는다. 누적 후 한 번에 갱신한다.
                losses = []
                for sample in chunk:
                    prediction = self.value_head(**self._inputs(sample))
                    target = torch.tensor(
                        [sample.progress], dtype=torch.float32, device=self.device
                    )
                    losses.append((prediction["progress"] - target).pow(2).mean())
                    total_abs += float((prediction["progress"] - target).abs().mean())
                    seen += 1
                loss = torch.stack(losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
                self.optimizer.step()
                steps += 1
        self.value_head.eval()
        return {
            "samples": float(len(flat)),
            "mae": total_abs / max(seen, 1),
            "steps": float(steps),
        }
