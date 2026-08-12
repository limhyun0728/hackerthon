"""프레임 → 학습 window (설계 1·2·7·8절).

window = [pre | a | h1 h2 | f1..f6] — 10틱 연속 구간.
- pre는 frame a의 vx,vy 계산에만 쓰고 모델 입력이 아니다.
- 잔차 레이블은 **전부 frame a 기준** (레이블이 빼는 프레임 = 조립이 더하는 프레임).
- 지형은 정적이라 프레임별로 복제하지 않고 한 벌만 담는다 (predictor에서 KV-only).
- 마스킹은 여기서 데이터에 굽지 않는다 — `sample_mask_slots`로 학습 시 배치마다 뽑는다.
  같은 window가 epoch마다 다른 마스크로 재사용된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import MaskConfig
from ..model.features import (
    ACTION_DIM,
    ACTION_ENGAGE,
    ACTION_MOVE,
    ACTION_STOP,
    ACTION_TURN,
    MAX_AMMO,
    MAX_HP,
    MISSION_DESTROY_ALL,
    MISSION_DESTROY_AND_REACH,
    MISSION_HOLD_OBJECTIVE,
    MISSION_REACH_OBJECTIVE,
    MAX_MOVE_PER_STEP,
    NUM_ACTION_TYPES,
    OBJECTIVE_RADIUS,
    TeamId,
    norm_x,
    norm_y,
)
from .episodes import Episode, UnitState

# window 구성 (설계 1절). HISTORY 3 = a + h1 + h2, PRED 6 = f1..f6.
HISTORY_FRAMES = 3
PRED_FRAMES = 6
TOTAL_FRAMES = HISTORY_FRAMES + PRED_FRAMES          # 모델이 보는 9프레임 (a..f6)
RAW_TICKS = 1 + TOTAL_FRAMES                          # pre 포함 10틱
LABEL_FRAMES = TOTAL_FRAMES - 1                       # h1..f6 8프레임 (a 기준 잔차)
ACTION_TICKS = TOTAL_FRAMES - 1                       # a..f5 발행분 — tick j는 프레임 j+1부터 영향


@dataclass(frozen=True)
class EpisodeLayout:
    """에피소드 안에서 불변인 슬롯 배치. 한 배치는 같은 layout끼리만 묶는다."""

    unit_ids: tuple[int, ...]      # id 오름차순 → BLUE(1xx)가 앞
    team_ids: np.ndarray           # (U,) TeamId 값
    num_blue: int
    terrain_features: np.ndarray   # (T, 9) 정적
    mission_type: int
    objective: tuple[float, float]
    duration_sec: float

    @property
    def num_units(self) -> int:
        return len(self.unit_ids)


@dataclass(frozen=True)
class Window:
    layout: EpisodeLayout
    anchor_tick: int
    unit_features: np.ndarray      # (9, U, 10)  a..f6
    mission_features: np.ndarray   # (9, 5)
    # 레이블 — 전부 frame a 기준 (h1..f6 순서, 8프레임)
    dpos: np.ndarray               # (8, U, 2) 월드 단위
    ddmg: np.ndarray               # (8, U)    (hp(a) − hp(t)) / MAX_HP
    dammo: np.ndarray              # (8, U)    (ammo(a) − ammo(t)) / MAX_AMMO
    heading: np.ndarray            # (8, U, 2) 절대 cos,sin
    completion: np.ndarray         # (6,)      f1..f6 유도 플래그
    pos_loss_mask: np.ndarray      # (U,) bool — 앵커 시점 생존 유닛만 위치 손실
    # 액션 토큰 (계획된 명령). 발행틱 a..f5 순서 — index j는 프레임 j+1부터 가시.
    actions: np.ndarray            # (8, num_blue, ACTION_DIM)


def _heading_pair(degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return math.cos(radians), math.sin(radians)


def _unit_vector(state: UnitState, prev: UnitState, team: int) -> np.ndarray:
    """UNIT 슬롯 특징 10개 (features.UNIT_FEATURES 순서)."""
    alive = state.hp > 0.0
    cos, sin = _heading_pair(state.heading_deg)
    # 사망 유닛은 동결이므로 속도도 0으로 둔다 — "정지 관측"과 구분할 필요가 없다.
    vx = (state.x - prev.x) / MAX_MOVE_PER_STEP if alive else 0.0
    vy = (state.y - prev.y) / MAX_MOVE_PER_STEP if alive else 0.0
    return np.asarray(
        [
            float(team),
            state.hp / MAX_HP,
            state.ammo / MAX_AMMO,
            norm_x(state.x),
            norm_y(state.y),
            cos,
            sin,
            float(alive),
            vx,
            vy,
        ],
        dtype=np.float32,
    )


def _terrain_features(episode: Episode) -> np.ndarray:
    from ..model.features import norm_height, norm_width

    rows = []
    for xmin, ymin, xmax, ymax in episode.obstacles:
        width, height = xmax - xmin, ymax - ymin
        rows.append(
            [
                1.0,
                norm_x(xmin + width / 2.0),
                norm_y(ymin + height / 2.0),
                norm_width(width),
                norm_height(height),
                0.0,  # traversability — 현재 장애물은 완전 엄폐물 (구 규약)
                1.0,  # movement_cost
                1.0,  # cover_value
                1.0,  # los_block
            ]
        )
    if not rows:
        return np.zeros((0, 9), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def derive_completion(
    *,
    mission_type: int,
    frame: dict[int, UnitState],
    blue_ids: tuple[int, ...],
    red_ids: tuple[int, ...],
    objective: tuple[float, float],
) -> bool:
    """승리조건에서 유도한 completion (구 mission_completed와 동일 의미)."""
    alive_blue = [frame[u] for u in blue_ids if u in frame and frame[u].hp > 0.0]
    if not alive_blue:
        return False
    red_alive = any(u in frame and frame[u].hp > 0.0 for u in red_ids)
    objective_reached = any(
        math.hypot(s.x - objective[0], s.y - objective[1]) <= OBJECTIVE_RADIUS
        for s in alive_blue
    )
    if mission_type == MISSION_DESTROY_AND_REACH:
        return (not red_alive) and objective_reached
    if mission_type == MISSION_DESTROY_ALL:
        return not red_alive
    if mission_type in (MISSION_REACH_OBJECTIVE, MISSION_HOLD_OBJECTIVE):
        return objective_reached
    raise ValueError(f"모르는 mission_type: {mission_type}")


def _mission_vector(
    episode: Episode, tick: int, frame: dict[int, UnitState]
) -> np.ndarray:
    time_remaining = max(0.0, (episode.duration_sec - float(tick)) / episode.duration_sec)
    completed = derive_completion(
        mission_type=episode.mission_type,
        frame=frame,
        blue_ids=episode.blue_ids,
        red_ids=episode.red_ids,
        objective=episode.objective,
    )
    return np.asarray(
        [
            float(episode.mission_type),
            norm_x(episode.objective[0]),
            norm_y(episode.objective[1]),
            time_remaining,
            float(completed),
        ],
        dtype=np.float32,
    )


def _parse_action(action: str, detail: str, red_ids: tuple[int, ...]) -> np.ndarray:
    """명령 하나 → 액션 특징. 파싱 실패는 조용히 삼키지 않고 STOP과 구분되게 issued만 남긴다."""
    vector = np.zeros(ACTION_DIM, dtype=np.float32)
    vector[0] = 1.0  # issued
    type_index = {"STOP": ACTION_STOP, "MOVE": ACTION_MOVE, "ENGAGE": ACTION_ENGAGE, "TURN": ACTION_TURN}.get(action)
    if type_index is None:
        return vector
    vector[1 + type_index] = 1.0
    move_offset = 1 + NUM_ACTION_TYPES
    if type_index == ACTION_MOVE:
        text = detail.strip().strip("()")
        try:
            x_str, y_str = text.split(",")
            vector[move_offset] = norm_x(float(x_str))
            vector[move_offset + 1] = norm_y(float(y_str))
        except ValueError:
            pass
    elif type_index == ACTION_ENGAGE:
        # detail 형식: "->R210"
        try:
            target_id = int(detail.strip().lstrip("->RB"))
            slot = red_ids.index(target_id)
            vector[move_offset + 2 + slot] = 1.0
        except ValueError:
            pass
    elif type_index == ACTION_TURN:
        try:
            cos, sin = _heading_pair(float(detail.strip()))
            vector[move_offset + 12] = cos
            vector[move_offset + 13] = sin
        except ValueError:
            pass
    return vector


def build_layout(episode: Episode) -> EpisodeLayout:
    unit_ids = tuple(sorted(episode.blue_ids) + sorted(episode.red_ids))
    team_ids = np.asarray(
        [int(TeamId.BLUE) if uid in episode.blue_ids else int(TeamId.RED) for uid in unit_ids],
        dtype=np.int64,
    )
    return EpisodeLayout(
        unit_ids=unit_ids,
        team_ids=team_ids,
        num_blue=len(episode.blue_ids),
        terrain_features=_terrain_features(episode),
        mission_type=episode.mission_type,
        objective=episode.objective,
        duration_sec=episode.duration_sec,
    )


def build_windows(episode: Episode, *, stride: int = 1) -> list[Window]:
    """에피소드 하나에서 모든 window를 만든다. 연속 10틱이 없으면 그 자리는 건너뛴다."""
    layout = build_layout(episode)
    ticks = episode.ticks
    tick_set = set(ticks)
    windows: list[Window] = []

    for start in range(ticks[0], ticks[-1] - RAW_TICKS + 2, stride):
        needed = list(range(start, start + RAW_TICKS))
        if any(t not in tick_set for t in needed):
            continue
        # needed[0]=pre, needed[1]=a, needed[2:4]=h, needed[4:]=f
        frames = [episode.frames[t] for t in needed]

        # 프레임 결측 유닛은 직전 상태로 동결 (로그 관례상 없어야 하지만 방어)
        def state_of(frame_index: int, uid: int) -> UnitState:
            for back in range(frame_index, -1, -1):
                if uid in frames[back]:
                    return frames[back][uid]
            raise ValueError(f"유닛 {uid}가 window 안에 한 번도 없다")

        unit_features = np.zeros((TOTAL_FRAMES, layout.num_units, 10), dtype=np.float32)
        for fi in range(TOTAL_FRAMES):  # fi=0 → frame a (raw index 1)
            for ui, uid in enumerate(layout.unit_ids):
                current = state_of(fi + 1, uid)
                previous = state_of(fi, uid)
                unit_features[fi, ui] = _unit_vector(current, previous, int(layout.team_ids[ui]))

        mission_features = np.stack(
            [_mission_vector(episode, needed[fi + 1], frames[fi + 1]) for fi in range(TOTAL_FRAMES)]
        )

        # 레이블: frame a(=raw 1) 기준 잔차, h1..f6
        anchor = {uid: state_of(1, uid) for uid in layout.unit_ids}
        dpos = np.zeros((LABEL_FRAMES, layout.num_units, 2), dtype=np.float32)
        ddmg = np.zeros((LABEL_FRAMES, layout.num_units), dtype=np.float32)
        dammo = np.zeros((LABEL_FRAMES, layout.num_units), dtype=np.float32)
        heading = np.zeros((LABEL_FRAMES, layout.num_units, 2), dtype=np.float32)
        for li in range(LABEL_FRAMES):  # li=0 → h1 (raw index 2)
            for ui, uid in enumerate(layout.unit_ids):
                s = state_of(li + 2, uid)
                base = anchor[uid]
                dpos[li, ui] = (s.x - base.x, s.y - base.y)
                ddmg[li, ui] = (base.hp - s.hp) / MAX_HP
                dammo[li, ui] = (base.ammo - s.ammo) / MAX_AMMO
                heading[li, ui] = _heading_pair(s.heading_deg)

        completion = np.asarray(
            [
                derive_completion(
                    mission_type=episode.mission_type,
                    frame=frames[fi],
                    blue_ids=episode.blue_ids,
                    red_ids=episode.red_ids,
                    objective=episode.objective,
                )
                for fi in range(4, RAW_TICKS)  # f1..f6 = raw 4..9
            ],
            dtype=np.float32,
        )

        pos_loss_mask = np.asarray(
            [anchor[uid].hp > 0.0 for uid in layout.unit_ids], dtype=bool
        )

        # 액션 토큰: 발행틱 a..f5 (raw 1..8). index j → 프레임 j+1부터 가시.
        red_ids_sorted = tuple(sorted(episode.red_ids))
        actions = np.zeros((ACTION_TICKS, layout.num_blue, ACTION_DIM), dtype=np.float32)
        blue_slot = {uid: i for i, uid in enumerate(layout.unit_ids[: layout.num_blue])}
        for j in range(ACTION_TICKS):
            for command in episode.commands.get(needed[j + 1], ()):
                slot = blue_slot.get(command.unit_id)
                if slot is None:
                    continue  # RED 명령은 토큰이 아니다 (외생)
                actions[j, slot] = _parse_action(command.action, command.detail, red_ids_sorted)

        windows.append(
            Window(
                layout=layout,
                anchor_tick=needed[1],
                unit_features=unit_features,
                mission_features=mission_features,
                dpos=dpos,
                ddmg=ddmg,
                dammo=dammo,
                heading=heading,
                completion=completion,
                pos_loss_mask=pos_loss_mask,
                actions=actions,
            )
        )
    return windows


def sample_mask_slots(
    window: Window, rng: np.random.Generator, config: MaskConfig
) -> np.ndarray:
    """h1·h2에서 가릴 유닛 슬롯 index를 뽑는다 (양 팀, 앵커 생존자만).

    데이터에 굽지 않고 배치마다 뽑는다 — 같은 window가 epoch마다 다른 마스크로 재사용.
    """
    if rng.random() >= config.mask_probability:
        return np.zeros(0, dtype=np.int64)
    alive = np.nonzero(window.pos_loss_mask)[0]
    if alive.size == 0:
        return np.zeros(0, dtype=np.int64)
    count = int(rng.integers(1, config.max_masked_units + 1))
    count = min(count, alive.size)
    return rng.choice(alive, size=count, replace=False)
