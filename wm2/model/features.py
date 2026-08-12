"""wm2 규약 상수와 슬롯 스키마의 유일한 정의처.

구 코드에서 import하지 않는다 — 값은 구 규약과 일치하도록 복사했고, 여기서 갈라지면
데이터 호환이 깨지므로 이 파일 밖에서 같은 상수를 재정의하는 것을 금지한다.
(설계: docs/wm2_설계.md 2·3절)
"""

from __future__ import annotations

from enum import IntEnum

# ── 세계 규약 (구 terrain.py와 동일 값) ──────────────────────────────────
WORLD_X_MIN, WORLD_X_MAX = -20.0, 20.0
WORLD_Y_MIN, WORLD_Y_MAX = -15.0, 10.0

# 1틱(1초) 이동 상한. 양 팀·모든 경로 공통 — 팀별로 갈리면 월드모델이 MOVE 좌표를
# "1초 뒤 위치"로 배우는 전제가 깨진다 (구 시스템에서 실측된 사고).
MAX_MOVE_PER_STEP = 1.0

MAX_HP = 100.0
MAX_AMMO = 30.0
OBJECTIVE_RADIUS = 1.0
# 사거리 (구 combat_config와 동일 값, 월드 단위). 유효 7 = 70m, 최대 10 = 100m.
EFFECTIVE_FIRE_RANGE_UNITS = 7.0
MAX_FIRE_RANGE_UNITS = 10.0

# ── 홀드아웃 (학습 투입 금지, scenarios.py가 강제) ───────────────────────
TRAIN_MAPS = (
    "seoultech", "yeouido", "gangnam", "itaewon", "sinchon",
    "daerim", "hongdae", "gunja", "myeongdong", "seongsu",
)
HELDOUT_MAPS = ("euljiro", "jamsil", "assembly", "yongsan")


class ObjectType(IntEnum):
    UNIT = 0
    TERRAIN = 1
    MISSION = 2


class TeamId(IntEnum):
    NONE = -1
    BLUE = 0
    RED = 1


MISSION_DESTROY_AND_REACH = 0
MISSION_DESTROY_ALL = 1
MISSION_REACH_OBJECTIVE = 2
MISSION_HOLD_OBJECTIVE = 3
MISSION_TYPE_BY_NAME = {
    "destroy_and_reach": MISSION_DESTROY_AND_REACH,
    "destroy_all": MISSION_DESTROY_ALL,
    "reach_objective": MISSION_REACH_OBJECTIVE,
    "hold_objective": MISSION_HOLD_OBJECTIVE,
}

# ── 슬롯 원시 특징 (설계 2절) ────────────────────────────────────────────
UNIT_FEATURES = (
    "team", "hp_ratio", "ammo_ratio", "x_norm", "y_norm",
    "heading_cos", "heading_sin", "alive", "vx_norm", "vy_norm",
)
TERRAIN_FEATURES = (
    "terrain_type", "x_norm", "y_norm", "width_norm", "height_norm",
    "traversability", "movement_cost", "cover_value", "los_block",
)
MISSION_FEATURES = (
    "mission_type", "objective_x_norm", "objective_y_norm",
    "time_remaining_ratio", "completion_flag",
)
FEATURES_BY_TYPE = {
    ObjectType.UNIT: UNIT_FEATURES,
    ObjectType.TERRAIN: TERRAIN_FEATURES,
    ObjectType.MISSION: MISSION_FEATURES,
}
MAX_FEATURE_DIM = max(len(names) for names in FEATURES_BY_TYPE.values())

# 자주 쓰는 특징 index (UNIT 기준)
UNIT_HP_INDEX = 1
UNIT_X_INDEX, UNIT_Y_INDEX = 3, 4
UNIT_VX_INDEX, UNIT_VY_INDEX = 8, 9


# ── 정규화 (구 slots.py와 동일 관례: [-1, 1]) ────────────────────────────
def norm_x(x: float) -> float:
    return (float(x) - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * 2.0 - 1.0


def norm_y(y: float) -> float:
    return (float(y) - WORLD_Y_MIN) / (WORLD_Y_MAX - WORLD_Y_MIN) * 2.0 - 1.0


def denorm_x(x_norm: float) -> float:
    return (float(x_norm) + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def denorm_y(y_norm: float) -> float:
    return (float(y_norm) + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def norm_width(w: float) -> float:
    return float(w) / (WORLD_X_MAX - WORLD_X_MIN)


def norm_height(h: float) -> float:
    return float(h) / (WORLD_Y_MAX - WORLD_Y_MIN)


# ── 액션 토큰 어휘 (설계 6절) ────────────────────────────────────────────
# 액션 토큰의 내용은 "계획된 명령"이다. rule 에피소드는 계획=실행이라 로그를 그대로 쓴다.
ACTION_STOP, ACTION_MOVE, ACTION_ENGAGE, ACTION_TURN = 0, 1, 2, 3
NUM_ACTION_TYPES = 4
MAX_RED_SLOTS = 10   # ENGAGE 표적 one-hot 폭. 팀 최대 10 규약과 같다.
# [issued(1) | type onehot(4) | move x,y norm(2) | engage red-slot onehot(10) | turn cos,sin(2)]
ACTION_DIM = 1 + NUM_ACTION_TYPES + 2 + MAX_RED_SLOTS + 2

TERRAIN_ENTITY_ID_BASE = 10_000   # 구 slots.py와 동일 대역

# ── 블록 layout (설계 3절) ───────────────────────────────────────────────
# position 25% | velocity 12.5% | state 25% | heading 12.5% | identity 나머지
# embedding 64에서 16 | 8 | 16 | 8 | 16.
BLOCK_RATIO = {"position": 0.25, "velocity": 0.125, "state": 0.25, "heading": 0.125}


def block_layout(embedding_dim: int) -> dict[str, slice]:
    """블록 이름 → embedding 차원 slice. identity가 나머지를 가져간다."""
    sizes = {name: max(1, int(embedding_dim * ratio)) for name, ratio in BLOCK_RATIO.items()}
    used = sum(sizes.values())
    if used >= embedding_dim:
        raise ValueError(f"embedding_dim {embedding_dim}이 블록 최소 크기보다 작다")
    layout: dict[str, slice] = {}
    offset = 0
    for name in ("position", "velocity", "state", "heading"):
        layout[name] = slice(offset, offset + sizes[name])
        offset += sizes[name]
    layout["identity"] = slice(offset, embedding_dim)
    return layout


# 블록별 원시 특징 index (설계 3절 표). 타입마다 각 블록에 어떤 원시 특징이 들어가나.
BLOCK_FEATURE_INDEX: dict[ObjectType, dict[str, tuple[int, ...]]] = {
    ObjectType.UNIT: {
        "position": (UNIT_X_INDEX, UNIT_Y_INDEX),
        "velocity": (UNIT_VX_INDEX, UNIT_VY_INDEX),
        "state": (UNIT_HP_INDEX, 2, 7),          # hp_ratio, ammo_ratio, alive
        "heading": (5, 6),                        # cos, sin
    },
    ObjectType.TERRAIN: {
        "position": (1, 2, 3, 4),                 # x, y, w, h — w,h는 기하라 위치 블록
        "velocity": (),
        "state": (5, 6, 7, 8),                    # trav, cost, cover, los
        "heading": (),
    },
    ObjectType.MISSION: {
        "position": (1, 2),                       # objective x, y
        "velocity": (),
        "state": (3, 4),                          # time_remaining, completion
        "heading": (),
    },
}
