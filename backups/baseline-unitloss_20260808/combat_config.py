"""combat_config.py

전투 관련 공유 상수의 단일 출처(single source of truth).

LLM에게 전달되는 술어 임계값(predicateExtractor)과 실제 사격 판정을 하는
executor/soldier 로직이 서로 다른 매직넘버를 들고 있으면, "mid_visible = 사거리 내"
라는 LLM 약속이 조용히 깨질 수 있다. 사거리 같은 값은 여기서만 정의한다.
"""

# 근거리 명중률 구간 경계. 피해량은 더 이상 거리에 따라 달라지지 않는다.
# 지휘관 belief map의 near/mid bucket 경계로도 사용한다.
SHOOTING_RANGE = 4.0

# Effective fire boundary. Red rule AI engages at this range because
# hit_probability(7u)=0.35 >= RED_ENGAGE_MIN_HIT_PROB.
EFFECTIVE_FIRE_RANGE = 7.0

# Maximum non-zero damage and perception boundary.
MAX_FIRE_RANGE = 10.0

# 아군 지원 거리. 이 거리 이내에 다른 아군이 있으면 supported, 없으면 isolated로 판정한다.
SUPPORT_DISTANCE = 3.0

# 수정: cho/ff Soldier.py와 commander_helpers.py의 perception_range 기본값에 맞춘다.
PERCEPTION_RANGE = MAX_FIRE_RANGE

DANGER_CLOSE_RANGE = 2.0

# 이 거리를 초과하면 limited_support, 아군이 아예 없으면 isolated로 판정한다.
ISOLATED_DISTANCE = 4.0

RED_ENGAGE_MIN_HIT_PROB = 0.25

# 명중했을 때 거리에 관계없이 적용하는 고정 피해량.
DAMAGE_PER_HIT = 15


def hit_probability(distance: float) -> float:
    distance = float(distance)
    if distance <= 2.0:
        return 0.85
    if distance <= SHOOTING_RANGE:
        return 0.65
    if distance <= EFFECTIVE_FIRE_RANGE:
        return 0.35
    if distance <= MAX_FIRE_RANGE:
        return 0.15
    return 0.0


def damage_for_distance(distance: float) -> int:
    """10 이내에서 명중하면 항상 같은 피해를 적용한다."""
    distance = float(distance)
    if distance <= MAX_FIRE_RANGE:
        return DAMAGE_PER_HIT
    return 0
