from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


AxisName = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW", "UNKNOWN"]
StrengthName = Literal["NONE", "LOW", "MEDIUM", "HIGH", "UNKNOWN"]
SuperiorityName = Literal["FRIENDLY", "ENEMY", "EVEN", "UNKNOWN"]
TacticName = Literal["FIX_AND_FLANK", "FOCUS_FIRE", "SEARCH_AND_BACK", "REGROUP"]
RoleName = Literal[
    "FIXER",
    "FLANKER",
    "SUPPORT",
    "SHOOTER",
    "SECURITY",
    "SCOUT",
    "RESERVE",
    "ANCHOR",
    "REGROUPER",
]
CommanderActionName = Literal["ENGAGE", "MOVE", "HOLD"]
ExecutableActionName = Literal["MOVE", "ENGAGE", "STOP"]


class StrictOutputModel(BaseModel):
    """정의하지 않은 LLM 출력 필드를 거부하는 공통 검증 모델."""

    model_config = ConfigDict(extra="forbid")


class SituationAssessment(StrictOutputModel):
    """Stage 1이 현재 지도에서 판단하는 순수 상황평가."""

    situation_summary: str
    main_enemy_axis: AxisName
    enemy_count_estimate: int = Field(ge=0)
    enemy_strength_estimate: StrengthName
    friendly_alive: int = Field(ge=0)
    friendly_wounded: List[int]
    local_superiority: SuperiorityName
    opportunity: str
    risk: str


class UnitRoleAssignment(StrictOutputModel):
    """Stage 2가 전술 안에서 유닛에 부여하는 역할."""

    unit_id: int
    role: RoleName
    reason: str


class TacticalPlan(StrictOutputModel):
    """5틱 동안 유지하는 전술과 유닛별 역할 계획."""

    tactic: TacticName
    rationale: str
    assignments: List[UnitRoleAssignment]


class UnitActionChoice(StrictOutputModel):
    """Stage 3이 현재 tick에 선택하는 유닛별 행동 종류."""

    unit_id: int
    action: CommanderActionName
    reason: str


class TickActionPlan(StrictOutputModel):
    """현재 tick에 모든 생존 유닛이 수행할 행동 종류."""

    rationale: str
    actions: List[UnitActionChoice]


class UnitMarkSelection(StrictOutputModel):
    """Stage 4가 행동을 구체화하기 위해 선택하는 표적 또는 grid mark."""

    unit_id: int
    mark: str
    reason: str


class MarkedActionPlan(StrictOutputModel):
    """모든 유닛의 Stage 4 mark 선택 결과."""

    rationale: str
    selections: List[UnitMarkSelection]


class SequenceCommand(StrictOutputModel):
    """Stage 5 helper가 mark를 변환해 만든 실행 명령."""

    unit_id: int
    action: ExecutableActionName
    x: Optional[float] = None
    y: Optional[float] = None
    target_id: Optional[int] = None
    reason: str

    @model_validator(mode="after")
    def 실행_필드_검증(self):
        """행동별 필수 실행 필드와 금지 필드를 검사한다."""
        if self.action == "MOVE":
            if self.x is None or self.y is None:
                raise ValueError("MOVE requires x and y")
            if self.target_id is not None:
                raise ValueError("MOVE must not include target_id")
        elif self.action == "ENGAGE":
            if self.target_id is None:
                raise ValueError("ENGAGE requires target_id")
            if self.x is not None or self.y is not None:
                raise ValueError("ENGAGE must not include x or y")
        elif self.x is not None or self.y is not None or self.target_id is not None:
            raise ValueError("STOP must not include x, y, or target_id")
        return self


class SequenceFrame(StrictOutputModel):
    """한 tick에 실행할 명령 묶음."""

    tick: int = Field(ge=0)
    commands: List[SequenceCommand]


class CommandSequence(StrictOutputModel):
    """Stage 5 실행 결과의 최종 검증 모델."""

    rationale: str
    sequence_duration_sec: float = Field(ge=1.0)
    frames: List[SequenceFrame]
