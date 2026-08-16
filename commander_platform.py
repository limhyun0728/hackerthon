"""지휘관 시뮬레이션 플랫폼 서버.

지휘관이 맵과 부대를 정하면, 결심 시점(6스텝)마다 CEM으로 후보를 다수 뽑고
태세 축(교전태세 x 부대 대형)으로 아카이브에 배치해 서로 다른 전개 시나리오를
보여준다. 지휘관이 하나를 고르면 그 계획을 6스텝 실행하고 다음 결심으로 넘어간다.
선택 이력은 트리로 남아 "그때 다른 안을 골랐다면"을 되짚을 수 있다.

후보는 무작위로 샘플링한다. 최적안 하나를 찾는 게 목적이 아니라 가능한 전개의
폭을 보여주는 게 목적이고, 지휘관에게는 셀별 elite만 제시된다.

사용법:
    python commander_platform.py --port 8900
    브라우저에서 http://127.0.0.1:8900/
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.combat_config import EFFECTIVE_FIRE_RANGE, MAX_FIRE_RANGE, PERCEPTION_RANGE
from hackerthon.terrain import (
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
    cell_of,
    component_points,
    largest_free_component,
    has_los,
    path_pad_for_unit_radius,
    set_path_pad,
    snap_to_component,
)
from hackerthon.worldmodel.actions import ActionType
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    CEMDistribution,
    FutureActionPlanBatch,
    ObservedActionWindow,
    build_initial_distribution,
    rollout_with_world_model,
    sample_future_action_plans,
    score_future_features_torch,
    update_distribution,
)
from hackerthon.worldmodel.object_slot_attention import (
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.actions import ACTION_DIM, NO_TARGET_ENTITY_ID
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.slots import (
    MAX_AMMO,
    MAX_HP,
    MISSION_TYPE_BY_NAME,
    MISSION_TYPE_NAMES,
    ObjectType,
    TeamId,
    build_slot_batch,
    mission_type_from_config,
)
from hackerthon.worldmodel.value_head import load_value_head
from hackerthon.worldmodel.value_scoring import make_value_score_fn

try:
    from local_env import load_local_env
except ModuleNotFoundError:
    from hackerthon.local_env import load_local_env

from hackerthon.platform_ui import PAGE_HTML

load_local_env()

UNIT_TEAM_INDEX = 0
UNIT_HP_INDEX = 1
UNIT_AMMO_INDEX = 2
UNIT_X_INDEX = 3
UNIT_Y_INDEX = 4
UNIT_COS_INDEX = 5
UNIT_SIN_INDEX = 6

# 태세 축 격자. 절대 단위로 고정해야 시나리오가 바뀌어도 셀 의미가 유지된다.
# 월드모델 rollout을 나눠 넣을 후보 수. 슬롯이 많은 실측맵에서 OOM을 막는다.
CHUNK_SIZE = 16

ENGAGE_EDGES = (0.0, 0.001, 0.15, 0.35, 0.60)
# 대형 축은 끝/시작 평균 쌍거리 "비"다. 절대 거리를 쓰면 전사로 인원이 줄 때
# 남은 유닛끼리 뭉쳐 보여 기동하지 않았는데도 밀집으로 분류된다.
SPREAD_EDGES = (0.0, 0.70, 0.90, 1.10, 1.50)
ENGAGE_LABELS = ("순수기동", "산발사격", "교전", "적극교전", "전력사격")
SPREAD_LABELS = ("급속집결", "집결", "대형유지", "산개", "급속분산")
# wm2 관점 채점 라벨 — safe: 생존 비용 포함(β=1), score: 임무 진행 전념(β=0)
LENS_LABELS = {"safe": "안전형", "score": "득점형"}
# 대형 변화를 못 재는 경우(생존 2명 미만, 시작 대형이 사실상 0)는 "대형유지"로 접는다.
SPREAD_NEUTRAL_BIN = 2
# 시작 평균 쌍거리가 이보다 작으면 비율이 폭주하므로 변화를 정의하지 않는다.
MIN_START_SPREAD_UNITS = 0.3


def _denorm_x(value: float) -> float:
    return (float(value) + 1.0) * 0.5 * (WORLD_X_MAX - WORLD_X_MIN) + WORLD_X_MIN


def _denorm_y(value: float) -> float:
    return (float(value) + 1.0) * 0.5 * (WORLD_Y_MAX - WORLD_Y_MIN) + WORLD_Y_MIN


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index in range(len(edges) - 1, -1, -1):
        if value >= edges[index]:
            return index
    return 0


@dataclass
class TreeNode:
    """결심 하나. 부모에서 어떤 셀을 골라 여기 왔는지 남긴다."""

    node_id: str
    parent_id: str | None
    time_sec: float
    unit_rows: list[dict[str, Any]]           # 계획·표시에 쓰는 belief 상태
    true_rows: list[dict[str, Any]] | None = None   # 실제 전장 상태 (사후 대조용)
    chosen_label: str | None = None
    children: list[str] = field(default_factory=list)
    # 이 노드 시점의 RED belief. 세션 전역으로 두면 결심 이력을 되돌려도 나중에 알게 된
    # 위치와 last_seen이 남아, 그 시점에 몰랐던 정보가 화면에 보인다.
    red_belief: dict[int, dict[str, Any]] = field(default_factory=dict)
    # 부모에서 여기까지 실제로 벌어진 6프레임. 후보 미리보기는 예측이지만 확정된
    # 구간은 DEVS 결과가 있으므로 그걸 남긴다. _advance_true가 어차피 계산한다.
    true_path: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Session:
    """한 지휘관 세션. 맵/임무/부대와 결심 트리를 들고 있다."""

    map_name: str
    config: dict[str, Any]
    mission_type: int
    objective: tuple[float, float]
    duration_sec: float
    horizon: int
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    current_id: str = ""
    # 월드모델 rollout에 필요한 최근 state history. 없으면 현재 state를 복제해 채운다.
    history: list[Any] = field(default_factory=list)
    # RED belief. 관측되면 실제로, 아니면 예측으로 유지된다.
    red_belief: dict[int, dict[str, Any]] = field(default_factory=dict)
    # RED rule 두뇌 상태(last_seen·순찰 인덱스) — 6틱 전개 구간을 넘어 승계해
    # 풀심(연속)의 RED와 같은 추격 지속성을 만든다.
    red_rollout_states: dict[int, dict[str, Any]] = field(default_factory=dict)
    # 후보 캐시. 지휘관이 셀을 고르면 그 plan의 결과 state를 그대로 쓴다.
    pending: dict[str, Any] = field(default_factory=dict)

    @property
    def current(self) -> TreeNode:
        return self.nodes[self.current_id]

    @property
    def obstacles(self) -> list:
        return self.config.get("obstacles", [])


# 배치 전용 여유. 경로 계산용 PATH_PAD(1m)는 통행 판정 기준이라 건물에 바짝
# 붙는 지점도 통과시킨다. 부대를 그 자리에 두면 화면에서 건물에 걸쳐 보이므로
# 초기 배치는 더 넉넉한 여유를 요구한다.
SPAWN_CLEARANCE_UNITS = 0.8


def _clear_of_buildings(point: tuple[float, float], obstacles, clearance: float) -> bool:
    """건물 경계에서 clearance 이상 떨어져 있는지."""
    for x0, y0, x1, y1 in obstacles:
        dx = max(x0 - point[0], 0.0, point[0] - x1)
        dy = max(y0 - point[1], 0.0, point[1] - y1)
        if math.hypot(dx, dy) < clearance:
            return False
    return True


def _spawn_points(config: dict[str, Any]) -> list[tuple[float, float]]:
    """건물에서 충분히 떨어진 이동 가능 지점만 남긴다."""
    obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
    component = largest_free_component(obstacles)
    if not component:
        raise ValueError("이동 가능한 자유 공간이 없다")
    points = component_points(component)
    if not obstacles:
        return points
    roomy = [p for p in points if _clear_of_buildings(p, obstacles, SPAWN_CLEARANCE_UNITS)]
    # 여유를 못 만족하는 좁은 맵이면 원래 자유 공간으로 되돌린다.
    return roomy if len(roomy) >= 20 else points


def _initial_rows(
    config: dict[str, Any],
    *,
    blue_count: int,
    red_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """자유 공간에서 진영을 나눠 초기 부대를 배치한다."""
    points = _spawn_points(config)
    center_x = (WORLD_X_MIN + WORLD_X_MAX) / 2.0
    blue_side = [p for p in points if p[0] < center_x]
    red_side = [p for p in points if p[0] >= center_x]
    if not blue_side or not red_side:
        raise ValueError("진영을 나눌 자유 공간이 부족하다")

    def pick(pool: list, count: int, first_id: int) -> list[dict[str, Any]]:
        chosen: list[tuple[float, float]] = []
        order = rng.permutation(len(pool))
        for index in order:
            point = pool[int(index)]
            if all(np.hypot(point[0] - p[0], point[1] - p[1]) >= 1.5 for p in chosen):
                chosen.append(point)
                if len(chosen) >= count:
                    break
        while len(chosen) < count:
            chosen.append(pool[int(order[len(chosen) % len(order)])])
        return [
            {
                "id": first_id + i,
                "x": float(x),
                "y": float(y),
                "heading": 0.0,
                "hp": MAX_HP,
                "ammo": int(MAX_AMMO),
                "time": 0.0,
            }
            for i, (x, y) in enumerate(chosen[:count])
        ]

    return pick(blue_side, blue_count, 101) + pick(red_side, red_count, 201)


def _placeable(config: dict[str, Any], point: tuple[float, float]) -> bool:
    """지휘관이 찍은 좌표가 실제로 부대를 둘 수 있는 곳인지 본다."""
    obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
    if not obstacles:
        return True
    return cell_of(point) in largest_free_component(obstacles)


def _snap_placeable(config: dict[str, Any], point: tuple[float, float]) -> tuple[float, float] | None:
    """건물에 붙거나 안쪽이면 여유 있는 가장 가까운 지점으로 밀어준다."""
    obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
    if not obstacles:
        return point
    if _clear_of_buildings(point, obstacles, SPAWN_CLEARANCE_UNITS):
        return point
    pool = _spawn_points(config)
    if not pool:
        return None
    best = min(pool, key=lambda p: math.hypot(p[0] - point[0], p[1] - point[1]))
    return (float(best[0]), float(best[1]))


def _slot_batch(session: Session, node: TreeNode):
    """현재 state를 slot batch로 만든다."""
    rows = [
        {
            "id": str(r["id"]),
            "x": str(r["x"]),
            "y": str(r["y"]),
            "heading": str(r["heading"]),
            "hp": str(r["hp"]),
            "ammo": str(r["ammo"]),
        }
        for r in node.unit_rows
    ]
    return build_slot_batch(
        unit_rows=rows,
        obstacles=session.obstacles,
        time_sec=node.time_sec,
        duration_sec=session.duration_sec,
        objective=session.objective,
        mission_type=session.mission_type,
    )


def _rows_from_features(batch, features: np.ndarray, time_sec: float) -> list[dict[str, Any]]:
    """rollout 결과 feature를 다시 unit row로 되돌린다."""
    rows: list[dict[str, Any]] = []
    for index, entity_id in enumerate(batch.entity_ids):
        if int(batch.type_ids[index]) != int(ObjectType.UNIT):
            continue
        vector = features[index]
        rows.append(
            {
                "id": int(entity_id),
                "x": _denorm_x(vector[UNIT_X_INDEX]),
                "y": _denorm_y(vector[UNIT_Y_INDEX]),
                "heading": float(np.degrees(np.arctan2(vector[UNIT_SIN_INDEX], vector[UNIT_COS_INDEX]))),
                "hp": float(np.clip(vector[UNIT_HP_INDEX], 0.0, 1.0) * MAX_HP),
                "ammo": int(round(float(np.clip(vector[UNIT_AMMO_INDEX], 0.0, 1.0)) * MAX_AMMO)),
                "time": float(time_sec),
            }
        )
    return rows


def _engage_allowed_mask(
    rows: list[dict[str, Any]], obstacles, device: torch.device
) -> torch.Tensor:
    """지금 실제로 사격 가능한 (BLUE, RED) 쌍.

    시뮬레이터(LosWorldAtomic)의 피해 판정과 같은 기준이다. World는 mode가
    ENGAGE이고 has_los가 트여 있으면 시야각과 무관하게 피해를 준다. 게이트를
    그보다 엄격하게 잡으면 실제로 가능한 사격을 막게 된다.
    """
    rects = [tuple(float(v) for v in rect) for rect in obstacles]
    blue = [r for r in rows if r["id"] < 200]
    red = [r for r in rows if r["id"] >= 200]
    mask = torch.zeros((len(blue), len(red)), dtype=torch.bool, device=device)
    for bi, b in enumerate(blue):
        if b["hp"] <= 0.0 or int(b["ammo"]) <= 0:
            continue
        for ri, e in enumerate(red):
            if e["hp"] <= 0.0:
                continue
            if math.hypot(e["x"] - b["x"], e["y"] - b["y"]) > MAX_FIRE_RANGE:
                continue
            if not has_los((b["x"], b["y"]), (e["x"], e["y"]), rects):
                continue
            mask[bi, ri] = True
    return mask


def _apply_engage_gate(distribution, allowed: torch.Tensor):
    """첫 스텝에서 불가능한 ENGAGE를 샘플링하지 않게 막는다.

    학습 루프의 CEMCommanderAtomic과 같은 규칙이다. 이게 없으면 후보 plan에
    건물 뒤 적을 쏘라는 지시가 들어가고, 화면에 관통 교전선으로 나타난다.
    """
    action_probs = distribution.action_probs.clone()
    target_probs = distribution.target_probs.clone()
    for unit_index in range(allowed.shape[0]):
        valid = allowed[unit_index].float()
        if bool(torch.any(allowed[unit_index])):
            target_probs[0, unit_index] = valid / valid.sum()
        else:
            action_probs[0, unit_index, int(ActionType.ENGAGE)] = 0.0
            target_probs[0, unit_index] = torch.full_like(
                target_probs[0, unit_index], 1.0 / float(target_probs.shape[-1])
            )
        total = action_probs[0, unit_index].sum()
        if float(total) > 0:
            action_probs[0, unit_index] = action_probs[0, unit_index] / total
    return CEMDistribution(
        action_probs=action_probs,
        move_mean=distribution.move_mean,
        move_std=distribution.move_std,
        turn_mean=distribution.turn_mean,
        turn_std=distribution.turn_std,
        target_probs=target_probs,
    )


def _normalize_angle_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _observed_red_ids(rows: list[dict[str, Any]], obstacles) -> set[int]:
    """BLUE 중 하나라도 파악하고 있는 RED id.

    판정을 사격 규칙(거리 + LOS)과 일치시킨다. 시뮬레이터는 시야각과 무관하게
    LOS만 트이면 피해를 주므로, 관측에만 FOV를 요구하면 "주황(미관측)인데
    사격선이 그려지는" 모순이 생긴다. 쏠 수 있으면 파악한 것으로 본다.
    지휘관 지도는 전지적 시점이 아니라 병사 관측 기반이다 — 이 반경을 사격보다
    넓히지 말 것 (2026-08-15 확인: 넓힘 시도는 설계 의도 위반으로 롤백됨).
    """
    rects = [tuple(float(v) for v in rect) for rect in obstacles]
    blue = [r for r in rows if r["id"] < 200 and r["hp"] > 0.0]
    red = [r for r in rows if r["id"] >= 200 and r["hp"] > 0.0]
    seen: set[int] = set()
    for b in blue:
        for e in red:
            if math.hypot(e["x"] - b["x"], e["y"] - b["y"]) > PERCEPTION_RANGE:
                continue
            if not has_los((b["x"], b["y"]), (e["x"], e["y"]), rects):
                continue
            seen.add(int(e["id"]))
    return seen


def _update_belief(
    session: "Session",
    true_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]] | None,
    time_sec: float,
) -> list[dict[str, Any]]:
    """관측된 RED는 실제값으로 갱신하고, 미관측은 예측으로 이어간다.

    지휘관이 보는 화면과 계획이 쓰는 state를 실제 전장이 아니라 belief로 맞춘다.
    관측되면 오차가 리셋되고, 계속 못 보면 예측 오차가 누적된다. 그 불일치가
    드러나는 것이 이 도구의 핵심 내용이다.
    """
    observed = _observed_red_ids(true_rows, session.obstacles)
    predicted = {int(r["id"]): r for r in (predicted_rows or [])}
    belief: list[dict[str, Any]] = []
    for row in true_rows:
        unit_id = int(row["id"])
        if unit_id < 200:
            belief.append(dict(row))          # BLUE는 항상 실측
            continue
        if unit_id in observed or row["hp"] <= 0.0:
            # 관측했거나 전사가 확인되면 belief를 실제로 리셋한다.
            entry = dict(row)
            entry["observed"] = True
            entry["last_seen"] = time_sec
        else:
            source = predicted.get(unit_id) or session.red_belief.get(unit_id) or row
            entry = dict(source)
            entry["id"] = unit_id
            entry["observed"] = False
            entry["last_seen"] = float(
                (session.red_belief.get(unit_id) or {}).get("last_seen", time_sec)
            )
            # 안 보이는 적은 위치만 예측으로 잇고 hp/ammo는 마지막 관측값에 묶는다.
            # 관측 판정과 사격 판정이 같은 조건(거리 MAX_FIRE_RANGE + LOS)이므로
            # 미관측 적은 BLUE의 사격을 받을 수도, BLUE를 쏠 수도 없다. 예측값을
            # 그대로 쓰면 멀쩡히 살아 있는 적이 belief에서 전사로 표시된다.
            last = session.red_belief.get(unit_id) or row
            entry["hp"] = float(last["hp"])
            entry["ammo"] = int(last["ammo"])
        entry["time"] = time_sec
        session.red_belief[unit_id] = entry
        belief.append(entry)
    return belief


def _plan_fire_pairs(
    plans, candidate: int, step: int, rows: list[dict[str, Any]], obstacles
) -> list[list[int]]:
    """이 후보가 이 스텝에 실제로 발사할 수 있는 (사수, 표적) 쌍.

    plan의 ENGAGE 지시를 쓰되, 그 프레임의 실제 위치로 사거리와 LOS를 다시 본다.
    교전 게이트는 첫 스텝에만 걸리므로 2스텝 이후 지시에는 건물 뒤 표적이 섞인다.
    시뮬레이터는 매 tick LOS를 확인해 실제로는 발사하지 않으니, 화면도 같은
    기준으로 걸러야 관통 사격처럼 보이지 않는다.
    """
    types = plans.action_type_ids[candidate, step].detach().cpu().numpy()
    targets = plans.target_entity_ids[candidate, step].detach().cpu().numpy()
    units = plans.action_unit_ids[candidate, step].detach().cpu().numpy()
    issued = plans.issued_mask[candidate, step].detach().cpu().numpy()
    by_id = {int(r["id"]): r for r in rows}
    rects = [tuple(float(v) for v in rect) for rect in obstacles]
    pairs: list[list[int]] = []
    for index in range(len(units)):
        if not issued[index] or int(types[index]) != int(ActionType.ENGAGE):
            continue
        target = int(targets[index])
        if target == NO_TARGET_ENTITY_ID:
            continue
        shooter_row, target_row = by_id.get(int(units[index])), by_id.get(target)
        if shooter_row is None or target_row is None:
            continue
        if shooter_row["hp"] <= 0.0 or target_row["hp"] <= 0.0:
            continue
        if math.hypot(target_row["x"] - shooter_row["x"], target_row["y"] - shooter_row["y"]) > MAX_FIRE_RANGE:
            continue
        if not has_los(
            (shooter_row["x"], shooter_row["y"]), (target_row["x"], target_row["y"]), rects
        ):
            continue
        pairs.append([int(units[index]), target])
    return pairs


def _build_path(
    batch,
    features: np.ndarray,
    plans,
    candidate: int,
    base_time: float,
    obstacles,
    belief_start: list[dict[str, Any]] | None = None,
):
    """후보 하나의 프레임별 (유닛 상태, 실제 발사 가능한 교전선)을 만든다.

    RED에는 프레임마다 관측 여부를 붙인다. 예측이 어긋나는 것은 재생 구간
    안에서 벌어지므로, 여기서 관측 여부를 안 붙이면 지휘관은 6초 내내 모든
    적을 '확실히 보고 있는 적'으로 보게 된다. 정작 belief가 흐려지는 구간이
    가장 확실해 보이는 셈이라 정보가 반대로 전달된다.

    판정은 예측 좌표 위에서 한다. 이 경로는 아직 일어나지 않은 일이므로
    "이대로 가면 이 시점엔 보일 것이다"가 지휘관에게 맞는 정보다.

    RED의 hp/ammo는 관측된 프레임에서만 줄어들 수 있다. 사격 판정과 관측 판정이
    같은 조건이라 미관측 적은 BLUE와 총알을 주고받을 수 없는데, 월드모델은 그걸
    모르고 안 보이는 적을 죽여 놓는다. 그대로 두면 모든 전개안이 적을 잡은 것처럼
    보이고, 그 상태가 다음 세그먼트 시작값으로 이어져 시나리오 끝까지 간다.
    """
    # 세그먼트 시작 시점의 last_seen을 이어받는다. 이전 결심에서 이미 놓친
    # 적은 재생 첫 프레임부터 그만큼 불확실한 상태로 시작해야 한다.
    last_seen = {
        int(r["id"]): float(r.get("last_seen", base_time))
        for r in (belief_start or [])
        if int(r["id"]) >= 200
    }
    # 미관측 구간에서 동결할 기준값. 없으면 첫 프레임 예측값으로 채워진다.
    held = {
        int(r["id"]): (float(r["hp"]), int(r["ammo"]))
        for r in (belief_start or [])
        if int(r["id"]) >= 200
    }
    path = []
    for step in range(features.shape[0]):
        time_sec = base_time + step + 1
        rows = _rows_from_features(batch, features[step], time_sec)

        # 관측 판정 전에 hp를 직전 값으로 되돌려 둔다. 모델이 미관측 적을 죽여 놓은
        # 채로 판정하면 그 적이 시신으로 취급돼 관측 집합에서 빠지고, 되살릴 근거도
        # 함께 사라진다.
        predicted_state: dict[int, tuple[float, int]] = {}
        for row in rows:
            unit_id = int(row["id"])
            if unit_id < 200 or unit_id not in held:
                continue
            predicted_state[unit_id] = (float(row["hp"]), int(row["ammo"]))
            row["hp"], row["ammo"] = held[unit_id]

        observed = _observed_red_ids(rows, obstacles)
        for row in rows:
            unit_id = int(row["id"])
            if unit_id < 200:
                continue
            if unit_id in observed and unit_id in predicted_state:
                # 보고 있는 동안 예측된 피해만 인정한다. 회복은 없으니 낮은 쪽을 남긴다.
                hp, ammo = predicted_state[unit_id]
                row["hp"] = min(float(row["hp"]), hp)
                row["ammo"] = min(int(row["ammo"]), ammo)
            held[unit_id] = (float(row["hp"]), int(row["ammo"]))
            # 전사 확인도 관측이다. 시신 위치에 불확실 원을 씌우면 안 된다.
            if unit_id in observed or row["hp"] <= 0.0:
                last_seen[unit_id] = time_sec
                row["observed"] = True
            else:
                row["observed"] = False
            row["last_seen"] = last_seen.get(unit_id, time_sec)
        path.append(
            {"units": rows, "fire": _plan_fire_pairs(plans, candidate, step, rows, obstacles)}
        )
    return path


def _merge_belief_path(
    true_path: list[dict[str, Any]],
    predicted_path: list[dict[str, Any]],
    obstacles,
    belief_start: list[dict[str, Any]],
    base_time: float,
) -> list[dict[str, Any]]:
    """프레임마다 관측된 것은 실측, 미관측 RED는 예측으로 합친다.

    지휘관이 실제로 볼 수 있는 화면이다. BLUE와 관측된 RED는 DEVS가 굴린 진짜
    상태이고, 못 보는 RED는 월드모델 예측 위치로 계속 움직인다. 마지막 관측 자리에
    고정하지 않는 이유는, 적이 실제로는 움직이고 있고 그 추정이 다음 결심의 입력이
    되기 때문이다.

    관측 판정은 **실측 좌표**로 한다. 보이느냐 마느냐는 적이 실제로 어디 있느냐로
    정해지지, 우리가 어디 있다고 믿느냐로 정해지지 않는다.

    hp/ammo는 미관측이면 동결한다. 관측 조건과 사격 조건이 같아서(거리 + LOS)
    안 보이는 적은 BLUE와 총알을 주고받을 수 없다.
    """
    last_seen = {
        int(r["id"]): float(r.get("last_seen", base_time))
        for r in belief_start
        if int(r["id"]) >= 200
    }
    held = {
        int(r["id"]): (float(r["hp"]), int(r["ammo"]))
        for r in belief_start
        if int(r["id"]) >= 200
    }
    merged: list[dict[str, Any]] = []
    for step, true_frame in enumerate(true_path):
        time_sec = base_time + step + 1
        true_rows = {int(r["id"]): r for r in true_frame["units"]}
        predicted = {}
        if step < len(predicted_path):
            predicted = {int(r["id"]): r for r in predicted_path[step]["units"]}
        observed = _observed_red_ids(list(true_rows.values()), obstacles)

        rows: list[dict[str, Any]] = []
        for unit_id, true_row in true_rows.items():
            if unit_id < 200:
                rows.append(dict(true_row))          # BLUE는 항상 실측
                continue
            if unit_id in observed or true_row["hp"] <= 0.0:
                entry = dict(true_row)
                entry["observed"] = True
                last_seen[unit_id] = time_sec
                held[unit_id] = (float(true_row["hp"]), int(true_row["ammo"]))
            else:
                source = predicted.get(unit_id) or true_row
                entry = dict(source)
                entry["id"] = unit_id
                entry["observed"] = False
                hp, ammo = held.get(unit_id, (float(true_row["hp"]), int(true_row["ammo"])))
                entry["hp"], entry["ammo"] = hp, ammo
            entry["time"] = time_sec
            entry["last_seen"] = last_seen.get(unit_id, time_sec)
            rows.append(entry)
        # 사격선은 실측을 쓴다. 쏘려면 보여야 하므로 미관측 적과의 사격선은 없다.
        merged.append({"units": rows, "fire": true_frame["fire"]})
    return merged


def _formation_spread(rows: list[dict[str, Any]], only_ids: set[int] | None = None) -> float:
    """BLUE 유닛 간 평균 쌍거리. only_ids를 주면 그 유닛들만 쓴다."""
    pts = [
        (r["x"], r["y"])
        for r in rows
        if r["id"] < 200 and r["hp"] > 0.0 and (only_ids is None or int(r["id"]) in only_ids)
    ]
    if len(pts) < 2:
        return float("nan")
    return float(
        np.mean([float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in itertools.combinations(pts, 2)])
    )


def _formation_change(
    start_rows: list[dict[str, Any]], end_rows: list[dict[str, Any]]
) -> float:
    """결심 시점 대비 horizon 끝의 대형 변화 배율.

    양쪽을 "끝까지 살아남은 같은 유닛 집합"으로 재는 것이 핵심이다. 생존자만으로
    끝 시점 대형을 재면, 바깥쪽 유닛을 잃은 부대가 남은 인원끼리 뭉쳐 보여
    기동하지 않았는데도 밀집으로 분류된다. 같은 집합으로 비를 내면 그 착시가
    사라지고 실제로 모였는지 흩어졌는지만 남는다.

    1.0이 대형 유지, 1보다 크면 산개, 작으면 집결이다. 배율이라 부대 규모나
    초기 대형이 달라도 같은 뜻으로 읽힌다.
    """
    survivors = {int(r["id"]) for r in end_rows if r["id"] < 200 and r["hp"] > 0.0}
    if len(survivors) < 2:
        return float("nan")
    start = _formation_spread(start_rows, survivors)
    end = _formation_spread(end_rows, survivors)
    if not np.isfinite(start) or not np.isfinite(end) or start < MIN_START_SPREAD_UNITS:
        return float("nan")
    return end / start


def build_archive(
    session: Session,
    *,
    candidates: int,
    seed: int,
    device: torch.device,
    model=None,
    model_config=None,
    value_head=None,
    iterations: int = 1,
    lens_heads=None,
    wm2_rollout=None,
) -> dict[str, Any]:
    """후보를 뽑아 태세 축 아카이브에 배치하고 셀별 elite를 만든다.

    lens_heads(wm2 안전형/득점형 value head)가 있으면 후보마다 두 관점으로 채점해
    셀×관점별 elite를 보존한다 — 아카이브 키가 (교전, 대형, 관점) 3축이 된다.
    분포 refit은 득점형 점수 기준(진행 지향 탐색), 안전형은 같은 풀에서 자기 elite를 뽑는다.
    """
    node = session.current
    batch = _slot_batch(session, node)
    alive_blue = sum(1 for r in node.unit_rows if r["id"] < 200 and r["hp"] > 0.0)
    alive_red = sum(1 for r in node.unit_rows if r["id"] >= 200 and r["hp"] > 0.0)
    if alive_blue == 0 or alive_red == 0:
        return {"cells": [], "finished": True, "alive_blue": alive_blue, "alive_red": alive_red}

    cem_config = CEMConfig(
        num_candidates=candidates,
        num_elites=max(2, candidates // 10),
        num_iterations=iterations,
        future_horizon=session.horizon,
        seed=seed,
        min_action_probability=0.0,
    )
    distribution = build_initial_distribution(batch, cem_config, device=device)
    distribution = _apply_engage_gate(
        distribution, _engage_allowed_mask(node.unit_rows, session.obstacles, device)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    snapshot = snapshot_from_slot_rows(
        unit_rows=node.unit_rows,
        obstacles=session.obstacles,
        base_time_sec=node.time_sec,
        episode_duration_sec=session.duration_sec,
        objective=session.objective,
        mission_type=session.mission_type,
    )
    if value_head is not None:
        # 학습 루프와 같은 채점을 쓴다. 휴리스틱 evaluator는 항별 가중치가 임의라
        # "아군을 버린 질주"가 격파의 32배로 평가되는 식의 왜곡이 있었다.
        score_fn = make_value_score_fn(
            value_head=value_head,
            current_batch=batch,
            mission_type=mission_type_from_config(session.config),
            device=device,
        )
    else:
        def score_fn(future_features):
            return score_future_features_torch(current_batch=batch, future_features=future_features)

    wm2_fn = None
    if wm2_rollout is not None:
        history_rows = getattr(node, "belief_tail", None)
        if not history_rows:
            history_rows = _warmup_history_rows(batch, snapshot, node.time_sec)
        wm2_fn = wm2_rollout.make_fn(
            current_rows=node.unit_rows, obstacles=session.obstacles,
            mission_type=session.mission_type, objective=session.objective,
            duration_sec=session.duration_sec, time_sec=node.time_sec, batch=batch,
            history_rows=history_rows,
        )

    def rollout(plans):
        """후보를 굴려 미래 feature를 낸다. wm2/구 model이 있으면 예측, 없으면 DEVS."""
        if wm2_fn is not None:
            return wm2_fn(plans)
        if model is None:
            return rollout_plans_with_devs(
                plans=plans, snapshot=snapshot, seed=seed, device=device,
                blue_max_step=1.0,   # 본게임 물리 (전원 1.0)
            )
        # history가 모자라면 현재 state를 복제해 채운다. 초반 결심에서도 예측이 되게.
        need = int(model_config.history_frames)
        hist = (session.history + [batch])[-need:]
        while len(hist) < need:
            hist = [batch] + hist
        num_units = int(plans.action_unit_ids.shape[2])
        zeros_f = torch.zeros((need - 1, num_units, ACTION_DIM), dtype=torch.float32, device=device)
        unit_ids = plans.action_unit_ids[0, 0].unsqueeze(0).expand(need - 1, num_units).contiguous()
        observed = ObservedActionWindow(
            action_features=zeros_f,
            action_unit_ids=unit_ids.to(device=device),
            issued_mask=torch.zeros((need - 1, num_units), dtype=torch.bool, device=device),
        )
        # 후보 전체를 한 번에 넣으면 슬롯 수(지형 100~200개)에 어텐션이 O(N^2)로
        # 곱해져 GPU 메모리가 터진다. 청크로 나눠 메모리를 상한에 묶는다.
        chunks = []
        for start in range(0, plans.action_features.shape[0], CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, plans.action_features.shape[0])
            index = torch.arange(start, stop, device=plans.action_features.device)
            chunks.append(
                rollout_with_world_model(
                    model=model,
                    history_batches=tuple(hist),
                    observed_actions=observed,
                    future_plans=plans.take_candidates(index),
                    device=device,
                )
            )
        return torch.cat(chunks, dim=0)

    # plan 축(BLUE만, 자체 순서)을 slot 축으로 잇는다. 두 축의 순서가 다르므로
    # entity id로 맞춰야 한다.
    slot_of = {
        int(entity_id): index
        for index, entity_id in enumerate(batch.entity_ids)
        if int(batch.type_ids[index]) == int(ObjectType.UNIT)
    }
    hp_now = {int(r["id"]): float(r["hp"]) for r in node.unit_rows}

    lens_scorer = None
    if lens_heads:
        from wm2_value_bridge import LensScorer

        lens_scorer = LensScorer(
            lens_heads, device=device, obstacles=session.obstacles,
            mission_type=session.mission_type, objective=session.objective,
            duration_sec=session.duration_sec, current_rows=node.unit_rows,
        )

    archive: dict[tuple[int, int, str], dict[str, Any]] = {}
    # CEM 반복을 돌리되 **모든 iteration의 후보를 전부 아카이빙**한다. 반복은 분포를
    # 좁혀 품질을 올리고, 아카이브는 그 과정에서 나온 후보를 태세 축에 흩뿌려 다양성을
    # 지킨다. 초반 iteration의 넓은 후보와 후반의 좋은 후보가 함께 남는다.
    for iteration in range(cem_config.num_iterations):
        plans = sample_future_action_plans(
            distribution=distribution, current_batch=batch, config=cem_config,
            generator=generator, device=device,
        )
        features = rollout(plans)
        if lens_scorer is not None:
            # wm2 관점 채점을 쓰면 legacy score_fn은 건너뛴다. refit용 점수는
            # 아래 후보 루프에서 득점형 관점으로 채워진다.
            scores = np.zeros(int(plans.action_features.shape[0]), dtype=np.float64)
        else:
            scores = score_fn(features).detach().cpu().numpy()
        features_np = features.detach().cpu().numpy()
        types = plans.action_type_ids.detach().cpu().numpy()
        issued = plans.issued_mask.detach().cpu().numpy()
        plan_unit_ids = plans.action_unit_ids[0, 0].detach().cpu().numpy()
        plan_slots = np.array([slot_of[int(unit_id)] for unit_id in plan_unit_ids])
        alive_now = np.array([hp_now.get(int(unit_id), 0.0) > 0.0 for unit_id in plan_unit_ids])

        for cand in range(features_np.shape[0]):
            # 전사자를 뺀다. 죽은 뒤 스텝의 명령은 실제로 나가지 않으므로, 그대로
            # 세면 일찍 전멸한 안이 오히려 적극적으로 교전한 것처럼 보인다.
            # 스텝 k의 명령은 k 시작 시점에 살아 있어야 유효하다. k 시작 = k-1 끝.
            alive_end = features_np[cand][:, plan_slots, UNIT_HP_INDEX] > 0.0
            alive_start = np.concatenate([alive_now[None, :], alive_end[:-1]], axis=0)
            engaged = (types[cand] == int(ActionType.ENGAGE)) & issued[cand] & alive_start
            engage = float(engaged.sum()) / float(max(int(alive_start.sum()), 1))
            rows = _rows_from_features(batch, features_np[cand, -1], node.time_sec + session.horizon)
            spread = _formation_change(node.unit_rows, rows)
            e_bin = _bin_index(engage, ENGAGE_EDGES)
            s_bin = SPREAD_NEUTRAL_BIN if np.isnan(spread) else _bin_index(spread, SPREAD_EDGES)
            if lens_scorer is not None:
                rows_prev = _rows_from_features(
                    batch, features_np[cand, -2], node.time_sec + session.horizon - 1
                )
                cand_scores = lens_scorer.score(
                    rows, rows_prev, node.time_sec + session.horizon
                )
                scores[cand] = cand_scores["score"]   # 분포 refit은 득점형 기준
            else:
                cand_scores = {"score": float(scores[cand])}
            path = None
            for lens, lens_score in cand_scores.items():
                key = (e_bin, s_bin, lens)
                if key in archive and lens_score <= archive[key]["score"]:
                    continue
                if path is None:
                    path = _build_path(
                        batch, features_np[cand], plans, cand,
                        node.time_sec, session.obstacles, belief_start=node.unit_rows,
                    )
                archive[key] = {
                    "score": float(lens_score),
                    "lens": lens,
                    # iteration마다 plans가 달라지므로 전역 index로는 못 찾는다.
                    # 선택 시 DEVS로 굴릴 수 있게 그 후보의 plan을 여기 들고 있는다.
                    "plan": plans.take_candidates(
                        torch.tensor([cand], dtype=torch.long, device=plans.action_features.device)
                    ),
                    "engage": engage,
                    "spread": 1.0 if np.isnan(spread) else float(spread),
                    # 마지막 프레임을 그대로 쓴다. 같은 좌표를 다시 만들면 관측 이력이
                    # 떨어져 나가, 추천 시나리오를 이어붙일 때 last_seen이 끊긴다.
                    "rows": path[-1]["units"],
                    "path": path,
                }

        if iteration + 1 < cem_config.num_iterations:
            elite = torch.topk(
                torch.as_tensor(scores, device=device), k=cem_config.num_elites, largest=True
            ).indices
            distribution = update_distribution(
                distribution=distribution, plans=plans, elite_indices=elite, config=cem_config
            )

    session.pending = {"archive": archive}
    cells = _cells_from_archive(archive)
    return {
        "cells": cells,
        "finished": False,
        "alive_blue": alive_blue,
        "alive_red": alive_red,
        "engage_labels": list(ENGAGE_LABELS),
        "spread_labels": list(SPREAD_LABELS),
        "lenses": ["safe", "score"] if lens_heads else ["score"],
    }


def _warmup_history_rows(batch, snapshot, time_sec: float) -> list[list[dict[str, Any]]]:
    """t=0(관측 이력 없음)용 합성 이력 — BLUE 정지 2틱을 DEVS로 굴려 RED 순찰 개시
    속도를 만든다.

    월드모델의 RED 예측은 이력 속도에 의존하는데, 배치 직후에는 이력이 없어 RED가
    제자리로 상상된다. 배치 시점엔 지휘관이 적을 직접 놓아 전원 관측 상태이므로
    실측 워밍업이 belief 규약을 어기지 않는다.
    """
    import torch as _t

    unit_ids = [int(e) for i, e in enumerate(batch.entity_ids)
                if int(batch.type_ids[i]) == int(ObjectType.UNIT)]
    blue_ids = sorted(u for u in unit_ids if u < 200)
    red_ids = sorted(u for u in unit_ids if u >= 200)
    shape3 = (1, 2, len(blue_ids))
    plans = FutureActionPlanBatch(
        action_features=_t.zeros(*shape3, ACTION_DIM),
        action_unit_ids=_t.as_tensor(blue_ids).reshape(1, 1, -1).expand(*shape3).clone(),
        issued_mask=_t.zeros(*shape3, dtype=_t.bool),
        action_type_ids=_t.zeros(*shape3, dtype=_t.long),
        target_entity_ids=_t.zeros(*shape3, dtype=_t.long),
        target_indices=_t.zeros(*shape3, dtype=_t.long),
        move_xy_norm=_t.zeros(*shape3, 2),
        theta_radians=_t.zeros(*shape3),
        red_target_ids=_t.as_tensor(red_ids, dtype=_t.long),
    )
    features = rollout_plans_with_devs(
        plans=plans, snapshot=snapshot, seed=0, device=torch.device("cpu"),
        blue_max_step=1.0,   # 본게임 물리 (전원 1.0)
    )
    array = np.asarray(features.detach().cpu()) if hasattr(features, "detach") else np.asarray(features)
    warm = [_rows_from_features(batch, array[0, k], time_sec) for k in range(array.shape[1])]
    # 워밍업은 미래 방향 순찰이다. 그대로 이력에 넣으면 h2 속도(현재−warm)가 역방향이
    # 되므로, 현재 기준 반사로 과거 프레임을 합성한다: past_k = current − (warm_k − current).
    # 속도열이 (warm2−warm1, warm1−current)로 순방향 순찰 스텝이 되고, anchor(h0)와
    # 예측 잔차의 기준 규약과도 자기일관이다. BLUE는 워밍업에서 정지라 그대로 남는다.
    current = {int(r["id"]): r for r in snapshot.unit_rows}
    def _reflect(frame_rows):
        out = []
        for r in frame_rows:
            c = current[int(r["id"])]
            out.append({**r,
                        "x": 2.0 * float(c["x"]) - float(r["x"]),
                        "y": 2.0 * float(c["y"]) - float(r["y"]),
                        "hp": float(c["hp"]), "ammo": c.get("ammo", r.get("ammo", 0))})
        return out
    return [_reflect(warm[-1]), _reflect(warm[0])]   # [h0(2틱 전), h1(1틱 전)]


def _cells_from_archive(archive: dict) -> list[dict[str, Any]]:
    """아카이브 엔트리를 표시용 셀 목록으로 요약한다 (build_archive·recommend 공용)."""
    cells = []
    for (e_bin, s_bin, lens), entry in sorted(archive.items()):
        blue_hp = sum(r["hp"] for r in entry["rows"] if r["id"] < 200)
        red_hp = sum(r["hp"] for r in entry["rows"] if r["id"] >= 200)
        cells.append(
            {
                "engage_bin": e_bin,
                "spread_bin": s_bin,
                "lens": lens,
                "label": f"{ENGAGE_LABELS[e_bin]} · {SPREAD_LABELS[s_bin]}",
                "score": entry["score"],
                "engage": entry["engage"],
                "spread": entry["spread"],
                "blue_hp": blue_hp,
                "red_hp": red_hp,
                "blue_alive": sum(1 for r in entry["rows"] if r["id"] < 200 and r["hp"] > 0),
                "red_alive": sum(1 for r in entry["rows"] if r["id"] >= 200 and r["hp"] > 0),
                "path": entry["path"],
                "rows": entry["rows"],
            }
        )
    return cells


class PlatformState:
    """서버 전역 상태. 맵 목록과 세션들을 들고 있다."""

    def __init__(
        self,
        maps_root: Path,
        *,
        device: torch.device,
        candidates: int,
        horizon: int,
        world_model_checkpoint: Path | None = None,
        archive_backend: str = "devs",
        recommend_backend: str = "devs",
        recommend_candidates: int = 24,
        value_head_checkpoint: Path | None = None,
        archive_iterations: int = 1,
        wm2_safe_value: Path | None = None,
        wm2_score_value: Path | None = None,
        wm2_checkpoint: Path | None = None,
    ):
        self.maps: dict[str, dict[str, Any]] = {}
        for path in sorted(maps_root.iterdir()):
            config_path = path / "config.json"
            if config_path.exists():
                self.maps[path.name] = json.loads(config_path.read_text(encoding="utf-8"))
        if not self.maps:
            raise ValueError(f"{maps_root} 아래에 맵이 없다")
        self.naver_client_id = (
            os.getenv("NAVER_MAP_CLIENT_ID")
            or os.getenv("NAVER_MAP_KEY_ID")
            or os.getenv("NCP_MAP_CLIENT_ID")
            or ""
        )
        self.device = device
        self.candidates = candidates
        self.horizon = horizon
        self.sessions: dict[str, Session] = {}
        self.model = None
        self.model_config = None
        self.archive_backend = archive_backend
        self.recommend_backend = recommend_backend
        self.recommend_candidates = recommend_candidates
        self.archive_iterations = archive_iterations
        if world_model_checkpoint is not None:
            payload = torch.load(world_model_checkpoint, map_location=device, weights_only=False)
            config_dict = dict(payload["model_config"])
            config_dict["maskable_type_ids"] = tuple(config_dict["maskable_type_ids"])
            self.model_config = ObjectSlotModelConfig(**config_dict)
            self.model = DEVSObjectCentricWorldModel(self.model_config).to(device)
            self.model.load_state_dict(payload["model_state_dict"])
            self.model.eval()
            print(
                f"월드모델 로드: {world_model_checkpoint.name} "
                f"(pred_frames={self.model_config.pred_frames}, history={self.model_config.history_frames})"
            )
        self.value_head = None
        if value_head_checkpoint is not None:
            self.value_head = load_value_head(value_head_checkpoint, device)
            print(f"value head 로드: {value_head_checkpoint.name} (아카이브 채점에 사용)")
        # wm2 안전형/득점형 관점 채점 (있으면 legacy value head 대신 이쪽을 쓴다)
        self.wm2_lens_heads = None
        lens_paths: dict[str, Path] = {}
        if wm2_safe_value is not None:
            lens_paths["safe"] = wm2_safe_value
        if wm2_score_value is not None:
            lens_paths["score"] = wm2_score_value
        if lens_paths:
            from wm2_value_bridge import load_lens_heads

            self.wm2_lens_heads = load_lens_heads(lens_paths, device)
        # wm2 월드모델 상상 백엔드 (--archive-backend wm2)
        self.wm2_rollout = None
        if wm2_checkpoint is not None:
            from wm2_value_bridge import Wm2Rollout

            self.wm2_rollout = Wm2Rollout(wm2_checkpoint, device)
        if archive_backend == "wm2" and self.wm2_rollout is None:
            raise ValueError("--archive-backend wm2에는 --wm2-checkpoint가 필요하다")

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        map_name = str(payload.get("map") or next(iter(self.maps)))
        config = self.maps[map_name]
        real_map = config.get("real_map", {})
        if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
            set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))

        blue = int(payload.get("blue", 5))
        red = int(payload.get("red", 7))
        mission = str(payload.get("mission", "destroy_all"))
        duration = float(payload.get("duration", 60.0))
        rng = np.random.default_rng(int(payload.get("seed", 0)))

        mission_type = MISSION_TYPE_BY_NAME.get(mission, 1)
        placed_blue = payload.get("blue_positions") or []
        placed_red = payload.get("red_positions") or []
        if placed_blue and placed_red:
            # 지휘관이 직접 찍은 배치. 건물 안이면 가장 가까운 통행 가능 지점으로 민다.
            rows = []
            for index, point in enumerate(placed_blue):
                snapped = _snap_placeable(config, (float(point[0]), float(point[1])))
                if snapped is None:
                    raise ValueError("아군 배치를 통행 가능 지점으로 옮길 수 없다")
                rows.append({"id": 101 + index, "x": snapped[0], "y": snapped[1],
                             "heading": 0.0, "hp": MAX_HP, "ammo": int(MAX_AMMO), "time": 0.0})
            for index, point in enumerate(placed_red):
                snapped = _snap_placeable(config, (float(point[0]), float(point[1])))
                if snapped is None:
                    raise ValueError("적군 배치를 통행 가능 지점으로 옮길 수 없다")
                rows.append({"id": 201 + index, "x": snapped[0], "y": snapped[1],
                             "heading": 0.0, "hp": MAX_HP, "ammo": int(MAX_AMMO), "time": 0.0})
        else:
            rows = _initial_rows(config, blue_count=blue, red_count=red, rng=rng)

        # 초기 heading을 상대 진영 중심으로 돌린다. 전원 0°(동쪽) 고정이면 서쪽의
        # 적은 120° 시야콘 밖이라, 바로 앞의 BLUE도 못 본 RED가 태연히 순찰을
        # 떠나는 왜곡이 생긴다 (2026-08-15 실측: "코앞인데 교전 안 함"의 원인 1).
        blues = [r for r in rows if r["id"] < 200]
        reds = [r for r in rows if r["id"] >= 200]
        for r in rows:
            foes = reds if r["id"] < 200 else blues
            if foes:
                cx = sum(f["x"] for f in foes) / len(foes)
                cy = sum(f["y"] for f in foes) / len(foes)
                r["heading"] = math.degrees(math.atan2(cy - r["y"], cx - r["x"]))

        points = _spawn_points(config)
        center_x = (WORLD_X_MIN + WORLD_X_MAX) / 2.0
        placed_obj = payload.get("objective")
        if placed_obj:
            snapped = _snap_placeable(config, (float(placed_obj[0]), float(placed_obj[1])))
            if snapped is None:
                raise ValueError("목표를 통행 가능 지점으로 옮길 수 없다")
            objective = snapped
        else:
            side = [p for p in points if p[0] < center_x] if mission == "hold_objective" else [
                p for p in points if p[0] >= center_x
            ]
            objective = tuple(float(v) for v in side[int(rng.integers(0, len(side)))])

        session = Session(
            map_name=map_name,
            config=config,
            mission_type=mission_type,
            objective=objective,
            duration_sec=duration,
            horizon=self.horizon,
        )
        root_id = uuid.uuid4().hex[:8]
        # 초기 적 위치는 지휘관이 안다는 전제라 belief = 실제로 시작한다.
        for row in rows:
            if row["id"] >= 200:
                session.red_belief[int(row["id"])] = {**row, "observed": True, "last_seen": 0.0}
        seeded = [
            {**r, "observed": True, "last_seen": 0.0} if r["id"] >= 200 else dict(r) for r in rows
        ]
        session.nodes[root_id] = TreeNode(
            node_id=root_id, parent_id=None, time_sec=0.0, unit_rows=seeded, true_rows=rows,
            red_belief={k: dict(v) for k, v in session.red_belief.items()},
        )
        session.current_id = root_id
        session_id = uuid.uuid4().hex[:8]
        self.sessions[session_id] = session
        return {"session": session_id, **self.session_view(session_id)}

    def random_placement(self, payload: dict[str, Any]) -> dict[str, Any]:
        """지휘관이 직접 안 찍을 때 쓸 무작위 배치를 만들어 준다."""
        map_name = str(payload.get("map") or next(iter(self.maps)))
        config = self.maps[map_name]
        real_map = config.get("real_map", {})
        if isinstance(real_map, dict) and real_map.get("unit_radius_units") is not None:
            set_path_pad(path_pad_for_unit_radius(float(real_map["unit_radius_units"])))
        blue = max(1, int(payload.get("blue", 5)))
        red = max(1, int(payload.get("red", 7)))
        mission = str(payload.get("mission", "destroy_all"))
        rng = np.random.default_rng(int(payload.get("seed", 0)) or None)

        rows = _initial_rows(config, blue_count=blue, red_count=red, rng=rng)
        points = _spawn_points(config)
        center_x = (WORLD_X_MIN + WORLD_X_MAX) / 2.0
        # 거점 방어는 아군이 지킬 거점이므로 아군 진영에, 나머지는 적 진영에 둔다.
        side = [p for p in points if p[0] < center_x] if mission == "hold_objective" else [
            p for p in points if p[0] >= center_x
        ]
        objective = [float(v) for v in side[int(rng.integers(0, len(side)))]]
        return {
            "blue_positions": [[r["x"], r["y"]] for r in rows if r["id"] < 200],
            "red_positions": [[r["x"], r["y"]] for r in rows if r["id"] >= 200],
            "objective": objective,
        }

    def session_view(self, session_id: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        node = session.current
        return {
            "map": session.map_name,
            "mission": MISSION_TYPE_NAMES[session.mission_type],
            "objective": list(session.objective),
            "duration": session.duration_sec,
            "time": node.time_sec,
            "units": node.unit_rows,
            "true_units": node.true_rows or node.unit_rows,
            # 확정된 직전 구간의 실제 경로(DEVS). 후보 미리보기와 달리 예측이 아니다.
            "true_path": node.true_path,
            "obstacles": session.config.get("obstacles", []),
            "building_polygons": session.config.get("building_polygons", []),
            "world": [WORLD_X_MIN, WORLD_Y_MIN, WORLD_X_MAX, WORLD_Y_MAX],
            "real_map": session.config.get("real_map", {}),
            "tree": [
                {
                    "id": n.node_id,
                    "parent": n.parent_id,
                    "time": n.time_sec,
                    "label": n.chosen_label,
                    "current": n.node_id == session.current_id,
                }
                for n in session.nodes.values()
            ],
        }

    def candidates_view(self, session_id: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        seed = int(session.current.time_sec) * 7919 + len(session.nodes)
        return build_archive(
            session, candidates=self.candidates, seed=seed, device=self.device,
            model=self.model if self.archive_backend == "model" else None,
            model_config=self.model_config,
            value_head=self.value_head,
            iterations=self.archive_iterations,
            lens_heads=self.wm2_lens_heads,
            wm2_rollout=self.wm2_rollout if self.archive_backend == "wm2" else None,
        )

    def recommend(
        self, session_id: str, *, candidates: int | None = None, lens: str = "score"
    ) -> dict[str, Any]:
        """현재 지점에서 매 결심마다 최고 점수 후보를 이어붙여 끝까지 전개한다.

        지휘관에게 먼저 보여줄 기준안이다. 후보를 전부 끝까지 굴리는 게 아니라
        6스텝마다 하나를 골라 이어붙이므로 비용이 결심 횟수에 비례한다.
        """
        session = self.sessions[session_id]
        node = session.current
        # belief(계획 입력)와 실제 상태(DEVS 시작점)를 나눠서 들고 간다. 지휘관은
        # 미관측 적의 진짜 위치를 모르므로, DEVS 결과를 그대로 다음 결심의 입력으로
        # 쓰면 알 수 없는 정보가 계획에 새어 들어간다. select()가 _update_belief로
        # 하는 것과 같은 처리를 여기서도 해야 한다.
        rows = [dict(r) for r in node.unit_rows]
        true_state = [dict(r) for r in (node.true_rows or node.unit_rows)]
        # 이 함수는 가정 전개라 세션의 belief를 건드리면 안 된다. 끝나면 되돌린다.
        saved_belief = {k: dict(v) for k, v in session.red_belief.items()}
        # 추천 전개가 세션의 후보 아카이브(pending)를 밟으면, 화면에 이미 떠 있는
        # 전개안 선택이 "후보가 없다"로 죽는다 (관점 탭 전환 → 추천 재계산 순서에서
        # 실제로 밟힘). 끝나면 원래 pending을 복원한다.
        saved_pending = session.pending
        time_sec = node.time_sec
        # **고를 때는 예측, 보여줄 때는 실측**이다.
        #
        # 후보 채점은 월드모델로 한다 — 그게 planning이고, 빨라서 후보를 많이 볼 수 있다.
        # 하지만 화면에 그리고 다음 구간으로 이어붙이는 것은 고른 plan 하나를 DEVS로
        # 다시 굴린 결과다. 예측을 그대로 이어붙이면 60초 동안 예측 위에 예측을 10번
        # 쌓게 되고, 실측(2026-08-11)에서 모델은 후보 간 RED 위치 변화를 실제 0.31m
        # 대비 3.05m로, HP 변화를 4.13 대비 17.06으로 지어낸다. 일어나지 않는 적
        # 반응을 지휘관에게 보여주게 된다.
        #
        # 비용은 구간당 DEVS rollout 1회다. 후보 전부를 DEVS로 굴리는 것보다 훨씬 싸다.
        #
        # 추천은 별도의 소형 탐색이 아니다 — 본 아카이브와 같은 CEM 탐색 예산으로
        # 아카이빙된 셀들 중 관점(lens) 최고 value를 결심마다 이어붙인다. 첫 구간은
        # 지휘관 화면에 이미 떠 있는 그 아카이브를 그대로 재사용한다.
        use_model = self.model if self.recommend_backend == "model" else None
        num = candidates or self.candidates
        reuse_entries = dict(saved_pending.get("archive") or {})
        prev_tail = None   # 다음 결심 윈도우의 관측 이력 (belief 프레임 꼬리)
        # 가정 전개용 RED 두뇌 상태 — 세션 상태를 복사해 쓰고 되돌리지 않는다
        chain_red = {k: dict(v) for k, v in session.red_rollout_states.items()}

        frames: list[dict[str, Any]] = [
            {"time": time_sec, "units": [dict(r) for r in rows], "fire": []}
        ]
        picks: list[dict[str, Any]] = []
        while time_sec + session.horizon <= session.duration_sec:
            alive_blue = sum(1 for r in rows if r["id"] < 200 and r["hp"] > 0.0)
            alive_red = sum(1 for r in rows if r["id"] >= 200 and r["hp"] > 0.0)
            if alive_blue == 0 or alive_red == 0:
                break
            probe = TreeNode(node_id="probe", parent_id=None, time_sec=time_sec, unit_rows=rows)
            if prev_tail:
                probe.belief_tail = prev_tail
            if reuse_entries:
                # 첫 구간: 지휘관이 보고 있는 그 아카이브에서 고른다 (재탐색 없음)
                entries, reuse_entries = reuse_entries, {}
                cells = _cells_from_archive(entries)
            else:
                saved_id, saved_nodes = session.current_id, session.nodes
                session.nodes = {**saved_nodes, "probe": probe}
                session.current_id = "probe"
                try:
                    archive = build_archive(
                        session, candidates=num, seed=int(time_sec) * 7919 + len(picks),
                        device=self.device, model=use_model, model_config=self.model_config,
                        value_head=self.value_head, iterations=self.archive_iterations,
                        lens_heads=self.wm2_lens_heads,
                        wm2_rollout=self.wm2_rollout if self.archive_backend == "wm2" else None,
                    )
                    # cells에는 plan이 없다(표시용 요약만 담는다). DEVS로 다시 굴리려면
                    # plan이 필요하므로 pending이 지워지기 전에 집어 둔다.
                    entries = dict(session.pending.get("archive") or {})
                finally:
                    session.nodes, session.current_id = saved_nodes, saved_id
                    session.pending = saved_pending
                cells = archive.get("cells") or []
            if not cells:
                break
            # 관점(lens)별 추천: 안전형/득점형이 각자의 채점으로 전개를 고른다.
            lens_cells = [c for c in cells if c.get("lens", "score") == lens] or cells
            best = max(lens_cells, key=lambda c: c["score"])

            # 고른 plan 하나를 DEVS로 굴려 실제 전개를 얻는다. probe의 true_rows를
            # 직전 구간의 DEVS 결과로 두므로 구간마다 실측으로 접지된다.
            entry = entries.get(
                (best["engage_bin"], best["spread_bin"], best.get("lens", "score"))
            )
            probe.true_rows = true_state
            if entry is None:
                true_rows, true_path = [dict(r) for r in best["rows"]], best["path"]
            else:
                true_rows, true_path = self._advance_true(
                    session, probe, entry, red_states=chain_red
                )

            # 관측된 것은 실측, 미관측 RED는 같은 plan의 월드모델 예측으로 합친다.
            belief_path = _merge_belief_path(
                true_path, best["path"], session.obstacles, rows, time_sec
            )
            for step, frame in enumerate(belief_path):
                frames.append(
                    {
                        "time": time_sec + step + 1,
                        "units": [dict(r) for r in frame["units"]],
                        "fire": frame["fire"],
                    }
                )
            # 요약도 예측이 아니라 실제 결과로 낸다. 지휘관이 읽는 "끝나면 몇 대 몇"이
            # 화면의 궤적과 어긋나면 안 된다.
            picks.append(
                {
                    "time": time_sec,
                    "label": best["label"],
                    "score": best["score"],
                    "blue_alive": sum(1 for r in true_rows if r["id"] < 200 and r["hp"] > 0.0),
                    "red_alive": sum(1 for r in true_rows if r["id"] >= 200 and r["hp"] > 0.0),
                    "blue_hp": sum(max(r["hp"], 0.0) for r in true_rows if r["id"] < 200),
                    "red_hp": sum(max(r["hp"], 0.0) for r in true_rows if r["id"] >= 200),
                }
            )
            # 화면에는 실제 전개를 그리되, 다음 결심의 계획 입력은 belief로 만든다.
            true_state = [dict(r) for r in true_rows]
            # 다음 결심의 계획 입력. 화면과 같은 belief여야 지휘관이 본 것과 계획이 맞는다.
            rows = [dict(r) for r in belief_path[-1]["units"]] if belief_path else true_state
            # 다음 윈도우의 관측 이력 — belief 프레임 꼬리 (미관측 RED는 예측 위치가 anchor)
            prev_tail = [
                [dict(u) for u in frame["units"]] for frame in belief_path[-2:]
            ] if belief_path else None
            time_sec += session.horizon
        session.red_belief = saved_belief
        return {
            "frames": frames, "picks": picks,
            "objective": list(session.objective), "lens": lens,
        }

    def select(
        self, session_id: str, engage_bin: int, spread_bin: int, lens: str = "score"
    ) -> dict[str, Any]:
        session = self.sessions[session_id]
        archive = session.pending.get("archive") or {}
        entry = archive.get((engage_bin, spread_bin, lens))
        if entry is None:
            raise ValueError("선택한 셀에 후보가 없다")
        parent = session.current
        # 실제 전개는 항상 DEVS로 굴린다. 후보 탐색은 근사여도 되지만 일어난 일은
        # 진짜여야 오차가 노드마다 누적되지 않는다.
        true_rows, true_path = self._advance_true(session, parent, entry)
        time_sec = parent.time_sec + session.horizon
        belief_rows = _update_belief(session, true_rows, entry["rows"], time_sec)
        node_id = uuid.uuid4().hex[:8]
        node = TreeNode(
            node_id=node_id,
            parent_id=parent.node_id,
            time_sec=time_sec,
            unit_rows=belief_rows,
            true_rows=true_rows,
            chosen_label=(
                (f"{LENS_LABELS[lens]} · " if self.wm2_lens_heads else "")
                + f"{ENGAGE_LABELS[engage_bin]} · {SPREAD_LABELS[spread_bin]}"
            ),
            red_belief={k: dict(v) for k, v in session.red_belief.items()},
            true_path=true_path,
        )
        parent.children.append(node_id)
        session.nodes[node_id] = node
        session.current_id = node_id
        session.pending = {}
        # 다음 결심 윈도우의 관측 이력 — 관측은 실측, 미관측 RED는 예측으로 병합한
        # belief 궤적의 꼬리 (recommend의 표시 규약과 동일)
        merged = _merge_belief_path(
            true_path, entry.get("path") or [], session.obstacles,
            parent.unit_rows, parent.time_sec,
        )
        node.belief_tail = [
            [dict(u) for u in frame["units"]] for frame in merged[-2:]
        ]
        return self.session_view(session_id)

    def _advance_true(
        self, session: Session, parent: TreeNode, entry: dict[str, Any],
        red_states: dict | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """선택된 plan 하나만 실제 DEVS로 굴려 진짜 다음 상태와 그 경로를 만든다.

        rollout_plans_with_devs는 horizon 전 프레임을 돌려준다. 마지막 프레임만 쓰고
        버리면 확정된 구간마저 예측으로 보여주게 되므로 경로도 함께 만들어 둔다.
        """
        plan = entry.get("plan")
        if plan is None:
            return entry["rows"], entry.get("path") or []
        base_rows = parent.true_rows or parent.unit_rows
        batch = _slot_batch(session, TreeNode("t", None, parent.time_sec, base_rows))
        snapshot = snapshot_from_slot_rows(
            unit_rows=base_rows,
            obstacles=session.obstacles,
            base_time_sec=parent.time_sec,
            episode_duration_sec=session.duration_sec,
            objective=session.objective,
            mission_type=session.mission_type,
        )
        features = rollout_plans_with_devs(
            plans=plan,
            snapshot=snapshot,
            seed=int(parent.time_sec) * 7919,
            device=self.device,
            blue_max_step=1.0,   # 본게임 물리 (전원 1.0) — BLUE 1.5 과속이 RED 무력화 원인
            red_policy_states=(
                session.red_rollout_states if red_states is None else red_states
            ),
        )
        frames = features.detach().cpu().numpy()[0]
        true_path = _build_path(
            batch, frames, plan, 0, parent.time_sec, session.obstacles,
            belief_start=parent.unit_rows,
        )
        return true_path[-1]["units"], true_path

    def goto(self, session_id: str, node_id: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        if node_id not in session.nodes:
            raise ValueError("없는 노드")
        session.current_id = node_id
        session.pending = {}
        # 그 시점에 알고 있던 것만 남긴다. 안 되돌리면 나중 결심에서 관측한 위치가
        # 과거 노드에도 그대로 보인다.
        session.red_belief = {k: dict(v) for k, v in session.nodes[node_id].red_belief.items()}
        return self.session_view(session_id)


class SessionRequest(BaseModel):
    """작전 개시 요청. 배치를 안 주면 무작위로 채운다."""

    map: str | None = None
    mission: str = "destroy_all"
    blue: int = Field(5, ge=1, le=10)
    red: int = Field(7, ge=1, le=10)
    duration: float = Field(60.0, gt=0)
    seed: int = 0
    blue_positions: list[tuple[float, float]] | None = None
    red_positions: list[tuple[float, float]] | None = None
    objective: tuple[float, float] | None = None


class RandomRequest(BaseModel):
    """무작위 배치 요청."""

    map: str | None = None
    mission: str = "destroy_all"
    blue: int = Field(5, ge=1, le=10)
    red: int = Field(7, ge=1, le=10)
    seed: int = 0


class SelectRequest(BaseModel):
    """아카이브 셀 선택."""

    session: str
    engage_bin: int = Field(ge=0)
    spread_bin: int = Field(ge=0)
    lens: str = "score"


class SessionOnly(BaseModel):
    session: str
    lens: str = "score"


class GotoRequest(BaseModel):
    session: str
    node: str


def create_app(state: PlatformState) -> FastAPI:
    """플랫폼 API. 무거운 계산은 threadpool로 빼 다른 요청을 막지 않는다."""
    app = FastAPI(title="지휘관 시뮬레이션 플랫폼", docs_url="/docs")

    def _guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"없는 세션/노드: {error}") from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE_HTML

    @app.get("/api/maps")
    async def maps() -> dict[str, Any]:
        return {
            "maps": [
                {
                    "name": name,
                    "real_map": config.get("real_map", {}),
                    "obstacles": config.get("obstacles", []),
                    "building_polygons": config.get("building_polygons", []),
                }
                for name, config in sorted(state.maps.items())
            ],
            "naver_client_id": state.naver_client_id,
        }

    @app.post("/api/session")
    async def create(request: SessionRequest) -> dict[str, Any]:
        return await run_in_threadpool(_guard, state.create_session, request.model_dump())

    @app.post("/api/random")
    async def random_placement(request: RandomRequest) -> dict[str, Any]:
        return await run_in_threadpool(_guard, state.random_placement, request.model_dump())

    @app.get("/api/session/{session_id}")
    async def view(session_id: str) -> dict[str, Any]:
        return _guard(state.session_view, session_id)

    @app.get("/api/candidates/{session_id}")
    async def candidates(session_id: str) -> dict[str, Any]:
        return await run_in_threadpool(_guard, state.candidates_view, session_id)

    @app.post("/api/recommend")
    async def recommend(request: SessionOnly) -> dict[str, Any]:
        return await run_in_threadpool(
            _guard, lambda: state.recommend(request.session, lens=request.lens)
        )

    @app.post("/api/select")
    async def select(request: SelectRequest) -> dict[str, Any]:
        return _guard(
            state.select, request.session, request.engage_bin, request.spread_bin,
            request.lens,
        )

    @app.post("/api/goto")
    async def goto(request: GotoRequest) -> dict[str, Any]:
        return _guard(state.goto, request.session, request.node)

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="지휘관 시뮬레이션 플랫폼")
    parser.add_argument("--maps-root", type=Path, default=Path("output/maps"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--archive-backend",
        choices=("devs", "model", "wm2"),
        default="devs",
        help="아카이브 후보 rollout 방식. model=구 JEPA, wm2=run13 상상(빠름, --wm2-checkpoint 필요)",
    )
    parser.add_argument(
        "--recommend-backend",
        choices=("devs", "model"),
        default="devs",
        help="추천 시나리오 rollout 방식. devs는 물리가 정확(느림), model은 빠르지만 건물 통과가 섞인다",
    )
    parser.add_argument(
        "--recommend-candidates",
        type=int,
        default=24,
        help="추천 시나리오 결심마다 뽑을 후보 수. devs 백엔드에서는 이 값이 속도를 좌우한다",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        default=None,
        help="지정하면 후보 rollout을 DEVS 대신 월드모델 예측으로 한다(훨씬 빠름, 근사)",
    )
    parser.add_argument(
        "--value-head-checkpoint",
        type=Path,
        default=None,
        help="지정하면 아카이브 채점을 휴리스틱 evaluator 대신 value head로 한다",
    )
    parser.add_argument(
        "--archive-iterations",
        type=int,
        default=1,
        help="CEM 반복 횟수. 매 반복의 후보를 전부 아카이빙하므로 늘리면 셀이 더 찬다",
    )
    parser.add_argument(
        "--wm2-safe-value", type=Path, default=None,
        help="wm2 안전형(β=1) value head — 실측 입력판(wm2_value_rtgs_real.pt)을 줄 것",
    )
    parser.add_argument(
        "--wm2-score-value", type=Path, default=None,
        help="wm2 득점형(β=0) value head — 실측 입력판(wm2_value_rtg_real.pt)을 줄 것",
    )
    parser.add_argument(
        "--wm2-checkpoint", type=Path, default=None,
        help="wm2 월드모델(run13) — --archive-backend wm2의 상상 롤아웃에 사용",
    )
    args = parser.parse_args(argv)

    state = PlatformState(
        args.maps_root,
        device=torch.device(args.device),
        candidates=args.candidates,
        horizon=args.horizon,
        world_model_checkpoint=args.world_model_checkpoint,
        archive_backend=args.archive_backend,
        recommend_backend=args.recommend_backend,
        recommend_candidates=args.recommend_candidates,
        value_head_checkpoint=args.value_head_checkpoint,
        archive_iterations=args.archive_iterations,
        wm2_safe_value=args.wm2_safe_value,
        wm2_score_value=args.wm2_score_value,
        wm2_checkpoint=args.wm2_checkpoint,
    )
    print(f"지휘관 플랫폼: http://{args.host}:{args.port}/  (API 문서 /docs)")
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
