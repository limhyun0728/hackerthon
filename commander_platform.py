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

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hackerthon.terrain import (
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
    cell_of,
    component_points,
    largest_free_component,
    path_pad_for_unit_radius,
    set_path_pad,
    snap_to_component,
)
from hackerthon.worldmodel.actions import ActionType
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    ObservedActionWindow,
    build_initial_distribution,
    rollout_with_world_model,
    sample_future_action_plans,
    score_future_features_torch,
)
from hackerthon.worldmodel.object_slot_attention import (
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
)
from hackerthon.worldmodel.actions import ACTION_DIM
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.slots import (
    MAX_AMMO,
    MAX_HP,
    MISSION_TYPE_BY_NAME,
    MISSION_TYPE_NAMES,
    ObjectType,
    TeamId,
    build_slot_batch,
)

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
SPREAD_EDGES = (0.0, 2.0, 5.0, 10.0, 15.0)
ENGAGE_LABELS = ("순수기동", "산발사격", "교전", "적극교전", "전력사격")
SPREAD_LABELS = ("밀집", "근접", "분진", "산개", "광역분산")


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
    unit_rows: list[dict[str, Any]]
    chosen_label: str | None = None
    children: list[str] = field(default_factory=list)


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
    # 후보 캐시. 지휘관이 셀을 고르면 그 plan의 결과 state를 그대로 쓴다.
    pending: dict[str, Any] = field(default_factory=dict)

    @property
    def current(self) -> TreeNode:
        return self.nodes[self.current_id]

    @property
    def obstacles(self) -> list:
        return self.config.get("obstacles", [])


def _initial_rows(
    config: dict[str, Any],
    *,
    blue_count: int,
    red_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """자유 공간에서 진영을 나눠 초기 부대를 배치한다."""
    obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
    component = largest_free_component(obstacles)
    if not component:
        raise ValueError("이동 가능한 자유 공간이 없다")
    points = component_points(component)
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
    """건물 안이면 가장 가까운 이동 가능 지점으로 밀어준다."""
    obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
    if not obstacles:
        return point
    snapped = snap_to_component(point, largest_free_component(obstacles))
    return None if snapped is None else (float(snapped[0]), float(snapped[1]))


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


def _formation_spread(rows: list[dict[str, Any]]) -> float:
    """생존 BLUE 유닛 간 평균 쌍거리."""
    pts = [(r["x"], r["y"]) for r in rows if r["id"] < 200 and r["hp"] > 0.0]
    if len(pts) < 2:
        return float("nan")
    return float(
        np.mean([float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in itertools.combinations(pts, 2)])
    )


def build_archive(
    session: Session,
    *,
    candidates: int,
    seed: int,
    device: torch.device,
    model=None,
    model_config=None,
) -> dict[str, Any]:
    """후보를 뽑아 태세 축 아카이브에 배치하고 셀별 elite를 만든다."""
    node = session.current
    batch = _slot_batch(session, node)
    alive_blue = sum(1 for r in node.unit_rows if r["id"] < 200 and r["hp"] > 0.0)
    alive_red = sum(1 for r in node.unit_rows if r["id"] >= 200 and r["hp"] > 0.0)
    if alive_blue == 0 or alive_red == 0:
        return {"cells": [], "finished": True, "alive_blue": alive_blue, "alive_red": alive_red}

    cem_config = CEMConfig(
        num_candidates=candidates,
        num_elites=max(2, candidates // 8),
        num_iterations=1,
        future_horizon=session.horizon,
        seed=seed,
        min_action_probability=0.0,
    )
    distribution = build_initial_distribution(batch, cem_config, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    plans = sample_future_action_plans(
        distribution=distribution, current_batch=batch, config=cem_config, generator=generator, device=device
    )
    snapshot = snapshot_from_slot_rows(
        unit_rows=node.unit_rows,
        obstacles=session.obstacles,
        base_time_sec=node.time_sec,
        episode_duration_sec=session.duration_sec,
        objective=session.objective,
        mission_type=session.mission_type,
    )
    if model is not None:
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
        features = torch.cat(chunks, dim=0)
    else:
        features = rollout_plans_with_devs(plans=plans, snapshot=snapshot, seed=seed, device=device)
    scores = score_future_features_torch(current_batch=batch, future_features=features).detach().cpu().numpy()
    features_np = features.detach().cpu().numpy()

    types = plans.action_type_ids.detach().cpu().numpy()
    issued = plans.issued_mask.detach().cpu().numpy()

    archive: dict[tuple[int, int], dict[str, Any]] = {}
    for cand in range(features_np.shape[0]):
        engage = float(((types[cand] == int(ActionType.ENGAGE)) & issued[cand]).sum()) / float(
            max(alive_blue * session.horizon, 1)
        )
        rows = _rows_from_features(batch, features_np[cand, -1], node.time_sec + session.horizon)
        spread = _formation_spread(rows)
        e_bin = _bin_index(engage, ENGAGE_EDGES)
        s_bin = 0 if np.isnan(spread) else _bin_index(spread, SPREAD_EDGES)
        key = (e_bin, s_bin)
        if key not in archive or scores[cand] > archive[key]["score"]:
            archive[key] = {
                "score": float(scores[cand]),
                "candidate": int(cand),
                "engage": engage,
                "spread": 0.0 if np.isnan(spread) else float(spread),
                "rows": rows,
                "path": [
                    _rows_from_features(batch, features_np[cand, step], node.time_sec + step + 1)
                    for step in range(features_np.shape[1])
                ],
            }

    session.pending = {"archive": archive}
    cells = []
    for (e_bin, s_bin), entry in sorted(archive.items()):
        blue_hp = sum(r["hp"] for r in entry["rows"] if r["id"] < 200)
        red_hp = sum(r["hp"] for r in entry["rows"] if r["id"] >= 200)
        cells.append(
            {
                "engage_bin": e_bin,
                "spread_bin": s_bin,
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
    return {
        "cells": cells,
        "finished": False,
        "alive_blue": alive_blue,
        "alive_red": alive_red,
        "engage_labels": list(ENGAGE_LABELS),
        "spread_labels": list(SPREAD_LABELS),
    }


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

        obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
        points = component_points(largest_free_component(obstacles))
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
        session.nodes[root_id] = TreeNode(node_id=root_id, parent_id=None, time_sec=0.0, unit_rows=rows)
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
        obstacles = [tuple(float(v) for v in rect) for rect in config.get("obstacles", [])]
        points = component_points(largest_free_component(obstacles))
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
            model=self.model, model_config=self.model_config,
        )

    def recommend(self, session_id: str, *, candidates: int | None = None) -> dict[str, Any]:
        """현재 지점에서 매 결심마다 최고 점수 후보를 이어붙여 끝까지 전개한다.

        지휘관에게 먼저 보여줄 기준안이다. 후보를 전부 끝까지 굴리는 게 아니라
        6스텝마다 하나를 골라 이어붙이므로 비용이 결심 횟수에 비례한다.
        """
        session = self.sessions[session_id]
        node = session.current
        rows = [dict(r) for r in node.unit_rows]
        time_sec = node.time_sec
        num = candidates or max(32, self.candidates // 4)

        frames: list[dict[str, Any]] = [{"time": time_sec, "units": [dict(r) for r in rows]}]
        picks: list[dict[str, Any]] = []
        while time_sec + session.horizon <= session.duration_sec:
            alive_blue = sum(1 for r in rows if r["id"] < 200 and r["hp"] > 0.0)
            alive_red = sum(1 for r in rows if r["id"] >= 200 and r["hp"] > 0.0)
            if alive_blue == 0 or alive_red == 0:
                break
            probe = TreeNode(node_id="probe", parent_id=None, time_sec=time_sec, unit_rows=rows)
            saved_id, saved_nodes = session.current_id, session.nodes
            session.nodes = {**saved_nodes, "probe": probe}
            session.current_id = "probe"
            try:
                archive = build_archive(
                    session, candidates=num, seed=int(time_sec) * 7919 + len(picks),
                    device=self.device, model=self.model, model_config=self.model_config,
                )
            finally:
                session.nodes, session.current_id = saved_nodes, saved_id
                session.pending = {}
            cells = archive.get("cells") or []
            if not cells:
                break
            best = max(cells, key=lambda c: c["score"])
            for step, frame_rows in enumerate(best["path"]):
                frames.append({"time": time_sec + step + 1, "units": [dict(r) for r in frame_rows]})
            picks.append(
                {
                    "time": time_sec,
                    "label": best["label"],
                    "score": best["score"],
                    "blue_alive": best["blue_alive"],
                    "red_alive": best["red_alive"],
                }
            )
            rows = [dict(r) for r in best["rows"]]
            time_sec += session.horizon
        return {"frames": frames, "picks": picks, "objective": list(session.objective)}

    def select(self, session_id: str, engage_bin: int, spread_bin: int) -> dict[str, Any]:
        session = self.sessions[session_id]
        archive = session.pending.get("archive") or {}
        entry = archive.get((engage_bin, spread_bin))
        if entry is None:
            raise ValueError("선택한 셀에 후보가 없다")
        parent = session.current
        node_id = uuid.uuid4().hex[:8]
        node = TreeNode(
            node_id=node_id,
            parent_id=parent.node_id,
            time_sec=parent.time_sec + session.horizon,
            unit_rows=entry["rows"],
            chosen_label=f"{ENGAGE_LABELS[engage_bin]} · {SPREAD_LABELS[spread_bin]}",
        )
        parent.children.append(node_id)
        session.nodes[node_id] = node
        session.current_id = node_id
        session.pending = {}
        return self.session_view(session_id)

    def goto(self, session_id: str, node_id: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        if node_id not in session.nodes:
            raise ValueError("없는 노드")
        session.current_id = node_id
        session.pending = {}
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


class SessionOnly(BaseModel):
    session: str


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
        return await run_in_threadpool(_guard, state.recommend, request.session)

    @app.post("/api/select")
    async def select(request: SelectRequest) -> dict[str, Any]:
        return _guard(state.select, request.session, request.engage_bin, request.spread_bin)

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
        "--world-model-checkpoint",
        type=Path,
        default=None,
        help="지정하면 후보 rollout을 DEVS 대신 월드모델 예측으로 한다(훨씬 빠름, 근사)",
    )
    args = parser.parse_args(argv)

    state = PlatformState(
        args.maps_root,
        device=torch.device(args.device),
        candidates=args.candidates,
        horizon=args.horizon,
        world_model_checkpoint=args.world_model_checkpoint,
    )
    print(f"지휘관 플랫폼: http://{args.host}:{args.port}/  (API 문서 /docs)")
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
