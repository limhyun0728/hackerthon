"""wm2 CEM (설계 11절).

한 계획 tick의 절차:
  후보 C개 샘플 (목적 지향 제안 + 전 스텝 ENGAGE 사실 마스크)
  → 월드모델 상상 → 물리 클램프 → 채점 (score.py) + 상상 feasibility 안전망
  → elite로 분포 갱신 ×iterations → 최고 후보 반환

휴리스틱 원칙 (합의): 제안 편향은 탐색 효율 장치일 뿐 채점에 안 낀다. mask는
"불가능 제거"만 한다 — step k의 ENGAGE는 자기 계획 궤적의 투영 위치(current)에서
현재 RED 배치로 거리+LOS를 검사해, 불가능하면 확률 0 + 표적은 가능한 RED로 한정
(실행층 _feasible_target의 재표적 규약 미러). 2026-08-14: 채점측 통짜 거부만 쓰면
원거리 상태에서 후보 전멸 → 마스크된 후보가 best로 실행되는 붕괴가 있어(-1e6 로그),
샘플링 차단이 1차, 채점측 마스크(_infeasible_engage)는 투영-상상 불일치 안전망.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..config import CEMConfig
from ..data.windows import Window
from ..model.features import (
    ACTION_DIM,
    ACTION_ENGAGE,
    ACTION_MOVE,
    ACTION_STOP,
    ACTION_TURN,
    MAX_AMMO,
    MAX_FIRE_RANGE_UNITS,
    MAX_HP,
    MAX_MOVE_PER_STEP,
    NUM_ACTION_TYPES,
    denorm_x,
    denorm_y,
    norm_x,
    norm_y,
)
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor
from ..model.rollout import assemble_hp, assemble_positions, clamp_physics
from ..value.head import WM2ValueHead
from .score import score_candidates

HORIZON = 6
ACTION_PRIOR = np.asarray([0.15, 0.40, 0.35, 0.10])   # STOP/MOVE/ENGAGE/TURN
GOAL_WEIGHTS = (0.45, 0.35, 0.20)                      # 목표 / 최근접 적 / 무작위 방향
GOAL_TEMPERATURE_RANGE = (0.1, 0.5)
STD_FLOOR = 0.3


@dataclass
class PlanCandidates:
    """(C, H, U_blue) 계획 묶음 — counterfactual.PlanSpec과 같은 배열 규약."""

    action_type_ids: np.ndarray
    move_xy: np.ndarray          # (C, H, U, 2) **월드 좌표**
    target_slots: np.ndarray     # (C, H, U) RED 슬롯 index (-1 = 없음)
    theta: np.ndarray
    issued: np.ndarray


@dataclass
class CEMResult:
    best_index: int
    candidates: PlanCandidates
    scores: np.ndarray
    gain: np.ndarray
    value: np.ndarray


def _current_state(window: Window):
    """h2(현재) 프레임의 월드 좌표·HP·탄약."""
    layout = window.layout
    feats = window.unit_features[2]                     # h2
    xs = np.array([denorm_x(v) for v in feats[:, 3]])
    ys = np.array([denorm_y(v) for v in feats[:, 4]])
    return np.stack([xs, ys], axis=-1), feats[:, 1] * MAX_HP, feats[:, 2] * MAX_AMMO


def _sample_candidates(
    window: Window,
    rng: np.random.Generator,
    config: CEMConfig,
    distribution: dict | None,
) -> PlanCandidates:
    layout = window.layout
    num_blue = layout.num_blue
    num_units = layout.num_units
    c = config.candidates
    positions, hp, _ = _current_state(window)
    blue_alive = hp[:num_blue] > 0.0
    red_alive = hp[num_blue:] > 0.0
    red_positions = positions[num_blue:]
    objective = np.asarray(layout.objective)

    types = np.zeros((c, HORIZON, num_blue), dtype=np.int64)
    move_xy = np.zeros((c, HORIZON, num_blue, 2), dtype=np.float32)
    target_slots = np.full((c, HORIZON, num_blue), -1, dtype=np.int64)
    theta = rng.uniform(-math.pi, math.pi, (c, HORIZON, num_blue)).astype(np.float32)
    issued = np.zeros((c, HORIZON, num_blue), dtype=bool)

    # 후보별 목적·온도 (탐색 편향 — 채점에는 안 낀다)
    goal_kind = rng.choice(3, size=c, p=GOAL_WEIGHTS)
    tau = rng.uniform(*GOAL_TEMPERATURE_RANGE, size=c)
    random_angle = rng.uniform(-math.pi, math.pi, size=c)

    alive_red_indices = np.nonzero(red_alive)[0]

    # 전 스텝 ENGAGE 사실 마스크: 자기 계획 궤적의 투영 위치(current)에서 현재 RED
    # 배치 기준 거리+LOS로 가능한 표적 집합. RED의 미래 이동은 모름 — 근사이며,
    # 투영-상상 불일치는 채점측 _infeasible_engage 안전망이 잡는다.
    rects = _decode_obstacle_rects(layout.terrain_features)
    feas_cache: dict[tuple[float, float], tuple[int, ...]] = {}

    def feasible_reds(pos) -> tuple[int, ...]:
        key = (round(float(pos[0]), 1), round(float(pos[1]), 1))
        hit = feas_cache.get(key)
        if hit is not None:
            return hit
        out = tuple(
            int(j) for j in alive_red_indices
            if np.linalg.norm(red_positions[j] - pos) <= MAX_FIRE_RANGE_UNITS
            and not _np_segment_blocked(pos, red_positions[j], rects)
        )
        feas_cache[key] = out
        return out

    for ci in range(c):
        for ui in range(num_blue):
            if not blue_alive[ui]:
                continue
            issued[ci, :, ui] = True
            # 목적 지점
            if goal_kind[ci] == 0:
                goal = objective
            elif goal_kind[ci] == 1 and alive_red_indices.size:
                dists = np.linalg.norm(red_positions[alive_red_indices] - positions[ui], axis=-1)
                goal = red_positions[alive_red_indices[int(np.argmin(dists))]]
            else:
                goal = positions[ui] + 30.0 * np.array(
                    [math.cos(random_angle[ci]), math.sin(random_angle[ci])]
                )
            current = positions[ui].copy()
            for step in range(HORIZON):
                feas = feasible_reds(current)
                if distribution is not None and distribution["counts"][step, ui] > 0:
                    probs = distribution["action_probs"][step, ui]
                else:
                    probs = ACTION_PRIOR
                if not feas:
                    probs = probs.copy(); probs[ACTION_ENGAGE] = 0.0; probs /= probs.sum()
                action = int(rng.choice(NUM_ACTION_TYPES, p=probs))
                types[ci, step, ui] = action
                if action == ACTION_MOVE:
                    if (
                        distribution is not None
                        and distribution["move_counts"][step, ui] > 1
                        and rng.random() < 0.7
                    ):
                        mean = distribution["move_mean"][step, ui]
                        std = np.maximum(distribution["move_std"][step, ui], STD_FLOOR)
                        dest = rng.normal(mean, std)
                    else:
                        direction = goal - current
                        norm = np.linalg.norm(direction)
                        base = math.atan2(direction[1], direction[0]) if norm > 1e-6 else rng.uniform(-math.pi, math.pi)
                        angle = base + rng.normal(0.0, tau[ci] * math.pi)
                        radius = rng.uniform(0.3, MAX_MOVE_PER_STEP)
                        dest = current + radius * np.array([math.cos(angle), math.sin(angle)])
                    dest = np.clip(dest, (-20.0, -15.0), (20.0, 10.0))
                    move_xy[ci, step, ui] = dest
                    current = dest
                elif action == ACTION_ENGAGE and feas:
                    feas_arr = np.asarray(feas)
                    if distribution is not None and distribution["target_counts"][step, ui] > 0:
                        target_probs = distribution["target_probs"][step, ui][feas_arr] + 1e-6
                        slot = int(rng.choice(feas_arr, p=target_probs / target_probs.sum()))
                    else:
                        slot = int(rng.choice(feas_arr))   # 가능 표적 중 균등 — elite가 조인다
                    target_slots[ci, step, ui] = slot
                elif action == ACTION_ENGAGE:
                    types[ci, step, ui] = ACTION_STOP
    return PlanCandidates(types, move_xy, target_slots, theta, issued)


def _refit(elites: PlanCandidates, num_blue: int, num_red: int) -> dict:
    e, h, _ = elites.action_type_ids.shape
    action_probs = np.zeros((h, num_blue, NUM_ACTION_TYPES))
    counts = np.zeros((h, num_blue))
    move_mean = np.zeros((h, num_blue, 2)); move_std = np.ones((h, num_blue, 2))
    move_counts = np.zeros((h, num_blue))
    target_probs = np.zeros((h, num_blue, num_red)); target_counts = np.zeros((h, num_blue))
    for step in range(h):
        for ui in range(num_blue):
            issued = elites.issued[:, step, ui]
            if not issued.any():
                continue
            types = elites.action_type_ids[issued, step, ui]
            hist = np.bincount(types, minlength=NUM_ACTION_TYPES).astype(np.float64)
            action_probs[step, ui] = (hist + 0.5) / (hist.sum() + 0.5 * NUM_ACTION_TYPES)
            counts[step, ui] = issued.sum()
            moves = elites.move_xy[issued & (elites.action_type_ids[:, step, ui] == ACTION_MOVE), step, ui]
            if len(moves) > 1:
                move_mean[step, ui] = moves.mean(axis=0)
                move_std[step, ui] = moves.std(axis=0)
                move_counts[step, ui] = len(moves)
            targets = elites.target_slots[issued & (elites.action_type_ids[:, step, ui] == ACTION_ENGAGE), step, ui]
            targets = targets[targets >= 0]
            if len(targets):
                thist = np.bincount(targets, minlength=num_red).astype(np.float64)
                target_probs[step, ui] = (thist + 0.2) / (thist.sum() + 0.2 * num_red)
                target_counts[step, ui] = len(targets)
    return {
        "action_probs": action_probs, "counts": counts,
        "move_mean": move_mean, "move_std": move_std, "move_counts": move_counts,
        "target_probs": target_probs, "target_counts": target_counts,
    }


def _decode_obstacle_rects(terrain_features: np.ndarray) -> np.ndarray:
    """terrain_features (T,9) → 장애물 사각형 (N,4) [xmin,ymin,xmax,ymax] 월드 좌표.

    windows._terrain_features의 인코딩([1, cx_n, cy_n, w_n, h_n, ...])을 역변환한다.
    """
    from ..model.features import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN

    rects = []
    for row in terrain_features:
        if row[0] < 0.5:
            continue
        cx, cy = denorm_x(float(row[1])), denorm_y(float(row[2]))
        w = float(row[3]) * (WORLD_X_MAX - WORLD_X_MIN)
        h = float(row[4]) * (WORLD_Y_MAX - WORLD_Y_MIN)
        rects.append([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
    return np.asarray(rects, dtype=np.float32).reshape(-1, 4)


def _np_segment_blocked(p, q, rects: np.ndarray) -> bool:
    """스칼라 선분판 slab 판정 — terrain.has_los의 미러 (샘플러용, rects (N,4))."""
    if rects.size == 0:
        return False
    p = np.asarray(p, dtype=np.float64)
    d = np.asarray(q, dtype=np.float64) - p
    tmin = np.zeros(len(rects)); tmax = np.ones(len(rects))
    for axis, (lo_i, hi_i) in enumerate(((0, 2), (1, 3))):
        lo, hi = rects[:, lo_i], rects[:, hi_i]
        if abs(d[axis]) < 1e-9:
            inside = (lo <= p[axis]) & (p[axis] <= hi)
            t1 = np.where(inside, -1e9, 1e9); t2 = np.where(inside, 1e9, -1e9)
        else:
            near = (lo - p[axis]) / d[axis]; far = (hi - p[axis]) / d[axis]
            t1 = np.minimum(near, far); t2 = np.maximum(near, far)
        tmin = np.maximum(tmin, t1); tmax = np.minimum(tmax, t2)
    return bool((tmin <= tmax).any())


def _segment_blocked(p: torch.Tensor, q: torch.Tensor, rects: torch.Tensor) -> torch.Tensor:
    """선분 p→q(..., 2)가 rects (N,4) 중 하나라도 지나면 True (terrain.has_los의 미러).

    slab 판정: 축별 진입/이탈 파라미터의 교집합이 [0,1]과 겹치면 교차.
    """
    if rects.numel() == 0:
        return torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)
    d = q - p
    eps = 1e-9
    tmin = torch.zeros_like(p[..., 0]).unsqueeze(-1)      # (..., 1) → 브로드캐스트 (..., N)
    tmax = torch.ones_like(p[..., 0]).unsqueeze(-1)
    for axis, (lo_i, hi_i) in enumerate(((0, 2), (1, 3))):
        pa = p[..., axis].unsqueeze(-1)
        da = d[..., axis].unsqueeze(-1)
        lo, hi = rects[:, lo_i], rects[:, hi_i]
        parallel = da.abs() < eps
        inside = (pa >= lo) & (pa <= hi)
        safe_da = torch.where(parallel, torch.ones_like(da), da)
        near = (lo - pa) / safe_da
        far = (hi - pa) / safe_da
        t1, t2 = torch.minimum(near, far), torch.maximum(near, far)
        # 평행축: 슬랩 안이면 (-inf,inf), 밖이면 빈 구간
        t1 = torch.where(parallel, torch.where(inside, torch.full_like(t1, -1e9), torch.full_like(t1, 1e9)), t1)
        t2 = torch.where(parallel, torch.where(inside, torch.full_like(t2, 1e9), torch.full_like(t2, -1e9)), t2)
        tmin = torch.maximum(tmin, t1)
        tmax = torch.minimum(tmax, t2)
    return (tmin <= tmax).any(dim=-1)


def _infeasible_engage(
    engage: torch.Tensor,        # (b, H, U_blue) bool — issued ENGAGE
    pos_start: torch.Tensor,     # (b, H, U, 2) 스텝 시작 시점 상상 위치
    hp_start: torch.Tensor,      # (b, H, U) 스텝 시작 시점 상상 HP
    num_blue: int,
    rects: torch.Tensor,
) -> torch.Tensor:
    """실행층 _feasible_target의 계획측 미러 (재표적 포함).

    스텝 시작 상태에서 사거리+LOS 안 생존 RED가 하나도 없는 ENGAGE(또는 상상 전사자의
    ENGAGE)가 든 후보 → True. 실행이 STOP으로 강등할 명령이므로 상상 점수도 무효.
    """
    shooter_pos = pos_start[:, :, :num_blue]
    shooter_alive = hp_start[:, :, :num_blue] > 0.0
    red_pos = pos_start[:, :, num_blue:]
    red_alive = hp_start[:, :, num_blue:] > 0.0
    if red_pos.shape[2] == 0:
        return engage.any(dim=(1, 2))
    diff = shooter_pos.unsqueeze(3) - red_pos.unsqueeze(2)          # (b,H,Ub,R,2)
    in_range = diff.norm(dim=-1) <= MAX_FIRE_RANGE_UNITS
    p = shooter_pos.unsqueeze(3).expand(-1, -1, -1, red_pos.shape[2], -1)
    q = red_pos.unsqueeze(2).expand(-1, -1, shooter_pos.shape[2], -1, -1)
    clear = ~_segment_blocked(p, q, rects)
    feasible_any = (red_alive.unsqueeze(2) & in_range & clear).any(dim=-1)   # (b,H,Ub)
    bad = engage & (~shooter_alive | ~feasible_any)
    return bad.any(dim=2).any(dim=1)


def _action_tokens(window: Window, plan: PlanCandidates) -> np.ndarray:
    """(C, 8, U_blue, ACTION_DIM). j=0,1 실측 명령, j=2..7 후보 계획 (계획된 명령이 토큰)."""
    c = plan.action_type_ids.shape[0]
    num_blue = window.layout.num_blue
    tokens = np.zeros((c, 8, num_blue, ACTION_DIM), dtype=np.float32)
    tokens[:, :2] = window.actions[:2]
    offset = 1 + NUM_ACTION_TYPES
    for ci in range(c):
        for step in range(HORIZON):
            for ui in range(num_blue):
                if not plan.issued[ci, step, ui]:
                    continue
                vec = tokens[ci, 2 + step, ui]
                vec[0] = 1.0
                action = plan.action_type_ids[ci, step, ui]
                vec[1 + action] = 1.0
                if action == ACTION_MOVE:
                    vec[offset] = norm_x(plan.move_xy[ci, step, ui, 0])
                    vec[offset + 1] = norm_y(plan.move_xy[ci, step, ui, 1])
                elif action == ACTION_ENGAGE and plan.target_slots[ci, step, ui] >= 0:
                    vec[offset + 2 + plan.target_slots[ci, step, ui]] = 1.0
                elif action == ACTION_TURN:
                    vec[offset + 12] = math.cos(plan.theta[ci, step, ui])
                    vec[offset + 13] = math.sin(plan.theta[ci, step, ui])
    return tokens


@torch.no_grad()
def plan(
    *,
    window: Window,
    model: WM2Predictor,
    heads: WM2Heads,
    value_head: WM2ValueHead | None,
    config: CEMConfig,
    device: torch.device,
    rng: np.random.Generator,
    lam: float = 1.0,
    chunk: int = 128,
) -> CEMResult:
    layout = window.layout
    num_blue, num_units = layout.num_blue, layout.num_units
    positions, hp, ammo = _current_state(window)
    anchor_feats = window.unit_features[0]
    anchor_xy = np.stack(
        [[denorm_x(v) for v in anchor_feats[:, 3]], [denorm_y(v) for v in anchor_feats[:, 4]]],
        axis=-1,
    )
    time_remaining = float(window.mission_features[2, 3]) - HORIZON / layout.duration_sec

    unit_t = torch.from_numpy(window.unit_features).to(device)
    terrain_t = torch.from_numpy(layout.terrain_features).to(device)
    mission_t = torch.from_numpy(window.mission_features).to(device)
    team_t = torch.from_numpy(np.asarray(layout.team_ids)).to(device)
    anchor_xy_t = torch.from_numpy(anchor_xy.astype(np.float32)).to(device)
    anchor_hp_t = torch.from_numpy((anchor_feats[:, 1] * MAX_HP).astype(np.float32)).to(device)
    anchor_ammo_t = torch.from_numpy((anchor_feats[:, 2] * MAX_AMMO).astype(np.float32)).to(device)
    current_pos_t = torch.from_numpy(positions.astype(np.float32)).to(device)
    current_hp_t = torch.from_numpy(hp.astype(np.float32)).to(device)

    rects_t = torch.from_numpy(_decode_obstacle_rects(layout.terrain_features)).to(device)

    distribution = None
    best = None
    for _ in range(config.iterations):
        candidates = _sample_candidates(window, rng, config, distribution)
        tokens = torch.from_numpy(_action_tokens(window, candidates)).to(device)
        c = tokens.shape[0]
        engage_t = torch.from_numpy(
            (candidates.action_type_ids == ACTION_ENGAGE) & candidates.issued
        ).to(device)
        scores = torch.zeros(c, device=device)
        gains = torch.zeros(c, device=device)
        values = torch.zeros(c, device=device)
        for start in range(0, c, chunk):
            end = min(start + chunk, c)
            b = end - start
            out = model(
                unit_features=unit_t.unsqueeze(0).expand(b, -1, -1, -1),
                terrain_features=terrain_t.unsqueeze(0).expand(b, -1, -1),
                mission_features=mission_t.unsqueeze(0).expand(b, -1, -1),
                actions=tokens[start:end],
                team_ids=team_t,
                masked_units=torch.zeros(b, num_units, dtype=torch.bool, device=device),
            )
            pred = heads(out["unit_tokens"], out["mission_tokens"])
            dpos_future = pred["dpos"][:, 2:]
            raw_positions = assemble_positions(anchor_xy_t.unsqueeze(0).expand(b, -1, -1), dpos_future)
            hp_pred = assemble_hp(anchor_hp_t.unsqueeze(0).expand(b, -1), pred["ddmg"][:, 2:])
            clamped = clamp_physics(
                raw_positions, current_pos_t.unsqueeze(0).expand(b, -1, -1),
                hp_pred, (current_hp_t > 0).unsqueeze(0).expand(b, -1),
            )
            ammo_final = (anchor_ammo_t.unsqueeze(0) - pred["dammo"][:, -1].clamp_min(0.0) * MAX_AMMO).clamp_min(0.0)
            result = score_candidates(
                positions=clamped, hp=hp_pred, ammo_final=ammo_final,
                heading_final=pred["heading"][:, -1],
                current_positions=current_pos_t, current_hp=current_hp_t,
                team_ids=team_t, terrain_features=terrain_t,
                mission_type=layout.mission_type, objective=layout.objective,
                time_remaining=time_remaining,
                value_head=value_head, lam=lam,
            )
            score = result["score"]
            if config.imagined_feasibility_mask:
                # 스텝 시작 시점(k=0은 실제 현재, k≥1은 상상 k-1 이후) 상태로 실행층
                # 강등 규칙을 미리 적용 — 불가능 ENGAGE가 든 후보는 선발 배제.
                pos_start = torch.cat(
                    [current_pos_t.view(1, 1, num_units, 2).expand(b, -1, -1, -1),
                     clamped[:, : HORIZON - 1]], dim=1,
                )
                hp_start = torch.cat(
                    [current_hp_t.view(1, 1, num_units).expand(b, -1, -1),
                     hp_pred[:, : HORIZON - 1]], dim=1,
                )
                bad = _infeasible_engage(
                    engage_t[start:end], pos_start, hp_start, num_blue, rects_t
                )
                score = score - bad.float() * 1e6
            scores[start:end] = score
            gains[start:end] = result["gain"]
            values[start:end] = result["value"]

        order = torch.argsort(scores, descending=True).cpu().numpy()
        elite_idx = order[: config.elites]
        elites = PlanCandidates(
            candidates.action_type_ids[elite_idx], candidates.move_xy[elite_idx],
            candidates.target_slots[elite_idx], candidates.theta[elite_idx],
            candidates.issued[elite_idx],
        )
        distribution = _refit(elites, num_blue, num_units - num_blue)
        top = int(order[0])
        top_score = float(scores[top])
        if best is None or top_score > best[0]:
            # scores/gain/value도 같은 반복의 것을 함께 보관 — 마지막 반복 배열에
            # best_index를 들이대면 반복이 어긋나 V 짝 수집까지 오염된다.
            best = (
                top_score, candidates, top,
                scores.cpu().numpy(), gains.cpu().numpy(), values.cpu().numpy(),
            )

    _, best_candidates, best_index, scores_np, gains_np, values_np = best
    return CEMResult(
        best_index=best_index,
        candidates=best_candidates,
        scores=scores_np,
        gain=gains_np,
        value=values_np,
    )
