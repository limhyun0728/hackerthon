"""wm2 CEM (설계 11절).

한 계획 tick의 절차:
  후보 C개 샘플 (목적 지향 제안 + step0 ENGAGE 사실 마스크)
  → 월드모델 상상 → 물리 클램프 → 채점 (score.py)
  → elite로 분포 갱신 ×iterations → 최고 후보 반환

휴리스틱 원칙 (합의): 제안 편향은 탐색 효율 장치일 뿐 채점에 안 낀다. step0 mask는
현재 관측 사실 기준 "불가능 제거"만 한다. ENGAGE 표적 초기 분포는 균등 — elite가 조인다.
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

    # step0 ENGAGE 사실 마스크: 현재 사거리 안 생존 RED가 없으면 step0 ENGAGE 금지
    step0_engage_ok = np.zeros(num_blue, dtype=bool)
    for ui in range(num_blue):
        if not blue_alive[ui]:
            continue
        dist = np.linalg.norm(red_positions - positions[ui], axis=-1)
        step0_engage_ok[ui] = bool((red_alive & (dist <= MAX_FIRE_RANGE_UNITS)).any())

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
                if distribution is not None and distribution["counts"][step, ui] > 0:
                    probs = distribution["action_probs"][step, ui]
                else:
                    probs = ACTION_PRIOR
                if step == 0 and not step0_engage_ok[ui]:
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
                elif action == ACTION_ENGAGE and alive_red_indices.size:
                    if distribution is not None and distribution["target_counts"][step, ui] > 0:
                        target_probs = distribution["target_probs"][step, ui][alive_red_indices]
                        target_probs = target_probs + 1e-6
                        slot = int(rng.choice(alive_red_indices, p=target_probs / target_probs.sum()))
                    else:
                        slot = int(rng.choice(alive_red_indices))   # 균등 — elite가 조인다
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

    distribution = None
    best = None
    for _ in range(config.iterations):
        candidates = _sample_candidates(window, rng, config, distribution)
        tokens = torch.from_numpy(_action_tokens(window, candidates)).to(device)
        c = tokens.shape[0]
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
                raw_positions, anchor_xy_t.unsqueeze(0).expand(b, -1, -1),
                hp_pred, (anchor_hp_t > 0).unsqueeze(0).expand(b, -1),
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
            scores[start:end] = result["score"]
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
