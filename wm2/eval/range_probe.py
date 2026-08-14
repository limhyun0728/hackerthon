"""사거리 프로브 — run별 재검 게이트 (CF v3의 검증 기준, 설계 14b절).

교착 거리대(기본 7.5~10유닛)의 실제 상태에서 세 가지 통제 계획의
"모델 상상 RED 피해"와 "DEVS 실측 RED 피해"를 비교한다:

    hold_fire   전원 제자리에서 최근접 표적 6틱 사격
    close_fire  전원 3틱 접근 후 3틱 사격
    approach    전원 6틱 접근 (사격 없음)

계보: run4에서 hold_fire 상상 112.4 vs 실측 4.8HP (23배) — 무작위 CF만으로는
사거리·LOS 인과가 분리되지 않았다. 게이트: ① 상상이 실측 순서(close > hold)를
재현 ② hold/close 과대평가 ≤2~3배.

    python -m wm2.eval.range_probe \
        --checkpoint output/wm2_run5/wm2_best.pt \
        --episode-dirs 'output/wm2_loop3/episode_p*' \
        --device cuda:0 --states 30
"""

from __future__ import annotations

import argparse
import glob
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig
from ..data.batch import collate
from ..data.counterfactual import _plan_action_tokens
from ..data.episodes import load_episode
from ..data.windows import build_windows
from ..model.features import (
    ACTION_ENGAGE,
    ACTION_MOVE,
    MAX_HP,
    MAX_MOVE_PER_STEP,
    TeamId,
    norm_x,
    norm_y,
)
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor
from ..model.rollout import assemble_hp

_REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_NAMES = ("hold_fire", "close_fire", "approach")


def _build_plans(episode, base_tick: int, horizon: int = 6):
    """h2(=base_tick) 상태 기준 통제 계획 3종. (PlanSpec candidates 축 = 계획 종류)"""
    from ..sim.adapter import PlanSpec

    frame = episode.frames[base_tick]
    blue_ids = sorted(episode.blue_ids)
    red_alive = [u for u in sorted(episode.red_ids) if frame[u].hp > 0.0]
    num_blue = len(blue_ids)
    n = len(PLAN_NAMES)

    types = np.zeros((n, horizon, num_blue), dtype=np.int64)
    move = np.zeros((n, horizon, num_blue, 2), dtype=np.float32)
    targets = np.zeros((n, horizon, num_blue), dtype=np.int64)
    theta = np.zeros((n, horizon, num_blue), dtype=np.float32)
    issued = np.zeros((n, horizon, num_blue), dtype=bool)
    approach_steps = horizon // 2

    for ui, uid in enumerate(blue_ids):
        state = frame[uid]
        if state.hp <= 0.0 or not red_alive:
            continue
        issued[:, :, ui] = True
        target = min(
            red_alive,
            key=lambda r: math.hypot(frame[r].x - state.x, frame[r].y - state.y),
        )
        tgt = frame[target]
        dist = math.hypot(tgt.x - state.x, tgt.y - state.y)
        ux, uy = (
            ((tgt.x - state.x) / dist, (tgt.y - state.y) / dist) if dist > 1e-6 else (0.0, 0.0)
        )

        def waypoint(k: int) -> tuple[float, float]:
            advance = min((k + 1) * MAX_MOVE_PER_STEP, max(dist - 0.5, 0.0))
            return norm_x(state.x + ux * advance), norm_y(state.y + uy * advance)

        # hold_fire
        types[0, :, ui] = ACTION_ENGAGE
        targets[0, :, ui] = target
        # close_fire
        for k in range(approach_steps):
            types[1, k, ui] = ACTION_MOVE
            move[1, k, ui] = waypoint(k)
        types[1, approach_steps:, ui] = ACTION_ENGAGE
        targets[1, approach_steps:, ui] = target
        # approach
        for k in range(horizon):
            types[2, k, ui] = ACTION_MOVE
            move[2, k, ui] = waypoint(k)

    return PlanSpec(
        action_type_ids=types, move_xy_norm=move, target_ids=targets,
        theta_radians=theta, issued=issued,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=30)
    parser.add_argument("--per-episode", type=int, default=2)
    parser.add_argument("--min-dist", type=float, default=7.5)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ..sim.adapter import rollout  # 구코드 접점 — 필요 시점에만 import
    from tqdm import tqdm

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WM2Predictor(ModelConfig()).to(device); model.load_state_dict(payload["model"]); model.eval()
    heads = WM2Heads(ModelConfig()).to(device); heads.load_state_dict(payload["heads"]); heads.eval()

    dirs = sorted(
        {d for p in args.episode_dirs for d in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p))) if Path(d).is_dir()}
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(dirs)

    imagined = {name: [] for name in PLAN_NAMES}
    # 무클립 진단: 조립(assemble_hp)의 유닛별 damage clamp_min(0)은 노이즈의 음수쪽만
    # 잘라 합계를 정류 편향시킨다. 부호 유지 합이 실측에 가깝다면 과대평가의 원인은
    # 피해 head의 편향이 아니라 클램프 정류다.
    imagined_signed = {name: [] for name in PLAN_NAMES}
    realized = {name: [] for name in PLAN_NAMES}
    started = time.time()
    progress = tqdm(dirs, desc="사거리 프로브", unit="ep")
    for d in progress:
        if len(imagined[PLAN_NAMES[0]]) >= args.states:
            break
        progress.set_postfix(states=len(imagined[PLAN_NAMES[0]]))
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            continue
        windows = build_windows(episode)
        team = None
        eligible = []
        for w in windows:
            base_tick = w.anchor_tick + 2
            if base_tick % 2 or w.anchor_tick < 6:
                continue
            team = np.asarray(w.layout.team_ids)
            f_h2 = w.unit_features[2]
            alive = f_h2[:, 7] > 0.5
            blue = (team == int(TeamId.BLUE)) & alive
            red = (team == int(TeamId.RED)) & alive
            if not blue.any() or not red.any():
                continue
            x = f_h2[:, 3] * 20.0
            y = f_h2[:, 4] * 12.5 - 2.5
            dmin = min(
                math.hypot(bx - rx, by - ry)
                for bx, by in zip(x[blue], y[blue])
                for rx, ry in zip(x[red], y[red])
            )
            if args.min_dist <= dmin <= args.max_dist:
                eligible.append(w)
        if not eligible:
            continue
        picks = [eligible[i] for i in rng.choice(len(eligible), size=min(args.per_episode, len(eligible)), replace=False)]
        red_ids = tuple(sorted(episode.red_ids))
        sorted_ids = sorted(list(episode.blue_ids) + list(episode.red_ids))
        red_mask = np.asarray([uid in episode.red_ids for uid in sorted_ids])

        for window in picks:
            if len(imagined[PLAN_NAMES[0]]) >= args.states:
                break
            base_tick = window.anchor_tick + 2
            plan = _build_plans(episode, base_tick)
            hp_now = np.asarray([episode.frames[base_tick][u].hp for u in sorted_ids])

            # 모델 상상: window 복제 + 미래 액션 토큰만 계획으로 교체 (마스킹 없음)
            probes = []
            for c in range(len(PLAN_NAMES)):
                actions = window.actions.copy()
                actions[2:] = _plan_action_tokens(plan, c, window.layout, red_ids)
                probes.append(replace(window, actions=actions))
            batch = collate(probes, [np.empty(0, dtype=np.int64)] * len(probes), device)
            with torch.no_grad():
                out = model(
                    unit_features=batch["unit_features"],
                    terrain_features=batch["terrain_features"],
                    mission_features=batch["mission_features"],
                    actions=batch["actions"],
                    team_ids=batch["team_ids"],
                    masked_units=batch["masked_units"],
                )
                preds = heads(out["unit_tokens"], out["mission_tokens"])
            hp_hat = assemble_hp(batch["anchor_hp"], preds["ddmg"])[:, -1].cpu().numpy()  # (3, U) f6
            dmg_signed = (preds["ddmg"][:, -1] * MAX_HP).cpu().numpy()                    # (3, U) 부호 유지

            # DEVS 실측
            units = rollout(episode=episode, base_tick=base_tick, plan=plan, horizon=6, seed=777)
            for c, name in enumerate(PLAN_NAMES):
                imagined[name].append(float(np.clip(hp_now - hp_hat[c], 0, None)[red_mask].sum()))
                imagined_signed[name].append(float(dmg_signed[c][red_mask].sum()))
                realized[name].append(float(np.clip(hp_now - units[c, -1, :, 2], 0, None)[red_mask].sum()))

    n = len(imagined[PLAN_NAMES[0]])
    print(f"\n프로브 상태 {n}개 (적거리 {args.min_dist}~{args.max_dist}유닛, 6초 RED 피해, {time.time()-started:.0f}s)")
    print(f"{'계획':<14}{'모델 상상':>10}{'무클립':>8}{'DEVS 실측':>10}{'배율':>8}")
    for name in PLAN_NAMES:
        im, re = float(np.mean(imagined[name])), float(np.mean(realized[name]))
        sg = float(np.mean(imagined_signed[name]))
        ratio = im / re if re > 0.1 else float("inf")
        print(f"{name:<14}{im:>10.1f}{sg:>8.1f}{re:>10.1f}{ratio:>8.1f}x")
    order_ok = np.mean(imagined["close_fire"]) > np.mean(imagined["hold_fire"])
    real_order = np.mean(realized["close_fire"]) > np.mean(realized["hold_fire"])
    print(f"순서 게이트: 실측 close>hold={real_order}, 상상 close>hold={order_ok}")


if __name__ == "__main__":
    main()
