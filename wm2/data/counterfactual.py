"""카운터팩추얼 window 생성기 (설계·계획의 2b).

rule 에피소드의 중간 상태에서 **무작위 계획(무리한 ENGAGE 포함)**을 DEVS로 굴려
(계획 토큰, 실제 결과) 쌍을 만든다. rule 데이터에는 "사거리 밖 ENGAGE" 같은 무리한
명령이 없어서, 이 데이터가 없으면 CEM이 학습 분포 밖 토큰을 던질 때 모델 반응이
무정의가 된다 — 구 시스템의 "상상 속 120m 저격" 구멍.

    python -m wm2.data.counterfactual \
        --episode-dirs 'output/statickv_rule/episode_*' \
        --output-dir output/wm2_cf --per-episode 6 --candidates 4

학습에서는 train_wm의 --counterfactual-dirs 로 읽는다.
"""

from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import numpy as np

from ..model.features import (
    ACTION_DIM,
    ACTION_ENGAGE,
    ACTION_MOVE,
    ACTION_STOP,
    ACTION_TURN,
    MAX_AMMO,
    MAX_HP,
    MAX_MOVE_PER_STEP,
    NUM_ACTION_TYPES,
    OBJECTIVE_RADIUS,
    norm_x,
    norm_y,
)
from .episodes import Episode, load_episode
from .scenarios import assert_not_heldout, heldout_signatures
from .windows import (
    EpisodeLayout,
    HISTORY_FRAMES,
    LABEL_FRAMES,
    PRED_FRAMES,
    TOTAL_FRAMES,
    Window,
    _heading_pair,
    _mission_vector,
    _parse_action,
    _unit_vector,
    build_layout,
)

# 무작위 계획의 액션 분포. ENGAGE를 후하게 줘야 "무리한 ENGAGE → 아무 일 없음"
# 표본이 충분히 나온다. 표적은 사거리·LOS 무관 균등 — 그게 목적이다.
ACTION_PROBS = {ACTION_STOP: 0.15, ACTION_MOVE: 0.40, ACTION_ENGAGE: 0.35, ACTION_TURN: 0.10}


def sample_plans(
    *,
    rng: np.random.Generator,
    candidates: int,
    horizon: int,
    episode: Episode,
    base_tick: int,
):
    from ..sim.adapter import PlanSpec

    blue_ids = sorted(episode.blue_ids)
    red_alive = [
        uid for uid in sorted(episode.red_ids)
        if episode.frames[base_tick][uid].hp > 0.0
    ]
    num_blue = len(blue_ids)
    frame = episode.frames[base_tick]

    types = rng.choice(
        list(ACTION_PROBS), size=(candidates, horizon, num_blue), p=list(ACTION_PROBS.values())
    )
    if not red_alive:
        types[types == ACTION_ENGAGE] = ACTION_STOP

    move_xy = np.zeros((candidates, horizon, num_blue, 2), dtype=np.float32)
    target_ids = np.zeros((candidates, horizon, num_blue), dtype=np.int64)
    theta = rng.uniform(-math.pi, math.pi, size=(candidates, horizon, num_blue)).astype(np.float32)
    issued = np.zeros((candidates, horizon, num_blue), dtype=bool)

    for ui, uid in enumerate(blue_ids):
        state = frame[uid]
        if state.hp <= 0.0:
            continue
        issued[:, :, ui] = True
        # MOVE 목적지: 1틱 예산 안의 무작위 걸음을 스텝마다 이어붙인다 (규약과 동일 스케일)
        angles = rng.uniform(-math.pi, math.pi, size=(candidates, horizon))
        radii = rng.uniform(0.2, MAX_MOVE_PER_STEP, size=(candidates, horizon))
        dx = np.cumsum(radii * np.cos(angles), axis=1)
        dy = np.cumsum(radii * np.sin(angles), axis=1)
        move_xy[:, :, ui, 0] = np.vectorize(norm_x)(state.x + dx)
        move_xy[:, :, ui, 1] = np.vectorize(norm_y)(state.y + dy)
        if red_alive:
            target_ids[:, :, ui] = rng.choice(red_alive, size=(candidates, horizon))

    return PlanSpec(
        action_type_ids=types.astype(np.int64),
        move_xy_norm=move_xy,
        target_ids=np.where(types == ACTION_ENGAGE, target_ids, 0),
        theta_radians=theta,
        issued=issued,
    )


def _plan_action_tokens(plan, candidate: int, layout: EpisodeLayout, red_ids: tuple[int, ...]) -> np.ndarray:
    """계획 스텝 0..5 → (6, U_blue, ACTION_DIM) 액션 토큰. **계획된** 명령이 곧 토큰이다."""
    horizon = plan.action_type_ids.shape[1]
    tokens = np.zeros((horizon, layout.num_blue, ACTION_DIM), dtype=np.float32)
    move_offset = 1 + NUM_ACTION_TYPES
    for step in range(horizon):
        for ui in range(layout.num_blue):
            if not plan.issued[candidate, step, ui]:
                continue
            vector = tokens[step, ui]
            vector[0] = 1.0
            action_type = int(plan.action_type_ids[candidate, step, ui])
            vector[1 + action_type] = 1.0
            if action_type == ACTION_MOVE:
                vector[move_offset : move_offset + 2] = plan.move_xy_norm[candidate, step, ui]
            elif action_type == ACTION_ENGAGE:
                target = int(plan.target_ids[candidate, step, ui])
                if target in red_ids:
                    vector[move_offset + 2 + red_ids.index(target)] = 1.0
            elif action_type == ACTION_TURN:
                angle = float(plan.theta_radians[candidate, step, ui])
                vector[move_offset + 12] = math.cos(angle)
                vector[move_offset + 13] = math.sin(angle)
    return tokens


def _future_completion(
    episode: Episode, future_units: np.ndarray, unit_ids: list[int]
) -> np.ndarray:
    """rollout 결과에서 f1..f6 completion을 유도한다. future_units: (H, U, 6)."""
    blue = [i for i, uid in enumerate(unit_ids) if uid in episode.blue_ids]
    red = [i for i, uid in enumerate(unit_ids) if uid in episode.red_ids]
    flags = np.zeros(future_units.shape[0], dtype=np.float32)
    for k in range(future_units.shape[0]):
        alive_blue = [i for i in blue if future_units[k, i, 2] > 0.0]
        if not alive_blue:
            continue
        red_alive = any(future_units[k, i, 2] > 0.0 for i in red)
        reached = any(
            math.hypot(
                future_units[k, i, 0] - episode.objective[0],
                future_units[k, i, 1] - episode.objective[1],
            )
            <= OBJECTIVE_RADIUS
            for i in alive_blue
        )
        from ..model.features import (
            MISSION_DESTROY_ALL,
            MISSION_DESTROY_AND_REACH,
        )

        if episode.mission_type == MISSION_DESTROY_AND_REACH:
            flags[k] = float((not red_alive) and reached)
        elif episode.mission_type == MISSION_DESTROY_ALL:
            flags[k] = float(not red_alive)
        else:
            flags[k] = float(reached)
    return flags


def windows_from_rollout(
    episode: Episode,
    layout: EpisodeLayout,
    base_tick: int,
    plan,
    rollout_units: np.ndarray,   # (C, H, U, 6) = x, y, hp, ammo, cos, sin
) -> list[Window]:
    """실측 history(a,h1,h2) + 카운터팩추얼 미래(f1..f6)로 window를 만든다."""
    unit_ids = list(layout.unit_ids)
    red_ids = tuple(sorted(episode.red_ids))
    ticks = [base_tick - 3, base_tick - 2, base_tick - 1, base_tick]  # pre, a, h1, h2
    real = [episode.frames[t] for t in ticks]
    anchor = real[1]

    # 실측 부분 (프레임 a..h2)
    history = np.zeros((HISTORY_FRAMES, layout.num_units, 10), dtype=np.float32)
    for fi in range(HISTORY_FRAMES):
        for ui, uid in enumerate(unit_ids):
            history[fi, ui] = _unit_vector(real[fi + 1][uid], real[fi][uid], int(layout.team_ids[ui]))
    mission_history = np.stack(
        [_mission_vector(episode, ticks[fi + 1], real[fi + 1]) for fi in range(HISTORY_FRAMES)]
    )

    # 실측 명령 (발행틱 a, h1)
    real_actions = np.zeros((2, layout.num_blue, ACTION_DIM), dtype=np.float32)
    blue_slot = {uid: i for i, uid in enumerate(unit_ids[: layout.num_blue])}
    for j, tick in enumerate((ticks[1], ticks[2])):
        for command in episode.commands.get(tick, ()):
            slot = blue_slot.get(command.unit_id)
            if slot is not None:
                real_actions[j, slot] = _parse_action(command.action, command.detail, red_ids)

    windows: list[Window] = []
    candidates = rollout_units.shape[0]
    for c in range(candidates):
        future = rollout_units[c]                     # (6, U, 6)
        unit_features = np.zeros((TOTAL_FRAMES, layout.num_units, 10), dtype=np.float32)
        unit_features[:HISTORY_FRAMES] = history
        prev_xy = np.stack(
            [[real[3][uid].x, real[3][uid].y] for uid in unit_ids]
        )                                              # h2 실측 위치
        for k in range(PRED_FRAMES):
            fi = HISTORY_FRAMES + k
            for ui, uid in enumerate(unit_ids):
                x, y, hp, ammo, cos, sin = future[k, ui]
                alive = hp > 0.0
                unit_features[fi, ui] = (
                    float(layout.team_ids[ui]), hp / MAX_HP, ammo / MAX_AMMO,
                    norm_x(x), norm_y(y), cos, sin, float(alive),
                    (x - prev_xy[ui, 0]) / MAX_MOVE_PER_STEP if alive else 0.0,
                    (y - prev_xy[ui, 1]) / MAX_MOVE_PER_STEP if alive else 0.0,
                )
            prev_xy = future[k, :, :2].copy()

        # 레이블 (frame a 기준): h1,h2는 실측, f1..f6은 rollout
        dpos = np.zeros((LABEL_FRAMES, layout.num_units, 2), dtype=np.float32)
        ddmg = np.zeros((LABEL_FRAMES, layout.num_units), dtype=np.float32)
        dammo = np.zeros((LABEL_FRAMES, layout.num_units), dtype=np.float32)
        heading = np.zeros((LABEL_FRAMES, layout.num_units, 2), dtype=np.float32)
        for ui, uid in enumerate(unit_ids):
            base = anchor[uid]
            for li, frame in ((0, real[2]), (1, real[3])):
                s = frame[uid]
                dpos[li, ui] = (s.x - base.x, s.y - base.y)
                ddmg[li, ui] = (base.hp - s.hp) / MAX_HP
                dammo[li, ui] = (base.ammo - s.ammo) / MAX_AMMO
                heading[li, ui] = _heading_pair(s.heading_deg)
            for k in range(PRED_FRAMES):
                x, y, hp, ammo, cos, sin = future[k, ui]
                dpos[2 + k, ui] = (x - base.x, y - base.y)
                ddmg[2 + k, ui] = (base.hp - hp) / MAX_HP
                dammo[2 + k, ui] = (base.ammo - ammo) / MAX_AMMO
                heading[2 + k, ui] = (cos, sin)

        mission_features = np.zeros((TOTAL_FRAMES, 5), dtype=np.float32)
        mission_features[:HISTORY_FRAMES] = mission_history
        completion = _future_completion(episode, future, unit_ids)
        for k in range(PRED_FRAMES):
            t = base_tick + 1 + k
            mission_features[HISTORY_FRAMES + k] = (
                float(episode.mission_type),
                norm_x(episode.objective[0]),
                norm_y(episode.objective[1]),
                max(0.0, (episode.duration_sec - t) / episode.duration_sec),
                completion[k],
            )

        actions = np.zeros((LABEL_FRAMES, layout.num_blue, ACTION_DIM), dtype=np.float32)
        actions[:2] = real_actions
        actions[2:] = _plan_action_tokens(plan, c, layout, red_ids)

        windows.append(
            Window(
                layout=layout,
                anchor_tick=ticks[1],
                unit_features=unit_features,
                mission_features=mission_features,
                dpos=dpos,
                ddmg=ddmg,
                dammo=dammo,
                heading=heading,
                completion=completion,
                pos_loss_mask=np.asarray(
                    [anchor[uid].hp > 0.0 for uid in unit_ids], dtype=bool
                ),
                actions=actions,
            )
        )
    return windows


# ── npz 저장/로드 ───────────────────────────────────────────────────────
_ARRAY_FIELDS = (
    "unit_features", "mission_features", "dpos", "ddmg", "dammo",
    "heading", "completion", "pos_loss_mask", "actions",
)


def save_windows(path: Path, windows: list[Window]) -> None:
    layout = windows[0].layout
    payload = {name: np.stack([getattr(w, name) for w in windows]) for name in _ARRAY_FIELDS}
    payload["anchor_ticks"] = np.asarray([w.anchor_tick for w in windows])
    payload["unit_ids"] = np.asarray(layout.unit_ids)
    payload["team_ids"] = np.asarray(layout.team_ids)
    payload["num_blue"] = np.asarray(layout.num_blue)
    payload["terrain_features"] = layout.terrain_features
    payload["mission_type"] = np.asarray(layout.mission_type)
    payload["objective"] = np.asarray(layout.objective)
    payload["duration_sec"] = np.asarray(layout.duration_sec)
    np.savez_compressed(path, **payload)


def load_windows(path: Path) -> list[Window]:
    data = np.load(path)
    layout = EpisodeLayout(
        unit_ids=tuple(int(v) for v in data["unit_ids"]),
        team_ids=data["team_ids"],
        num_blue=int(data["num_blue"]),
        terrain_features=data["terrain_features"],
        mission_type=int(data["mission_type"]),
        objective=tuple(float(v) for v in data["objective"]),
        duration_sec=float(data["duration_sec"]),
    )
    count = data["unit_features"].shape[0]
    return [
        Window(
            layout=layout,
            anchor_tick=int(data["anchor_ticks"][i]),
            **{name: data[name][i] for name in _ARRAY_FIELDS},
        )
        for i in range(count)
    ]


def main() -> None:
    from ..sim.adapter import rollout

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="output/wm2_cf")
    parser.add_argument("--per-episode", type=int, default=6, help="에피소드당 base tick 수")
    parser.add_argument("--candidates", type=int, default=4, help="base tick당 무작위 계획 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dirs = sorted(
        {
            d
            for p in args.episode_dirs
            for d in (glob.glob(p) or glob.glob(str(repo_root / p)))
            if Path(d).is_dir()
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    signatures = heldout_signatures()

    from tqdm import tqdm

    started = time.time()
    total_windows = 0
    progress = tqdm(dirs, desc="counterfactual 생성", unit="ep")
    for index, d in enumerate(progress):
        progress.set_postfix(windows=total_windows)
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError, NotADirectoryError):
            continue
        assert_not_heldout(episode, signatures)
        layout = build_layout(episode)
        ticks = episode.ticks
        # 짝수 틱만: RED 결정 파이프라인(관측→1초 지연→1초 실행)은 에피소드 시작에
        # 전역 위상이 잠겨 있고, rollout은 항상 "t+1 대기, t+2 이동" 리듬으로 시작한다.
        # 홀수 틱에서 시작하면 원 세계와 위상이 반전된 미래를 만들어 RED 레이블이
        # 계통적으로 오염된다 (실측: 반전 시 t+2 오차 2.6→11.5m).
        valid = [t for t in ticks if t - 3 >= ticks[0] and t % 2 == 0]
        if len(valid) < args.per_episode:
            continue
        base_ticks = sorted(rng.choice(valid, size=args.per_episode, replace=False).tolist())

        episode_windows: list[Window] = []
        for base_tick in base_ticks:
            plan = sample_plans(
                rng=rng, candidates=args.candidates, horizon=PRED_FRAMES,
                episode=episode, base_tick=base_tick,
            )
            units = rollout(
                episode=episode, base_tick=base_tick, plan=plan,
                horizon=PRED_FRAMES, seed=int(rng.integers(0, 2**31 - 1)),
            )
            episode_windows.extend(
                windows_from_rollout(episode, layout, base_tick, plan, units)
            )
        if episode_windows:
            name = Path(d).name + ".npz"
            save_windows(output_dir / name, episode_windows)
            total_windows += len(episode_windows)
        if (index + 1) % 20 == 0:
            print(
                f"{index+1}/{len(dirs)} episodes, windows={total_windows}, "
                f"elapsed={time.time()-started:.0f}s", flush=True,
            )
    print(f"done episodes={len(dirs)} windows={total_windows} elapsed={time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
