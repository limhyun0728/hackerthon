"""게이트 2 (설계 12절): CEM 계획의 실행 가능성과 채점의 순위 능력.

    python -m wm2.eval.planning \
        --checkpoint output/wm2_run2/wm2_best.pt \
        --value-checkpoint checkpoints/wm2_value.pt \
        --episode-dirs 'output/validation_shared_v2/episode_*' \
        --device cuda:0 --episodes 8 --lam 1.0

지표:
1. ENGAGE 실행률 — best 계획을 DEVS로 실제 굴려, 계획된 ENGAGE 중 실제 발사(탄약 감소)
   비율. 구 시스템 22% → 목표 70%+.
2. 총점 vs 실현 진전 순위상관 — 후보를 점수 분위로 뽑아 DEVS로 굴리고, 실현된 6초
   progress 증가와 score의 Spearman. 채점이 실제와 같은 순서로 세우는가.
3. 접근 거동 — best 계획 실행 후 최근접 적과의 거리 변화.
base tick은 짝수만 (위상 정합 — counterfactual과 같은 이유).
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch

from ..config import CEMConfig, ModelConfig
from ..data.episodes import load_episode
from ..data.windows import build_windows
from ..model.features import ACTION_ENGAGE, TeamId
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor
from ..plan.cem import CEMResult, PlanCandidates, plan
from ..plan.score import progress_batch
from ..value.head import load_value_head
from ..value.train_value import _spearman

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_planspec(candidates: PlanCandidates, indices: list[int], red_ids: tuple[int, ...]):
    from ..sim.adapter import PlanSpec

    target_ids = np.zeros_like(candidates.target_slots[indices])
    slots = candidates.target_slots[indices]
    for red_slot, red_id in enumerate(red_ids):
        target_ids[slots == red_slot] = red_id
    return PlanSpec(
        action_type_ids=candidates.action_type_ids[indices],
        move_xy_norm=np.stack(
            [
                (candidates.move_xy[indices][..., 0] + 20.0) / 40.0 * 2.0 - 1.0,
                (candidates.move_xy[indices][..., 1] + 15.0) / 25.0 * 2.0 - 1.0,
            ],
            axis=-1,
        ).astype(np.float32),
        target_ids=target_ids,
        theta_radians=candidates.theta[indices],
        issued=candidates.issued[indices],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", default=None)
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks-per-episode", type=int, default=2)
    parser.add_argument("--corr-samples", type=int, default=12, help="순위상관용 DEVS rollout 후보 수")
    parser.add_argument(
        "--realize-seconds", type=int, default=6, choices=(6, 21),
        help="실현 진전 측정 구간. 21 = 6초 계획 + 15초 rule 연속 (score의 의미와 정합)",
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ..sim.adapter import rollout  # 구코드 접점 — 필요 시점에만 import

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WM2Predictor(ModelConfig()).to(device); model.load_state_dict(payload["model"]); model.eval()
    heads = WM2Heads(ModelConfig()).to(device); heads.load_state_dict(payload["heads"]); heads.eval()
    value_head = None
    if args.value_checkpoint and args.lam != 0.0:
        value_head = load_value_head(Path(args.value_checkpoint), device)

    dirs = sorted(
        {d for p in args.episode_dirs for d in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p))) if Path(d).is_dir()}
    )[: args.episodes]
    rng = np.random.default_rng(args.seed)
    config = CEMConfig(candidates=args.candidates)

    from tqdm import tqdm

    planned_engage = 0; fired_engage = 0
    rhos = []
    approach = []
    per_mission: dict[int, list[float]] = {}
    progress_bar = tqdm(dirs, desc="게이트 2 측정", unit="ep")
    for d in progress_bar:
        progress_bar.set_postfix(plans=len(rhos), engage=f"{fired_engage}/{planned_engage}")
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            continue
        windows = build_windows(episode)
        even = [w for w in windows if (w.anchor_tick + 2) % 2 == 0 and w.anchor_tick >= 6]
        if not even:
            continue
        chosen = [even[i] for i in rng.choice(len(even), size=min(args.ticks_per_episode, len(even)), replace=False)]
        red_ids = tuple(sorted(episode.red_ids))
        for window in chosen:
            base_tick = window.anchor_tick + 2
            result = plan(
                window=window, model=model, heads=heads, value_head=value_head,
                config=config, device=device, rng=rng, lam=args.lam,
            )
            team = np.asarray(window.layout.team_ids)
            current = window.unit_features[2]
            # ── 순위상관: 점수 분위에서 후보를 뽑아 실제로 굴린다 ──
            order = np.argsort(result.scores)[::-1]
            take = np.unique(
                np.linspace(0, len(order) - 1, min(args.corr_samples, len(order))).astype(int)
            )
            indices = [int(order[i]) for i in take]
            spec = _to_planspec(result.candidates, indices, red_ids)
            units = rollout(episode=episode, base_tick=base_tick, plan=spec, horizon=6, seed=777)
            team_t = torch.from_numpy(team)
            pos0 = torch.from_numpy(
                np.stack([[episode.frames[base_tick][u].x, episode.frames[base_tick][u].y]
                          for u in sorted(episode.frames[base_tick])])
            ).float().unsqueeze(0)
            hp0 = torch.from_numpy(
                np.asarray([episode.frames[base_tick][u].hp for u in sorted(episode.frames[base_tick])])
            ).float().unsqueeze(0)
            p0 = progress_batch(pos0, hp0, team_t, window.layout.mission_type, window.layout.objective)
            realized = []
            sorted_ids = sorted(episode.frames[base_tick])
            for k in range(len(indices)):
                if args.realize_seconds > 6:
                    from ..sim.adapter import continue_with_rules

                    final = continue_with_rules(
                        unit_states=units[k, -1], unit_ids=sorted_ids,
                        obstacles=episode.obstacles, objective=window.layout.objective,
                        seconds=args.realize_seconds - 6, seed=777,
                    )
                    pos_end = torch.from_numpy(final[:, 0:2]).float().unsqueeze(0)
                    hp_end = torch.from_numpy(final[:, 2]).float().unsqueeze(0)
                else:
                    pos_end = torch.from_numpy(units[k, -1, :, 0:2]).float().unsqueeze(0)
                    hp_end = torch.from_numpy(units[k, -1, :, 2]).float().unsqueeze(0)
                p_end = progress_batch(pos_end, hp_end, team_t, window.layout.mission_type, window.layout.objective)
                realized.append(float(p_end - p0))
            rho = _spearman(result.scores[indices], np.asarray(realized))
            if math.isfinite(rho):
                rhos.append(rho)
                per_mission.setdefault(window.layout.mission_type, []).append(rho)

            # ── best 계획: ENGAGE 실행률(탄약 감소 프록시) + 접근 거동 ──
            best_units = units[0] if indices[0] == result.best_index else rollout(
                episode=episode, base_tick=base_tick,
                plan=_to_planspec(result.candidates, [result.best_index], red_ids),
                horizon=6, seed=777,
            )[0]
            ammo_prev = np.asarray(
                [episode.frames[base_tick][u].ammo for u in sorted(episode.frames[base_tick])]
            )[: window.layout.num_blue]
            for step in range(6):
                ammo_now = best_units[step, : window.layout.num_blue, 3]
                for ui in range(window.layout.num_blue):
                    if (
                        result.candidates.issued[result.best_index, step, ui]
                        and result.candidates.action_type_ids[result.best_index, step, ui] == ACTION_ENGAGE
                    ):
                        planned_engage += 1
                        if ammo_now[ui] < ammo_prev[ui] - 1e-6:
                            fired_engage += 1
                ammo_prev = ammo_now

            blue = team == int(TeamId.BLUE); red = team == int(TeamId.RED)
            def nearest(units_pos, hp_row):
                bp = units_pos[blue][hp_row[blue] > 0]; rp = units_pos[red][hp_row[red] > 0]
                if len(bp) == 0 or len(rp) == 0:
                    return float("nan")
                return float(np.min(np.linalg.norm(bp[:, None] - rp[None], axis=-1)))
            d0 = nearest(pos0[0].numpy(), hp0[0].numpy())
            d6 = nearest(best_units[-1, :, 0:2], best_units[-1, :, 2])
            if math.isfinite(d0) and math.isfinite(d6):
                approach.append(d6 - d0)

    print(f"\n게이트 2 결과 (λ={args.lam}, 실현 {args.realize_seconds}초, {len(rhos)} 계획)")
    execution = 100.0 * fired_engage / max(1, planned_engage)
    print(f"ENGAGE 실행률: {fired_engage}/{planned_engage} = {execution:.0f}%  (기준 70%+, 구 시스템 22%)")
    print(f"score vs 실현 {args.realize_seconds}초 진전 순위상관: 평균 rho={np.mean(rhos):.3f}")
    for mission, values in sorted(per_mission.items()):
        print(f"  mission {mission}: rho={np.mean(values):.3f} (n={len(values)})")
    print(f"최근접 적 거리 변화 (best 계획, 6초): 평균 {np.mean(approach):+.2f} 유닛 (음수=접근)")


if __name__ == "__main__":
    main()
