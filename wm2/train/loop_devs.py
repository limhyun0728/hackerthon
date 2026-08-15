"""cem+devs 평가 러너 — 후보 롤아웃을 월드모델 상상 대신 DEVS 실측으로 (2026-08-15).

발표 지표 "미션별 달성도 (rule / cem+devs / ours)"의 가운데 팔. 같은 계획기(샘플러+
마스크+생존 채점+γ⁶V)에서 롤아웃만 진짜 시뮬레이터로 바꿔, ours와의 격차가 곧
"월드모델 오차의 대가"가 되게 한다. 계획시간 지표도 같은 로그에서 나온다.

기존 파일은 수정하지 않는다: episode.py의 계획 호출 심볼(cem_plan)을 이 프로세스
안에서만 DEVS 판으로 갈아끼운다(run_cem_episode 재사용). DEVS가 강등·사거리·LOS를
실제로 실행하므로 채점측 feasibility 안전망은 불필요하고, 샘플러 마스크는 그대로 탐색
효율로 작동한다.

    python -m wm2.train.loop_devs \
        --checkpoint output/wm2_run13/wm2_best.pt \
        --value-checkpoint checkpoints/wm2_value_rtgs_real.pt \
        --scenario-dirs $(cat output/loop3_scenarios.txt) \
        --pairs 30 --candidates 32 --iterations 3 \
        --device cuda:0 --output-root output/wm2_loop17_devs

주의: --value-checkpoint는 **실측 입력판** V를 줄 것 (DEVS 롤아웃 상태는 실측 분포).
상상 입력판을 꽂으면 입력 분포가 어긋난다. 시나리오 선택·episode_seed는 loop_lam의
lam-a 팔과 동일해 ours 팔들과 같은 판 짝비교가 된다.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import loop as lp
from ..model.features import (
    MAX_AMMO,
    MAX_HP,
    denorm_x,
    denorm_y,
    norm_x,
    norm_y,
)
from ..plan import cem
from ..plan.score import score_candidates
from ..sim.adapter import (
    OLD_ACTION_DIM,
    OLD_UNIT_AMMO,
    OLD_UNIT_COS,
    OLD_UNIT_HP,
    OLD_UNIT_SIN,
    OLD_UNIT_X,
    OLD_UNIT_Y,
    FutureActionPlanBatch,
    RolloutSnapshot,
    rollout_plans_with_devs,
)


def _rows_from_window(window) -> tuple:
    """window h2 특징을 DEVS 스냅샷 unit_rows로 되돌린다."""
    layout = window.layout
    feats = window.unit_features[2]
    rows = []
    for ui, uid in enumerate(layout.unit_ids):
        rows.append({
            "id": int(uid),
            "x": denorm_x(float(feats[ui, 3])),
            "y": denorm_y(float(feats[ui, 4])),
            "heading": math.degrees(math.atan2(float(feats[ui, 6]), float(feats[ui, 5]))),
            "hp": float(feats[ui, 1]) * MAX_HP,
            "ammo": float(feats[ui, 2]) * MAX_AMMO,
        })
    return tuple(rows)


def _old_plans(candidates: cem.PlanCandidates, layout) -> FutureActionPlanBatch:
    """wm2 PlanCandidates → 구 FutureActionPlanBatch (adapter.rollout과 같은 규약)."""
    c, h, num_blue = candidates.action_type_ids.shape
    blue_ids = np.asarray(layout.unit_ids[:num_blue], dtype=np.int64)
    red_ids = np.asarray(layout.unit_ids[num_blue:], dtype=np.int64)
    target_ids = np.zeros_like(candidates.target_slots)
    valid = candidates.target_slots >= 0
    target_ids[valid] = red_ids[candidates.target_slots[valid]]
    move_norm = np.stack(
        [np.vectorize(norm_x)(candidates.move_xy[..., 0]),
         np.vectorize(norm_y)(candidates.move_xy[..., 1])], axis=-1,
    ).astype(np.float32)
    as_t = torch.as_tensor
    shape3 = (c, h, num_blue)
    return FutureActionPlanBatch(
        action_features=torch.zeros(*shape3, OLD_ACTION_DIM),
        action_unit_ids=as_t(blue_ids).reshape(1, 1, -1).expand(*shape3).clone(),
        issued_mask=as_t(candidates.issued.astype(bool)),
        action_type_ids=as_t(candidates.action_type_ids.astype(np.int64)),
        target_entity_ids=as_t(target_ids.astype(np.int64)),
        target_indices=torch.zeros(*shape3, dtype=torch.long),
        move_xy_norm=as_t(move_norm),
        theta_radians=as_t(candidates.theta.astype(np.float32)),
        red_target_ids=as_t(np.asarray(sorted(int(u) for u in red_ids), dtype=np.int64)),
    )


def make_devs_planner(duration_sec: float, obstacles: tuple):
    """cem_plan과 같은 시그니처의 DEVS 롤아웃 계획기를 만든다."""

    def devs_plan(*, window, model, heads, value_head, config, device, rng, lam, chunk=128):
        layout = window.layout
        num_blue, num_units = layout.num_blue, layout.num_units
        rows = _rows_from_window(window)
        tick = window.anchor_tick + 2                       # h2 = 현재
        snapshot = RolloutSnapshot(
            unit_rows=rows,
            obstacles=obstacles,
            base_time_sec=float(tick),
            episode_duration_sec=duration_sec,
            objective=layout.objective,
            mission_type=layout.mission_type,
        )
        current_pos, current_hp, _ = cem._current_state(window)
        current_pos_t = torch.from_numpy(current_pos.astype(np.float32)).to(device)
        current_hp_t = torch.from_numpy(current_hp.astype(np.float32)).to(device)
        team_t = torch.from_numpy(np.asarray(layout.team_ids)).to(device)
        terrain_t = torch.from_numpy(layout.terrain_features).to(device)
        time_remaining = float(window.mission_features[2, 3]) - cem.HORIZON / duration_sec

        distribution = None
        best = None
        for _ in range(config.iterations):
            candidates = cem._sample_candidates(window, rng, config, distribution)
            plans = _old_plans(candidates, layout)
            features = rollout_plans_with_devs(
                plans=plans, snapshot=snapshot,
                seed=int(rng.integers(1 << 31)), device=torch.device("cpu"),
            )
            arr = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
            units = arr[:, :, :num_units, :]
            xs = np.vectorize(denorm_x)(units[..., OLD_UNIT_X])
            ys = np.vectorize(denorm_y)(units[..., OLD_UNIT_Y])
            positions = torch.from_numpy(
                np.stack([xs, ys], axis=-1).astype(np.float32)
            ).to(device)
            hp = torch.from_numpy((units[..., OLD_UNIT_HP] * MAX_HP).astype(np.float32)).to(device)
            ammo = torch.from_numpy(
                (units[:, -1, :, OLD_UNIT_AMMO] * MAX_AMMO).astype(np.float32)
            ).to(device)
            heading = torch.from_numpy(
                np.stack([units[:, -1, :, OLD_UNIT_COS], units[:, -1, :, OLD_UNIT_SIN]], axis=-1).astype(np.float32)
            ).to(device)
            result = score_candidates(
                positions=positions, hp=hp, ammo_final=ammo, heading_final=heading,
                current_positions=current_pos_t, current_hp=current_hp_t,
                team_ids=team_t, terrain_features=terrain_t,
                mission_type=layout.mission_type, objective=layout.objective,
                time_remaining=time_remaining, value_head=value_head, lam=lam,
            )
            scores = result["score"].detach().cpu().numpy()
            top = int(np.argmax(scores))
            if best is None or scores[top] > best[0]:
                best = (
                    float(scores[top]), candidates, top, scores,
                    result["gain"].detach().cpu().numpy(),
                    result["value"].detach().cpu().numpy(),
                )
            order = np.argsort(-scores)
            elite_idx = order[: config.elites]
            elites = cem.PlanCandidates(
                candidates.action_type_ids[elite_idx], candidates.move_xy[elite_idx],
                candidates.target_slots[elite_idx], candidates.theta[elite_idx],
                candidates.issued[elite_idx],
            )
            distribution = cem._refit(elites, num_blue, num_units - num_blue)

        _, cand, idx, scores, gains, values = best
        return cem.CEMResult(
            best_index=idx, candidates=cand, scores=scores, gain=gains, value=values
        )

    return devs_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="브리지용 WM (v_pair 기록에만 쓰임)")
    parser.add_argument("--value-checkpoint", required=True, help="실측 입력판 V")
    parser.add_argument("--scenario-dirs", nargs="+", required=True)
    parser.add_argument("--lam", type=float, default=0.83)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="output/wm2_loop_devs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tqdm import tqdm

    from ..sim import episode as se

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = lp.WM2Predictor(lp.ModelConfig()).to(device)
    model.load_state_dict(payload["model"]); model.eval()
    heads = lp.WM2Heads(lp.ModelConfig()).to(device)
    heads.load_state_dict(payload["heads"]); heads.eval()
    value_head = lp.load_value_head(Path(args.value_checkpoint), device)

    dirs = sorted({
        d for p in args.scenario_dirs
        for d in (glob.glob(p) or glob.glob(str(lp._REPO_ROOT / p)))
        if Path(d).is_dir() and (Path(d) / "config.json").exists()
    })
    signatures = lp.heldout_signatures()
    rng = np.random.default_rng(args.seed)
    chosen = [dirs[i] for i in rng.choice(len(dirs), size=min(100, len(dirs)), replace=False)]
    chosen = chosen[: args.pairs]   # 앞쪽 pair들 = ours 팔들과 같은 판

    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "loop_summary.jsonl"
    cem_config = lp.CEMConfig(
        candidates=args.candidates, elites=args.elites, iterations=args.iterations
    )

    wins = 0
    for pair_index, scenario_dir in enumerate(tqdm(chosen, desc="cem+devs", unit="ep")):
        scenario = json.loads((Path(scenario_dir) / "config.json").read_text())
        sig_probe = type("S", (), {
            "run_dir": scenario_dir,
            "obstacles": tuple(tuple(float(v) for v in r) for r in scenario["obstacles"]),
        })
        lp.assert_not_heldout(sig_probe, signatures)
        layout = lp._layout_from_scenario(scenario)
        obstacles = tuple(tuple(float(v) for v in r) for r in scenario["obstacles"])
        # 계획 심볼 교체: 이 프로세스 안에서만 cem_plan → DEVS 판 (episode.py 무수정)
        se.cem_plan = make_devs_planner(layout.duration_sec, obstacles)
        episode_seed = args.seed + pair_index * 2 + 0
        result = se.run_cem_episode(
            scenario=scenario, layout=layout, model=model, heads=heads,
            value_head=value_head, cem_config=cem_config, device=device,
            lam=args.lam, seed=episode_seed, duration=layout.duration_sec,
            label=f"[{pair_index+1}/{len(chosen)} devs]",
        )
        wins += result.outcome == "WIN"
        lp._save_episode_dir(
            output_root, f"episode_p{pair_index:03d}_devs", result, scenario, args.lam, episode_seed
        )
        with summary_path.open("a") as f:
            f.write(json.dumps({
                "pair": pair_index, "lam": args.lam, "scenario": scenario_dir,
                "outcome": result.outcome, "final_progress": round(result.final_progress, 4),
                "planned": len(result.planned_commands),
                "backend": "devs", "candidates": args.candidates, "iterations": args.iterations,
            }, ensure_ascii=False) + "\n")

    print(f"\ndone cem+devs: {wins}/{len(chosen)} 승")


if __name__ == "__main__":
    main()
