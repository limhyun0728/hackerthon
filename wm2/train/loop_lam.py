"""에피소딕 루프 — λ 인자화판. 기존 loop.py는 수정하지 않고 헬퍼를 재사용한다.

배경 (2026-08-13 진단): 교착에서 접근의 진짜 보상(V 프리미엄 +0.018)이 제자리 사격의
팬텀 보상(피해 과대의 argmax 꼬리)과 박빙이라 접근이 낙선한다. V 지평 연장(h30)은
프리미엄을 못 키웠다(기각). 남은 레버는 λ — V항을 증폭해 박빙을 깬다.

    python -m wm2.train.loop_lam \
        --checkpoint output/wm2_run11/wm2_latest.pt \
        --value-checkpoint checkpoints/wm2_value_run11.pt \
        --scenario-dirs $(cat output/loop3_scenarios.txt) \
        --lam-a 3.0 --lam-b 1.0 --pairs 100 --device cuda:0 \
        --output-root output/wm2_loop6

loop.py와의 차이: λ 쌍이 (1, 0) 고정이 아니라 --lam-a/--lam-b 인자. episode_seed는
arm 인덱스(0=b, 1=a) 기준이라 loop4/5와 시드가 다른 독립 반복이다.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from . import loop as lp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--scenario-dirs", nargs="+", required=True)
    parser.add_argument("--lam-a", type=float, default=3.0, help="실험 arm의 λ")
    parser.add_argument("--lam-b", type=float, default=1.0, help="대조 arm의 λ")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="output/wm2_loop_lam")
    parser.add_argument("--v-update-every", type=int, default=10)
    parser.add_argument("--v-update-steps", type=int, default=200)
    parser.add_argument("--v-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tqdm import tqdm
    from ..sim.episode import run_cem_episode

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
    chosen = [dirs[i] for i in rng.choice(len(dirs), size=min(args.pairs, len(dirs)), replace=False)]

    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "loop_summary.jsonl"
    cem_config = lp.CEMConfig(
        candidates=args.candidates, elites=args.elites, iterations=args.iterations
    )

    lams = (args.lam_a, args.lam_b)
    wins = {l: 0 for l in lams}; totals = {l: 0 for l in lams}
    v_buffer = []
    live_pred, live_real = [], []
    holdout_pred, holdout_real = [], []
    progress_bar = tqdm(chosen, desc=f"loop λ={args.lam_a} vs {args.lam_b}", unit="pair")
    for pair_index, scenario_dir in enumerate(progress_bar):
        scenario = json.loads((Path(scenario_dir) / "config.json").read_text())
        sig_probe = type("S", (), {
            "run_dir": scenario_dir,
            "obstacles": tuple(tuple(float(v) for v in r) for r in scenario["obstacles"]),
        })
        lp.assert_not_heldout(sig_probe, signatures)
        layout = lp._layout_from_scenario(scenario)
        for arm, lam in enumerate(lams):
            episode_seed = args.seed + pair_index * 2 + arm
            result = run_cem_episode(
                scenario=scenario, layout=layout, model=model, heads=heads,
                value_head=value_head, cem_config=cem_config, device=device,
                lam=lam, seed=episode_seed, duration=layout.duration_sec,
                label=f"[{pair_index+1}/{len(chosen)} λ={lam:g}]",
            )
            totals[lam] += 1
            if result.outcome == "WIN":
                wins[lam] += 1
            lp._save_episode_dir(
                output_root, f"episode_p{pair_index:03d}_lam{lam:g}",
                result, scenario, lam, episode_seed,
            )
            is_holdout = pair_index % 5 == 4
            for p in result.v_pairs:
                if p.label is None:
                    continue
                if is_holdout:
                    holdout_pred.append(p.predicted_value); holdout_real.append(p.label)
                else:
                    live_pred.append(p.predicted_value); live_real.append(p.label)
            if not is_holdout:
                samples = lp._samples_from_pairs(result.v_pairs, layout)
                if samples is not None:
                    v_buffer.append(samples)
            with summary_path.open("a") as f:
                f.write(json.dumps({
                    "pair": pair_index, "lam": lam, "scenario": scenario_dir,
                    "outcome": result.outcome, "final_progress": round(result.final_progress, 4),
                    "planned": len(result.planned_commands), "v_pairs": len(result.v_pairs),
                }, ensure_ascii=False) + "\n")
        progress_bar.set_postfix(
            **{f"win{l:g}": f"{wins[l]}/{totals[l]}" for l in lams},
            vbuf=sum(len(s.labels) for s in v_buffer),
        )
        if (pair_index + 1) % args.v_update_every == 0 and v_buffer:
            recent = min(len(live_pred), 200)
            rho_live = lp._spearman(np.asarray(live_pred[-recent:]), np.asarray(live_real[-recent:]))
            recent_h = min(len(holdout_pred), 200)
            rho_holdout = (
                lp._spearman(np.asarray(holdout_pred[-recent_h:]), np.asarray(holdout_real[-recent_h:]))
                if recent_h >= 10 else float("nan")
            )
            loss = lp._finetune_value(
                value_head, v_buffer, device,
                steps=args.v_update_steps, lr=args.v_lr, rng=rng,
            )
            lp.save_value_head(output_root / "wm2_value_online.pt", value_head)
            print(
                f"\nV online 갱신 pair={pair_index+1} buffer={sum(len(s.labels) for s in v_buffer)} "
                f"loss={loss:.5f} V_live_rho={rho_live:+.3f} V_holdout_rho={rho_holdout:+.3f}",
                flush=True,
            )

    print(f"\ndone λ={args.lam_a:g}: {wins[args.lam_a]}/{totals[args.lam_a]} 승  |  "
          f"λ={args.lam_b:g}: {wins[args.lam_b]}/{totals[args.lam_b]} 승")
    print(f"V online 체크포인트: {output_root/'wm2_value_online.pt'}")


if __name__ == "__main__":
    main()
