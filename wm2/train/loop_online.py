"""에피소딕 루프 — WM+V 동시 온라인판 (2026-08-14). loop_lam.py 복사 기반.

기존 loop.py / loop_lam.py / train_wm.py는 수정하지 않고 재사용한다.

3단 사다리의 셋째 팔: loop8(전부 동결) / loop9(V만 온라인) / 본 스크립트(V+WM 온라인).
단일 팔(λ 하나)로 돌고, 시나리오 선택(seed 42 choice)과 episode_seed(seed+pair*2+0)를
loop_lam의 lam-a 팔과 동일하게 맞춰 loop8/9의 λ=0.83 팔과 같은 판 짝비교가 된다.

WM 온라인 갱신 (--wm-update-every 쌍마다, 낮은 LR):
- 표본 = 이 런이 방금 만든 on-policy 윈도우 + rule 리플레이 + CF 리플레이 혼합.
  리플레이 없이는 협소 분포(같은 100판) 과적합과, CF로만 분리된 사거리·LOS 피해
  인과의 침식 위험이 있다. 손실·마스크·클립은 train_wm 규약 그대로 (LossConfig 기본).
- 갱신마다 output_root/wm2_wm_online.pt 저장 + 공유 검증셋 RED f2/정지가정 로그로
  드리프트 감시. 입력 체크포인트 파일은 읽기 전용 — 스냅샷 사본을 넘길 것.

    python -m wm2.train.loop_online \
        --checkpoint checkpoints/snapshot_0814_preonline/wm2_run12_latest.pt \
        --value-checkpoint checkpoints/snapshot_0814_preonline/wm2_value_rtg_run12.pt \
        --scenario-dirs $(cat output/loop3_scenarios.txt) \
        --pairs 100 --device cuda:2 --output-root output/wm2_loop11_wm_online

주의: CF 리플레이는 train_wm의 교차 중복 제거를 생략한다(추출 가중 목적이 아니라 보정
유지 목적이라 중복의 해가 작다). 정식 재학습은 train_wm 코스로.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from . import loop as lp
from . import train_wm as tw
from ..config import LossConfig, MaskConfig, TrainConfig
from ..data.batch import collate
from ..data.episodes import load_episode
from ..data.windows import build_windows, sample_mask_slots
from ..model.losses import compute_losses

RULE_REPLAY_DEFAULT = [
    "output/statickv_rule/episode_*", "output/blockfix/episode_*",
    "output/noblock/episode_*", "output/blockfix2/episode_*",
]
CF_REPLAY_DEFAULT = ["output/wm2_cf4_mid/*.npz", "output/wm2_cf4_nofire/*.npz"]


def _finetune_wm(model, heads, optimizer, online_groups, replay_groups, device, *,
                 steps, batch_size, online_share, rng, loss_config, mask_config, clip):
    """train_wm의 학습 스텝을 온라인 버퍼+리플레이 혼합으로 재현한다."""
    model.train(); heads.train()
    parameters = list(model.parameters()) + list(heads.parameters())
    pools = []
    if online_groups:
        w = np.asarray([len(g) for g in online_groups], dtype=np.float64)
        pools.append((online_groups, w / w.sum(), online_share))
    if replay_groups:
        w = np.asarray([len(g) for g in replay_groups], dtype=np.float64)
        pools.append((replay_groups, w / w.sum(), 1.0 - online_share if online_groups else 1.0))
    total_share = sum(p[2] for p in pools)
    last = {}
    for _ in range(steps):
        r = rng.random() * total_share
        for groups, weights, share in pools:
            if r < share:
                break
            r -= share
        g = groups[int(rng.choice(len(groups), p=weights))]
        size = min(batch_size, len(g))
        chosen = [g[i] for i in rng.choice(len(g), size=size, replace=False)]
        masks = [sample_mask_slots(w_, rng, mask_config) for w_ in chosen]
        batch = collate(chosen, masks, device)
        pred = tw._forward(model, heads, batch)
        losses = compute_losses(
            pred, batch["labels"],
            pos_loss_mask=batch["pos_loss_mask"],
            masked_units=batch["masked_units"], config=loss_config,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        last = {k: float(v) for k, v in losses.items()}
    model.eval(); heads.eval()
    return last


def _load_replay(patterns, count, rng):
    """rule 에피소드 리플레이 풀 — 홀드아웃 가드 포함, count개 표본."""
    from tqdm import tqdm

    dirs = tw._expand(patterns)
    if len(dirs) > count:
        dirs = [dirs[i] for i in rng.choice(len(dirs), size=count, replace=False)]
    signatures = lp.heldout_signatures()
    groups = []
    for d in tqdm(dirs, desc="rule 리플레이 로드", unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            continue
        lp.assert_not_heldout(episode, signatures)
        windows = build_windows(episode)
        if windows:
            groups.append(windows)
    return groups


def _load_cf_replay(patterns, count, rng):
    from tqdm import tqdm

    from ..data.counterfactual import load_windows

    files = sorted({
        f for p in patterns
        for f in (glob.glob(p) or glob.glob(str(lp._REPO_ROOT / p)))
        if f.endswith(".npz")
    })
    if len(files) > count:
        files = [files[i] for i in rng.choice(len(files), size=count, replace=False)]
    groups = []
    for f in tqdm(files, desc="CF 리플레이 로드", unit="npz"):
        windows = load_windows(Path(f))
        if windows:
            groups.append(windows)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="스냅샷 사본 경로 권장 (읽기 전용)")
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--scenario-dirs", nargs="+", required=True)
    parser.add_argument("--lam", type=float, default=0.83, help="γ^HORIZON — rtg V의 벨만 가중")
    parser.add_argument("--survival-beta", type=float, default=None,
                        help="온라인 V 라벨의 β (기본 config.SURVIVAL_BETA=1). "
                             "β ablation 시 꽂은 V의 학습 β와 반드시 일치시킬 것 — "
                             "불일치면 온라인 갱신이 V를 다른 목적함수로 끌어간다")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-root", default="output/wm2_loop_online")
    parser.add_argument("--v-update-every", type=int, default=10)
    parser.add_argument("--v-update-steps", type=int, default=200)
    parser.add_argument("--v-lr", type=float, default=1e-4)
    parser.add_argument("--wm-update-every", type=int, default=10)
    parser.add_argument("--wm-update-steps", type=int, default=200)
    parser.add_argument("--wm-lr", type=float, default=1e-5)
    parser.add_argument("--wm-batch-size", type=int, default=32)
    parser.add_argument("--online-share", type=float, default=0.5,
                        help="WM 갱신 배치 중 on-policy 윈도우 비중 (나머지 rule+CF 리플레이)")
    parser.add_argument("--replay-dirs", nargs="+", default=RULE_REPLAY_DEFAULT)
    parser.add_argument("--replay-episodes", type=int, default=150)
    parser.add_argument("--replay-cf-dirs", nargs="+", default=CF_REPLAY_DEFAULT)
    parser.add_argument("--replay-cf-files", type=int, default=40)
    parser.add_argument("--validation-dirs", nargs="+",
                        default=["output/validation_shared_v2/episode_*"])
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

    train_config = TrainConfig()
    loss_config = LossConfig()
    mask_config = MaskConfig()
    wm_optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(heads.parameters()),
        lr=args.wm_lr, weight_decay=train_config.weight_decay,
    )

    dirs = sorted({
        d for p in args.scenario_dirs
        for d in (glob.glob(p) or glob.glob(str(lp._REPO_ROOT / p)))
        if Path(d).is_dir() and (Path(d) / "config.json").exists()
    })
    signatures = lp.heldout_signatures()
    rng = np.random.default_rng(args.seed)
    chosen = [dirs[i] for i in rng.choice(len(dirs), size=min(args.pairs, len(dirs)), replace=False)]

    replay_rng = np.random.default_rng(args.seed + 1)
    replay_groups = _load_replay(args.replay_dirs, args.replay_episodes, replay_rng)
    replay_groups += _load_cf_replay(args.replay_cf_dirs, args.replay_cf_files, replay_rng)
    val_episodes = tw._load_all(args.validation_dirs)
    print(f"리플레이 그룹 {len(replay_groups)}개 "
          f"(windows={sum(len(g) for g in replay_groups)}) | 검증 에피소드 {len(val_episodes)}")

    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "loop_summary.jsonl"
    cem_config = lp.CEMConfig(
        candidates=args.candidates, elites=args.elites, iterations=args.iterations
    )

    wins = 0; totals = 0
    v_buffer = []
    online_groups = []
    live_pred, live_real = [], []
    holdout_pred, holdout_real = [], []
    progress_bar = tqdm(chosen, desc=f"loop online λ={args.lam:g}+WM", unit="ep")
    for pair_index, scenario_dir in enumerate(progress_bar):
        scenario = json.loads((Path(scenario_dir) / "config.json").read_text())
        sig_probe = type("S", (), {
            "run_dir": scenario_dir,
            "obstacles": tuple(tuple(float(v) for v in r) for r in scenario["obstacles"]),
        })
        lp.assert_not_heldout(sig_probe, signatures)
        layout = lp._layout_from_scenario(scenario)
        episode_seed = args.seed + pair_index * 2 + 0   # loop_lam lam-a 팔과 동일
        result = run_cem_episode(
            scenario=scenario, layout=layout, model=model, heads=heads,
            value_head=value_head, cem_config=cem_config, device=device,
            lam=args.lam, seed=episode_seed, duration=layout.duration_sec,
            label=f"[{pair_index+1}/{len(chosen)} λ={args.lam:g}+WM]",
            survival_beta=args.survival_beta,
        )
        totals += 1
        if result.outcome == "WIN":
            wins += 1
        run_name = f"episode_p{pair_index:03d}_lam{args.lam:g}"
        lp._save_episode_dir(output_root, run_name, result, scenario, args.lam, episode_seed)
        # on-policy 윈도우 풀 — 방금 저장한 디렉터리를 다시 읽어 기존 로더 경로를 재사용
        try:
            windows = build_windows(load_episode(str(output_root / run_name)))
            if windows:
                online_groups.append(windows)
        except (FileNotFoundError, ValueError) as e:
            print(f"\non-policy 윈도우 생성 실패 p{pair_index}: {e}", flush=True)

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
                "pair": pair_index, "lam": args.lam, "scenario": scenario_dir,
                "outcome": result.outcome, "final_progress": round(result.final_progress, 4),
                "planned": len(result.planned_commands), "v_pairs": len(result.v_pairs),
            }, ensure_ascii=False) + "\n")
        progress_bar.set_postfix(
            win=f"{wins}/{totals}",
            vbuf=sum(len(s.labels) for s in v_buffer),
            onwin=sum(len(g) for g in online_groups),
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

        if (pair_index + 1) % args.wm_update_every == 0 and (online_groups or replay_groups):
            losses = _finetune_wm(
                model, heads, wm_optimizer, online_groups, replay_groups, device,
                steps=args.wm_update_steps, batch_size=args.wm_batch_size,
                online_share=args.online_share, rng=rng,
                loss_config=loss_config, mask_config=mask_config,
                clip=train_config.gradient_clip_norm,
            )
            torch.save(
                {"model": model.state_dict(), "heads": heads.state_dict(),
                 "pair": pair_index + 1},
                output_root / "wm2_wm_online.pt",
            )
            _, red_m, stat_m = tw.validate(
                model, heads, val_episodes, device, loss_config, max_windows=400
            )
            model.eval(); heads.eval()   # validate가 train()으로 되돌려 놓는다
            table = " ".join(f"f{k+1}={red_m[k]:.1f}/{stat_m[k]:.1f}" for k in (1, 5))
            print(
                f"WM online 갱신 pair={pair_index+1} steps={args.wm_update_steps} "
                f"loss={losses.get('loss', float('nan')):.3f} dmg={losses.get('loss_dmg', float('nan')):.4f} "
                f"| 검증 RED오차/정지가정(m): {table}",
                flush=True,
            )

    print(f"\ndone λ={args.lam:g}+WM online: {wins}/{totals} 승")
    print(f"체크포인트: {output_root/'wm2_wm_online.pt'} / {output_root/'wm2_value_online.pt'}")


if __name__ == "__main__":
    main()
