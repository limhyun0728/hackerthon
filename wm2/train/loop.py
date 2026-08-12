"""4a-1 에피소딕 루프: λ 교대 A/B + V online 재라벨 (월드모델 동결).

    python -m wm2.train.loop \
        --checkpoint output/wm2_run2/wm2_best.pt \
        --value-checkpoint checkpoints/wm2_value_imagined.pt \
        --scenario-dirs 'output/blockfix/episode_*' \
        --pairs 100 --device cuda:0 --output-root output/wm2_loop1

- 시나리오는 기존 에피소드 config를 재사용한다 (스폰·맵·임무 그대로) — rule 런의
  실제 결과와 같은 판에서의 짝비교가 공짜로 생긴다.
- 같은 시나리오를 λ=1과 λ=0으로 연달아 돌린다 (짝지은 A/B).
- 매 재계획의 (상상 ŝ₆, 실제 15초 증가분) 짝을 모아 N쌍마다 V를 online 미세학습.
  V_live rho(수집 시점 V 예측 vs 실현)를 함께 찍는다 — online 학습이 듣는지의
  실시간 지표 (오프라인 진단의 V_cont −0.21에서 양수로 올라와야 한다).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import CEMConfig, ModelConfig
from ..data.scenarios import assert_not_heldout, heldout_signatures
from ..data.windows import EpisodeLayout
from ..model.heads import WM2Heads
from ..model.predictor import WM2Predictor
from ..model.features import MISSION_TYPE_BY_NAME, TeamId
from ..value.head import load_value_head, save_value_head
from ..value.train_value import EpisodeSamples, _forward, _spearman

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _layout_from_scenario(scenario: dict) -> EpisodeLayout:
    from ..data.windows import _terrain_features as build_terrain  # episodes 기반 헬퍼 재사용

    blue = tuple(sorted(int(v) for v in scenario["blue_ids"]))
    red = tuple(sorted(int(v) for v in scenario["red_ids"]))
    mission_raw = scenario.get("mission_type", "destroy_and_reach")
    mission = mission_raw if isinstance(mission_raw, int) else MISSION_TYPE_BY_NAME[str(mission_raw)]

    class _E:  # _terrain_features가 기대하는 최소 속성
        obstacles = tuple(tuple(float(v) for v in r) for r in scenario["obstacles"])

    team_ids = np.asarray(
        [int(TeamId.BLUE)] * len(blue) + [int(TeamId.RED)] * len(red), dtype=np.int64
    )
    return EpisodeLayout(
        unit_ids=blue + red,
        team_ids=team_ids,
        num_blue=len(blue),
        terrain_features=build_terrain(_E),
        mission_type=int(mission),
        objective=tuple(float(v) for v in scenario["objective"]),
        duration_sec=float(scenario.get("duration", 60.0)),
    )


def _samples_from_pairs(pairs, layout: EpisodeLayout) -> EpisodeSamples | None:
    labeled = [p for p in pairs if p.label is not None]
    if not labeled:
        return None
    return EpisodeSamples(
        unit_features=np.stack([p.unit_features for p in labeled]).astype(np.float32),
        mission_features=np.stack([p.mission_features for p in labeled]).astype(np.float32),
        terrain_features=layout.terrain_features,
        team_ids=np.asarray(layout.team_ids),
        labels=np.asarray([p.label for p in labeled], dtype=np.float32),
        objective_dist=np.full(len(labeled), np.nan, dtype=np.float32),
        mission_type=layout.mission_type,
    )


def _finetune_value(value_head, buffer, device, *, steps, lr, rng):
    optimizer = torch.optim.AdamW(value_head.parameters(), lr=lr, weight_decay=1e-4)
    weights = np.asarray([len(s.labels) for s in buffer], dtype=np.float64)
    weights /= weights.sum()
    value_head.train()
    for _ in range(steps):
        s = buffer[int(rng.choice(len(buffer), p=weights))]
        index = rng.choice(len(s.labels), size=min(64, len(s.labels)), replace=False)
        pred = _forward(value_head, s, index, device)
        target = torch.from_numpy(s.labels[index]).to(device)
        loss = (pred - target).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
        optimizer.step()
    value_head.eval()
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--scenario-dirs", nargs="+", required=True)
    parser.add_argument("--pairs", type=int, default=100, help="시나리오 쌍 수 (각 쌍 = λ1 + λ0)")
    parser.add_argument("--candidates", type=int, default=300, help="원본 Push-T와 동일 (300)")
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="output/wm2_loop")
    parser.add_argument("--v-update-every", type=int, default=10, help="N 쌍마다 V 미세학습")
    parser.add_argument("--v-update-steps", type=int, default=200)
    parser.add_argument("--v-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from tqdm import tqdm

    from ..sim.episode import run_cem_episode

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WM2Predictor(ModelConfig()).to(device); model.load_state_dict(payload["model"]); model.eval()
    heads = WM2Heads(ModelConfig()).to(device); heads.load_state_dict(payload["heads"]); heads.eval()
    value_head = load_value_head(Path(args.value_checkpoint), device)

    dirs = sorted({
        d for p in args.scenario_dirs
        for d in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p)))
        if Path(d).is_dir() and (Path(d) / "config.json").exists()
    })
    signatures = heldout_signatures()
    rng = np.random.default_rng(args.seed)
    chosen = [dirs[i] for i in rng.choice(len(dirs), size=min(args.pairs, len(dirs)), replace=False)]

    output_root = Path(args.output_root); output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "loop_summary.jsonl"
    cem_config = CEMConfig(
        candidates=args.candidates, elites=args.elites, iterations=args.iterations
    )

    wins = {0.0: 0, 1.0: 0}; totals = {0.0: 0, 1.0: 0}
    v_buffer: list[EpisodeSamples] = []
    live_pred, live_real = [], []
    # 매 5번째 쌍은 측정 전용: V 짝을 학습 버퍼에 넣지 않는다. V_live_rho(학습 풀,
    # prequential)와 별도로 "한 번도 학습에 안 쓴 시나리오"의 rho를 따로 찍어
    # V가 시나리오 풀을 외우는 오염을 감시한다.
    holdout_pred, holdout_real = [], []
    progress_bar = tqdm(chosen, desc="loop 4a-1", unit="pair")
    for pair_index, scenario_dir in enumerate(progress_bar):
        scenario = json.loads((Path(scenario_dir) / "config.json").read_text())
        # 홀드아웃 가드 (장애물 서명, duck-typed probe)
        sig_probe = type("S", (), {
            "run_dir": scenario_dir,
            "obstacles": tuple(tuple(float(v) for v in r) for r in scenario["obstacles"]),
        })
        assert_not_heldout(sig_probe, signatures)
        layout = _layout_from_scenario(scenario)
        for lam in (1.0, 0.0):
            episode_seed = args.seed + pair_index * 2 + int(lam)
            result = run_cem_episode(
                scenario=scenario, layout=layout, model=model, heads=heads,
                value_head=value_head, cem_config=cem_config, device=device,
                lam=lam, seed=episode_seed, duration=layout.duration_sec,
                label=f"[{pair_index+1}/{len(chosen)} λ={int(lam)}]",
            )
            totals[lam] += 1
            if result.outcome == "WIN":
                wins[lam] += 1
            is_holdout = pair_index % 5 == 4
            for p in result.v_pairs:
                if p.label is None:
                    continue
                if is_holdout:
                    holdout_pred.append(p.predicted_value); holdout_real.append(p.label)
                else:
                    live_pred.append(p.predicted_value); live_real.append(p.label)
            if not is_holdout:
                samples = _samples_from_pairs(result.v_pairs, layout)
                if samples is not None:
                    v_buffer.append(samples)
            with summary_path.open("a") as f:
                f.write(json.dumps({
                    "pair": pair_index, "lam": lam, "scenario": scenario_dir,
                    "outcome": result.outcome, "final_progress": round(result.final_progress, 4),
                    "planned": len(result.planned_commands), "v_pairs": len(result.v_pairs),
                }, ensure_ascii=False) + "\n")
        progress_bar.set_postfix(
            win1=f"{wins[1.0]}/{totals[1.0]}", win0=f"{wins[0.0]}/{totals[0.0]}",
            vbuf=sum(len(s.labels) for s in v_buffer),
        )
        if (pair_index + 1) % args.v_update_every == 0 and v_buffer:
            recent = min(len(live_pred), 200)
            rho_live = _spearman(np.asarray(live_pred[-recent:]), np.asarray(live_real[-recent:]))
            recent_h = min(len(holdout_pred), 200)
            rho_holdout = (
                _spearman(np.asarray(holdout_pred[-recent_h:]), np.asarray(holdout_real[-recent_h:]))
                if recent_h >= 10 else float("nan")
            )
            loss = _finetune_value(value_head, v_buffer, device, steps=args.v_update_steps, lr=args.v_lr, rng=rng)
            save_value_head(output_root / "wm2_value_online.pt", value_head)
            print(
                f"\nV online 갱신 pair={pair_index+1} buffer={sum(len(s.labels) for s in v_buffer)} "
                f"loss={loss:.5f} V_live_rho={rho_live:+.3f} V_holdout_rho={rho_holdout:+.3f} "
                f"(둘 다 양수로 올라와야 하고, 벌어지면 풀 암기)", flush=True,
            )

    print(f"\ndone λ=1: {wins[1.0]}/{totals[1.0]} 승  |  λ=0: {wins[0.0]}/{totals[0.0]} 승")
    print(f"V online 체크포인트: {output_root/'wm2_value_online.pt'}")


if __name__ == "__main__":
    main()
