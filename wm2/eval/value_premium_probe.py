"""접근 프리미엄 프로브 — V 재라벨의 게이트 (2026-08-14 계획 Step 2).

교착 거리대(기본 7.5~12u)의 destroy 계열 실측 상태에서, "전원 최근접 RED 방향으로
3u 전진한 상태"와 "제자리 상태"의 V 차이(ΔV)를 잰다. 기존 V15 상상판의 실측치는
+0.018(84% 양수)로, CEM 채점에서 정지의 팬텀 동점을 못 이기는 크기였다.

게이트 (사전 합의): 평균 ΔV ≥ +0.05 그리고 양수 비율 ≥ 80% → 통과.
- rtg 상상판 통과 → 라벨 정의가 범인. 다음 단계(전 스텝 feasibility 마스크) 진행.
- rtg 상상판 탈락 + rtg 실측판 통과 → WM 상상이 접근 신호를 뭉갬 → WM 재진단.
- 둘 다 탈락 → 라벨/입력이 아니라 특징 표현·데이터 문제 → 중지 후 재진단.

단순화: 전진 상태는 위치(x,y)만 이동하고 heading·속도는 그대로 둔다 — V 입력에서
위치 대비 부차 특징이며, 세 체크포인트가 같은 조건으로 비교되므로 게이트 판정에 공정.

    python -m wm2.eval.value_premium_probe \
        --episode-dirs 'output/blockfix2/episode_*' \
        --value-checkpoints checkpoints/wm2_value_run11.pt \
                            checkpoints/wm2_value_rtg_real.pt \
                            checkpoints/wm2_value_rtg_run12.pt \
        --states 50 --device cuda:0
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..model.features import TeamId, denorm_x, denorm_y, norm_x, norm_y
from ..value.head import load_value_head
from ..value import train_value as tv

X_NORM, Y_NORM, ALIVE = 3, 4, 7   # UNIT_FEATURES 인덱스
GATE_MEAN, GATE_POS = 0.05, 0.80


def _positions(uf_row: np.ndarray) -> np.ndarray:
    return np.stack(
        [[denorm_x(v) for v in uf_row[:, X_NORM]], [denorm_y(v) for v in uf_row[:, Y_NORM]]],
        axis=-1,
    )


def _stalemate_states(samples_list, *, dist_min, dist_max, rng, count):
    """(에피소드 인덱스, 표본 인덱스, 전진판 unit_features) 목록."""
    candidates = []
    for si, s in enumerate(samples_list):
        if s.mission_type not in (0, 1):   # destroy 계열만
            continue
        blue = np.asarray(s.team_ids) == int(TeamId.BLUE)
        for i in range(len(s.labels)):
            uf = s.unit_features[i]
            alive = uf[:, ALIVE] > 0.5
            ab, ar = alive & blue, alive & ~blue
            if not ab.any() or not ar.any():
                continue
            pos = _positions(uf)
            diff = pos[ab][:, None, :] - pos[ar][None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            if dist_min <= float(dists.min()) <= dist_max:
                candidates.append((si, i))
    if len(candidates) > count:
        pick = rng.choice(len(candidates), size=count, replace=False)
        candidates = [candidates[int(k)] for k in pick]
    return candidates


def _advanced_row(s, i: int, advance: float) -> np.ndarray:
    """살아있는 BLUE 전원을 각자 최근접 생존 RED 방향으로 advance만큼 이동시킨 사본."""
    uf = s.unit_features[i].copy()
    blue = np.asarray(s.team_ids) == int(TeamId.BLUE)
    alive = uf[:, ALIVE] > 0.5
    pos = _positions(uf)
    red_pos = pos[alive & ~blue]
    for ui in np.flatnonzero(alive & blue):
        vec = red_pos - pos[ui]
        d = np.linalg.norm(vec, axis=-1)
        j = int(np.argmin(d))
        if d[j] <= 1e-6:
            continue
        step = min(advance, max(d[j] - 0.1, 0.0))
        new = pos[ui] + vec[j] / d[j] * step
        uf[ui, X_NORM] = norm_x(float(new[0]))
        uf[ui, Y_NORM] = norm_y(float(new[1]))
    return uf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", default=["output/blockfix2/episode_*"])
    parser.add_argument("--value-checkpoints", nargs="+", required=True)
    parser.add_argument("--states", type=int, default=50)
    parser.add_argument("--dist-min", type=float, default=7.5)
    parser.add_argument("--dist-max", type=float, default=12.0)
    parser.add_argument("--advance", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    samples_list = tv.load_samples(args.episode_dirs)
    picked = _stalemate_states(
        samples_list, dist_min=args.dist_min, dist_max=args.dist_max,
        rng=rng, count=args.states,
    )
    print(f"교착 상태 {len(picked)}개 선택 (거리대 {args.dist_min}~{args.dist_max}u, "
          f"destroy 계열, 전진 {args.advance}u)")
    if not picked:
        raise SystemExit("교착 상태가 없다 — 거리대나 에피소드 소스를 확인")

    # 상태별 (제자리, 전진) 입력 쌍을 미리 구성
    pairs = []
    for si, i in picked:
        s = samples_list[si]
        adv = replace(
            s,
            unit_features=_advanced_row(s, i, args.advance)[None],
            mission_features=s.mission_features[i][None],
            labels=s.labels[i : i + 1],
            objective_dist=s.objective_dist[i : i + 1],
        )
        pairs.append((s, i, adv))

    print(f"\n게이트: 평균 ΔV ≥ +{GATE_MEAN} 그리고 양수 ≥ {GATE_POS:.0%}\n")
    for ckpt in args.value_checkpoints:
        model = load_value_head(Path(ckpt), device)
        model.eval()
        deltas = []
        with torch.no_grad():
            for s, i, adv in pairs:
                base = float(tv._forward(model, s, np.asarray([i]), device)[0])
                moved = float(tv._forward(model, adv, np.asarray([0]), device)[0])
                deltas.append(moved - base)
        deltas = np.asarray(deltas)
        mean, med, pos = deltas.mean(), np.median(deltas), (deltas > 0).mean()
        verdict = "통과" if (mean >= GATE_MEAN and pos >= GATE_POS) else "탈락"
        print(f"{Path(ckpt).name:<28} ΔV 평균 {mean:+.4f} 중앙값 {med:+.4f} "
              f"양수 {pos:.0%}  → {verdict}")


if __name__ == "__main__":
    main()
