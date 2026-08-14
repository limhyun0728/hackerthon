"""wm2 오프라인 학습 (설계 15절 3단계).

rule 에피소드 디렉터리에서 window를 만들어 월드모델을 학습한다. 시뮬 불필요.

    python -m hackerthon.wm2.train.train_wm \
        --episode-dirs 'output/statickv_rule/episode_*' 'output/blockfix/episode_*' \
        --device cuda:0 --steps 20000 --output-root output/wm2_run1

검증은 에피소드 단위로 분리(매 5번째)하고, 게이트 지표(RED 미래 위치 오차 vs 정지
가정)를 주기적으로 찍는다. 체크포인트는 latest와 best(검증 RED f2)를 남긴다.
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import LossConfig, MaskConfig, ModelConfig, RunConfig, TrainConfig
from ..data.batch import collate
from ..data.episodes import load_episode
from ..data.scenarios import assert_not_heldout, heldout_signatures
from ..data.windows import Window, build_windows, sample_mask_slots
from ..model.heads import WM2Heads
from ..model.losses import compute_losses
from ..model.predictor import WM2Predictor


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _expand(patterns: list[str]) -> list[str]:
    """glob을 cwd 기준으로 먼저, 안 잡히면 리포 루트 기준으로 푼다. 디렉터리만."""
    dirs: set[str] = set()
    for pattern in patterns:
        matched = glob.glob(pattern) or glob.glob(str(_REPO_ROOT / pattern))
        dirs.update(d for d in matched if Path(d).is_dir())
    return sorted(dirs)


def _load_all(patterns: list[str]) -> list[list[Window]]:
    from tqdm import tqdm

    dirs = _expand(patterns)
    if not dirs:
        raise ValueError(
            f"에피소드 디렉터리가 없다: {patterns} (cwd와 {_REPO_ROOT} 기준 모두 확인함)"
        )
    signatures = heldout_signatures()
    episodes: list[list[Window]] = []
    skipped = 0
    for d in tqdm(dirs, desc="window 생성", unit="ep"):
        try:
            episode = load_episode(d)
        except (FileNotFoundError, ValueError):
            skipped += 1
            continue
        assert_not_heldout(episode, signatures)
        windows = build_windows(episode)
        if windows:
            episodes.append(windows)
    print(f"episodes={len(episodes)} skipped={skipped} windows={sum(len(w) for w in episodes)}")
    return episodes


def _forward(model, heads, batch):
    out = model(
        unit_features=batch["unit_features"],
        terrain_features=batch["terrain_features"],
        mission_features=batch["mission_features"],
        actions=batch["actions"],
        team_ids=batch["team_ids"],
        masked_units=batch["masked_units"],
    )
    return heads(out["unit_tokens"], out["mission_tokens"])


@torch.no_grad()
def validate_counterfactual(model, heads, cf_groups, device, *, max_windows=600):
    """무작위 계획 하의 BLUE 위치 오차(m)와 전 유닛 피해 MAE(HP).

    BLUE 오차가 낮다 = 계획 토큰→실행 결과(우회·클램프·강등·사망)를 배웠다.
    피해 MAE가 낮다 = 무리한 ENGAGE에 데미지를 상상하지 않는다.
    """
    model.eval(); heads.eval()
    blue_err = np.zeros(6); dmg_abs = 0.0; blue_n = 0; dmg_n = 0
    budget = max_windows
    for windows in cf_groups:
        if budget <= 0:
            break
        chunk = windows[: min(len(windows), 24, budget)]
        budget -= len(chunk)
        batch = collate(chunk, [np.zeros(0, dtype=np.int64)] * len(chunk), device)
        pred = _forward(model, heads, batch)
        blue = batch["team_ids"] == 0
        alive = batch["pos_loss_mask"][:, blue]
        diff = pred["dpos"][:, 2:][:, :, blue] - batch["labels"]["dpos"][:, 2:][:, :, blue]
        err = diff.norm(dim=-1)
        mask = alive.unsqueeze(1).expand_as(err)
        for k in range(6):
            blue_err[k] += float(err[:, k][mask[:, k]].sum())
        blue_n += int(mask[:, 0].sum())
        alive_all = batch["pos_loss_mask"]
        dmg_diff = (pred["ddmg"][:, 2:] - batch["labels"]["ddmg"][:, 2:]).abs()
        mask_all = alive_all.unsqueeze(1).expand_as(dmg_diff)
        dmg_abs += float(dmg_diff[mask_all].sum())
        dmg_n += int(mask_all.sum())
    model.train(); heads.train()
    return blue_err / max(1, blue_n) * 10.0, dmg_abs / max(1, dmg_n) * 100.0


@torch.no_grad()
def validate(model, heads, val_episodes, device, loss_config, *, max_windows=400):
    """검증 손실 + RED 미래 위치 오차(m) vs 정지 가정. 마스킹 없이 잰다."""
    model.eval(); heads.eval()
    losses_sum: dict[str, float] = {}
    error_sum = np.zeros(6); stationary_sum = np.zeros(6); count = 0
    batches = 0
    budget = max_windows
    for windows in val_episodes:
        if budget <= 0:
            break
        size = min(len(windows), 32, budget)
        # 에피소드 앞부분만 보지 않게 window를 고르게 편다
        indices = np.unique(np.linspace(0, len(windows) - 1, size).astype(int))
        chunk = [windows[i] for i in indices]
        budget -= len(chunk)
        empty = [np.zeros(0, dtype=np.int64)] * len(chunk)
        batch = collate(chunk, empty, device)
        pred = _forward(model, heads, batch)
        losses = compute_losses(
            pred, batch["labels"],
            pos_loss_mask=batch["pos_loss_mask"],
            masked_units=batch["masked_units"], config=loss_config,
        )
        batches += 1
        for key, value in losses.items():
            losses_sum[key] = losses_sum.get(key, 0.0) + float(value)
        red = batch["team_ids"] == 1
        alive = batch["pos_loss_mask"][:, red]                       # (B, R)
        diff = (pred["dpos"][:, 2:, :, :][:, :, red] - batch["labels"]["dpos"][:, 2:, :, :][:, :, red])
        err = diff.norm(dim=-1)                                       # (B, 6, R) f1..f6
        # 정지가정 기준선: "현재(h2) 위치에서 안 움직인다"의 오차 = 현재 이후 실제 변위.
        # 레이블 잔차는 anchor(a) 기준이므로 h2 잔차를 빼서 현재 기준으로 되돌린다.
        # (a 기준 그대로 쓰면 이미 관측된 2틱 이동까지 기준선에 얹혀 기준선이 부풀고,
        #  구 측정표의 t+k 정지가정 수치와 비교할 수 없게 된다)
        stat = (
            batch["labels"]["dpos"][:, 2:, :, :][:, :, red]
            - batch["labels"]["dpos"][:, 1:2, :, :][:, :, red]
        ).norm(dim=-1)
        mask3 = alive.unsqueeze(1).expand_as(err)
        for k in range(6):
            error_sum[k] += float(err[:, k][mask3[:, k]].sum())
            stationary_sum[k] += float(stat[:, k][mask3[:, k]].sum())
        count += int(mask3[:, 0].sum())
    model.train(); heads.train()
    mean_losses = {k: v / max(1, batches) for k, v in losses_sum.items()}
    red_error_m = error_sum / max(1, count) * 10.0
    stationary_m = stationary_sum / max(1, count) * 10.0
    return mean_losses, red_error_m, stationary_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dirs", nargs="+", required=True)
    parser.add_argument(
        "--validation-dirs",
        nargs="+",
        default=["output/validation_shared_v2/episode_*"],
        help="전용 검증 에피소드 (rule 생성 공유 검증셋). 주면 학습 glob은 전부 학습에 쓴다",
    )
    parser.add_argument(
        "--counterfactual-dirs",
        nargs="+",
        default=[],
        help="counterfactual.py가 만든 npz glob. 무리한 계획의 실행 의미론 커버리지 (2b)",
    )
    parser.add_argument(
        "--cf-repeat", type=int, default=1,
        help="CF window 그룹 반복 계수 — rule 데이터 대비 실행 의미론의 gradient 비중 확대",
    )
    parser.add_argument(
        "--val-counterfactual-dirs",
        nargs="+",
        default=[],
        help="검증 에피소드에서 만든 CF npz. 무작위 계획 하의 BLUE 실행 의미론을 잰다",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="output/wm2_run")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    # early stopping: 게이트 지표(검증 RED f2, m)가 patience회 연속 개선 없으면 중단.
    # 개선 인정은 min-delta(m)를 넘어야 한다 — 검증 잡음으로 best가 우연히 갱신되는 것 방지.
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.1)
    args = parser.parse_args()

    config = RunConfig(
        train=TrainConfig(
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, device=args.device,
        ),
        output_root=args.output_root,
    )
    snapshot = config.snapshot()
    print(f"config_snapshot={snapshot}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    if args.validation_dirs:
        train_episodes = _load_all(args.episode_dirs)
        val_episodes = _load_all(args.validation_dirs)
        # 같은 run 디렉터리가 양쪽 glob에 걸리면 검증이 무의미해진다
        train_dirs = set(_expand(args.episode_dirs))
        val_dirs = set(_expand(args.validation_dirs))
        overlap = train_dirs & val_dirs
        if overlap:
            raise ValueError(f"학습/검증 에피소드가 겹친다: {sorted(overlap)[:3]} ...")
    else:
        episodes = _load_all(args.episode_dirs)
        val_episodes = [w for i, w in enumerate(episodes) if i % 5 == 4]
        train_episodes = [w for i, w in enumerate(episodes) if i % 5 != 4]
    if args.counterfactual_dirs:
        from ..data.counterfactual import load_windows

        cf_files = sorted(
            {
                f
                for p in args.counterfactual_dirs
                for f in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p)))
                if f.endswith(".npz")
            }
        )
        # 교차 셋 중복 제거: 여러 CF 셋이 같은 에피소드의 같은 base tick을 뽑으면
        # 결정적 패턴(hold/close/approach)의 계획이 똑같이 재생성된다 — run7 실측:
        # loop3 교착대 hold/close 고유 404개 중 145개 2중, 37개 3중 수록. cf-repeat와
        # 곱해져 특정 (상태, 계획)만 최대 6배 가중되어 상태 조건부 피해 평균을 끌어올렸다.
        # 키 = (에피소드 npz 이름, anchor tick, 계획 토큰) — 먼저 온 셋의 창이 남는다.
        seen_plans: set = set()
        dropped = 0
        cf_groups = []
        for f in cf_files:
            kept = []
            for w in load_windows(Path(f)):
                key = (Path(f).name, w.anchor_tick, hash(w.actions.tobytes()))
                if key in seen_plans:
                    dropped += 1
                    continue
                seen_plans.add(key)
                kept.append(w)
            if kept:
                cf_groups.append(kept)
        if dropped:
            print(f"counterfactual 교차 중복 제거: {dropped} windows")
        # 반복 계수: CF는 rule 대비 window 수가 1/10 수준이라 그대로 섞으면 실행
        # 의미론(사거리·LOS 게이팅)의 gradient가 밀린다 (run5b: 표본 밀집 대역만 보정).
        train_episodes += cf_groups * max(1, args.cf_repeat)
        cf_windows = sum(len(g) for g in cf_groups)
        rule_windows = sum(len(g) for g in train_episodes) - cf_windows * max(1, args.cf_repeat)
        share = cf_windows * max(1, args.cf_repeat) / max(1, rule_windows + cf_windows * max(1, args.cf_repeat))
        print(
            f"counterfactual groups={len(cf_groups)} windows={cf_windows} "
            f"repeat={max(1, args.cf_repeat)} -> 학습 비중 {share:.0%}"
        )
    cf_val_groups = []
    if args.val_counterfactual_dirs:
        from ..data.counterfactual import load_windows as _load_cf

        cf_val_files = sorted(
            {
                f
                for p in args.val_counterfactual_dirs
                for f in (glob.glob(p) or glob.glob(str(_REPO_ROOT / p)))
                if f.endswith(".npz")
            }
        )
        cf_val_groups = [g for g in (_load_cf(Path(f)) for f in cf_val_files) if g]
        print(f"cf_val groups={len(cf_val_groups)} windows={sum(len(g) for g in cf_val_groups)}")
    print(f"train_episodes={len(train_episodes)} val_episodes={len(val_episodes)}")

    model = WM2Predictor(config.model).to(device)
    heads = WM2Heads(config.model).to(device)
    parameters = list(model.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    loss_config = config.loss
    mask_config = config.mask

    # window 수 비례로 episode를 뽑아 window 균등 추출과 같게 만든다
    weights = np.asarray([len(w) for w in train_episodes], dtype=np.float64)
    weights = weights / weights.sum()

    checkpoint_dir = Path(args.output_root)
    best_red_f2 = float("inf")   # 정체 판정 기준값 (min-delta 넘는 개선만 갱신)
    best_saved_f2 = float("inf")  # wm2_best.pt에 실제 저장된 f2
    bad_checks = 0
    started = time.time()

    for step in range(1, args.steps + 1):
        windows = train_episodes[int(rng.choice(len(train_episodes), p=weights))]
        size = min(config.train.batch_size, len(windows))
        chosen = [windows[i] for i in rng.choice(len(windows), size=size, replace=False)]
        masks = [sample_mask_slots(w, rng, mask_config) for w in chosen]
        batch = collate(chosen, masks, device)

        pred = _forward(model, heads, batch)
        losses = compute_losses(
            pred, batch["labels"],
            pos_loss_mask=batch["pos_loss_mask"],
            masked_units=batch["masked_units"], config=loss_config,
        )
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.train.gradient_clip_norm)
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"step={step} loss={float(losses['loss']):.4f} pos={float(losses['loss_pos']):.4f} "
                f"dmg={float(losses['loss_dmg']):.4f} ammo={float(losses['loss_ammo']):.4f} "
                f"head={float(losses['loss_heading']):.4f} comp={float(losses['loss_completion']):.4f} "
                f"elapsed={time.time()-started:.0f}s",
                flush=True,
            )

        if step % args.val_every == 0 and val_episodes:
            _, red_m, stat_m = validate(
                model, heads, val_episodes, device, loss_config,
                max_windows=2500 if args.validation_dirs else 400,
            )
            table = " ".join(f"f{k+1}={red_m[k]:.1f}/{stat_m[k]:.1f}" for k in range(6))
            payload = {
                "model": model.state_dict(),
                "heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {"model": vars(config.model) if hasattr(config.model, "__dict__") else None},
                "step": step,
                "red_error_m": red_m.tolist(),
            }
            torch.save(payload, checkpoint_dir / "wm2_latest.pt")
            # best 저장은 절대 개선이면 충분하다 — 검증 셋이 고정이고 eval이 결정적이라
            # 측정 잡음이 없다. min-delta에 묶으면 느린 개선 구간에서 best가 동결된다
            # (run5 실측: f2 2.7→2.6 개선이 저장 안 됨). min-delta는 아래 정체 판정 전용.
            if red_m[1] < best_saved_f2:
                best_saved_f2 = red_m[1]
                torch.save(payload, checkpoint_dir / "wm2_best.pt")
            # early stopping: 게이트 지표(RED f2)가 min-delta 넘게 좋아져야 개선으로 인정
            if red_m[1] < best_red_f2 - args.early_stop_min_delta:
                best_red_f2 = red_m[1]
                bad_checks = 0
            else:
                bad_checks += 1
            cf_text = ""
            if cf_val_groups:
                blue_m, dmg_mae = validate_counterfactual(model, heads, cf_val_groups, device)
                cf_text = (
                    f" | CF BLUE f2={blue_m[1]:.1f} f6={blue_m[5]:.1f}m dmgMAE={dmg_mae:.2f}HP"
                )
            print(
                f"validation step={step} RED오차/정지가정(m): {table} "
                f"best_f2={best_saved_f2:.1f} bad={bad_checks}/{args.early_stop_patience}{cf_text}",
                flush=True,
            )
            if args.early_stop_patience > 0 and bad_checks >= args.early_stop_patience:
                print(f"early_stopping step={step} best_red_f2={best_red_f2:.1f}m", flush=True)
                break

    print(f"done steps={step} best_red_f2={best_red_f2:.1f}m elapsed={time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
