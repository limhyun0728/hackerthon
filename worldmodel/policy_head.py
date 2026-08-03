"""상호작용 표현 위에서 joint action을 제안하는 policy head.

DEVS-CEM planner가 고른 최적 액션(planner의 결론)을 behavior cloning으로
암기해, CEM의 초기 제안 분포를 학습된 분포로 바꾼다. 평가는 계속 DEVS가
담당하므로 policy가 틀려도 결과의 정확성은 깨지지 않는다 — 제안 품질은
"policy 제안 후보가 DEVS 점수 상위에 드는 비율"로 잰다.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.worldmodel.actions import build_action_batch_from_v2_run
from hackerthon.worldmodel.cem_planner import (
    CEMConfig,
    CEMDistribution,
    build_initial_distribution,
    sample_future_action_plans,
    score_future_features_torch,
)
from hackerthon.worldmodel.devs_rollout import rollout_plans_with_devs, snapshot_from_slot_rows
from hackerthon.worldmodel.object_slot_attention import (
    ObjectSlotModelConfig,
    ObjectSlotTransformer,
    TypedObjectSlotEncoder,
    build_object_attention_mask,
)
from hackerthon.worldmodel.slots import ObjectType, SlotBatch, TeamId, build_slot_batch_from_v2_run, load_v2_config

MOVE_X_INDEX = 2
MOVE_Y_INDEX = 3
THETA_KNOWN_INDEX = 8
THETA_COS_INDEX = 9
THETA_SIN_INDEX = 10


@dataclasses.dataclass(frozen=True)
class PolicyHeadConfig:
    """policy head 구조 설정."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """구조 계약을 즉시 검증한다."""
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("embedding_dim/hidden_dim은 0보다 커야 한다")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers/num_heads는 0보다 커야 한다")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim은 num_heads로 나누어떨어져야 한다")


class PolicyHead(nn.Module):
    """객체 slot 위 cross-object attention으로 유닛별 액션 분포를 낸다."""

    def __init__(self, config: PolicyHeadConfig):
        super().__init__()
        self.config = config
        encoder_config = ObjectSlotModelConfig(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_predictor_layers=1,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.slot_encoder = TypedObjectSlotEncoder(encoder_config)
        self.interaction = ObjectSlotTransformer(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        dim = config.embedding_dim
        self.type_head = nn.Linear(dim, 4)
        self.move_head = nn.Linear(dim, 2)
        self.theta_head = nn.Linear(dim, 2)
        self.target_bilinear = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        type_ids: torch.Tensor,
        team_ids: torch.Tensor,
        alive_mask: torch.Tensor,
        blue_indices: torch.Tensor,
        red_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """현재 state에서 blue 유닛별 액션 분포 파라미터를 계산한다."""
        tokens = self.slot_encoder(features, feature_mask, type_ids, team_ids)
        allowed = build_object_attention_mask(type_ids, alive_mask)
        tokens = self.interaction(tokens, allowed)
        blue_tokens = tokens.index_select(1, blue_indices)
        red_tokens = tokens.index_select(1, red_indices)
        theta_vec = self.theta_head(blue_tokens)
        theta_vec = theta_vec / theta_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target_logits = torch.einsum(
            "bud,brd->bur", self.target_bilinear(blue_tokens), red_tokens
        ) / math.sqrt(self.config.embedding_dim)
        return {
            "type_logits": self.type_head(blue_tokens),
            "move_mu": torch.tanh(self.move_head(blue_tokens)),
            "theta_vec": theta_vec,
            "target_logits": target_logits,
        }


def _blue_red_indices(batch: SlotBatch) -> tuple[np.ndarray, np.ndarray]:
    """SlotBatch에서 blue/red unit slot index를 정렬 순서로 얻는다."""
    unit = batch.type_ids == int(ObjectType.UNIT)
    blue = np.flatnonzero(unit & (batch.team_ids == int(TeamId.BLUE)))
    red = np.flatnonzero(unit & (batch.team_ids == int(TeamId.RED)))
    if blue.size == 0 or red.size == 0:
        raise ValueError("blue/red unit slot이 없다")
    return blue, red


def build_tick_sample(run_dir: Path, time_sec: float) -> dict[str, np.ndarray]:
    """한 tick의 (state, planner가 고른 joint action) 학습 샘플을 만든다."""
    batch = build_slot_batch_from_v2_run(run_dir, time_sec)
    actions = build_action_batch_from_v2_run(run_dir, command_time_sec=time_sec, state_time_sec=time_sec)
    blue, red = _blue_red_indices(batch)
    red_ids = batch.entity_ids[red]
    target_index = np.full(actions.unit_ids.shape, -1, dtype=np.int64)
    for i, target_id in enumerate(actions.target_entity_ids):
        if int(target_id) >= 0:
            matches = np.flatnonzero(red_ids == int(target_id))
            if matches.size == 1:
                target_index[i] = int(matches[0])
    return {
        "features": batch.features,
        "feature_mask": batch.feature_mask,
        "type_ids": batch.type_ids,
        "team_ids": batch.team_ids,
        "alive_mask": batch.alive_mask,
        "blue_indices": blue,
        "red_indices": red,
        "issued": actions.issued_mask,
        "action_type": actions.action_type_ids,
        "move_dest": actions.features[:, [MOVE_X_INDEX, MOVE_Y_INDEX]],
        "theta_known": actions.features[:, THETA_KNOWN_INDEX] >= 0.5,
        "theta_vec": actions.features[:, [THETA_COS_INDEX, THETA_SIN_INDEX]],
        "target_index": target_index,
    }


def load_dataset(run_dirs: Iterable[Path]) -> list[dict[str, np.ndarray]]:
    """episode 디렉터리들에서 tick 샘플 전체를 모은다."""
    samples: list[dict[str, np.ndarray]] = []
    for run_dir in run_dirs:
        times: set[float] = set()
        import csv

        with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                times.add(float(row["time"]))
        for time_sec in sorted(times):
            try:
                samples.append(build_tick_sample(run_dir, time_sec))
            except ValueError:
                continue
    if not samples:
        raise ValueError("policy 학습 샘플이 없다")
    return samples


def _collate(samples: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    """같은 layout의 tick 샘플들을 batch tensor로 묶는다."""
    def stack(key, dtype):
        return torch.as_tensor(np.stack([s[key] for s in samples]), dtype=dtype, device=device)

    first = samples[0]
    return {
        "features": stack("features", torch.float32),
        "feature_mask": stack("feature_mask", torch.bool),
        "type_ids": stack("type_ids", torch.long),
        "team_ids": stack("team_ids", torch.long),
        "alive_mask": stack("alive_mask", torch.bool),
        "blue_indices": torch.as_tensor(first["blue_indices"], dtype=torch.long, device=device),
        "red_indices": torch.as_tensor(first["red_indices"], dtype=torch.long, device=device),
        "issued": stack("issued", torch.bool),
        "action_type": stack("action_type", torch.long),
        "move_dest": stack("move_dest", torch.float32),
        "theta_known": stack("theta_known", torch.bool),
        "theta_vec": stack("theta_vec", torch.float32),
        "target_index": stack("target_index", torch.long),
    }


def _categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """logit categorical 분포의 평균 엔트로피를 계산한다."""
    if logits.ndim != 2:
        raise ValueError(f"logits rank는 2이어야 한다: shape={tuple(logits.shape)}")
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()


def compute_policy_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    entropy_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """behavior cloning 손실을 계산한다. 명령이 내려간 유닛만 대상."""
    if entropy_weight < 0.0:
        raise ValueError("entropy_weight는 음수일 수 없다")
    issued = batch["issued"]
    if not bool(issued.any()):
        raise ValueError("issued 유닛이 하나도 없는 batch다")
    type_loss = F.cross_entropy(
        output["type_logits"][issued], batch["action_type"][issued].clamp_min(0)
    )
    is_move = issued & (batch["action_type"] == 1)
    move_loss = (
        F.mse_loss(output["move_mu"][is_move], batch["move_dest"][is_move])
        if bool(is_move.any())
        else output["move_mu"].new_zeros(())
    )
    is_engage = issued & (batch["action_type"] == 2) & (batch["target_index"] >= 0)
    target_loss = (
        F.cross_entropy(output["target_logits"][is_engage], batch["target_index"][is_engage])
        if bool(is_engage.any())
        else output["target_logits"].new_zeros(())
    )
    is_turn = issued & (batch["action_type"] == 3) & batch["theta_known"]
    theta_loss = (
        (1.0 - (output["theta_vec"][is_turn] * batch["theta_vec"][is_turn]).sum(-1)).mean()
        if bool(is_turn.any())
        else output["theta_vec"].new_zeros(())
    )
    # entropy 항은 policy가 CEM teacher를 너무 빨리 확신하지 않게 하는 exploration 보너스다.
    type_entropy = _categorical_entropy(output["type_logits"][issued])
    target_entropy = (
        _categorical_entropy(output["target_logits"][is_engage])
        if bool(is_engage.any())
        else output["target_logits"].new_zeros(())
    )
    entropy_bonus = type_entropy + target_entropy
    entropy_loss = -float(entropy_weight) * entropy_bonus
    loss = type_loss + move_loss + target_loss + 0.5 * theta_loss + entropy_loss
    return {
        "loss": loss,
        "type_loss": type_loss,
        "move_loss": move_loss,
        "target_loss": target_loss,
        "theta_loss": theta_loss,
        "entropy_loss": entropy_loss,
        "type_entropy": type_entropy,
        "target_entropy": target_entropy,
    }


def build_policy_guided_distribution(
    *,
    policy: PolicyHead,
    current_batch: SlotBatch,
    cem_config: CEMConfig,
    device: torch.device,
    prior_mix: float = 0.25,
) -> CEMDistribution:
    """policy 출력으로 CEM 초기 분포를 만든다. prior_mix만큼 기존 prior를 섞는다."""
    if not 0.0 <= prior_mix <= 1.0:
        raise ValueError("prior_mix는 [0, 1] 범위여야 한다")
    base = build_initial_distribution(current_batch, cem_config, device=device)
    blue, red = _blue_red_indices(current_batch)

    def tensor(value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=device).unsqueeze(0)

    policy.eval()
    with torch.no_grad():
        output = policy(
            features=tensor(current_batch.features, torch.float32),
            feature_mask=tensor(current_batch.feature_mask, torch.bool),
            type_ids=tensor(current_batch.type_ids, torch.long),
            team_ids=tensor(current_batch.team_ids, torch.long),
            alive_mask=tensor(current_batch.alive_mask, torch.bool),
            blue_indices=torch.as_tensor(blue, dtype=torch.long, device=device),
            red_indices=torch.as_tensor(red, dtype=torch.long, device=device),
        )
    horizon = base.action_probs.shape[0]
    action_probs = torch.softmax(output["type_logits"][0], dim=-1)
    action_probs = (1.0 - prior_mix) * action_probs + prior_mix * base.action_probs[0]
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    target_probs = torch.softmax(output["target_logits"][0], dim=-1)
    target_probs = (1.0 - prior_mix) * target_probs + prior_mix * base.target_probs[0]
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    move_mean = output["move_mu"][0]
    turn_mean = torch.atan2(output["theta_vec"][0, :, 1], output["theta_vec"][0, :, 0])
    expand = lambda value: value.unsqueeze(0).expand(horizon, *value.shape).contiguous()
    return dataclasses.replace(
        base,
        action_probs=expand(action_probs),
        target_probs=expand(target_probs),
        move_mean=expand(move_mean),
        turn_mean=expand(turn_mean),
    )


def save_policy(path: Path, policy: PolicyHead) -> None:
    """policy checkpoint를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": dataclasses.asdict(policy.config), "state_dict": policy.state_dict()},
        path,
    )


def load_policy(path: Path, device: torch.device) -> PolicyHead:
    """policy checkpoint를 복원한다."""
    payload = torch.load(path, map_location=device)
    policy = PolicyHead(PolicyHeadConfig(**payload["config"])).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def train_policy(args: argparse.Namespace) -> None:
    """episode 로그에서 policy head를 behavior cloning으로 학습한다."""
    if args.entropy_weight < 0.0:
        raise ValueError("--entropy-weight는 음수일 수 없다")
    device = torch.device(args.device)
    run_dirs = [d for root in args.run_dirs for d in sorted(Path(root).iterdir()) if d.is_dir()]
    samples = load_dataset(run_dirs)
    print(f"tick 샘플 {len(samples)}개 (episode dir {len(run_dirs)}개)")
    policy = PolicyHead(PolicyHeadConfig()).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        policy.train()
        order = torch.randperm(len(samples), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), args.batch_size):
            group = [samples[i] for i in order[start:start + args.batch_size]]
            batch = _collate(group, device)
            output = policy(
                features=batch["features"],
                feature_mask=batch["feature_mask"],
                type_ids=batch["type_ids"],
                team_ids=batch["team_ids"],
                alive_mask=batch["alive_mask"],
                blue_indices=batch["blue_indices"],
                red_indices=batch["red_indices"],
            )
            metrics = compute_policy_loss(output, batch, entropy_weight=args.entropy_weight)
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(metrics["loss"].detach()))
        print(f"epoch={epoch} loss={sum(losses)/len(losses):.4f}")
    save_policy(Path(args.out), policy)
    print(f"saved {args.out}")


def evaluate_proposer(args: argparse.Namespace) -> None:
    """policy 제안 분포와 기존 prior의 DEVS 점수 uplift를 비교한다."""
    import csv

    device = torch.device(args.device)
    policy = load_policy(Path(args.checkpoint), device)
    run_dirs = [d for root in args.run_dirs for d in sorted(Path(root).iterdir()) if d.is_dir()]
    cem = CEMConfig(
        num_candidates=args.candidates,
        num_elites=max(2, args.candidates // 8),
        num_iterations=1,
        future_horizon=args.horizon,
        seed=args.seed,
    )
    rows_out = []
    for run_dir in run_dirs:
        if len(rows_out) >= args.max_ticks:
            break
        config = load_v2_config(run_dir)
        times = set()
        with (run_dir / "commands_log.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                times.add(float(row["time"]))
        for time_sec in sorted(times)[:: args.tick_stride]:
            if len(rows_out) >= args.max_ticks:
                break
            try:
                batch = build_slot_batch_from_v2_run(run_dir, time_sec)
                snapshot_rows = []
                with (run_dir / "soldier_log.csv").open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if float(row["time"]) == time_sec:
                            snapshot_rows.append(
                                {
                                    "time": time_sec,
                                    "id": int(row["id"]),
                                    "x": float(row["x"]),
                                    "y": float(row["y"]),
                                    "heading": float(row["heading"]),
                                    "hp": float(row["hp"]),
                                    "ammo": int(float(row["ammo"])),
                                }
                            )
                snapshot = snapshot_from_slot_rows(
                    unit_rows=snapshot_rows,
                    obstacles=config["obstacles"],
                    base_time_sec=time_sec,
                    episode_duration_sec=float(config["duration"]),
                )
                scores = {}
                for name, dist_fn in (
                    ("prior", lambda: build_initial_distribution(batch, cem, device=device)),
                    ("policy", lambda: build_policy_guided_distribution(
                        policy=policy, current_batch=batch, cem_config=cem, device=device
                    )),
                ):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(args.seed)
                    plans = sample_future_action_plans(
                        distribution=dist_fn(), current_batch=batch, config=cem,
                        generator=generator, device=device,
                    )
                    features = rollout_plans_with_devs(
                        plans=plans, snapshot=snapshot, seed=args.seed, device=device
                    )
                    scores[name] = score_future_features_torch(
                        current_batch=batch, future_features=features
                    ).cpu().numpy()
                top_k = max(1, args.candidates // 8)
                rows_out.append(
                    {
                        "mean_uplift": float(scores["policy"].mean() - scores["prior"].mean()),
                        "top_uplift": float(
                            np.sort(scores["policy"])[-top_k:].mean()
                            - np.sort(scores["prior"])[-top_k:].mean()
                        ),
                        "best_uplift": float(scores["policy"].max() - scores["prior"].max()),
                    }
                )
            except ValueError:
                continue
    if not rows_out:
        raise ValueError("평가 가능한 tick이 없다")
    mean_u = np.mean([r["mean_uplift"] for r in rows_out])
    top_u = np.mean([r["top_uplift"] for r in rows_out])
    best_u = np.mean([r["best_uplift"] for r in rows_out])
    win = np.mean([r["top_uplift"] > 0 for r in rows_out])
    print(f"ticks={len(rows_out)}")
    print(f"평균 점수 uplift (policy - prior): {mean_u:+.3f}")
    print(f"상위 {max(1, args.candidates // 8)}개 평균 uplift: {top_u:+.3f}")
    print(f"최고 후보 uplift: {best_u:+.3f}")
    print(f"policy가 이긴 tick 비율(top 기준): {100 * win:.0f}%")


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint: train 또는 eval."""
    parser = argparse.ArgumentParser(description="policy head 학습/평가")
    sub = parser.add_subparsers(dest="mode", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("run_dirs", nargs="+")
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--learning-rate", type=float, default=3e-4)
    train_p.add_argument("--entropy-weight", type=float, default=0.05)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--device", type=str, default="cuda:0")
    train_p.add_argument("--out", type=str, default="checkpoints/policy_head.pt")
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("run_dirs", nargs="+")
    eval_p.add_argument("--checkpoint", type=str, required=True)
    eval_p.add_argument("--candidates", type=int, default=64)
    eval_p.add_argument("--horizon", type=int, default=6)
    eval_p.add_argument("--max-ticks", type=int, default=20)
    eval_p.add_argument("--tick-stride", type=int, default=7)
    eval_p.add_argument("--seed", type=int, default=7)
    eval_p.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.mode == "train":
        train_policy(args)
    else:
        evaluate_proposer(args)


if __name__ == "__main__":
    main()
