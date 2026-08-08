"""학습된 객체 중심 월드모델의 slot attention 행렬을 시각화한다.

행은 query slot, 열은 key slot이다. 값이 클수록 해당 query 객체가 다음 표현을
만들 때 그 key 객체 정보를 더 강하게 읽었다는 뜻이다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.worldmodel.actions import ACTION_DIM
from hackerthon.worldmodel.object_slot_attention import (
    DEVSObjectCentricWorldModel,
    ObjectSlotModelConfig,
    build_object_attention_mask,
)
from hackerthon.worldmodel.slots import ObjectType
from hackerthon.worldmodel.train_object_centric_jepa import TrainingWindow, collate_training_batch, load_training_window


@dataclass(frozen=True)
class AttentionBundle:
    """한 window에서 뽑은 attention 행렬 묶음."""

    slot_names: tuple[str, ...]
    type_ids: np.ndarray
    context: np.ndarray
    dynamics: np.ndarray
    start_time: float
    action_time: float


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> DEVSObjectCentricWorldModel:
    """checkpoint에서 model config와 weight를 읽어 평가용 모델을 만든다."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ObjectSlotModelConfig(**checkpoint["model_config"])
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_window(run_dir: Path, *, start_time: float, config: ObjectSlotModelConfig) -> tuple[TrainingWindow, object]:
    """attention 분석에 필요한 연속 state/action window를 로그에서 읽는다."""
    total_frames = config.history_frames + config.pred_frames
    state_times = tuple(float(start_time + offset) for offset in range(total_frames))
    action_times = tuple(state_times[:-1])
    spec = TrainingWindow(run_dir=run_dir, state_times=state_times, action_times=action_times)
    return spec, load_training_window(spec)


def _attention_mask_for_block(
    *,
    num_heads: int,
    attention_allowed: torch.Tensor,
) -> torch.Tensor:
    """MultiheadAttention에 넣을 head별 차단 mask를 만든다."""
    batch_size, num_slots, _ = attention_allowed.shape
    blocked = ~attention_allowed
    attn_mask = blocked.unsqueeze(1).expand(batch_size, num_heads, num_slots, num_slots)
    return attn_mask.reshape(batch_size * num_heads, num_slots, num_slots)


@torch.no_grad()
def _run_object_transformer_with_attention(
    transformer,
    tokens: torch.Tensor,
    attention_allowed: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """ObjectSlotTransformer를 실행하며 layer별 head attention을 함께 수집한다."""
    weights: list[torch.Tensor] = []
    for layer in transformer.layers:
        attn_mask = _attention_mask_for_block(num_heads=layer.num_heads, attention_allowed=attention_allowed)
        attn_input = layer.attn_norm(tokens)
        attn_out, attn_weight = layer.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + layer.ffn(layer.ffn_norm(tokens))
        weights.append(attn_weight.detach().cpu())
    return transformer.output_norm(tokens), tuple(weights)


@torch.no_grad()
def _extract_attention(
    *,
    model: DEVSObjectCentricWorldModel,
    run_dir: Path,
    start_time: float,
    action_index: int,
    device: torch.device,
) -> AttentionBundle:
    """지정 window에서 context/dynamics attention 행렬을 계산한다."""
    config = model.config
    _, loaded_window = _load_window(run_dir, start_time=start_time, config=config)
    if action_index < 0 or action_index >= config.history_frames + config.pred_frames - 1:
        raise ValueError("action_index가 window action 범위를 벗어났다")
    batch = collate_training_batch((loaded_window,), device=device)
    if batch.action_features.shape[-1] != ACTION_DIM:
        raise ValueError(f"action feature 마지막 차원은 {ACTION_DIM}이어야 한다")

    state_frame = action_index
    features = batch.features[:, state_frame]
    feature_mask = batch.feature_mask[:, state_frame]
    type_ids = batch.type_ids[:, state_frame]
    team_ids = batch.team_ids[:, state_frame]
    entity_ids = batch.entity_ids[:, state_frame]
    alive_mask = batch.alive_mask[:, state_frame]
    attention_allowed = build_object_attention_mask(type_ids, alive_mask)

    slot_tokens = model.slot_encoder(features, feature_mask, type_ids, team_ids)
    state_tokens, context_weights = _run_object_transformer_with_attention(
        model.context_encoder,
        slot_tokens,
        attention_allowed,
    )
    conditioned = model.action_conditioner(
        state_tokens,
        type_ids=type_ids,
        team_ids=team_ids,
        entity_ids=entity_ids,
        action_features=batch.action_features[:, action_index],
        action_unit_ids=batch.action_unit_ids[:, action_index],
        issued_mask=batch.issued_mask[:, action_index],
    )
    _, dynamics_weights = _run_object_transformer_with_attention(
        model.dynamics_predictor,
        conditioned,
        attention_allowed,
    )

    context = torch.stack(context_weights, dim=0).squeeze(1).numpy()
    dynamics = torch.stack(dynamics_weights, dim=0).squeeze(1).numpy()
    slot_names = tuple(loaded_window.states[state_frame].names)
    return AttentionBundle(
        slot_names=slot_names,
        type_ids=loaded_window.states[state_frame].type_ids.copy(),
        context=context,
        dynamics=dynamics,
        start_time=float(start_time),
        action_time=float(loaded_window.actions[action_index].time_sec),
    )


def _group_boundaries(type_ids: np.ndarray) -> tuple[int, ...]:
    """slot type이 바뀌는 경계 index를 계산한다."""
    boundaries: list[int] = []
    for index in range(1, type_ids.shape[0]):
        if int(type_ids[index]) != int(type_ids[index - 1]):
            boundaries.append(index)
    return tuple(boundaries)


def _draw_group_boundaries(ax: Axes, type_ids: np.ndarray) -> None:
    """unit/terrain/mission 구간을 heatmap 위에 선으로 표시한다."""
    for boundary in _group_boundaries(type_ids):
        line_position = boundary - 0.5
        ax.axhline(line_position, color="white", lw=0.8, alpha=0.85)
        ax.axvline(line_position, color="white", lw=0.8, alpha=0.85)


def _set_slot_ticks(ax: Axes, slot_names: tuple[str, ...], *, show_x: bool, show_y: bool) -> None:
    """heatmap 축에 slot 이름을 붙인다."""
    ticks = np.arange(len(slot_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(slot_names if show_x else [], rotation=90, fontsize=6)
    ax.set_yticklabels(slot_names if show_y else [], fontsize=6)
    ax.set_xlabel("Key slot" if show_x else "")
    ax.set_ylabel("Query slot" if show_y else "")


def _plot_attention_grid(
    *,
    bundle: AttentionBundle,
    out_path: Path,
    head: int | None,
) -> None:
    """context/dynamics layer별 attention heatmap을 하나의 PNG로 저장한다."""
    context = bundle.context.mean(axis=1) if head is None else bundle.context[:, head]
    dynamics = bundle.dynamics.mean(axis=1) if head is None else bundle.dynamics[:, head]
    if context.shape[0] != dynamics.shape[0]:
        raise ValueError("context와 dynamics layer 수가 같아야 한다")

    num_layers = context.shape[0]
    vmax = float(max(context.max(), dynamics.max()))
    fig, axes = plt.subplots(2, num_layers, figsize=(4.3 * num_layers, 8.8), dpi=150, squeeze=False)
    head_label = "head mean" if head is None else f"head {head}"
    fig.suptitle(
        f"Slot Attention Matrix | t={bundle.action_time:.1f} | {head_label} | row=query, col=key",
        fontsize=12,
    )
    rows = (("context encoder", context), ("action-conditioned dynamics", dynamics))
    image = None
    for row_index, (row_name, matrices) in enumerate(rows):
        for layer_index in range(num_layers):
            ax = axes[row_index][layer_index]
            image = ax.imshow(matrices[layer_index], cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"{row_name}\nlayer {layer_index + 1}")
            _draw_group_boundaries(ax, bundle.type_ids)
            _set_slot_ticks(
                ax,
                bundle.slot_names,
                show_x=row_index == len(rows) - 1,
                show_y=layer_index == 0,
            )
    if image is None:
        raise ValueError("attention matrix를 그리지 못했다")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.12, hspace=0.24, wspace=0.16)
    colorbar_axis = fig.add_axes((0.92, 0.18, 0.014, 0.62))
    fig.colorbar(image, cax=colorbar_axis, label="attention weight")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_top_pairs(bundle: AttentionBundle, out_path: Path, *, top_k: int) -> None:
    """BLUE query별 높은 attention key를 CSV로 저장한다."""
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 한다")
    rows: list[dict[str, object]] = []
    matrices_by_source = (
        ("context", bundle.context.mean(axis=1)),
        ("dynamics", bundle.dynamics.mean(axis=1)),
    )
    blue_indices = [index for index, name in enumerate(bundle.slot_names) if name.startswith("B")]
    for source, matrices in matrices_by_source:
        for layer_index, matrix in enumerate(matrices, start=1):
            for query_index in blue_indices:
                order = np.argsort(matrix[query_index])[::-1]
                selected = [int(index) for index in order if int(index) != query_index][:top_k]
                for rank, key_index in enumerate(selected, start=1):
                    rows.append(
                        {
                            "source": source,
                            "layer": layer_index,
                            "query": bundle.slot_names[query_index],
                            "rank": rank,
                            "key": bundle.slot_names[key_index],
                            "weight": float(matrix[query_index, key_index]),
                        }
                    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "layer", "query", "rank", "key", "weight"])
        writer.writeheader()
        writer.writerows(rows)


def _save_npz(bundle: AttentionBundle, out_path: Path) -> None:
    """후처리를 위해 원본 attention tensor를 npz로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        slot_names=np.asarray(bundle.slot_names),
        type_ids=bundle.type_ids,
        context=bundle.context,
        dynamics=bundle.dynamics,
        start_time=np.asarray([bundle.start_time], dtype=np.float32),
        action_time=np.asarray([bundle.action_time], dtype=np.float32),
    )


def _parse_head(value: str) -> int | None:
    """head 인자를 평균 또는 정수 index로 해석한다."""
    if value == "mean":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("head index는 음수일 수 없다")
    return parsed


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="학습된 CEM-JEPA slot attention 행렬 렌더링")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--head", type=_parse_head, default=None)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    device = torch.device(args.device)
    model = _load_checkpoint_model(args.checkpoint_path, device)
    if args.head is not None and args.head >= model.config.num_heads:
        raise ValueError(f"head index는 num_heads({model.config.num_heads})보다 작아야 한다")
    bundle = _extract_attention(
        model=model,
        run_dir=args.run_dir,
        start_time=args.start_time,
        action_index=args.action_index,
        device=device,
    )
    prefix = args.output_prefix
    if prefix is None:
        prefix = args.run_dir / f"attention_t{args.start_time:g}_a{args.action_index}"
    png_path = prefix.with_suffix(".png")
    csv_path = prefix.with_name(prefix.name + "_top_pairs.csv")
    npz_path = prefix.with_suffix(".npz")
    _plot_attention_grid(bundle=bundle, out_path=png_path, head=args.head)
    _write_top_pairs(bundle, csv_path, top_k=args.top_k)
    _save_npz(bundle, npz_path)
    print(f"png={png_path}")
    print(f"csv={csv_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
