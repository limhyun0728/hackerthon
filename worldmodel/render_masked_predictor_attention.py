"""C-JEPA masked predictor의 시간-객체 attention을 시각화한다.

각 BLUE unit을 하나씩 강제로 self-mask한 뒤, 미래 BLUE query token이
어느 시간의 어느 객체 token을 참조하는지 확인한다.
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hackerthon.worldmodel.actions import unit_name
from hackerthon.worldmodel.object_slot_attention import DEVSObjectCentricWorldModel, ObjectSlotModelConfig
from hackerthon.worldmodel.slots import ObjectType
from hackerthon.worldmodel.train_object_centric_jepa import TrainingWindow, collate_training_batch, load_training_window


@dataclass(frozen=True)
class PredictorAttentionBundle:
    """미래 BLUE query별 predictor attention 결과."""

    slot_names: tuple[str, ...]
    object_slot_count: int
    type_ids: np.ndarray
    state_times: tuple[float, ...]
    query_labels: tuple[str, ...]
    query_frames: tuple[int, ...]
    query_slots: tuple[int, ...]
    layer_attention: np.ndarray
    aggregate_rows: tuple[dict[str, object], ...]


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> DEVSObjectCentricWorldModel:
    """checkpoint에서 model config와 weight를 읽어 평가용 모델을 만든다."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ObjectSlotModelConfig(**checkpoint["model_config"])
    model = DEVSObjectCentricWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_window(run_dir: Path, *, start_time: float, config: ObjectSlotModelConfig):
    """연속 state/action window를 로그에서 읽는다."""
    total_frames = config.history_frames + config.pred_frames
    state_times = tuple(float(start_time + offset) for offset in range(total_frames))
    action_times = tuple(state_times[:-1])
    spec = TrainingWindow(run_dir=run_dir, state_times=state_times, action_times=action_times)
    return load_training_window(spec)


def _prepare_input_with_forced_mask(
    *,
    model: DEVSObjectCentricWorldModel,
    history_tokens: torch.Tensor,
    action_tokens: torch.Tensor,
    masked_indices: torch.Tensor,
) -> torch.Tensor:
    """지정 slot의 t=1 이후 history를 query token으로 바꾼 predictor 입력을 만든다."""
    predictor = model.masked_predictor
    batch_size, history_frames, num_slots, embedding_dim = history_tokens.shape
    if action_tokens.shape[:2] != (batch_size, predictor.total_frames):
        raise ValueError("action_tokens shape 앞쪽은 (B, T_total)이어야 한다")
    if action_tokens.shape[-1] != embedding_dim:
        raise ValueError("action_tokens 마지막 차원은 history token embedding 차원과 같아야 한다")
    num_action_tokens = action_tokens.shape[2]
    masked_slot_mask = torch.zeros((batch_size, num_slots), dtype=torch.bool, device=history_tokens.device)
    masked_slot_mask[:, masked_indices] = True

    anchors = history_tokens[:, 0]
    anchor_queries = predictor.anchor_projector(anchors)
    query_grid = predictor.mask_token.expand(batch_size, predictor.total_frames, num_slots, embedding_dim)
    query_grid = query_grid + predictor.time_pos_embedding.expand(
        batch_size,
        predictor.total_frames,
        num_slots,
        embedding_dim,
    )
    query_grid = query_grid + anchor_queries.unsqueeze(1).expand(
        batch_size,
        predictor.total_frames,
        num_slots,
        embedding_dim,
    )
    model_input = query_grid.clone()

    # t=0은 모든 객체를 identity anchor로 보여준다.
    model_input[:, 0] = history_tokens[:, 0] + predictor.time_pos_embedding[:, 0]

    if history_frames > 1:
        unmasked_indices = torch.nonzero(~masked_slot_mask[0], as_tuple=False).flatten()
        if unmasked_indices.numel() > 0:
            history_pos = predictor.time_pos_embedding[:, 1:history_frames].expand(
                batch_size,
                history_frames - 1,
                num_slots,
                embedding_dim,
            )
            model_input[:, 1:history_frames, unmasked_indices] = (
                history_tokens[:, 1:, unmasked_indices]
                + history_pos[:, :, unmasked_indices]
            )
    # action은 객체 query에 더하지 않고 predictor slot 축 뒤의 visible token으로 붙인다.
    action_pos = predictor.time_pos_embedding.expand(
        batch_size,
        predictor.total_frames,
        num_action_tokens,
        embedding_dim,
    )
    action_input = action_tokens + action_pos
    return torch.cat([model_input, action_input], dim=2)


@torch.no_grad()
def _run_predictor_transformer_with_attention(
    transformer,
    flat_input: torch.Tensor,
    *,
    attn_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Predictor transformer를 실행하며 causal mask가 적용된 layer별 attention을 수집한다."""
    tokens = flat_input
    weights: list[torch.Tensor] = []
    for layer in transformer.layers:
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
    return tuple(weights)


def _key_category(
    *,
    key_frame: int,
    key_slot: int,
    query_slot: int,
    history_frames: int,
    slot_names: tuple[str, ...],
    object_slot_count: int,
    type_ids: np.ndarray,
) -> str:
    """attention key token의 의미 범주를 계산한다."""
    if key_slot >= object_slot_count:
        phase = "future" if key_frame >= history_frames else "history"
        source_name = slot_names[key_slot][2:] if slot_names[key_slot].startswith("A_") else slot_names[key_slot]
        if source_name == slot_names[query_slot]:
            return f"{phase}_self_action_tokens"
        if source_name.startswith("B"):
            return f"{phase}_other_blue_action_tokens"
        if source_name.startswith("R"):
            return f"{phase}_red_action_tokens"
        raise ValueError(f"정의되지 않은 action token 이름: {source_name}")
    if key_frame >= history_frames:
        return "future_query_tokens"
    if key_slot == query_slot and key_frame == 0:
        return "self_anchor_t0"
    if key_slot == query_slot:
        return "self_masked_history"
    object_type = ObjectType(int(type_ids[key_slot]))
    if object_type == ObjectType.UNIT:
        return "other_blue_history" if slot_names[key_slot].startswith("B") else "red_history"
    if object_type == ObjectType.TERRAIN:
        return "terrain_history"
    if object_type == ObjectType.MISSION:
        return "mission_history"
    raise ValueError(f"정의되지 않은 slot type: {type_ids[key_slot]}")


def _aggregate_attention(
    *,
    vector: np.ndarray,
    layer: int,
    query_label: str,
    query_slot: int,
    slot_names: tuple[str, ...],
    object_slot_count: int,
    type_ids: np.ndarray,
    state_times: tuple[float, ...],
    history_frames: int,
) -> tuple[dict[str, object], ...]:
    """query attention vector를 의미 범주별 합으로 요약한다."""
    num_slots = len(slot_names)
    totals: dict[str, float] = {}
    for flat_index, weight in enumerate(vector):
        key_frame = flat_index // num_slots
        key_slot = flat_index % num_slots
        category = _key_category(
            key_frame=key_frame,
            key_slot=key_slot,
            query_slot=query_slot,
            history_frames=history_frames,
            slot_names=slot_names,
            object_slot_count=object_slot_count,
            type_ids=type_ids,
        )
        totals[category] = totals.get(category, 0.0) + float(weight)
    return tuple(
        {
            "layer": layer,
            "query": query_label,
            "category": category,
            "weight_sum": totals[category],
        }
        for category in sorted(totals)
    )


@torch.no_grad()
def extract_self_masked_predictor_attention(
    *,
    model: DEVSObjectCentricWorldModel,
    run_dir: Path,
    start_time: float,
    device: torch.device,
) -> PredictorAttentionBundle:
    """각 BLUE unit을 self-mask하고 미래 query attention row를 뽑는다."""
    loaded_window = _load_window(run_dir, start_time=start_time, config=model.config)
    batch = collate_training_batch((loaded_window,), device=device)
    config = model.config
    total_frames = config.history_frames + config.pred_frames
    num_slots = batch.features.shape[2]

    history_tokens = model.encode_state_sequence(
        features=batch.features[:, :config.history_frames],
        feature_mask=batch.feature_mask[:, :config.history_frames],
        type_ids=batch.type_ids[:, :config.history_frames],
        team_ids=batch.team_ids[:, :config.history_frames],
        alive_mask=batch.alive_mask[:, :config.history_frames],
    )
    action_tokens = model.build_transition_action_tokens(
        source_tokens=history_tokens[:, 0],
        type_ids=batch.type_ids,
        team_ids=batch.team_ids,
        entity_ids=batch.entity_ids,
        action_features=batch.action_features,
        action_unit_ids=batch.action_unit_ids,
        issued_mask=batch.issued_mask,
    )

    object_slot_names = tuple(loaded_window.states[0].names)
    action_slot_names = tuple(f"A_{unit_name(int(unit_id))}" for unit_id in loaded_window.actions[0].unit_ids.tolist())
    slot_names = object_slot_names + action_slot_names
    type_ids = loaded_window.states[0].type_ids.copy()
    state_times = tuple(float(state.time_sec) for state in loaded_window.states)
    blue_slots = tuple(index for index, name in enumerate(object_slot_names) if name.startswith("B"))
    if not blue_slots:
        raise ValueError("BLUE slot이 없다")

    query_labels: list[str] = []
    query_frames: list[int] = []
    query_slots: list[int] = []
    layer_rows: list[list[np.ndarray]] = [[] for _ in range(config.num_predictor_layers)]
    aggregate_rows: list[dict[str, object]] = []

    for future_frame in range(config.history_frames, total_frames):
        for query_slot in blue_slots:
            masked_indices = torch.as_tensor([query_slot], dtype=torch.long, device=device)
            model_input = _prepare_input_with_forced_mask(
                model=model,
                history_tokens=history_tokens,
                action_tokens=action_tokens,
                masked_indices=masked_indices,
            )
            total_token_slots = num_slots + action_tokens.shape[2]
            flat_input = model_input.reshape(1, total_frames * total_token_slots, config.embedding_dim)
            attn_mask = model.masked_predictor.temporal_attention_mask(
                total_frames=total_frames,
                total_token_slots=total_token_slots,
                device=flat_input.device,
            )
            weights = _run_predictor_transformer_with_attention(
                model.masked_predictor.transformer,
                flat_input,
                attn_mask=attn_mask,
            )
            query_flat_index = future_frame * total_token_slots + query_slot
            query_label = f"t={state_times[future_frame]:.0f} {object_slot_names[query_slot]}"
            query_labels.append(query_label)
            query_frames.append(future_frame)
            query_slots.append(query_slot)
            for layer_index, layer_weight in enumerate(weights):
                head_mean = layer_weight[0].mean(dim=0).numpy()
                vector = head_mean[query_flat_index]
                layer_rows[layer_index].append(vector)
                aggregate_rows.extend(
                    _aggregate_attention(
                        vector=vector,
                        layer=layer_index + 1,
                        query_label=query_label,
                        query_slot=query_slot,
                        slot_names=slot_names,
                        object_slot_count=num_slots,
                        type_ids=type_ids,
                        state_times=state_times,
                        history_frames=config.history_frames,
                    )
                )

    layer_attention = np.stack([np.stack(rows, axis=0) for rows in layer_rows], axis=0)
    return PredictorAttentionBundle(
        slot_names=slot_names,
        object_slot_count=num_slots,
        type_ids=type_ids,
        state_times=state_times,
        query_labels=tuple(query_labels),
        query_frames=tuple(query_frames),
        query_slots=tuple(query_slots),
        layer_attention=layer_attention,
        aggregate_rows=tuple(aggregate_rows),
    )


def _plot_query_attention(bundle: PredictorAttentionBundle, out_path: Path) -> None:
    """미래 BLUE query별 key token attention heatmap을 저장한다."""
    num_layers, _, key_count = bundle.layer_attention.shape
    num_slots = len(bundle.slot_names)
    vmax = float(bundle.layer_attention.max())
    fig, axes = plt.subplots(num_layers, 1, figsize=(18.0, 3.2 * num_layers), dpi=150, squeeze=False)
    fig.suptitle("Masked Predictor Attention | self-masked BLUE future queries", fontsize=13)

    for layer_index in range(num_layers):
        ax = axes[layer_index][0]
        image = ax.imshow(bundle.layer_attention[layer_index], aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"predictor layer {layer_index + 1}")
        ax.set_yticks(np.arange(len(bundle.query_labels)))
        ax.set_yticklabels(bundle.query_labels, fontsize=7)
        centers = [frame * num_slots + (num_slots - 1) / 2.0 for frame in range(len(bundle.state_times))]
        ax.set_xticks(centers)
        ax.set_xticklabels([f"t={time_value:.0f}" for time_value in bundle.state_times], fontsize=8)
        for boundary in range(1, len(bundle.state_times)):
            ax.axvline(boundary * num_slots - 0.5, color="white", lw=0.8, alpha=0.9)
        for query_boundary in range(5, len(bundle.query_labels), 5):
            ax.axhline(query_boundary - 0.5, color="white", lw=0.8, alpha=0.9)
        ax.set_xlabel("Key tokens grouped by time; object slots first, then BLUE action tokens")
        ax.set_ylabel("Future query")
    fig.subplots_adjust(left=0.09, right=0.91, top=0.90, bottom=0.08, hspace=0.48)
    colorbar_axis = fig.add_axes((0.93, 0.16, 0.012, 0.68))
    fig.colorbar(image, cax=colorbar_axis, label="attention weight")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_top_pairs(bundle: PredictorAttentionBundle, out_path: Path, *, top_k: int) -> None:
    """미래 BLUE query별 높은 attention key token을 CSV로 저장한다."""
    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 한다")
    num_slots = len(bundle.slot_names)
    rows: list[dict[str, object]] = []
    for layer_index in range(bundle.layer_attention.shape[0]):
        for query_index, query_label in enumerate(bundle.query_labels):
            vector = bundle.layer_attention[layer_index, query_index]
            order = np.argsort(vector)[::-1][:top_k]
            for rank, key_flat_index in enumerate(order, start=1):
                key_frame = int(key_flat_index // num_slots)
                key_slot = int(key_flat_index % num_slots)
                rows.append(
                    {
                        "layer": layer_index + 1,
                        "query": query_label,
                        "rank": rank,
                        "key_time": bundle.state_times[key_frame],
                        "key_slot": bundle.slot_names[key_slot],
                        "key_is_same_unit": key_slot == bundle.query_slots[query_index],
                        "weight": float(vector[key_flat_index]),
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "query", "rank", "key_time", "key_slot", "key_is_same_unit", "weight"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate(bundle: PredictorAttentionBundle, out_path: Path) -> None:
    """의미 범주별 attention 합을 CSV로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "query", "category", "weight_sum"])
        writer.writeheader()
        writer.writerows(bundle.aggregate_rows)


def _save_npz(bundle: PredictorAttentionBundle, out_path: Path) -> None:
    """후처리용 원본 배열을 npz로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        slot_names=np.asarray(bundle.slot_names),
        type_ids=bundle.type_ids,
        state_times=np.asarray(bundle.state_times, dtype=np.float32),
        query_labels=np.asarray(bundle.query_labels),
        query_frames=np.asarray(bundle.query_frames, dtype=np.int64),
        query_slots=np.asarray(bundle.query_slots, dtype=np.int64),
        object_slot_count=np.asarray(bundle.object_slot_count, dtype=np.int64),
        layer_attention=bundle.layer_attention,
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    """CLI 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="C-JEPA masked predictor self-mask attention 분석")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    device = torch.device(args.device)
    model = _load_checkpoint_model(args.checkpoint_path, device)
    bundle = extract_self_masked_predictor_attention(
        model=model,
        run_dir=args.run_dir,
        start_time=args.start_time,
        device=device,
    )
    prefix = args.output_prefix
    if prefix is None:
        prefix = args.run_dir / f"masked_predictor_attention_t{args.start_time:g}"
    png_path = prefix.with_suffix(".png")
    top_path = prefix.with_name(prefix.name + "_top_pairs.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    npz_path = prefix.with_suffix(".npz")
    _plot_query_attention(bundle, png_path)
    _write_top_pairs(bundle, top_path, top_k=args.top_k)
    _write_aggregate(bundle, aggregate_path)
    _save_npz(bundle, npz_path)
    print(f"png={png_path}")
    print(f"top_pairs={top_path}")
    print(f"aggregate={aggregate_path}")
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()
