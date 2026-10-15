"""Paper seven-step Waymo rollout training with frozen VGGT-Omega.

Training encodes the 12 observed images once, uses one S3-FRONT paired
VGGT/LiDAR metric scale, and optimizes the autoregressive S4--S10 rollout
through decoded FRONT depth. Growing-prefix states are used only for the
teacher-forced validation diagnostics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, Sampler

from GeoState.core.geometry import paired_depth_scale
from GeoState.evaluation.checkpoint import load_geostate_checkpoint as _load_model
from GeoState.models.world_model import SceneStatePredictorConfig, build_next_scene_state_predictor
from GeoState.training.waymo_online_one_step import (
    _atomic_json_dump,
    _atomic_torch_save,
    _cleanup_distributed,
    _distributed_context,
    _set_seed,
)
from GeoState.training.waymo_one_step import (
    ManifestWaymoDataset,
    TARGET_STATE_INDICES,
    _decode_meters,
    _region_log_loss,
)


ROLLOUT_STEPS = len(TARGET_STATE_INDICES)
TRAIN_ROLLOUT_STEPS = ROLLOUT_STEPS


class _RankShardSampler(Sampler[int]):
    """Deterministically shard a dataset across DDP ranks."""

    def __init__(
        self,
        length: int,
        rank: int,
        world_size: int,
        *,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.length, generator=generator).tolist()
        else:
            indices = list(range(self.length))
        if self.drop_last:
            indices = indices[: self.length - (self.length % self.world_size)]
        return iter(indices[self.rank :: self.world_size])

    def __len__(self) -> int:
        if self.drop_last:
            return self.length // self.world_size
        return (self.length - self.rank + self.world_size - 1) // self.world_size


class _ResumeOffsetRankSampler(_RankShardSampler):
    """Skip already-consumed local indices only in the resumed first epoch."""

    def __init__(self, *args: Any, start_offset: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_offset = int(start_offset)
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")

    def __iter__(self):
        iterator = super().__iter__()
        if self.epoch == 0 and self.start_offset:
            return itertools.islice(iterator, self.start_offset, None)
        return iterator

    def __len__(self) -> int:
        length = super().__len__()
        return max(0, length - self.start_offset) if self.epoch == 0 else length


def _collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _replace_hardlink(source: Path, destination: Path) -> None:
    """Atomically repoint a small alias without duplicating a multi-GB checkpoint."""
    temporary = destination.with_name(destination.name + ".tmp-link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)


def _loader(
    dataset: ManifestWaymoDataset,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    start_offset: int = 0,
) -> tuple[DataLoader, _RankShardSampler]:
    sampler = _ResumeOffsetRankSampler(
        len(dataset), rank, world_size, shuffle=shuffle, drop_last=drop_last, seed=seed,
        start_offset=start_offset,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        multiprocessing_context="spawn" if workers > 0 else None,
        drop_last=drop_last,
    )
    return loader, sampler


def _next_batch(iterator: Any, loader: DataLoader, sampler: _RankShardSampler, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def _anchor_mask(batch: int, views: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch, views), dtype=torch.bool, device=device)
    mask[:, 0] = True
    return mask


def _context(encoding: Any, item: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "reference_c2w_student": encoding.reference_c2w_student[batch_index].float().cpu(),
        "dataset_to_student_scale": encoding.dataset_to_student.scale[batch_index].float().cpu(),
        "dataset_to_student_rotation": encoding.dataset_to_student.rotation[batch_index].float().cpu(),
        "dataset_to_student_translation": encoding.dataset_to_student.translation[batch_index].float().cpu(),
        "scene_scale": encoding.scene_scale[batch_index].float().cpu(),
        "target_c2w": item["target_c2w"].float().cpu(),
        "target_intrinsics": item["target_intrinsics"].float().cpu(),
        "target_depth": item["target_depth"].half().cpu(),
        "target_valid": item["target_valid"].cpu(),
        "motion_mask": item["motion_mask"].cpu(),
        "window_id": item["window_id"],
    }


@torch.inference_mode()
def extract_growing_rollout_states(
    geometry_model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    future_steps: int = ROLLOUT_STEPS,
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, torch.Tensor] | None:
    """Run the exact 12/15/.../33 protocol and return S0--S10.

    Every pass estimates its own dataset->VGGT gauge from the same S3 FRONT
    image and the same condition LiDAR.  Its geometry is then canonicalized to
    the physical scene unit fixed by the 12-image pass.  No future LiDAR enters
    either training or inference.
    """
    batch = len(items)
    images = torch.stack([item["images"] for item in items]).to(device, non_blocking=True)
    c2w = torch.stack([item["input_c2w"] for item in items]).to(device, non_blocking=True)
    intrinsics = torch.stack([item["input_intrinsics"] for item in items]).to(
        device, non_blocking=True
    )
    view_states = torch.stack([item["view_states"] for item in items]).to(
        device, non_blocking=True
    )

    output12 = geometry_model.vggt(images[:, :12])
    lidar = torch.stack([item["input_scale_depth"] for item in items]).to(
        device, non_blocking=True
    )
    paired_scale, scale_valid, scale_support = paired_depth_scale(
        output12.depth[:, 0], lidar, output12.confidence[:, 0]
    )
    if batch == 1 and not bool(scale_valid[0]):
        return None
    history_mask = torch.stack(
        [(view_states[:, :12] == state) for state in range(4)], dim=1
    )
    history, history_encoding = geometry_model.encode_temporal_scene_states_from_vggt_output(
        output12,
        c2w[:, :12],
        input_intrinsics=intrinsics[:, :12],
        state_view_mask=history_mask,
        single_view_scale=paired_scale,
        alignment_view_mask=_anchor_mask(batch, 12, device),
        scene_scale_view_mask=torch.ones((batch, 12), dtype=torch.bool, device=device),
        frame_ids=[item["frame_ids"][:12] for item in items],
        reference_index=0,
    )
    history_alignment = history_encoding.metadata["alignment"]
    # L0 is the one physical unit available at inference: meters represented by
    # one condition-12 normalized scene unit.  It remains fixed for S0--S10.
    condition_meters_per_unit = (
        history_encoding.scene_scale / paired_scale.float().clamp_min(1e-8)
    )
    contexts = [_context(history_encoding, item, index) for index, item in enumerate(items)]
    valid = scale_valid.bool() & history_alignment.valid.bool()
    states = [history]
    del output12, history_encoding

    if not 0 <= future_steps <= ROLLOUT_STEPS:
        raise ValueError(f"future_steps must be in [0, {ROLLOUT_STEPS}]")
    for target_state in TARGET_STATE_INDICES[:future_steps]:
        views = 3 * (target_state + 1)
        output = geometry_model.vggt(images[:, :views])
        prefix_scale, prefix_scale_valid, _ = paired_depth_scale(
            output.depth[:, 0], lidar, output.confidence[:, 0]
        )
        # Student units = prefix_scale * meters.  Dividing all geometry by
        # prefix_scale * L0 therefore gives meters / L0, exactly the coordinate
        # system used by the condition-12 decoder context.
        canonical_scene_scale = prefix_scale * condition_meters_per_unit
        target_mask = (view_states[:, :views] == target_state)[:, None]
        target, encoding = geometry_model.encode_temporal_scene_states_from_vggt_output(
            output,
            c2w[:, :views],
            input_intrinsics=intrinsics[:, :views],
            state_view_mask=target_mask,
            single_view_scale=prefix_scale,
            scene_scale_override=canonical_scene_scale,
            alignment_view_mask=_anchor_mask(batch, views, device),
            scene_scale_view_mask=view_states[:, :views] < 4,
            frame_ids=[item["frame_ids"][:views] for item in items],
            reference_index=0,
        )
        valid &= prefix_scale_valid.bool() & encoding.metadata["alignment"].valid.bool()
        states.append(target)
        del output, encoding
    del images, c2w, intrinsics, view_states
    return torch.cat(states, dim=1).detach(), contexts, valid, scale_support


def _rollout(
    predictor: torch.nn.Module,
    history: torch.Tensor,
    *,
    use_checkpoint: bool,
    steps: int | None = None,
) -> torch.Tensor:
    window = history
    predictions: list[torch.Tensor] = []
    rollout_steps = ROLLOUT_STEPS if steps is None else int(steps)
    if not 1 <= rollout_steps <= ROLLOUT_STEPS:
        raise ValueError(
            f"rollout steps must be in [1, {ROLLOUT_STEPS}], got {rollout_steps}"
        )
    for _ in range(rollout_steps):
        if use_checkpoint and torch.is_grad_enabled():
            previous_based = activation_checkpoint(predictor, window, use_reentrant=False)
        else:
            previous_based = predictor(window)
        next_state = previous_based
        predictions.append(next_state)
        window = torch.cat((window[:, 1:], next_state[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def _decode_rollout(
    geometry_model: torch.nn.Module,
    predicted: torch.Tensor,
    contexts: list[dict[str, Any]],
    device: torch.device,
    *,
    use_checkpoint: bool,
) -> torch.Tensor:
    depths: list[torch.Tensor] = []
    # Training and validation may deliberately use different rollout lengths
    # (for example, train through 1.4 s while validating through 2.0 s).  The
    # prediction tensor is the source of truth for how many horizons should be
    # decoded in this invocation.
    for horizon in range(predicted.shape[1]):
        horizon_contexts = []
        for item in contexts:
            selected = dict(item)
            for name in ("target_c2w", "target_intrinsics", "target_depth", "target_valid", "motion_mask"):
                selected[name] = item[name][horizon]
            horizon_contexts.append(selected)

        def decode(
            value: torch.Tensor,
            horizon_contexts: list[dict[str, Any]] = horizon_contexts,
        ) -> torch.Tensor:
            return _decode_meters(geometry_model, value, horizon_contexts, device)

        value = predicted[:, horizon]
        if use_checkpoint and torch.is_grad_enabled():
            depth = activation_checkpoint(decode, value, use_reentrant=False)
        else:
            depth = decode(value)
        depths.append(depth)
    return torch.stack(depths, dim=1)


def _teacher_forced_predictions(
    predictor: torch.nn.Module,
    states: torch.Tensor,
) -> torch.Tensor:
    """Predict each future state from its four ground-truth predecessors."""
    return torch.stack(
        [predictor(states[:, offset : offset + 4]) for offset in range(ROLLOUT_STEPS)],
        dim=1,
    )


@torch.inference_mode()
def validate(
    predictor: torch.nn.Module,
    geometry_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    history_only: bool = False,
) -> dict[str, Any]:
    model = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
    model.eval()
    method_names = ("predicted", "teacher_forced", "copy_last")
    # [method, horizon, absrel_sum, pixels, aligned_absrel_sum,
    #  aligned_pixels, scale_bias_sum, images, delta1_good, delta1_pixels]
    physical = torch.zeros((3, ROLLOUT_STEPS, 10), dtype=torch.float64, device=device)
    state_totals = torch.zeros((ROLLOUT_STEPS, 3), dtype=torch.float64, device=device)
    invalid_totals = torch.zeros((3, ROLLOUT_STEPS), dtype=torch.float64, device=device)
    counts = torch.zeros(5, dtype=torch.float64, device=device)
    for items in loader:
        extracted = (
            extract_growing_rollout_states(
                geometry_model, items, device, future_steps=0
            )
            if history_only
            else extract_growing_rollout_states(geometry_model, items, device)
        )
        if extracted is None:
            counts[1] += len(items)
            continue
        states, contexts, valid, supports = extracted
        # The first growing-prefix pass is the standalone history path.
        copy_last = states[:, 3]
        copy_contexts = contexts
        counts[3] += float(supports.sum())
        counts[4] += len(items)
        indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        counts[1] += len(items) - len(indices)
        if not indices:
            continue
        states = states[indices].float()
        contexts = [contexts[index] for index in indices]
        copy_last = copy_last[indices].float()
        copy_contexts = [copy_contexts[index] for index in indices]
        predicted = _rollout(
            model, states[:, :4], use_checkpoint=False,
        )
        pred_depth = _decode_rollout(
            geometry_model, predicted, contexts, device, use_checkpoint=False
        )
        teacher_depth = None
        if not history_only:
            teacher_forced = _teacher_forced_predictions(model, states)
            teacher_depth = _decode_rollout(
                geometry_model, teacher_forced, contexts, device, use_checkpoint=False
            )
        copy_states = copy_last[:, None].expand(-1, ROLLOUT_STEPS, -1, -1)
        copy_depth = _decode_rollout(
            geometry_model, copy_states, copy_contexts, device, use_checkpoint=False
        )
        if not history_only:
            target = states[:, 4:]
            state_totals[:, 0] += (
                (predicted - target).abs().mean(dim=(2, 3)).sum(0).double()
            )
            state_totals[:, 1] += (
                1 - F.cosine_similarity(
                    predicted.flatten(2), target.flatten(2), dim=2
                )
            ).sum(0).double()
            state_totals[:, 2] += len(indices)
        gt = torch.stack([item["target_depth"] for item in contexts]).to(device).float()
        target_valid = torch.stack([item["target_valid"] for item in contexts]).to(device).bool()
        for method, depth in enumerate((pred_depth, teacher_depth, copy_depth)):
            if depth is None:
                continue
            invalid = ~torch.isfinite(depth) | (depth <= 0)
            invalid_totals[method] += (
                invalid & target_valid
            ).sum(dim=(0, 2, 3)).double()
            depth = torch.nan_to_num(
                depth, nan=1e-3, posinf=80.0, neginf=1e-3
            ).clamp_min(1e-3)
            for batch_index in range(depth.shape[0]):
                for horizon in range(ROLLOUT_STEPS):
                    mask = target_valid[batch_index, horizon]
                    pixels = int(mask.sum())
                    if not pixels:
                        continue
                    pred_values = depth[batch_index, horizon][mask]
                    gt_values = gt[batch_index, horizon][mask].clamp_min(1e-6)
                    ratio = pred_values / gt_values
                    scale_bias = ratio.median() - 1
                    aligned = pred_values * (gt_values / pred_values.clamp_min(1e-6)).median()
                    physical[method, horizon, 0] += (
                        (pred_values - gt_values).abs() / gt_values
                    ).sum().double()
                    physical[method, horizon, 1] += pixels
                    physical[method, horizon, 2] += (
                        (aligned - gt_values).abs() / gt_values
                    ).sum().double()
                    physical[method, horizon, 3] += pixels
                    physical[method, horizon, 4] += scale_bias.double()
                    physical[method, horizon, 5] += 1
                    physical[method, horizon, 6] += (
                        torch.maximum(ratio, ratio.reciprocal()) < 1.25
                    ).sum().double()
                    physical[method, horizon, 7] += pixels
                    # Formal random100 reports an arithmetic mean over frames,
                    # rather than allowing dense-LiDAR frames to dominate.
                    physical[method, horizon, 8] += (
                        (pred_values - gt_values).abs() / gt_values
                    ).mean().double()
                    physical[method, horizon, 9] += 1
        counts[0] += len(indices)
        counts[2] += 1
    packed = torch.cat((
        physical.flatten(), state_totals.flatten(),
        invalid_totals.flatten(), counts,
    ))
    if dist.is_initialized():
        dist.all_reduce(packed)
    cursor = 0
    physical = packed[cursor : cursor + 3 * ROLLOUT_STEPS * 10].reshape(3, ROLLOUT_STEPS, 10); cursor += 3 * ROLLOUT_STEPS * 10
    state_totals = packed[cursor : cursor + ROLLOUT_STEPS * 3].reshape(ROLLOUT_STEPS, 3); cursor += ROLLOUT_STEPS * 3
    invalid_totals = packed[cursor : cursor + 3 * ROLLOUT_STEPS].reshape(3, ROLLOUT_STEPS); cursor += 3 * ROLLOUT_STEPS
    counts = packed[cursor : cursor + 5]
    horizons: dict[str, Any] = {}
    for offset in range(ROLLOUT_STEPS):
        row: dict[str, float] = {
            "state_l1": float(state_totals[offset, 0] / state_totals[offset, 2].clamp_min(1)),
            "state_cosine": float(state_totals[offset, 1] / state_totals[offset, 2].clamp_min(1)),
        }
        for method, name in enumerate(method_names):
            values = physical[method, offset]
            row[f"{name}_all_absrel"] = float(values[0] / values[1].clamp_min(1))
            row[f"{name}_scale_aligned_absrel"] = float(values[2] / values[3].clamp_min(1))
            row[f"{name}_scale_bias"] = float(values[4] / values[5].clamp_min(1))
            row[f"{name}_scale_bias_percent"] = 100 * row[f"{name}_scale_bias"]
            row[f"{name}_delta1"] = float(values[6] / values[7].clamp_min(1))
            row[f"{name}_frame_mean_absrel"] = float(
                values[8] / values[9].clamp_min(1)
            )
            row[f"{name}_invalid_lidar_pixels"] = int(invalid_totals[method, offset])
            row[f"{name}_pixels"] = int(values[1])
        row["predicted_minus_copy_all_absrel"] = (
            row["predicted_all_absrel"] - row["copy_last_all_absrel"]
        )
        row["exposure_gap_absrel"] = (
            row["predicted_all_absrel"] - row["teacher_forced_all_absrel"]
        )
        horizons[f"t+{0.2 * (offset + 1):.1f}"] = row
    summary: dict[str, float] = {}
    for method, name in enumerate(method_names):
        values = physical[method]
        summary[f"{name}_all_horizons_absrel"] = float(values[:, 0].sum() / values[:, 1].sum().clamp_min(1))
        summary[f"{name}_all_horizons_scale_aligned_absrel"] = float(values[:, 2].sum() / values[:, 3].sum().clamp_min(1))
        summary[f"{name}_all_horizons_scale_bias"] = float(values[:, 4].sum() / values[:, 5].sum().clamp_min(1))
        summary[f"{name}_all_horizons_scale_bias_percent"] = 100 * summary[f"{name}_all_horizons_scale_bias"]
        summary[f"{name}_all_horizons_delta1"] = float(values[:, 6].sum() / values[:, 7].sum().clamp_min(1))
        summary[f"{name}_all_horizons_frame_mean_absrel"] = float(
            values[:, 8].sum() / values[:, 9].sum().clamp_min(1)
        )
        first_seven = values[: min(7, ROLLOUT_STEPS)]
        summary[f"{name}_first7_horizons_absrel"] = float(
            first_seven[:, 0].sum() / first_seven[:, 1].sum().clamp_min(1)
        )
        summary[f"{name}_first7_horizons_frame_mean_absrel"] = float(
            first_seven[:, 8].sum() / first_seven[:, 9].sum().clamp_min(1)
        )
    summary["predicted_minus_copy_all_horizons_absrel"] = (
        summary["predicted_all_horizons_absrel"] - summary["copy_last_all_horizons_absrel"]
    )
    summary["exposure_gap_all_horizons_absrel"] = (
        summary["predicted_all_horizons_absrel"] - summary["teacher_forced_all_horizons_absrel"]
    )
    return {
        "copy_last_protocol": "standalone_12_history_vggt_lidar_paired",
        "evaluation_mask": "target_valid_lidar_only",
        "teacher_forced_available": not history_only,
        "rollout_steps": ROLLOUT_STEPS,
        "sequences": int(counts[0]),
        "rejected_sequences": int(counts[1]),
        "batches": int(counts[2]),
        "lidar_scale_mean_support": float(counts[3] / counts[4].clamp_min(1)),
        **summary,
        "predicted_one_step_absrel": horizons["t+0.2"]["predicted_all_absrel"],
        "teacher_forced_one_step_absrel": horizons["t+0.2"]["teacher_forced_all_absrel"],
        "copy_last_one_step_absrel": horizons["t+0.2"]["copy_last_all_absrel"],
        "horizons": horizons,
    }


def train(args: argparse.Namespace) -> None:
    rank, world_size, _, device = _distributed_context()
    _set_seed(args.seed, rank)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _, _, geometry_model = _load_model(args.config, args.scene_checkpoint, device)
    geometry_model.eval().requires_grad_(False)
    train_dataset = ManifestWaymoDataset(
        args.train_prepared_root,
        args.train_manifest,
        args.train_motion_mask_root,
        max_windows=args.train_sequences,
        seed=args.seed,
        validation=False,
        image_view_count=12,
        target_steps=TRAIN_ROLLOUT_STEPS,
    )
    validation_datasets = {
        "main": ManifestWaymoDataset(
            args.validation_prepared_root,
            args.validation_manifest,
            args.validation_motion_mask_root,
            max_windows=args.validation_sequences,
            seed=args.seed,
            validation=True,
            image_view_count=None,
            target_steps=ROLLOUT_STEPS,
        )
    }
    train_loader, train_sampler = _loader(
        train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        workers=args.loader_workers,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    validation_loaders = {
        name: _loader(
            dataset,
            rank=rank,
            world_size=world_size,
            batch_size=args.validation_batch_size,
            workers=args.validation_loader_workers,
            shuffle=False,
            seed=args.seed,
            drop_last=False,
        )[0]
        for name, dataset in validation_datasets.items()
    }

    predictor_config = SceneStatePredictorConfig(
        scene_tokens=512,
        scene_dim=512,
        hidden_dim=1152,
        layers=12,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    predictor = build_next_scene_state_predictor(predictor_config).to(device)
    source = torch.load(args.resume, map_location="cpu", mmap=True, weights_only=True)
    if source.get("config") != asdict(predictor_config):
        raise RuntimeError("World checkpoint predictor config mismatch")
    predictor.load_state_dict(source["model"], strict=True)
    source_step = int(source.get("step", 0))
    del source
    if world_size > 1:
        predictor = DistributedDataParallel(
            predictor,
            device_ids=[device.index],
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    final_ratio = args.final_lr / args.lr

    def lr_multiplier(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return args.warmup_start_factor + (
                1 - args.warmup_start_factor
            ) * step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return final_ratio + (1 - final_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(
            {
                "format": "geostate_waymo_depth_rollout_v1",
                "arguments": vars(args) | {"function": None},
                "predictor_config": asdict(predictor_config),
                "source_checkpoint_step": source_step,
                "train_sequences": len(train_dataset),
                "rollout_steps": ROLLOUT_STEPS,
                "vggt_protocol": "growing_eight_pass",
                "training_vggt_views": 12,
                "objective": "uniform_all_horizon_front_log_depth_l1",
                "future_lidar_used_for_scale": False,
                "effective_global_batch": args.batch_size * world_size * args.grad_accumulation,
            },
            output_dir / "run_config.json",
        )

    def validation_slices() -> dict[str, dict[str, Any]]:
        return {
            name: validate(
                predictor,
                geometry_model,
                loader,
                device,
                history_only=False,
            )
            for name, loader in validation_loaders.items()
        }

    best_absrel = math.inf
    if not args.skip_initial_validation:
        initial = validation_slices()
        best_absrel = sum(
            float(metrics["predicted_all_horizons_absrel"])
            for metrics in initial.values()
        ) / len(initial)
        if rank == 0:
            print(json.dumps({"event": "initial_validation", "slices": initial}), flush=True)
    if args.eval_only:
        _cleanup_distributed()
        return

    predictor.train()
    iterator = iter(train_loader)
    epoch = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, device=device)
        for micro_step in range(args.grad_accumulation):
            while True:
                items, iterator, epoch = _next_batch(iterator, train_loader, train_sampler, epoch)
                extracted = extract_growing_rollout_states(
                    geometry_model, items, device, future_steps=0
                )
                if extracted is None:
                    totals[3] += len(items)
                    continue
                states, contexts, valid, _ = extracted
                indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
                totals[3] += len(items) - len(indices)
                if indices:
                    break
            states = states[indices].float()
            contexts = [contexts[index] for index in indices]
            sync = micro_step + 1 == args.grad_accumulation
            gradient_context = (
                predictor.no_sync()
                if isinstance(predictor, DistributedDataParallel) and not sync
                else nullcontext()
            )
            with gradient_context:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    predicted = _rollout(
                        predictor,
                        states[:, :4],
                        use_checkpoint=args.activation_checkpointing,
                        steps=TRAIN_ROLLOUT_STEPS,
                    )
                    pred_depth = _decode_rollout(
                        geometry_model,
                        predicted,
                        contexts,
                        device,
                        use_checkpoint=args.activation_checkpointing,
                    )
                    gt = torch.stack([item["target_depth"] for item in contexts]).to(device).float()
                    target_valid = torch.stack([item["target_valid"] for item in contexts]).to(device).bool()
                    valid_depth = target_valid & torch.isfinite(pred_depth) & (pred_depth > 0)
                    horizon_losses = torch.stack([
                        _region_log_loss(
                            pred_depth[:, horizon],
                            gt[:, horizon],
                            valid_depth[:, horizon],
                        )
                        for horizon in range(TRAIN_ROLLOUT_STEPS)
                    ])
                    loss = horizon_losses.mean()
                    scaled_loss = loss / args.grad_accumulation
                if not torch.isfinite(scaled_loss):
                    raise RuntimeError(f"Non-finite rollout loss at step {step}")
                scaled_loss.backward()
            totals[0] += loss.detach()
            totals[1] += (predicted[:, 0] - states[:, 3]).square().mean().sqrt().detach()
            totals[2] += len(indices)
        grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite rollout gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        totals[:2] /= args.grad_accumulation
        if dist.is_initialized():
            dist.all_reduce(totals)
            totals[:2] /= world_size
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            print(
                json.dumps({
                    "event": "train",
                    "step": step,
                    "front_depth_log_l1": float(totals[0]),
                    "residual_rms": float(totals[1]),
                    "accepted_sequences": int(totals[2]),
                    "rejected_sequences": int(totals[3]),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }),
                flush=True,
            )
        if args.validate_every and step % args.validate_every == 0:
            slices = validation_slices()
            score = sum(
                float(metrics["predicted_all_horizons_absrel"])
                for metrics in slices.values()
            ) / len(slices)
            if rank == 0:
                unwrapped = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
                payload = {
                    "step": step,
                    "config": asdict(predictor_config),
                    "model": {key: value.detach().cpu() for key, value in unwrapped.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "validation": slices["main"],
                    "validation_slices": slices,
                    "source_checkpoint_step": source_step,
                    "train_mode": "closed_depth",
                }
                checkpoint_path = checkpoint_dir / f"world_model_step_{step:06d}.pt"
                _atomic_torch_save(payload, checkpoint_path)
                _replace_hardlink(checkpoint_path, output_dir / "latest.pt")
                if score < best_absrel:
                    best_absrel = score
                    _replace_hardlink(checkpoint_path, output_dir / "best_rollout.pt")
                print(json.dumps({
                    "event": "validation",
                    "step": step,
                    "equal_pool_absrel": score,
                    "best_equal_pool_absrel": best_absrel,
                    "slices": slices,
                }), flush=True)
            predictor.train()
    if rank == 0:
        unwrapped = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
        _atomic_torch_save(
            {
                "step": args.steps,
                "config": asdict(predictor_config),
                "model": {key: value.detach().cpu() for key, value in unwrapped.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "source_checkpoint_step": source_step,
                "train_mode": "closed_depth",
            },
            output_dir / "final.pt",
        )
    _cleanup_distributed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--resume", required=True, help="One-step world-model checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-prepared-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-motion-mask-root", required=True)
    parser.add_argument("--validation-prepared-root", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--validation-motion-mask-root", required=True)
    parser.add_argument("--train-sequences", type=int, default=139367)
    parser.add_argument("--validation-sequences", type=int, default=102)
    parser.add_argument("--steps", type=int, default=35000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--minimum-effective-batch", type=int, default=1)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--validation-loader-workers", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--final-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    effective = args.batch_size * int(os.environ.get("WORLD_SIZE", "1")) * args.grad_accumulation
    if effective < args.minimum_effective_batch:
        raise ValueError(
            f"Effective global batch must be at least {args.minimum_effective_batch}, got {effective}"
        )
    if args.steps < 1 or args.grad_accumulation < 1:
        raise ValueError("steps and grad-accumulation must be positive")
    if not 0 < args.final_lr <= args.lr:
        raise ValueError("final-lr must be positive and no greater than lr")
    train(args)


if __name__ == "__main__":
    main()
