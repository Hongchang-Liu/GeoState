"""Paper one-step Waymo world-model training with frozen VGGT-Omega.

Every logical sample is independent.  ``two_pass`` encodes four observed
timestamps with a 12-image VGGT pass and the target with a separate 15-image
pass. Each pass is calibrated against the same observed FRONT LiDAR depth;
target geometry uses the physical scene unit fixed by the history pass.
No prediction is rolled into a later sample.
"""

from __future__ import annotations

import argparse
import itertools
import io
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from GeoState.evaluation.checkpoint import load_geostate_checkpoint as _load_model
from GeoState.core.geometry import paired_depth_scale
from GeoState.models.world_model import SceneStatePredictorConfig, build_next_scene_state_predictor
from GeoState.data.waymo import CAMERA_NAMES
from GeoState.training.waymo_one_step import (
    _decode_meters,
    _region_log_loss,
)


PAIR_FORMAT = "waymo_world_online_independent_4to1_v1"
LOCAL_STATE_ORDER = (3, 0, 1, 2, 4)
TARGET_OFFSETS = tuple(range(4, 11))
VGGT_PROTOCOLS = ("two_pass",)


def _distributed_context() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if world_size > 1:
            raise RuntimeError("Multi-process world-model training requires CUDA/NCCL")
        return rank, world_size, local_rank, torch.device("cpu")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        timeout_seconds = int(os.environ.get("TORCH_DISTRIBUTED_TIMEOUT_SECONDS", "600"))
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
            timeout=timedelta(seconds=timeout_seconds),
        )
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_seed(seed: int, rank: int) -> None:
    value = int(seed) + rank * 100_003
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_sequences(
    rows: list[dict[str, Any]],
    prepared_root: Path,
    *,
    limit: int,
    seed: int,
    validation: bool,
) -> list[dict[str, Any]]:
    rows = [row for row in rows if (prepared_root / f"{row['segment']}.pt").is_file()]
    if validation:
        by_segment: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_segment.setdefault(row["segment"], []).append(row)
        for values in by_segment.values():
            values.sort(key=lambda value: (-float(value["dynamic_score"]), value["first_slot"]))
        selected: list[dict[str, Any]] = []
        cursor = 0
        segments = sorted(by_segment)
        while len(selected) < min(limit, len(rows)):
            progressed = False
            for segment in segments:
                if cursor < len(by_segment[segment]):
                    selected.append(by_segment[segment][cursor])
                    progressed = True
                    if len(selected) == min(limit, len(rows)):
                        break
            if not progressed:
                break
            cursor += 1
        return selected
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(limit, len(rows))]


class IndependentWaymoPairDataset(Dataset[dict[str, Any]]):
    """Expand every 11-state manifest row into seven independent 4->1 pairs."""

    def __init__(
        self,
        prepared_root: str | Path,
        manifest: str | Path,
        motion_mask_root: str | Path,
        *,
        sequence_limit: int,
        seed: int,
        validation: bool,
    ) -> None:
        self.prepared_root = Path(prepared_root)
        self.motion_mask_root = Path(motion_mask_root)
        self.records = _select_sequences(
            _read_jsonl(manifest),
            self.prepared_root,
            limit=sequence_limit,
            seed=seed,
            validation=validation,
        )
        if len(self.records) != sequence_limit:
            raise ValueError(f"Requested {sequence_limit} sequences, found {len(self.records)}")
        self._cached_path: Path | None = None
        self._cached_payload: dict[str, Any] | None = None

    @staticmethod
    def _decode(value: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(value)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    def _payload(self, path: Path) -> dict[str, Any]:
        if path != self._cached_path:
            self._cached_payload = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_path = path
        assert self._cached_payload is not None
        return self._cached_payload

    def __len__(self) -> int:
        return len(self.records) * len(TARGET_OFFSETS)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, offset_index = divmod(index, len(TARGET_OFFSETS))
        target_state = TARGET_OFFSETS[offset_index]
        row = self.records[sequence_index]
        logical_slots = tuple(int(value) for value in row["logical_slots"])
        local_global_states = tuple(range(target_state - 4, target_state + 1))
        ordered_global_states = tuple(local_global_states[state] for state in LOCAL_STATE_ORDER)
        segment = row["segment"]
        payload = self._payload(self.prepared_root / f"{segment}.pt")
        pairs = [
            (logical_slots[global_state], camera)
            for global_state in ordered_global_states
            for camera in range(3)
        ]
        images = torch.stack(
            [self._decode(payload["images_jpeg"][slot][camera]) for slot, camera in pairs]
        )
        world_c2w = torch.as_tensor(payload["world_from_camera_cv"]).double()
        physical_c2w = torch.stack([world_c2w[slot, camera] for slot, camera in pairs])
        anchor_from_world = torch.linalg.inv(physical_c2w[0])
        local_c2w = (anchor_from_world[None] @ physical_c2w).float()
        local_c2w[0] = torch.eye(4)
        intrinsics = torch.as_tensor(payload["intrinsics"]).float()
        input_intrinsics = torch.stack([intrinsics[camera] for _, camera in pairs])
        timestamps = torch.as_tensor(payload["timestamps"], dtype=torch.int64)
        target_slot = logical_slots[target_state]
        target_timestamp = int(timestamps[target_slot])
        depth_rows = torch.as_tensor(payload["depth_cm"])
        scale_slot = logical_slots[target_state - 1]
        input_scale_depth = depth_rows[scale_slot, 0].float() / 100.0
        input_scale_valid = (
            torch.isfinite(input_scale_depth)
            & (input_scale_depth > 0)
            & (input_scale_depth <= 80.0)
        )
        input_scale_depth = torch.where(
            input_scale_valid, input_scale_depth, torch.zeros_like(input_scale_depth)
        )
        target_depth = depth_rows[target_slot, 0].float() / 100.0
        target_valid = torch.isfinite(target_depth) & (target_depth > 0) & (target_depth <= 80.0)
        target_depth = torch.where(target_valid, target_depth, torch.zeros_like(target_depth))
        mask_path = self.motion_mask_root / segment / "FRONT" / f"{target_timestamp}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing motion mask {mask_path}")
        with Image.open(mask_path) as mask_image:
            motion_mask = torch.from_numpy(np.asarray(mask_image, dtype=np.uint8).copy()) > 0
        if motion_mask.shape != target_depth.shape:
            raise ValueError(f"Motion mask shape {motion_mask.shape} != depth {target_depth.shape}")
        view_states = torch.tensor(
            [state for state in LOCAL_STATE_ORDER for _ in range(3)], dtype=torch.long
        )
        return {
            "format": PAIR_FORMAT,
            "window_id": f"{row['window_id']}:target_s{target_state}",
            "segment": segment,
            "target_state_index": target_state,
            "dynamic_score": float(row["dynamic_score"]),
            "images": images,
            "input_c2w": local_c2w,
            "input_intrinsics": input_intrinsics,
            "view_states": view_states,
            "frame_ids": tuple(
                f"{int(timestamps[slot])}_{CAMERA_NAMES[camera]}" for slot, camera in pairs
            ),
            "target_c2w": local_c2w[12],
            "target_intrinsics": input_intrinsics[12],
            "target_depth": target_depth,
            "target_valid": target_valid,
            "input_scale_depth": input_scale_depth,
            "input_scale_valid": input_scale_valid,
            "motion_mask": motion_mask,
        }


def _collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def _context(
    encoding: Any,
    item: dict[str, Any],
    batch_index: int,
    *,
    paired_scale: torch.Tensor,
    scale_valid: torch.Tensor,
    scale_support: torch.Tensor,
) -> dict[str, Any]:
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
        "target_state_index": item["target_state_index"],
        "window_id": item["window_id"],
        "paired_lidar_scale": paired_scale[batch_index].float().cpu(),
        "paired_lidar_scale_valid": scale_valid[batch_index].bool().cpu(),
        "paired_lidar_scale_support": scale_support[batch_index].long().cpu(),
    }


def _paired_history_scale(
    output: Any,
    items: list[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix the dataset->VGGT gauge from matched input-time FRONT depths."""
    lidar_depth = torch.stack([item["input_scale_depth"] for item in items]).to(
        device, non_blocking=True
    )
    return paired_depth_scale(
        output.depth[:, 0],
        lidar_depth,
        output.confidence[:, 0],
    )


def _single_anchor_mask(batch: int, views: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch, views), dtype=torch.bool, device=device)
    mask[:, 0] = True
    return mask


@torch.inference_mode()
def extract_independent_pairs(
    geometry_model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    protocol: str,
    *,
    unified_validation: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, list[str]]:
    if not items:
        raise ValueError("Raw online extraction batch cannot be empty")
    images = torch.stack([item["images"] for item in items]).to(device, non_blocking=True)
    c2w = torch.stack([item["input_c2w"] for item in items]).to(device)
    intrinsics = torch.stack([item["input_intrinsics"] for item in items]).to(device)
    view_states = torch.stack([item["view_states"] for item in items]).to(device)

    if protocol != "two_pass":
        raise ValueError(f"Unknown VGGT protocol {protocol!r}; expected {VGGT_PROTOCOLS}")
    else:
        output12 = geometry_model.vggt(images[:, :12])
        paired_scale, scale_valid, scale_support = _paired_history_scale(
            output12, items, device
        )
        history_alignment_mask = view_states[:, :12] < 4
        anchor_alignment_mask12 = _single_anchor_mask(len(items), 12, device)
        history_mask = torch.stack(
            [(view_states[:, :12] == state) for state in range(4)], dim=1
        )
        history, history_encoding = geometry_model.encode_temporal_scene_states_from_vggt_output(
            output12,
            c2w[:, :12],
            input_intrinsics=intrinsics[:, :12],
            state_view_mask=history_mask,
            single_view_scale=paired_scale,
            alignment_view_mask=anchor_alignment_mask12,
            scene_scale_view_mask=history_alignment_mask,
            frame_ids=[item["frame_ids"][:12] for item in items],
            reference_index=0,
        )
        history_alignment = history_encoding.metadata["alignment"]
        # L0 is meters per normalized history unit. Keep it fixed even if the
        # separate target pass changes VGGT's raw reconstruction scale.
        condition_meters_per_unit = (
            history_encoding.scene_scale / paired_scale.float().clamp_min(1e-8)
        )
        contexts = [
            _context(
                history_encoding,
                item,
                i,
                paired_scale=paired_scale,
                scale_valid=scale_valid,
                scale_support=scale_support,
            )
            for i, item in enumerate(items)
        ]
        del output12, history_encoding

        output15 = geometry_model.vggt(images)
        target_paired_scale, target_scale_valid, _ = _paired_history_scale(
            output15, items, device
        )
        target_mask = (view_states == 4)[:, None]
        target_history_mask = view_states < 4
        anchor_alignment_mask15 = _single_anchor_mask(len(items), 15, device)
        target, target_encoding = geometry_model.encode_temporal_scene_states_from_vggt_output(
            output15,
            c2w,
            input_intrinsics=intrinsics,
            state_view_mask=target_mask,
            single_view_scale=target_paired_scale,
            scene_scale_override=target_paired_scale * condition_meters_per_unit,
            alignment_view_mask=anchor_alignment_mask15,
            scene_scale_view_mask=target_history_mask,
            frame_ids=[item["frame_ids"] for item in items],
            reference_index=0,
        )
        target_alignment = target_encoding.metadata["alignment"]
        valid = (
            scale_valid.bool()
            & target_scale_valid.bool()
            & history_alignment.valid.bool()
            & target_alignment.valid.bool()
        )
        conditions = [
            f"history={history_condition};target={target_condition}"
            for history_condition, target_condition in zip(
                history_alignment.condition, target_alignment.condition
            )
        ]
        states = torch.cat((history, target), dim=1)
        del output15, target_encoding
    del images
    return states.detach(), contexts, valid, conditions


@torch.inference_mode()
def extract_independent_pairs_chunked(
    geometry_model: torch.nn.Module,
    items: list[dict[str, Any]],
    device: torch.device,
    protocol: str,
    *,
    unified_validation: bool = False,
    geometry_microbatch_size: int = 0,
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor, list[str]]:
    """Extract frozen geometry in smaller chunks, then restore batch order.

    The predictor still receives the original compute batch.  This only lowers
    peak memory in the frozen VGGT/tokenizer front end and therefore leaves the
    world-model loss reduction and optimizer batch unchanged.
    """
    if geometry_microbatch_size <= 0 or len(items) <= geometry_microbatch_size:
        return extract_independent_pairs(
            geometry_model,
            items,
            device,
            protocol,
            unified_validation=unified_validation,
        )
    state_chunks: list[torch.Tensor] = []
    contexts: list[dict[str, Any]] = []
    valid_chunks: list[torch.Tensor] = []
    conditions: list[str] = []
    for start in range(0, len(items), geometry_microbatch_size):
        states, chunk_contexts, valid, chunk_conditions = extract_independent_pairs(
            geometry_model,
            items[start : start + geometry_microbatch_size],
            device,
            protocol,
            unified_validation=unified_validation,
        )
        state_chunks.append(states)
        contexts.extend(chunk_contexts)
        valid_chunks.append(valid)
        conditions.extend(chunk_conditions)
    return torch.cat(state_chunks, dim=0), contexts, torch.cat(valid_chunks, dim=0), conditions


class _ResumeOffsetDistributedSampler(DistributedSampler):
    """Skip consumed samples only in the first epoch after a restart."""

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


def _make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = True,
    start_offset: int = 0,
) -> tuple[DataLoader, DistributedSampler]:
    sampler = _ResumeOffsetDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
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
        drop_last=drop_last,
    )
    return loader, sampler


def _next_batch(iterator: Any, loader: DataLoader, sampler: DistributedSampler, epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


@torch.inference_mode()
def validate_online(
    predictor: torch.nn.Module,
    geometry_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    protocol: str,
    *,
    unified_all_pairs: bool = False,
    geometry_microbatch_size: int = 0,
) -> dict[str, Any]:
    model = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
    model.eval()
    totals = {name: 0.0 for name in (
        "state_l1", "state_cosine",
        "all_abs", "all_pixels", "dynamic_abs", "dynamic_pixels", "static_abs", "static_pixels",
        "aligned_all_abs", "aligned_all_pixels", "aligned_dynamic_abs", "aligned_dynamic_pixels", "aligned_static_abs", "aligned_static_pixels",
        "copy_all_abs", "copy_all_pixels", "copy_dynamic_abs", "copy_dynamic_pixels",
        "copy_static_abs", "copy_static_pixels",
        "copy_aligned_all_abs", "copy_aligned_all_pixels", "copy_aligned_dynamic_abs", "copy_aligned_dynamic_pixels",
        "copy_aligned_static_abs", "copy_aligned_static_pixels",
        "lidar_scale_valid", "lidar_scale_invalid", "lidar_scale_support",
    )}
    per_target = {state: [0.0, 0.0] for state in TARGET_OFFSETS}
    copy_per_target = {state: [0.0, 0.0] for state in TARGET_OFFSETS}
    accepted = rejected = alignment_valid = alignment_invalid = 0
    for items in loader:
        states, contexts, valid, _ = extract_independent_pairs_chunked(
            geometry_model,
            items,
            device,
            protocol,
            unified_validation=unified_all_pairs,
            geometry_microbatch_size=geometry_microbatch_size,
        )
        alignment_valid += int(valid.sum())
        alignment_invalid += len(items) - int(valid.sum())
        for context in contexts:
            scale_ok = bool(context["paired_lidar_scale_valid"])
            totals["lidar_scale_valid" if scale_ok else "lidar_scale_invalid"] += 1
            totals["lidar_scale_support"] += int(context["paired_lidar_scale_support"])
        indices = (
            list(range(len(items)))
            if unified_all_pairs
            else torch.nonzero(valid, as_tuple=False).flatten().tolist()
        )
        rejected += len(items) - len(indices)
        if not indices:
            continue
        states = states[indices].float()
        contexts = [contexts[i] for i in indices]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted = model(states[:, :4])
            pred_depth = _decode_meters(geometry_model, predicted, contexts, device)
            copy_depth = _decode_meters(geometry_model, states[:, 3], contexts, device)
        target = states[:, 4]
        totals["state_l1"] += float((predicted.float() - target).abs().mean(dim=(1, 2)).sum())
        totals["state_cosine"] += float((1 - F.cosine_similarity(predicted.float().flatten(1), target.flatten(1))).sum())
        for i, item in enumerate(contexts):
            gt = item["target_depth"].to(device).float()
            valid_pixels = (
                item["target_valid"].to(device).bool()
                & torch.isfinite(pred_depth[i])
                & (pred_depth[i] > 0)
                & torch.isfinite(copy_depth[i])
                & (copy_depth[i] > 0)
            )
            motion = item["motion_mask"].to(device).bool()
            if bool(valid_pixels.any()):
                valid_gt = gt[valid_pixels].clamp_min(1e-6)
                predicted_scale = (
                    valid_gt / pred_depth[i][valid_pixels].clamp_min(1e-6)
                ).median()
                copy_scale = (
                    valid_gt / copy_depth[i][valid_pixels].clamp_min(1e-6)
                ).median()
            else:
                predicted_scale = pred_depth.new_ones(())
                copy_scale = copy_depth.new_ones(())
            for region, mask in (("all", valid_pixels), ("dynamic", valid_pixels & motion), ("static", valid_pixels & ~motion)):
                pixels = int(mask.sum())
                if pixels:
                    error = float(((pred_depth[i][mask] - gt[mask]).abs() / gt[mask].clamp_min(1e-6)).sum())
                    copy_error = float(((copy_depth[i][mask] - gt[mask]).abs() / gt[mask].clamp_min(1e-6)).sum())
                    totals[f"{region}_abs"] += error
                    totals[f"{region}_pixels"] += pixels
                    totals[f"copy_{region}_abs"] += copy_error
                    totals[f"copy_{region}_pixels"] += pixels
                    totals[f"aligned_{region}_abs"] += float((
                        (pred_depth[i][mask] * predicted_scale - gt[mask]).abs()
                        / gt[mask].clamp_min(1e-6)
                    ).sum())
                    totals[f"aligned_{region}_pixels"] += pixels
                    totals[f"copy_aligned_{region}_abs"] += float((
                        (copy_depth[i][mask] * copy_scale - gt[mask]).abs()
                        / gt[mask].clamp_min(1e-6)
                    ).sum())
                    totals[f"copy_aligned_{region}_pixels"] += pixels
                    if region == "all":
                        per_target[int(item["target_state_index"])][0] += error
                        per_target[int(item["target_state_index"])][1] += pixels
                        copy_per_target[int(item["target_state_index"])][0] += copy_error
                        copy_per_target[int(item["target_state_index"])][1] += pixels
        accepted += len(indices)
    total_names = tuple(totals)
    packed = torch.tensor(
        [totals[name] for name in total_names]
        + [float(accepted), float(rejected), float(alignment_valid), float(alignment_invalid)]
        + [value for state in TARGET_OFFSETS for value in per_target[state]]
        + [value for state in TARGET_OFFSETS for value in copy_per_target[state]],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(packed)
    cursor = 0
    totals = {name: float(packed[index]) for index, name in enumerate(total_names)}
    cursor += len(total_names)
    accepted, rejected = int(packed[cursor]), int(packed[cursor + 1])
    alignment_valid = int(packed[cursor + 2])
    alignment_invalid = int(packed[cursor + 3])
    cursor += 4
    per_target = {}
    for state in TARGET_OFFSETS:
        per_target[state] = [float(packed[cursor]), float(packed[cursor + 1])]
        cursor += 2
    copy_per_target = {}
    for state in TARGET_OFFSETS:
        copy_per_target[state] = [float(packed[cursor]), float(packed[cursor + 1])]
        cursor += 2
    predicted_all = totals["all_abs"] / max(1.0, totals["all_pixels"])
    predicted_dynamic = totals["dynamic_abs"] / max(1.0, totals["dynamic_pixels"])
    predicted_static = totals["static_abs"] / max(1.0, totals["static_pixels"])
    copy_all = totals["copy_all_abs"] / max(1.0, totals["copy_all_pixels"])
    copy_dynamic = totals["copy_dynamic_abs"] / max(1.0, totals["copy_dynamic_pixels"])
    copy_static = totals["copy_static_abs"] / max(1.0, totals["copy_static_pixels"])
    predicted_aligned = {
        region: totals[f"aligned_{region}_abs"]
        / max(1.0, totals[f"aligned_{region}_pixels"])
        for region in ("all", "dynamic", "static")
    }
    copy_aligned = {
        region: totals[f"copy_aligned_{region}_abs"]
        / max(1.0, totals[f"copy_aligned_{region}_pixels"])
        for region in ("all", "dynamic", "static")
    }
    result: dict[str, Any] = {
        "samples": accepted,
        "rejected": rejected,
        "alignment_valid_samples": alignment_valid,
        "alignment_invalid_samples": alignment_invalid,
        "state_l1": totals["state_l1"] / max(1, accepted),
        "state_cosine": totals["state_cosine"] / max(1, accepted),
        "lidar_scale_valid_samples": int(totals["lidar_scale_valid"]),
        "lidar_scale_invalid_samples": int(totals["lidar_scale_invalid"]),
        "lidar_scale_mean_support": totals["lidar_scale_support"]
        / max(1.0, totals["lidar_scale_valid"] + totals["lidar_scale_invalid"]),
        "predicted_all_absrel": predicted_all,
        "predicted_dynamic_absrel": predicted_dynamic,
        "predicted_static_absrel": predicted_static,
        "predicted_all_scale_aligned_absrel": predicted_aligned["all"],
        "predicted_dynamic_scale_aligned_absrel": predicted_aligned["dynamic"],
        "predicted_static_scale_aligned_absrel": predicted_aligned["static"],
        "copy_last_all_absrel": copy_all,
        "copy_last_dynamic_absrel": copy_dynamic,
        "copy_last_static_absrel": copy_static,
        "copy_last_all_scale_aligned_absrel": copy_aligned["all"],
        "copy_last_dynamic_scale_aligned_absrel": copy_aligned["dynamic"],
        "copy_last_static_scale_aligned_absrel": copy_aligned["static"],
        "predicted_minus_copy_all_absrel": predicted_all - copy_all,
        "predicted_minus_copy_dynamic_absrel": predicted_dynamic - copy_dynamic,
        "predicted_minus_copy_static_absrel": predicted_static - copy_static,
        "per_target_absrel": {
            f"s{state}": values[0] / max(1.0, values[1]) for state, values in per_target.items()
        },
        "copy_last_per_target_absrel": {
            f"s{state}": values[0] / max(1.0, values[1])
            for state, values in copy_per_target.items()
        },
    }
    return result


def _one_step_loss_terms(
    state_loss: torch.Tensor,
    cosine_loss: torch.Tensor,
    depth_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixed paper objective and its state/depth contributions."""
    state_contribution = state_loss + 0.05 * cosine_loss
    depth_contribution = 0.24 * depth_loss
    return state_contribution + depth_contribution, state_contribution, depth_contribution


def train(args: argparse.Namespace) -> None:
    rank, world_size, _, device = _distributed_context()
    if args.resume_sampler_global_offset % world_size:
        raise ValueError(
            "--resume-sampler-global-offset must be divisible by WORLD_SIZE; "
            f"got {args.resume_sampler_global_offset} and {world_size}"
        )
    if args.geometry_microbatch_size < 0:
        raise ValueError("--geometry-microbatch-size must be non-negative")
    _set_seed(args.seed, rank)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    config, _, geometry_model = _load_model(args.config, args.scene_checkpoint, device)
    geometry_model.eval().requires_grad_(False)
    train_dataset = IndependentWaymoPairDataset(
        args.train_prepared_root,
        args.train_manifest,
        args.train_motion_mask_root,
        sequence_limit=args.train_sequences,
        seed=args.seed,
        validation=False,
    )
    validation_dataset = IndependentWaymoPairDataset(
        args.validation_prepared_root,
        args.validation_manifest,
        args.validation_motion_mask_root,
        sequence_limit=args.validation_sequences,
        seed=args.seed,
        validation=True,
    )
    train_loader, train_sampler = _make_loader(
        train_dataset, rank=rank, world_size=world_size, batch_size=args.batch_size,
        workers=args.loader_workers, shuffle=True, seed=args.seed,
        start_offset=args.resume_sampler_global_offset // world_size,
    )
    validation_loader, _ = _make_loader(
        validation_dataset, rank=rank, world_size=world_size, batch_size=args.validation_batch_size,
        workers=args.validation_loader_workers, shuffle=False, seed=args.seed, drop_last=False,
    )
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
    resume_step = 0
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_config = resume_checkpoint.get("config")
        if saved_config != asdict(predictor_config):
            raise RuntimeError(
                "Resume checkpoint predictor config mismatch: "
                f"saved={saved_config}, requested={asdict(predictor_config)}"
            )
        predictor.load_state_dict(resume_checkpoint["model"], strict=True)
        resume_step = 0 if args.resume_model_only else int(resume_checkpoint["step"])
        if not 0 <= resume_step < args.steps:
            raise ValueError(
                f"Resume step must be in [0, {args.steps}), got {resume_step}"
            )
    frozen_parameters = sum(parameter.numel() for parameter in geometry_model.parameters())
    if any(parameter.requires_grad for parameter in geometry_model.parameters()):
        raise RuntimeError("VGGT, scene tokenizer, and depth decoder must all be frozen")
    trainable_parameters = sum(parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad)
    if world_size > 1:
        predictor = DistributedDataParallel(
            predictor,
            device_ids=[device.index],
            static_graph=args.grad_accumulation == 1,
        )
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    optimizer_state_restored = False
    if (
        resume_checkpoint is not None
        and not args.resume_model_only
        and "optimizer" in resume_checkpoint
    ):
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        optimizer_state_restored = True
    resume_lr = float(optimizer.param_groups[0]["lr"])
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup_steps must be in [0, steps)")
    if args.resume_rebase_scheduler and not (resume_step and optimizer_state_restored):
        raise ValueError(
            "--resume-rebase-scheduler requires a non-model-only checkpoint with optimizer state"
        )
    schedule_base_lr = resume_lr if args.resume_rebase_scheduler else args.lr
    final_ratio = args.final_lr / schedule_base_lr

    def lr_multiplier(step: int) -> float:
        if args.resume_rebase_scheduler:
            progress = (step - resume_step) / max(1, args.steps - resume_step)
            progress = min(1.0, max(0.0, progress))
            return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if args.warmup_steps and step < args.warmup_steps:
            return args.warmup_start_factor + (1 - args.warmup_start_factor) * step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    if resume_step:
        for group in optimizer.param_groups:
            group["initial_lr"] = schedule_base_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_multiplier,
        last_epoch=resume_step - 1 if resume_step else -1,
    )
    scheduler_state_restored = False
    if (
        resume_checkpoint is not None
        and not args.resume_model_only
        and "scheduler" in resume_checkpoint
        and not args.resume_rebase_scheduler
    ):
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scheduler_state_restored = True
    del resume_checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        _atomic_json_dump(
            {
                "format": "waymo_world_online_independent_4to1_lidar_scale_training_v2",
                "arguments": vars(args) | {"function": None},
                "predictor_config": asdict(predictor_config),
                "train_sequences": args.train_sequences,
                "logical_train_pairs": len(train_dataset),
                "validation_sequences": args.validation_sequences,
                "logical_validation_pairs": len(validation_dataset),
                "vggt_protocol": "two_pass",
                "vggt_passes_per_sample": [12, 15],
                "geometry_microbatch_size": args.geometry_microbatch_size,
                "scene_scale_source": "history_12_vggt_depths",
                "dataset_to_student_scale_source": (
                    "median_log_matched_last_history_front_vggt_depth_over_lidar"
                ),
                "scale_anchor": "last_history_front",
                "scale_uses_future_lidar": False,
                "scale_min_valid_pixels": 128,
                "validation_scene_scale_source": "history_12_vggt_depths",
                "validation_inclusion": "alignment_valid_only",
                "resume_checkpoint": str(args.resume) if args.resume is not None else None,
                "resume_sampler_global_offset": args.resume_sampler_global_offset,
                "resume_model_only": args.resume_model_only,
                "resume_step": resume_step,
                "optimizer_state_restored": optimizer_state_restored,
                "scheduler_state_restored": scheduler_state_restored,
                "scheduler_rebased_at_resume": args.resume_rebase_scheduler,
                "resume_lr": resume_lr if resume_step else None,
                "schedule_base_lr": schedule_base_lr,
                "rollout": False,
                "loss_scale_term": 0.0,
                "primary_posthoc_gt_alignment": False,
                "frozen_geometry_parameters": frozen_parameters,
                "trainable_predictor_parameters": trainable_parameters,
                "training_losses": ["scene_state_smooth_l1", "scene_state_cosine", "front_all_pixel_log_depth_l1"],
                "dynamic_static_metrics_validation_only": True,
            },
            output_dir / "run_config.json",
        )
    initial = validate_online(
        predictor,
        geometry_model,
        validation_loader,
        device,
        "two_pass",
        unified_all_pairs=False,
        geometry_microbatch_size=args.geometry_microbatch_size,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if rank == 0:
        print(json.dumps({
            "event": "validation",
            "step": resume_step,
            "resumed_initial": bool(resume_step),
            **initial,
        }), flush=True)
    if args.validation_only:
        _cleanup_distributed()
        return
    best_absrel = float(initial["predicted_all_absrel"])
    best_static_absrel = float(initial["predicted_static_absrel"])
    iterator = iter(train_loader)
    epoch = 0
    started = time.time()
    predictor.train()
    for step in range(resume_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(8, device=device)
        for micro_step in range(args.grad_accumulation):
            while True:
                items, iterator, epoch = _next_batch(iterator, train_loader, train_sampler, epoch)
                states, contexts, valid, _ = extract_independent_pairs_chunked(
                    geometry_model,
                    items,
                    device,
                    "two_pass",
                    geometry_microbatch_size=args.geometry_microbatch_size,
                )
                indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
                sums[7] += len(items) - len(indices)
                if indices:
                    break
            states = states[indices].float()
            contexts = [contexts[i] for i in indices]
            target = states[:, 4]
            synchronize = micro_step + 1 == args.grad_accumulation
            gradient_context = (
                predictor.no_sync()
                if isinstance(predictor, DistributedDataParallel) and not synchronize
                else nullcontext()
            )
            with gradient_context:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    predicted = predictor(states[:, :4])
                    state_loss = F.smooth_l1_loss(predicted, target, beta=0.1)
                    cosine_loss = 1 - F.cosine_similarity(predicted.flatten(1), target.flatten(1)).mean()
                    pred_depth = _decode_meters(geometry_model, predicted, contexts, device)
                    gt = torch.stack([item["target_depth"] for item in contexts]).to(device).float()
                    target_valid = torch.stack([item["target_valid"] for item in contexts]).to(device).bool()
                    valid_depth = target_valid & torch.isfinite(pred_depth) & (pred_depth > 0)
                    depth_loss = _region_log_loss(pred_depth, gt, valid_depth)
                    loss, state_contribution, depth_contribution = _one_step_loss_terms(
                        state_loss,
                        cosine_loss,
                        depth_loss,
                    )
                    loss = loss / args.grad_accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}")
                loss.backward()
            sums[:6] += torch.stack(
                (
                    loss.detach() * args.grad_accumulation,
                    state_loss.detach(),
                    cosine_loss.detach(),
                    depth_loss.detach(),
                    state_contribution.detach(),
                    depth_contribution.detach(),
                )
            )
            sums[6] += len(indices)
        grad_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        sums[:6] /= args.grad_accumulation
        if dist.is_initialized():
            dist.all_reduce(sums)
            sums[:6] /= world_size
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            print(json.dumps({
                "event": "train", "step": step, "total_loss": float(sums[0]),
                "state_loss": float(sums[1]), "cosine_loss": float(sums[2]),
                "front_depth_log_l1": float(sums[3]),
                "state_loss_contribution": float(sums[4]),
                "depth_loss_contribution": float(sums[5]),
                "state_loss_fraction": float(sums[4] / (sums[4] + sums[5]).clamp_min(1e-8)),
                "accepted_samples": int(sums[6]),
                "rejected_samples": int(sums[7]), "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "vggt_forwards_per_raw_batch": 2,
                "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "elapsed_seconds": time.time() - started,
            }), flush=True)
        if args.validate_every and step % args.validate_every == 0:
            metrics = validate_online(
                predictor,
                geometry_model,
                validation_loader,
                device,
                "two_pass",
                unified_all_pairs=False,
                geometry_microbatch_size=args.geometry_microbatch_size,
            )
            if rank == 0:
                print(json.dumps({"event": "validation", "step": step, **metrics}), flush=True)
                unwrapped = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
                checkpoint = {
                    "step": step,
                    "config": asdict(predictor_config),
                    "model": {
                        key: value.detach().cpu()
                        for key, value in unwrapped.state_dict().items()
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "validation": metrics,
                }
                _atomic_torch_save(checkpoint, output_dir / "latest.pt")
                if float(metrics["predicted_all_absrel"]) < best_absrel:
                    best_absrel = float(metrics["predicted_all_absrel"])
                    _atomic_torch_save(checkpoint, output_dir / "best_onestep.pt")
                if float(metrics["predicted_static_absrel"]) < best_static_absrel:
                    best_static_absrel = float(metrics["predicted_static_absrel"])
                    _atomic_torch_save(checkpoint, output_dir / "best_static.pt")
            predictor.train()
    _cleanup_distributed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Load predictor weights but restart optimizer, scheduler, and step count.",
    )
    parser.add_argument(
        "--resume-rebase-scheduler",
        action="store_true",
        help=(
            "Restore optimizer moments and keep the checkpoint LR exactly at the resume step, "
            "then cosine-decay continuously from that LR to --final-lr at --steps."
        ),
    )
    parser.add_argument("--train-prepared-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-motion-mask-root", required=True)
    parser.add_argument("--validation-prepared-root", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--validation-motion-mask-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-sequences", type=int, default=139367)
    parser.add_argument(
        "--resume-sampler-global-offset",
        type=int,
        default=0,
        help="Number of globally consumed training samples to skip in resumed epoch 0.",
    )
    parser.add_argument("--validation-sequences", type=int, default=102)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--validation-loader-workers", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument(
        "--geometry-microbatch-size",
        type=int,
        default=0,
        help="Chunk only frozen VGGT/tokenizer extraction; 0 uses the full compute batch.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser


def main() -> None:
    args = _parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
