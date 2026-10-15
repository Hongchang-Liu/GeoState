from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from GeoState.data.batching import (
    collate_scene_targets,
    slice_depth_prediction,
)
from GeoState.core.config import ExperimentConfig
from GeoState.data.common import CachedSceneDataset
from GeoState.core.geometry import paired_depth_scale, robust_scene_scale
from GeoState.evaluation.metrics import depth_metrics
from GeoState.training.losses import geometry_losses_per_scene
from GeoState.models.tokenizer import GeometrySceneModel
from GeoState.training.supervision import build_target_supervision
from GeoState.training.teacher_cache import TeacherCacheBuilder
from GeoState.core.types import VGGTOutput
from GeoState.models.vggt_adapter import VGGTOmegaAdapter
from GeoState.data.waymo import WaymoPreparedDataset


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible worker process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _remap_dataset_root(
    record: object,
    dataset_name: str,
    new_root: str | Path | None,
) -> object:
    """Return a copied scene record rooted at a runtime-local dataset copy."""
    if new_root is None or not isinstance(record, dict):
        return record
    if str(record.get("dataset", "")).lower() != dataset_name.lower():
        return record
    remapped = dict(record)
    old_path = Path(str(remapped["path"]))
    anchor_index = next(
        (
            index
            for index, part in enumerate(old_path.parts)
            if part.lower() == f"{dataset_name.lower()}_prepared"
        ),
        None,
    )
    relative = (
        Path(*old_path.parts[anchor_index + 1 :])
        if anchor_index is not None
        else Path(old_path.name)
    )
    new_path = Path(new_root) / relative
    if not (new_path / "transforms.json").is_file():
        raise FileNotFoundError(
            f"Runtime {dataset_name} scene is incomplete or missing: {new_path}"
        )
    remapped["path"] = str(new_path)
    image_root = remapped.get("image_root")
    if image_root:
        old_image_root = Path(str(image_root))
        try:
            image_relative = old_image_root.relative_to(old_path)
        except ValueError:
            image_relative = None
        if image_relative is not None:
            remapped["image_root"] = str(new_path / image_relative)
    return remapped


class _SceneBatchPrefetcher:
    """Deterministic single-producer bounded queue for CPU scene batches."""

    def __init__(
        self,
        dataset: object,
        *,
        start_cursor: int,
        batch_size: int,
        start_forward_index: int,
        input_views_for_forward: object,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("prefetch capacity must be positive")
        self.dataset = dataset
        self.cursor = int(start_cursor)
        self.batch_size = int(batch_size)
        self.forward_index = int(start_forward_index)
        self.input_views_for_forward = input_views_for_forward
        self.queue: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._produce,
            name="scene-batch-producer",
            daemon=True,
        )
        self.thread.start()

    def _put(self, item: object) -> bool:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        while not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                batches, next_cursor = self.dataset.get_compatible_scene_batch(
                    self.cursor,
                    self.batch_size,
                    self.input_views_for_forward(self.forward_index),
                )
            except BaseException as exc:
                self._put(("error", exc))
                return
            load_seconds = time.perf_counter() - started
            if not self._put(
                (
                    "batch",
                    batches,
                    next_cursor,
                    self.forward_index,
                    load_seconds,
                )
            ):
                return
            self.cursor = next_cursor
            self.forward_index += 1

    def get(
        self,
    ) -> tuple[list[dict[str, object]], int, int, float, float, int]:
        started = time.perf_counter()
        item = self.queue.get()
        wait_seconds = time.perf_counter() - started
        if item[0] == "error":
            raise item[1]
        _, batches, cursor, forward_index, load_seconds = item
        return (
            batches,
            int(cursor),
            int(forward_index),
            float(load_seconds),
            wait_seconds,
            self.queue.qsize(),
        )

    def close(self) -> None:
        self.stop_event.set()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.thread.join()


def _adapter(config: ExperimentConfig, device: torch.device) -> VGGTOmegaAdapter:
    return VGGTOmegaAdapter(
        config.vggt.source_root,
        config.vggt.checkpoint,
        config.vggt.checkpoint_min_bytes,
        device,
    )


def _scene_record(
    dataset: str, path: Path, convention: str, *, image_root: Path | None = None
) -> dict[str, str]:
    cache_key = path.name if dataset == "dl3dv" else f"{dataset}__{path.name}"
    record = {
        "dataset": dataset,
        "scene_id": f"{dataset}/{path.name}",
        "path": str(path),
        "cache_key": cache_key,
        "source_c2w_convention": convention,
    }
    if image_root is not None:
        record["image_root"] = str(image_root)
    return record


def cache_shard(args: argparse.Namespace) -> None:
    config = ExperimentConfig.load(args.config)
    device = torch.device(args.device)
    teacher = _adapter(config, device)
    builder = TeacherCacheBuilder(
        teacher,
        config.data.cache_root,
        config.data.image_resolution,
        config.data.teacher_frames,
        config.data.teacher_sampling,
        config.alignment.max_rotation_error_deg,
        config.alignment.max_relative_baseline_error,
        config.alignment.max_relative_center_error,
        config.data.confidence_quantile,
        config.data.source_c2w_convention,
    )
    scenes = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))["scenes"]
    requested_datasets = {value.strip() for value in args.datasets.split(",") if value.strip()}
    scenes = [scene for scene in scenes if scene["dataset"] in requested_datasets]
    if args.max_candidates > 0:
        scenes = scenes[: args.max_candidates]
    records = []
    for index in range(args.rank, len(scenes), args.world_size):
        scene = scenes[index]
        cache_path = Path(config.data.cache_root) / f"{scene['cache_key']}.pt"
        if cache_path.is_file() and cache_path.stat().st_size > 1_000_000:
            record = {**scene, "status": "existing", "cache": str(cache_path)}
        else:
            try:
                output = builder.build_scene(
                    scene["path"],
                    dataset_name=scene["dataset"],
                    scene_id=scene["scene_id"],
                    cache_key=scene["cache_key"],
                    source_c2w_convention=scene["source_c2w_convention"],
                    image_root=scene.get("image_root"),
                )
                record = {**scene, "status": "cached", "cache": str(output)}
            except Exception as error:
                record = {**scene, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rank": args.rank, "records": records}, indent=2), encoding="utf-8")


def aggregate_cache(args: argparse.Namespace) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path().glob(args.inputs))]
    records = [record for payload in payloads for record in payload["records"]]
    valid_by_cache_key: dict[str, dict[str, str]] = {}
    for record in records:
        if record["status"] not in {"existing", "cached"}:
            continue
        compact = {
            key: record[key]
            for key in ("dataset", "scene_id", "path", "cache_key", "source_c2w_convention", "image_root")
            if key in record
        }
        valid_by_cache_key[str(record["cache_key"])] = compact
    valid = list(valid_by_cache_key.values())
    requested = len(valid) if args.scenes <= 0 else args.scenes
    if len(valid) < requested:
        failures = [record for record in records if record["status"] == "failed"]
        raise RuntimeError(f"Only {len(valid)} valid caches, need {requested}; failures={failures}")
    selected = valid[:requested]
    counts = Counter(record["dataset"] for record in selected)
    result = {
        "scenes": selected,
        "valid_cache_count": len(valid),
        "duplicate_valid_records_removed": sum(
            record["status"] in {"existing", "cached"} for record in records
        ) - len(valid),
        "selected_scene_count": len(selected),
        "dataset_counts": dict(counts),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


def filter_cache_manifest(args: argparse.Namespace) -> None:
    """Keep only scenes that can form the requested input/target tuple.

    A cache may be valid for teacher generation while still containing too few
    frames to support a training tuple.  Filtering this once, before DDP
    sharding and RAM preload, avoids a late single-rank exception that would
    otherwise terminate the whole job.
    """
    payload = json.loads(Path(args.scene_manifest).read_text(encoding="utf-8"))
    required_frames = args.min_input_views + args.targets_per_input
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for scene in payload["scenes"]:
        cache_path = Path(args.cache_root) / f"{scene['cache_key']}.pt"
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            frame_count = len(cache["frame_indices"])
            del cache
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            rejected.append({
                "scene": scene,
                "reason": f"cache_read_error:{type(error).__name__}",
            })
            continue
        if frame_count < required_frames:
            rejected.append({
                "scene": scene,
                "reason": "insufficient_teacher_frames",
                "frame_count": frame_count,
            })
            continue
        accepted.append(scene)
    if not accepted:
        raise RuntimeError("No train scenes remain after teacher-frame filtering")
    result = {
        "scenes": accepted,
        "scene_count": len(accepted),
        "dataset_counts": dict(Counter(str(scene["dataset"]) for scene in accepted)),
        "min_input_views": args.min_input_views,
        "targets_per_input": args.targets_per_input,
        "required_teacher_frames": required_frames,
        "dropped_scene_count": len(rejected),
        "dropped_by_reason": dict(Counter(str(row["reason"]) for row in rejected)),
        "dropped_by_dataset": dict(
            Counter(str(row["scene"].get("dataset", "default")) for row in rejected)
        ),
        "dropped_scenes": rejected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"scenes", "dropped_scenes"}}))


def create_validation_split(args: argparse.Namespace) -> None:
    """Reserve deterministic scene-level tuples and remove them from training."""
    payload = json.loads(Path(args.scene_manifest).read_text(encoding="utf-8"))
    scenes = list(payload["scenes"])
    grouped: dict[str, list[dict[str, object]]] = {}
    for scene in scenes:
        grouped.setdefault(str(scene["dataset"]), []).append(scene)
    dataset_names = sorted(grouped)
    if args.count < len(dataset_names):
        raise ValueError("Validation count must cover every dataset")
    base, remainder = divmod(args.count, len(dataset_names))
    selected: list[dict[str, object]] = []
    for dataset_index, dataset_name in enumerate(dataset_names):
        candidates = sorted(grouped[dataset_name], key=lambda row: str(row["cache_key"]))
        random.Random(args.seed + dataset_index).shuffle(candidates)
        take = base + int(dataset_index < remainder)
        if len(candidates) < take:
            raise RuntimeError(f"Dataset {dataset_name} has only {len(candidates)} scenes; need {take}")
        selected.extend(candidates[:take])

    tuples = []
    for tuple_index, scene in enumerate(selected):
        cache_path = Path(args.cache_root) / f"{scene['cache_key']}.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing validation teacher cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        frame_indices = [int(value) for value in cache["frame_indices"]]
        register = int(cache.get("register_frame_index", frame_indices[0]))
        candidates = [value for value in frame_indices if value != register]
        if len(candidates) < args.input_views:
            raise RuntimeError(f"Scene {scene['scene_id']} has too few cached frames")
        chosen = random.Random(args.seed * 1009 + tuple_index).sample(candidates, args.input_views)
        input_indices = [register, *chosen[: args.input_views - 1]]
        target_index = chosen[-1]
        tuples.append(
            {
                "tuple_id": f"val-{tuple_index:04d}",
                "scene": scene,
                "input_indices": input_indices,
                "target_index": target_index,
            }
        )

    heldout_keys = {str(item["scene"]["cache_key"]) for item in tuples}
    train_scenes = [scene for scene in scenes if str(scene["cache_key"]) not in heldout_keys]
    train_result = {
        "scenes": train_scenes,
        "scene_count": len(train_scenes),
        "dataset_counts": dict(Counter(str(scene["dataset"]) for scene in train_scenes)),
        "validation_scene_count": len(tuples),
    }
    validation_result = {
        "tuples": tuples,
        "tuple_count": len(tuples),
        "input_views": args.input_views,
        "seed": args.seed,
        "dataset_counts": dict(Counter(str(item["scene"]["dataset"]) for item in tuples)),
        "scene_level_heldout": True,
    }
    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.write_text(json.dumps(train_result, indent=2), encoding="utf-8")
    validation_output.write_text(json.dumps(validation_result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "train": {key: value for key, value in train_result.items() if key != "scenes"},
                "validation": {key: value for key, value in validation_result.items() if key != "tuples"},
            }
        )
    )


def sample_memory_view_dropout_mask(
    batch_size: int,
    view_count: int,
    anchor_probability: float,
    random_nonanchor_probability: float,
    device: torch.device,
    *,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample mutually exclusive anchor/random view-memory dropout.

    The reference view is index zero in the training tuples. For V=1 or
    evaluation every view remains visible. For V>=2 each scene independently
    chooses exactly one of: hide anchor, hide one random non-anchor, or keep
    every view. This keeps the information budget matched between the two
    dropout interventions and can never hide every view.
    """
    if batch_size < 1 or view_count < 1:
        raise ValueError("batch_size and view_count must be positive")
    mask = torch.ones(
        batch_size,
        view_count,
        device=device,
        dtype=torch.bool,
    )
    no_dropout = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if (
        not enabled
        or view_count == 1
        or (anchor_probability == 0.0 and random_nonanchor_probability == 0.0)
    ):
        return mask, no_dropout, no_dropout.clone()

    choices = torch.rand(batch_size, device=device)
    anchor_dropout = choices < anchor_probability
    random_dropout = (
        (choices >= anchor_probability)
        & (
            choices
            < anchor_probability + random_nonanchor_probability
        )
    )
    mask[anchor_dropout, 0] = False
    random_rows = random_dropout.nonzero(as_tuple=False).flatten()
    if random_rows.numel():
        random_views = torch.randint(
            1,
            view_count,
            (random_rows.numel(),),
            device=device,
        )
        mask[random_rows, random_views] = False
    return mask, anchor_dropout, random_dropout


class DistributedLossModule(nn.Module):
    def __init__(self, model: GeometrySceneModel, config: ExperimentConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def _dummy_loss(self) -> torch.Tensor:
        return sum(parameter.reshape(-1)[0] * 0.0 for parameter in self.model.trainable_parameters())

    @torch.no_grad()
    def extract_vggt_output(self, batch: dict[str, object]):
        """Run the frozen input-only backbone once for a multi-target update group."""
        device = next(self.model.tokenizer.parameters()).device
        self.model.vggt.assert_frozen()
        input_images_cpu = torch.as_tensor(batch["input_images"])
        if device.type == "cuda" and not input_images_cpu.is_pinned():
            input_images_cpu = input_images_cpu.pin_memory()
        input_images = input_images_cpu.to(device=device, non_blocking=True).unsqueeze(0)
        if input_images.dtype == torch.uint8:
            input_images = input_images.to(torch.float32).mul_(1.0 / 255.0)
        return self.model.vggt(input_images)

    @torch.no_grad()
    def extract_vggt_batch(self, batches: list[dict[str, object]]) -> VGGTOutput:
        """Run one frozen-VGGT forward for an equal-shaped scene batch."""
        if not batches:
            raise ValueError("A scene batch must contain at least one scene")
        shapes = {tuple(torch.as_tensor(batch["input_images"]).shape) for batch in batches}
        if len(shapes) != 1:
            raise ValueError("A full compute batch requires equal input image shapes")
        device = next(self.model.tokenizer.parameters()).device
        self.model.vggt.assert_frozen()
        image_tensors = [torch.as_tensor(batch["input_images"]) for batch in batches]
        if device.type == "cuda":
            first = image_tensors[0]
            images_cpu = torch.empty(
                (len(image_tensors), *first.shape),
                dtype=first.dtype,
                pin_memory=True,
            )
            torch.stack(image_tensors, out=images_cpu)
        else:
            images_cpu = torch.stack(image_tensors)
        images = images_cpu.to(device=device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.to(torch.float32).mul_(1.0 / 255.0)
        return self.model.vggt(images)

    def _encode_scene_batch(
        self,
        batches: list[dict[str, object]],
        vggt_output: VGGTOutput,
        target_slot: int,
    ) -> tuple[object, torch.Tensor]:
        """Build the target-independent trainable scene encoding once.

        ``target_slot`` selects the held-out query while the frozen VGGT
        features are shared across targets from the same input scene.
        """
        batch = collate_scene_targets(batches, target_slot)
        device = next(self.model.tokenizer.parameters()).device
        input_c2w = torch.as_tensor(batch["input_c2w"], device=device)
        input_intrinsics = torch.as_tensor(batch["input_intrinsics"], device=device)
        batch_size = len(batches)
        if vggt_output.c2w.shape[0] != batch_size:
            raise ValueError("VGGT output batch size does not match scene batch size")

        register_valid = torch.tensor(
            [
                int(batch["input_frame_indices"][index][0])
                == int(batch["teacher_register_frame_index"][index])
                and str(batch["input_frame_ids"][index][0])
                == str(batch["teacher_register_frame_id"][index])
                for index in range(batch_size)
            ],
            device=device,
            dtype=torch.bool,
        )
        strict_scene_mask = batch.get("scene_tokenizer_view_mask")
        if strict_scene_mask is not None:
            strict_scene_mask = torch.as_tensor(
                strict_scene_mask, device=device, dtype=torch.bool
            )
        alignment_scene_mask = strict_scene_mask
        if strict_scene_mask is not None:
            # Waymo supplies calibrated camera poses and permits target-LiDAR
            # scale anchoring.  One visible same-time camera plus that scale
            # uniquely fixes dataset->VGGT Sim(3).  Keep every same-time camera
            # in tokenizer attention, but do not let VGGT camera-head relative
            # pose noise make the optimizer batch size stochastic.
            first_visible = strict_scene_mask.to(torch.int64).argmax(dim=1)
            alignment_scene_mask = torch.zeros_like(strict_scene_mask)
            alignment_scene_mask.scatter_(1, first_visible[:, None], True)
        alignment_view_counts = (
            alignment_scene_mask.sum(dim=1)
            if alignment_scene_mask is not None
            else torch.full(
                (batch_size,), input_c2w.shape[1], device=device, dtype=torch.long
            )
        )
        single_view_scale = None
        if bool((alignment_view_counts == 1).any()):
            if "input_scale_depth_dataset" in batch:
                scale_view_indices = torch.as_tensor(
                    batch["input_scale_view_index"], device=device, dtype=torch.long
                ).reshape(batch_size)
                if not torch.equal(
                    scale_view_indices,
                    alignment_scene_mask.to(torch.int64).argmax(dim=1),
                ):
                    raise ValueError(
                        "Paired-scale view must be the camera used for one-view Sim(3) alignment"
                    )
                scale_rows = torch.arange(batch_size, device=device)
                input_scale_depth = torch.as_tensor(
                    batch["input_scale_depth_dataset"], device=device
                )
                input_scale_valid = torch.as_tensor(
                    batch["input_scale_valid_dataset"],
                    device=device,
                    dtype=torch.bool,
                )
                input_scale_depth = torch.where(
                    input_scale_valid,
                    input_scale_depth,
                    torch.zeros_like(input_scale_depth),
                )
                single_view_scale, paired_valid, paired_support = paired_depth_scale(
                    vggt_output.depth[scale_rows, scale_view_indices],
                    input_scale_depth,
                    vggt_output.confidence[scale_rows, scale_view_indices],
                )
                register_valid &= paired_valid
            elif any(value is None for value in batch["teacher_scene_scale"]):
                register_valid.zero_()
            else:
                scale_view_indices = (
                    alignment_scene_mask.to(torch.int64).argmax(dim=1)
                    if alignment_scene_mask is not None
                    else torch.zeros(batch_size, device=device, dtype=torch.long)
                )
                scale_rows = torch.arange(batch_size, device=device)
                student_scene_scale = robust_scene_scale(
                    vggt_output.depth[scale_rows, scale_view_indices].unsqueeze(1),
                    vggt_output.confidence[scale_rows, scale_view_indices].unsqueeze(1),
                )
                teacher_scene_scale = torch.stack(
                    [
                        torch.as_tensor(
                            value,
                            device=device,
                            dtype=student_scene_scale.dtype,
                        )
                        for value in batch["teacher_scene_scale"]
                    ]
                ).reshape(batch_size)
                dataset_to_teacher_scale = torch.as_tensor(
                    batch["dataset_to_teacher"]["scale"],
                    device=device,
                    dtype=student_scene_scale.dtype,
                ).reshape(batch_size)
                uses_dataset_depth = torch.as_tensor(
                    batch["target_uses_dataset_depth"],
                    device=device,
                    dtype=torch.bool,
                ).reshape(batch_size)
                # Waymo's input-view LiDAR is the formal metric gauge. Dense
                # pseudo depth may supervise decoder pixels, but it must never
                # provide a future-target scale to the scene encoding.
                uses_dataset_scale = uses_dataset_depth
                dataset_scene_scale = torch.as_tensor(
                    batch["target_scene_scale_dataset"],
                    device=device,
                    dtype=student_scene_scale.dtype,
                ).reshape(batch_size)
                teacher_scene_scale = torch.where(
                    uses_dataset_scale,
                    dataset_scene_scale,
                    teacher_scene_scale,
                )
                dataset_to_teacher_scale = torch.where(
                    uses_dataset_scale,
                    torch.ones_like(dataset_to_teacher_scale),
                    dataset_to_teacher_scale,
                )
                single_view_scale = (
                    dataset_to_teacher_scale
                    * student_scene_scale
                    / teacher_scene_scale.clamp_min(1e-8)
                )

        attention_view_mask, anchor_dropout, random_dropout = (
            sample_memory_view_dropout_mask(
                batch_size,
                input_c2w.shape[1],
                self.config.train.anchor_view_dropout_probability,
                self.config.train.random_view_dropout_probability,
                device,
                enabled=self.model.training,
            )
        )
        if strict_scene_mask is not None:
            if strict_scene_mask.shape != attention_view_mask.shape:
                raise ValueError(
                    "scene_tokenizer_view_mask must match the frozen-VGGT view shape"
                )
            attention_view_mask = attention_view_mask & strict_scene_mask
            if not attention_view_mask.any(dim=1).all():
                raise ValueError("Waymo scene-time masking removed every tokenizer view")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoding = self.model.encode_vggt_output(
                vggt_output,
                input_c2w,
                input_intrinsics=input_intrinsics,
                frame_ids=batch["input_frame_ids"],
                single_view_scale=single_view_scale,
                tokenizer_attention_view_mask=attention_view_mask,
                alignment_view_mask=alignment_scene_mask,
            )
        encoding.metadata["anchor_view_dropout_applied"] = anchor_dropout
        encoding.metadata["random_view_dropout_applied"] = random_dropout
        if "input_scale_depth_dataset" in batch:
            encoding.metadata["scale_alignment_protocol"] = "input_same_pixel_vggt_lidar"
            encoding.metadata["input_paired_scale"] = single_view_scale
            encoding.metadata["input_paired_scale_support"] = paired_support
        return encoding, register_valid

    def _forward_scene_batch(
        self,
        batches: list[dict[str, object]],
        vggt_output: VGGTOutput,
        target_slot: int,
        shared_encoding=None,
        shared_register_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Evaluate the paper log-depth and gradient objective for one target slot."""
        batch = collate_scene_targets(batches, target_slot)
        device = next(self.model.tokenizer.parameters()).device
        batch_size = len(batches)
        try:
            if shared_encoding is None:
                encoding, register_valid = self._encode_scene_batch(
                    batches, vggt_output, target_slot
                )
            else:
                if shared_register_valid is None:
                    raise ValueError("Shared encoding requires its register-valid mask")
                encoding, register_valid = shared_encoding, shared_register_valid
            with torch.autocast(device_type="cuda", enabled=False):
                supervision = build_target_supervision(
                    batch,
                    encoding,
                    min_overlap=self.config.alignment.min_overlap,
                    max_projection_relative_error=self.config.alignment.max_projection_relative_error,
                )
            valid_scenes = (
                register_valid
                & encoding.metadata["alignment"].valid
                & (supervision.alignment_valid | supervision.uses_dataset_depth)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.model.decode_depth(
                    encoding,
                    supervision.query_c2w_reference,
                    torch.as_tensor(batch["target_intrinsics"], device=device),
                    tuple(int(value) for value in batch["image_hw"]),
                    query_is_dataset_camera=False,
                )
                losses = geometry_losses_per_scene(
                    prediction,
                    supervision.normalized_z_depth,
                    supervision.valid,
                    supervision.weight,
                    log_depth_weight=self.config.train.log_depth_weight,
                    gradient_weight=self.config.train.gradient_weight,
                )
            rows: list[dict[str, object]] = []
            for index in range(batch_size):
                if not bool(valid_scenes[index]):
                    rows.append({
                        "skipped": True,
                        "reason": "alignment-or-projection-sanity",
                        "dataset_name": batch["dataset_name"][index],
                        "scene_id": batch["scene_id"][index],
                    })
                    continue
                item = slice(index, index + 1)
                item_prediction = slice_depth_prediction(prediction, index)
                metrics = depth_metrics(
                    item_prediction.z_depth,
                    supervision.normalized_z_depth[item],
                    supervision.valid[item],
                )
                observed = depth_metrics(
                    item_prediction.z_depth,
                    supervision.normalized_z_depth[item],
                    supervision.observed[item],
                )
                unobserved = depth_metrics(
                    item_prediction.z_depth,
                    supervision.normalized_z_depth[item],
                    supervision.valid[item] & ~supervision.observed[item],
                )
                total_loss = losses.total[index]
                rows.append({
                    **metrics,
                    "skipped": False,
                    "loss": float(total_loss.detach()),
                    "log_depth_loss": float(losses.log_depth[index].detach()),
                    "gradient_loss": float(losses.gradient[index].detach()),
                    "abs_rel_observed": observed["abs_rel"],
                    "abs_rel_unobserved": unobserved["abs_rel"],
                    "overlap": float(supervision.overlap[index]),
                    "projection_relative_error": float(supervision.projection_relative_error[index]),
                    "scene_id": batch["scene_id"][index],
                    "dataset_name": batch["dataset_name"][index],
                    "num_inputs": len(batch["input_frame_ids"][index]),
                    "scene_token_views": int(batch.get(
                        "scene_token_view_count",
                        [len(batch["input_frame_ids"][index])] * batch_size,
                    )[index]),
                    "reconstruction_task": str(batch.get(
                        "reconstruction_task", ["standard"] * batch_size
                    )[index]),
                    "student_vggt_timestamp_count": int(batch.get(
                        "student_vggt_timestamp_count", [1] * batch_size
                    )[index]),
                    "target_horizon_seconds": float(batch.get(
                        "target_horizon_seconds", [float("nan")] * batch_size
                    )[index]),
                    "target_dataset_gt": float(supervision.uses_dataset_depth[index]),
                    "dynamic_downweighted_fraction": float(torch.as_tensor(batch.get(
                        "dynamic_downweighted_fraction", torch.zeros(batch_size)
                    ))[index]),
                    "anchor_view_dropout_applied": float(
                        encoding.metadata["anchor_view_dropout_applied"][index]
                    ),
                    "random_view_dropout_applied": float(
                        encoding.metadata["random_view_dropout_applied"][index]
                    ),
                    "target_slot": target_slot,
                })
            valid_count = int(valid_scenes.sum())
            loss_sum = losses.total[valid_scenes].sum() if valid_count else self._dummy_loss()
            return loss_sum, {
                "samples": rows,
                "valid_count": valid_count,
                "batch_size": batch_size,
            }
        except RuntimeError as error:
            if os.environ.get("VGGT_RERAISE_BATCH_ERRORS") == "1":
                raise
            return self._dummy_loss(), {
                "samples": [{
                    "skipped": True,
                    "reason": str(error),
                    "dataset_name": batch["dataset_name"][index],
                    "scene_id": batch["scene_id"][index],
                } for index in range(batch_size)],
                "valid_count": 0,
                "batch_size": batch_size,
            }

    def _forward_scene_targets_batch(
        self,
        batches: list[dict[str, object]],
        vggt_output: VGGTOutput,
        target_slots: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not target_slots:
            raise ValueError("At least one target slot is required")
        encoding, register_valid = self._encode_scene_batch(
            batches, vggt_output, target_slots[0]
        )
        losses: list[torch.Tensor] = []
        rows: list[dict[str, object]] = []
        valid_count = 0
        for target_slot in target_slots:
            loss, metrics = self._forward_scene_batch(
                batches,
                vggt_output,
                target_slot,
                shared_encoding=encoding,
                shared_register_valid=register_valid,
            )
            losses.append(loss)
            rows.extend(metrics["samples"])
            valid_count += int(metrics["valid_count"])
        return sum(losses), {
            "samples": rows,
            "valid_count": valid_count,
            "batch_size": len(rows),
        }

    def forward(
        self,
        batch: dict[str, object] | list[dict[str, object]],
        vggt_output=None,
        target_slot: int | tuple[int, ...] = 0,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        is_single = not isinstance(batch, list)
        batches = [batch] if is_single else batch
        if vggt_output is None:
            if not is_single:
                raise ValueError("Batched training requires a precomputed VGGT output")
            vggt_output = self.extract_vggt_output(batch)
        if isinstance(target_slot, tuple):
            loss, result = self._forward_scene_targets_batch(
                batches, vggt_output, target_slot
            )
        else:
            loss, result = self._forward_scene_batch(
                batches, vggt_output, target_slot
            )
        return (loss, result["samples"][0]) if is_single else (loss, result)

def shard_scene_records_by_dataset(
    scene_records: list[dict[str, object]], rank: int, world_size: int
) -> list[dict[str, object]]:
    """Give each rank a disjoint, source-balanced cache shard."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in scene_records:
        grouped.setdefault(str(record["dataset"]), []).append(record)
    local = [
        record
        for dataset_name in sorted(grouped)
        for record in grouped[dataset_name][rank::world_size]
    ]
    if not local:
        raise RuntimeError(f"Rank {rank} received no scenes from {len(scene_records)} records")
    return local


def _move_vggt_output(output: VGGTOutput, device: torch.device) -> VGGTOutput:
    """Move cached frozen features without changing fp32 geometry tensors."""
    is_cpu = device.type == "cpu"
    return VGGTOutput(
        patch_features=[
            value.detach().to(device=device, dtype=torch.bfloat16 if is_cpu else value.dtype)
            for value in output.patch_features
        ],
        depth=output.depth.detach().to(device),
        confidence=output.confidence.detach().to(device),
        world_points=output.world_points.detach().to(device),
        c2w=output.c2w.detach().to(device),
        intrinsics=output.intrinsics.detach().to(device),
        patch_size=output.patch_size,
        patch_hw=output.patch_hw,
    )


def prepare_fixed_validation(
    wrapper: DistributedLossModule,
    validation_manifest: str | Path,
    config: ExperimentConfig,
    patch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    use_tartanair_gt_depth: bool = False,
    view_counts: tuple[int, ...] = (1, 2, 3, 4),
    tartanair_runtime_root: str | Path | None = None,
) -> list[tuple[dict[str, object], VGGTOutput]]:
    """Run frozen VGGT once per fixed tuple and retain only CPU outputs."""
    tuples = json.loads(Path(validation_manifest).read_text(encoding="utf-8"))["tuples"]
    if tartanair_runtime_root is not None:
        tuples = [
            {
                **item,
                "scene": _remap_dataset_root(
                    item["scene"], "tartanair", tartanair_runtime_root
                ),
            }
            for item in tuples
        ]
    local_specs = tuples[rank::world_size]
    dataset = CachedSceneDataset(
        [item["scene"] for item in local_specs],
        config.data.cache_root,
        patch_size,
        config.data.image_resolution,
        min_views=1,
        max_views=4,
        samples_per_scene=1,
        seed=config.train.seed,
        source_c2w_convention=config.data.source_c2w_convention,
        require_teacher_register_match=True,
        use_tartanair_gt_depth=use_tartanair_gt_depth,
        scene_load_workers=1,
    )
    cached = []
    for local_index, spec in enumerate(local_specs):
        all_inputs = [int(value) for value in spec["input_indices"]]
        for view_count in view_counts:
            if view_count > len(all_inputs):
                continue
            batch = dataset.build_tuple(
                local_index,
                all_inputs[:view_count],
                int(spec["target_index"]),
            )
            output = wrapper.extract_vggt_output(batch)
            batch = dict(batch)
            batch.pop("input_images")
            batch["validation_tuple_id"] = str(spec["tuple_id"])
            batch["validation_num_inputs"] = view_count
            cached.append((batch, _move_vggt_output(output, torch.device("cpu"))))
    print(
        f"[validation] rank={rank} precomputed frozen VGGT for {len(cached)} tuples",
        flush=True,
    )
    return cached


def prepare_waymo_validation(
    wrapper: DistributedLossModule,
    prepared_root: str | Path,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    examples: int,
    seed: int,
    static_temporal_manifest: str | Path,
) -> list[tuple[dict[str, object], VGGTOutput]]:
    """Build fixed Waymo tuples and cache their frozen-VGGT outputs.

    Static validation uses three source-time cameras once, then queries FRONT
    at every 0.2-second horizon through +2.0 seconds.  Future RGB is absent.
    """
    dataset = WaymoPreparedDataset(
        prepared_root,
        seed=seed,
        samples_per_segment=1,
        decode_workers=4,
        static_temporal_manifest=static_temporal_manifest,
    )
    cached: list[tuple[dict[str, object], VGGTOutput]] = []
    for example_index in range(rank, examples, world_size):
        cpu_output = None
        source_frame_ids = None
        for horizon_index in range(1, 11):
            batch = dataset.build_static_validation_tuple(
                example_index, horizon_index
            )
            if cpu_output is None:
                output = wrapper.extract_vggt_output(batch)
                cpu_output = _move_vggt_output(output, torch.device("cpu"))
                source_frame_ids = list(batch["input_frame_ids"])
            elif list(batch["input_frame_ids"]) != source_frame_ids:
                raise AssertionError("Future validation changed source VGGT inputs")
            batch = dict(batch)
            batch.pop("input_images")
            horizon = float(batch["target_horizon_seconds"])
            batch["dataset_name"] = f"waymo_future_t+{horizon:.1f}"
            batch["validation_tuple_id"] = (
                f"waymo_future_{example_index:04d}_t+{horizon:.1f}"
            )
            batch["validation_num_inputs"] = 3
            cached.append((batch, cpu_output))
    dataset.close()
    print(
        f"[validation] rank={rank} precomputed frozen VGGT for "
        f"{len(range(rank, examples, world_size))} static anchors and "
        f"{len(cached)} future FRONT queries",
        flush=True,
    )
    return cached


@torch.inference_mode()
def run_fixed_validation(
    wrapper: DistributedLossModule,
    cached: list[tuple[dict[str, object], VGGTOutput]],
    device: torch.device,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[int, dict[str, float]],
    dict[str, dict[int, dict[str, float]]],
]:
    """Evaluate deterministic held-out tuples and aggregate across all ranks."""
    metric_names = (
        "loss", "log_l1", "abs_rel", "delta1", "abs_rel_observed", "abs_rel_unobserved",
        "coverage", "overlap",
    )
    was_training = wrapper.model.training
    wrapper.model.eval()
    local_rows: list[dict[str, object]] = []
    for batch, cpu_output in cached:
        output = _move_vggt_output(cpu_output, device)
        _, metrics = wrapper(batch, output, 0)
        if bool(metrics["skipped"]):
            local_rows.append(
                {"dataset": str(batch["dataset_name"]), "skipped": True, "reason": str(metrics.get("reason", "unknown"))}
            )
            continue
        local_rows.append(
            {
                "dataset": str(batch["dataset_name"]),
                "num_inputs": int(batch.get("validation_num_inputs", 4)),
                "valid_pixels": int(metrics.get("valid_pixels", 0)),
                "target_horizon_seconds": float(
                    batch.get("target_horizon_seconds", float("nan"))
                ),
                "skipped": False,
                **{name: float(metrics[name]) for name in metric_names},
            }
        )
    if was_training:
        wrapper.model.train()
    gathered: list[list[dict[str, object]]] = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_rows)
    rows = [row for rank_rows in gathered for row in rank_rows if not bool(row["skipped"])]
    if not rows:
        raise RuntimeError("Every fixed validation tuple was rejected")

    def aggregate(items: list[dict[str, object]]) -> dict[str, float]:
        result = {}
        for name in metric_names:
            finite_items = [
                item
                for item in items
                if math.isfinite(float(item[name]))
            ]
            if name in ("abs_rel", "delta1"):
                pixel_count = sum(int(item["valid_pixels"]) for item in finite_items)
                result[name] = (
                    sum(
                        float(item[name]) * int(item["valid_pixels"])
                        for item in finite_items
                    )
                    / pixel_count
                    if pixel_count
                    else float("nan")
                )
            else:
                values = [float(item[name]) for item in finite_items]
                result[name] = float(np.mean(values)) if values else float("nan")
        result["valid_pixels"] = float(
            sum(int(item["valid_pixels"]) for item in items)
        )
        result["accepted_tuples"] = float(len(items))
        return result

    headline_rows = [row for row in rows if int(row["num_inputs"]) == 4] or rows
    overall = aggregate(headline_rows)
    per_dataset = {
        dataset_name: aggregate(
            [
                row
                for row in headline_rows
                if str(row["dataset"]) == dataset_name
            ]
        )
        for dataset_name in sorted({str(row["dataset"]) for row in rows})
    }
    overall["attempted_tuples"] = float(sum(len(rank_rows) for rank_rows in gathered))
    per_view = {
        view_count: aggregate(
            [row for row in rows if int(row["num_inputs"]) == view_count]
        )
        for view_count in sorted({int(row["num_inputs"]) for row in rows})
    }
    per_dataset_view = {
        dataset_name: {
            view_count: aggregate(
                [
                    row
                    for row in rows
                    if str(row["dataset"]) == dataset_name
                    and int(row["num_inputs"]) == view_count
                ]
            )
            for view_count in sorted({int(row["num_inputs"]) for row in rows})
        }
        for dataset_name in sorted({str(row["dataset"]) for row in rows})
    }
    return overall, per_dataset, per_view, per_dataset_view


@torch.inference_mode()
def run_streaming_waymo_static_validation(
    wrapper: DistributedLossModule,
    dataset: WaymoPreparedDataset,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float]],
    dict[int, dict[str, float]],
    dict[str, dict[int, dict[str, float]]],
]:
    """Evaluate every static validation window without retaining VGGT features.

    One three-view VGGT forward is reused for all ten held-out future FRONT
    queries of a window. Only sufficient statistics are kept, so the complete
    5,827-window pool has bounded host and GPU memory.
    """
    metric_names = (
        "loss", "log_l1", "abs_rel", "delta1", "abs_rel_observed",
        "abs_rel_unobserved", "coverage", "overlap",
    )
    horizon_count = 10
    sums = torch.zeros(horizon_count, len(metric_names), device=device, dtype=torch.float64)
    accepted = torch.zeros(horizon_count, device=device, dtype=torch.float64)
    attempted = torch.zeros(horizon_count, device=device, dtype=torch.float64)
    valid_pixels = torch.zeros(horizon_count, device=device, dtype=torch.float64)
    was_training = wrapper.model.training
    wrapper.model.eval()
    record_count = len(dataset._static_records)
    started = time.time()
    for local_position, example_index in enumerate(range(rank, record_count, world_size), 1):
        cpu_batch = dataset.build_static_validation_tuple(example_index, 1)
        output = wrapper.extract_vggt_output(cpu_batch)
        source_frame_ids = list(cpu_batch["input_frame_ids"])
        for horizon_index in range(1, horizon_count + 1):
            batch = (
                cpu_batch
                if horizon_index == 1
                else dataset.build_static_validation_tuple(example_index, horizon_index)
            )
            if list(batch["input_frame_ids"]) != source_frame_ids:
                raise AssertionError("Streaming validation changed source VGGT inputs")
            batch = dict(batch)
            batch.pop("input_images")
            horizon = float(batch["target_horizon_seconds"])
            batch["dataset_name"] = f"waymo_future_t+{horizon:.1f}"
            attempted[horizon_index - 1] += 1
            _, metrics = wrapper(batch, output, 0)
            if bool(metrics["skipped"]):
                continue
            pixels = float(metrics.get("valid_pixels", 0))
            accepted[horizon_index - 1] += 1
            valid_pixels[horizon_index - 1] += pixels
            for metric_index, name in enumerate(metric_names):
                value = float(metrics[name])
                if not math.isfinite(value):
                    continue
                weight = pixels if name in ("abs_rel", "delta1") else 1.0
                sums[horizon_index - 1, metric_index] += value * weight
        if local_position % 100 == 0:
            print(
                json.dumps({
                    "event": "full_static_validation_progress",
                    "rank": rank,
                    "anchors_done": local_position,
                    "anchors_local_total": len(range(rank, record_count, world_size)),
                    "elapsed_seconds": time.time() - started,
                }),
                flush=True,
            )
    for tensor in (sums, accepted, attempted, valid_pixels):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if was_training:
        wrapper.model.train()

    name_to_index = {name: index for index, name in enumerate(metric_names)}
    per_dataset: dict[str, dict[str, float]] = {}
    for horizon_index in range(1, horizon_count + 1):
        row: dict[str, float] = {}
        count = float(accepted[horizon_index - 1])
        pixels = float(valid_pixels[horizon_index - 1])
        for name, metric_index in name_to_index.items():
            denominator = pixels if name in ("abs_rel", "delta1") else count
            row[name] = (
                float(sums[horizon_index - 1, metric_index]) / denominator
                if denominator > 0 else float("nan")
            )
        row["valid_pixels"] = pixels
        row["accepted_tuples"] = count
        row["attempted_tuples"] = float(attempted[horizon_index - 1])
        per_dataset[f"waymo_future_t+{0.2 * horizon_index:.1f}"] = row

    total_count = float(accepted.sum())
    total_pixels = float(valid_pixels.sum())
    overall: dict[str, float] = {}
    for name, metric_index in name_to_index.items():
        denominator = total_pixels if name in ("abs_rel", "delta1") else total_count
        overall[name] = (
            float(sums[:, metric_index].sum()) / denominator
            if denominator > 0 else float("nan")
        )
    overall.update({
        "valid_pixels": total_pixels,
        "accepted_tuples": total_count,
        "attempted_tuples": float(attempted.sum()),
        "static_windows": float(record_count),
    })
    return (
        overall,
        per_dataset,
        {3: overall},
        {name: {3: metrics} for name, metrics in per_dataset.items()},
    )


def _save_checkpoint(
    path: Path,
    model: GeometrySceneModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: ExperimentConfig,
    scenes: list[object],
    runtime_state: dict[str, int | float] | None = None,
) -> None:
    state = {key: value.cpu() for key, value in model.state_dict().items() if not key.startswith("vggt.")}
    payload = {
            "step": step,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "scenes": scenes,
    }
    if runtime_state is not None:
        payload["runtime_state"] = dict(runtime_state)
    torch.save(payload, path)


def train_ddp(args: argparse.Namespace) -> None:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if args.resume_exact_scheduler and not args.resume:
        raise ValueError("--resume-exact-scheduler requires --resume")
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be at least 1")
    if args.scene_batch_size < 1:
        raise ValueError("--scene-batch-size must be at least 1")
    if args.targets_per_vggt < 1:
        raise ValueError("--targets-per-vggt must be at least 1")
    if args.grad_accumulation % args.scene_batch_size:
        raise ValueError("grad-accumulation must be divisible by scene-batch-size")
    if (
        args.shared_scene_encoding
        and args.grad_accumulation
        % (args.scene_batch_size * args.targets_per_vggt)
    ):
        raise ValueError(
            "Shared scene encoding requires grad-accumulation to be divisible "
            "by scene-batch-size * targets-per-vggt"
        )
    if args.resume_warmup_steps < 0:
        raise ValueError("--resume-warmup-steps must be non-negative")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.final_lr < 0 or args.final_lr > args.lr:
        raise ValueError("--final-lr must satisfy 0 <= final_lr <= lr")
    if not 0.0 <= args.anchor_view_dropout <= 1.0:
        raise ValueError("--anchor-view-dropout must be in [0, 1]")
    if not 0.0 <= args.random_view_dropout <= 1.0:
        raise ValueError("--random-view-dropout must be in [0, 1]")
    if args.anchor_view_dropout + args.random_view_dropout > 1.0:
        raise ValueError(
            "--anchor-view-dropout + --random-view-dropout must not exceed 1"
        )
    if not 4 <= args.prefetch_batches <= 8:
        raise ValueError("--prefetch-batches must be in [4, 8]")
    if args.scene_load_workers < 1:
        raise ValueError("--scene-load-workers must be positive")
    if (args.validation_manifest or args.waymo_validation_prepared_root) and args.validate_every < 1:
        raise ValueError("--validate-every must be positive when validation is enabled")
    if args.validation_manifest and args.waymo_validation_prepared_root:
        raise ValueError(
            "--validation-manifest and --waymo-validation-prepared-root are mutually exclusive"
        )
    data_sources = [bool(args.scene_manifest), bool(args.waymo_prepared_root)]
    if sum(data_sources) != 1:
        raise ValueError(
            "Provide exactly one of --scene-manifest or --waymo-prepared-root"
        )
    if args.waymo_prepared_root and args.targets_per_vggt != 1:
        raise ValueError("Waymo static-future training requires --targets-per-vggt 1")
    if args.waymo_validation_prepared_root and not args.waymo_prepared_root:
        raise ValueError("Waymo validation is only supported with Waymo fine-tuning")
    if args.waymo_prepared_root and not args.waymo_static_temporal_manifest:
        raise ValueError("Waymo adaptation requires --waymo-static-temporal-manifest")
    if args.waymo_static_temporal_manifest and not args.waymo_prepared_root:
        raise ValueError("Static Waymo temporal training requires --waymo-prepared-root")
    if args.waymo_validation_static_temporal_manifest and not args.waymo_validation_prepared_root:
        raise ValueError(
            "Static future validation requires --waymo-validation-prepared-root"
        )
    if bool(args.waymo_static_temporal_manifest) != bool(
        args.waymo_validation_static_temporal_manifest
    ):
        raise ValueError(
            "Use static temporal manifests for both Waymo training and validation"
        )
    if args.waymo_validation_examples < 1:
        raise ValueError("--waymo-validation-examples must be positive")
    if args.waymo_validation_stream_all and not args.waymo_validation_static_temporal_manifest:
        raise ValueError(
            "--waymo-validation-stream-all requires the static validation manifest"
        )
    if args.waymo_context_max_frames < 1:
        raise ValueError("--waymo-context-max-frames must be positive")
    config = ExperimentConfig.load(args.config)
    if args.vggt_checkpoint:
        config.vggt.checkpoint = args.vggt_checkpoint
    config.model.memory_mode = "perceiver"
    config.train.steps = args.steps
    config.train.learning_rate = args.lr
    config.train.grad_accumulation = args.grad_accumulation
    config.train.anchor_view_dropout_probability = args.anchor_view_dropout
    config.train.random_view_dropout_probability = args.random_view_dropout
    if args.alignment_max_relative_baseline_error is not None:
        config.alignment.max_relative_baseline_error = (
            args.alignment_max_relative_baseline_error
        )
    if args.alignment_max_relative_center_error is not None:
        config.alignment.max_relative_center_error = (
            args.alignment_max_relative_center_error
        )
    if args.min_input_views is not None:
        config.data.min_input_views = args.min_input_views
    if args.max_input_views is not None:
        config.data.max_input_views = args.max_input_views
    if (
        config.data.min_input_views < 1
        or config.data.max_input_views < config.data.min_input_views
    ):
        raise ValueError("Input view range must satisfy 1 <= min <= max")
    if args.waymo_prepared_root and not (
        3 <= config.data.min_input_views <= config.data.max_input_views <= 8
    ):
        raise ValueError(
            "Waymo frozen-VGGT input view range must satisfy 3 <= min <= max <= 8 "
            "after YAML/CLI overrides; "
            f"got min={config.data.min_input_views}, max={config.data.max_input_views}. "
            "For paper adaptation, set --min-input-views 3 --max-input-views 5."
        )
    # CLI overrides must be opt-in so YAML values remain authoritative.
    for argument, field in ((args.gradient_weight, "gradient_weight"),):
        if argument is not None:
            setattr(config.train, field, argument)
    config.train.output_dir = args.output_dir
    if config.train.gradient_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if (
        config.alignment.max_relative_baseline_error <= 0
        or config.alignment.max_relative_center_error <= 0
    ):
        raise ValueError("Camera alignment residual thresholds must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    seed_everything(args.seed)
    if args.waymo_prepared_root:
        scene_records = [
            {
                "dataset": "waymo",
                "scene_id": "waymo/prepared_front3_train",
                "path": str(args.waymo_prepared_root),
                "source_c2w_convention": "opencv",
                "cache_key": "waymo_prepared_front3_train",
            }
        ]
    else:
        manifest = json.loads(Path(args.scene_manifest).read_text(encoding="utf-8"))
        scene_records = list(manifest["scenes"])
        scene_records = [
            _remap_dataset_root(
                record, "tartanair", args.tartanair_runtime_root
            )
            for record in scene_records
        ]
        if args.max_scenes > 0:
            scene_records = scene_records[: args.max_scenes]
        scene_records = [
            record
            if isinstance(record, dict)
            else _scene_record(
                "dl3dv", Path(config.data.root) / str(record), "opengl"
            )
            for record in scene_records
        ]
    adapter = _adapter(config, device)
    model = GeometrySceneModel(adapter, config.model, config.alignment).to(device).train()
    resume_payload: dict[str, object] | None = None
    start_step = 0
    if args.init_from:
        init_payload = torch.load(args.init_from, map_location="cpu", weights_only=True)
        source_state = init_payload["model"]
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state
            and target_state[key].shape == value.shape
        }
        expanded_feature_projections = 0
        for key, target_value in target_state.items():
            if (
                key.startswith("tokenizer.feature_projections.")
                and ".1." in key
                and key in source_state
                and source_state[key].ndim == target_value.ndim
                and source_state[key].shape[0] < target_value.shape[0]
                and source_state[key].shape[1:] == target_value.shape[1:]
            ):
                # A wider feature front-end reuses every shape-compatible row;
                # newly added rows retain their normal initialization.
                expanded = target_value.clone()
                expanded[: source_state[key].shape[0]].copy_(source_state[key])
                compatible[key] = expanded
                expanded_feature_projections += 1
        if not compatible:
            raise RuntimeError("--init-from checkpoint has no compatible parameters")
        model.load_state_dict(compatible, strict=False)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "initialized_from": args.init_from,
                        "loaded_compatible_tensors": len(compatible),
                        "loaded_compatible_decoder": True,
                        "expanded_feature_projection_tensors": expanded_feature_projections,
                        "source_step": int(init_payload.get("step", 0)),
                    }
                )
            )
    if args.resume:
        # Memory-map multi-gigabyte checkpoints so every DDP rank does not
        # redundantly stream the complete archive into private CPU buffers.
        resume_payload = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        missing, unexpected = model.load_state_dict(resume_payload["model"], strict=False)
        bad_missing = [key for key in missing if not key.startswith("vggt.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Incompatible resume checkpoint: missing={bad_missing}, unexpected={unexpected}"
            )
        start_step = int(resume_payload.get("step", 0))
        if start_step >= args.steps:
            raise ValueError(f"Resume step {start_step} must be below target step {args.steps}")
    wrapper = DistributedLossModule(model, config).to(device)
    ddp = DistributedDataParallel(wrapper, device_ids=[local_rank], output_device=local_rank)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=config.train.weight_decay)
    remaining_steps = max(1, args.steps - start_step)
    if resume_payload is not None and args.resume_exact_scheduler:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler_state = resume_payload["scheduler"]
        checkpoint_t_max = int(scheduler_state.get("T_max", args.steps))
        if checkpoint_t_max != args.steps:
            raise ValueError(
                "Exact scheduler resume requires the requested final step to match "
                f"the checkpoint scheduler: requested={args.steps}, checkpoint_T_max={checkpoint_t_max}"
            )
        # An exact resume keeps the original global schedule.  ``remaining_steps``
        # would incorrectly compress the cosine tail after a mid-run restart.
        schedule_steps = max(1, args.steps)
        warmup_steps = min(args.warmup_steps, max(0, schedule_steps - 1))
        if warmup_steps:
            final_factor = args.final_lr / args.lr

            def fresh_lr_factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                cosine_step = step - warmup_steps
                cosine_steps = max(1, schedule_steps - warmup_steps)
                cosine = 0.5 * (
                    1.0
                    + math.cos(
                        math.pi * min(cosine_step, cosine_steps) / cosine_steps
                    )
                )
                return final_factor + (1.0 - final_factor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, fresh_lr_factor)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.steps)
            )
        scheduler.load_state_dict(scheduler_state)
        # LambdaLR performs an initialization step in its constructor. Restore
        # the optimizer LR saved at the checkpoint after loading scheduler state;
        # otherwise an exact resume can silently start at the warmup step-0 LR.
        for group, resumed_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = float(resumed_lr)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "resumed_exact_scheduler": True,
                        "scheduler_last_epoch": int(scheduler.last_epoch),
                        "resume_lr": float(optimizer.param_groups[0]["lr"]),
                        "scheduler_t_max": checkpoint_t_max,
                    }
                )
            )
    elif resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        checkpoint_lr = float(optimizer.param_groups[0]["lr"])
        # Raise the resumed LR gradually to the requested value, then use a
        # fresh cosine tail ending at the new target step. Loading the old
        # scheduler would still end at its former 20k T_max.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["initial_lr"] = args.lr
        warmup_steps = min(args.resume_warmup_steps, max(0, remaining_steps - 1))
        start_factor = checkpoint_lr / args.lr

        final_factor = args.final_lr / args.lr

        def resumed_lr_factor(step: int) -> float:
            # With no resumed warmup, keep the checkpoint LR as the starting
            # point and decay it directly.  The previous expression started
            # at 1.0 here, which reset a decayed checkpoint back to args.lr.
            if warmup_steps == 0:
                progress = min(step, remaining_steps) / remaining_steps
                return final_factor + (start_factor - final_factor) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            if warmup_steps > 0 and step <= warmup_steps:
                return start_factor + (1.0 - start_factor) * step / warmup_steps
            cosine_step = step - warmup_steps
            cosine_steps = max(1, remaining_steps - warmup_steps)
            cosine = 0.5 * (
                1.0
                + math.cos(
                    math.pi * min(cosine_step, cosine_steps) / cosine_steps
                )
            )
            return final_factor + (1.0 - final_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, resumed_lr_factor)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "optimizer_state_restored": True,
                        "resume_step": start_step,
                        "checkpoint_lr": checkpoint_lr,
                        "requested_lr": args.lr,
                        "final_lr": args.final_lr,
                        "resume_warmup_steps": warmup_steps,
                    }
                )
            )
    else:
        warmup_steps = min(args.warmup_steps, max(0, remaining_steps - 1))
        final_factor = args.final_lr / args.lr
        if warmup_steps:
            def fresh_lr_factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                cosine_step = step - warmup_steps
                cosine_steps = max(1, remaining_steps - warmup_steps)
                cosine = 0.5 * (
                    1.0
                    + math.cos(
                        math.pi * min(cosine_step, cosine_steps) / cosine_steps
                    )
                )
                return final_factor + (1.0 - final_factor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, fresh_lr_factor)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, remaining_steps, eta_min=args.final_lr
            )
    if args.waymo_prepared_root:
        local_scene_records = scene_records
        dataset = WaymoPreparedDataset(
            args.waymo_prepared_root,
            seed=args.seed + rank * 1_000_003,
            samples_per_segment=args.waymo_samples_per_segment,
            decode_workers=args.scene_load_workers,
            max_context_frame_delta=args.waymo_context_max_frames,
            static_temporal_manifest=args.waymo_static_temporal_manifest,
        )
    else:
        local_scene_records = (
            scene_records[rank::world_size]
            if args.global_scene_sampling
            else shard_scene_records_by_dataset(scene_records, rank, world_size)
        )
        dataset = CachedSceneDataset(
            local_scene_records,
            config.data.cache_root,
            adapter.patch_size,
            config.data.image_resolution,
            config.data.min_input_views,
            config.data.max_input_views,
            samples_per_scene=max(args.steps * world_size * 2, 1024),
            seed=args.seed,
            source_c2w_convention=config.data.source_c2w_convention,
            require_teacher_register_match=True,
            dataset_balanced=not args.global_scene_sampling,
            shuffle_scenes_each_epoch=args.global_scene_sampling,
            scene_shuffle_seed=args.seed + rank * 1_000_003,
            targets_per_input=args.targets_per_vggt,
            preload_caches=args.preload_caches,
            cache_metadata_path=args.cache_metadata_index,
            use_tartanair_gt_depth=args.use_tartanair_gt_depth,
            scene_load_workers=args.scene_load_workers,
        )
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    writer = (
        __import__("torch.utils.tensorboard", fromlist=["SummaryWriter"]).SummaryWriter(
            output_dir / "tensorboard",
            # A checkpoint can lag the final event written before a controlled
            # stop. Hide only those non-restorable orphan events so the new
            # run has one optimizer state for every visible global step.
            purge_step=(start_step + 1 if args.resume else None),
        )
        if rank == 0
        else None
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "training_confidence_supervision": "cached-hard-mask",
                    "fixed_validation_confidence_supervision": "cached-hard-mask",
                    "tartanair_depth_supervision": (
                        "dataset_gt" if args.use_tartanair_gt_depth else "cached_teacher"
                    ),
                    "shared_scene_encoding": bool(args.shared_scene_encoding),
                    "memory_view_dropout": {
                        "anchor_probability": config.train.anchor_view_dropout_probability,
                        "random_nonanchor_probability": (
                            config.train.random_view_dropout_probability
                        ),
                        "sampling": "mutually_exclusive",
                        "validation": "disabled",
                    },
                    "prefetch_batches": args.prefetch_batches,
                    "scene_load_workers": args.scene_load_workers,
                    "tartanair_runtime_root": args.tartanair_runtime_root,
                    "waymo_scale_alignment": (
                        "input_same_pixel_vggt_lidar_once_before_encoding"
                        if args.waymo_prepared_root
                        else "not_applicable"
                    ),
                    "decoder_posthoc_scale_alignment": False,
                    "waymo_static_temporal_manifest": args.waymo_static_temporal_manifest,
                }
            )
        )
    validation_cache: list[tuple[dict[str, object], VGGTOutput]] = []
    validation_stream_dataset: WaymoPreparedDataset | None = None
    checkpoint_runtime = (
        dict(resume_payload.get("runtime_state", {}))
        if resume_payload is not None
        else {}
    )
    best_validation_abs_rel = float(
        checkpoint_runtime.get("best_validation_abs_rel", float("inf"))
    )
    if args.validation_manifest:
        validation_cache = prepare_fixed_validation(
            ddp.module,
            args.validation_manifest,
            config,
            adapter.patch_size,
            device,
            rank,
            world_size,
            use_tartanair_gt_depth=args.use_tartanair_gt_depth,
            view_counts=tuple(
                int(value) for value in args.validation_view_counts.split(",")
            ),
            tartanair_runtime_root=args.tartanair_runtime_root,
        )
        dist.barrier()
    elif args.waymo_validation_prepared_root and args.waymo_validation_stream_all:
        validation_stream_dataset = WaymoPreparedDataset(
            args.waymo_validation_prepared_root,
            seed=args.seed + 91_337,
            samples_per_segment=1,
            decode_workers=args.scene_load_workers,
            static_temporal_manifest=args.waymo_validation_static_temporal_manifest,
        )
        if rank == 0:
            print(json.dumps({
                "event": "full_static_validation_configured",
                "windows": len(validation_stream_dataset._static_records),
                "queries_per_window": 10,
                "bounded_memory": True,
            }), flush=True)
        dist.barrier()
    elif args.waymo_validation_prepared_root:
        validation_cache = prepare_waymo_validation(
            ddp.module,
            args.waymo_validation_prepared_root,
            device,
            rank,
            world_size,
            examples=args.waymo_validation_examples,
            seed=args.seed + 91_337,
            static_temporal_manifest=(
                args.waymo_validation_static_temporal_manifest
            ),
        )
        dist.barrier()

    def validate_and_log(step: int) -> None:
        nonlocal best_validation_abs_rel
        if validation_stream_dataset is not None:
            overall, per_dataset, per_view, per_dataset_view = (
                run_streaming_waymo_static_validation(
                    ddp.module, validation_stream_dataset, device, rank, world_size
                )
            )
        else:
            overall, per_dataset, per_view, per_dataset_view = run_fixed_validation(
                ddp.module, validation_cache, device
            )
        if rank != 0:
            return
        assert writer is not None
        for name, value in overall.items():
            writer.add_scalar(f"validation/{name}", value, step)
        for dataset_name, metrics in per_dataset.items():
            for name, value in metrics.items():
                writer.add_scalar(f"validation_dataset/{dataset_name}/{name}", value, step)
        for view_count, metrics in per_view.items():
            for name, value in metrics.items():
                writer.add_scalar(
                    f"validation_views/V{view_count}/{name}", value, step
                )
        for dataset_name, view_metrics in per_dataset_view.items():
            for view_count, metrics in view_metrics.items():
                for name, value in metrics.items():
                    writer.add_scalar(
                        f"validation_dataset_views/{dataset_name}/V{view_count}/{name}",
                        value,
                        step,
                    )
        writer.flush()
        print(
            json.dumps(
                {
                    "validation_step": step,
                    **overall,
                    "per_dataset": per_dataset,
                    "per_view": per_view,
                    "per_dataset_view": per_dataset_view,
                }
            )
        )
        if math.isfinite(overall["abs_rel"]) and overall["abs_rel"] < best_validation_abs_rel:
            best_validation_abs_rel = overall["abs_rel"]
            _save_checkpoint(
                output_dir / "best_validation.pt",
                model,
                optimizer,
                scheduler,
                step,
                config,
                scene_records,
                runtime_state={
                    "vggt_forwards": vggt_forwards,
                    "scene_cursor": scene_cursor,
                    "update_attempt": update_attempt,
                    "micro_attempt": micro_attempt,
                    "best_validation_abs_rel": best_validation_abs_rel,
                },
            )

    successful = start_step
    update_attempt = start_step
    micro_attempt = start_step * args.grad_accumulation
    vggt_forwards = int(checkpoint_runtime.get("vggt_forwards", 0))
    session_optimizer_updates = 0
    session_micro_attempts = 0
    accepted_samples = 0
    skip_reasons: Counter[str] = Counter()
    skip_reasons_by_dataset: Counter[str] = Counter()
    recent: deque[dict[str, float]] = deque(maxlen=100)
    metric_names = (
        "loss", "log_depth_loss", "gradient_loss",
        "target_dataset_gt", "dynamic_downweighted_fraction",
        "log_l1", "abs_rel", "abs_rel_observed", "abs_rel_unobserved",
        "coverage", "overlap",
        "anchor_view_dropout_applied", "random_view_dropout_applied",
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated_micro = 0
    accumulated_valid = 0
    accumulated_metrics = torch.zeros(len(metric_names), device=device)
    accumulated_dataset_rows: list[dict[str, object]] = []
    accumulated_skips: list[dict[str, str]] = []
    accumulated_io_rows: list[tuple[float, float, int]] = []
    scene_cursor = int(checkpoint_runtime.get("scene_cursor", 0))
    if rank == 0 and args.resume:
        print(
            json.dumps(
                {
                    "resumed_runtime_state": True,
                    "vggt_forwards": vggt_forwards,
                    "scene_cursor": scene_cursor,
                    "source": "checkpoint",
                }
            )
        )

    def input_views_for_forward(forward_index: int) -> int:
        # All ranks use the same view count for a given frozen-VGGT forward.
        # Scene/view identities remain rank-local, while synchronized tensor
        # workload prevents the 1-view rank waiting for a 4-view peer.
        view_rng = random.Random(args.seed * 1009 + forward_index)
        return view_rng.randint(
            config.data.min_input_views,
            config.data.max_input_views,
        )

    if (validation_cache or validation_stream_dataset is not None) and not args.skip_initial_validation:
        validate_and_log(start_step)
    # A single producer owns the cursor and fills a bounded queue.  This keeps
    # sampling deterministic and lossless while overlapping 4--8 CPU batches
    # with GPU work.
    prefetcher = _SceneBatchPrefetcher(
        dataset,
        start_cursor=scene_cursor,
        batch_size=args.scene_batch_size,
        start_forward_index=vggt_forwards,
        input_views_for_forward=input_views_for_forward,
        capacity=args.prefetch_batches,
    )
    optimizer_step_started = time.perf_counter()
    measured_training_seconds = 0.0
    measured_training_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while successful < args.steps:
        (
            batches,
            scene_cursor,
            prefetched_forward_index,
            producer_load_seconds,
            consumer_wait_seconds,
            queue_depth,
        ) = prefetcher.get()
        if prefetched_forward_index != vggt_forwards:
            raise RuntimeError(
                "Prefetch sequence changed: "
                f"expected forward {vggt_forwards}, got {prefetched_forward_index}"
            )
        accumulated_io_rows.append(
            (producer_load_seconds, consumer_wait_seconds, queue_depth)
        )
        vggt_output = ddp.module.extract_vggt_batch(batches)
        vggt_forwards += 1
        target_groups = (
            [tuple(range(args.targets_per_vggt))]
            if args.shared_scene_encoding
            else [(target_slot,) for target_slot in range(args.targets_per_vggt)]
        )
        for target_slots in target_groups:
            if successful >= args.steps:
                break
            group_samples = args.scene_batch_size * len(target_slots)
            next_accumulated = accumulated_micro + group_samples
            if next_accumulated > args.grad_accumulation:
                raise RuntimeError("Compute batch crosses a gradient-accumulation boundary")
            should_sync = next_accumulated == args.grad_accumulation
            sync_context = nullcontext() if should_sync else ddp.no_sync()
            with sync_context:
                loss_sum, batch_metrics = ddp(
                    batches,
                    vggt_output,
                    target_slots if args.shared_scene_encoding else target_slots[0],
                )
                (loss_sum * world_size).backward()
            rows = batch_metrics["samples"]
            local_valid_count = int(batch_metrics["valid_count"])
            session_micro_attempts += group_samples
            micro_attempt += group_samples
            accumulated_micro = next_accumulated
            accumulated_valid += local_valid_count
            for metrics in rows:
                if not bool(metrics["skipped"]):
                    accumulated_metrics += torch.tensor(
                        [float(metrics[name]) for name in metric_names],
                        device=device,
                    )
                    metric_row = {
                        "dataset": str(metrics["dataset_name"]),
                        "num_inputs": int(metrics["num_inputs"]),
                        **{name: float(metrics[name]) for name in metric_names},
                        "abs_rel": float(metrics["abs_rel"]),
                        "abs_rel_observed": float(metrics["abs_rel_observed"]),
                        "abs_rel_unobserved": float(metrics["abs_rel_unobserved"]),
                        "coverage": float(metrics["coverage"]),
                        "overlap": float(metrics["overlap"]),
                        "delta1": float(metrics.get("delta1", float("nan"))),
                        "valid_pixels": int(metrics.get("valid_pixels", 0)),
                        "scene_token_views": int(metrics.get("scene_token_views", metrics["num_inputs"])),
                        "reconstruction_task": str(metrics.get("reconstruction_task", "standard")),
                        "student_vggt_timestamp_count": int(
                            metrics.get("student_vggt_timestamp_count", 1)
                        ),
                        "target_horizon_seconds": float(
                            metrics.get("target_horizon_seconds", float("nan"))
                        ),
                    }
                    accumulated_dataset_rows.append(metric_row)
                    continue
                accumulated_skips.append({
                    "dataset": str(metrics["dataset_name"]),
                    "reason": str(metrics.get("reason", "unknown")),
                })

            if accumulated_micro < args.grad_accumulation:
                continue

            update_attempt += 1
            valid_count_tensor = torch.tensor(
                accumulated_valid, device=device, dtype=torch.int32
            )
            dist.all_reduce(valid_count_tensor, op=dist.ReduceOp.SUM)
            accumulated_valid = int(valid_count_tensor.item())
            any_valid = accumulated_valid > 0
            if any_valid:
                dist.all_reduce(accumulated_metrics, op=dist.ReduceOp.SUM)
            accepted_samples += accumulated_valid
            if accumulated_valid < world_size * args.grad_accumulation:
                skipped_by_rank: list[list[dict[str, str]]] = [
                    [] for _ in range(world_size)
                ]
                dist.all_gather_object(skipped_by_rank, accumulated_skips)
                if rank == 0:
                    for rank_skips in skipped_by_rank:
                        for item in rank_skips:
                            skip_reasons[item["reason"]] += 1
                            skip_reasons_by_dataset[f"{item['dataset']}:{item['reason']}"] += 1
            if any_valid:
                # DDP averaged across ranks; the backward scale converted it to
                # a sum over valid samples, so divide by their exact count.
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_valid)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer_step_seconds = time.perf_counter() - optimizer_step_started
                measured_training_seconds += optimizer_step_seconds
                measured_training_samples += accumulated_valid
                successful += 1
                session_optimizer_updates += 1
                averaged = accumulated_metrics / accumulated_valid
                recent.append({
                    **dict(zip(metric_names, averaged.cpu().tolist())),
                    "valid_ranks": float(accumulated_valid) / args.grad_accumulation,
                    "effective_batch": float(accumulated_valid),
                })
                if writer is not None:
                    for name, value in recent[-1].items():
                        writer.add_scalar(f"train/{name}", value, successful)
                    if accumulated_io_rows:
                        writer.add_scalar(
                            "system/rank0_batch_load_ms",
                            1000.0
                            * float(np.mean([row[0] for row in accumulated_io_rows])),
                            successful,
                        )
                        writer.add_scalar(
                            "system/rank0_data_wait_ms",
                            1000.0
                            * float(np.mean([row[1] for row in accumulated_io_rows])),
                            successful,
                        )
                        writer.add_scalar(
                            "system/rank0_prefetch_queue_depth",
                            float(np.mean([row[2] for row in accumulated_io_rows])),
                            successful,
                        )
                    writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], successful)
                    writer.add_scalar(
                        "system/optimizer_step_seconds",
                        optimizer_step_seconds,
                        successful,
                    )
                    writer.add_scalar(
                        "system/effective_samples_per_second",
                        accumulated_valid / max(optimizer_step_seconds, 1e-9),
                        successful,
                    )
                    if device.type == "cuda":
                        writer.add_scalar(
                            "system/peak_gpu_memory_allocated_gib",
                            torch.cuda.max_memory_allocated(device) / (1024**3),
                            successful,
                        )
                        writer.add_scalar(
                            "system/peak_gpu_memory_reserved_gib",
                            torch.cuda.max_memory_reserved(device) / (1024**3),
                            successful,
                        )
                    writer.add_scalar("train/attempt", update_attempt, successful)
                    writer.add_scalar("train/vggt_forwards", vggt_forwards, successful)
                    writer.add_scalar("train/targets_per_vggt", args.targets_per_vggt, successful)
                    writer.add_scalar("train/grad_accumulation", args.grad_accumulation, successful)
                    local_accept_rate = accepted_samples / (session_micro_attempts * world_size)
                    writer.add_scalar("train/local_accept_rate", local_accept_rate, successful)
                    writer.add_scalar("train/local_skip_rate", 1.0 - local_accept_rate, successful)
                    writer.add_scalar("train/skip_rate", 1.0 - local_accept_rate, successful)
                    writer.add_scalar(
                        "train/optimizer_step_rate",
                        session_optimizer_updates / max(1, update_attempt - start_step),
                        successful,
                    )

            if any_valid and successful % args.log_every == 0:
                per_rank: list[list[dict[str, object]]] = [[] for _ in range(world_size)]
                dist.all_gather_object(per_rank, accumulated_dataset_rows)
                if writer is not None:
                    grouped: dict[str, list[dict[str, object]]] = {}
                    for rank_rows in per_rank:
                        for row in rank_rows:
                            grouped.setdefault(str(row["dataset"]), []).append(row)
                    for dataset_name, rows in grouped.items():
                        for name in (
                            "loss", "log_depth_loss", "gradient_loss",
                            "target_dataset_gt", "dynamic_downweighted_fraction", "abs_rel",
                            "abs_rel_observed", "abs_rel_unobserved", "coverage", "overlap",
                            "anchor_view_dropout_applied",
                            "random_view_dropout_applied",
                        ):
                            writer.add_scalar(
                                f"dataset/{dataset_name}/{name}",
                                float(np.mean([float(row[name]) for row in rows])), successful,
                            )
                    grouped_views: dict[int, list[dict[str, object]]] = {}
                    for rank_rows in per_rank:
                        for row in rank_rows:
                            grouped_views.setdefault(int(row["num_inputs"]), []).append(row)
                    for view_count, rows in grouped_views.items():
                        for name in metric_names:
                            writer.add_scalar(
                                f"train_views/V{view_count}/{name}",
                                float(np.mean([float(row[name]) for row in rows])),
                                successful,
                            )
                        writer.add_scalar(
                            f"train_views/V{view_count}/sample_count",
                            len(rows),
                            successful,
                        )
                    writer.flush()
            if rank == 0 and any_valid and successful % args.log_every == 0:
                tail_rows = list(recent)[-min(len(recent), 50):]
                summary = {
                    key: float(np.median([row[key] for row in tail_rows]))
                    for key in tail_rows[0]
                }
                print(json.dumps({
                    "step": successful, "attempt": update_attempt,
                    "vggt_forwards": vggt_forwards, "lr": scheduler.get_last_lr()[0],
                    **summary,
                }))
            if rank == 0 and any_valid and successful % args.checkpoint_every == 0:
                _save_checkpoint(
                    output_dir / f"step_{successful:06d}.pt", model, optimizer,
                    scheduler, successful, config, scene_records,
                    runtime_state={
                        "vggt_forwards": vggt_forwards,
                        "scene_cursor": scene_cursor,
                        "update_attempt": update_attempt,
                        "micro_attempt": micro_attempt,
                        "best_validation_abs_rel": best_validation_abs_rel,
                    },
                )
            if any_valid and validation_cache and successful % args.validate_every == 0:
                validate_and_log(successful)
            optimizer.zero_grad(set_to_none=True)
            accumulated_micro = 0
            accumulated_valid = 0
            accumulated_metrics.zero_()
            accumulated_dataset_rows = []
            accumulated_skips = []
            accumulated_io_rows = []
            optimizer_step_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if update_attempt - start_step > (args.steps - start_step) * 5:
                raise RuntimeError("Too many all-rank-invalid DDP attempts")
    prefetcher.close()
    close_dataset = getattr(dataset, "close", None)
    if close_dataset is not None:
        close_dataset()
    if rank == 0 and not args.no_final_checkpoint:
        _save_checkpoint(
            output_dir / "final.pt", model, optimizer, scheduler, successful,
            config, scene_records,
            runtime_state={
                "vggt_forwards": vggt_forwards,
                "scene_cursor": scene_cursor,
                "update_attempt": update_attempt,
                "micro_attempt": micro_attempt,
                "best_validation_abs_rel": best_validation_abs_rel,
            },
        )
    if rank == 0:
        result = {
            "steps": successful,
            "attempts": update_attempt,
            "micro_attempts": micro_attempt,
            "vggt_forwards": vggt_forwards,
            "targets_per_vggt": args.targets_per_vggt,
            "shared_scene_encoding": bool(args.shared_scene_encoding),
            "scene_batch_size": args.scene_batch_size,
            "min_input_views": config.data.min_input_views,
            "max_input_views": config.data.max_input_views,
            "optimizer_updates_per_vggt": session_optimizer_updates / max(1, vggt_forwards),
            "world_size": world_size,
            "grad_accumulation": args.grad_accumulation,
            "nominal_global_batch": world_size * args.grad_accumulation,
            "scene_count": len(scene_records),
            "dataset_counts": dict(Counter(record["dataset"] for record in scene_records)),
            "teacher_student_register_matched": True,
            "dataset_balanced_sampling": not args.global_scene_sampling,
            "rank_sharded_cache_preload": bool(args.preload_caches),
            "validation_manifest": args.validation_manifest,
            "validate_every": args.validate_every,
            "training_confidence_supervision": "cached-hard-mask",
            "best_validation_abs_rel": best_validation_abs_rel,
            "accepted_samples": accepted_samples,
            "attempted_samples": session_micro_attempts * world_size,
            "local_accept_rate": accepted_samples / max(1, session_micro_attempts * world_size),
            "optimizer_step_rate": session_optimizer_updates / max(1, update_attempt - start_step),
            "measured_training_seconds": measured_training_seconds,
            "measured_effective_samples_per_second": (
                measured_training_samples / max(measured_training_seconds, 1e-9)
            ),
            "skip_reasons": dict(skip_reasons),
            "skip_reasons_by_dataset": dict(skip_reasons_by_dataset),
            "tail": {
                key: float(np.median([row[key] for row in recent]))
                for key in recent[0].keys()
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("LARGE_SCALE_RESULT=" + json.dumps(result))
        assert writer is not None
        writer.flush()
        writer.close()
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache-shard")
    cache.add_argument("--config", default="configs/geostate.yaml")
    cache.add_argument("--rank", type=int, required=True)
    cache.add_argument("--world-size", type=int, default=4)
    cache.add_argument("--max-candidates", type=int, default=0, help="0 means every scene in the candidate manifest")
    cache.add_argument("--datasets", default="dl3dv,co3d,re10k,tartanair,wildrgbd")
    cache.add_argument("--candidate-manifest", required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--output", required=True)
    cache.set_defaults(function=cache_shard)

    aggregate = commands.add_parser("aggregate-cache")
    aggregate.add_argument("--inputs", required=True)
    aggregate.add_argument("--scenes", type=int, default=0, help="0 means every valid cache")
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(function=aggregate_cache)

    filter_manifest = commands.add_parser("filter-cache-manifest")
    filter_manifest.add_argument("--scene-manifest", required=True)
    filter_manifest.add_argument("--cache-root", default="data/cache/teacher")
    filter_manifest.add_argument("--min-input-views", type=int, default=3)
    filter_manifest.add_argument("--targets-per-input", type=int, default=1)
    filter_manifest.add_argument("--output", required=True)
    filter_manifest.set_defaults(function=filter_cache_manifest)

    split = commands.add_parser("split-validation")
    split.add_argument("--scene-manifest", required=True)
    split.add_argument("--cache-root", default="data/cache/teacher")
    split.add_argument("--count", type=int, default=64)
    split.add_argument("--input-views", type=int, default=4)
    split.add_argument("--seed", type=int, default=20260720)
    split.add_argument("--train-output", required=True)
    split.add_argument("--validation-output", required=True)
    split.set_defaults(function=create_validation_split)

    train = commands.add_parser("train-ddp")
    train.add_argument("--config", default="configs/geostate.yaml")
    train.add_argument(
        "--vggt-checkpoint",
        help="Optional frozen VGGT checkpoint override (for example, an NVMe replica).",
    )
    train.add_argument("--scene-manifest")
    train.add_argument(
        "--waymo-prepared-root",
        help=(
            "Use prepared Waymo front-three-camera segments. Frozen VGGT may see "
            "multiple timestamps while scene-token memory is restricted to one timestamp."
        ),
    )
    train.add_argument(
        "--waymo-samples-per-segment",
        type=int,
        default=128,
        help="Number of sampled tuples to amortize each prepared Waymo segment load",
    )
    train.add_argument(
        "--waymo-static-temporal-manifest",
        help=(
            "Static temporal-window JSONL. Scene tokens read the source-time "
            "front three cameras and predict a hidden future FRONT camera."
        ),
    )
    train.add_argument(
        "--waymo-context-max-frames",
        type=int,
        default=10,
        help=(
            "Maximum frame-index distance for hidden multi-time VGGT context; "
            "Waymo is about 10 Hz, so 10 keeps context within roughly one second"
        ),
    )
    train.add_argument(
        "--waymo-validation-prepared-root",
        help="Prepared Waymo validation segments for fixed static-future evaluation",
    )
    train.add_argument(
        "--waymo-validation-static-temporal-manifest",
        help=(
            "Static checkpoint102 validation-window JSONL for t+0.2 through t+2.0 FRONT queries"
        ),
    )
    train.add_argument(
        "--waymo-validation-examples",
        type=int,
        default=30,
        help="Number of fixed static Waymo anchors, each evaluated at ten future horizons",
    )
    train.add_argument(
        "--waymo-validation-stream-all",
        action="store_true",
        help=(
            "Evaluate every static validation window and all ten horizons with "
            "bounded memory instead of caching frozen VGGT outputs"
        ),
    )
    train.add_argument(
        "--skip-initial-validation",
        action="store_true",
        help="Start optimizer updates immediately; run full validation at the first interval",
    )
    train.add_argument("--max-scenes", type=int, default=0, help="0 means every manifest scene")
    train.add_argument(
        "--global-scene-sampling",
        action="store_true",
        help="Sample every scene uniformly through a deterministic per-epoch random permutation",
    )
    train.add_argument("--steps", type=int, default=200000)
    train.add_argument("--lr", type=float, default=1e-5)
    train.add_argument(
        "--final-lr",
        type=float,
        default=0.0,
        help="Cosine-decay floor.",
    )
    train.add_argument(
        "--alignment-max-relative-baseline-error",
        type=float,
        help="Override the camera Sim(3) median relative-baseline sanity threshold",
    )
    train.add_argument(
        "--alignment-max-relative-center-error",
        type=float,
        help="Override the camera Sim(3) median center-residual sanity threshold",
    )
    train.add_argument("--gradient-weight", type=float)
    train.add_argument(
        "--use-tartanair-gt-depth",
        action="store_true",
        help="Use materialized TartanAir z-depth and its metric scale when available",
    )
    train.add_argument(
        "--anchor-view-dropout",
        type=float,
        default=0.4,
        help=(
            "For V>=2, mutually exclusively hide the reference/anchor patch "
            "memory with this probability; disabled in validation"
        ),
    )
    train.add_argument(
        "--random-view-dropout",
        type=float,
        default=0.4,
        help=(
            "For V>=2, mutually exclusively hide one uniformly sampled "
            "non-anchor patch-memory view with this probability"
        ),
    )
    train.add_argument(
        "--grad-accumulation",
        type=int,
        default=9,
        help="Per-rank accumulation; the four-GPU paper run uses 9 for global batch 36.",
    )
    train.add_argument(
        "--scene-batch-size",
        type=int,
        default=1,
        help="Shared batch size for frozen VGGT, tokenizer, decoder, geometry and losses",
    )
    train.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="Bounded deterministic CPU batch queue depth (4--8)",
    )
    train.add_argument(
        "--scene-load-workers",
        type=int,
        default=4,
        help="Permanent per-rank scene-loading worker count",
    )
    train.add_argument(
        "--tartanair-runtime-root",
        help=(
            "Remap TartanAir scene paths from the manifest to this runtime "
            "root without modifying the manifest"
        ),
    )
    train.add_argument("--min-input-views", type=int)
    train.add_argument("--max-input-views", type=int)
    train.add_argument(
        "--targets-per-vggt", type=int, default=1,
        help="Held-out targets reused from each frozen VGGT forward; accumulation may span forward groups",
    )
    train.add_argument(
        "--shared-scene-encoding",
        action="store_true",
        help=(
            "Encode each frozen-VGGT scene batch once and share the trainable "
            "scene encoding across all held-out targets"
        ),
    )
    train.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Fresh-run linear warmup steps before cosine decay",
    )
    train.add_argument(
        "--preload-caches", action="store_true",
        help="Preload only this DDP rank's disjoint teacher-cache shard into host RAM",
    )
    train.add_argument(
        "--cache-metadata-index",
        help="JSON cache_key->image_hw index used for metadata-only shape bucketing",
    )
    train.add_argument("--validation-manifest")
    train.add_argument("--validate-every", type=int, default=500)
    train.add_argument(
        "--validation-view-counts",
        default="1,2,3,4",
        help="Comma-separated prefix view counts for fixed validation",
    )
    train.add_argument("--resume")
    train.add_argument(
        "--resume-exact-scheduler",
        action="store_true",
        help="Restore the checkpoint scheduler state exactly; requires the same final step",
    )
    train.add_argument(
        "--init-from",
        help=(
            "Initialize all shape-compatible tokenizer and decoder tensors from a "
            "pretraining checkpoint and start a fresh optimizer and step counter."
        ),
    )
    train.add_argument("--resume-warmup-steps", type=int, default=250)
    train.add_argument("--seed", type=int, default=20260718)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--checkpoint-every", type=int, default=500)
    train.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="Skip the final checkpoint (useful for short smoke tests)",
    )
    train.add_argument("--output-dir", required=True)
    train.set_defaults(
        function=train_ddp,
    )
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
