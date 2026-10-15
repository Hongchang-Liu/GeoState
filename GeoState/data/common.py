from __future__ import annotations

import json
import math
import random
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRecord:
    """Source paths and camera metadata for one RGB frame."""
    frame_id: str
    image_path: Path
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    original_hw: tuple[int, int]
    depth_path: Path | None = None
    depth_convention: str = "z"


@dataclass
class PreparedFrame:
    """Preprocessed image and camera tensors for one frame."""
    frame_id: str
    image: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    pixel_transform: torch.Tensor
    image_hw: tuple[int, int]


class NerfScene:
    """Reader for a scene represented by transforms metadata and image files."""

    def __init__(
        self,
        scene_path: str | Path,
        source_c2w_convention: str = "opengl",
        *,
        dataset_name: str = "default",
        scene_id: str | None = None,
        cache_key: str | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        self.path = Path(scene_path)
        self.dataset_name = dataset_name
        self.scene_id = scene_id or self.path.name
        self.cache_key = cache_key or self.path.name
        self.image_root = Path(image_root) if image_root is not None else None

        with (self.path / "transforms.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        model = str(data.get("camera_model", "OPENCV")).upper()
        if model not in {"OPENCV", "PINHOLE"}:
            raise ValueError(f"Unsupported camera model {model}; images must be undistorted first")

        self.frames: list[FrameRecord] = []
        image_sizes: dict[Path, tuple[int, int]] = {}
        for index, frame in enumerate(data.get("frames", [])):
            if "transform_matrix" not in frame:
                continue

            relative = Path(frame["file_path"])
            candidates = [
                self.path / relative,
                *(([self.image_root / relative.name]) if self.image_root is not None else []),
                self.path / "images_8" / relative.name,
                self.path / "images_4" / relative.name,
                self.path / "images" / relative.name,
            ]
            image_path = next((candidate for candidate in candidates if candidate.is_file()), None)

            if image_path is None:
                continue

            declared_width = int(frame.get("w", data.get("w", 0)))
            declared_height = int(frame.get("h", data.get("h", 0)))

            is_materialized_pyramid = image_path.parent.name in {"images_8", "images_4"}
            needs_image_size = declared_width <= 0 or declared_height <= 0 or is_materialized_pyramid
            if needs_image_size and image_path.parent not in image_sizes:
                with Image.open(image_path) as image:
                    image_sizes[image_path.parent] = image.size
            if needs_image_size:
                width, height = image_sizes[image_path.parent]
            else:
                width, height = declared_width, declared_height
            calibration_width = declared_width if declared_width > 0 else width
            calibration_height = declared_height if declared_height > 0 else height

            fx = float(frame.get("fl_x", data.get("fl_x", 0)))
            fy = float(frame.get("fl_y", data.get("fl_y", fx)))
            cx = float(frame.get("cx", data.get("cx", calibration_width / 2)))
            cy = float(frame.get("cy", data.get("cy", calibration_height / 2)))
            if fx <= 0 or fy <= 0:
                raise ValueError(f"Missing focal length for {image_path}")

            fx *= width / calibration_width
            fy *= height / calibration_height
            cx *= width / calibration_width
            cy *= height / calibration_height
            intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            if source_c2w_convention == "opengl":

                camera_gl_to_cv = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
                c2w = c2w @ camera_gl_to_cv
            elif source_c2w_convention == "opencv_w2c":

                c2w = torch.linalg.inv(c2w)
            elif source_c2w_convention != "opencv":
                raise ValueError(f"Unknown source c2w convention {source_c2w_convention}")

            depth_path = None
            if frame.get("depth_file_path"):
                candidate = self.path / str(frame["depth_file_path"])
                if candidate.is_file():
                    depth_path = candidate
            frame_id = str(frame.get("frame_id", frame.get("colmap_im_id", index)))
            self.frames.append(
                FrameRecord(
                    frame_id,
                    image_path,
                    c2w,
                    intrinsics,
                    (height, width),
                    depth_path,
                    str(frame.get("depth_convention", "z")).lower(),
                )
            )

        if not self.frames:
            raise ValueError(f"No calibrated frames in {self.path}")


def _crop_supported(image: Image.Image) -> tuple[Image.Image, int, int]:

    width, height = image.size
    aspect = height / width
    if aspect < 0.5:
        crop_width = max(1, round(height / 0.5))
        left = (width - crop_width) // 2
        return image.crop((left, 0, left + crop_width, height)), left, 0
    if aspect > 2.0:
        crop_height = max(1, round(width * 2.0))
        top = (height - crop_height) // 2
        return image.crop((0, top, width, top + crop_height)), 0, top
    return image, 0, 0


def _target_shape(height: int, width: int, resolution: int, patch_size: int) -> tuple[int, int]:
    """Choose a balanced image shape with an approximately fixed patch count."""
    aspect = height / width
    token_count = (resolution // patch_size) ** 2
    width_patches = max(1, round(math.sqrt(token_count / aspect)))
    height_patches = max(1, round(token_count / width_patches))
    return height_patches * patch_size, width_patches * patch_size


def preprocess_frame(
    record: FrameRecord,
    resolution: int,
    patch_size: int,
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> PreparedFrame:
    """Preprocess frame."""
    loader = image_loader or (lambda path: Image.open(path).convert("RGB"))
    loaded = loader(record.image_path)
    image = loaded.convert("RGB")

    cropped, left, top = _crop_supported(image)
    cropped_width, cropped_height = cropped.size

    target_height, target_width = _target_shape(cropped_height, cropped_width, resolution, patch_size)
    resized = cropped.resize((target_width, target_height), Image.Resampling.BICUBIC)

    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

    # T = [[sx,  0,  -left*sx],
    #      [0,   sy, -top*sy],
    #      [0,   0,   1       ]]
    scale_x, scale_y = target_width / cropped_width, target_height / cropped_height
    pixel_transform = torch.tensor(
        [[scale_x, 0, -left * scale_x], [0, scale_y, -top * scale_y], [0, 0, 1]], dtype=torch.float32
    )

    intrinsics = pixel_transform @ record.intrinsics

    if loaded is not image:
        loaded.close()
    return PreparedFrame(record.frame_id, tensor, record.c2w.clone(), intrinsics, pixel_transform, (target_height, target_width))


def preprocess_z_depth(
    record: FrameRecord,
    target_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the RGB crop/resize geometry to a dataset z-depth map."""
    if record.depth_path is None:
        raise FileNotFoundError(f"Frame {record.frame_id} has no dataset depth")
    if record.depth_convention != "z":
        raise ValueError(
            f"Frame {record.frame_id} depth convention is {record.depth_convention}, expected z"
        )
    depth = np.asarray(np.load(record.depth_path, mmap_mode="r"), dtype=np.float32)
    if depth.shape != record.original_hw:
        raise ValueError(
            f"Depth/RGB shape mismatch for {record.frame_id}: "
            f"depth={depth.shape}, rgb={record.original_hw}"
        )
    height, width = record.original_hw
    aspect = height / width
    left = top = 0
    crop_height, crop_width = height, width
    if aspect < 0.5:
        crop_width = max(1, round(height / 0.5))
        left = (width - crop_width) // 2
    elif aspect > 2.0:
        crop_height = max(1, round(width * 2.0))
        top = (height - crop_height) // 2
    depth = depth[top : top + crop_height, left : left + crop_width].copy()
    # TartanAir encodes missing/far-plane depth with very large finite values.
    # Exclude those sentinels before resize, scale alignment, and supervision.
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1000.0)
    depth_tensor = torch.from_numpy(np.where(valid, depth, 0.0))[None, None]
    valid_tensor = torch.from_numpy(valid.astype(np.float32))[None, None]
    # Nearest-exact preserves discontinuities and never invents depth across
    # invalid holes. TartanAir GT is dense, but this also supports sparse GT.
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor, size=target_hw, mode="nearest-exact"
    )[0, 0]
    valid_tensor = torch.nn.functional.interpolate(
        valid_tensor, size=target_hw, mode="nearest-exact"
    )[0, 0] > 0.5
    return depth_tensor, valid_tensor


def uniform_frame_indices(frame_count: int, requested: int) -> list[int]:
    """Uniform frame indices."""
    if frame_count < requested:
        raise ValueError(f"Scene has {frame_count} frames, fewer than requested {requested}")
    return np.linspace(0, frame_count - 1, requested).round().astype(np.int64).tolist()


def overlap_clip_frame_indices(frame_count: int, requested: int) -> list[int]:
    """Overlap clip frame indices."""
    if frame_count < requested:
        raise ValueError(f"Scene has {frame_count} frames, fewer than requested {requested}")
    start = max(0, (frame_count - requested) // 2)
    return list(range(start, start + requested))


class CachedSceneDataset(Dataset[dict[str, object]]):
    """Sample input and target views from scenes with cached VGGT supervision."""

    def __init__(
        self,
        scene_paths: list[str | Path | dict[str, object]],
        cache_root: str | Path,
        patch_size: int,
        image_resolution: int = 512,
        min_views: int = 3,
        max_views: int = 8,
        samples_per_scene: int = 1024,
        seed: int = 42,
        image_loader: Callable[[Path], Image.Image] | None = None,
        source_c2w_convention: str = "opengl",
        require_teacher_register_match: bool = False,
        dataset_balanced: bool = False,
        shuffle_scenes_each_epoch: bool = False,
        scene_shuffle_seed: int | None = None,
        targets_per_input: int = 1,
        preload_caches: bool = False,
        cache_metadata_path: str | Path | None = None,
        use_tartanair_gt_depth: bool = False,
        scene_load_workers: int = 4,
    ) -> None:

        self.scenes = []
        for item in scene_paths:
            if isinstance(item, dict):

                self.scenes.append(
                    NerfScene(
                        str(item["path"]),
                        str(item.get("source_c2w_convention", source_c2w_convention)),
                        dataset_name=str(item.get("dataset", "default")),
                        scene_id=str(item.get("scene_id", Path(str(item["path"])).name)),
                        cache_key=str(item.get("cache_key", Path(str(item["path"])).name)),
                        image_root=str(item["image_root"]) if item.get("image_root") else None,
                    )
                )
            else:
                self.scenes.append(NerfScene(item, source_c2w_convention))

        if not self.scenes:
            raise ValueError("CachedSceneDataset requires at least one scene")

        self.cache_root = Path(cache_root)
        self.patch_size = patch_size
        self.image_resolution = image_resolution
        self.min_views = min_views
        self.max_views = max_views
        self.samples_per_scene = samples_per_scene
        self.seed = seed
        self.image_loader = image_loader
        self.use_tartanair_gt_depth = use_tartanair_gt_depth
        if scene_load_workers < 1:
            raise ValueError("scene_load_workers must be positive")
        self.scene_load_workers = scene_load_workers
        # One pool lives for the lifetime of the dataset.  Recreating this
        # pool for every scene batch caused periodic thread startup/teardown
        # stalls and prevented the outer prefetcher from building a queue.
        self._scene_load_pool = (
            ThreadPoolExecutor(
                max_workers=scene_load_workers,
                thread_name_prefix="scene-load",
            )
            if scene_load_workers > 1
            else None
        )
        self.require_teacher_register_match = require_teacher_register_match
        self.dataset_balanced = dataset_balanced
        self.shuffle_scenes_each_epoch = shuffle_scenes_each_epoch
        self.scene_shuffle_seed = seed if scene_shuffle_seed is None else scene_shuffle_seed
        self._scene_permutations: dict[int, list[int]] = {}
        if targets_per_input < 1:
            raise ValueError("targets_per_input must be positive")
        self.targets_per_input = targets_per_input
        self._teacher_image_hw_by_key: dict[str, tuple[int, int]] = {}
        self._input_original_hw_by_key: dict[str, tuple[int, int]] = {}
        if cache_metadata_path is not None:
            metadata_path = Path(cache_metadata_path)
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = payload.get("image_hw", payload)
            self._teacher_image_hw_by_key = {
                str(key): tuple(int(value) for value in image_hw)
                for key, image_hw in records.items()
            }
            self._input_original_hw_by_key = {
                str(key): tuple(int(value) for value in image_hw)
                for key, image_hw in payload.get("input_original_hw", {}).items()
            }

        self._scene_indices_by_dataset: dict[str, list[int]] = {}
        for scene_index, scene in enumerate(self.scenes):
            self._scene_indices_by_dataset.setdefault(scene.dataset_name, []).append(scene_index)
        self._dataset_names = sorted(self._scene_indices_by_dataset)

        self._cache_memory: dict[str, dict[str, object]] = {}
        if preload_caches:
            self._preload_caches()

        # Equal-shaped scene batches use metadata-only queues.  The queues hold
        # sampling indices, not loaded tensors, so incompatible candidates are
        # delayed rather than discarded or repeatedly read from disk.
        self._compatible_scene_buckets: dict[
            tuple[tuple[int, int], tuple[int, int]], deque[tuple[int, int]]
        ] = {}
        self._compatible_scene_cursor = 0
        self._compatible_batch_size: int | None = None
        self._compatible_shape_scene_counts: Counter[
            tuple[tuple[int, int], tuple[int, int]]
        ] | None = None
        if cache_metadata_path is not None:
            missing = [
                scene.cache_key
                for scene in self.scenes
                if (
                    scene.cache_key not in self._teacher_image_hw_by_key
                    or scene.cache_key not in self._input_original_hw_by_key
                )
            ]
            if missing:
                raise ValueError(
                    f"Cache metadata index misses {len(missing)} local scenes; "
                    f"first={missing[0]}"
                )
            self._compatible_shape_scene_counts = Counter(
                self._scene_metadata_batch_key(scene_index)
                for scene_index in range(len(self.scenes))
            )
            undersized = {
                key: count
                for key, count in self._compatible_shape_scene_counts.items()
                if count < 9
            }
            print(
                f"[CachedSceneDataset] metadata shape groups="
                f"{len(self._compatible_shape_scene_counts)}, "
                f"groups_below_B9={undersized}",
                flush=True,
            )

    def _preload_caches(self) -> None:

        tensor_bytes = 0
        for scene in self.scenes:
            if scene.cache_key in self._cache_memory:
                continue
            cache = torch.load(self._cache_path(scene), map_location="cpu", weights_only=True)
            self._cache_memory[scene.cache_key] = cache
            tensor_bytes += sum(
                value.numel() * value.element_size()
                for value in cache.values()
                if isinstance(value, torch.Tensor)
            )
        print(
            f"[CachedSceneDataset] preloaded {len(self._cache_memory)} rank-local caches "
            f"({tensor_bytes / 2**30:.2f} GiB)",
            flush=True,
        )

    def _get_cache(self, scene: NerfScene) -> dict[str, object]:

        return self._cache_memory.get(scene.cache_key) or torch.load(
            self._cache_path(scene),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

    def __len__(self) -> int:

        return len(self.scenes) * self.samples_per_scene

    def _cache_path(self, scene: NerfScene) -> Path:

        return self.cache_root / f"{scene.cache_key}.pt"

    def _scene_index(self, index: int) -> int:

        if not self.dataset_balanced:
            if self.shuffle_scenes_each_epoch:
                epoch, offset = divmod(index, len(self.scenes))
                permutation = self._scene_permutations.get(epoch)
                if permutation is None:
                    permutation = list(range(len(self.scenes)))
                    random.Random(self.scene_shuffle_seed + epoch).shuffle(permutation)
                    self._scene_permutations[epoch] = permutation
                return permutation[offset]
            return index % len(self.scenes)
        dataset_slot = index % len(self._dataset_names)
        cycle = index // len(self._dataset_names)
        candidates = self._scene_indices_by_dataset[self._dataset_names[dataset_slot]]
        return candidates[cycle % len(candidates)]

    def __getitem__(self, index: int) -> dict[str, object]:

        return self.get_with_num_inputs(index)

    def get_with_num_inputs(
        self,
        index: int,
        num_inputs: int | None = None,
    ) -> dict[str, object]:
        """Get with num inputs."""
        scene_index = self._scene_index(index)
        scene = self.scenes[scene_index]
        cache = self._get_cache(scene)
        cache_indices = [int(value) for value in cache["frame_indices"]]

        rng = random.Random(self.seed + index)

        max_views = min(self.max_views, len(cache_indices) - self.targets_per_input)
        if max_views < self.min_views:
            raise RuntimeError("Teacher pool has too few frames for input/target split")
        if num_inputs is None:
            num_inputs = rng.randint(self.min_views, max_views)
        elif not self.min_views <= num_inputs <= max_views:
            raise ValueError(
                f"Requested {num_inputs} input views outside [{self.min_views}, {max_views}]"
            )

        if self.require_teacher_register_match:

            register_index = int(cache.get("register_frame_index", cache_indices[0]))
            candidates = [value for value in cache_indices if value != register_index]
            chosen = rng.sample(candidates, num_inputs - 1 + self.targets_per_input)
            input_indices = [register_index, *chosen[: num_inputs - 1]]
            target_indices = chosen[num_inputs - 1 :]
        else:

            chosen = rng.sample(cache_indices, num_inputs + self.targets_per_input)
            input_indices = chosen[:num_inputs]
            target_indices = chosen[num_inputs:]

        if self.targets_per_input == 1:
            return self.build_tuple(scene_index, input_indices, target_indices[0], cache=cache)
        return self.build_multi_target_tuple(scene_index, input_indices, target_indices, cache=cache)

    def get_compatible_scene_batch(
        self,
        start_index: int,
        batch_size: int,
        num_inputs: int,
    ) -> tuple[list[dict[str, object]], int]:
        """Return a fair, lossless metadata-bucketed equal-shaped scene batch.

        Only scene/frame metadata is inspected while filling queues. Sampling
        indices that do not fit the next full batch remain queued for later
        calls, so uncommon shapes are delayed but never thrown away. Among
        ready queues, the one with the oldest head index is emitted first.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        cursor = int(start_index)
        buckets = getattr(self, "_compatible_scene_buckets", None)
        expected_cursor = getattr(self, "_compatible_scene_cursor", None)
        expected_batch_size = getattr(self, "_compatible_batch_size", None)
        if (
            buckets is None
            or expected_cursor != cursor
            or expected_batch_size not in (None, batch_size)
        ):
            buckets = {}
            self._compatible_scene_buckets = buckets
        self._compatible_batch_size = batch_size

        def shape_scene_count(
            shape_key: tuple[tuple[int, int], tuple[int, int]],
        ) -> int:
            shape_counts = getattr(self, "_compatible_shape_scene_counts", None)
            return (
                shape_counts.get(shape_key, batch_size)
                if shape_counts is not None
                else batch_size
            )

        def queue_is_ready(
            shape_key: tuple[tuple[int, int], tuple[int, int]],
            queue: deque[tuple[int, int]],
        ) -> bool:
            if shape_scene_count(shape_key) < batch_size:
                return len(queue) >= batch_size
            return len({scene_index for _, scene_index in queue}) >= batch_size

        def ready_key() -> tuple[tuple[int, int], tuple[int, int]] | None:
            ready = [
                (queue[0][0], key)
                for key, queue in buckets.items()
                if queue_is_ready(key, queue)
            ]
            return min(ready)[1] if ready else None

        key = ready_key()
        scanned = 0
        max_scan = max(len(self.scenes) * batch_size, batch_size * 64)
        while key is None and scanned < max_scan:
            sample_index = cursor % len(self)
            scene_index = self._scene_index(sample_index)
            shape_key = self._scene_metadata_batch_key(scene_index)
            queue = buckets.setdefault(shape_key, deque())
            # Never discard a sampling index. Duplicate scene identities wait
            # in the queue until a later distinct-scene batch can use them.
            queue.append((sample_index, scene_index))
            cursor += 1
            scanned += 1
            key = ready_key()

        if key is None:
            counts = sorted((len(queue), shape) for shape, queue in buckets.items())
            raise RuntimeError(
                "Unable to assemble a distinct equal-shaped scene batch from metadata after "
                f"{max_scan} candidates; largest buckets={counts[-5:]}"
            )

        queue = buckets[key]
        if shape_scene_count(key) < batch_size:
            selected = [queue.popleft()[0] for _ in range(batch_size)]
        else:
            selected_entries: list[tuple[int, int]] = []
            deferred: deque[tuple[int, int]] = deque()
            selected_scenes: set[int] = set()
            while queue:
                entry = queue.popleft()
                if (
                    len(selected_entries) < batch_size
                    and entry[1] not in selected_scenes
                ):
                    selected_entries.append(entry)
                    selected_scenes.add(entry[1])
                else:
                    deferred.append(entry)
            queue.extend(deferred)
            selected = [sample_index for sample_index, _ in selected_entries]
        if not queue:
            del buckets[key]
        self._compatible_scene_cursor = cursor
        # Overlap independent mmap faults, JPEG reads and PIL decoding while
        # preserving ``selected`` order exactly.  The executor is permanent:
        # worker threads are not recreated for every batch.
        scene_load_pool = getattr(self, "_scene_load_pool", None)
        if scene_load_pool is None or batch_size == 1:
            batches = [
                self.get_with_num_inputs(sample_index, num_inputs)
                for sample_index in selected
            ]
        else:
            batches = list(
                scene_load_pool.map(
                    lambda sample_index: self.get_with_num_inputs(
                        sample_index,
                        num_inputs,
                    ),
                    selected,
                )
            )
        actual_keys = {
            (
                tuple(torch.as_tensor(batch["input_images"]).shape[-2:]),
                tuple(
                    torch.as_tensor(
                        batch.get(
                            "target_depth_teacher_multi",
                            batch.get("target_depth_teacher"),
                        )
                    ).shape[-2:]
                ),
            )
            for batch in batches
        }
        image_hws = {
            tuple(int(value) for value in batch["image_hw"])
            for batch in batches
        }
        if (
            len(actual_keys) != 1
            or next(iter(actual_keys)) != key
            or image_hws != {key[1]}
        ):
            raise RuntimeError(
                "Scene metadata shape disagrees with the teacher/input tensors: "
                f"metadata={key}, actual={sorted(actual_keys)}, image_hw={sorted(image_hws)}"
            )
        return batches, cursor

    def close(self) -> None:
        """Release the persistent scene-loader workers."""
        if self._scene_load_pool is not None:
            self._scene_load_pool.shutdown(wait=True, cancel_futures=True)
            self._scene_load_pool = None

    def _scene_metadata_batch_key(
        self,
        scene_index: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Compute input/teacher shapes without opening RGB/cache files."""
        scene = self.scenes[scene_index]
        indexed_input_hw = self._input_original_hw_by_key.get(scene.cache_key)
        input_shapes = (
            {self._preprocessed_image_hw(indexed_input_hw, self.patch_size)}
            if indexed_input_hw is not None
            else {
                self._preprocessed_image_hw(frame.original_hw, self.patch_size)
                for frame in scene.frames
            }
        )
        if len(input_shapes) != 1:
            raise RuntimeError(
                f"Scene {scene.scene_id} has mixed frame shapes in metadata: "
                f"{sorted(input_shapes)}"
            )
        teacher_hw = self._teacher_image_hw_by_key.get(scene.cache_key)
        if teacher_hw is None:
            # Exact metadata fallback. mmap parses the small archive header but
            # does not read the large depth/confidence tensor storages.
            cache = torch.load(
                self._cache_path(scene),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            teacher_hw = tuple(int(value) for value in cache["image_hw"])
            self._teacher_image_hw_by_key[scene.cache_key] = teacher_hw
        return next(iter(input_shapes)), teacher_hw

    def _preprocessed_image_hw(
        self,
        original_hw: tuple[int, int],
        patch_size: int,
    ) -> tuple[int, int]:
        """Mirror crop/resize shape arithmetic using only ``FrameRecord.original_hw``."""
        height, width = original_hw
        aspect = height / width
        if aspect < 0.5:
            width = max(1, round(height / 0.5))
        elif aspect > 2.0:
            height = max(1, round(width * 2.0))
        return _target_shape(
            height,
            width,
            self.image_resolution,
            patch_size,
        )

    def build_tuple(
        self,
        scene_index: int,
        input_indices: list[int],
        target_index: int,
        *,
        cache: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build tuple."""
        result = self.build_multi_target_tuple(scene_index, input_indices, [target_index], cache=cache)

        result["target_frame_id"] = result.pop("target_frame_ids")[0]
        result["target_frame_index"] = result.pop("target_frame_indices")[0]
        for plural, singular in (
            ("target_c2w_dataset_multi", "target_c2w_dataset"),
            ("target_intrinsics_multi", "target_intrinsics"),
            ("target_depth_teacher_multi", "target_depth_teacher"),
            ("target_confidence_multi", "target_confidence"),
            ("target_valid_multi", "target_valid"),
            ("target_c2w_teacher_multi", "target_c2w_teacher"),
            ("target_depth_dataset_multi", "target_depth_dataset"),
            ("target_valid_dataset_multi", "target_valid_dataset"),
            ("target_confidence_dataset_multi", "target_confidence_dataset"),
            ("target_scene_scale_dataset_multi", "target_scene_scale_dataset"),
            ("target_uses_dataset_depth_multi", "target_uses_dataset_depth"),
            ("dynamic_downweighted_fraction_multi", "dynamic_downweighted_fraction"),
        ):
            if plural in result:
                result[singular] = result.pop(plural)[0]
        return result

    def build_multi_target_tuple(
        self,
        scene_index: int,
        input_indices: list[int],
        target_indices: list[int],
        *,
        cache: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build multi target tuple."""
        if not target_indices:
            raise ValueError("At least one target frame is required")
        scene = self.scenes[scene_index]
        cache = cache or self._get_cache(scene)
        cache_indices = [int(value) for value in cache["frame_indices"]]

        if set(target_indices) & set(input_indices):
            raise ValueError("Target frame must be held out from input frames")
        if len(set(target_indices)) != len(target_indices):
            raise ValueError("Target frames must be unique")
        if any(index not in cache_indices for index in [*input_indices, *target_indices]):
            raise ValueError("Explicit tuple contains a frame outside the teacher cache")

        register_index = int(cache.get("register_frame_index", cache_indices[0]))
        register_slot = cache_indices.index(register_index)
        register_frame_id = str(cache.get("register_frame_id", cache["frame_ids"][register_slot]))
        if self.require_teacher_register_match and input_indices[0] != register_index:
            raise ValueError(
                "Student VGGT first frame must match the teacher register frame: "
                f"student={input_indices[0]}, teacher={register_index}"
            )

        prepared = [
            preprocess_frame(
                scene.frames[frame_index],
                self.image_resolution,
                self.patch_size,
                self.image_loader,
            )
            for frame_index in input_indices
        ]

        target_slots = [cache_indices.index(index) for index in target_indices]

        return {
            "scene_id": scene.scene_id,
            "dataset_name": scene.dataset_name,

            "input_images": torch.stack([frame.image for frame in prepared]),        # [V, 3, H, W]
            "input_c2w": torch.stack([frame.c2w for frame in prepared]),            # [V, 4, 4]
            "input_intrinsics": torch.stack([frame.intrinsics for frame in prepared]),  # [V, 3, 3]
            "input_frame_ids": [frame.frame_id for frame in prepared],
            "input_frame_indices": list(input_indices),

            "teacher_register_frame_index": register_index,
            "teacher_register_frame_id": register_frame_id,
            "teacher_scene_scale": cache.get("teacher_scene_scale"),

            "target_frame_ids": [cache["frame_ids"][slot] for slot in target_slots],
            "target_frame_indices": list(target_indices),

            "target_c2w_dataset_multi": cache["dataset_c2w"][target_slots],       # [T, 4, 4]
            "target_intrinsics_multi": cache["intrinsics"][target_slots],          # [T, 3, 3]
            "target_depth_teacher_multi": cache["z_depth"][target_slots],          # [T, H, W]
            "target_confidence_multi": cache["confidence"][target_slots],          # [T, H, W]
            "target_valid_multi": cache["valid"][target_slots],                    # [T, H, W]
            "target_c2w_teacher_multi": cache["teacher_c2w"][target_slots],        # [T, 4, 4]
            **self._dataset_depth_payload(scene, target_indices, cache),
            "dataset_to_teacher": cache["dataset_to_teacher"],
            "image_hw": tuple(cache["image_hw"]),                                  # (H, W)
        }

    def _dataset_depth_payload(
        self,
        scene: NerfScene,
        target_indices: list[int],
        cache: dict[str, object],
    ) -> dict[str, object]:
        """Return explicit dataset GT tensors, with teacher tensors as shape-safe fallback."""
        teacher_depth = cache["z_depth"][
            [[int(value) for value in cache["frame_indices"]].index(index) for index in target_indices]
        ].float()
        use_gt = self.use_tartanair_gt_depth and scene.dataset_name.lower() == "tartanair"
        if use_gt:
            pairs = [
                preprocess_z_depth(scene.frames[index], tuple(int(v) for v in cache["image_hw"]))
                for index in target_indices
            ]
            depth = torch.stack([pair[0] for pair in pairs])
            valid = torch.stack([pair[1] for pair in pairs])
            confidence = valid.float()
            scales = torch.stack(
                [
                    item[mask].median() if mask.any() else item.new_tensor(1.0)
                    for item, mask in zip(depth, valid)
                ]
            )
        else:
            depth = teacher_depth
            valid = torch.zeros_like(depth, dtype=torch.bool)
            confidence = torch.zeros_like(depth)
            scales = torch.ones(len(target_indices), dtype=torch.float32)
        return {
            "target_depth_dataset_multi": depth,
            "target_valid_dataset_multi": valid,
            "target_confidence_dataset_multi": confidence,
            "target_scene_scale_dataset_multi": scales,
            "target_uses_dataset_depth_multi": torch.full(
                (len(target_indices),), use_gt, dtype=torch.bool
            ),
        }


def select_target(batch: dict[str, object], target_slot: int) -> dict[str, object]:
    """Select target."""
    if "target_frame_ids" not in batch:
        if target_slot != 0:
            raise IndexError("Single-target batch only has target slot zero")
        return batch

    target_ids = batch["target_frame_ids"]
    if not isinstance(target_ids, list) or not 0 <= target_slot < len(target_ids):
        raise IndexError("target_slot is outside the multi-target batch")

    selected = dict(batch)
    selected["target_frame_id"] = target_ids[target_slot]
    selected["target_frame_index"] = batch["target_frame_indices"][target_slot]
    for plural, singular in (
        ("target_c2w_dataset_multi", "target_c2w_dataset"),
        ("target_intrinsics_multi", "target_intrinsics"),
        ("target_depth_teacher_multi", "target_depth_teacher"),
        ("target_confidence_multi", "target_confidence"),
        ("target_valid_multi", "target_valid"),
        ("target_c2w_teacher_multi", "target_c2w_teacher"),
        ("target_depth_dataset_multi", "target_depth_dataset"),
        ("target_valid_dataset_multi", "target_valid_dataset"),
        ("target_confidence_dataset_multi", "target_confidence_dataset"),
        ("target_scene_scale_dataset_multi", "target_scene_scale_dataset"),
        ("target_uses_dataset_depth_multi", "target_uses_dataset_depth"),
        ("dynamic_downweighted_fraction_multi", "dynamic_downweighted_fraction"),
    ):
        if plural in batch:
            selected[singular] = batch[plural][target_slot]
    return selected
