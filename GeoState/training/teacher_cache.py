from __future__ import annotations

from pathlib import Path

import torch

from GeoState.data.common import NerfScene, overlap_clip_frame_indices, preprocess_frame, uniform_frame_indices
from GeoState.core.geometry import estimate_camera_sim3, invert_rigid, robust_scene_scale, transform_points
from GeoState.models.vggt_adapter import VGGTOmegaAdapter


class TeacherCacheBuilder:
    VERSION = 4

    def __init__(
        self,
        teacher: VGGTOmegaAdapter,
        cache_root: str | Path,
        image_resolution: int = 512,
        teacher_frames: int = 24,
        sampling: str = "overlap_clip",
        max_rotation_error_deg: float = 10.0,
        max_relative_baseline_error: float = 0.25,
        max_relative_center_error: float = 0.25,
        confidence_quantile: float = 0.1,
        source_c2w_convention: str = "opengl",
    ) -> None:
        self.teacher = teacher
        self.cache_root = Path(cache_root)
        self.image_resolution = image_resolution
        self.teacher_frames = teacher_frames
        self.sampling = sampling
        self.max_rotation_error_deg = max_rotation_error_deg
        self.max_relative_baseline_error = max_relative_baseline_error
        self.max_relative_center_error = max_relative_center_error
        self.confidence_quantile = confidence_quantile
        self.source_c2w_convention = source_c2w_convention

    @torch.no_grad()
    def build_scene(
        self,
        scene_path: str | Path,
        frame_indices: list[int] | None = None,
        *,
        dataset_name: str = "default",
        scene_id: str | None = None,
        cache_key: str | None = None,
        source_c2w_convention: str | None = None,
        image_root: str | Path | None = None,
    ) -> Path:
        scene = NerfScene(
            scene_path,
            source_c2w_convention or self.source_c2w_convention,
            dataset_name=dataset_name,
            scene_id=scene_id,
            cache_key=cache_key,
            image_root=image_root,
        )
        requested_frames = min(self.teacher_frames, len(scene.frames))
        if requested_frames < 4:
            raise ValueError("A teacher scene needs at least four frames (3 inputs + 1 held-out target)")
        if frame_indices is not None:
            indices = frame_indices
        elif self.sampling == "overlap_clip":
            indices = overlap_clip_frame_indices(len(scene.frames), requested_frames)
        elif self.sampling == "uniform":
            indices = uniform_frame_indices(len(scene.frames), requested_frames)
        else:
            raise ValueError(f"Unknown teacher sampling strategy {self.sampling}")
        prepared = [
            preprocess_frame(
                scene.frames[index], self.image_resolution, self.teacher.patch_size
            )
            for index in indices
        ]
        image_hw = prepared[0].image_hw
        if any(frame.image_hw != image_hw for frame in prepared):
            raise ValueError("Teacher frames must share a native preprocessed shape")
        images = torch.stack([frame.image for frame in prepared], dim=0).unsqueeze(0)
        output = self.teacher(images)
        dataset_c2w = torch.stack([frame.c2w for frame in prepared]).to(output.c2w)
        dataset_to_teacher, diagnostics = estimate_camera_sim3(
            dataset_c2w,
            output.c2w[0],
            max_rotation_error_deg=self.max_rotation_error_deg,
            max_relative_baseline_error=self.max_relative_baseline_error,
            max_relative_center_error=self.max_relative_center_error,
        )
        if not diagnostics.valid.all():
            raise RuntimeError(
                "Teacher camera alignment failed: "
                f"condition={diagnostics.condition}, rotation_deg={diagnostics.rotation_error_deg.tolist()}, "
                f"rotation_p90_deg={diagnostics.rotation_error_p90_deg.tolist()}, "
                f"baseline_rel={diagnostics.relative_baseline_error.tolist()}, "
                f"baseline_p90={diagnostics.relative_baseline_error_p90.tolist()}, "
                f"center_rel={diagnostics.center_error.tolist()}"
            )

        # Recompute camera z explicitly from the world pointmaps.
        teacher_w2c = invert_rigid(output.c2w)
        camera_points = transform_points(teacher_w2c[:, :, None, None], output.world_points)
        z_depth = camera_points[..., 2]
        confidence = output.confidence
        confidence_threshold = torch.quantile(confidence.flatten(2).float(), self.confidence_quantile, dim=-1)
        valid = (
            torch.isfinite(z_depth)
            & (z_depth > 0)
            & torch.isfinite(confidence)
            & (confidence >= confidence_threshold[..., None, None])
        )
        teacher_scale = robust_scene_scale(z_depth[:, :1], confidence[:, :1], valid[:, :1])[0]
        payload = {
            "version": self.VERSION,
            "scene_id": scene.scene_id,
            "dataset_name": scene.dataset_name,
            "cache_key": scene.cache_key,
            "frame_indices": torch.tensor(indices, dtype=torch.long),
            "frame_ids": [frame.frame_id for frame in prepared],
            # VGGT's first image establishes its register/gauge. Student batches
            # must put this exact frame first, while all later views may differ.
            "register_frame_index": int(indices[0]),
            "register_frame_id": prepared[0].frame_id,
            "image_paths": [str(scene.frames[index].image_path.relative_to(scene.path)) for index in indices],
            "image_hw": list(image_hw),
            "z_depth": z_depth[0].cpu().to(torch.float16),
            "confidence": confidence[0].cpu().to(torch.float16),
            "valid": valid[0].cpu(),
            "intrinsics": output.intrinsics[0].cpu(),
            "dataset_c2w": dataset_c2w.cpu(),
            "teacher_c2w": output.c2w[0].cpu(),
            "pixel_transform": torch.stack([frame.pixel_transform for frame in prepared]),
            "dataset_to_teacher": {
                "scale": dataset_to_teacher.scale[0].cpu(),
                "rotation": dataset_to_teacher.rotation[0].cpu(),
                "translation": dataset_to_teacher.translation[0].cpu(),
            },
            "teacher_scene_scale": teacher_scale.cpu(),
            "alignment": {
                "rotation_error_deg": diagnostics.rotation_error_deg[0].cpu(),
                "rotation_error_p90_deg": diagnostics.rotation_error_p90_deg[0].cpu(),
                "relative_baseline_error": diagnostics.relative_baseline_error[0].cpu(),
                "relative_baseline_error_p90": diagnostics.relative_baseline_error_p90[0].cpu(),
                "center_error": diagnostics.center_error[0].cpu(),
                "center_error_p90": diagnostics.center_error_p90[0].cpu(),
                "condition": diagnostics.condition[0],
            },
        }
        self.cache_root.mkdir(parents=True, exist_ok=True)
        output_path = self.cache_root / f"{scene.cache_key}.pt"
        temporary = output_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(output_path)
        return output_path
