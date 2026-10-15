from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from GeoState.core.config import AlignmentConfig, ModelConfig
from GeoState.core.geometry import (
    apply_sim3_to_c2w,
    estimate_camera_sim3,
    full_plucker_ray_map,
    invert_rigid,
    robust_scene_scale,
    transform_points,
)
from GeoState.models.networks import (
    SceneTokenizer,
    VGGTPatchFeatureDecoder,
)
from GeoState.core.types import AlignmentDiagnostics, DepthPrediction, SceneEncoding, Sim3, VGGTOutput
from GeoState.models.vggt_adapter import VGGTOmegaAdapter


def confidence_pool_dense(
    points: torch.Tensor,
    confidence: torch.Tensor,
    patch_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Confidence pool dense."""
    batch, views, height, width, _ = points.shape

    points_cf = points.reshape(batch * views, height, width, 3).permute(0, 3, 1, 2)
    # [B, V, H, W] -> [B*V, 1, H, W]
    confidence_cf = confidence.reshape(batch * views, 1, height, width).clamp_min(0)

    valid_cf = torch.isfinite(points_cf).all(1, keepdim=True) & torch.isfinite(confidence_cf) & (confidence_cf > 0)

    weights = confidence_cf * valid_cf

    denominator = F.adaptive_avg_pool2d(weights, patch_hw).clamp_min(1e-8)
    pooled_points = F.adaptive_avg_pool2d(torch.nan_to_num(points_cf) * weights, patch_hw) / denominator
    pooled_confidence = F.adaptive_avg_pool2d(weights, patch_hw)
    pooled_valid = F.adaptive_max_pool2d(valid_cf.float(), patch_hw) > 0

    patch_count = patch_hw[0] * patch_hw[1]

    return (
        pooled_points.permute(0, 2, 3, 1).reshape(batch, views, patch_count, 3),
        pooled_confidence.reshape(batch, views, patch_count),
        pooled_valid.reshape(batch, views, patch_count),
    )


class GeometrySceneModel(nn.Module):
    """Encode multi-view geometry into scene tokens and decode queried depth."""

    def __init__(
        self,
        vggt: VGGTOmegaAdapter,
        model_config: ModelConfig,
        alignment_config: AlignmentConfig,
    ) -> None:
        super().__init__()
        self.vggt = vggt
        self.model_config = model_config
        self.alignment_config = alignment_config
        if model_config.memory_mode != "perceiver" or model_config.view_mode != "all":
            raise ValueError("The released model requires memory_mode=perceiver and view_mode=all")

        self.tokenizer = SceneTokenizer(
            vggt.feature_dims,
            dim=model_config.dim,
            num_tokens=model_config.scene_tokens,
            layers=model_config.perceiver_layers,
            heads=model_config.heads,
            mlp_ratio=model_config.mlp_ratio,
            fourier_bands=model_config.point_fourier_bands,
            use_3d_pe=model_config.use_3d_pe,
            patch_size=vggt.patch_size,
            use_relative_camera_pose=model_config.use_relative_camera_pose,
            use_input_patch_rays=model_config.use_input_patch_rays,
            camera_translation_fourier_bands=model_config.camera_translation_fourier_bands,
            input_ray_fourier_bands=model_config.input_ray_fourier_bands,
            feature_projection_dim=model_config.vggt_feature_projection_dim,
            feature_fusion_hidden_dim=model_config.vggt_feature_fusion_hidden_dim,
        )

        if model_config.decoder_arch != "frozen_vggt_dpt":
            raise ValueError("The paper release supports decoder_arch=frozen_vggt_dpt only")
        self.decoder = VGGTPatchFeatureDecoder(
            vggt.patch_size,
            vggt.feature_dims,
            scene_dim=model_config.dim,
            dim=model_config.dim,
            layers=model_config.vggt_dpt_decoder_layers,
            capture_layers=model_config.vggt_dpt_decoder_capture_layers,
            state_conditioner_layers=model_config.vggt_dpt_state_conditioner_layers,
            heads=model_config.heads,
            mlp_ratio=model_config.mlp_ratio,
            ray_fourier_bands=model_config.ray_fourier_bands,
            hidden_dim=model_config.vggt_dpt_decoder_hidden_dim,
            branch_layers=model_config.vggt_dpt_branch_layers,
        )

    def train(self, mode: bool = True) -> "GeometrySceneModel":
        """Set the module training mode."""
        super().train(mode)
        self.vggt.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Trainable parameters."""
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("vggt.") and parameter.requires_grad]

    def encode_scene(
        self,
        input_images: torch.Tensor,
        input_c2w: torch.Tensor,
        input_intrinsics: torch.Tensor,
        frame_ids: Sequence[Sequence[str]] | None = None,
    ) -> SceneEncoding:
        """Encode scene."""
        self.vggt.assert_frozen()
        output = self.vggt(input_images)
        return self.encode_vggt_output(
            output,
            input_c2w,
            input_intrinsics=input_intrinsics,
            frame_ids=frame_ids,
            reference_index=0,
        )

    def encode_vggt_output(
        self,
        output: VGGTOutput,
        input_c2w: torch.Tensor,
        *,
        input_intrinsics: torch.Tensor | None = None,
        frame_ids: Sequence[Sequence[str]] | None = None,
        reference_index: int = 0,
        single_view_scale: torch.Tensor | None = None,
        scene_scale_override: torch.Tensor | None = None,
        tokenizer_attention_view_mask: torch.Tensor | None = None,
        alignment_view_mask: torch.Tensor | None = None,
        scene_scale_view_mask: torch.Tensor | None = None,
        defer_scene_tokenization: bool = False,
        input_camera_conditioning_source: str = "dataset",
    ) -> SceneEncoding:
        """Encode vggt output."""
        if input_camera_conditioning_source not in {"dataset", "vggt"}:
            raise ValueError(
                "input_camera_conditioning_source must be 'dataset' or 'vggt', "
                f"got {input_camera_conditioning_source!r}"
            )
        if not 0 <= reference_index < output.c2w.shape[1]:
            raise IndexError("reference_index is outside the input view set")
        if tokenizer_attention_view_mask is not None:
            expected_shape = output.c2w.shape[:2]
            if tokenizer_attention_view_mask.shape != expected_shape:
                raise ValueError(
                    "tokenizer_attention_view_mask must have shape "
                    f"{tuple(expected_shape)}, got "
                    f"{tuple(tokenizer_attention_view_mask.shape)}"
                )
            tokenizer_attention_view_mask = tokenizer_attention_view_mask.to(
                device=output.c2w.device,
                dtype=torch.bool,
            )
            if not tokenizer_attention_view_mask.any(dim=1).all():
                raise ValueError(
                    "Every scene must leave at least one input view visible "
                    "to the scene tokenizer"
                )
        if alignment_view_mask is not None:
            expected_shape = output.c2w.shape[:2]
            if alignment_view_mask.shape != expected_shape:
                raise ValueError(
                    "alignment_view_mask must have shape "
                    f"{tuple(expected_shape)}, got {tuple(alignment_view_mask.shape)}"
                )
            alignment_view_mask = alignment_view_mask.to(
                device=output.c2w.device,
                dtype=torch.bool,
            )
            if not alignment_view_mask.any(dim=1).all():
                raise ValueError("Every scene must keep at least one camera for alignment")
        if scene_scale_view_mask is not None:
            expected_shape = output.c2w.shape[:2]
            if scene_scale_view_mask.shape != expected_shape:
                raise ValueError(
                    "scene_scale_view_mask must have shape "
                    f"{tuple(expected_shape)}, got {tuple(scene_scale_view_mask.shape)}"
                )
            scene_scale_view_mask = scene_scale_view_mask.to(
                device=output.depth.device,
                dtype=torch.bool,
            )
            if not scene_scale_view_mask.any(dim=1).all():
                raise ValueError("Every scene must keep at least one view for scene scale")

        with torch.autocast(device_type=output.c2w.device.type, enabled=False):
            source_c2w = input_c2w.to(output.c2w)
            alignment_kwargs = {
                "max_rotation_error_deg": self.alignment_config.max_rotation_error_deg,
                "max_relative_baseline_error": self.alignment_config.max_relative_baseline_error,
                "max_relative_center_error": self.alignment_config.max_relative_center_error,
                "min_baseline": self.alignment_config.min_baseline,
            }
            if alignment_view_mask is None or bool(alignment_view_mask.all()):
                dataset_to_student, diagnostics = estimate_camera_sim3(
                    source_c2w,
                    output.c2w,
                    single_view_scale=single_view_scale,
                    **alignment_kwargs,
                )
            else:
                # Waymo may give the frozen VGGT extra temporal context that is
                # intentionally hidden from the scene tokenizer.  Those cameras
                # must not influence the dataset-to-student frame fit.  The
                # number of visible same-time cameras can differ per scene, so
                # fit each compact camera set and stack the tiny diagnostics.
                transforms: list[Sim3] = []
                diagnostic_rows: list[AlignmentDiagnostics] = []
                for batch_index, visible in enumerate(alignment_view_mask):
                    selected_count = int(visible.sum())
                    item_scale = None
                    if selected_count == 1:
                        if single_view_scale is None:
                            raise ValueError(
                                "Single-camera masked alignment requires an explicit scale"
                            )
                        item_scale = torch.as_tensor(
                            single_view_scale,
                            device=source_c2w.device,
                            dtype=source_c2w.dtype,
                        ).reshape(-1)[batch_index : batch_index + 1]
                    transform, item_diagnostics = estimate_camera_sim3(
                        source_c2w[batch_index, visible],
                        output.c2w[batch_index, visible],
                        single_view_scale=item_scale,
                        **alignment_kwargs,
                    )
                    transforms.append(transform)
                    diagnostic_rows.append(item_diagnostics)
                dataset_to_student = Sim3(
                    scale=torch.cat([value.scale.reshape(-1) for value in transforms]),
                    rotation=torch.cat([value.rotation for value in transforms]),
                    translation=torch.cat([value.translation for value in transforms]),
                )
                diagnostics = AlignmentDiagnostics(
                    rotation_error_deg=torch.cat(
                        [value.rotation_error_deg for value in diagnostic_rows]
                    ),
                    rotation_error_p90_deg=torch.cat(
                        [value.rotation_error_p90_deg for value in diagnostic_rows]
                    ),
                    relative_baseline_error=torch.cat(
                        [value.relative_baseline_error for value in diagnostic_rows]
                    ),
                    relative_baseline_error_p90=torch.cat(
                        [value.relative_baseline_error_p90 for value in diagnostic_rows]
                    ),
                    center_error=torch.cat(
                        [value.center_error for value in diagnostic_rows]
                    ),
                    center_error_p90=torch.cat(
                        [value.center_error_p90 for value in diagnostic_rows]
                    ),
                    min_baseline=torch.cat(
                        [value.min_baseline for value in diagnostic_rows]
                    ),
                    condition=[
                        condition
                        for value in diagnostic_rows
                        for condition in value.condition
                    ],
                    valid=torch.cat([value.valid for value in diagnostic_rows]),
                )
        # A full compute batch may contain a rejected scene beside valid
        # scenes. Preserve the single-scene exception;
        # the batched training wrapper masks rejected elements individually.
        if output.c2w.shape[0] == 1 and not diagnostics.valid.all():
            raise RuntimeError(f"Sim(3) alignment failed: {diagnostics.condition}")

        scale_valid = None
        if scene_scale_view_mask is not None:
            depth_for_scale = output.depth
            if depth_for_scale.ndim == 5 and depth_for_scale.shape[-1] == 1:
                depth_for_scale = depth_for_scale[..., 0]
            confidence_for_scale = output.confidence
            scale_valid = scene_scale_view_mask
            while scale_valid.ndim < depth_for_scale.ndim:
                scale_valid = scale_valid.unsqueeze(-1)
            scale_valid = scale_valid.expand_as(depth_for_scale)
            scale_valid = (
                scale_valid
                & torch.isfinite(depth_for_scale)
                & (depth_for_scale > 0)
                & torch.isfinite(confidence_for_scale)
                & (confidence_for_scale > 0)
            )
        natural_scale = robust_scene_scale(
            output.depth, output.confidence, valid=scale_valid
        )
        if scene_scale_override is None:
            scale = natural_scale
        else:
            scale = torch.as_tensor(
                scene_scale_override,
                device=output.depth.device,
                dtype=torch.float32,
            ).reshape(-1)
            if scale.shape != natural_scale.shape:
                raise ValueError(
                    "scene_scale_override must contain one value per batch item; "
                    f"expected {tuple(natural_scale.shape)}, got {tuple(scale.shape)}"
                )
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError("scene_scale_override must be finite and positive")

        reference_c2w = output.c2w[:, reference_index]
        reference_w2c = invert_rigid(reference_c2w)

        points_reference = transform_points(reference_w2c[:, None], output.world_points)

        points_patch, confidence_patch, valid_patch = confidence_pool_dense(
            points_reference, output.confidence, output.patch_hw
        )

        # points_patch: [B, V, N, 3] / scale[:, None, None, None] -> [B, V, N, 3]
        points_patch = points_patch / scale[:, None, None, None]

        view_order = [reference_index] + [index for index in range(output.c2w.shape[1]) if index != reference_index]

        relative_c2w_normalized = None
        input_ray_maps = None
        camera_source = "disabled"
        intrinsics_source = "disabled"

        needs_patch_conditioning = (
            self.model_config.use_relative_camera_pose
            or self.model_config.use_input_patch_rays
        )

        if needs_patch_conditioning:
            with torch.autocast(device_type=output.c2w.device.type, enabled=False):
                # Keep dataset-to-student estimation independent from the
                # camera/ray conditioning source.  This permits a strict
                # mode in which VGGT poses condition the tokenizer while
                # the dataset Sim(3) is retained for metric query cameras and
                # metre conversion.
                if input_camera_conditioning_source == "vggt":
                    conditioning_c2w = output.c2w.float()
                    camera_source = "vggt"
                else:

                    dataset_c2w = input_c2w.to(device=output.c2w.device, dtype=torch.float32)
                    use_dataset_cameras = (
                        dataset_c2w.shape == output.c2w.shape
                        and bool(torch.isfinite(dataset_c2w).all())
                    )
                    if use_dataset_cameras:

                        conditioning_c2w = apply_sim3_to_c2w(dataset_to_student, dataset_c2w)
                        camera_source = "dataset"
                    else:
                        conditioning_c2w = output.c2w.float()
                        camera_source = "vggt-fallback"

                reference_w2c_float = invert_rigid(reference_c2w.float())
                relative_c2w_normalized = reference_w2c_float[:, None] @ conditioning_c2w
                relative_c2w_normalized = relative_c2w_normalized.clone()

                relative_c2w_normalized[..., :3, 3] /= scale[:, None, None].float().clamp_min(1e-8)

                if input_intrinsics is not None:
                    candidate_intrinsics = input_intrinsics.to(device=output.c2w.device, dtype=torch.float32)
                else:
                    candidate_intrinsics = None
                use_dataset_intrinsics = (
                    candidate_intrinsics is not None
                    and candidate_intrinsics.shape == output.intrinsics.shape
                    and bool(torch.isfinite(candidate_intrinsics).all())
                    and bool((candidate_intrinsics[..., 0, 0] > 0).all())
                    and bool((candidate_intrinsics[..., 1, 1] > 0).all())
                )
                conditioning_intrinsics = candidate_intrinsics if use_dataset_intrinsics else output.intrinsics.float()
                intrinsics_source = "dataset" if use_dataset_intrinsics else "vggt-fallback"

                if self.model_config.use_input_patch_rays:
                    batch, views = relative_c2w_normalized.shape[:2]
                    height, width = output.depth.shape[-2:]

                    input_ray_maps = full_plucker_ray_map(
                        relative_c2w_normalized.reshape(batch * views, 4, 4),
                        conditioning_intrinsics.reshape(batch * views, 3, 3),
                        (height, width),
                        torch.ones(batch * views, device=output.c2w.device, dtype=torch.float32),
                    ).reshape(batch, views, 6, height, width)

        ordered_valid_patch = valid_patch[:, view_order]
        ordered_attention_view_mask = None
        if tokenizer_attention_view_mask is not None:
            ordered_attention_view_mask = tokenizer_attention_view_mask[:, view_order]
            ordered_valid_patch = (
                ordered_valid_patch
                & ordered_attention_view_mask[:, :, None]
            )
        memory, memory_mask = self.tokenizer.build_memory(
            [feature[:, view_order] for feature in output.patch_features],
            points_patch[:, view_order],
            confidence_patch[:, view_order],
            ordered_valid_patch,
            relative_c2w_normalized=(
                relative_c2w_normalized[:, view_order] if relative_c2w_normalized is not None else None
            ),
            input_ray_maps=input_ray_maps[:, view_order] if input_ray_maps is not None else None,
        )

        if defer_scene_tokenization:
            if self.model_config.memory_mode != "perceiver":
                raise ValueError(
                    "Deferred scene tokenization requires perceiver memory mode"
                )
            scene_tokens = memory.new_empty(memory.shape[0], 0, memory.shape[-1])
        else:
            scene_tokens = self.tokenizer(memory, memory_mask)  # [B, K, D]

        patch_rays = None  # [B, V, N_patches, 6]
        patch_c2w = None   # [B, V, N_patches, 4, 4]
        if input_ray_maps is not None and relative_c2w_normalized is not None:
            batch, views, channels, height, width = input_ray_maps.shape

            # [B*V, 6, H, W] -> adaptive_avg_pool2d -> [B*V, 6, H_p, W_p]
            # -> [B, V, N_patches, 6]
            patch_rays = F.adaptive_avg_pool2d(
                input_ray_maps.reshape(batch * views, channels, height, width),
                output.patch_hw,
            ).permute(0, 2, 3, 1).reshape(batch, views, -1, channels)

            patch_c2w = relative_c2w_normalized[:, :, None].expand(
                -1, -1, patch_rays.shape[2], -1, -1
            )

        return SceneEncoding(
            memory=memory,
            memory_mask=memory_mask,
            scene_tokens=scene_tokens,
            reference_c2w_student=reference_c2w,
            dataset_to_student=dataset_to_student,
            scene_scale=scale,
            input_depth=output.depth,
            input_confidence=output.confidence,
            input_points_reference=points_reference,
            patch_hw=output.patch_hw,
            metadata={
                "frame_ids": frame_ids,
                "alignment": diagnostics,
                # Retain the unconstrained VGGT scale for drift diagnostics.  The
                # public scene_scale is the scale actually used by every geometry,
                # camera and ray normalization path above.
                "natural_scene_scale": natural_scale.detach(),
                "scene_scale_was_overridden": scene_scale_override is not None,
                "scene_scale_view_mask": (
                    scene_scale_view_mask.detach()
                    if scene_scale_view_mask is not None
                    else torch.ones(
                        output.c2w.shape[:2],
                        device=output.c2w.device,
                        dtype=torch.bool,
                    )
                ),
                "student_c2w": output.c2w,
                "reference_index": reference_index,
                "tokenizer_attention_view_mask": (
                    ordered_attention_view_mask.detach()
                    if ordered_attention_view_mask is not None
                    else torch.ones(
                        output.c2w.shape[:2],
                        device=output.c2w.device,
                        dtype=torch.bool,
                    )
                ),
                "input_camera_conditioning_source": camera_source,
                "input_intrinsics_conditioning_source": intrinsics_source,
                "relative_input_c2w_normalized": relative_c2w_normalized,
                "student_intrinsics": output.intrinsics,
                "input_patch_points_normalized": points_patch[:, view_order].detach(),
                "input_patch_rays": (
                    patch_rays[:, view_order].detach() if patch_rays is not None else None
                ),
                "input_patch_c2w_normalized": (
                    patch_c2w[:, view_order].detach() if patch_c2w is not None else None
                ),
                "input_patch_valid": valid_patch[:, view_order].detach(),
                "input_patch_feature_targets": tuple(
                    feature[:, view_order].detach() for feature in output.patch_features
                ),
                "scene_tokenization_deferred": defer_scene_tokenization,
            },
        )

    def encode_temporal_scene_states_from_vggt_output(
        self,
        output: VGGTOutput,
        input_c2w: torch.Tensor,
        *,
        input_intrinsics: torch.Tensor | None,
        state_view_mask: torch.Tensor,
        single_view_scale: torch.Tensor | None = None,
        scene_scale_override: torch.Tensor | None = None,
        alignment_view_mask: torch.Tensor | None = None,
        scene_scale_view_mask: torch.Tensor | None = None,
        frame_ids: Sequence[Sequence[str]] | None = None,
        reference_index: int = 0,
        input_camera_conditioning_source: str = "dataset",
    ) -> tuple[torch.Tensor, SceneEncoding]:
        """Tokenize several disjoint temporal states from one shared VGGT geometry pass."""
        if self.model_config.memory_mode != "perceiver":
            raise ValueError("Temporal scene states require perceiver memory mode")
        if self.model_config.view_mode != "all":
            raise ValueError("Temporal scene states require view_mode='all'")
        if state_view_mask.ndim != 3:
            raise ValueError("state_view_mask must have shape [B, T, V]")
        batch, state_count, view_count = state_view_mask.shape
        if (batch, view_count) != tuple(output.c2w.shape[:2]):
            raise ValueError("state_view_mask disagrees with the VGGT output shape")
        state_view_mask = state_view_mask.to(
            device=output.c2w.device,
            dtype=torch.bool,
        )
        visible_counts = state_view_mask.sum(dim=-1)
        if not visible_counts.gt(0).all():
            raise ValueError("Every temporal state must expose at least one view")
        if not torch.equal(
            visible_counts,
            visible_counts.new_full(visible_counts.shape, int(visible_counts[0, 0])),
        ):
            raise ValueError("Every temporal state must expose the same number of views")

        encoding = self.encode_vggt_output(
            output,
            input_c2w,
            input_intrinsics=input_intrinsics,
            frame_ids=frame_ids,
            reference_index=reference_index,
            single_view_scale=single_view_scale,
            scene_scale_override=scene_scale_override,
            defer_scene_tokenization=True,
            alignment_view_mask=alignment_view_mask,
            scene_scale_view_mask=scene_scale_view_mask,
            input_camera_conditioning_source=input_camera_conditioning_source,
        )
        if reference_index != 0:
            raise ValueError("Optimized temporal tokenization currently requires reference_index=0")
        if encoding.memory.shape[1] % view_count:
            raise ValueError("Patch memory cannot be divided evenly across input views")
        patches_per_view = encoding.memory.shape[1] // view_count
        scene_dim = encoding.memory.shape[-1]
        memory_by_view = encoding.memory.reshape(
            batch,
            view_count,
            patches_per_view,
            scene_dim,
        )
        if encoding.memory_mask is None:
            valid_by_view = torch.ones(
                batch,
                view_count,
                patches_per_view,
                device=encoding.memory.device,
                dtype=torch.bool,
            )
        else:
            valid_by_view = encoding.memory_mask.reshape(
                batch,
                view_count,
                patches_per_view,
            )
        views_per_state = int(visible_counts[0, 0])
        selected_memory = memory_by_view[:, None].expand(
            -1, state_count, -1, -1, -1
        )[state_view_mask].reshape(
            batch * state_count,
            views_per_state * patches_per_view,
            scene_dim,
        )
        selected_valid = valid_by_view[:, None].expand(
            -1, state_count, -1, -1
        )[state_view_mask].reshape(
            batch * state_count,
            views_per_state * patches_per_view,
        )
        scene_states = self.tokenizer(selected_memory, selected_valid).reshape(
            batch,
            state_count,
            -1,
            scene_dim,
        )
        encoding.scene_tokens = scene_states[:, -1]
        encoding.metadata["temporal_state_view_mask"] = state_view_mask.detach()
        encoding.metadata["tokenizer_attention_view_mask"] = state_view_mask[:, -1].detach()
        encoding.metadata["scene_tokenization_deferred"] = False
        return scene_states, encoding

    def query_camera_in_reference(self, encoding: SceneEncoding, query_c2w_dataset: torch.Tensor) -> torch.Tensor:
        """Query camera in reference."""

        query_student = apply_sim3_to_c2w(encoding.dataset_to_student, query_c2w_dataset)

        return invert_rigid(encoding.reference_c2w_student) @ query_student

    def decode_depth(
        self,
        encoding: SceneEncoding,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        output_hw: tuple[int, int],
        *,
        query_is_dataset_camera: bool = True,
    ) -> DepthPrediction:
        """Decode depth."""

        if query_is_dataset_camera:

            query_c2w = self.query_camera_in_reference(encoding, query_c2w)

        # [B, 4, 4] + [B, 3, 3] + (H, W) + [B] -> [B, 6, H, W]
        rays = full_plucker_ray_map(query_c2w, query_intrinsics, output_hw, encoding.scene_scale)

        patch_features, patch_hw, _ = self.decoder(rays, encoding.scene_tokens, None)
        expected_hw = (output_hw[0] // self.vggt.patch_size, output_hw[1] // self.vggt.patch_size)
        if patch_hw != expected_hw:
            raise RuntimeError("Feature decoder patch grid disagrees with frozen DenseHead")
        z_depth, _ = self.vggt.decode_patch_features(patch_features, output_hw)
        return DepthPrediction(
            z_depth=z_depth,
            log_z_depth=torch.log(z_depth.clamp_min(1e-6)),
        )
