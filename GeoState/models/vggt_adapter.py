from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from GeoState.core.geometry import homogenize_transform, invert_rigid, unproject_z_depth
from GeoState.core.types import VGGTOutput


def resolve_checkpoint(path: str | Path, source_root: str | Path, min_bytes: int = 1_000_000_000) -> Path:
    """Require the exact VGGT checkpoint selected by the configuration."""
    requested = Path(path)
    if requested.is_file() and requested.stat().st_size >= min_bytes:
        return requested
    raise FileNotFoundError(
        f"VGGT-Omega checkpoint {requested} is missing or smaller than "
        f"{min_bytes} bytes; select the exact checkpoint explicitly."
    )


class VGGTOmegaAdapter(nn.Module):
    """Frozen input-only VGGT-Omega feature and geometry adapter."""

    def __init__(
        self,
        source_root: str,
        checkpoint: str,
        checkpoint_min_bytes: int = 1_000_000_000,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()

        # Source and checkpoint overrides support site-local installations
        # without changing the paper configuration.
        source_root_path = Path(
            os.environ.get("VGGT_OMEGA_IMPORT_ROOT_OVERRIDE", source_root)
        )
        if str(source_root_path) not in sys.path:
            sys.path.insert(0, str(source_root_path))
        model_module = importlib.import_module("vggt_omega.models")
        pose_module = importlib.import_module("vggt_omega.utils.pose_enc")

        model_class = getattr(model_module, "VGGTOmega")
        self._encoding_to_camera = getattr(pose_module, "encoding_to_camera")

        self.model = model_class()
        checkpoint_request = os.environ.get(
            "VGGT_OMEGA_CHECKPOINT_OVERRIDE", checkpoint
        )
        resolved = resolve_checkpoint(
            checkpoint_request, source_root, checkpoint_min_bytes
        )
        state = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)

        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.checkpoint_path = str(resolved)

        self.patch_size = int(self.model.aggregator.patch_size)

        self.feature_layers = tuple(self.model.dense_head.intermediate_layer_idx)

        self.feature_dims = tuple(
            2 * int(self.model.aggregator.patch_embed.embed_dim) for _ in self.feature_layers
        )

    def train(self, mode: bool = True) -> "VGGTOmegaAdapter":
        """Keep VGGT frozen in evaluation mode regardless of the caller state."""
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> VGGTOutput:
        """Extract contextual patch features, cameras, depth, and world points."""

        if images.ndim == 4:
            images = images.unsqueeze(0)

        images = images.to(self.device, non_blocking=True)
        batch, views, _, height, width = images.shape

        if height % self.patch_size or width % self.patch_size:
            raise ValueError("VGGT input dimensions must be divisible by patch size")

        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with amp:
            aggregated, patch_start = self.model.aggregator(images)

        patch_features = []
        for layer_index in self.feature_layers:
            tokens = aggregated[layer_index]  # [B*V, 1+N_patches, D_vggt]
            if tokens is None:
                raise RuntimeError(f"VGGT did not cache requested layer {layer_index}")

            # [B*V, 1+N_patches, D] -> [B*V, N_patches, D]
            patch_features.append(tokens[:, :, patch_start:].float())

        with torch.autocast("cuda", enabled=False) if self.device.type == "cuda" else nullcontext():

            pose = self.model.camera_head(
                aggregated,
                patch_token_start=patch_start,
            ).float()

            depth, confidence = self.model.dense_head(
                aggregated, images=images, patch_token_start=patch_start
            )

        w2c, intrinsics = self._encoding_to_camera(pose, (height, width))

        # [B*V, 3, 4] -> [B*V, 4, 4] -> invert -> [B*V, 4, 4]
        c2w = invert_rigid(homogenize_transform(w2c.float()))

        depth = depth[..., 0].float()
        confidence = confidence.float()

        world_points = unproject_z_depth(depth, intrinsics.float(), c2w)

        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        if any(feature.shape[2] != expected_patches for feature in patch_features):
            raise RuntimeError("VGGT patch count disagrees with dynamic image grid")

        return VGGTOutput(
            patch_features=patch_features,  # list of [B*V, N_patches, D_vggt]
            depth=depth,                    # [B*V, H, W]
            confidence=confidence,          # [B*V, H, W]
            world_points=world_points,      # [B*V, H, W, 3]
            c2w=c2w,                        # [B*V, 4, 4]
            intrinsics=intrinsics.float(),  # [B*V, 3, 3]
            patch_size=self.patch_size,     # int
            patch_hw=(height // self.patch_size, width // self.patch_size),  # (H_p, W_p)
        )

    def decode_patch_features(
        self,
        patch_features: list[torch.Tensor],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode synthesized patch features with the frozen native DenseHead."""
        self.assert_frozen()
        dense_head = self.model.dense_head
        if dense_head is None:
            raise RuntimeError("VGGT-Omega checkpoint has no DenseHead")

        if len(patch_features) != len(self.feature_layers):
            raise ValueError("Expected one synthesized feature tensor per DenseHead layer")

        height, width = (int(value) for value in output_hw)
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Frozen DenseHead output dimensions must be divisible by patch size")

        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        batch = patch_features[0].shape[0]

        aggregated: list[torch.Tensor | None] = [None] * (max(self.feature_layers) + 1)
        for layer_index, expected_dim, feature in zip(
            self.feature_layers, self.feature_dims, patch_features
        ):

            if feature.ndim != 3 or feature.shape != (batch, expected_patches, expected_dim):
                raise ValueError(
                    f"Synthesized layer {layer_index} has {tuple(feature.shape)}, expected "
                    f"{(batch, expected_patches, expected_dim)}"
                )

            aggregated[layer_index] = feature.float()[:, None]

        shape_only_images = torch.zeros(
            batch, 1, 3, height, width,
            device=patch_features[0].device,
            dtype=torch.float32,
        )

        with torch.autocast(device_type=self.device.type, enabled=False):
            depth, confidence = dense_head(
                aggregated,
                images=shape_only_images,
                patch_token_start=0,
                frames_chunk_size=None,
            )

        return depth[..., 0], confidence

    def assert_frozen(self) -> None:
        """Verify that VGGT remains frozen and in evaluation mode."""
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("VGGT-Omega must remain frozen and in eval mode")
