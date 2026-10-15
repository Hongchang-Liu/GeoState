from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

#

#     [sin(2^0 * π * x), cos(2^0 * π * x), sin(2^1 * π * x), cos(2^1 * π * x), ...]

#


class FourierFeatures(nn.Module):
    """Multi-frequency sinusoidal embedding for continuous coordinates."""

    def __init__(self, bands: int, include_input: bool = True) -> None:
        super().__init__()
        self.bands = bands
        self.include_input = include_input

        self.register_buffer("frequencies", (2.0 ** torch.arange(bands)) * torch.pi, persistent=False)

    @property
    def multiplier(self) -> int:
        """Multiplier."""
        return 2 * self.bands + int(self.include_input)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        # [..., d] -> [..., d, 1] * [bands] -> [..., d, bands]
        encoded = value[..., None] * self.frequencies.to(value.dtype)

        parts = [torch.sin(encoded), torch.cos(encoded)]
        # cat -> [..., d, 2*bands], flatten(-2) -> [..., d*2*bands]
        flattened = torch.cat(parts, dim=-1).flatten(-2)

        return torch.cat((value, flattened), dim=-1) if self.include_input else flattened


class FeedForward(nn.Module):
    """Transformer feed-forward sublayer."""

    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AttentionBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, cross_attention: bool) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim) if cross_attention else None
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forward pass."""

        normalized_query = self.query_norm(query)
        if self.cross_attention:
            if context is None or self.context_norm is None:
                raise ValueError("Cross-attention requires context")
            normalized_context = self.context_norm(context)
        else:
            normalized_context = normalized_query

        attended, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            key_padding_mask=None if context_mask is None else ~context_mask,
            need_weights=False,
        )
        query = query + attended

        return query + self.ff(self.ff_norm(query))


class PerceiverLayer(nn.Module):
    """Cross-attend latent tokens to memory and then refine them jointly."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.cross = AttentionBlock(dim, heads, mlp_ratio, cross_attention=True)
        self.self_attention = AttentionBlock(dim, heads, mlp_ratio, cross_attention=False)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Run the forward pass."""
        queries = self.cross(queries, memory, mask)
        return self.self_attention(queries)


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to 6d."""
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in [3,3]")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


class RelativeCameraEncoder(nn.Module):
    """Embed relative camera rotation and translation."""

    def __init__(self, dim: int = 512, translation_fourier_bands: int = 4) -> None:
        super().__init__()
        self.translation_fourier = FourierFeatures(translation_fourier_bands)

        input_dim = 6 + 3 * self.translation_fourier.multiplier
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, relative_c2w_normalized: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        if relative_c2w_normalized.shape[-2:] != (4, 4):
            raise ValueError("relative_c2w_normalized must end in [4,4]")

        rotation = rotation_matrix_to_6d(relative_c2w_normalized[..., :3, :3])

        translation = self.translation_fourier(relative_c2w_normalized[..., :3, 3])

        # [..., 6] + [..., 3*multiplier] -> [..., 6+3*multiplier] -> [..., D]
        return self.projection(torch.cat((rotation, translation), dim=-1).float())


class SceneTokenizer(nn.Module):
    """Fuse multi-view VGGT geometry into a fixed-size scene state."""

    def __init__(
        self,
        feature_dims: Sequence[int],
        dim: int = 512,
        num_tokens: int = 512,
        layers: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        fourier_bands: int = 10,
        use_3d_pe: bool = True,
        patch_size: int = 16,
        use_relative_camera_pose: bool = False,
        use_input_patch_rays: bool = False,
        camera_translation_fourier_bands: int = 4,
        input_ray_fourier_bands: int = 4,
        feature_projection_dim: int | None = None,
        feature_fusion_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim % len(feature_dims):
            raise ValueError("dim must be divisible by the number of VGGT feature layers")

        per_layer = dim // len(feature_dims) if feature_projection_dim is None else int(feature_projection_dim)
        if per_layer < 1:
            raise ValueError("feature_projection_dim must be positive")

        self.feature_projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(size), nn.Linear(size, per_layer)) for size in feature_dims]
        )

        appearance_dim = per_layer * len(feature_dims)
        if appearance_dim == dim:

            self.appearance_fusion = None
        else:
            hidden = (
                max(2 * dim, appearance_dim)
                if feature_fusion_hidden_dim is None
                else int(feature_fusion_hidden_dim)
            )
            if hidden < 1:
                raise ValueError("feature_fusion_hidden_dim must be positive")
            # LayerNorm -> Linear -> GELU -> Linear (appearance_dim -> hidden -> dim)
            self.appearance_fusion = nn.Sequential(
                nn.LayerNorm(appearance_dim),
                nn.Linear(appearance_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )

        self.point_fourier = FourierFeatures(fourier_bands)
        geometry_dim = 3 * self.point_fourier.multiplier if use_3d_pe else 3
        # LayerNorm -> Linear -> GELU -> Linear

        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(geometry_dim + 1), nn.Linear(geometry_dim + 1, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.camera_encoder = (
            RelativeCameraEncoder(dim, camera_translation_fourier_bands) if use_relative_camera_pose else None
        )

        self.input_ray_encoder = (
            FullResolutionRayEncoder(patch_size, dim, input_ray_fourier_bands) if use_input_patch_rays else None
        )

        if use_relative_camera_pose or use_input_patch_rays:
            branch_count = 2 + int(use_relative_camera_pose) + int(use_input_patch_rays)
            fusion_dim = branch_count * dim
            self.modality_fusion = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, 2 * dim),
                nn.GELU(),
                nn.Linear(2 * dim, dim),
            )
        else:

            self.modality_fusion = None

        self.input_norm = nn.LayerNorm(dim)

        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

        self.layers = nn.ModuleList([PerceiverLayer(dim, heads, mlp_ratio) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.use_3d_pe = use_3d_pe

    def build_memory(
        self,
        patch_features: Sequence[torch.Tensor],
        points: torch.Tensor,                       # [B, V, N_patches, 3]
        confidence: torch.Tensor,                   # [B, V, N_patches]
        valid: torch.Tensor,                        # [B, V, N_patches]
        relative_c2w_normalized: torch.Tensor | None = None,
        input_ray_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build memory."""
        if len(patch_features) != len(self.feature_projections):
            raise ValueError("Unexpected number of VGGT feature layers")
        # 4 * [B, V, N, 2048] -> 4 * [B, V, N, per_layer] -> cat -> [B, V, N, 4*per_layer]
        projected = [projection(feature.float()) for projection, feature in zip(self.feature_projections, patch_features)]
        appearance = torch.cat(projected, dim=-1)

        if self.appearance_fusion is not None:
            appearance = self.appearance_fusion(appearance)
        # appearance: [B, V, N_patches, D]

        # [B, V, N, 3] -> [B, V, N, 3*multiplier]
        geometry = self.point_fourier(points) if self.use_3d_pe else points

        # [B, V, N] -> [B, V, N, 1]
        confidence_feature = torch.log1p(confidence.clamp_min(0))[..., None]
        # [B, V, N, 3*multiplier+1] -> MLP -> [B, V, N, D]
        geometry_feature = self.geometry_projection(torch.cat((geometry, confidence_feature), dim=-1))

        branches = [appearance, geometry_feature]

        if self.camera_encoder is not None:
            if relative_c2w_normalized is None:
                raise ValueError("relative camera conditioning is enabled but no relative cameras were provided")
            # [B, V, 4, 4] -> [B, V, D] -> expand -> [B, V, N, D]
            camera_feature = self.camera_encoder(relative_c2w_normalized)
            branches.append(camera_feature[:, :, None].expand_as(appearance))

        if self.input_ray_encoder is not None:
            if input_ray_maps is None:
                raise ValueError("input patch ray conditioning is enabled but no ray maps were provided")
            batch, views, channels, height, width = input_ray_maps.shape
            if channels != 6:
                raise ValueError("input_ray_maps must have shape [B,V,6,H,W]")

            ray_feature, ray_hw = self.input_ray_encoder(input_ray_maps.reshape(batch * views, 6, height, width))
            if ray_hw[0] * ray_hw[1] != appearance.shape[2]:
                raise ValueError("input ray patch grid does not match VGGT patch tokens")
            # [B*V, N, D] -> [B, V, N, D]
            branches.append(ray_feature.reshape(batch, views, appearance.shape[2], appearance.shape[3]))

        if self.modality_fusion is None:

            memory = self.input_norm(appearance + geometry_feature)
        else:

            # cat([B,V,N,D], [B,V,N,D], ...) -> [B,V,N, branch_count*D]
            # -> modality_fusion -> [B,V,N,D]
            fused = self.modality_fusion(torch.cat(branches, dim=-1))
            memory = self.input_norm(appearance + fused)

        # [B, V, N, D] -> [B, V*N, D]
        batch = memory.shape[0]
        return memory.reshape(batch, -1, memory.shape[-1]), valid.reshape(batch, -1)

    def forward(self, memory: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Run the forward pass."""

        queries = self.queries[None].expand(memory.shape[0], -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory, mask)

        return self.output_norm(queries)


class FullResolutionRayEncoder(nn.Module):
    """Encode dense query rays on the decoder patch grid."""

    def __init__(self, patch_size: int, dim: int = 512, fourier_bands: int = 6) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.fourier = FourierFeatures(fourier_bands)

        channels = 6 * self.fourier.multiplier

        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ray_map: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Run the forward pass."""
        height, width = ray_map.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Ray map {(height, width)} must be divisible by patch_size={self.patch_size}")

        # [B, 6, H, W] -> permute -> [B, H, W, 6]
        # -> Fourier -> [B, H, W, 6 * multiplier]
        # -> permute -> [B, 6 * multiplier, H, W]
        encoded = self.fourier(ray_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        # [B, channels, H, W] -> Conv2d -> [B, D, H/ps, W/ps]
        patches = self.patch_embed(encoded)
        patch_hw = patches.shape[-2:]

        # [B, D, H/ps, W/ps] -> flatten(2) -> [B, D, N_patches]
        # -> transpose(1,2) -> [B, N_patches, D]
        tokens = patches.flatten(2).transpose(1, 2)

        return self.norm(tokens), patch_hw


class FinalSkipFeatureBranch(nn.Module):
    """Refine one synthesized VGGT feature scale from final and skip states."""

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        *,
        heads: int,
        mlp_ratio: float,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("Each final+skip feature branch needs at least one refinement layer")

        self.final_norm = nn.LayerNorm(dim)
        self.skip_norm = nn.LayerNorm(dim)
        self.skip_projection = nn.Linear(dim, dim)
        self.skip_gate = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )
        self.feature_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.refinement = nn.ModuleList(
            [PerceiverLayer(dim, heads, mlp_ratio) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(feature_dim)),
        )

    def forward(
        self,
        final_rays: torch.Tensor,
        skip_rays: torch.Tensor,
        final_scene: torch.Tensor,
        scene_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final = self.final_norm(final_rays)
        skip = self.skip_norm(skip_rays)
        gate = self.skip_gate(torch.cat((final, skip), dim=-1))
        value = final + gate * self.skip_projection(skip)
        value = value + self.feature_embedding[None, None]

        for layer in self.refinement:
            value = layer(value, final_scene, scene_mask)

        value = self.output_norm(value)
        return self.output_head(value), value


class VGGTPatchFeatureDecoder(nn.Module):
    """Synthesize the four patch-feature levels consumed by frozen DenseHead."""

    def __init__(
        self,
        patch_size: int,
        feature_dims: Sequence[int],
        scene_dim: int = 512,
        dim: int = 512,
        layers: int = 16,
        capture_layers: Sequence[int] = (4, 8, 12, 16),
        state_conditioner_layers: int = 2,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        ray_fourier_bands: int = 6,
        hidden_dim: int = 1024,
        branch_layers: int = 2,
    ) -> None:
        super().__init__()
        capture_layers = tuple(int(value) for value in capture_layers)
        if len(feature_dims) != 4 or len(capture_layers) != 4:
            raise ValueError("Frozen VGGT DPT decoding requires exactly four feature layers")
        if layers < 1 or state_conditioner_layers < 0:
            raise ValueError("Decoder layer counts are invalid")
        if sorted(capture_layers) != list(capture_layers) or len(set(capture_layers)) != 4:
            raise ValueError("capture_layers must contain four increasing unique 1-based indices")
        if capture_layers[0] < 1 or capture_layers[-1] > layers:
            raise ValueError("capture_layers must lie within the decoder depth")
        if capture_layers[-1] != layers:
            raise ValueError("The last capture layer must equal the decoder depth for final+skip decoding")
        if dim % heads:
            raise ValueError("Decoder dimension must be divisible by attention heads")

        self.ray_encoder = FullResolutionRayEncoder(patch_size, dim, ray_fourier_bands)

        self.scene_adapter = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, dim))

        self.state_conditioner = nn.ModuleList(
            [AttentionBlock(dim, heads, mlp_ratio, cross_attention=False) for _ in range(state_conditioner_layers)]
        )

        self.layers = nn.ModuleList(
            [AttentionBlock(dim, heads, mlp_ratio, cross_attention=False) for _ in range(layers)]
        )

        self.capture_indices = tuple(value - 1 for value in capture_layers)

        self.feature_branches = nn.ModuleList(
            [
                FinalSkipFeatureBranch(
                    dim,
                    int(feature_dim),
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    hidden_dim=hidden_dim,
                    layers=branch_layers,
                )
                for feature_dim in feature_dims
            ]
        )

    def forward(
        self,
        ray_map: torch.Tensor,             # [B, 6, H, W]
        scene_memory: torch.Tensor,        # [B, K, scene_dim]
        scene_mask: torch.Tensor | None,   # [B, K] | None
    ) -> tuple[list[torch.Tensor], tuple[int, int], list[torch.Tensor]]:
        """Run the forward pass."""

        # [B, 6, H, W] -> [B, N_q, D], patch_hw
        rays, patch_hw = self.ray_encoder(ray_map)

        # [B, K, scene_dim] -> [B, K, D]
        scene = self.scene_adapter(scene_memory)

        for layer in self.state_conditioner:
            scene = layer(scene, context_mask=scene_mask)

        # tokens: [B, K+N_q, D]
        scene_length = scene.shape[1]
        tokens = torch.cat((scene, rays), dim=1)

        joint_mask = None
        if scene_mask is not None:
            ray_mask = torch.ones(rays.shape[:2], device=rays.device, dtype=torch.bool)
            joint_mask = torch.cat((scene_mask.bool(), ray_mask), dim=1)

        captured = []
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens, context_mask=joint_mask)
            if index in self.capture_indices:
                captured.append(tokens[:, scene_length:])

        if len(captured) != 4:
            raise RuntimeError("Feature decoder did not capture all four requested layers")

        final_rays = captured[-1]
        final_scene = tokens[:, :scene_length]
        decoded = [
            branch(final_rays, skip, final_scene, scene_mask)
            for branch, skip in zip(self.feature_branches, captured)
        ]
        features = [feature for feature, _ in decoded]
        branch_states = [state for _, state in decoded]

        # [B, N_q, D] -> transpose -> [B, D, N_q] -> reshape -> [B, D, H_p, W_p]
        feature_maps = [
            value.transpose(1, 2).reshape(value.shape[0], value.shape[-1], *patch_hw)
            for value in branch_states
        ]
        return features, patch_hw, feature_maps
