"""Paper-facing depth metrics shared across datasets."""

from __future__ import annotations

import torch


@torch.no_grad()
def depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    """Compute log-L1, AbsRel, delta1, valid pixels, and coverage."""
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    values = (
        valid
        & torch.isfinite(target)
        & (target > 0)
        & torch.isfinite(prediction)
        & (prediction > 0)
    )
    if not values.any():
        return {
            "log_l1": float("inf"),
            "abs_rel": float("inf"),
            "delta1": float("nan"),
            "valid_pixels": 0,
            "coverage": 0.0,
        }
    log_l1 = (
        torch.log(prediction[values].clamp_min(1e-6))
        - torch.log(target[values])
    ).abs().mean()
    abs_rel = ((prediction[values] - target[values]).abs() / target[values]).mean()
    ratio = torch.maximum(
        prediction[values] / target[values].clamp_min(1e-8),
        target[values] / prediction[values].clamp_min(1e-8),
    )
    return {
        "log_l1": float(log_l1),
        "abs_rel": float(abs_rel),
        "delta1": float(ratio.lt(1.25).float().mean()),
        "valid_pixels": int(values.sum()),
        "coverage": float(values.float().mean()),
    }

