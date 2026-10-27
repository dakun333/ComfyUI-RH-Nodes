"""Mask-only whole-sample routing; not semantic hair or light classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ALPHA_LOW = 8.0 / 255.0
ALPHA_HIGH = 0.98
NEAR_BLACK_MAX = 16.0 / 255.0


@dataclass(frozen=True)
class ThinEdgeDecision:
    thin_edge_dominant: bool
    effective_semi_pixels: int
    near_low_alpha_pixels: int
    radius_px: float

    @property
    def coverage(self) -> float | None:
        if not self.effective_semi_pixels:
            return None
        return self.near_low_alpha_pixels / self.effective_semi_pixels


def scaled_radius(shape: tuple[int, int]) -> float:
    """Long side 1K/2K/4K gives 4/8/16 pixels, with a four-pixel floor."""
    return max(4.0, 4.0 * max(shape) / 1024.0)


@torch.no_grad()
def is_near_black_background(background: torch.Tensor | None) -> bool:
    """Known background with every channel in [0,16/255]; None is unknown."""
    if background is None:
        return False
    return bool(
        (torch.isfinite(background) & (background >= 0.0)
         & (background <= NEAR_BLACK_MAX)).all().item()
    )


@torch.no_grad()
def detect_thin_edge(alpha: torch.Tensor) -> ThinEdgeDecision:
    """Require 80% of effective semi pixels to be near low-alpha exterior."""
    radius = scaled_radius(tuple(alpha.shape))
    if alpha.device.type == "cpu":
        import numpy as np

        a = alpha.detach().numpy()
        semi = (a > ALPHA_LOW) & (a < ALPHA_HIGH)
        count = int(np.count_nonzero(semi))
        if not count:
            return ThinEdgeDecision(False, 0, 0, radius)
        exterior = a <= ALPHA_LOW
        if not exterior.any():
            return ThinEdgeDecision(False, count, 0, radius)
    else:
        semi = (alpha > ALPHA_LOW) & (alpha < ALPHA_HIGH)
        count = int(torch.count_nonzero(semi).item())
        if not count:
            return ThinEdgeDecision(False, 0, 0, radius)
        exterior = alpha <= ALPHA_LOW
        if not bool(exterior.any().item()):
            return ThinEdgeDecision(False, count, 0, radius)
        exterior, semi = exterior.cpu().numpy(), semi.cpu().numpy()

    import cv2
    import numpy as np

    distance = cv2.distanceTransform(
        (~exterior).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    hits = int(np.count_nonzero(semi & (distance <= radius)))
    return ThinEdgeDecision(hits * 5 >= count * 4, count, hits, radius)
