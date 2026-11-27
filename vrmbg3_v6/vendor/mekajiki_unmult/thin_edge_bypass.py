"""Mask-only thin-edge geometry and near-black exception for Mask-Guided only.

This is a whole-sample product policy, NOT semantic hair/light classification.
Mask values and RGB are never edited here. CPU uses NumPy views; other devices
transfer at most two boolean maps, not RGB. OpenCV supplies exact Euclidean
distances. No process-wide OpenCV or PyTorch thread settings are changed.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
import cv2
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
    """Long side 1K/2K/4K -> 4/8/16px; native crops retain a 4px floor."""
    return max(4.0, 4.0 * max(shape) / 1024.0)

def is_near_black_background(background: torch.Tensor | None) -> bool:
    """Only a known background with EVERY channel in [0, 16/255] is exempt.

    This is not weighted luminance, and not the estimator's unrelated 64-code
    dark-cluster threshold. None must not silently become an assumed black.
    """
    if background is None:
        return False
    return bool((torch.isfinite(background) & (background >= 0.0) & (background <= NEAR_BLACK_MAX)).all().item())

def detect_thin_edge(alpha: torch.Tensor) -> ThinEdgeDecision:
    """Inspect one already-normalized HW Mask without editing it.

    Effective semi: 8/255 < A < .98. Exterior: A <= 8/255, irrespective of RGB.
    At least 80% of effective semi pixels must lie within the scaled Euclidean
    radius of exterior evidence. No semi or no exterior means no thin-edge
    decision. The caller separately enforces the near-black exception.
    """
    radius = scaled_radius(tuple(alpha.shape))
    if alpha.device.type == 'cpu':
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
        exterior, semi = (exterior.cpu().numpy(), semi.cpu().numpy())
    distance = cv2.distanceTransform((~exterior).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    hits = int(np.count_nonzero(semi & (distance <= radius)))
    return ThinEdgeDecision(hits * 5 >= count * 4, count, hits, radius)
