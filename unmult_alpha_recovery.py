"""Necessary-only alpha recovery for bounded straight RGBA (gray17, 2026-10-09).

The over-range reference is internal. This module registers no nodes and performs
no mask blur, dilation, or neighbor recoloring.
"""

from __future__ import annotations

import torch

GRAY_ANCHOR = 56 / 255
ALPHA_INCREASE_CAP = 43 / 255
_NECESSARY_EPS = 1e-7


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    maximum, index = rgb.max(-1)
    minimum = rgb.amin(-1)
    delta = maximum - minimum
    safe = delta.clamp_min(1e-12)
    r, g, b = rgb.unbind(-1)
    sectors = torch.stack(((g - b) / safe, 2 + (b - r) / safe, 4 + (r - g) / safe), -1)
    hue = sectors.gather(-1, index[..., None])[..., 0].div(6).remainder(1)
    hue = torch.where(delta > 0, hue, torch.zeros_like(hue))
    saturation = torch.where(
        maximum > 0, delta / maximum.clamp_min(1e-12), torch.zeros_like(maximum)
    )
    return torch.stack((hue, saturation, maximum), -1)


def _code_y(rgb: torch.Tensor) -> torch.Tensor:
    """Weighted encoded values, not linear-light luminance."""
    return (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)


def _eligible_info(premult: torch.Tensor, alpha: torch.Tensor):
    raw = premult / alpha.clamp_min(1 / 255)[..., None]
    severe = ((premult < -2 / 255) & (raw < -0.25)).any(-1)
    eligible = (alpha > 0) & ~severe & ((premult.amax(-1) - alpha) * 255 > 0.50001)
    return raw, severe, eligible


def _power_reference(premult: torch.Tensor, alpha: torch.Tensor, base: torch.Tensor):
    """Frozen power_k1_h8_s10 reference, including the 23-step search."""
    raw, _, eligible = _eligible_info(premult, alpha)
    result = base.clone()
    if eligible.any():
        peak = raw[eligible].clamp_min(0).amax(-1)
        q = base[eligible]
        original = _rgb_to_hsv(q)
        requested = peak.pow(-1.0).clamp_min(1e-5)
        lo = requested.clone()
        hi = torch.ones_like(lo)

        def valid(exponent):
            hsv = _rgb_to_hsv(q.pow(exponent[:, None]))
            hue_difference = ((hsv[:, 0] - original[:, 0] + 0.5).remainder(1) - 0.5).abs() * 360
            stable = original[:, 1] > 0.01
            return ((~stable) | (hue_difference <= 8.0)) & (hsv[:, 1] >= original[:, 1] - 0.10)

        for _ in range(23):
            mid = (lo + hi) * 0.5
            accepted = valid(mid)
            hi = torch.where(accepted, mid, hi)
            lo = torch.where(accepted, lo, mid)
        exponent = torch.where(valid(requested), requested, hi)
        result[eligible] = q.pow(exponent[:, None])
    return result


def _target_extra(premult: torch.Tensor, alpha: torch.Tensor, base: torch.Tensor):
    """Private reference contribution; never returned as an output layer."""
    _, _, eligible = _eligible_info(premult, alpha)
    reference = _power_reference(premult, alpha, base)
    extra = torch.zeros_like(base)
    if eligible.any():
        q = base[eligible]
        scale = (_code_y(reference[eligible]) / _code_y(q).clamp_min(1e-12)).clamp_min(1)
        total = q * scale[:, None]
        extra[eligible] = (total - q) * alpha[eligible][:, None]
    return extra


def recover_gray17(
    premult: torch.Tensor, alpha: torch.Tensor, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded RGB and necessary local alpha, increasing by at most 43/255.

    Noneligible and already-feasible pixels retain Method 1. Unknown background
    is bypassed by the caller. Residual unrepresentable RGB is clipped at the cap.
    """
    _, _, eligible = _eligible_info(premult, alpha)
    extra = _target_extra(premult, alpha, base)
    target_rgb = base + extra / alpha[..., None].clamp_min(1e-12)
    target = target_rgb * alpha[..., None] + GRAY_ANCHOR * (1 - alpha[..., None])
    needed = torch.maximum(
        ((target - GRAY_ANCHOR) / (1 - GRAY_ANCHOR)).amax(-1),
        (1 - target / GRAY_ANCHOR).amax(-1),
    )
    active = eligible & (alpha > 0) & (alpha < 1) & (needed > alpha + _NECESSARY_EPS)
    adjusted_alpha = torch.where(
        active,
        torch.maximum(alpha, torch.minimum(needed, (alpha + ALPHA_INCREASE_CAP).clamp_max(1))),
        alpha,
    )
    rgb = base.clone()
    if active.any():
        rgb[active] = (
            (target[active] - GRAY_ANCHOR * (1 - adjusted_alpha[active, None]))
            / adjusted_alpha[active, None]
        ).clamp(0, 1)
    return rgb, adjusted_alpha
