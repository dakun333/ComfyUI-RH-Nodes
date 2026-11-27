"""Compositing conventions: sRGB display images, linear-light RGB, straight alpha."""
import numpy as np


def linear_to_srgb(x):
    x = np.asarray(x, dtype=np.float32)
    absolute = np.abs(x)
    result = np.where(absolute <= 0.0031308, absolute * 12.92,
                      1.055 * np.maximum(absolute, 0.0031308)**(1.0 / 2.4) - 0.055)
    return np.copysign(result, x).astype(np.float32, copy=False)


def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    absolute = np.abs(x)
    result = np.where(absolute <= 0.04045, absolute / 12.92,
                      ((absolute + 0.055) / 1.055)**2.4)
    return np.copysign(result, x).astype(np.float32, copy=False)


def checker_preview(fg_linear, alpha, square=32):
    height, width = alpha.shape[-2:]
    yy, xx = np.ogrid[:height, :width]
    grid = np.where(((yy // square + xx // square) % 2) == 0, 0.35, 0.65).astype(np.float32)
    bg = srgb_to_linear(grid)[..., None]
    linear = fg_linear * alpha[..., None] + bg * (1.0 - alpha[..., None])
    return np.clip(linear_to_srgb(linear), 0.0, 1.0)
