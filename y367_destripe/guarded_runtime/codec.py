"""Isolated, numerically unchanged historical directed-8 encoder."""
import numpy as np
try:
    from .native_output import directed_round as _native_directed_round
except ImportError:
    _native_directed_round=None

def quantize(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(np.rint(image), 0, 255))

def directed_round(target: np.ndarray, enabled: np.ndarray,
                   fallback: np.ndarray) -> np.ndarray:
    """First-order error feedback along y, only inside enabled runs.

    Each result is floor(target) or ceil(target), hence <1 RGB level from the
    floating-point target. A deterministic bit-reversed column phase decorrelates
    the counters. There is no PRNG. This trades coherent rounding bands for faint
    fine quantization texture; it is NOT recovery of unknown original decimals.
    The counter resets at disabled pixels and never writes into protected areas.
    """
    if _native_directed_round is not None and target.dtype == np.float32:
        return _native_directed_round(target, enabled, fallback)
    h, w, _ = target.shape
    out = fallback.copy()
    indices = np.arange(w, dtype=np.uint32) % 256
    phase = np.zeros(w, np.float32)
    for bit in range(8):
        phase += ((indices >> bit) & 1) * 2. ** (-bit - 1)
    phase = phase[:, None] - 0.5
    error = np.broadcast_to(phase, (w, 3)).copy()
    for y in range(h):
        row = target[y]
        value = row + error
        rounded = np.clip(np.floor(value + 0.5), np.floor(row), np.ceil(row))
        out[y, enabled[y]] = rounded[enabled[y]].astype(np.uint8)
        error = np.where(enabled[y, :, None], value - rounded, phase)
    return out

def encode_directed8(original, target):
    """Exact historical encode_ablation(..., 'directed8') numeric path."""
    target = np.clip(target, 0, 255)
    nearest = quantize(target)
    enabled = np.max(np.abs(original.astype(np.float32) - target), axis=2) > 1e-5
    return directed_round(target, enabled, nearest)
