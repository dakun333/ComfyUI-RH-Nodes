"""One-time linear alpha levels, performed on native assets before resampling."""
from numbers import Integral
import numpy as np


def alpha_levels(alpha, black=3, white=252):
    if alpha.dtype != np.uint8:
        raise ValueError('Alpha levels requires uint8 input')
    if (isinstance(black, bool) or isinstance(white, bool) or
        not isinstance(black, Integral) or not isinstance(white, Integral) or
        not 0 <= black < white <= 255):
        raise ValueError('Require integer levels 0 <= black < white <= 255')
    # Deterministic half-up integer rounding, no float/tie-to-even dependency.
    numerator = np.clip(np.arange(256, dtype=np.int32)-black, 0, white-black)*255
    lut = ((2*numerator+(white-black))//(2*(white-black))).astype(np.uint8)
    return lut[alpha]


def normalize_alpha(alpha, metadata, cfg, stage):
    """Return new alpha and copied metadata; refuse accidental double levels.

    Output files carry the source-grid normalization stamp. Reopening them for
    compositing or calibration must not expand their midtones for a second time.
    """
    meta = dict(metadata)
    if not cfg.alpha_levels:
        return alpha.copy(), meta
    existing = meta.get('alpha_normalization')
    if existing:
        if existing.get('method') != 'linear_levels' or existing.get('black') != cfg.alpha_black or existing.get('white') != cfg.alpha_white:
            raise ValueError('Asset already normalized with different levels; rerun from original assets')
        return alpha.copy(), meta
    normalized = alpha_levels(alpha, cfg.alpha_black, cfg.alpha_white)
    meta['alpha_normalization'] = {
        'method': 'linear_levels', 'black': cfg.alpha_black, 'white': cfg.alpha_white,
        'rounding': 'half_up', 'stage': stage, 'input_grid': [alpha.shape[1], alpha.shape[0]],
        'changed_pixels': int(np.count_nonzero(normalized != alpha)),
        'to_zero_pixels': int(np.count_nonzero((alpha > 0) & (normalized == 0))),
        'to_opaque_pixels': int(np.count_nonzero((alpha < 255) & (normalized == 255))),
        'midrange_changed_pixels': int(np.count_nonzero((normalized != alpha) & (normalized > 0) & (normalized < 255)))}
    return normalized, meta
