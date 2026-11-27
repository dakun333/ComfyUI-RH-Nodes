"""Fixed-Alpha foreground color recovery in input encoded RGB space."""
import numpy as np
import torch
from pymatting import estimate_foreground_ml
from .vendor.mekajiki_unmult.core import MekajikiMaskedUnMult


def recover_foreground_rgb(rgb, alpha, method="multilevel"):
    if method == "legacy_q":
        rgba, _ = MekajikiMaskedUnMult().apply(torch.from_numpy(rgb)[None], torch.from_numpy(alpha)[None])
        return rgba[0, ..., :3].numpy()
    if method != "multilevel":
        raise ValueError("Unsupported foreground color recovery method")
    # Joint local foreground/background estimation; no alpha recomputation.
    foreground = estimate_foreground_ml(rgb, alpha)
    foreground[alpha >= 1.] = rgb[alpha >= 1.]
    foreground[alpha <= 0.] = 0.
    return np.clip(foreground, 0., 1.)
