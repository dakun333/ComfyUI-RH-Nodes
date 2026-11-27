"""Output-resolution alpha cleanup; never change the predicted straight RGB."""
import math

import cv2
import numpy as np


def validate_alpha_settings(alpha_shrink, alpha_feather):
    if not math.isfinite(alpha_shrink) or alpha_shrink != int(alpha_shrink) or not 0 <= alpha_shrink <= 64:
        raise ValueError("alpha_shrink must be an integer pixel radius between 0 and 64")
    if not math.isfinite(alpha_feather) or not 0 <= alpha_feather <= 64:
        raise ValueError("alpha_feather must be a finite Gaussian sigma between 0 and 64 pixels")


def adjust_alpha(alpha, alpha_shrink=0, alpha_feather=0.0):
    if alpha_shrink:
        radius = int(alpha_shrink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        alpha = cv2.erode(alpha, kernel, borderType=cv2.BORDER_REPLICATE)
    if alpha_feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(alpha_feather),
                                 sigmaY=float(alpha_feather), borderType=cv2.BORDER_REPLICATE)
        np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha
