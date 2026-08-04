"""Pure-OpenCV comic and manga outline extraction for ComfyUI IMAGE tensors."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


def _keep_large_components(
    mask: np.ndarray, min_area: int, *, keep_largest: bool = False
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary, dtype=np.uint8)
    if count <= 1:
        return output

    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest:
        output[labels == int(np.argmax(areas)) + 1] = 255
        return output

    for label, area in enumerate(areas, start=1):
        if area >= min_area:
            output[labels == label] = 255
    return output


def _foreground_mask(
    rgba: np.ndarray, alpha_threshold: int, background_threshold: float
) -> np.ndarray:
    """Use alpha when available, otherwise estimate background from image borders."""
    rgb = rgba[..., :3].astype(np.uint8)
    alpha = rgba[..., 3].astype(np.uint8)
    height, width = alpha.shape

    if alpha.min() < 250:
        mask = (alpha > alpha_threshold).astype(np.uint8) * 255
        mask = _keep_large_components(mask, max(20, int(height * width * 0.00003)))
    else:
        border = np.concatenate(
            (rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0
        ).astype(np.uint8)
        background_rgb = np.median(border, axis=0).astype(np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2LAB).astype(
            np.float32
        )[0, 0]
        delta = lab - background_lab
        distance = np.sqrt(
            (0.65 * delta[..., 0]) ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
        )
        distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
        otsu_threshold, _ = cv2.threshold(
            distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = max(background_threshold, float(otsu_threshold) * 0.55)
        mask = (distance > threshold).astype(np.uint8) * 255
        mask = _keep_large_components(mask, max(50, int(height * width * 0.00015)))

    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_5, iterations=1)
    return cv2.dilate(mask, kernel_3, iterations=1)


def _auto_canny(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = gray[mask > 0] if np.any(mask) else gray.reshape(-1)
    median = float(np.median(values))
    lower = int(max(0, (1.0 - 0.28) * median))
    upper = int(min(255, (1.0 + 0.28) * median))
    if upper <= lower + 10:
        lower = max(0, upper - 40)
    return cv2.Canny(gray, lower, upper, L2gradient=True)


def _remove_edge_noise(edges: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (edges > 0).astype(np.uint8), 8
    )
    output = np.zeros_like(edges, dtype=np.uint8)
    for label in range(1, count):
        _, _, width, height, area = stats[label]
        if area >= min_area or max(width, height) >= 12:
            output[labels == label] = 255
    return output


def _outline_from_rgba(
    rgba: np.ndarray,
    *,
    alpha_threshold: int,
    line_width: int,
    edge_percentile: float,
    min_edge_area: Optional[int],
    background_threshold: float,
    add_outer_contour: bool,
    invert: bool,
) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.uint8)
    alpha = rgba[..., 3].astype(np.uint8)
    height, width = alpha.shape
    alpha_float = alpha.astype(np.float32) / 255.0
    composited = (
        rgb.astype(np.float32) * alpha_float[..., None]
        + 255.0 * (1.0 - alpha_float[..., None])
    ).astype(np.uint8)

    mask = _foreground_mask(rgba, alpha_threshold, background_threshold)
    edge_mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    smoothed = cv2.bilateralFilter(composited, d=7, sigmaColor=45, sigmaSpace=7)
    smoothed = cv2.medianBlur(smoothed, 3)

    luminance_edges = _auto_canny(
        cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY), edge_mask
    )
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
    magnitude = np.zeros((height, width), dtype=np.float32)
    for channel in range(3):
        gradient_x = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0)
        gradient_y = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1)
        magnitude += gradient_x * gradient_x + gradient_y * gradient_y
    magnitude = np.sqrt(magnitude)
    valid_values = magnitude[edge_mask > 0]
    threshold_values = valid_values if valid_values.size else magnitude.reshape(-1)
    color_edges = (
        magnitude > np.percentile(threshold_values, edge_percentile)
    ).astype(np.uint8) * 255

    if add_outer_contour:
        outer_edges = cv2.morphologyEx(
            mask,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    else:
        outer_edges = np.zeros_like(mask)

    edges = cv2.bitwise_or(luminance_edges, color_edges)
    edges = cv2.bitwise_or(edges, outer_edges)
    edges = cv2.bitwise_and(edges, edge_mask)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    effective_min_area = (
        max(8, int(height * width * 0.000006))
        if min_edge_area is None
        else min_edge_area
    )
    edges = _remove_edge_noise(edges, effective_min_area)
    if line_width > 1:
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_width, line_width)),
            iterations=1,
        )
    if invert:
        return edges

    output = np.full_like(edges, 255)
    output[edges > 0] = 0
    return output


def _as_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a ComfyUI IMAGE torch.Tensor")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "image must use BHWC layout with 1, 3, or 4 channels; "
            f"received {tuple(image.shape)}"
        )
    return image.detach().float().cpu().clamp(0.0, 1.0)


def _to_rgba(image: torch.Tensor) -> np.ndarray:
    if image.shape[-1] == 1:
        rgb = image.repeat(1, 1, 3)
        alpha = torch.ones((*image.shape[:2], 1), dtype=image.dtype)
    else:
        rgb = image[..., :3]
        alpha = image[..., 3:4] if image.shape[-1] == 4 else torch.ones(
            (*image.shape[:2], 1), dtype=image.dtype
        )
    return torch.cat((rgb, alpha), dim=-1).mul(255).round().byte().numpy()


class ComicOutlineDetect:
    """Extract clean black-and-white comic outlines without a model download."""

    DESCRIPTION = (
        "Fuses luminance, LAB colour-boundary, and optional foreground-outline edges. "
        "Works entirely with OpenCV and supports IMAGE batches."
    )
    CATEGORY = "🎨 漫画轮廓"
    FUNCTION = "detect"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("轮廓图",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "line_width": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                "edge_percentile": (
                    "FLOAT", {"default": 90.0, "min": 75.0, "max": 99.0, "step": 0.5}
                ),
                "alpha_thr": (
                    "INT", {"default": 24, "min": 0, "max": 255, "step": 1}
                ),
                "bg_threshold": (
                    "FLOAT", {"default": 18.0, "min": 1.0, "max": 100.0, "step": 1.0}
                ),
                "invert": ("BOOLEAN", {"default": False}),
                "add_outer_contour": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "min_edge_area": (
                    "INT", {"default": -1, "min": -1, "max": 10000, "step": 1}
                )
            },
        }

    def detect(
        self,
        图像: torch.Tensor,
        line_width: int = 3,
        edge_percentile: float = 90.0,
        alpha_thr: int = 24,
        bg_threshold: float = 18.0,
        invert: bool = False,
        add_outer_contour: bool = True,
        min_edge_area: int = -1,
    ):
        min_area = None if min_edge_area < 0 else min_edge_area
        outlines = []
        for item in _as_image_batch(图像):
            outline = _outline_from_rgba(
                _to_rgba(item),
                alpha_threshold=alpha_thr,
                line_width=line_width,
                edge_percentile=edge_percentile,
                min_edge_area=min_area,
                background_threshold=bg_threshold,
                add_outer_contour=add_outer_contour,
                invert=invert,
            )
            outline_rgb = np.repeat(outline[..., None], 3, axis=-1).astype(np.float32)
            outlines.append(outline_rgb / 255.0)
        return (torch.from_numpy(np.stack(outlines)),)
