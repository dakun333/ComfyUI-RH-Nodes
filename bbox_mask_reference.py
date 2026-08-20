"""ComfyUI node that turns an aligned image + BBOX mask into a 1M reference image."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


STEP = 16
DEFAULT_AREA = 1024 * 1024
RED = (1.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)


def _round_nearest_step(value: float, step: int = STEP) -> int:
    return max(step, int(math.floor(value / step + 0.5)) * step)


def _round_down_step(value: float, step: int = STEP) -> int:
    return max(step, int(math.floor(value / step)) * step)


def _round_up_step(value: int, step: int = STEP) -> int:
    """Round an integer up so a BBOX can contain, rather than crop, content."""
    return max(step, int(math.ceil(value / step)) * step)


def _aligned_canvas_size(
    source_width: int, source_height: int, target_area: int, allow_upscale: bool
) -> tuple[int, int]:
    """Return same-aspect dimensions near target_area, both divisible by 16."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("image has an invalid size")
    scale = math.sqrt(max(1, target_area) / (source_width * source_height))
    if not allow_upscale:
        scale = min(1.0, scale)
    if scale >= 1.0:
        width = _round_nearest_step(source_width * scale)
        height = _round_nearest_step(source_height * scale)
    else:
        width = _round_down_step(source_width * scale)
        height = _round_down_step(source_height * scale)
    if not allow_upscale:
        width = min(width, max(STEP, (source_width // STEP) * STEP))
        height = min(height, max(STEP, (source_height // STEP) * STEP))
    return width, height


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    """Scale a source mask BBOX to a 16-aligned box without cropping content.

    The old nearest-16 calculation could round a scaled white-mask extent down.
    Here the continuous source extent is first conservatively converted to
    output pixels (floor at the start, ceil at the exclusive end). The final
    width/height are then rounded *up* to 16 and the frame is shifted only as
    needed to stay on canvas. Thus the selected content always remains inside
    the red-frame interior; any alignment adjustment adds context outside it.
    """
    x, y, width, height = bbox
    scale_x = output_width / source_width
    scale_y = output_height / source_height

    content_x0 = max(0, min(output_width, int(math.floor(x * scale_x))))
    content_y0 = max(0, min(output_height, int(math.floor(y * scale_y))))
    content_x1 = max(0, min(output_width, int(math.ceil((x + width) * scale_x))))
    content_y1 = max(0, min(output_height, int(math.ceil((y + height) * scale_y))))
    if content_x1 <= content_x0 or content_y1 <= content_y0:
        raise ValueError(f"scaled BBOX is empty: {bbox}")

    final_width = min(output_width, _round_up_step(content_x1 - content_x0))
    final_height = min(output_height, _round_up_step(content_y1 - content_y0))

    # Keep the nominal scaled origin whenever it already contains the content.
    # Otherwise move only enough to retain the entire content extent, then
    # clamp against the output canvas for BBOXs that touch an image edge.
    nominal_x = int(round(x * scale_x))
    nominal_y = int(round(y * scale_y))
    min_x_to_contain = content_x1 - final_width
    min_y_to_contain = content_y1 - final_height
    final_x = min(max(nominal_x, min_x_to_contain), content_x0)
    final_y = min(max(nominal_y, min_y_to_contain), content_y0)
    final_x = max(0, min(final_x, output_width - final_width))
    final_y = max(0, min(final_y, output_height - final_height))

    if not (
        final_x <= content_x0
        and final_y <= content_y0
        and final_x + final_width >= content_x1
        and final_y + final_height >= content_y1
    ):
        raise AssertionError("16-aligned BBOX would crop mask content")
    return final_x, final_y, final_width, final_height


def _as_bhwc(image: torch.Tensor, name: str) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must be HWC or BHWC, got {tuple(image.shape)}")
    if image.shape[-1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    return image.float().clamp(0.0, 1.0)


def _match_batch(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.shape[0] == mask.shape[0]:
        return image, mask
    if image.shape[0] == 1:
        return image.repeat(mask.shape[0], 1, 1, 1), mask
    if mask.shape[0] == 1:
        return image, mask.repeat(image.shape[0], 1, 1, 1)
    raise ValueError(f"image batch {image.shape[0]} does not match bbox_mask batch {mask.shape[0]}")


def _mask_bbox(mask: torch.Tensor, threshold: float) -> tuple[int, int, int, int]:
    """Find one XYWH BBOX around every white mask pixel in an HWC mask image."""
    luminance = mask[..., : min(3, mask.shape[-1])].mean(dim=-1)
    ys, xs = torch.where(luminance >= threshold)
    if xs.numel() == 0:
        raise ValueError("bbox_mask has no white pixels at the configured threshold")
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _paint_outline(
    image: torch.Tensor,
    left: int,
    top: int,
    right: int,
    bottom: int,
    thickness: int,
    color: tuple[float, float, float],
) -> None:
    """Paint an inward outline on a single HWC image, clipping at canvas edges."""
    height, width = image.shape[:2]
    color_tensor = image.new_tensor(color)

    def fill(x0: int, y0: int, x1: int, y1: int) -> None:
        x0, x1 = max(0, x0), min(width - 1, x1)
        y0, y1 = max(0, y0), min(height - 1, y1)
        if x0 <= x1 and y0 <= y1:
            image[y0 : y1 + 1, x0 : x1 + 1, :3] = color_tensor

    fill(left, top, right, top + thickness - 1)
    fill(left, bottom - thickness + 1, right, bottom)
    fill(left, top, left + thickness - 1, bottom)
    fill(right - thickness + 1, top, right, bottom)


def _draw_bbox_topmost(base: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    """Draw the outer white / inner red frame while keeping BBOX interior exact."""
    height, width = base.shape[:2]
    x, y, box_width, box_height = bbox
    x1, y1 = x + box_width - 1, y + box_height - 1
    frame_scale = math.sqrt(width * height / DEFAULT_AREA)
    red_width = max(2, round(frame_scale * 8))
    white_width = max(1, round(frame_scale * 4))

    result = base.clone()
    _paint_outline(
        result,
        x - red_width - white_width,
        y - red_width - white_width,
        x1 + red_width + white_width,
        y1 + red_width + white_width,
        white_width,
        WHITE,
    )
    _paint_outline(
        result,
        x - red_width,
        y - red_width,
        x1 + red_width,
        y1 + red_width,
        red_width,
        RED,
    )
    # Never let a rasterized outline affect pixels represented by the mask.
    result[y : y1 + 1, x : x1 + 1, :3] = base[y : y1 + 1, x : x1 + 1, :3]
    return result


class BBoxMaskToReferenceImage:
    """Build a 1M / 16-aligned red-frame reference image from image + mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_mask": ("IMAGE", {"tooltip": "Aligned black/white mask image: white pixels define the BBOX."}),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "target_area": (
                    "INT",
                    {"default": DEFAULT_AREA, "min": STEP * STEP, "max": 64 * DEFAULT_AREA, "step": STEP * STEP},
                ),
                "allow_upscale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("reference_image", "bbox_crop")
    FUNCTION = "make_reference"
    CATEGORY = "BBOX Tools"
    DESCRIPTION = (
        "Input an original image and a same-size black/white BBOX mask. "
        "Outputs a roughly 1M reference image with a 16-aligned BBOX and an exact crop of its BBOX interior."
    )

    def make_reference(
        self,
        image: torch.Tensor,
        bbox_mask: torch.Tensor,
        mask_threshold: float = 0.5,
        target_area: int = DEFAULT_AREA,
        allow_upscale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = _as_bhwc(image, "image")
        bbox_mask = _as_bhwc(bbox_mask, "bbox_mask")
        image, bbox_mask = _match_batch(image, bbox_mask)
        if image.shape[1:3] != bbox_mask.shape[1:3]:
            raise ValueError(
                "image and bbox_mask must have exactly the same HxW dimensions; "
                f"got {tuple(image.shape[1:3])} and {tuple(bbox_mask.shape[1:3])}"
            )

        source_height, source_width = image.shape[1:3]
        output_width, output_height = _aligned_canvas_size(
            source_width, source_height, int(target_area), bool(allow_upscale)
        )
        resized = F.interpolate(
            image[..., :3].permute(0, 3, 1, 2),
            size=(output_height, output_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1).contiguous()

        references: list[torch.Tensor] = []
        crops: list[torch.Tensor] = []
        for batch_index in range(resized.shape[0]):
            source_bbox = _mask_bbox(bbox_mask[batch_index], float(mask_threshold))
            final_bbox = _scale_bbox(
                source_bbox,
                source_width,
                source_height,
                output_width,
                output_height,
            )
            x, y, box_width, box_height = final_bbox
            # Crop from the same already-resized base that is used to make the
            # reference. _draw_bbox_topmost restores this exact interior after
            # drawing, so bbox_crop can be pasted back with no pixel change.
            crop = resized[batch_index, y : y + box_height, x : x + box_width, :].clone()
            reference = _draw_bbox_topmost(resized[batch_index], final_bbox)
            if not torch.equal(reference[y : y + box_height, x : x + box_width, :], crop):
                raise AssertionError("BBOX frame changed crop pixels")
            references.append(reference)
            crops.append(crop)

        crop_sizes = {tuple(crop.shape[:2]) for crop in crops}
        if len(crop_sizes) != 1:
            raise ValueError(
                "batch samples produced different BBOX crop sizes; process them separately "
                "so ComfyUI can represent bbox_crop as one IMAGE batch"
            )
        return torch.stack(references, dim=0), torch.stack(crops, dim=0)


NODE_CLASS_MAPPINGS = {
    "BBoxMaskToReferenceImage": BBoxMaskToReferenceImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBoxMaskToReferenceImage": "BBOX: Mask → 1M Reference Image",
}
