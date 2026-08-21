"""ComfyUI node that turns an aligned image + BBOX mask into a 1M reference image."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from PIL import Image


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


def _logical_canvas_size(
    source_width: int, source_height: int, target_area: int, allow_upscale: bool
) -> tuple[int, int]:
    """Return a near-target canvas that preserves aspect ratio but need not be 16-aligned.

    This is the logical (un-stretched) 1M canvas. The actual AI working canvas
    is separately rounded to 16-pixel multiples by _aligned_canvas_size.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError("image has an invalid size")
    scale = math.sqrt(max(1, target_area) / (source_width * source_height))
    if not allow_upscale:
        scale = min(1.0, scale)
    return max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))


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


def _scale_bbox_to_canvas(
    bbox: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    content_width: int,
    content_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Map a source BBOX through content scaling, then 16-align it on a canvas.

    content_width/height describe the scaled real image. canvas_width/height
    can be larger when gray padding is used. A final BBOX may extend into that
    padding, but it can never crop any scaled white-mask content.
    """
    x, y, width, height = bbox
    scale_x = content_width / source_width
    scale_y = content_height / source_height

    content_x0 = max(0, min(content_width, int(math.floor(x * scale_x))))
    content_y0 = max(0, min(content_height, int(math.floor(y * scale_y))))
    content_x1 = max(0, min(content_width, int(math.ceil((x + width) * scale_x))))
    content_y1 = max(0, min(content_height, int(math.ceil((y + height) * scale_y))))
    if content_x1 <= content_x0 or content_y1 <= content_y0:
        raise ValueError(f"scaled BBOX is empty: {bbox}")

    final_width = min(canvas_width, _round_up_step(content_x1 - content_x0))
    final_height = min(canvas_height, _round_up_step(content_y1 - content_y0))

    # Keep the nominal scaled origin whenever it contains the content. At a
    # padded right/bottom edge, the extra canvas may be used as harmless BBOX
    # context instead of shifting the BBOX inward over real image pixels.
    nominal_x = int(round(x * scale_x))
    nominal_y = int(round(y * scale_y))
    min_x_to_contain = content_x1 - final_width
    min_y_to_contain = content_y1 - final_height
    final_x = min(max(nominal_x, min_x_to_contain), content_x0)
    final_y = min(max(nominal_y, min_y_to_contain), content_y0)
    final_x = max(0, min(final_x, canvas_width - final_width))
    final_y = max(0, min(final_y, canvas_height - final_height))

    if not (
        final_x <= content_x0
        and final_y <= content_y0
        and final_x + final_width >= content_x1
        and final_y + final_height >= content_y1
    ):
        raise AssertionError("16-aligned BBOX would crop mask content")
    return final_x, final_y, final_width, final_height


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    """Stretch-mode compatibility wrapper for BBOX mapping."""
    return _scale_bbox_to_canvas(
        bbox,
        source_width,
        source_height,
        output_width,
        output_height,
        output_width,
        output_height,
    )


def _pad_bhwc_gray(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Append neutral gray pixels to the right/bottom without resampling image pixels."""
    batch, source_height, source_width, channels = image.shape
    if source_width > width or source_height > height:
        raise ValueError("gray pad canvas must not be smaller than its content")
    if source_width == width and source_height == height:
        return image
    result = torch.full((batch, height, width, channels), 0.5, device=image.device, dtype=image.dtype)
    result[:, :source_height, :source_width, :] = image
    return result


def _resize_bhwc_lanczos(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize RGB BHWC data with Pillow Lanczos, preserving batch and device."""
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"Lanczos resize expects BHWC RGB input, got {tuple(image.shape)}")
    if tuple(image.shape[1:3]) == (height, width):
        return image

    source_device = image.device
    source_dtype = image.dtype
    # ComfyUI IMAGE inputs are normally [0, 1]. Pillow's Lanczos is 8-bit,
    # which is appropriate for the PNG/JPEG image pipeline and yields visibly
    # sharper reductions than the previous bicubic working path.
    cpu_uint8 = image.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    cpu_uint8 = cpu_uint8.mul(255.0).round().to(torch.uint8).numpy()
    resized_items: list[np.ndarray] = []
    for item in cpu_uint8:
        pil_image = Image.fromarray(item, mode="RGB")
        resized_items.append(np.asarray(pil_image.resize((width, height), Image.Resampling.LANCZOS)).copy())
    resized = torch.from_numpy(np.stack(resized_items, axis=0)).to(device=source_device, dtype=source_dtype)
    return resized.div(255.0)


def _as_bhwc(image: torch.Tensor, name: str, clamp: bool = True) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"{name} must be HWC or BHWC, got {tuple(image.shape)}")
    if image.shape[-1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    image = image.float()
    return image.clamp(0.0, 1.0) if clamp else image


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
                "alignment_mode": (
                    ["stretch", "pad_gray"],
                    {"default": "stretch", "tooltip": "stretch: resize to 16 multiples; pad_gray: preserve logical aspect and append #808080 pixels to the right/bottom."},
                ),
            }
        }

    # The first two outputs deliberately retain their prior positions so
    # existing workflows keep working. crop_info is a workflow-only object;
    # crop_info_json is a readable/exportable copy of the same data.
    RETURN_TYPES = ("IMAGE", "IMAGE", "BBOX_CROP_INFO", "STRING", "BBOX_STRETCH_INFO")
    RETURN_NAMES = ("reference_image", "bbox_crop", "crop_info", "crop_info_json", "stretch_info")
    FUNCTION = "make_reference"
    CATEGORY = "BBOX Tools"

    def make_reference(
        self,
        image: torch.Tensor,
        bbox_mask: torch.Tensor,
        mask_threshold: float = 0.5,
        target_area: int = DEFAULT_AREA,
        allow_upscale: bool = True,
        alignment_mode: str = "stretch",
    ) -> tuple[torch.Tensor, torch.Tensor, dict, str, dict]:
        image = _as_bhwc(image, "image")
        bbox_mask = _as_bhwc(bbox_mask, "bbox_mask")
        image, bbox_mask = _match_batch(image, bbox_mask)
        if image.shape[1:3] != bbox_mask.shape[1:3]:
            raise ValueError(
                "image and bbox_mask must have exactly the same HxW dimensions; "
                f"got {tuple(image.shape[1:3])} and {tuple(bbox_mask.shape[1:3])}"
            )

        source_height, source_width = image.shape[1:3]
        logical_width, logical_height = _logical_canvas_size(
            source_width, source_height, int(target_area), bool(allow_upscale)
        )
        if alignment_mode not in {"stretch", "pad_gray"}:
            raise ValueError(f"unsupported alignment_mode: {alignment_mode!r}")
        if alignment_mode == "stretch":
            output_width, output_height = _aligned_canvas_size(
                source_width, source_height, int(target_area), bool(allow_upscale)
            )
            resized = _resize_bhwc_lanczos(image[..., :3], output_width, output_height)
            padding_ltrb = [0, 0, 0, 0]
        else:
            # Overall resize to the logical near-1M size preserves aspect. The
            # 16 alignment itself is then achieved only by adding #808080 pad.
            output_width, output_height = _round_up_step(logical_width), _round_up_step(logical_height)
            logical_image = _resize_bhwc_lanczos(image[..., :3], logical_width, logical_height)
            resized = _pad_bhwc_gray(logical_image, output_width, output_height)
            padding_ltrb = [0, 0, output_width - logical_width, output_height - logical_height]

        references: list[torch.Tensor] = []
        crops: list[torch.Tensor] = []
        info_items: list[dict[str, object]] = []
        for batch_index in range(resized.shape[0]):
            source_bbox = _mask_bbox(bbox_mask[batch_index], float(mask_threshold))
            final_bbox = _scale_bbox_to_canvas(
                source_bbox,
                source_width,
                source_height,
                logical_width if alignment_mode == "pad_gray" else output_width,
                logical_height if alignment_mode == "pad_gray" else output_height,
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
            info_items.append(
                {
                    "canvas_size_wh": [output_width, output_height],
                    "bbox_xywh": [x, y, box_width, box_height],
                    "crop_size_wh": [box_width, box_height],
                }
            )

        crop_sizes = {tuple(crop.shape[:2]) for crop in crops}
        if len(crop_sizes) != 1:
            raise ValueError(
                "batch samples produced different BBOX crop sizes; process them separately "
                "so ComfyUI can represent bbox_crop as one IMAGE batch"
            )
        crop_info = {"version": 1, "items": info_items}
        # The working canvas is 16-aligned for AI. logical_size_wh is the
        # near-1M, aspect-preserving size to which Restore can later un-stretch.
        # Source → working uses one resize only; logical_size is metadata, not
        # an extra pre-AI resize pass.
        stretch_info = {
            "version": 1,
            "source_size_wh": [source_width, source_height],
            "logical_size_wh": [logical_width, logical_height],
            "working_size_wh": [output_width, output_height],
            "target_area": int(target_area),
            "resize_filter": "lanczos",
            "alignment_mode": alignment_mode,
            "padding_ltrb": padding_ltrb,
            "align_corners": None,
            "antialias": True,
        }
        return (
            torch.stack(references, dim=0),
            torch.stack(crops, dim=0),
            crop_info,
            json.dumps(crop_info, ensure_ascii=False, separators=(",", ":")),
            stretch_info,
        )


def _parse_crop_info(crop_info: object) -> list[tuple[int, int, int, int, int, int]]:
    """Validate BBOX_CROP_INFO and return (canvas_w, canvas_h, x, y, w, h) items."""
    if not isinstance(crop_info, dict):
        raise ValueError("crop_info must be connected from BBOX: Mask → 1M Reference Image")
    if crop_info.get("version") != 1:
        raise ValueError(f"unsupported crop_info version: {crop_info.get('version')!r}")
    items = crop_info.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("crop_info contains no items")

    validated: list[tuple[int, int, int, int, int, int]] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"crop_info item {item_index} is invalid")
        canvas = item.get("canvas_size_wh")
        bbox = item.get("bbox_xywh")
        crop_size = item.get("crop_size_wh")
        if not (isinstance(canvas, (list, tuple)) and len(canvas) == 2):
            raise ValueError(f"crop_info item {item_index} has invalid canvas_size_wh")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError(f"crop_info item {item_index} has invalid bbox_xywh")
        if not (isinstance(crop_size, (list, tuple)) and len(crop_size) == 2):
            raise ValueError(f"crop_info item {item_index} has invalid crop_size_wh")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (*canvas, *bbox, *crop_size)):
            raise ValueError(f"crop_info item {item_index} must contain integer coordinates")
        canvas_width, canvas_height = canvas
        x, y, width, height = bbox
        crop_width, crop_height = crop_size
        if canvas_width <= 0 or canvas_height <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"crop_info item {item_index} has non-positive dimensions")
        if crop_width != width or crop_height != height:
            raise ValueError(f"crop_info item {item_index} crop_size_wh does not match bbox_xywh")
        if x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height:
            raise ValueError(f"crop_info item {item_index} BBOX lies outside its canvas")
        validated.append((canvas_width, canvas_height, x, y, width, height))
    return validated


def _background_rgb(background: str, custom_color: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    colors = {
        "alpha": (0.0, 0.0, 0.0),
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
    }
    if background == "custom":
        color_text = str(custom_color).strip().lstrip("#")
        if len(color_text) == 3:
            color_text = "".join(channel * 2 for channel in color_text)
        if len(color_text) != 6:
            raise ValueError("custom_color must be a hex RGB color such as #RRGGBB")
        try:
            rgb = tuple(int(color_text[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4))
        except ValueError as exc:
            raise ValueError("custom_color must be a hex RGB color such as #RRGGBB") from exc
    else:
        rgb = colors.get(background)
        if rgb is None:
            raise ValueError(f"unsupported background: {background!r}")
    return torch.tensor(rgb, device=device, dtype=dtype)


def _parse_stretch_info(
    stretch_info: object, working_width: int, working_height: int
) -> tuple[int, int, str]:
    """Validate optional 16-alignment stretch metadata and return logical WH."""
    if not isinstance(stretch_info, dict):
        raise ValueError("stretch_info must be connected from BBOX: Mask → 1M Reference Image")
    if stretch_info.get("version") != 1:
        raise ValueError(f"unsupported stretch_info version: {stretch_info.get('version')!r}")
    logical_size = stretch_info.get("logical_size_wh")
    working_size = stretch_info.get("working_size_wh")
    if not (isinstance(logical_size, (list, tuple)) and len(logical_size) == 2):
        raise ValueError("stretch_info has invalid logical_size_wh")
    if not (isinstance(working_size, (list, tuple)) and len(working_size) == 2):
        raise ValueError("stretch_info has invalid working_size_wh")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (*logical_size, *working_size)):
        raise ValueError("stretch_info dimensions must be integers")
    logical_width, logical_height = logical_size
    info_working_width, info_working_height = working_size
    if logical_width <= 0 or logical_height <= 0:
        raise ValueError("stretch_info has non-positive logical_size_wh")
    if (info_working_width, info_working_height) != (working_width, working_height):
        raise ValueError(
            "stretch_info working_size_wh does not match crop_info canvas_size_wh; "
            "connect both outputs from the same BBOX reference node"
        )
    alignment_mode = stretch_info.get("alignment_mode", "stretch")
    if alignment_mode not in {"stretch", "pad_gray"}:
        raise ValueError(f"unsupported stretch_info alignment_mode: {alignment_mode!r}")
    return logical_width, logical_height, alignment_mode


def _resize_rgba_premultiplied(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize RGBA with Lanczos without dark/bright transparent-edge fringes."""
    if image.shape[-1] != 4:
        raise ValueError("_resize_rgba_premultiplied expects BHWC RGBA input")
    if tuple(image.shape[1:3]) == (height, width):
        return image

    alpha = image[..., 3:4].clamp(0.0, 1.0)
    # Keep opaque backgrounds exactly opaque and avoid needless alpha math.
    if bool(torch.all(alpha == 1.0)):
        rgb = _resize_bhwc_lanczos(image[..., :3], width, height)
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)

    # Lanczos RGB resize is applied to premultiplied values. Alpha is resized
    # with the same kernel (replicated to RGB only because Pillow's RGB mode is
    # used by the shared helper), then RGB is un-premultiplied.
    premultiplied_rgb = _resize_bhwc_lanczos(image[..., :3] * alpha, width, height)
    resized_alpha = _resize_bhwc_lanczos(alpha.expand(-1, -1, -1, 3), width, height)[..., :1].clamp(0.0, 1.0)
    rgb = torch.where(
        resized_alpha > 1e-6,
        premultiplied_rgb / resized_alpha.clamp_min(1e-6),
        torch.zeros_like(premultiplied_rgb),
    )
    return torch.cat((rgb, resized_alpha), dim=-1)


class BBoxRestoreCropToCanvas:
    """Paste an edited BBOX crop back, optionally undoing the 16-alignment stretch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop": ("IMAGE", {"tooltip": "Edited bbox_crop. Its pixels are pasted without any resize."}),
                "crop_info": ("BBOX_CROP_INFO", {"tooltip": "Connect crop_info from BBOX: Mask → 1M Reference Image."}),
                "background": (
                    ["alpha", "black", "white", "red", "green", "blue", "yellow", "custom"],
                    {"default": "alpha"},
                ),
                "custom_color": ("STRING", {"default": "#000000", "multiline": False}),
            },
            "optional": {
                "stretch_info": (
                    "BBOX_STRETCH_INFO",
                    {"tooltip": "Optional. Connect it to undo only the final 16-alignment stretch after the crop has been pasted."},
                ),
            },
        }

    # First two outputs retain their prior positions. restored_image is logical
    # (un-stretched) when stretch_info is supplied; working_image_16 always
    # remains the exact 16-aligned canvas used for crop placement.
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("restored_image", "alpha_mask", "working_image_16", "working_alpha_mask")
    FUNCTION = "restore"
    CATEGORY = "BBOX Tools"

    def restore(
        self,
        crop: torch.Tensor,
        crop_info: object,
        background: str = "alpha",
        custom_color: str = "#000000",
        stretch_info: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Do not clamp here: edited crop pixels must be pasted back bit-for-bit unchanged.
        crop = _as_bhwc(crop, "crop", clamp=False)
        if crop.shape[-1] not in (3, 4):
            raise ValueError(f"crop must have 3 (RGB) or 4 (RGBA) channels, got {crop.shape[-1]}")
        info_items = _parse_crop_info(crop_info)
        if crop.shape[0] != len(info_items):
            raise ValueError(
                f"crop batch size ({crop.shape[0]}) does not match crop_info item count ({len(info_items)}); "
                "do not mix crop_info from another batch"
            )

        canvas_sizes = {(canvas_width, canvas_height) for canvas_width, canvas_height, *_ in info_items}
        if len(canvas_sizes) != 1:
            raise ValueError(
                "crop_info contains different canvas sizes; process each item separately so ComfyUI can form one IMAGE batch"
            )
        canvas_width, canvas_height = canvas_sizes.pop()
        rgb = _background_rgb(background, custom_color, crop.device, crop.dtype)
        working = torch.empty((crop.shape[0], canvas_height, canvas_width, 4), device=crop.device, dtype=crop.dtype)
        working[..., :3] = rgb
        working[..., 3] = 0.0 if background == "alpha" else 1.0

        for batch_index, (_, _, x, y, width, height) in enumerate(info_items):
            one_crop = crop[batch_index]
            if tuple(one_crop.shape[:2]) != (height, width):
                raise ValueError(
                    f"crop item {batch_index} is {one_crop.shape[1]}x{one_crop.shape[0]}, but crop_info requires {width}x{height}; "
                    "the crop is never resized during restoration"
                )
            working[batch_index, y : y + height, x : x + width, :3] = one_crop[..., :3]
            working[batch_index, y : y + height, x : x + width, 3] = (
                one_crop[..., 3] if one_crop.shape[-1] == 4 else 1.0
            )

        if stretch_info is None:
            restored = working
        else:
            logical_width, logical_height, alignment_mode = _parse_stretch_info(stretch_info, canvas_width, canvas_height)
            if alignment_mode == "pad_gray":
                # Padding was appended only to the right/bottom, so removing it
                # is a crop operation: no interpolation and no added blur.
                restored = working[:, :logical_height, :logical_width, :].clone()
            else:
                # Pasting happens before this resize. Therefore the inverse
                # stretch samples a continuous canvas and cannot create a
                # crop-edge seam.
                restored = _resize_rgba_premultiplied(working, logical_width, logical_height)

        return restored, restored[..., 3], working, working[..., 3]


NODE_CLASS_MAPPINGS = {
    "BBoxMaskToReferenceImage": BBoxMaskToReferenceImage,
    "BBoxRestoreCropToCanvas": BBoxRestoreCropToCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BBoxMaskToReferenceImage": "BBOX: Mask → 1M Reference Image",
    "BBoxRestoreCropToCanvas": "BBOX: Restore Crop to Canvas",
}
