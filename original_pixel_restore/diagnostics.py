"""Render arrays captured by the restore node without rerunning the pipeline."""

from io import BytesIO
import json

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont
import torch

from .core.pipeline import blur, gray, resize_like, rms
from .core.precision import quantize8


def make_panels(result, profile):
    def uint8(array):
        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    def show(array):
        image = Image.fromarray(uint8(array))
        if profile and profile.get("icc"):
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(BytesIO(profile["icc"])),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        return np.array(image)

    def heat(array, maximum):
        return cv2.cvtColor(
            cv2.applyColorMap(
                uint8(np.clip(array / maximum, 0, 1) * 255),
                cv2.COLORMAP_INFERNO,
            ),
            cv2.COLOR_BGR2RGB,
        )

    def mask(array):
        return uint8(array.astype(float) * 255)

    def edges(first, second):
        first_edges = cv2.Canny(uint8(gray(first)), 55, 120) > 0
        second_edges = cv2.Canny(uint8(gray(second)), 55, 120) > 0
        output = np.full(first.shape, 20, np.uint8)
        output[first_edges] = [255, 75, 160]
        output[second_edges] = [55, 220, 225]
        output[first_edges & second_edges] = 245
        return output

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    panels, titles = [], []

    def panel(title, items, note):
        canvas = Image.new("RGB", (1200, 520), (17, 23, 31))
        draw = ImageDraw.Draw(canvas)
        draw.text((22, 15), title, font=font(24), fill="white")
        titles.append(title)
        for index, (array, label) in enumerate(items):
            image = Image.fromarray(uint8(array))
            image.thumbnail((376, 376), Image.Resampling.LANCZOS)
            x = 16 + index * 400
            draw.text((x + 4, 58), label, font=font(17), fill="white")
            canvas.paste(
                image,
                (x + (376 - image.width) // 2, 91 + (376 - image.height) // 2),
            )
        draw.text((22, 484), note, font=font(15), fill=(169, 184, 200))
        panels.append(np.array(canvas))

    original = result["original"]
    edited = result["edited"]
    trace = result["trace"]
    global_aligned = trace["global_aligned"]
    aligned = result["aligned"]
    first_corrected = trace["first_corrected"]
    stats = result["stats"]
    height, width = original.shape[:2]
    names = [
        "Input sizes",
        "Global alignment",
        "Effective local flow",
        "First color fit",
        "Edit evidence",
        "Refit on trusted pixels",
        "Final segmentation",
        "Restricted blending",
        "Output and exactness",
    ]
    panel(
        "01 / " + names[0],
        [
            (show(original), f"Original {width}x{height}"),
            (edited, f"AI {edited.shape[1]}x{edited.shape[0]}"),
            (resize_like(edited, original), "AI resized to original grid"),
        ],
        "Original pixel grid is never resampled.",
    )
    panel(
        "02 / " + names[1],
        [
            (edges(original, resize_like(edited, original)), "Before: magenta=O cyan=AI"),
            (edges(original, global_aligned), "After: white=matched edges"),
            (heat(rms(original.astype(float) - global_aligned), 20), "RGB residual 0..20+"),
        ],
        f'Global transform accepted: {stats["registration"]["method"]}',
    )
    flow = result["flow"]
    magnitude = (
        np.zeros((height, width), np.float32)
        if flow is None
        else np.linalg.norm(flow, axis=2)
    )
    local = stats["local_alignment"]
    accepted = bool(local and local["accepted"])
    panel(
        "03 / " + names[2],
        [
            (heat(magnitude, 5), "Effective displacement 0..5px"),
            (edges(original, global_aligned), "Before local motion"),
            (edges(original, aligned), "After local motion"),
        ],
        f"Local flow accepted: {accepted}; rejected flow is zero. Method A does not estimate local flow.",
    )
    panel(
        "04 / " + names[3],
        [
            (show(aligned), "Before color fit"),
            (show(first_corrected), "First corrected generated"),
            (heat(rms(first_corrected - aligned), 15), "Total correction 0..15+"),
        ],
        "Robust color model plus broad illumination field. Not the final composite.",
    )
    difference = original.astype(np.float32) - first_corrected
    panel(
        "05 / " + names[4],
        [
            (heat(rms(blur(difference, 0.65)), 15), "Color residual 0..15+"),
            (
                heat(rms(blur(difference, 0.6) - blur(difference, 3)), 10),
                "Detail residual 0..10+",
            ),
            (heat(trace["first_score"], 6), "Normalized evidence 0..6+"),
        ],
        "Black to yellow indicates increasing evidence; panel scales differ.",
    )
    panel(
        "06 / " + names[5],
        [
            (mask(trace["first_core"]), "First candidate core"),
            (mask(trace["trusted"]), "White = trusted color anchors"),
            (show(result["corrected"]), "Final corrected generated"),
        ],
        "Exclude first candidates and neighbors, refit color, then recompute the mask.",
    )
    overlay = show(original).astype(float)
    overlay[result["core"]] = (
        overlay[result["core"]] * 0.45 + np.array([255, 85, 60]) * 0.55
    )
    panel(
        "07 / " + names[6],
        [
            (heat(result["probability"], 1), "Uncalibrated evidence 0..1"),
            (mask(result["core"]), "Final core mask"),
            (overlay, "Red = candidate edit"),
        ],
        f'Core area {result["core"].mean():.2%}; manual override: {stats["manual_mask"]}. Evidence is not semantic truth.',
    )
    y_values, x_values = np.where(result["core"])
    center_y = int(np.median(y_values)) if len(y_values) else height // 2
    center_x = int(np.median(x_values)) if len(x_values) else width // 2
    crop_height, crop_width = min(140, height), min(140, width)
    y = max(0, min(height - crop_height, center_y - crop_height // 2))
    x = max(0, min(width - crop_width, center_x - crop_width // 2))
    crop = np.s_[y : y + crop_height, x : x + crop_width]
    alpha = result["alpha"]
    zones = np.zeros_like(original)
    zones[alpha == 0] = [35, 44, 57]
    zones[(alpha > 0) & (alpha < 1)] = [255, 186, 63]
    zones[alpha == 1] = [56, 201, 198]
    panel(
        "08 / " + names[7],
        [
            (mask(result["core"])[crop], "Crop: edit core"),
            (mask(alpha)[crop], "Crop: alpha / active region"),
            (zones[crop], "Cyan=active gold=blend dark=O"),
        ],
        f'Padding {stats["config"]["padding"]}px / feather {stats["config"]["feather"]}px. Mode {stats["blend"]}.',
    )
    preview = quantize8(result["output_float"], result["support"], True, 421)
    exact = np.all(preview == quantize8(original), axis=2)
    check = np.zeros_like(original)
    check[exact] = [35, 145, 134]
    check[~exact] = [255, 181, 74]
    panel(
        "09 / " + names[8],
        [
            (show(preview), "Dithered 8-bit preview"),
            (check, "Green=exact gold=changed"),
            (heat(rms(preview.astype(float) - original), 20), "Output change 0..20+"),
        ],
        f'8-bit exact {exact.mean():.2%}; outside-mask max error {stats["outside_max_channel_error"]}.',
    )
    return torch.from_numpy(np.stack(panels).astype(np.float32) / 255), json.dumps(
        {
            "steps": titles,
            "stats": stats,
            "note": (
                "Panels are display-only dithered 8-bit previews (seed 421). "
                "Restore IMAGE stays float; outside-mask tensor samples are copied."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )
