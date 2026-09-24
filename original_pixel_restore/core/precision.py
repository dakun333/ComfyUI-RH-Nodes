"""Precision-preserving RGB IO and support-restricted quantization."""

from pathlib import Path
import struct
import zlib

import cv2
import numpy as np
from PIL import Image, ImageOps, PngImagePlugin


OUTPUT_MODES = ("16bit", "8bit_dither", "8bit_round")


def _rgb(image):
    array = np.asarray(image)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.size == 0
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError("Expected H x W x 3 numeric RGB")
    if (
        not np.isfinite(array).all()
        or array.min() < 0
        or array.max() > 255
    ):
        raise ValueError(
            "RGB values must be finite and in 0..255; scale uint16 by 1/257 first"
        )
    return array


def _support(mask, shape):
    value = np.asarray(mask)
    if value.shape != shape[:2] or value.dtype != np.bool_:
        raise ValueError("support must be an H x W bool mask matching the image")
    return value


def quantize8(image, support=None, dither=False, seed=421):
    array = _rgb(image)
    output = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if dither:
        if support is None:
            raise ValueError("8-bit dithering requires an explicit support mask")
        active = _support(support, array.shape)
        noise = np.random.default_rng(seed).uniform(-0.5, 0.5, (*active.shape, 1))
        output[active] = np.clip(
            np.rint(array[active].astype(np.float64) + noise[active]), 0, 255
        ).astype(np.uint8)
    return output


def quantize_supported(original, candidate, support, seed=421):
    """Copy decoded original uint8 pixels outside support."""
    if not np.isfinite(candidate).all():
        raise ValueError("candidate must be finite")
    original = _rgb(original)
    candidate = _rgb(np.clip(candidate, 0, 255))
    active = _support(support, original.shape)
    if original.dtype != np.uint8 or candidate.shape != original.shape:
        raise ValueError("original must be uint8 and candidate must match it")
    output = quantize8(candidate, active, True, seed)
    output[~active] = original[~active]
    return output


def _chunk(kind, body):
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def write_png16(path, image, icc_profile=None, metadata=None):
    """Write true 16-bit RGB PNG with optional ICC and UTF-8 text metadata."""
    array = _rgb(image)
    values = np.rint(array.astype(np.float64) * 257).astype(np.uint16)
    height, width = array.shape[:2]
    rows = values.astype(">u2").view(np.uint8).reshape(height, width * 6)
    sub = rows.copy()
    sub[:, 6:] = rows[:, 6:] - rows[:, :-6]
    filtered = np.concatenate([np.ones((height, 1), np.uint8), sub], axis=1).tobytes()
    data = b"\x89PNG\r\n\x1a\n" + _chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    )
    if icc_profile:
        data += _chunk(
            b"iCCP", b"Original ICC\x00\x00" + zlib.compress(icc_profile)
        )
    for key, text in (metadata or {}).items():
        encoded_key = str(key).encode("latin1")
        if not 1 <= len(encoded_key) <= 79 or b"\x00" in encoded_key:
            raise ValueError("PNG metadata keys must be 1..79 Latin-1 bytes without NUL")
        data += _chunk(
            b"iTXt",
            encoded_key
            + b"\x00\x00\x00\x00\x00"
            + str(text).encode("utf8"),
        )
    data += _chunk(b"IDAT", zlib.compress(filtered, 6)) + _chunk(b"IEND", b"")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return values


def write_rgb16(path, original, candidate, support, icc_profile=None):
    """Write candidate values inside support and original uint8 values elsewhere."""
    if not np.isfinite(candidate).all():
        raise ValueError("candidate must be finite")
    original = _rgb(original)
    candidate = _rgb(np.clip(candidate, 0, 255))
    active = _support(support, original.shape)
    if original.dtype != np.uint8 or candidate.shape != original.shape:
        raise ValueError("original must be uint8 and candidate must match it")
    values = original.astype(np.float64)
    values[active] = candidate[active]
    return write_png16(path, values, icc_profile)


def save_png(
    path,
    image,
    mode="16bit",
    support=None,
    seed=421,
    icc_profile=None,
    metadata=None,
):
    if mode not in OUTPUT_MODES:
        raise ValueError("Unknown PNG output mode")
    if mode == "16bit":
        return write_png16(path, image, icc_profile, metadata)
    values = quantize8(image, support, mode == "8bit_dither", seed)
    info = PngImagePlugin.PngInfo()
    for key, text in (metadata or {}).items():
        info.add_itxt(str(key), str(text))
    options = {"icc_profile": icc_profile} if icc_profile else {}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(
        output_path, pnginfo=info, compress_level=4, **options
    )
    return values


def load_rgb(path):
    """Read 8-bit images or 16-bit RGB/gray PNG without ICC conversion."""
    path = Path(path)
    raw = path.read_bytes()
    is_16_bit = (
        raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 25 and raw[24] == 16
    )
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Animated/multipage inputs are unsupported")
        profile = image.info.get("icc_profile")
        orientation = image.getexif().get(274, 1)
        if image.format == "TIFF":
            bits = image.tag_v2.get(258, 8)
            bits = (bits,) if isinstance(bits, int) else bits
            if max(bits) > 8:
                raise ValueError(
                    "High-bit-depth TIFF is unsupported; use 16-bit RGB PNG"
                )
        if is_16_bit:
            array = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if array is None or array.dtype != np.uint16:
                raise ValueError("Unable to decode 16-bit PNG")
            if "transparency" in image.info:
                raise ValueError("Transparent PNG input is unsupported")
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            elif array.shape[2] == 4:
                if not np.all(array[:, :, 3] == 65535):
                    raise ValueError("Transparent input is unsupported")
                array = array[:, :, :3][:, :, ::-1]
            elif array.shape[2] == 3:
                array = array[:, :, ::-1]
            else:
                raise ValueError("Unsupported 16-bit PNG channels")
            if orientation == 2:
                array = array[:, ::-1]
            elif orientation == 3:
                array = array[::-1, ::-1]
            elif orientation == 4:
                array = array[::-1]
            elif orientation == 5:
                array = array.transpose(1, 0, 2)
            elif orientation == 6:
                array = np.rot90(array, 3)
            elif orientation == 7:
                array = array.transpose(1, 0, 2)[::-1, ::-1]
            elif orientation == 8:
                array = np.rot90(array, 1)
            rgb = np.ascontiguousarray(array, dtype=np.float32) / np.float32(257)
        else:
            if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                raise ValueError(
                    "Use SDR 8-bit RGB/gray or 16-bit PNG; HDR/CMYK unsupported"
                )
            if (
                "A" in image.getbands()
                and image.getchannel("A").getextrema() != (255, 255)
            ):
                raise ValueError("Transparent input is unsupported")
            if "transparency" in image.info:
                raise ValueError("Transparent palette/RGB input is unsupported")
            rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    return rgb, {
        "icc": profile,
        "source": path.name,
        "bit_depth": 16 if is_16_bit else 8,
    }
