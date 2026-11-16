"""Native ComfyUI VIDEO interop and mutually exclusive output allocation.

This module imports no model, worker, private dependency, or global sys.path.
ComfyUI's current VideoFromComponents drops alpha/metadata in get_components;
the small subclass below preserves them and never silently saves RGBA as H264.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path

import torch


def validate_output_mode(output_mode):
    if output_mode not in ("images", "video"):
        raise ValueError("output_mode must be 'images' or 'video'")


def resolve_media(images=None, video=None, fps=24.0, preserve_input_alpha=False):
    """Decode exactly one source. VIDEO fps/audio/metadata take precedence.

    Internal matting still needs frames in BOTH modes; output_mode does not
    magically turn the model into a streaming codec or remove input decoding.
    """
    if (images is None) == (video is None):
        raise ValueError("Connect exactly one input: images (IMAGE) OR video (VIDEO), not both.")
    audio = metadata = None
    input_kind = "images"
    if video is not None:
        if not callable(getattr(video, "get_components", None)):
            raise TypeError("video must be a native ComfyUI VIDEO object")
        components = video.get_components()
        images = components.images
        fps = components.frame_rate
        audio = components.audio
        metadata = getattr(components, "metadata", None)
        input_kind = "video"
        # File VIDEO inputs often expose RGB and a separate [N,H,W,1] alpha.
        # CorridorKey intentionally ignores source alpha, as before. UnMult
        # multiplies it; combine only when needed and never multiply it twice.
        source_alpha = getattr(components, "alpha", None)
        if preserve_input_alpha and source_alpha is not None and isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[-1] == 3:
            if not isinstance(source_alpha, torch.Tensor):
                raise TypeError("VIDEO alpha must be a tensor")
            if source_alpha.ndim == 4 and source_alpha.shape[-1] == 1:
                source_alpha = source_alpha[..., 0]
            if tuple(source_alpha.shape) != tuple(images.shape[:3]):
                raise ValueError("VIDEO alpha dimensions must match VIDEO images")
            images = torch.cat((images, source_alpha.to(images).unsqueeze(-1)), dim=-1)
    try:
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be finite and positive")
        frame_rate = fps if isinstance(fps, Fraction) else Fraction(str(float(fps))).limit_denominator(1000000)
    except (TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("fps must be finite and positive") from exc
    return images, frame_rate, audio, metadata, input_kind


@lru_cache(maxsize=1)
def _video_class():
    # Lazy import: image mode does not initialize ComfyUI's VIDEO API.
    from comfy_api.latest import InputImpl, Types

    class MattingVideo(InputImpl.VideoFromComponents):
        """Native VIDEO with straight RGBA and an alpha view, not a checker."""
        def __init__(self, components):
            super().__init__(components)
            self._matting_components = components

        def get_components(self):
            c = self._matting_components
            # RGBA IMAGE is intentional: native GetVideoComponents can then
            # pass transparency to SaveWEBM(vp9). alpha aliases the same storage.
            return Types.VideoComponents(images=c.images, frame_rate=c.frame_rate,
                                         alpha=c.alpha, audio=c.audio, metadata=c.metadata)

        def get_dimensions(self):
            c = self._matting_components
            return int(c.images.shape[2]), int(c.images.shape[1])

        def get_frame_count(self):
            return int(self._matting_components.images.shape[0])

        def get_frame_rate(self):
            return self._matting_components.frame_rate

        def get_duration(self):
            return float(self.get_frame_count() / self.get_frame_rate())

        def get_container_format(self):
            return "mov"

        def save_to(self, path, format="auto", codec="auto", metadata=None):
            """Optional lossless RGBA MOV/transparent WebM export.

            No encoding happens when the matting node creates a VIDEO. Native
            SaveVideo currently exposes only MP4/H264; reject that alpha-loss
            path instead of producing a black rectangle or an invalid rgb24.
            """
            import av
            from comfy import model_management as mm
            fmt = getattr(format, "value", format)
            cod = getattr(codec, "value", codec)
            suffix = Path(path).suffix.lower() if isinstance(path, (str, Path)) else ""
            if fmt == "mp4" or cod == "h264" or suffix == ".mp4":
                raise ValueError("Transparent matting VIDEO cannot be saved as MP4/H264. "
                                 "Use GetVideoComponents → SaveWEBM (codec=vp9), "
                                 "or an alpha-aware VIDEO saver (.mov/.webm). No alpha was discarded.")
            target = "webm" if fmt == "webm" or suffix == ".webm" else "mov"
            if fmt not in ("auto", "mov", "webm") or cod not in ("auto", "qtrle", "vp9", "libvpx-vp9"):
                raise ValueError("Matting VIDEO export supports RGBA MOV/qtrle or WebM/vp9 only")
            c = self._matting_components
            options = {"movflags": "use_metadata_tags"} if target == "mov" else {}
            with av.open(str(path) if isinstance(path, Path) else path, mode="w", format=target, options=options) as out:
                for key, value in (metadata or {}).items():
                    out.metadata[str(key)] = json.dumps(value, ensure_ascii=False)
                stream = out.add_stream("qtrle" if target == "mov" else "libvpx-vp9", rate=c.frame_rate)
                stream.width, stream.height = self.get_dimensions()
                stream.pix_fmt = "argb" if target == "mov" else "yuva420p"
                if target == "webm":
                    stream.options = {"lossless": "1", "crf": "0", "auto-alt-ref": "0"}
                astream = None
                if c.audio is not None:
                    waveform = c.audio["waveform"]
                    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2, 6):
                        raise ValueError("Audio waveform must be [1,1/2/6,samples]")
                    rate = int(c.audio["sample_rate"])
                    layout = {1: "mono", 2: "stereo", 6: "5.1"}[int(waveform.shape[1])]
                    samples = min(int(waveform.shape[-1]), math.ceil(rate * self.get_duration()))
                    if samples > 0:
                        # Register all streams BEFORE the first video packet
                        # writes the container header (otherwise audio time_base is zero).
                        astream = out.add_stream("aac" if target == "mov" else "libopus",
                                                 rate=rate if target == "mov" else 48000, layout=layout)
                for tensor in c.images:
                    mm.throw_exception_if_processing_interrupted()
                    array = (tensor.clamp(0, 1) * 255).round().to(device="cpu", dtype=torch.uint8).numpy()
                    frame = av.VideoFrame.from_ndarray(array, format="rgba")
                    out.mux(stream.encode(frame))
                out.mux(stream.encode(None))
                if astream is not None:
                    aframe = av.AudioFrame.from_ndarray(waveform[0, :, :samples].float().cpu().contiguous().numpy(), format="fltp", layout=layout)
                    aframe.sample_rate, aframe.pts = rate, 0
                    mm.throw_exception_if_processing_interrupted()
                    out.mux(astream.encode(aframe))
                    out.mux(astream.encode(None))

        def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
            c = self._matting_components
            total = self.get_duration()
            start = float(start_time or 0.0)
            if start < 0:
                start = max(0.0, total + start)
            length = float(duration or 0.0)
            if not math.isfinite(start) or not math.isfinite(length) or length < 0:
                raise ValueError("Trim start/duration must be finite; duration must be nonnegative")
            if start >= total or (strict_duration and length > 0 and start + length > total):
                return None
            first = max(0, min(self.get_frame_count(), round(start * float(c.frame_rate))))
            last = self.get_frame_count() if length == 0 else min(self.get_frame_count(), first + round(length * float(c.frame_rate)))
            if last <= first:
                return None
            audio = c.audio
            if audio is not None:
                rate = int(audio["sample_rate"])
                audio = {**audio, "waveform": audio["waveform"][..., round(first / c.frame_rate * rate):round(last / c.frame_rate * rate)]}
            rgba = c.images[first:last]
            return type(self)(Types.VideoComponents(images=rgba, frame_rate=c.frame_rate,
                                                   alpha=rgba[..., 3], audio=audio, metadata=c.metadata))

    return MattingVideo


def make_matting_video(rgba, frame_rate, audio=None, metadata=None):
    from comfy_api.latest import Types
    return _video_class()(Types.VideoComponents(images=rgba, frame_rate=frame_rate,
                                               alpha=rgba[..., 3], audio=audio, metadata=metadata))


class MattingOutputBuffer:
    """Allocate ONLY the selected representation. No dummy tensors/videos."""
    def __init__(self, mode, n, h, w, frame_rate, audio=None, metadata=None):
        validate_output_mode(mode)
        self.mode, self.frame_rate = mode, frame_rate
        self.audio, self.metadata = audio, metadata
        self.rgba = torch.empty((n, h, w, 4), dtype=torch.float32, device="cpu")
        self.foreground = self.alpha = self.checker = None
        if mode == "images":
            self.foreground = torch.empty((n, h, w, 3), dtype=torch.float32, device="cpu")
            self.alpha = torch.empty((n, h, w), dtype=torch.float32, device="cpu")
            self.checker = torch.empty((n, h, w, 3), dtype=torch.float32, device="cpu")

    @property
    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in
                   (self.rgba, self.foreground, self.alpha, self.checker) if x is not None)

    def write(self, i, foreground, alpha, preview):
        fg = torch.from_numpy(foreground) if not isinstance(foreground, torch.Tensor) else foreground
        a = torch.from_numpy(alpha) if not isinstance(alpha, torch.Tensor) else alpha
        self.rgba[i, ..., :3].copy_(fg)
        self.rgba[i, ..., 3].copy_(a)
        if self.mode == "images":
            self.foreground[i].copy_(self.rgba[i, ..., :3])
            self.alpha[i].copy_(self.rgba[i, ..., 3])
            # Preview is not even called in VIDEO mode.
            image = preview()
            self.checker[i].copy_(torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image)

    def finish(self, report):
        from comfy_execution.graph_utils import ExecutionBlocker
        report.update(output_mode=self.mode, output_tensor_bytes=self.tensor_bytes,
                      output_frame_rate=str(self.frame_rate), audio_preserved_in_video=self.audio is not None and self.mode == "video",
                      inactive_outputs="video" if self.mode == "images" else "foreground_rgb, alpha, rgba, checker_preview")
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if self.mode == "images":
            return self.foreground, self.alpha, self.rgba, self.checker, text, ExecutionBlocker(None)
        metadata = {**(self.metadata or {}), "matting_report": report}
        video = make_matting_video(self.rgba, self.frame_rate, self.audio, metadata)
        return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None),
                ExecutionBlocker(None), text, video)
