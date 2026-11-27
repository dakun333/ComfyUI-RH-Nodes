"""Lightweight ComfyUI bridge; never import CorridorKey/transformers in the host."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np
import torch
import folder_paths
from comfy_api.latest import io
from .checkpoint_io import BUNDLE_FILENAME
from .alpha import adjust_alpha, validate_alpha_settings
from .color import checker_preview, linear_to_srgb
from .video_io import MattingOutputBuffer, resolve_media, validate_output_mode

ROOT = Path(__file__).resolve().parent
_RUN_LOCK = threading.Lock()


def checkpoint_options():
    names = [name for name in folder_paths.get_filename_list("checkpoints") if name.lower().endswith(".safetensors")]
    return [BUNDLE_FILENAME] + [name for name in names if name != BUNDLE_FILENAME]


def resolve_checkpoint(name):
    if not isinstance(name, str):
        raise ValueError("checkpoint must be a filename from ComfyUI checkpoints")
    relative = PureWindowsPath(name)
    if relative.drive or relative.root or ".." in relative.parts or ":" in name or relative.suffix.lower() != ".safetensors":
        raise ValueError("checkpoint must be a relative .safetensors filename from ComfyUI checkpoints")
    existing = folder_paths.get_full_path("checkpoints", name)
    if existing:
        return Path(existing).resolve()
    if name == BUNDLE_FILENAME:
        return Path(folder_paths.models_dir) / "checkpoints" / BUNDLE_FILENAME
    raise FileNotFoundError(f"Checkpoint not found in ComfyUI checkpoints: {name}")


def validate_inputs(images, hint_mask, fps, hann_chunk, hann_stride, model_scale, despill_strength):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("images must be a ComfyUI IMAGE batch [frames,height,width,3 or 4]")
    n, h, w, _ = images.shape
    if min(n, h, w) < 1:
        raise ValueError("Input frame batch is empty")
    if not images.is_floating_point() or not torch.isfinite(images).all().item():
        raise ValueError("IMAGE values must be finite floating-point sRGB values")
    for name, value in [("fps", fps), ("model_scale", model_scale)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if model_scale > 1:
        raise ValueError("model_scale must be <=1 (1 is native resolution)")
    if not 0 <= despill_strength <= 1:
        raise ValueError("despill_strength must be in [0,1]")
    if hann_chunk < 1 or not 1 <= hann_stride <= hann_chunk:
        raise ValueError("hann_stride must be between 1 and hann_chunk")
    if hint_mask is None:
        return None
    if hint_mask.ndim == 2:
        hint_mask = hint_mask.unsqueeze(0)
    if hint_mask.ndim != 3 or hint_mask.shape[0] not in (1, n):
        raise ValueError("hint_mask must be [1,H,W] (first-frame hint) or [frame_count,H,W]")
    if tuple(hint_mask.shape[-2:]) != (h, w):
        raise ValueError("hint_mask height/width must match input images; resize it explicitly first")
    if not torch.isfinite(hint_mask).all().item():
        raise ValueError("hint_mask contains NaN or Inf")
    return hint_mask


def terminate_worker(process):
    """Terminate only this worker's process tree (Windows venv launchers spawn a child)."""
    if process.poll() is None:
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "taskkill.exe"
            subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        if process.poll() is None:
            process.kill()
    process.wait(timeout=30)


class CorridorKey2Native(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CorridorKey2Native",
            display_name="纯色底去背（宿主依赖版）",
            category="RH-Nodes/matting",
            description=("CorridorKey 2.0 with unchanged ComfyUI dependencies. "
                         "Existing utils3d is tried first; bundled source is used only when absent. "
                         "Select the CorridorKey full checkpoint from ComfyUI checkpoints. "
                         "Connect IMAGE or VIDEO, not both. Only the selected output mode is computed."),
            inputs=[
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01),
                io.Boolean.Input("carry_hint", default=True),
                io.Float.Input("despill_strength", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Extra despill; 0 preserves predicted clean foreground colors."),
                io.Boolean.Input("low_vram", default=True),
                io.Int.Input("hann_chunk", default=4, min=1, max=128),
                io.Int.Input("hann_stride", default=2, min=1, max=128),
                io.Float.Input("model_scale", default=1.0, min=0.1, max=1.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=2**32 - 1),
                io.Image.Input("images", optional=True, tooltip="IMAGE 帧批次，与 video 二选一。"),
                io.Mask.Input("hint_mask", optional=True),
                io.Float.Input("hint_quality", default=0.95, min=0.0, max=1.0, step=0.01, optional=True),
                io.Boolean.Input("invert_hint_mask", default=False, optional=True),
                io.String.Input("temp_directory", default="", optional=True),
                io.Video.Input("video", optional=True, tooltip="VIDEO 输入；保留源帧率、音频。"),
                io.Combo.Input("output_mode", options=["images", "video"], default="images", optional=True),
                io.Combo.Input("checkpoint", options=checkpoint_options(), default=BUNDLE_FILENAME, optional=True,
                               tooltip="CorridorKey full checkpoint in ComfyUI/models/checkpoints; not an SD checkpoint."),
                io.Int.Input("alpha_shrink", default=0, min=0, max=64, step=1, optional=True,
                             tooltip="Alpha 向内收缩的像素半径（最终输出分辨率），0 为关闭；先收缩再羽化。"),
                io.Float.Input("alpha_feather", default=0.0, min=0.0, max=64.0, step=0.1, optional=True,
                               tooltip="Alpha 高斯羽化强度 sigma（最终输出像素），0 为关闭；不改变前景 RGB。"),
            ],
            outputs=[
                io.Image.Output("foreground_rgb"), io.Mask.Output("alpha"),
                io.Image.Output("rgba"), io.Image.Output("checker_preview"),
                io.String.Output("report"), io.Video.Output("video"),
            ],
        )

    @classmethod
    def execute(cls, images=None, fps=24.0, carry_hint=True, despill_strength=0.0,
                low_vram=True, hann_chunk=4, hann_stride=2, model_scale=1.0, seed=0,
                hint_mask=None, hint_quality=0.95, invert_hint_mask=False, temp_directory="",
                video=None, output_mode="images", checkpoint=BUNDLE_FILENAME,
                alpha_shrink=0, alpha_feather=0.0):
        from comfy import model_management as mm
        from comfy.utils import ProgressBar
        validate_output_mode(output_mode)
        validate_alpha_settings(alpha_shrink, alpha_feather)
        checkpoint_path = resolve_checkpoint(checkpoint)
        mm.throw_exception_if_processing_interrupted()
        images, frame_rate, audio, source_metadata, input_kind = resolve_media(images, video, fps)
        mm.throw_exception_if_processing_interrupted()
        fps = float(frame_rate)
        hint_mask = validate_inputs(images, hint_mask, fps, hann_chunk, hann_stride,
                                    model_scale, despill_strength)
        if not math.isfinite(hint_quality) or not 0 <= hint_quality <= 1:
            raise ValueError("hint_quality must be in [0,1]")
        settings = {"fps": float(fps), "carry_hint": bool(carry_hint),
                    "despill_strength": float(despill_strength), "low_vram": bool(low_vram),
                    "hann_chunk": int(hann_chunk), "hann_stride": int(hann_stride),
                    "model_scale": float(model_scale), "seed": int(seed),
                    "hint_quality": float(hint_quality), "invert_hint_mask": bool(invert_hint_mask)}
        # Never retain another model behind ComfyUI's back. Release models managed by ComfyUI only.
        while not _RUN_LOCK.acquire(timeout=0.25):
            mm.throw_exception_if_processing_interrupted()
        try:
            mm.throw_exception_if_processing_interrupted()
            mm.unload_all_models()
            mm.soft_empty_cache()
            parent = Path(temp_directory).expanduser().resolve() if temp_directory.strip() else None
            if parent:
                parent.mkdir(parents=True, exist_ok=True)
            disk_parent = parent or Path(tempfile.gettempdir())
            n, h, w, _ = images.shape
            # Input/result EXRs + .npy + foundation scratch. This is a conservative estimate.
            estimated_temp = n * h * w * 64 + 512 * 1024**2
            if shutil.disk_usage(disk_parent).free < estimated_temp:
                raise RuntimeError(f"Not enough temp disk space at {disk_parent}. "
                                   f"Allow approximately {estimated_temp / 1024**3:.1f} GiB, "
                                   "or set temp_directory to another drive / use fewer frames.")
            progress = ProgressBar(100)
            logs = ROOT / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / (time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".log")
            with tempfile.TemporaryDirectory(prefix="corridorkey2_", dir=parent) as tmp:
                directory = Path(tmp)
                rgb = images[..., :3].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
                np.save(directory / "images.npy", np.clip(rgb, 0.0, 1.0), allow_pickle=False)
                del rgb
                if hint_mask is not None:
                    hint = hint_mask.detach().to(device="cpu", dtype=torch.float32).numpy()
                    hint = np.clip(hint, 0.0, 1.0)
                    if invert_hint_mask:
                        hint = 1.0 - hint
                    np.save(directory / "hints.npy", hint, allow_pickle=False)
                    del hint
                (directory / "job.json").write_text(json.dumps({"settings": settings, "checkpoint": str(checkpoint_path)}), encoding="utf-8")
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                environment["PYTHONUNBUFFERED"] = "1"
                environment["PYTHONIOENCODING"] = "utf-8"
                command = [sys.executable, "-I", "-u", str(ROOT / "worker.py"),
                           "--job", str(directory / "job.json")]
                print(f"[CorridorKey2] Starting CUDA worker with unchanged ComfyUI dependencies; log: {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    last_percent, last_message = -1, None
                    try:
                        while process.poll() is None:
                            mm.throw_exception_if_processing_interrupted()
                            try:
                                state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                                percent = max(last_percent, int(state["progress"]))
                                if percent != last_percent:
                                    progress.update_absolute(percent)
                                    last_percent = percent
                                if state["message"] != last_message:
                                    print(f"[CorridorKey2] {state['message']}", flush=True)
                                    last_message = state["message"]
                            except (OSError, ValueError, KeyError):
                                pass
                            time.sleep(0.3)
                    except BaseException:
                        terminate_worker(process)
                        raise
                if process.returncode != 0:
                    detail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                    raise RuntimeError(f"CorridorKey2 worker failed (exit {process.returncode}).\n"
                                       f"Log: {log_path}\n{detail}")
                mm.throw_exception_if_processing_interrupted()
                linear_rgba = np.load(directory / "linear_rgba.npy", mmap_mode="r", allow_pickle=False)
                outputs = MattingOutputBuffer(output_mode, n, h, w, frame_rate, audio, source_metadata)
                linear = a = None
                try:
                    for i in range(n):
                        mm.throw_exception_if_processing_interrupted()
                        linear = np.asarray(linear_rgba[i, ..., :3])
                        a = np.clip(linear_rgba[i, ..., 3], 0.0, 1.0)
                        a = adjust_alpha(a, alpha_shrink, alpha_feather)
                        fg = np.clip(linear_to_srgb(linear), 0.0, 1.0)
                        outputs.write(i, fg, a, lambda: checker_preview(linear, a))
                    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
                    report["log_path"] = str(log_path)
                    report["node_directory"] = str(ROOT)
                    report["input_kind"] = input_kind
                    report["alpha_postprocess"] = {
                        "alpha_shrink": int(alpha_shrink), "alpha_feather": float(alpha_feather),
                        "units": "output pixels; feather is Gaussian sigma", "order": "shrink_then_feather",
                    }
                finally:
                    # Close memmap explicitly: Windows refuses cleanup while it is open.
                    del linear, a
                    linear_rgba._mmap.close()
                    del linear_rgba
                progress.update_absolute(100)
                return io.NodeOutput(*outputs.finish(report))
        finally:
            _RUN_LOCK.release()
            mm.soft_empty_cache()
