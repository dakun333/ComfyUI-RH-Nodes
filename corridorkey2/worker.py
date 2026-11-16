"""One-shot isolated worker. Its dependency versions never enter ComfyUI's process."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
# This also works with python -I (isolated mode).
sys.path[:0] = [str(ROOT / "python_deps"), str(ROOT / "runtime"), str(ROOT)]
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "runtime" / ".hf_cache"))

from asset_manager import atomic_json, ensure_assets
from dependency_bootstrap import ensure_worker_dependencies


def dependencies():
    try:
        private = (ROOT / "python_deps").resolve()
        for relative in ("numpy/__init__.py", "transformers/__init__.py"):
            if not (private / relative).is_file():
                raise RuntimeError("Private dependency directory is missing; first execution installs it automatically")
        import numpy as np
        import cv2
        import torch
        import torchvision
        import transformers
        import scipy
        import timm
        import utils3d
        import safetensors
        if not Path(np.__file__).resolve().is_relative_to(private) or not Path(transformers.__file__).resolve().is_relative_to(private):
            raise RuntimeError("Refusing host NumPy/Transformers fallback")
        if np.__version__ != "1.26.4" or transformers.__version__ != "4.57.6":
            raise RuntimeError("Private dependencies missing or incompatible")
        return {"python": sys.version.split()[0], "numpy": np.__version__,
                "torch": torch.__version__, "torchvision": torchvision.__version__,
                "transformers": transformers.__version__, "opencv": cv2.__version__,
                "cuda_available": torch.cuda.is_available()}
    except Exception as exc:
        raise RuntimeError(f"CorridorKey2 worker dependency check failed: {exc}\n"
                           f"Run with ComfyUI Python: {ROOT / 'install_worker_deps.py'}\n"
                           "Do NOT install isolated_requirements.txt globally.") from exc


def run(job_path):
    job_path = Path(job_path).resolve()
    directory = job_path.parent
    job = json.loads(job_path.read_text(encoding="utf-8"))
    status_path = directory / "status.json"

    def status(percent, message):
        atomic_json(status_path, {"progress": percent, "message": message})
        print(f"[CorridorKey2 {percent}%] {message}", flush=True)

    start = time.monotonic()
    dependency_start = time.monotonic()
    ensure_worker_dependencies(progress=lambda message: status(1, message), status_file=status_path)
    dependency_setup_seconds = round(time.monotonic() - dependency_start, 3)
    status(2, "Private dependencies ready; checking isolated worker")
    versions = dependencies()
    import numpy as np
    import cv2
    import torch
    from color import srgb_to_linear
    if not torch.cuda.is_available():
        raise RuntimeError("CorridorKey2 requires an NVIDIA CUDA GPU; CPU inference is unsupported.")
    status(4, "Checking node-local model assets")
    downloaded = ensure_assets(progress=lambda text: status(4, text))
    from corridorkey import runtime as ck
    from corridorkey import video_io as vio

    # Unicode-safe image reads (upstream's cv2 filename APIs fail on Windows Unicode).
    def decode(path, flags):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
        if image is None:
            raise IOError(f"Failed to decode frame: {path}")
        return image

    def read_rgb(path):
        image = decode(path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image = image[..., :3][..., ::-1].copy()
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        if image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        return image.astype(np.float32, copy=False)

    def read_hint(path):
        image = decode(path, cv2.IMREAD_UNCHANGED)
        source_dtype = image.dtype
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
        if source_dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif source_dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        return image.astype(np.float32, copy=False)

    vio.read_image_rgb = read_rgb
    vio.read_hint_image = read_hint
    def read_hint_until(reader, t_end):
        while reader.next_t < t_end:
            frame = decode(reader.paths[reader.next_t], cv2.IMREAD_UNCHANGED)
            reader.cache[reader.next_t] = vio.preprocess_hint_frame(frame, reader.meta)
            reader.next_t += 1
    vio.SequentialHintFrameDirWindowReader._read_until = read_hint_until
    upstream_remove = ck.safe_remove
    def defer_locked_remove(path):
        try:
            upstream_remove(path)
        except PermissionError:
            print(f"[CorridorKey2] Defer locked scratch cleanup until worker exit: {path}", flush=True)
    ck.safe_remove = defer_locked_remove
    clip = directory / "clip"
    clip.mkdir()
    params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
              cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]

    def write_exr(path, image):
        ok, encoded = cv2.imencode(".exr", np.ascontiguousarray(image), params)
        if not ok:
            raise IOError(f"Failed to encode EXR: {path}")
        Path(path).write_bytes(encoded.tobytes())

    source_path = directory / "images.npy"
    images = np.load(source_path, mmap_mode="r", allow_pickle=False)
    n, h, w, c = images.shape
    if c != 3:
        raise ValueError("Worker expects RGB NHWC input")
    model_h = max(1, round(h * job["settings"]["model_scale"]))
    model_w = max(1, round(w * job["settings"]["model_scale"]))
    status(7, f"Preparing {n} frames at {w}x{h} (float32 linear EXR; no 8-bit quantization)")
    for i in range(n):
        linear = srgb_to_linear(images[i])
        if (model_h, model_w) != (h, w):
            linear = cv2.resize(linear, (model_w, model_h), interpolation=cv2.INTER_LANCZOS4)
        write_exr(clip / f"{i:06d}.exr", linear[..., ::-1])
    images._mmap.close()
    del images
    source_path.unlink()

    initial_hint, full_hints = None, None
    hint_path = directory / "hints.npy"
    if hint_path.is_file():
        hints = np.load(hint_path, mmap_mode="r", allow_pickle=False)
        hint_dir = directory / "hints"
        hint_dir.mkdir()
        for i in range(len(hints)):
            hint = hints[i]
            if (model_h, model_w) != (h, w):
                hint = cv2.resize(hint, (model_w, model_h), interpolation=cv2.INTER_LANCZOS4)
            write_exr(hint_dir / f"{i:06d}.exr", np.clip(hint, 0.0, 1.0))
        if len(hints) == 1:
            initial_hint = str(hint_dir / "000000.exr")
        elif len(hints) == n:
            full_hints = str(hint_dir)
        else:
            raise ValueError("Hint count must be 1 or the input frame count")
        del hint
        hints._mmap.close()
        del hints
        hint_path.unlink()

    # Report actual rendered frames without altering any inference math.
    upstream_writer = ck.ExrSequenceWriter
    class ProgressWriter(upstream_writer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completed_frames = 0
        def write_many(self, frames):
            super().write_many(frames)
            self.completed_frames += len(frames)
            status(30 + int(60 * self.completed_frames / n),
                   f"Rendering foreground + alpha: {self.completed_frames}/{n}")
    ck.ExrSequenceWriter = ProgressWriter

    settings = job["settings"]
    cfg = ck.InferenceConfig(
        hann_chunk=settings["hann_chunk"], hann_stride=settings["hann_stride"],
        carry_hint=settings["carry_hint"], hint_quality=settings["hint_quality"],
        carry_hint_quality=settings["hint_quality"], low_vram=settings["low_vram"],
        low_vram_foundation_chunk=1, model_scale=1.0,
        seed=settings["seed"], outputs={"cutout"},
        temp_dir=directory / "scratch", frame_dir_fps=settings["fps"],
        frame_dir_linear=True, despill_strength=settings["despill_strength"],
        output_format="exr", exr_dtype="float")
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    status(12, "Loading CorridorKey 2.0 and foundation models")
    model = ck.load_model(ck.RuntimePaths.from_package(ROOT / "runtime"), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    status(30, "Running native CorridorKey 2.0 inference")
    outputs = directory / "predictions"
    ck.run_inference(model, str(clip), str(outputs), cfg, ck.WriterConfig(),
                     initial_hint_path=initial_hint, hint_video_path=full_hints)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    del model
    torch.cuda.empty_cache()
    # Reclaim input EXRs before constructing the final lossless result array.
    for path in clip.glob("*.exr"):
        path.unlink()
    result = np.lib.format.open_memmap(directory / "linear_rgba.npy", mode="w+",
                                       dtype=np.float32, shape=(n, h, w, 4))
    frames = sorted((outputs / "clip_cutout_exr").glob("cutout_*.exr"))
    if len(frames) != n:
        raise RuntimeError(f"Output frame count mismatch: expected {n}, received {len(frames)}")
    for i, path in enumerate(frames):
        bgra = decode(path, cv2.IMREAD_UNCHANGED)
        if bgra.shape != (model_h, model_w, 4) or not np.isfinite(bgra).all():
            raise RuntimeError(f"Invalid prediction frame: {path}")
        if (model_h, model_w) != (h, w):
            bgra = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_LANCZOS4)
        result[i, ..., :3] = bgra[..., [2, 1, 0]]
        result[i, ..., 3] = np.clip(bgra[..., 3], 0.0, 1.0)
        path.unlink()
    result.flush()
    del result
    report = {"model": "CorridorKey 2.0", "frames": n, "width": w, "height": h,
              "inference_input_width": model_w, "inference_input_height": model_h,
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "peak_gpu_allocated_gib": round(peak_vram, 3), "settings": settings,
              "versions": versions, "downloaded_assets": downloaded,
              "foreground": "predicted clean RGB, straight/unassociated alpha",
              "internal_result": "float32 scene-linear RGBA EXR, no 8-bit quantization",
              "comfy_images": "sRGB float32 clipped to [0,1]; alpha=foreground coverage"}
    atomic_json(directory / "report.json", report)
    status(100, "Inference complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path)
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--bootstrap-deps", action="store_true", help="Explicitly trigger automatic dependency setup only")
    parser.add_argument("--ensure-assets", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.check_deps:
        print(json.dumps(dependencies(), indent=2), flush=True)
    elif args.bootstrap_deps:
        ensure_worker_dependencies(progress=lambda text: print(text, flush=True))
        print(json.dumps(dependencies(), indent=2), flush=True)
    elif args.ensure_assets:
        print(json.dumps({"downloaded": ensure_assets(verify=args.verify)}, indent=2), flush=True)
    elif args.job:
        run(args.job)
    else:
        parser.error("Specify --job, --check-deps, --bootstrap-deps, or --ensure-assets")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
