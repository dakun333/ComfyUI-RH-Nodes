"""Private VRMBG3 nodes; default localhalo381 ep4. Foreground MASK=1; per-video autoregression, no alpha min/max."""
from __future__ import annotations
import contextlib, hashlib, json, math, os, subprocess, threading, time
from pathlib import Path
from fractions import Fraction
import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher
from comfy.utils import ProgressBar
from .foreground_color import recover_foreground_rgb

ROOT = Path(__file__).parent
REVISION = "59716e19a6cc97f91311edea190938219c097b76"
DEFAULT_MODEL_NAME = "localhalo381_epoch004_s000071232.safetensors"
WEIGHT_SHA = "49cc2001d47d73a3f06000e5a0c39e5a8deed6d5175e80278390199b62c0da06"
VERIFIED_MODELS = {
    WEIGHT_SHA: {"model": "BRIA VRMBG-3.0 / private localhalo381 ep4",
        "epoch": 4, "step": 71232, "training_dataset": "381 local_halo30"},
    "ed4e58962171ce1043cbc6ee1b7b2e72a1cfe20e7c6ac383d171c40e47640144": {
        "model": "BRIA VRMBG-3.0 / private V6 ep2", "epoch": 2, "step": 37440,
        "training_dataset": "400 online2d V6"},
}
SOURCE_SHA = {
    "model.py": "7cf51fe6907f79be8eadbd7cc75648eac2e8415152721d6e296d608eb0333b2b",
    "vrmbg3_config.py": "051ce339a8aa419d6e41d08204879b5cc528b7b90d9e665ca05b516a80e2515e",
    "config.json": "833cbc1fdfe95799afa4c070dbeb957ca885cdca5e0a173619d5c3852ccc1c0a",
}
# Prefer the shared checkpoints folder; retain previous installation paths.
MODEL_FOLDER = Path(folder_paths.models_dir) / "checkpoints"
LEGACY_MODEL_FOLDER = Path(folder_paths.models_dir) / "VRMBG3"
folder_paths.add_model_folder_path("vrmbg3_v6", str(LEGACY_MODEL_FOLDER))
folder_paths.add_model_folder_path("vrmbg3_v6", str(ROOT / "models"))
folder_paths.add_model_folder_path("vrmbg3_v6", str(MODEL_FOLDER), is_default=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def resolution_plan(w, h, max_pixels=1048576):
    if min(w, h, max_pixels) < 1: raise ValueError("Invalid resolution/pixel budget")
    def shape(scale):
        cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
        return cw, ch, math.ceil(cw / 32) * 32, math.ceil(ch / 32) * 32
    low, high = 0., min(1., math.sqrt(max_pixels / (w * h)))
    if shape(high)[2] * shape(high)[3] > max_pixels:
        for _ in range(60):
            mid = (low + high) / 2
            if shape(mid)[2] * shape(mid)[3] <= max_pixels: low = mid
            else: high = mid
        high = low
    cw, ch, pw, ph = shape(high)
    if min(cw, ch) < 64: raise ValueError("VRMBG3 needs both model content dimensions >=64")
    return {"native_width": w, "native_height": h, "content_width": cw, "content_height": ch,
            "width": pw, "height": ph, "pixels": pw * ph, "pad_right": pw - cw, "pad_bottom": ph - ch}


def check_images(images):
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("Expected nonempty ComfyUI IMAGE batch [frames,H,W,3 or 4]")
    if not images.is_floating_point(): raise ValueError("IMAGE must be floating point RGB in [0,1]")
    # Validate frame by frame; a whole-batch isfinite temporary can itself be GiB.
    for frame in images:
        if not torch.isfinite(frame).all() or frame.min() < -1e-6 or frame.max() > 1.000001:
            raise ValueError("IMAGE contains nonfinite/out-of-range values; not silently normalized")


class ComfyVRMBGWrapper(torch.nn.Module):
    """Keep BRIA untouched; ComfyUI needs writable device bookkeeping."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        # transformers.PreTrainedModel.device is a read-only property.
        # ModelPatcher sets device on load/unload, so patch this wrapper instead.
        self.device = torch.device("cpu")

    def forward(self, x):
        return self.net(x)


class ModelHandle:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.patcher = None
        self.lock = threading.RLock()
        self.loaded_identity = None
        self.model_info = {}

    def ensure_loaded(self):
        identity = (self.path.stat().st_size, self.path.stat().st_mtime_ns)
        if self.patcher is not None and self.loaded_identity == identity: return self.patcher
        if self.patcher is not None: mm.unload_model_and_clones(self.patcher, unload_additional_models=False)
        digest = sha256(self.path)
        if digest not in VERIFIED_MODELS:
            raise ValueError("Unverified VRMBG3 weights; SHA256 mismatch. Obtain the authorized checkpoint separately; RH-nodes does not distribute these weights.")
        self.model_info = {**VERIFIED_MODELS[digest], "weights_sha256": digest,
                           "weights_path": str(self.path)}
        source = ROOT / "vendor" / "bria_vrmbg3"
        for name, digest in SOURCE_SHA.items():
            if sha256(source / name) != digest: raise ValueError(f"Pinned BRIA source changed: {name}")
        from .vendor.bria_vrmbg3.model import BiRefNet
        from safetensors.torch import load_file
        model = BiRefNet().float().eval()
        model.load_state_dict(load_file(str(self.path), device="cpu"), strict=True)
        self.patcher = ModelPatcher(ComfyVRMBGWrapper(model).eval(), load_device=mm.get_torch_device(), offload_device=torch.device("cpu"))
        self.loaded_identity = identity
        return self.patcher


class VRMBG3V6Loader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for root in folder_paths.get_folder_paths("vrmbg3_v6"):
            paths += [str(p.relative_to(root)).replace("\\", "/") for p in Path(root).rglob("*.safetensors")]
        # Always default to ep4, even if missing: report missing weights rather
        # than silently substituting an old checkpoint on another machine.
        names = [DEFAULT_MODEL_NAME] + sorted(set(paths) - {DEFAULT_MODEL_NAME})
        return {"required": {"model_name": (names, {"default": DEFAULT_MODEL_NAME})}}
    RETURN_TYPES = ("VRMBG3_V6_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "RH-Nodes/matting"
    def load(self, model_name=DEFAULT_MODEL_NAME):
        for root in folder_paths.get_folder_paths("vrmbg3_v6"):
            p = (Path(root) / model_name).resolve()
            if p.is_relative_to(Path(root).resolve()) and p.is_file(): return (ModelHandle(p),)
        raise FileNotFoundError(f"VRMBG3 weights not found: {model_name}. Place the authorized safetensors file in {MODEL_FOLDER}; weights are not distributed in RH-nodes.")
    @classmethod
    def IS_CHANGED(cls, model_name):
        for root in folder_paths.get_folder_paths("vrmbg3_v6"):
            p = (Path(root) / model_name).resolve()
            if p.is_relative_to(Path(root).resolve()) and p.is_file(): return (p.stat().st_size, p.stat().st_mtime_ns)
        return float("nan")


def alpha_postprocess_inputs():
    return {
        "alpha_shrink": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
            "tooltip": "Alpha收缩半径（原视频像素）；0关闭，先收缩再羽化。"}),
        "alpha_feather": ("FLOAT", {"default": 0., "min": 0., "max": 64., "step": .1,
            "tooltip": "Alpha羽化强度（高斯sigma，原视频像素）；0关闭。"}),
        "enable_alpha_unmult": ("BOOLEAN", {"default": False, "display_name": "开启alpha反算",
            "tooltip": "使用输出遮罩恢复前景RGB，不改Alpha；关闭保留输入RGB。"}),
        "color_recovery_method": (["multilevel", "legacy_q"], {"default": "multilevel",
            "display_name": "颜色反算方法", "tooltip": "multilevel：邻域前景/背景估计，适合实体与模糊；legacy_q：旧Q纯色底反算，适合纯光效。"}),
    }


def postprocess_alpha(alpha, shrink_kernel=None, feather=0.):
    # Continuous grayscale erosion/blur; the model's temporal history is unchanged.
    if shrink_kernel is not None:
        alpha = cv2.erode(alpha, shrink_kernel, borderType=cv2.BORDER_REPLICATE)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather, sigmaY=feather,
                                 borderType=cv2.BORDER_REPLICATE)
    return alpha


class VRMBG3V6Matting:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("VRMBG3_V6_MODEL",), "images": ("IMAGE",),
                "max_pixels": ("INT", {"default": 1048576, "min": 65536, "max": 1048576, "step": 1024}),
                "precision": (["bf16", "auto", "fp32"],),
                "temporal": ("BOOLEAN", {"default": True, "tooltip": "Treat IMAGE batch as consecutive frames; resets at every node call. Turn off for unrelated images."}),
                "offload_after": ("BOOLEAN", {"default": True})},
                "optional": alpha_postprocess_inputs()}
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("rgba", "alpha", "checker_preview", "run_info")
    FUNCTION = "infer"
    CATEGORY = "RH-Nodes/matting"
    def infer(self, model, images, max_pixels=1048576, precision="bf16", temporal=True, offload_after=True, _file_backed=False, _rgba_buffer=None, alpha_shrink=0, alpha_feather=0., enable_alpha_unmult=False, color_recovery_method="multilevel"):
        total_start = time.perf_counter()
        if not isinstance(alpha_shrink, int) or alpha_shrink < 0:
            raise ValueError("alpha_shrink must be a nonnegative integer pixel radius")
        if not math.isfinite(alpha_feather) or alpha_feather < 0:
            raise ValueError("alpha_feather must be a finite nonnegative sigma")
        shrink_kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                         (2 * alpha_shrink + 1, 2 * alpha_shrink + 1)) if alpha_shrink else None)
        check_images(images)
        validate_seconds = time.perf_counter() - total_start
        n, h, w, _ = images.shape
        rgb = images[..., :3].detach().to("cpu", dtype=torch.float32)
        plan = resolution_plan(w, h, max_pixels)
        device = mm.get_torch_device()
        if precision not in ("auto", "bf16", "fp32"): raise ValueError("Unsupported precision")
        use_bf16 = device.type == "cuda" and precision != "fp32" and torch.cuda.is_bf16_supported()
        if precision == "bf16" and device.type == "cuda" and not use_bf16:
            raise ValueError("CUDA BF16 unsupported; choose fp32 (no silent precision change)")
        allocation_start = time.perf_counter()
        if _file_backed:
            from .batch_memory import mapped_tensor
            if _rgba_buffer is not None:
                if (_rgba_buffer.shape != (n, h, w, 4) or _rgba_buffer.dtype != torch.float32
                    or _rgba_buffer.device.type != "cpu" or rgb.data_ptr() != _rgba_buffer.data_ptr()
                    or rgb.stride() != _rgba_buffer[..., :3].stride()):
                    raise ValueError("Invalid internal shared RGB/RGBA buffer")
                rgba = _rgba_buffer
            else:
                rgba = mapped_tensor((n, h, w, 4))
            # Foreground alpha is a view of the RGBA storage, not another full batch.
            result = rgba[..., 3]
        else:
            result = torch.empty((n, h, w), dtype=torch.float32, device="cpu")
        allocation_seconds = time.perf_counter() - allocation_start
        progress = ProgressBar(n)
        start = time.perf_counter()
        preprocess_seconds = forward_alpha_seconds = 0.
        cuda_events = []
        with model.lock:
            load_start = time.perf_counter()
            patcher = model.ensure_loaded()
            try:
                # Native BRIA modules need full FP32 weights; ComfyUI manages other model offloading.
                mm.load_models_gpu([patcher], memory_required=int(3.5 * (1 << 30) * plan["pixels"] / 1048576), force_full_load=True)
                net = patcher.model.eval()
                model_load_seconds = time.perf_counter() - load_start
                mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
                previous_rgb = previous_alpha = None
                with torch.inference_mode():
                    for i in range(n):
                        mm.throw_exception_if_processing_interrupted()
                        prepare_start = time.perf_counter()
                        if (w, h) != (plan["content_width"], plan["content_height"]):
                            # Match training RGB area resize, without letterboxing/cropping.
                            rgb8 = np.rint(rgb[i].numpy() * 255).astype(np.uint8)
                            resized = cv2.resize(rgb8, (plan["content_width"], plan["content_height"]), interpolation=cv2.INTER_AREA)
                            x = torch.from_numpy(resized).permute(2, 0, 1)[None].to(device).float().div_(255)
                        else:
                            x = rgb[i:i + 1].permute(0, 3, 1, 2).to(device).contiguous()
                        x = F.pad(x, (0, plan["pad_right"], 0, plan["pad_bottom"]), mode="replicate")
                        x = (x - mean) / std
                        history = torch.zeros_like(x) if previous_rgb is None else previous_rgb * previous_alpha
                        paired = torch.cat([x, history], dim=1).contiguous(memory_format=torch.channels_last)
                        preprocess_seconds += time.perf_counter() - prepare_start
                        forward_start = time.perf_counter()
                        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
                        if events: events[0].record()
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                        with context:
                            logits = net(paired)[-1]
                            if logits.shape != x[:, :1].shape:
                                raise ValueError(f"Unexpected BRIA output shape: {tuple(logits.shape)}")
                            # History matches the trained VideoAdapter: sigmoid in model output dtype.
                            history_alpha = logits.sigmoid()
                        if events:
                            events[1].record()
                            cuda_events.append(events)
                        alpha = logits.float().sigmoid()[:, :, :plan["content_height"], :plan["content_width"]]
                        if alpha.shape[-2:] != (h, w): alpha = F.interpolate(alpha, (h, w), mode="bilinear", align_corners=False)
                        if not torch.isfinite(alpha).all(): raise ValueError("Nonfinite alpha; output not repaired")
                        # Writing a strided alpha plane with NumPy avoids a parallel
                        # CPU tensor copy without changing the FP32 values.
                        output_alpha = postprocess_alpha(alpha[0, 0].cpu().numpy(), shrink_kernel, alpha_feather)
                        np.copyto(result[i].numpy(), output_alpha)
                        forward_alpha_seconds += time.perf_counter() - forward_start
                        previous_rgb, previous_alpha = (x, history_alpha.detach()) if temporal else (None, None)
                        progress.update(1)
                del previous_rgb, previous_alpha, paired, logits
            except torch.cuda.OutOfMemoryError as ex:
                raise RuntimeError("VRMBG3 ran out of VRAM. Lower max_pixels or unload other models; no automatic quality change was applied.") from ex
            finally:
                if offload_after:
                    mm.unload_model_and_clones(patcher, unload_additional_models=False)
                    mm.soft_empty_cache()
        render_start = time.perf_counter()
        yy = torch.arange(h).view(h, 1); xx = torch.arange(w).view(1, w)
        checker = torch.where(((yy // 32 + xx // 32) % 2).bool(), 210 / 255., 170 / 255.).unsqueeze(-1)
        if _file_backed or enable_alpha_unmult:
            if _file_backed:
                preview = mapped_tensor((n, h, w, 3))
            else:
                rgba = torch.cat([rgb, result.unsqueeze(-1)], dim=-1)
                preview = torch.empty((n, h, w, 3), dtype=torch.float32, device="cpu")
            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                if enable_alpha_unmult:
                    frame_rgb = recover_foreground_rgb(rgb[i].numpy(), result[i].numpy(), color_recovery_method)
                else:
                    frame_rgb = rgb[i].numpy()
                if enable_alpha_unmult or _rgba_buffer is None:
                    np.copyto(rgba[i, ..., :3].numpy(), frame_rgb)
                a = result[i].numpy()[..., None]
                np.copyto(preview[i].numpy(), frame_rgb * a + checker.numpy() * (1 - a))
        else:
            rgba = torch.cat([rgb, result.unsqueeze(-1)], dim=-1)
            preview = rgb * result.unsqueeze(-1) + checker * (1 - result.unsqueeze(-1))
        render_seconds = time.perf_counter() - render_start
        cuda_forward_seconds = sum(a.elapsed_time(b) for a, b in cuda_events) / 1000 if cuda_events else None
        timings = {"input_validation": validate_seconds, "output_allocation": allocation_seconds,
                   "model_load": model_load_seconds, "preprocess_and_transfer_wall": preprocess_seconds,
                   "forward_and_alpha_wall": forward_alpha_seconds, "CUDA_forward_event": cuda_forward_seconds,
                   "RGBA_preview_prepare": render_seconds, "total": time.perf_counter() - total_start}
        info = json.dumps({**getattr(model, "model_info", {}), "frames": n, "native_size": [w, h], "plan": plan,
            "precision": "BF16 forward / FP32 sigmoid" if use_bf16 else "FP32", "temporal": bool(temporal),
            "history_reset": "every infer call", "alpha": "foreground=1, no minmax/threshold",
            "alpha_postprocess": {"shrink_radius_px": alpha_shrink, "feather_sigma_px": alpha_feather,
                                  "order": "shrink then feather", "resolution": "native output", "temporal_history_modified": False},
            "foreground_color_recovery": {"enabled": bool(enable_alpha_unmult),
                "method": color_recovery_method,
                "algorithm": "PyMatting fixed-Alpha multilevel foreground/background estimation" if color_recovery_method == "multilevel" else "Mekajiki Mask-Guided UnMult / Q / 2026-10-27",
                "alpha_adjustment": False, "color_space": "source encoded RGB",
                "thin_edge_protection": "legacy Q only: non-near-black whole-frame bypass" if color_recovery_method == "legacy_q" else "local foreground/background regularization",
                "background": "spatially varying estimate" if color_recovery_method == "multilevel" else "one estimated representative color per frame"},
            "seconds": time.perf_counter() - start, "timings_seconds": timings}, ensure_ascii=False)
        return (rgba, result, preview, info)


class VRMBG3LoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": "14622bc351ce4d70b2520e8ac346cbe5.mov", "tooltip": "ComfyUI/input-relative path or absolute video path"}),
               "frame_limit": ("INT", {"default": 300, "min": 0, "max": 10000, "tooltip": "0=all; do not concatenate unrelated videos."}),
               "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000})}}
    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("images", "fps", "source_path", "video_info")
    FUNCTION = "load"
    CATEGORY = "RH-Nodes/matting"
    def load(self, video_path, frame_limit=300, start_frame=0):
        p = Path(video_path.strip().strip('"')).expanduser()
        if not p.is_absolute(): p = Path(folder_paths.get_input_directory()) / p
        p = p.resolve()
        if not p.is_file(): raise FileNotFoundError(str(p))
        frames, pts = [], []
        with av.open(str(p)) as container:
            stream = container.streams.video[0]
            if not stream.average_rate: raise ValueError("Video has no valid frame rate")
            fps = float(stream.average_rate)
            for idx, frame in enumerate(container.decode(stream)):
                mm.throw_exception_if_processing_interrupted()
                if idx < start_frame: continue
                arr = frame.to_ndarray(format="rgb24")
                if frames and arr.shape != frames[0].shape: raise ValueError("Changing frame dimensions unsupported")
                frames.append(arr); pts.append(float(frame.pts * frame.time_base) if frame.pts is not None else idx / fps)
                if len(frames) * arr.size * 4 > 4 * (1 << 30): raise ValueError("Decoded IMAGE batch would exceed 4GiB; reduce frame_limit")
                if frame_limit and len(frames) >= frame_limit: break
        if not frames: raise ValueError("Empty video slice")
        if len(pts) > 1 and not np.allclose(np.diff(pts), 1 / fps, atol=1e-4):
            raise ValueError("Variable frame rate detected. Convert to constant fps before matting; not silently retimed.")
        images = torch.from_numpy(np.stack(frames)).float().div_(255)
        return images, fps, str(p), json.dumps({"path": str(p), "frames": len(frames), "fps": fps, "start_frame": start_frame,
                  "pts": pts, "size": [images.shape[2], images.shape[1]], "audio_note": "Use original source audio externally if needed; saver is silent."})
    @classmethod
    def IS_CHANGED(cls, video_path, frame_limit=300, start_frame=0):
        p = Path(video_path.strip().strip('"')).expanduser()
        if not p.is_absolute(): p = Path(folder_paths.get_input_directory()) / p
        return (p.stat().st_mtime_ns, p.stat().st_size, frame_limit, start_frame) if p.is_file() else float("nan")


from .video_nodes import VRMBG3SaveVideo, VRMBG3V6MattingVideo


NODE_CLASS_MAPPINGS = {"VRMBG3V6Loader": VRMBG3V6Loader, "VRMBG3V6Matting": VRMBG3V6Matting,
                       "VRMBG3V6LoadVideo": VRMBG3LoadVideo, "VRMBG3V6SaveVideo": VRMBG3SaveVideo,
                       "VRMBG3V6MattingVideo": VRMBG3V6MattingVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"VRMBG3V6Loader": "VRMBG3 · 模型加载 (默认 381 ep4)", "VRMBG3V6Matting": "VRMBG3 · 视频时序去背 (V6)",
                            "VRMBG3V6LoadVideo": "VRMBG3 · 加载视频帧", "VRMBG3V6SaveVideo": "VRMBG3 · 保存透明 MOV + 节点预览",
                            "VRMBG3V6MattingVideo": "VRMBG3 · 视频去背 (原生 Load Video)"}
