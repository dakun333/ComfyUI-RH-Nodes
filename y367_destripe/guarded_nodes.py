"""Internal IMAGE/MASK adapter for CF1 ST1 + PO1 protection."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from .engine import MODEL_SHA256, FIXED_CYCLES
from .guarded_engine import run_guarded_frame, runtime_versions

ACTUAL8 = "actual8"
FLOAT32 = "float32"
ORIENTATIONS = ("vertical", "horizontal", "both")


def _interrupt():
    management = sys.modules.get("comfy.model_management")
    if management is not None:
        management.throw_exception_if_processing_interrupted()


def _as_numpy(tensor, label):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Y367-FAST {label} must be a torch.Tensor")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"Y367-FAST {label} contains NaN or infinity")
    if array.size and (array.min() < 0 or array.max() > 1):
        raise ValueError(f"Y367-FAST {label} must be in [0,1]; no implicit HDR conversion")
    return array


def _prepare_mask(mask, batch, height, width):
    if mask is None:
        return None
    array = _as_numpy(mask, "mask")
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[0] not in (1, batch):
        raise ValueError("Y367-FAST mask must be H×W or B×H×W; batch must be 1 or match image")
    if array.shape[1:] != (height, width):
        # An opaque Load Image has a zero 64×64 placeholder. Never resize real alpha.
        if not np.all(array == 0):
            raise ValueError("Y367-FAST nonzero mask size must match native image size")
        array = np.zeros((array.shape[0], height, width), np.float32)
    if array.shape[0] == 1 and batch > 1:
        array = np.broadcast_to(array, (batch, height, width))
    return array


def _mask_alpha(mask):
    alpha = np.asarray(1.0 - mask, np.float32)
    byte = np.rint(alpha * 255.0)
    canonicalized = bool(np.max(np.abs(alpha * 255.0 - byte)) <= 3e-5)
    if canonicalized:
        # Recover original PNG alpha bytes after Load Image's float32 inversion.
        alpha = np.asarray(byte / 255.0, np.float32)
    return alpha, canonicalized


class _Y367GuardedDestripe:
    VARIANT = None
    NODE_ID = None
    METHOD_NAME = None
    CATEGORY = "RH-Nodes/restore"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("处理图像", "删除量×16", "透明度MASK", "处理报告")
    OUTPUT_TOOLTIPS = (
        "actual8使用本版冻结编码与最终颜色/轮廓保护，可接标准Save Image；float32返回保护后的FLOAT。",
        "128灰为零、放大16倍的有符号改动。actual8：输入RGB−实际8位输出；float32：输入RGB−最终FLOAT。不是纯删除选择蒙版。",
        "ComfyUI透明度MASK，1透明、0不透明；不是去纹范围。",
        "算法版本、颜色/轮廓保护、方向、完整五轮、处理耗时、FLOAT/实际8位改动量与局限。",
    )
    SEARCH_ALIASES = ["Y367", "Y367-FAST", "destripe", "横纹", "竖纹", "周期去纹"]
    DESCRIPTION = "接原生8位图片；MASK 为透明度（1透明），推荐 actual8 输出。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "原生8位图的B×H×W×RGB/RGBA，0–1；支持Load Image，不静默量化非8位输入。"}),
                "orientation": (list(ORIENTATIONS), {"default": "both", "tooltip": "vertical竖纹，horizontal横纹，both联合；固定48周期。"}),
                "output_mode": ([ACTUAL8, FLOAT32], {"default": ACTUAL8, "tooltip": "actual8：本版冻结编码与保护；float32：有效FLOAT，标准Save Image仍为8位且不等于冻结编码。"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "连接Load Image透明度MASK：1透明，0不透明。非零MASK须原尺寸；单张可广播批次。RGBA+MASK须一致。"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image, orientation="both", output_mode=ACTUAL8, mask=None):
        root = Path(__file__).parent
        digest = hashlib.sha256(cls.NODE_ID.encode("utf-8"))
        files = sorted((root / "guarded_runtime").glob("*.py"))
        files += [root / "guarded_runtime/model.npz", root / "__init__.py",
                  root / "engine.py", root / "guarded_engine.py", root / "guarded_nodes.py",
                  root / "protected_nodes.py"]
        digest.update(json.dumps(runtime_versions(), sort_keys=True).encode("utf-8"))
        for path in files:
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def process(self, image, orientation="both", output_mode=ACTUAL8, mask=None):
        started = time.perf_counter()
        if orientation not in ORIENTATIONS:
            raise ValueError("Y367-FAST orientation must be vertical, horizontal or both")
        if output_mode not in (ACTUAL8, FLOAT32):
            raise ValueError("Y367-FAST output_mode must be actual8 or float32")
        frames = _as_numpy(image, "image")
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError("Y367-FAST image must be B×H×W×3 or B×H×W×4")
        batch, height, width, channels = frames.shape
        if batch < 1 or height < 2 or width < 2:
            raise ValueError("Y367-FAST empty batches/images are unsupported")
        scale = min(1.0, 640.0 / max(height, width))
        aw, ah = max(32, round(width * scale)), max(32, round(height * scale))
        if orientation in ("vertical", "both") and FIXED_CYCLES / aw >= 0.4:
            raise ValueError("Y367-FAST image is too narrow for the frozen 48-cycle operator")
        if orientation in ("horizontal", "both") and FIXED_CYCLES / ah >= 0.4:
            raise ValueError("Y367-FAST image is too short for the frozen 48-cycle operator")
        mask_array = _prepare_mask(mask, batch, height, width)
        output_rgba = channels == 4 or mask_array is not None
        outputs, previews, masks, reports = [], [], [], []
        utilities = sys.modules.get("comfy.utils")
        progress = utilities.ProgressBar(batch) if utilities is not None else None
        for index, frame in enumerate(frames):
            _interrupt()
            frame_started = time.perf_counter()
            rgb = np.ascontiguousarray(frame[..., :3] * 255.0, dtype=np.float32)
            byte_grid = bool(np.max(np.abs(rgb - np.rint(rgb))) <= 3e-5)
            if byte_grid:
                # Recover byte-grid input exactly; do not quantize arbitrary FLOAT inputs.
                rgb = np.ascontiguousarray(np.rint(rgb), dtype=np.float32)
            else:
                raise ValueError(f"{self.METHOD_NAME} requires native8 RGB (for example Load Image from an 8-bit PNG); arbitrary FLOAT input is not silently quantized")
            canonicalized = False
            if channels == 4:
                alpha = np.ascontiguousarray(frame[..., 3], dtype=np.float32)
                if mask_array is not None and np.max(np.abs(alpha - (1.0 - mask_array[index]))) > 2e-6:
                    raise ValueError(f"Y367-FAST frame {index}: RGBA alpha and MASK disagree")
            elif mask_array is not None:
                alpha, canonicalized = _mask_alpha(mask_array[index])
            else:
                alpha = np.ones((height, width), np.float32)
            target, eight, report = run_guarded_frame(rgb, alpha, orientation, variant=self.VARIANT)
            selected255 = eight.astype(np.float32) if output_mode == ACTUAL8 else target
            selected = selected255 / 255.0
            if output_rgba:
                selected = np.dstack((selected, alpha))
            outputs.append(torch.from_numpy(np.ascontiguousarray(selected, dtype=np.float32)))
            diagnostic = 128.0 + 16.0 * (rgb - selected255)
            preview = np.uint8(np.clip(np.rint(diagnostic), 0, 255)).astype(np.float32) / 255.0
            previews.append(torch.from_numpy(np.ascontiguousarray(preview)))
            returned_mask = mask_array[index] if mask_array is not None else 1.0 - alpha
            masks.append(torch.from_numpy(np.array(returned_mask, dtype=np.float32, copy=True)))
            report.update(
                index=index, wrapper_frame_seconds=time.perf_counter() - frame_started,
                output_mode=output_mode, output_channels=int(selected.shape[-1]),
                rgb_byte_grid_input=byte_grid, rgb_byte_grid_recovered=byte_grid,
                mask_byte_grid_alpha_recovered=canonicalized,
                alpha_input="RGBA" if channels == 4 else "ComfyUI MASK" if mask_array is not None else "opaque RGB",
                diagnostic_basis="actual8" if output_mode == ACTUAL8 else "final_FLOAT",
                diagnostic_gain=16, diagnostic_zero=128,
                diagnostic_clipped_fraction=float(np.any((diagnostic < 0) | (diagnostic > 255), axis=2).mean()),
            )
            reports.append(report)
            _interrupt()
            if progress is not None:
                progress.update(1)
        report_text = json.dumps({
            "method": self.METHOD_NAME, "variant": self.VARIANT, "node_id": self.NODE_ID,
            "independent_node_folder": "Y367-FAST", "orientation": orientation,
            "cycles": FIXED_CYCLES, "auto_frequency_search": False,
            "steps_per_axis": 5, "model_inferences_per_image": 12 if orientation == "both" else 6,
            "model_sha256": MODEL_SHA256, "output_mode": output_mode, "batch": batch,
            "wrapper_total_seconds": time.perf_counter() - started,
            "diagnostic_definition": "round(clip(128+16*(input_RGB-actual8_output),0,255))" if output_mode == ACTUAL8 else "round(clip(128+16*(input_RGB-final_FLOAT),0,255))",
            "frames": reports,
            "warning": "Current-runtime output/save consistency is not a visual-quality certificate; historical experiment-environment bytes may differ slightly. Color permission does not prove an artifact; edge-following true stripes may be protected. Non-native8 RGB inputs are rejected, not silently quantized. IMAGE carries no original ICC metadata.",
        }, ensure_ascii=False, indent=2, allow_nan=False)
        return torch.stack(outputs), torch.stack(previews), torch.stack(masks), report_text
