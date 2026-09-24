"""ComfyUI adapters for original-pixel restoration."""

from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, PngImagePlugin
import torch

from .core.pipeline import Config, process_pair
from .core.precision import OUTPUT_MODES, load_rgb, save_png


CATEGORY = "image/Original Pixel Restore"


def _folder_paths():
    import folder_paths

    return folder_paths


def to_u8(tensor):
    return np.clip(
        np.rint(tensor.detach().cpu().float().numpy() * 255), 0, 255
    ).astype(np.uint8)


def check_image(tensor, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 4
        or tensor.shape[-1] != 3
        or tensor.shape[0] < 1
        or min(tensor.shape[1:3]) < 16
    ):
        raise ValueError(
            f"{name}: expected nonempty BxHxWx3 RGB, each side >=16"
        )
    if (
        not torch.isfinite(tensor).all()
        or tensor.min() < 0
        or tensor.max() > 1
    ):
        raise ValueError(f"{name}: expected finite SDR values in [0,1]")


class RestoreOriginalPixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": (
                    "IMAGE",
                    {
                        "tooltip": "原图；尺寸及遮罩外张量值保持不变。"
                    },
                ),
                "edited": (
                    "IMAGE",
                    {
                        "tooltip": "AI 编辑图；尺寸可不同，白色 edit_mask 表示使用生成内容。"
                    },
                ),
                "method": (["B", "A"], {"default": "B"}),
                "blend": (
                    ["feather", "poisson", "adaptive"],
                    {
                        "default": "feather",
                        "tooltip": "feather 适合细小编辑；adaptive 适合大物体边界；poisson 为旧版求解器。",
                    },
                ),
                "high": (
                    "FLOAT",
                    {"default": 2.7, "min": 0.1, "max": 20.0, "step": 0.1},
                ),
                "low": (
                    "FLOAT",
                    {"default": 1.55, "min": 0.05, "max": 20.0, "step": 0.05},
                ),
                "padding": ("INT", {"default": 1, "min": 0, "max": 32}),
                "feather": ("INT", {"default": 2, "min": 0, "max": 32}),
                "flow_limit": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "B 方法在原图网格上的局部位移上限。",
                    },
                ),
                "capture_steps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "保留 Steps 节点所需中间数组；关闭可减少内存。",
                    },
                ),
            },
            "optional": {
                "solver_precision": (
                    ["float32", "float64"],
                    {
                        "default": "float32",
                        "tooltip": "float64 同时收紧 Poisson 收敛条件。",
                    },
                ),
                "edit_mask": (
                    "MASK",
                    {
                        "tooltip": "原图分辨率二值遮罩，>0.5 为编辑区；严格限制时将 padding 和 feather 设为 0。"
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "STRING",
        "OPR_DIAGNOSTICS",
    )
    RETURN_NAMES = (
        "restored",
        "support_mask",
        "core_mask",
        "alpha",
        "corrected_generated",
        "diagnostics_json",
        "steps_data",
    )
    FUNCTION = "restore"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "B 方法包含受限配准、鲁棒校色、图割和限制范围融合；A 为全局配准基线。"
        "输出保持浮点，最终量化只在 Save Precision 中发生。自动遮罩不保证语义正确。"
    )

    def restore(
        self,
        original,
        edited,
        method="B",
        blend="feather",
        high=2.7,
        low=1.55,
        padding=1,
        feather=2,
        flow_limit=5.0,
        capture_steps=True,
        edit_mask=None,
        solver_precision="float32",
    ):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from comfy.utils import ProgressBar

        check_image(original, "original")
        check_image(edited, "edited")
        if len(edited) not in (1, len(original)):
            raise ValueError("edited batch must be 1 or match original batch")
        if edit_mask is not None:
            if (
                not isinstance(edit_mask, torch.Tensor)
                or edit_mask.ndim != 3
                or len(edit_mask) not in (1, len(original))
                or tuple(edit_mask.shape[1:]) != tuple(original.shape[1:3])
            ):
                raise ValueError(
                    "edit_mask must be BxHxW at original resolution, batch 1 or "
                    "original batch"
                )
            if (
                not torch.isfinite(edit_mask).all()
                or edit_mask.min() < 0
                or edit_mask.max() > 1
            ):
                raise ValueError("edit_mask must contain finite values in [0,1]")

        config = Config(
            high=high,
            low=low,
            padding=padding,
            feather=feather,
            flow_limit=flow_limit,
            blend=blend,
            solver_precision=solver_precision,
        )
        results, images, stats = [], [], []
        progress = ProgressBar(len(original))
        for index in range(len(original)):
            throw_exception_if_processing_interrupted()
            original_array = (
                original[index].detach().cpu().float().numpy() * np.float32(255)
            )
            edited_array = (
                edited[min(index, len(edited) - 1)].detach().cpu().float().numpy()
                * np.float32(255)
            )
            mask = (
                None
                if edit_mask is None
                else edit_mask[min(index, len(edit_mask) - 1)]
                .detach()
                .cpu()
                .numpy()
                > 0.5
            )
            result = process_pair(
                original_array,
                edited_array,
                method,
                config,
                mask,
                capture=capture_steps,
            )
            dtype = (
                torch.float64 if original.dtype == torch.float64 else torch.float32
            )
            restored = original[index].detach().cpu().to(dtype).clone()
            active = torch.from_numpy(result["support"])
            restored[active] = torch.from_numpy(
                result["output_float"] / np.float32(255)
            ).to(restored.dtype)[active]
            assert torch.equal(
                restored[~active], original[index].detach().cpu()[~active]
            )
            result["stats"]["outside_tensor_exact"] = True
            result["stats"]["exact_original_tensor_fraction"] = float(
                torch.all(restored == original[index].detach().cpu(), dim=-1)
                .float()
                .mean()
            )
            if capture_steps:
                result["original"] = original_array
                result["edited"] = edited_array
            images.append(restored)
            results.append(result)
            stats.append(result["stats"])
            progress.update(1)
            throw_exception_if_processing_interrupted()

        def masks(key):
            return torch.from_numpy(
                np.stack([result[key] for result in results]).astype(np.float32)
            )

        corrected = torch.from_numpy(
            np.stack([result["corrected"] for result in results]).astype(np.float32)
            / 255
        )
        steps = {
            "results": results if capture_steps else None,
            "stats": stats,
            "captured": capture_steps,
        }
        return (
            torch.stack(images),
            masks("support"),
            masks("core"),
            masks("alpha"),
            corrected,
            json.dumps(stats, ensure_ascii=False, indent=2),
            steps,
        )


class LoadOriginalICC:
    @classmethod
    def INPUT_TYPES(cls):
        folder_paths = _folder_paths()
        root = Path(folder_paths.get_input_directory())
        files = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        )
        return {"required": {"image": (files, {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "OPR_ICC")
    RETURN_NAMES = ("image", "icc_profile")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "保留已解码 8 位图像和 16 位 RGB/灰度 PNG 数值及原始 ICC；拒绝透明、"
        "动画和 HDR/CMYK 输入。"
    )

    def load(self, image):
        folder_paths = _folder_paths()
        path = folder_paths.get_annotated_filepath(image)
        rgb, profile = load_rgb(path)
        return torch.from_numpy(rgb.astype(np.float32) / np.float32(255))[None], profile

    @classmethod
    def IS_CHANGED(cls, image):
        folder_paths = _folder_paths()
        return hashlib.sha256(
            Path(folder_paths.get_annotated_filepath(image)).read_bytes()
        ).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        folder_paths = _folder_paths()
        return (
            True
            if folder_paths.exists_annotated_filepath(image)
            else f"Image not found: {image}"
        )


class SaveOriginalICC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": "OriginalPixelRestore/B"},
                ),
            },
            "optional": {
                "icc_profile": ("OPR_ICC",),
                "diagnostics_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "旧版 8 位就近舍入保存。平滑渐变请使用 Save Precision 的 16bit 或 "
        "8bit_dither。"
    )

    def save(
        self,
        images,
        filename_prefix="OriginalPixelRestore/B",
        icc_profile=None,
        diagnostics_json=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        from comfy.cli_args import args

        folder_paths = _folder_paths()
        check_image(images, "images")
        output = folder_paths.get_output_directory()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output, images.shape[2], images.shape[1]
        )
        files = []
        for index, tensor in enumerate(images):
            metadata = PngImagePlugin.PngInfo()
            if not args.disable_metadata:
                if prompt is not None:
                    metadata.add_text(
                        "prompt", json.dumps(prompt, ensure_ascii=False)
                    )
                for key, value in (extra_pnginfo or {}).items():
                    metadata.add_text(key, json.dumps(value, ensure_ascii=False))
                if diagnostics_json:
                    metadata.add_text("original_pixel_restore", diagnostics_json)
            filename = (
                f'{name.replace("%batch_num%", str(index))}_{counter:05}_.png'
            )
            options = (
                {"icc_profile": icc_profile["icc"]}
                if icc_profile and icc_profile.get("icc")
                else {}
            )
            Image.fromarray(to_u8(tensor)).save(
                Path(folder) / filename,
                pnginfo=metadata,
                compress_level=4,
                **options,
            )
            files.append(
                {"filename": filename, "subfolder": subfolder, "type": "output"}
            )
            counter += 1
        return {"ui": {"images": files}}


class SavePrecisionICC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "support_mask": (
                    "MASK",
                    {
                        "tooltip": "连接 Restore Pixels 的 support_mask；抖动仅限此范围。"
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "OriginalPixelRestore/Precision"},
                ),
                "output_mode": (
                    list(OUTPUT_MODES),
                    {
                        "default": "16bit",
                        "tooltip": "16bit 保留渐变；8bit_dither 仅在 support 内抖动。",
                    },
                ),
                "dither_seed": (
                    "INT",
                    {"default": 421, "min": 0, "max": 2147483647},
                ),
            },
            "optional": {
                "icc_profile": ("OPR_ICC",),
                "diagnostics_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "保存为 16 位 RGB PNG 或仅在 support 内抖动的 8 位 PNG，并保留 ICC 和"
        "工作流元数据。"
    )

    def save(
        self,
        images,
        support_mask,
        filename_prefix="OriginalPixelRestore/Precision",
        output_mode="16bit",
        dither_seed=421,
        icc_profile=None,
        diagnostics_json=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        from comfy.cli_args import args

        folder_paths = _folder_paths()
        check_image(images, "images")
        if output_mode not in OUTPUT_MODES:
            raise ValueError("Invalid output mode")
        if (
            not isinstance(support_mask, torch.Tensor)
            or support_mask.ndim != 3
            or len(support_mask) not in (1, len(images))
            or tuple(support_mask.shape[1:]) != tuple(images.shape[1:3])
        ):
            raise ValueError(
                "support_mask must match image H/W, with batch 1 or image batch size"
            )
        if (
            not torch.isfinite(support_mask).all()
            or support_mask.min() < 0
            or support_mask.max() > 1
        ):
            raise ValueError("support_mask must be finite and in [0,1]")
        output = folder_paths.get_output_directory()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output, images.shape[2], images.shape[1]
        )
        files = []
        for index, tensor in enumerate(images):
            metadata = {}
            if not args.disable_metadata:
                if prompt is not None:
                    metadata["prompt"] = json.dumps(prompt, ensure_ascii=False)
                for key, value in (extra_pnginfo or {}).items():
                    metadata[key] = json.dumps(value, ensure_ascii=False)
                if diagnostics_json:
                    metadata["original_pixel_restore"] = diagnostics_json
                metadata["opr_output"] = json.dumps(
                    {
                        "mode": output_mode,
                        "seed": int(dither_seed),
                        "dither_scope": "support_only",
                        "pipeline": "2.0-precision",
                    }
                )
            filename = (
                f'{name.replace("%batch_num%", str(index))}_{counter:05}_.png'
            )
            values = tensor.detach().cpu().double().numpy() * 255.0
            support = (
                support_mask[min(index, len(support_mask) - 1)]
                .detach()
                .cpu()
                .numpy()
                > 0
            )
            save_png(
                Path(folder) / filename,
                values,
                mode=output_mode,
                support=support,
                seed=dither_seed,
                icc_profile=(icc_profile or {}).get("icc"),
                metadata=metadata,
            )
            files.append(
                {"filename": filename, "subfolder": subfolder, "type": "output"}
            )
            counter += 1
        return {"ui": {"images": files}}


class RestoreSteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps_data": ("OPR_DIAGNOSTICS",),
                "batch_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000},
                ),
            },
            "optional": {"icc_profile": ("OPR_ICC",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("nine_step_panels", "step_summary")
    FUNCTION = "render"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "渲染一个批次项目的九步诊断图；Restore Original Pixels 必须启用 "
        "capture_steps。"
    )

    def render(self, steps_data, batch_index=0, icc_profile=None):
        if not steps_data["captured"]:
            raise ValueError("Enable capture_steps on Restore Original Pixels")
        if batch_index >= len(steps_data["results"]):
            raise ValueError("batch_index outside captured batch")
        from .diagnostics import make_panels

        return make_panels(steps_data["results"][batch_index], icc_profile)
