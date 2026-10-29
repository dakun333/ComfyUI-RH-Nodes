"""CPU ComfyUI wrapper for shared-color vertical artifact removal."""
from __future__ import annotations
from dataclasses import replace
import json
import logging
import time

import numpy as np
import torch

from .core import Parameters, Prior, Settings, encode_directed8, restore
from .adaptive_scale import estimate_scale
from .transparent import estimate_scale_alpha, restore_alpha

LOGGER = logging.getLogger(__name__)
DIRECTED = "directed8（复现原8位结果）"
FLOAT = "float32（保留处理精度）"


class SharedColorBroadbandDestripe:
    CATEGORY = "RH-Nodes/restore"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("处理图像", "处理报告")
    OUTPUT_TOOLTIPS = (
        "原尺寸RGB/RGBA图像。连接透明度MASK或传入RGBA时保留Alpha，可直接Save Image保存透明PNG。默认定向8位取整；float32不进行输出8位量化。",
        "每张图的处理/拒绝状态、颜色证据、参数和改动量。无证据原样返回。",
    )
    DESCRIPTION = (
        "共享颜色宽带去竖纹：从单图学习共同RGB污染方向，分离结构和材质明暗，"
        "两轮扣除宽带残差。CPU逐张处理，不缩图、不锁亮度，无输出门控或扣除限幅。"
        "无证据返回原图不代表清理成功；真实同色波浪纹理可能误删。"
        "透明PNG请同时连接Load Image的IMAGE与MASK；隐藏RGB不参与证据，Alpha不修复、不烘焙底色。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "原始SDR图，BHWC RGB/RGBA、0–1；勿重复处理输出。"}),
            "period_min": ("FLOAT", {"default": 22.0, "min": 2.0, "max": 1023.0, "step": .5, "tooltip": "手动最小尺度，原图px；auto_scale关闭时生效，须小于最大尺度。"}),
            "period_max": ("FLOAT", {"default": 80.0, "min": 2.5, "max": 1024.0, "step": .5, "tooltip": "手动最大尺度，原图px；同时控制结构分离窗口。"}),
            "gain_first": ("FLOAT", {"default": .8, "min": 0.0, "max": 1.0, "step": .05, "tooltip": "第一轮扣除比例，越大越积极；两轮均0为完全旁路。"}),
            "gain_second": ("FLOAT", {"default": .9, "min": 0.0, "max": 1.0, "step": .05, "tooltip": "对第一轮结果重新估计残留并扣除的比例。"}),
            "guide_sigma": ("FLOAT", {"default": 12.0, "min": .1, "max": 64.0, "step": .5, "tooltip": "结构分离正则σ，0–255码值；越小越保护结构。"}),
            "along_ratio": ("FLOAT", {"default": .14, "min": .01, "max": 1.0, "step": .01, "tooltip": "竖向平滑σ=ratio×√(min×max)；过大可能漏掉快速弯曲纹带。"}),
            "across_sigma": ("FLOAT", {"default": .65, "min": .05, "max": 8.0, "step": .05, "tooltip": "污染场横向高斯σ，原图px；不模糊最终RGB。"}),
            "min_color_support": ("INT", {"default": 6, "min": 3, "max": 100, "step": 1, "tooltip": "一致颜色证据块最少数量；增加更保守。"}),
            "max_color_gradient": ("FLOAT", {"default": 1.8, "min": .1, "max": 10.0, "step": .1, "tooltip": "拟合证据块最大梯度，码值/px；仅影响学习来源。"}),
            "color_consensus": ("FLOAT", {"default": .94, "min": .5, "max": .999, "step": .001, "tooltip": "颜色方向余弦一致性门槛，越高越严格。"}),
            "output_mode": ([DIRECTED, FLOAT], {"default": DIRECTED, "tooltip": "定向8位误差反馈取整，或float32保留小数；标准Save Image仍为8位保存。"}),
        }, "optional": {
            "auto_scale": ("BOOLEAN", {"default": True, "tooltip": "逐图从共享颜色残差估计尺度，只替换上下界。固定3–96px探测范围，无证据原样返回。旧API省略保持手动。"}),
            "include_near_white": ("BOOLEAN", {"default": True, "tooltip": "允许近白证据参与颜色和尺度学习，不改变其余门槛；校正仍应用全图。严格复现历史须同时关闭auto_scale。"}),
            "mask": ("MASK", {"default": None, "tooltip": "可选透明度：Load Image MASK，1透明、0不透明；输出RGBA。不是选择性去纹蒙版。RGBA和MASK同时输入须一致。"}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, period_min, period_max):
        if period_min >= period_max:
            return "period_min 必须小于 period_max（以原图像素为单位）。"
        return True

    def process(self, image, period_min=22.0, period_max=80.0,
                gain_first=.8, gain_second=.9, guide_sigma=12.0,
                along_ratio=.14, across_sigma=.65, min_color_support=6,
                max_color_gradient=1.8, color_consensus=.94, output_mode=DIRECTED,
                auto_scale=False, include_near_white=True, mask=None):
        if not isinstance(include_near_white, bool):
            raise ValueError("include_near_white须为布尔值")
        if not isinstance(auto_scale, bool):
            raise ValueError("auto_scale必须为布尔值")
        prior = Prior(Parameters(period_min=period_min, period_max=period_max, guide_sigma=guide_sigma))
        settings = Settings(gains=(gain_first, gain_second), along_ratio=along_ratio,
                            min_support=min_color_support, max_gradient=max_color_gradient,
                            cluster_cosine=color_consensus, include_near_white=include_near_white)
        prior.validate()
        settings.validate()
        if not np.isfinite(across_sigma) or across_sigma <= 0:
            raise ValueError("across_sigma必须为正数")
        if output_mode not in (DIRECTED, FLOAT):
            raise ValueError("未知output_mode")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError("需要B×H×W×3（或RGBA）的IMAGE张量")
        if image.shape[0] < 1 or min(image.shape[1:3]) < 2 or not image.is_floating_point():
            raise ValueError("IMAGE必须是非空浮点批次，宽高至少2px")
        source = image.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(source).all() or source.min() < 0 or source.max() > 1:
            raise ValueError("IMAGE需为0–1有限数值；本节点不自动修正HDR、NaN或越界输入")

        mask_alpha = None
        mask_roundtrip_corrected = False
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or not mask.is_floating_point() or mask.ndim not in (2, 3):
                raise ValueError("mask须为HW或B×H×W浮点MASK，1透明、0不透明")
            masks = mask.detach().to(device="cpu", dtype=torch.float32).numpy()
            if masks.ndim == 2:
                masks = masks[None]
            if not masks.size or not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
                raise ValueError("mask须为非空0–1有限值")
            if masks.shape[0] not in (1, source.shape[0]):
                raise ValueError("mask批次数须为1或与IMAGE一致")
            if masks.shape[1:] != source.shape[1:3]:
                if masks.shape[1:] == (64, 64) and not np.any(masks):
                    masks = np.zeros((masks.shape[0], *source.shape[1:3]), np.float32)
                else:
                    raise ValueError("透明度mask尺寸须与IMAGE一致，不自动缩放")
            mask_alpha = np.broadcast_to(1-masks, source.shape[:3]).copy()
            alpha_codes = mask_alpha * 255
            nearest_alpha = np.rint(alpha_codes)
            if np.all(np.abs(alpha_codes-nearest_alpha) < 1e-4):
                mask_alpha = nearest_alpha.astype(np.float32) / 255
                mask_roundtrip_corrected = True
            if source.shape[-1] == 4 and not np.allclose(mask_alpha, source[..., 3], atol=1e-6, rtol=0):
                raise ValueError("RGBA的alpha与输入mask不一致")

        try:
            from comfy.utils import ProgressBar
            progress = ProgressBar(len(source))
        except ImportError:
            progress = None
        results, reports = [], []
        for index, frame in enumerate(source):
            started = time.perf_counter()
            alpha = frame[:, :, 3] if frame.shape[2] == 4 else (mask_alpha[index] if mask_alpha is not None else None)
            alpha_aware = alpha is not None and bool(np.any(alpha < 1))
            alpha_source = "rgba" if frame.shape[2] == 4 else ("mask" if mask_alpha is not None else "none")
            codes = np.ascontiguousarray(frame[:, :, :3] * 255, dtype=np.float32)
            rgb8 = np.clip(np.rint(codes), 0, 255).astype(np.uint8)
            frac = codes - rgb8.astype(np.float32)
            frac[np.abs(frac) < 1e-4] = 0
            effective_prior = prior
            scale_info = None
            if gain_first == 0 and gain_second == 0:
                z = rgb8.astype(np.float32)
                info = {"status": "bypassed_zero_gains", "evidence": {}}
                if auto_scale:
                    effective_prior = None
                    scale_info = {"status": "skipped_zero_gains"}
            else:
                if auto_scale:
                    if alpha_aware:
                        selected, scale_info = estimate_scale_alpha(rgb8, alpha, include_near_white=include_near_white)
                    else:
                        selected, scale_info = estimate_scale(rgb8, include_near_white=include_near_white)
                    effective_prior = (None if selected is None else
                        replace(prior, base=replace(prior.base,
                            period_min=selected.base.period_min,
                            period_max=selected.base.period_max)))
                if effective_prior is None:
                    z = rgb8.astype(np.float32)
                    info = {"status": "abstained_scale", "evidence": {},
                            "abstain_reason": scale_info["status"]}
                else:
                    if alpha_aware:
                        z, info, _ = restore_alpha(rgb8, alpha, effective_prior, settings, across_sigma=across_sigma)
                    else:
                        z, info, _ = restore(rgb8, effective_prior, settings, across_sigma=across_sigma)
            processed = info["status"] == "processed_not_quality_approved"
            if not processed:
                restored = (np.concatenate((frame[:, :, :3], alpha[:, :, None]), axis=2)
                            if alpha is not None else frame.copy())
            else:
                target = np.clip(z + frac, 0, 255)
                encoding_target = target
                if alpha_aware and output_mode == DIRECTED:
                    encoding_target = target.copy()
                    encoding_target[alpha < .05] = rgb8[alpha < .05]
                rgb = (encode_directed8(rgb8, encoding_target).astype(np.float32) / 255
                       if output_mode == DIRECTED else target / 255)
                if alpha_aware:
                    protected = alpha < .05
                    rgb[protected] = frame[:, :, :3][protected]
                restored = np.concatenate((rgb, alpha[:, :, None]), axis=2) if alpha is not None else rgb
            results.append(torch.from_numpy(np.ascontiguousarray(restored)))
            evidence = info.get("evidence", {})
            report = {
                "batch_index": index, "status": info["status"],
                "size": [int(frame.shape[1]), int(frame.shape[0])],
                "output_mode": output_mode,
                "include_near_white": include_near_white,
                "estimated_color_direction": evidence.get("color_direction"),
                "candidate_tiles": evidence.get("candidate_tile_count", 0),
                "support_tiles": evidence.get("support_tile_count", 0),
                "abstain_reason": info.get("abstain_reason", evidence.get("reason")),
                "correction_rgb_rmse_codes": float(np.sqrt(np.mean((rgb8.astype(np.float32) - z)**2))),
                "input_had_subcode_precision": bool(np.any(frac != 0)),
                "alpha_passed_through": alpha is not None,
                "alpha_source": alpha_source,
                "alpha_aware": alpha_aware,
                "alpha_modified": False,
                "mask_8bit_roundtrip_corrected": mask_roundtrip_corrected,
                "transparent_pixel_count": int(np.count_nonzero(alpha == 0)) if alpha is not None else 0,
                "hidden_rgb_excluded": alpha_aware,
                "seconds": round(time.perf_counter() - started, 4),
            }
            report["scale_mode"] = "auto" if auto_scale else "manual"
            report["effective_period_min"] = (effective_prior.base.period_min if effective_prior else None)
            report["effective_period_max"] = (effective_prior.base.period_max if effective_prior else None)
            if scale_info is not None:
                report["scale_estimation"] = {k: v for k, v in scale_info.items()
                    if k not in ("color_evidence", "spectral_tiles", "frequency", "normalized_spectrum")}
                color_evidence = scale_info.get("color_evidence", {})
                report["scale_estimation"]["color_evidence"] = {k: v for k, v in color_evidence.items()
                    if k not in ("tiles", "supporting_tiles")}
                if effective_prior is not None:
                    LOGGER.info("Shared-color broadband frame %s: auto scale %.4f–%.4f px",
                                index, effective_prior.base.period_min, effective_prior.base.period_max)
            reports.append(report)
            if not processed:
                LOGGER.info("Shared-color broadband frame %s: %s (%s)", index, report["status"], report["abstain_reason"])
            if progress is not None:
                progress.update(1)
        report_text = json.dumps({
            "method": "shared_color_broadband", "not_continuous_field": True,
            "luminance_locked": False, "output_gates": False, "correction_caps": False,
            "parameters": {"period_min": period_min, "period_max": period_max, "auto_scale": auto_scale,
                "include_near_white": include_near_white,
                "gain_first": gain_first, "gain_second": gain_second, "guide_sigma": guide_sigma,
                "along_ratio": along_ratio, "across_sigma": across_sigma,
                "min_color_support": min_color_support, "max_color_gradient": max_color_gradient,
                "color_consensus": color_consensus},
            "frames": reports,
            "warning": "单图估计，不保证无损。无证据返回原图≠清理成功；真实同色波浪纹理仍可能误删。",
        }, ensure_ascii=False, indent=2, allow_nan=False)
        return (torch.stack(results), report_text)


NODE_CLASS_MAPPINGS = {"SharedColorBroadbandDestripe": SharedColorBroadbandDestripe}
NODE_DISPLAY_NAME_MAPPINGS = {"SharedColorBroadbandDestripe": "共享颜色宽带去竖纹"}
