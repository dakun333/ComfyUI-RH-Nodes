"""Original-pixel restoration node family."""

from .nodes import (
    LoadOriginalICC,
    RestoreOriginalPixels,
    RestoreSteps,
    SaveOriginalICC,
    SavePrecisionICC,
)
from .task_nodes import LargeObjectMask, RestrictedComposite

NODE_CLASS_MAPPINGS = {
    "OPR_LargeObjectMask": LargeObjectMask,
    "OPR_RestrictedComposite": RestrictedComposite,
    "OPR_RestoreOriginalPixels": RestoreOriginalPixels,
    "OPR_LoadImageICC": LoadOriginalICC,
    "OPR_SaveImageICC": SaveOriginalICC,
    "OPR_SaveImagePrecision": SavePrecisionICC,
    "OPR_DiagnosticSteps": RestoreSteps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OPR_LargeObjectMask": "OPR · 大物体删除遮罩 / Large Object Mask",
    "OPR_RestrictedComposite": "OPR · 仅遮罩合成 / Restricted Composite",
    "OPR_RestoreOriginalPixels": "OPR · 原像素恢复 / Restore Pixels (B/A)",
    "OPR_LoadImageICC": "OPR · 加载图像＋ICC / Load Image",
    "OPR_SaveImageICC": "OPR · 旧版8位保存 / Legacy Save PNG",
    "OPR_SaveImagePrecision": "OPR · 高精度保存＋ICC / Save Precision PNG",
    "OPR_DiagnosticSteps": "OPR · 九步诊断图 / Diagnostic Steps",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
