"""CF1 with ST1 + PO1 abnormal-correction protection."""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
try:
    from .protected_nodes import NODE_CLASS_MAPPINGS as PROTECTED_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as PROTECTED_NAMES
except ImportError as exc:
    print(f"[RH Nodes] Y367 CF1 protection requires ComfyUI V3 API: {exc}")
else:
    NODE_CLASS_MAPPINGS.update(PROTECTED_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(PROTECTED_NAMES)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
