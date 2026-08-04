"""Dakun333's maintained collection of standalone ComfyUI image nodes."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
_IMPORT_ERRORS = []


def _register(mapping, display_mapping):
    """Register one node family without replacing keys owned by another family."""
    duplicate_keys = set(NODE_CLASS_MAPPINGS).intersection(mapping)
    if duplicate_keys:
        raise RuntimeError(f"Duplicate ComfyUI node IDs: {sorted(duplicate_keys)}")
    NODE_CLASS_MAPPINGS.update(mapping)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_mapping)


try:
    from .nodes import (
        NODE_CLASS_MAPPINGS as _COLOR_RESTORE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _COLOR_RESTORE_DISPLAY_MAPPINGS,
    )

    _register(_COLOR_RESTORE_MAPPINGS, _COLOR_RESTORE_DISPLAY_MAPPINGS)
except Exception as exc:  # Keep unrelated node families available when one dependency fails.
    _IMPORT_ERRORS.append(f"Reference Color Restore: {exc}")

try:
    from .comic_outline import ComicOutlineDetect

    _register(
        {"ComicOutlineDetect": ComicOutlineDetect},
        {"ComicOutlineDetect": "🎨 漫画轮廓检测 (Comic Outline)"},
    )
except Exception as exc:  # Keep unrelated node families available when one dependency fails.
    _IMPORT_ERRORS.append(f"Comic Outline: {exc}")

if _IMPORT_ERRORS:
    print("[Dakun333 Nodes] Some node families could not be loaded:")
    for error in _IMPORT_ERRORS:
        print(f"  - {error}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
