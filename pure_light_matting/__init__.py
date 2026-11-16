"""Standalone lightweight pure-light IMAGE/VIDEO matting node. No AI weights."""
from .light_effects import CorridorKey2LightEffectsMatting

# Preserve the old ID so workflows survive moving between plugin folders.
NODE_CLASS_MAPPINGS = {"CorridorKey2LightEffectsMatting": CorridorKey2LightEffectsMatting}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKey2LightEffectsMatting": "纯光效去背"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
