"""Standalone lightweight pure-light IMAGE/VIDEO matting node. No AI weights."""
from .light_effects import PureLightMatting

NODE_CLASS_MAPPINGS = {"PureLightMatting": PureLightMatting}
NODE_DISPLAY_NAME_MAPPINGS = {"PureLightMatting": "纯光效去背"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
