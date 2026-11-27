"""CorridorKey 2.0 host-dependency experiment; no installs on import or execution."""
from comfy_api.latest import ComfyExtension, io
from .nodes import CorridorKey2Native

NODE_CLASS_MAPPINGS = {"CorridorKey2Native": CorridorKey2Native}
NODE_DISPLAY_NAME_MAPPINGS = {"CorridorKey2Native": "纯色底去背（宿主依赖版）"}


class CorridorKey2NativeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CorridorKey2Native]


async def comfy_entrypoint() -> CorridorKey2NativeExtension:
    return CorridorKey2NativeExtension()
