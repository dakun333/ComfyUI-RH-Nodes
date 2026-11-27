"""Use installed libraries unchanged; utils3d has a source-only missing-package fallback."""
from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_utils3d():
    spec = importlib.util.find_spec("utils3d")
    if spec is None:
        vendor = str(ROOT / "vendor")
        if vendor not in sys.path:
            sys.path.append(vendor)
        importlib.invalidate_caches()
    try:
        return importlib.import_module("utils3d")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "utils3d import failed. The existing package has not been replaced, upgraded or downgraded. "
            f"Fix the reported import/API problem in this environment: {exc}"
        ) from exc


def dependencies():
    versions = {"python": sys.version.split()[0], "dependency_mode": "host"}
    paths = {}
    for name, label in (("numpy", "numpy"), ("cv2", "opencv"), ("torch", "torch"),
                        ("torchvision", "torchvision"), ("transformers", "transformers"),
                        ("scipy", "scipy"), ("timm", "timm"), ("safetensors", "safetensors"),
                        ("einops", "einops"), ("huggingface_hub", "huggingface_hub")):
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"Cannot use the installed {name}: {exc}. No packages were installed or changed."
            ) from exc
        versions[label] = str(getattr(module, "__version__", "unknown"))
        paths[label] = str(Path(module.__file__).resolve())
    utils3d = load_utils3d()
    paths["utils3d"] = str(Path(utils3d.__file__).resolve())
    versions["utils3d"] = str(getattr(utils3d, "__version__", "unknown (source package)"))
    versions["utils3d_source"] = "bundled" if Path(utils3d.__file__).resolve().is_relative_to(ROOT / "vendor") else "host"
    versions["module_paths"] = paths
    versions["cuda_available"] = sys.modules["torch"].cuda.is_available()
    return versions
