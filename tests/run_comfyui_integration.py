"""Manual real-ComfyUI smoke test using an isolated temporary base directory."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    comfyui = args.comfyui.resolve()
    if not (comfyui / "folder_paths.py").is_file():
        raise ValueError(f"Not a ComfyUI directory: {comfyui}")

    with tempfile.TemporaryDirectory(prefix="rh-nodes-comfy-") as directory:
        work = Path(directory)
        sys.path.insert(0, str(comfyui))
        sys.argv = [
            sys.argv[0],
            "--cpu",
            "--base-directory",
            str(work),
        ]
        spec = importlib.util.spec_from_file_location(
            "comfyui_rh_nodes_integration",
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        assert spec and spec.loader
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)

        expected = {
            "RCMRobustMaskedColorMatch",
            "RCMCoreFeatherMask",
            "OPR_LargeObjectMask",
            "OPR_RestrictedComposite",
            "OPR_RestoreOriginalPixels",
            "OPR_LoadImageICC",
            "OPR_SaveImageICC",
            "OPR_SaveImagePrecision",
            "OPR_DiagnosticSteps",
        }
        registered = set(package.NODE_CLASS_MAPPINGS)
        if not expected <= registered:
            raise RuntimeError(
                f"Missing nodes: {sorted(expected - registered)}; "
                f"import errors: {package._IMPORT_ERRORS}"
            )

        import folder_paths
        import torch

        input_directory = work / "input"
        output_directory = work / "output"
        input_directory.mkdir()
        output_directory.mkdir()
        folder_paths.set_input_directory(str(input_directory))
        folder_paths.set_output_directory(str(output_directory))

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 0] = np.arange(64, dtype=np.uint8)[None, :] + 55
        image[..., 1] = np.arange(64, dtype=np.uint8)[:, None] + 60
        image[..., 2] = 90
        Image.fromarray(image).save(input_directory / "source.png")

        loaded, profile = package.NODE_CLASS_MAPPINGS["OPR_LoadImageICC"]().load(
            "source.png"
        )
        restored = package.NODE_CLASS_MAPPINGS[
            "OPR_RestoreOriginalPixels"
        ]().restore(loaded, loaded, capture_steps=False)
        support = torch.zeros((1, 64, 64), dtype=torch.float32)
        saved = package.NODE_CLASS_MAPPINGS["OPR_SaveImagePrecision"]().save(
            restored[0], support, "e2e/result", "16bit", icc_profile=profile
        )
        entry = saved["ui"]["images"][0]
        output = output_directory / entry["subfolder"] / entry["filename"]
        if not output.is_file():
            raise RuntimeError(f"Output was not written: {output}")
        raw = output.read_bytes()
        if raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[24] != 16:
            raise RuntimeError("Save Precision did not write a 16-bit PNG")
        with Image.open(output) as saved_image:
            if saved_image.size != (64, 64):
                raise RuntimeError(f"Unexpected saved size: {saved_image.size}")
        print(
            json.dumps(
                {"registered_nodes": len(registered), "output_written": True}
            )
        )


if __name__ == "__main__":
    main()
