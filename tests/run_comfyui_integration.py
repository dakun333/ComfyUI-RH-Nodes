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
            "MekajikiUnMult",
            "MekajikiMaskedUnMult",
            "SeedreamLayerColorReuseV64",
            "SeedreamLayerColorReuseV64Fast",
            "SeedreamV64LoadAlignedLayers",
            "SharedColorBroadbandDestripe",
            "CorridorKey2Matting",
            "PureLightMatting",
            "Y367CF1Destripe",
            "Y367CE4Destripe",
            "Y367CF1ProtectedDestripe",
            "Y367CE4ProtectedDestripe",
        }
        registered = set(package.NODE_CLASS_MAPPINGS)
        if not expected <= registered:
            raise RuntimeError(
                f"Missing nodes: {sorted(expected - registered)}; "
                f"import errors: {package._IMPORT_ERRORS}"
            )

        import folder_paths
        import torch

        yy, xx = np.mgrid[:160, :192]
        direction = np.array([.3, -.6, .8]); direction /= np.linalg.norm(direction)
        waves = np.cos(2*np.pi*48*xx/192) + .7*np.cos(2*np.pi*48*yy/160)
        stripes = torch.from_numpy(np.rint(150 + 3.5*waves[..., None]*direction).astype(np.float32)/255)[None]
        for node_id in ("Y367CF1Destripe", "Y367CE4Destripe", "Y367CF1ProtectedDestripe", "Y367CE4ProtectedDestripe"):
            cls = package.NODE_CLASS_MAPPINGS[node_id]
            if "Protected" in node_id:
                schema = cls.define_schema()
                if schema.category != "RH-Nodes/restore" or len(schema.outputs) != 3:
                    raise RuntimeError("Y367 V3 schema changed")
                y367_result = cls.execute(stripes, orientation="both").result
            else:
                y367_result = cls().process(stripes, orientation="both")
            if y367_result[0].shape != stripes.shape or not torch.isfinite(y367_result[0]).all():
                raise RuntimeError("Y367 returned invalid output")

        light = package.NODE_CLASS_MAPPINGS["PureLightMatting"]()
        glow = torch.zeros(2, 16, 24, 3)
        glow[:, 4:12, 6:18] = torch.tensor([.04, .2, .08])
        result = light.matte(glow, auto_background=False, processing_device="cpu")
        torch.testing.assert_close(result[0] * result[1][..., None], glow)
        from comfy_api.latest import Types, InputImpl
        from fractions import Fraction
        source_video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=glow, frame_rate=Fraction(30000, 1001), metadata={"test": "matting"}))
        video_result = light.matte(video=source_video, output_mode="video", auto_background=False, processing_device="cpu")
        video_components = video_result[5].get_components()
        if video_components.frame_rate != Fraction(30000, 1001) or video_components.images.shape != (2, 16, 24, 4):
            raise RuntimeError("Pure-light VIDEO lost frame rate or RGBA")
        if not torch.equal(video_components.alpha, video_components.images[..., 3]):
            raise RuntimeError("Pure-light VIDEO alpha differs from RGBA")

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
        cleaned, report = package.NODE_CLASS_MAPPINGS[
            "SharedColorBroadbandDestripe"
        ]().process(loaded.repeat(2, 1, 1, 1), auto_scale=True)
        if cleaned.shape != (2, 64, 64, 3) or len(json.loads(report)["frames"]) != 2:
            raise RuntimeError("Destripe returned unexpected batch outputs")
        # Exercise the real standard LoadImage -> destripe -> SaveImage alpha path.
        import nodes as comfy_nodes
        y, x = np.mgrid[:192, :320]
        color = np.array([.35, -.5, .8]); color /= np.linalg.norm(color)
        transparent = np.empty((192, 320, 4), dtype=np.uint8)
        transparent[..., :3] = np.rint(150 + 3.5*np.cos(2*np.pi*x/12 + .2*np.sin(y/25))[:, :, None]*color)
        transparent[..., 3] = 0
        transparent[24:168, 24:296, 3] = 255
        transparent[20:24, 24:296, 3] = 77
        transparent[168:172, 24:296, 3] = 77
        transparent[:4, :256, 3] = np.arange(256, dtype=np.uint8)
        Image.fromarray(transparent).save(input_directory / "transparent.png")
        rgb, mask = comfy_nodes.LoadImage().load_image("transparent.png")
        cleaned_rgba, report = package.NODE_CLASS_MAPPINGS[
            "SharedColorBroadbandDestripe"
        ]().process(rgb, mask=mask, auto_scale=True)
        if json.loads(report)["frames"][0]["status"] != "processed_not_quality_approved":
            raise RuntimeError("Transparent destripe did not process the typical input")
        saved_rgba = comfy_nodes.SaveImage().save_images(cleaned_rgba, filename_prefix="transparent/result")
        entry = saved_rgba["ui"]["images"][0]
        with Image.open(output_directory / entry["subfolder"] / entry["filename"]) as output_rgba:
            if output_rgba.mode != "RGBA":
                raise RuntimeError("Standard SaveImage did not save RGBA")
            np.testing.assert_array_equal(np.asarray(output_rgba)[..., 3], transparent[..., 3])
        layer_folder = input_directory / "layers"
        layer_folder.mkdir()
        Image.fromarray(image).convert("RGBA").save(layer_folder / "layer_1.png")
        layers, masks, _ = package.NODE_CLASS_MAPPINGS[
            "SeedreamV64LoadAlignedLayers"
        ]().load("layers")
        for node_id in ("SeedreamLayerColorReuseV64", "SeedreamLayerColorReuseV64Fast"):
            result = package.NODE_CLASS_MAPPINGS[node_id]().correct_and_reuse(
                [loaded], [layers], normalize_alpha_levels=[False], image_masks=[masks]
            )
            if len(result) != 8 or result[7].shape != (1, 64, 64, 4):
                raise RuntimeError("Seedream layer reuse returned unexpected outputs")
        restored = package.NODE_CLASS_MAPPINGS[
            "OPR_RestoreOriginalPixels"
        ]().restore(loaded, loaded, capture_steps=False)
        unmult_rgba, unmult_alpha = package.NODE_CLASS_MAPPINGS[
            "MekajikiUnMult"
        ]().apply(loaded, auto_background=False)
        if unmult_rgba.shape != (1, 64, 64, 4) or unmult_alpha.shape != (
            1,
            64,
            64,
        ):
            raise RuntimeError("UnMult returned unexpected output shapes")
        guided_rgba, guided_alpha = package.NODE_CLASS_MAPPINGS[
            "MekajikiMaskedUnMult"
        ]().apply(loaded, unmult_alpha)
        if guided_rgba.shape != (1, 64, 64, 4) or not torch.equal(
            guided_alpha, unmult_alpha
        ):
            raise RuntimeError("Mask-Guided UnMult returned unexpected outputs")
        overflow_image = torch.zeros((1, 64, 64, 3))
        overflow_image[:, 20:44, 20:44] = torch.tensor([0.8, 0.4, 0.1])
        overflow_mask = torch.zeros((1, 64, 64))
        overflow_mask[:, 20:44, 20:44] = 0.2
        adjusted_rgba, adjusted_alpha = package.NODE_CLASS_MAPPINGS[
            "MekajikiMaskedUnMult"
        ]().apply(overflow_image, overflow_mask, allow_alpha_adjustment=True)
        if not torch.equal(adjusted_rgba[..., 3], adjusted_alpha):
            raise RuntimeError("Adjusted RGBA and alpha outputs differ")
        if not 0 < float((adjusted_alpha - overflow_mask).max()) <= 43 / 255 + 1e-7:
            raise RuntimeError("Alpha adjustment did not respect its increase cap")
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
                {"registered_nodes": len(registered), "output_written": True,
                 "standard_transparent_png_roundtrip": True,
                 "y367_four_nodes_executed": True}
            )
        )


if __name__ == "__main__":
    main()
