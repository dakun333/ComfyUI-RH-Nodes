"""Full CorridorKey state checkpoints; configs are data and model code stays in the node."""
from __future__ import annotations
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
if __package__:
    from .asset_manager import asset_lock, ensure_assets
else:
    from asset_manager import asset_lock, ensure_assets

ROOT = Path(__file__).resolve().parent
BUNDLE_FILENAME = "corridorkey2_full.safetensors"
BUNDLE_FORMAT = "corridorkey_v2_full_checkpoint"


def default_checkpoint_path():
    """Resolve the normal ComfyUI layout for optional standalone CLI commands."""
    for parent in ROOT.parents:
        if (parent / "folder_paths.py").is_file():
            return parent / "models" / "checkpoints" / BUNDLE_FILENAME
    raise ValueError("ComfyUI root not found; pass an explicit --checkpoint or --output path")


def prepare_runtime(runtime_root):
    runtime_root = Path(runtime_root)
    for path in (runtime_root, runtime_root / "third_party/MoGe"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def read_checkpoint_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if metadata.get("format") != BUNDLE_FORMAT or metadata.get("format_version") != "1":
            raise ValueError(f"{path} is not a CorridorKey v2 full checkpoint. Do not select an SD/Flux checkpoint.")
        for key in ("model_config", "foundation_config"):
            try:
                value = json.loads(metadata[key])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid checkpoint {key}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Invalid checkpoint {key}")
            if key == "foundation_config":
                for name in ("moge", "cradio", "rvm"):
                    if not isinstance(value.get(name), dict):
                        raise ValueError(f"Invalid checkpoint {name} config")
        keys = set(reader.keys())
        for prefix in ("stage1.", "stage2.", "foundations.moge.", "foundations.cradio.", "foundations.rvm."):
            if not any(key.startswith(prefix) for key in keys):
                raise ValueError(f"Checkpoint is missing {prefix} weights")
        for name in ("cradio", "moge", "rvm"):
            for stat in ("mean", "std"):
                if f"foundations.{name}_{stat}" not in keys:
                    raise ValueError(f"Checkpoint is missing {name} {stat}")
        if len(keys) != int(metadata["tensor_count"]):
            raise ValueError("Checkpoint tensor count does not match its metadata")
    return metadata


def convert_checkpoint(runtime_root, destination, progress=print, overwrite=False):
    runtime_root, destination = Path(runtime_root).resolve(), Path(destination).resolve()
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Checkpoint output must end in .safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with asset_lock(destination.parent):
        if destination.exists() and not overwrite:
            read_checkpoint_metadata(destination)
            return {"checkpoint": str(destination), "reused": True, "bytes": destination.stat().st_size}
        started = time.monotonic()
        prepare_runtime(runtime_root)
        from corridorkey import runtime as ck
        paths = ck.RuntimePaths.from_package(runtime_root)
        paths.validate()
        progress("Loading the original complete models on CPU")
        model = ck.load_model(paths, device="cpu")
        with safe_open(str(paths.weights), framework="pt", device="cpu") as reader:
            main_header = reader.metadata()
        moge = torch.load(str(paths.moge_checkpoint), map_location="cpu", weights_only=True, mmap=True)
        foundation_config = {
            "moge": moge["model_config"],
            "cradio": json.loads((paths.cradio_dir / "config.json").read_text(encoding="utf-8")),
            "rvm": {"variant": model.foundations.rvm.variant},
        }
        del moge
        state, storage_seen = {}, set()
        groups, dtypes = defaultdict(int), defaultdict(int)
        for key, tensor in model.state_dict().items():
            tensor = tensor.detach().cpu().contiguous()
            storage = tensor.untyped_storage().data_ptr()
            if tensor.numel() and storage in storage_seen:
                tensor = tensor.clone()
            if tensor.numel():
                storage_seen.add(tensor.untyped_storage().data_ptr())
            state[key] = tensor
            if key.startswith("foundations."):
                group = key.split(".")[1]
                if group not in {"moge", "cradio", "rvm"}:
                    group = "stats"
            else:
                group = key.split(".")[0]
            size = tensor.numel() * tensor.element_size()
            groups[group] += size
            dtypes[str(tensor.dtype)] += size
        metadata = {
            "format": BUNDLE_FORMAT, "format_version": "1",
            "model_config": main_header["model_config"],
            "foundation_config": json.dumps(foundation_config),
            "metadata": main_header.get("metadata", "{}"),
            "tensor_count": str(len(state)), "torch_version": str(torch.__version__),
            "precision_policy": "preserve original runtime state dtypes; no quantization",
        }
        temporary = destination.with_name(destination.name + f".{os.getpid()}.partial")
        progress(f"Writing {len(state)} tensors to {destination}")
        save_file(state, str(temporary), metadata=metadata)
        progress("Comparing every saved tensor against the original runtime state")
        read_checkpoint_metadata(temporary)
        with safe_open(str(temporary), framework="pt", device="cpu") as reader:
            for key, expected in state.items():
                if not torch.equal(reader.get_tensor(key), expected):
                    raise RuntimeError(f"Checkpoint tensor verification failed: {key}")
        os.replace(temporary, destination)
        report = {"checkpoint": str(destination), "bytes": destination.stat().st_size,
                  "tensor_count": len(state), "groups_bytes": dict(groups), "dtypes_bytes": dict(dtypes),
                  "all_tensors_verified": True, "elapsed_seconds": round(time.monotonic() - started, 3)}
        progress(f"Checkpoint ready: {destination} ({destination.stat().st_size / 1e9:.3f} GB)")
        return report


def ensure_checkpoint(path, progress=print):
    path = Path(path).resolve()
    if path.is_file():
        read_checkpoint_metadata(path)
        return []
    downloaded = ensure_assets(ROOT, progress=progress)
    convert_checkpoint(ROOT / "runtime", path, progress=progress)
    return downloaded


def load_checkpoint(path, runtime_root=None, device="cuda"):
    path = Path(path).resolve()
    metadata = read_checkpoint_metadata(path)
    runtime_root = Path(runtime_root or ROOT / "runtime")
    prepare_runtime(runtime_root)
    from corridorkey.model import CorridorKeyV3
    from corridorkey.foundations import build_cradio_from_config
    from corridorkey.rvm import RVMVideoSiamMAE
    from moge.model.v2 import MoGeModel

    foundation_config = json.loads(metadata["foundation_config"])
    cradio_config = foundation_config["cradio"]
    # Never interpret pretrained/initial-checkpoint paths embedded in a model config as downloads.
    cradio_config["args"]["pretrained"] = False
    cradio_config["args"]["initial_checkpoint"] = None
    moge = MoGeModel(**foundation_config["moge"])
    moge.enable_pytorch_native_sdpa()
    models = {
        "moge": moge,
        "cradio": build_cradio_from_config(cradio_config, runtime_root / "third_party/C-RADIO"),
        "rvm": RVMVideoSiamMAE(variant=foundation_config["rvm"]["variant"]),
    }
    config = json.loads(metadata["model_config"])
    config.update(device="cpu", gradient_checkpoint=False, stage1_recurrent_64_sites=(),
                  cradio_repo=str(runtime_root / "third_party/C-RADIO"), moge_checkpoint=None,
                  rvm_ckpt=None, foundation_stats_path=None, foundation_models=models)
    model = CorridorKeyV3(**config)
    state = load_file(str(path), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    # Match the reference loader: foundations on the compute device; stages moved by run_inference.
    model.foundations.to(device)
    return model
