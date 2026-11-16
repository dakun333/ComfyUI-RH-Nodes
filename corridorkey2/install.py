"""Install/check using ComfyUI's Python; never change host torch or numpy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_environment():
    print(f"Python: {sys.executable}\nVersion: {sys.version.split()[0]}", flush=True)
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 12):
        raise RuntimeError("This dependency set is verified for Windows x64 CPython 3.12. "
                           "Run with the SAME Python as ComfyUI; Python 3.13+, CPU/Mac are not supported here.")
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError("ComfyUI's torch/torchvision could not import. Repair ComfyUI itself first; "
                           "this installer intentionally does not install/upgrade torch.") from exc
    print(f"Host torch={torch.__version__}, torchvision={torchvision.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is unavailable in this Python. Use ComfyUI's CUDA environment.")
    # Test a real CUDA operation, not just driver detection.
    value = torch.ones(1, device="cuda").sum().item()
    torch.cuda.synchronize()
    assert value == 1
    print(f"CUDA operation OK; GPU: {torch.cuda.get_device_name(0)}", flush=True)


def run(command):
    print("\n> " + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def check_private_deps():
    for relative in ("numpy/__init__.py", "transformers/__init__.py"):
        if not (ROOT / "python_deps" / relative).is_file():
            raise RuntimeError("Private dependencies are missing. Execute the node once for automatic setup, or run install.py without --check. "
                               "Installing requirements.txt alone does NOT complete installation.")
    run([sys.executable, "-I", "-u", ROOT / "worker.py", "--check-deps"])


def asset_status():
    manifest = json.loads((ROOT / "assets_manifest.json").read_text(encoding="utf-8"))
    total = sum(asset["size"] for asset in manifest["files"])
    pending = []
    runtime = (ROOT / "runtime").resolve()
    for asset in manifest["files"]:
        path = (runtime / asset["path"]).resolve()
        if not path.is_relative_to(runtime):
            raise RuntimeError("Unsafe asset path in manifest")
        if not path.is_file() or path.stat().st_size != asset["size"]:
            pending.append(asset)
    print(f"\nModel resources: {total / 10**9:.2f} GB ({total / 2**30:.2f} GiB), "
          f"{len(manifest['files'])} pinned files", flush=True)
    print(f"Node disk free: {shutil.disk_usage(ROOT).free / 2**30:.2f} GiB", flush=True)
    if pending:
        print(f"Missing/wrong-size files: {len(pending)}. These download automatically on FIRST inference.", flush=True)
    else:
        print("All files have expected sizes. This is NOT a full SHA256 check.", flush=True)
    required = sum(asset["size"] for asset in pending)
    if required and shutil.disk_usage(ROOT).free < required + 512 * 1024**2:
        print("WARNING: Insufficient node disk space for model download. Free space before inference.", flush=True)
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check environment/deps/file sizes only; no installs/downloads")
    parser.add_argument("--index-url", help="Optional pip index (model downloads still use official URLs)")
    parser.add_argument("--download-models", action="store_true", help="Also download models now; default downloads on first inference")
    parser.add_argument("--verify", action="store_true", help="With --download-models: force full SHA256 verification/repair")
    args = parser.parse_args()
    if args.check and (args.index_url or args.download_models or args.verify):
        parser.error("--check is read-only; do not combine it with installation/download options")
    if args.verify and not args.download_models:
        parser.error("--verify requires --download-models")
    check_environment()
    if not args.check:
        if not shutil.which("git"):
            raise RuntimeError("Git must be installed and available on PATH for the pinned utils3d dependency.")
        run([sys.executable, "-m", "pip", "--version"])
        if shutil.disk_usage(ROOT).free < 600 * 1024**2:
            raise RuntimeError("Allow at least 600 MiB free for private dependency installation, plus 5 GB for models.")
        command = [sys.executable, ROOT / "install_worker_deps.py"]
        if args.index_url:
            command += ["--index-url", args.index_url]
        run(command)
    check_private_deps()
    required = asset_status()
    if args.download_models:
        if shutil.disk_usage(ROOT).free < required + 512 * 1024**2:
            raise RuntimeError("Insufficient disk space. Free space before downloading model resources.")
        command = [sys.executable, "-I", "-u", ROOT / "worker.py", "--ensure-assets"]
        if args.verify:
            command.append("--verify")
        run(command)
    print("\nEnvironment/dependency check PASS. Restart ComfyUI and search for 纯色底去背.", flush=True)
    if not args.check:
        print("No host torch/numpy/transformers packages were installed or upgraded.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nINSTALL/CHECK FAILED: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
