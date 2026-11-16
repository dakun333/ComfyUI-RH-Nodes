"""Stdlib-only first-execution bootstrap. Never imported by the ComfyUI host."""
from __future__ import annotations
import contextlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


@contextlib.contextmanager
def dependency_lock(root, progress=None, timeout=1200):
    """A process lock shared by manual/automatic installation; OS releases on exit."""
    path = Path(root) / ".deps.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0"); handle.flush()
        if os.name == "nt":
            import msvcrt
            def acquire():
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire(): fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release(): fcntl.flock(handle, fcntl.LOCK_UN)
        start = time.monotonic()
        last_notice = float('-inf')
        while True:
            try:
                acquire(); break
            except OSError:
                now = time.monotonic()
                if now - start >= timeout:
                    raise TimeoutError("Timed out waiting for another private dependency installation")
                if progress and now - last_notice >= 5:
                    progress("Waiting for another dependency installer; no duplicate pip installation")
                    last_notice = now
                time.sleep(0.25)
        try:
            yield
        finally:
            release()


def probe_private_dependencies(root=ROOT, target=None):
    """Check all pinned metadata and real imports in a fresh isolated subprocess.

    The caller never imports a possibly wrong host NumPy/Transformers before installation.
    """
    root = Path(root).resolve()
    target = (Path(target) if target is not None else root / "python_deps").resolve()
    for relative in ("numpy/__init__.py", "transformers/__init__.py"):
        if not (target / relative).is_file():
            return False, f"Missing private {relative}"
    expected = json.loads((root / "dependency_versions.json").read_text(encoding="utf-8"))["versions"]
    code = f'''
import sys, json, re, importlib, importlib.metadata as md
from pathlib import Path
private = Path({str(target)!r}).resolve()
sys.path.insert(0, str(private))
normal = lambda name: re.sub(r"[-_.]+", "-", name).lower()
installed = {{normal(d.metadata["Name"]): d.version for d in md.distributions(path=[str(private)])}}
expected = {expected!r}
for name, version in expected.items():
    assert installed.get(normal(name)) == version, f"Private {{name}} expected {{version}}, found {{installed.get(normal(name))}}"
for name in ("numpy", "cv2", "transformers", "scipy", "timm", "utils3d", "safetensors", "huggingface_hub"):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(private), f"{{name}} fell back to host packages"
import torch, torchvision
print("CK2_PRIVATE_DEPS_OK")
'''
    try:
        result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode == 0 and "CK2_PRIVATE_DEPS_OK" in result.stdout:
        return True, "All pinned private dependencies/imports OK"
    return False, (result.stderr or result.stdout or "Private import check failed")[-4000:]


def quick_private_version_check(root=ROOT):
    """No heavy imports on subsequent jobs; the worker performs real imports once.

    Missing payloads/metadata or changed pins fall back to locked setup. This avoids
    importing torch/transformers again in a separate probe process on every inference.
    """
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 12):
        return False
    root = Path(root).resolve()
    target = root / "python_deps"
    for module in ("numpy", "cv2", "transformers", "scipy", "timm", "utils3d", "safetensors", "huggingface_hub"):
        if not (target / module / "__init__.py").is_file():
            return False
    try:
        import importlib.metadata as md
        import re
        expected = json.loads((root / "dependency_versions.json").read_text(encoding="utf-8"))["versions"]
        normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
        actual = {normalize(d.metadata["Name"]): d.version for d in md.distributions(path=[str(target)])}
        return all(actual.get(normalize(name)) == version for name, version in expected.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def check_install_environment():
    """Reject unsupported Python/missing host CUDA before starting downloads."""
    if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Automatic dependency installation requires Windows x64 CPython 3.12. "
                           "Use ComfyUI's Python 3.12; no automatic Python/torch upgrade is performed.")
    import shutil
    if not shutil.which("git"):
        raise RuntimeError("Git is missing from PATH; the pinned utils3d dependency needs Git. "
                           "Install Git and restart ComfyUI, or install dependencies manually.")
    code = "import torch,torchvision; assert torch.cuda.is_available(), 'NVIDIA CUDA unavailable'; torch.ones(1,device='cuda').sum().item(); torch.cuda.synchronize(); print('CUDA preflight OK')"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError("ComfyUI's torch/torchvision/CUDA environment is not usable. "
                           "This installer will not replace it.\n" + (result.stderr or result.stdout)[-4000:])


def ensure_worker_dependencies(root=ROOT, progress=None, status_file=None):
    """Called ONLY by an actual worker job/explicit bootstrap command, not node import."""
    root = Path(root).resolve()
    if os.environ.get("CORRIDORKEY2_AUTO_INSTALL_DEPS", "1").strip().lower() in ("0", "false", "off"):
        valid, reason = probe_private_dependencies(root)
        if not valid:
            raise RuntimeError("Automatic dependency installation is disabled. "
                               f"Run with ComfyUI Python: {root / 'install.py'}\n{reason}")
        if progress: progress("Private dependencies ready; automatic installation disabled")
        return
    if quick_private_version_check(root):
        if progress: progress("Fast private dependency check OK; skipping pip")
        importlib.invalidate_caches()
        return
    if progress: progress("Checking private dependencies; missing packages will install automatically")
    command = [sys.executable, "-I", "-u", str(root / "install_worker_deps.py"), "--if-needed"]
    if status_file is not None:
        command += ["--status-file", str(status_file)]
    index = os.environ.get("CORRIDORKEY2_PIP_INDEX_URL", "").strip()
    if index:
        command += ["--index-url", index]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    try:
        # Inherit worker stdout/stderr: complete pip output goes to the node's log.
        subprocess.run(command, check=True, cwd=root, env=env)
        # python_deps may not have existed when this worker imported stdlib modules.
        # Clear PathFinder negative caches before the first NumPy/Transformers import.
        importlib.invalidate_caches()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Automatic private dependency setup failed. See this job's worker log for "
                           "pip/network/Git/CUDA details; retry after fixing them, or run install.py manually. "
                           "No host packages were installed/upgraded.") from exc
