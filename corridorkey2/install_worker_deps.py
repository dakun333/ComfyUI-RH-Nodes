"""Install private worker dependencies with ComfyUI Python; never install torch."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
# Required for python -I script execution; no private deps enter the parent process.
sys.path.insert(0, str(ROOT))
from dependency_bootstrap import dependency_lock, probe_private_dependencies, check_install_environment
from asset_manager import atomic_json


def safe_remove(path):
    path = Path(path).resolve()
    if path.parent != ROOT or not path.name.startswith("python_deps"):
        raise ValueError(f"Refusing to recursively remove unexpected path: {path}")
    if path.exists():
        shutil.rmtree(path)


def install(*, if_needed=False, index_url=None, progress=None):
    say = progress or (lambda message: print(f"[CorridorKey2 deps] {message}", flush=True))
    with dependency_lock(ROOT, progress=say):
        if if_needed:
            say("Checking pinned private versions and real imports")
            valid, reason = probe_private_dependencies(ROOT)
            if valid:
                say("Private dependencies ready; skipping pip")
                return False
            say("Private dependencies missing/incompatible; preparing first-run installation")
            print(f"[CorridorKey2 deps] Check detail: {reason}", flush=True)
        check_install_environment()
        if shutil.disk_usage(ROOT).free < 600 * 1024**2:
            raise RuntimeError("Allow at least 600 MiB free on the node disk for private dependencies, plus model space.")
        tag = uuid.uuid4().hex[:8]
        staging = ROOT / ("python_deps_new_" + tag)
        backup = ROOT / ("python_deps_backup_" + tag)
        target = ROOT / "python_deps"
        command = [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(staging),
                   "-r", str(ROOT / "isolated_requirements.txt")]
        if index_url:
            command += ["--index-url", index_url]
        try:
            say("Installing private dependencies (first run); see worker log for pip download progress")
            subprocess.run(command, check=True)
            say("Verifying new private dependencies before switching directories")
            valid, reason = probe_private_dependencies(ROOT, staging)
            if not valid:
                raise RuntimeError("New private dependency verification failed; existing installation preserved.\n" + reason)
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                if backup.exists():
                    os.replace(backup, target)
                raise
            safe_remove(backup)
        finally:
            safe_remove(staging)
        say("Private dependency installation complete; continuing matting without restarting ComfyUI")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", help="Optional pip index")
    parser.add_argument("--if-needed", action="store_true", help="Under the shared install lock, skip a verified existing installation")
    parser.add_argument("--status-file", type=Path, help="Internal worker job status JSON (optional)")
    args = parser.parse_args()
    def progress(message):
        print(f"[CorridorKey2 deps] {message}", flush=True)
        if args.status_file is not None:
            atomic_json(args.status_file, {"progress": 1, "message": message})
    install(if_needed=args.if_needed, index_url=args.index_url, progress=progress)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
