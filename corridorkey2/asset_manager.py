"""Pinned, verified node-local assets; no Hugging Face cache or host dependencies."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def asset_lock(root):
    # OS releases this lock if the worker is interrupted or killed.
    path = Path(root) / ".assets.lock"
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.3)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def download_asset(asset, destination, progress=None, attempts=4):
    """Resume incomplete transfers; verify before atomically replacing destination.

    Never write an existing destination in-place: installed assets may be hardlinks.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".partial")
    expected_size = int(asset["size"])
    if not asset["url"].startswith("https://"):
        raise ValueError("Only pinned HTTPS weight URLs are permitted")
    if part.exists() and part.stat().st_size > expected_size:
        part.unlink()
    last_error = None
    for attempt in range(attempts):
        try:
            offset = part.stat().st_size if part.exists() else 0
            remaining = expected_size - offset
            if remaining > 0:
                if shutil.disk_usage(destination.parent).free < remaining + 128 * 1024**2:
                    raise OSError(f"Insufficient free disk space for {destination.name}: "
                                  f"need {remaining / 1024**3:.2f} GiB plus a 128 MiB margin")
                headers = {"User-Agent": "ComfyUI-CorridorKey2/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(asset["url"], headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    status = response.status
                    if status == 206:
                        prefix = f"bytes {offset}-"
                        if not response.headers.get("Content-Range", "").startswith(prefix):
                            raise IOError("Server returned an incorrect HTTP range")
                        mode = "ab" if offset else "wb"
                    elif status == 200:
                        # Server ignored Range: restart the temporary file, not the weight.
                        offset, mode = 0, "wb"
                        if shutil.disk_usage(destination.parent).free < expected_size + 128 * 1024**2:
                            raise OSError("Insufficient disk space to restart download")
                    else:
                        raise IOError(f"Unexpected HTTP status {status}")
                    with part.open(mode) as f:
                        while True:
                            block = response.read(2 * 1024 * 1024)
                            if not block:
                                break
                            f.write(block)
                            offset += len(block)
                            if offset > expected_size:
                                raise IOError("Download exceeds the pinned asset size")
                            if progress:
                                progress(f"Downloading {destination.name}: "
                                         f"{offset / expected_size:.0%}")
            if not part.exists() or part.stat().st_size != expected_size:
                raise IOError(f"Incomplete transfer for {destination.name}")
            if sha256(part) != asset["sha256"]:
                part.unlink()
                raise IOError(f"SHA256 mismatch for {destination.name}")
            os.replace(part, destination)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                print(f"[CorridorKey2] Download retry {attempt+1}: {exc}", flush=True)
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Cannot download {destination.name}: {last_error}\n"
                       f"Pinned source: {asset['url']}") from last_error


def ensure_assets(root=ROOT, *, verify=False, progress=None):
    root = Path(root).resolve()
    runtime = (root / "runtime").resolve()
    manifest = json.loads((root / "assets_manifest.json").read_text(encoding="utf-8"))
    cache_path = root / ".verified_assets.json"
    downloaded = []
    with asset_lock(root):
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        for asset in manifest["files"]:
            relative = asset["path"]
            path = (runtime / relative).resolve()
            if not path.is_relative_to(runtime):
                raise ValueError(f"Asset path escapes node runtime: {relative}")
            valid = False
            if path.is_file() and path.stat().st_size == asset["size"]:
                stat = path.stat()
                signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, asset["sha256"]]
                if not verify and cache.get(relative) == signature:
                    valid = True
                else:
                    if progress:
                        progress(f"Verifying {relative}")
                    valid = sha256(path) == asset["sha256"]
            if not valid:
                print(f"[CorridorKey2] Missing or invalid asset: {relative}", flush=True)
                download_asset(asset, path, progress=progress)
                downloaded.append(relative)
            stat = path.stat()
            cache[relative] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, asset["sha256"]]
        atomic_json(cache_path, cache)
    return downloaded
