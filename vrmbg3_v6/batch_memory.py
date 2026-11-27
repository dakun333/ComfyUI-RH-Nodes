"""File-backed CPU tensors. Keep FP32 and native resolution, not an inference downgrade."""
import logging, math, mmap, os, shutil, uuid
from pathlib import Path
import torch
import folder_paths

def _resize_empty_file(file, size):
    if os.name == "nt":
        # CPython/CRT truncate on Windows writes zeros through the entire file.
        # Sparse holes read as zero, but only touched pages need physical writes.
        # Never use SetFileValidData (would expose uninitialized disk contents).
        import ctypes, msvcrt
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        ioctl = kernel.DeviceIoControl
        ioctl.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                          ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        ioctl.restype = ctypes.c_int
        returned = ctypes.c_ulong()
        sparse = ioctl(msvcrt.get_osfhandle(file.fileno()), 0x900C4, None, 0, None, 0,
                       ctypes.byref(returned), None)  # FSCTL_SET_SPARSE
        if sparse:
            file.seek(size - 1)
            file.write(b"\0")
            file.flush()
            return
        logging.warning("VRMBG3 temp volume does not support sparse files; safe slow allocation fallback (Windows error %s)", ctypes.get_last_error())
    file.truncate(size)


class BatchBuffer(mmap.mmap):
    def __new__(cls, path, size):
        try:
            # Windows closes these handles even on abrupt process exit. The
            # duplicated handle retained by mmap keeps the file alive for views.
            if os.name == "nt":
                fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_TEMPORARY, 0o600)
                file = os.fdopen(fd, "w+b")
            else:
                file = open(path, "w+b")
            with file as f:
                _resize_empty_file(f, size)
                obj = super().__new__(cls, f.fileno(), size, access=mmap.ACCESS_WRITE)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        obj.cache_path = Path(path)
        return obj
    def __del__(self):
        # torch.frombuffer retains the actual buffer owner through tensor storage,
        # including views: don't unlink just because the original tensor was dropped.
        try:
            self.close()
            self.cache_path.unlink(missing_ok=True)
        except (BufferError, OSError, AttributeError):
            logging.warning("VRMBG3 temporary mapped batch remains until restart: %s", getattr(self, "cache_path", ""))

def mapped_tensor(shape):
    root = Path(folder_paths.get_temp_directory()) / "vrmbg3_v6_batches"
    root.mkdir(parents=True, exist_ok=True)
    size = math.prod(shape) * 4
    if shutil.disk_usage(root).free < size + (1 << 30):
        raise ValueError(f"VRMBG3 mapped batch needs {size/(1<<30):.2f}GiB temporary disk space plus 1GiB reserve; free disk space or use Video Slice")
    path = root / f"batch_{os.getpid()}_{uuid.uuid4().hex}.f32"
    buffer = BatchBuffer(path, size)
    return torch.frombuffer(buffer, dtype=torch.float32).reshape(shape)
