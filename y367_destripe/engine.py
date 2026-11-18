"""Shared, scoped CPU budget for CE4 / CF1 and their protection variants."""
from contextlib import contextmanager
import threading
import cv2
from threadpoolctl import threadpool_limits
MODEL_SHA256 = "32c4375a718762bec6352c12f320d3f760a04cc03a8b39ba46f2d09967870268"
FIXED_CYCLES = 48.0
_LOCK = threading.RLock()

@contextmanager
def inference_threads():
    # Source CPU budget, retaining both original axis workers; restore on error too.
    with _LOCK:
        previous = cv2.getNumThreads()
        try:
            cv2.setNumThreads(1)
            with threadpool_limits(limits=1, user_api="blas"):
                yield
        finally:
            cv2.setNumThreads(previous)
