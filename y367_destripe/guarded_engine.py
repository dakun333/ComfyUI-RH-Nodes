"""CE4 / CF1 in the same Python environment and scoped CPU budget as Y367-FAST."""
from pathlib import Path
from contextlib import contextmanager
import hashlib, sys, threading, time
import cv2, numba, PIL, scipy
import numpy as np
from .engine import MODEL_SHA256, FIXED_CYCLES, inference_threads
from .guarded_runtime import envelope, protected_outlier
from .guarded_runtime.chroma_guard import admission_mask
MODEL = Path(__file__).parent / "guarded_runtime" / "model.npz"


def runtime_versions():
    """Loaded numerical modules, for reports and cache invalidation; no version pin."""
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "PIL": PIL.__version__, "cv2": cv2.__version__, "numba": numba.__version__}


def verify_model():
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("CE4/CF1 frozen model hash mismatch")
verify_model()


def _interrupt():
    management = sys.modules.get("comfy.model_management")
    if management is not None:
        management.throw_exception_if_processing_interrupted()


@contextmanager
def _cpu4_threads(orientation):
    """CPU4 budget inside the original shared engine lock, never at import time.

    Worker-local masks are set by frozen CPU4 at both axis entry points.
    Restore the caller mask and the module's worker budget even on cancellation.
    """
    hot = envelope._hot
    if hot is None:
        raise RuntimeError("CPU4 requires working Numba hot kernels in the shared ComfyUI environment")
    previous_numba = numba.get_num_threads()
    previous_hot = hot._THREAD_COUNT
    previous_cv = cv2.getNumThreads()
    budget = min(4, numba.config.NUMBA_NUM_THREADS)
    try:
        if orientation == "both" and numba.threading_layer() == "workqueue":
            raise RuntimeError("CPU4 joint processing requires a concurrency-safe Numba backend, not workqueue")
        hot.configure_threads(budget)
        cv2.setNumThreads(budget)
        yield
    finally:
        hot.configure_threads(previous_hot)
        numba.set_num_threads(previous_numba)
        cv2.setNumThreads(previous_cv)


def _validated_inference_schedule(report, orientation, count):
    """Validate the five logical updates without pretending fixed-point reuse is a CNN call."""
    directions = ["vertical", "horizontal"] if orientation == "both" else [orientation]
    branches = report.get("branches")
    if not isinstance(branches, list) or [b.get("direction") for b in branches] != directions or report.get("steps") != 5:
        raise RuntimeError("CPU4 missing or invalid five-update branch schedule")
    expected_calls = 0
    total_reused = 0
    for branch in branches:
        executed = branch.get("executed_refreshes")
        reused = branch.get("exact_fixed_point_reused_refreshes")
        steps = branch.get("steps")
        if (type(executed) is not int or type(reused) is not int or
                not 1 <= executed <= 5 or not 0 <= reused <= 4 or executed + reused != 5 or
                branch.get("approximate_early_stop") is not False or not isinstance(steps, list) or
                len(steps) != 5 or [step.get("step") for step in steps] != [1, 2, 3, 4, 5]):
            raise RuntimeError("CPU4 incomplete or approximate update schedule")
        if reused:
            stable = {k: v for k, v in steps[executed - 1].items() if k != "step"}
            if any({k: v for k, v in step.items() if k != "step"} != stable for step in steps[executed:]):
                raise RuntimeError("CPU4 fixed-point trace does not preserve the stable update")
        expected_calls += 1 + executed  # One initial model call per axis, then physical refreshes.
        total_reused += reused
    if count != expected_calls:
        raise RuntimeError(f"CPU4 actual inference count mismatch: {count} != {expected_calls}")
    return dict(model_inferences_executed=count,
                full_logical_model_inferences=6 * len(branches),
                logical_refreshes_per_axis=5,
                exact_fixed_point_refreshes_reused=total_reused,
                approximate_early_stop=False)


def _in_process_frame(rgb, alpha, orientation, variant):
    counter_lock = threading.Lock()
    count = 0
    with inference_threads(), _cpu4_threads(orientation):
        _interrupt()
        original_infer = envelope.infer_numpy

        def counted_infer(*args, **kwargs):
            nonlocal count
            _interrupt()
            with counter_lock:
                count += 1
            result = original_infer(*args, **kwargs)
            _interrupt()
            return result

        # Existing shared lock serializes all four node variants and instrumentation.
        envelope.infer_numpy = counted_infer
        started = time.perf_counter()
        try:
            target, eight, report = protected_outlier.process(
                rgb, alpha, method="CF1", variant="PO1",
                orientation=orientation, keep_edge_protection=False)
            _interrupt()
        finally:
            envelope.infer_numpy = original_infer
    schedule = _validated_inference_schedule(report, orientation, count)
    report = dict(report)
    report.update(**schedule, in_process_compute_seconds=time.perf_counter() - started)
    return target, eight, report


def run_guarded_frame(rgb, alpha, orientation, *, variant):
    started = time.perf_counter()
    if variant != "CF1_PO1":
        raise ValueError("Unknown Y367-FAST guarded variant")
    if orientation not in ("vertical", "horizontal", "both"):
        raise ValueError("Unknown Y367-FAST orientation")
    verify_model()
    if envelope._native is None:
        raise RuntimeError("CE4/CF1 requires working Numba CPU kernels in the ComfyUI environment")
    rgb = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if not np.array_equal(rgb, np.uint8(rgb).astype(np.float32)):
        raise ValueError(f"{variant} requires native integer-valued 8-bit RGB; no implicit input quantization")
    # Keep the operator's already-guarded actual8 bytes, never re-encode
    # after the whole-pixel color / edge / contour veto.
    target, eight, report = _in_process_frame(rgb, alpha, orientation, variant)
    original8 = np.uint8(rgb)
    protected = alpha < 0.95
    if not np.array_equal(target[protected], rgb[protected]):
        raise RuntimeError(f"{variant} FLOAT alpha protection failed")
    if not np.array_equal(eight[protected], original8[protected]):
        raise RuntimeError(f"{variant} actual8 alpha protection failed")
    removal = rgb - target
    actual = rgb - eight.astype(np.float32)
    active = np.any(actual != 0, axis=2)
    display_vector = np.clip(128.0 + 16.0 * actual, 0, 255) - 128.0
    if not all(np.all(admission_mask(v)[active]) for v in (removal, actual, display_vector)):
        raise RuntimeError(f"{variant} frozen FLOAT/actual8/x16 color contract failed")
    if np.max(np.abs(actual)) > 8:
        raise RuntimeError(f"{variant} original actual8 change cap exceeded")
    if np.max(np.abs(eight.astype(np.float32) - target)) >= 1.00001:
        raise RuntimeError(f"{variant} actual8 differs by >=1 code value from FLOAT")
    report = dict(report)
    report.update(
        method_name=variant, ComfyUI_deployed=True,
        wrapper_core_and_encode_seconds=time.perf_counter() - started,
        native_size=[int(rgb.shape[1]), int(rgb.shape[0])],
        weights_retrained=False, source_algorithm_modified=False,
        reference_images_read=False, RE2_called=False, SR3_called=False,
        geometric_edge_halo_enabled=False, fixed_cycles=FIXED_CYCLES,
        protected_pixels=int(protected.sum()),
        total_FLOAT_RMS=float(np.sqrt(np.mean(removal.astype(np.float64)**2))),
        total_actual8_RMS=float(np.sqrt(np.mean(actual.astype(np.float64)**2))),
        max_FLOAT_change=float(np.max(np.abs(removal))),
        max_actual8_change=float(np.max(np.abs(actual))),
        modified_FLOAT_pixels=int(np.any(removal != 0, axis=2).sum()),
        modified_actual8_pixels=int(active.sum()),
        FLOAT_to_actual8_max=float(np.max(np.abs(eight.astype(np.float32) - target))),
        protected_FLOAT_exact=True, protected_actual8_exact=True,
        color_contract_FLOAT_actual8_x16=True,
        final_actual8_returned_directly_from_frozen_operator=True,
        whole_pixel_veto_not_reencoded=True,
        runtime_implementation="CPU4_ST1_PO1" if variant.endswith("_PO1") else "CPU4",
        speed_optimization_only=not variant.endswith("_PO1"),
        runtime_mode="ComfyUI_in_process", runtime_python=sys.executable,
        shared_ComfyUI_environment=True, shared_dependency_file="requirements.txt",
        isolated_numerical_worker=False, ComfyUI_global_dependencies_changed=False,
        historical_experiment_byte_identity_required=False,
        numerical_runtime_versions=runtime_versions(),
    )
    return target, eight, report
