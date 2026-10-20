"""Reporting/export endpoint for color-conditioned maps without pixel reuse."""
import numpy as np


def finish_color_only(layers, original, source_rgba, cfg, before, output_alpha,
                      eligible, protected, weights, stats, supports, shadow_guards,
                      shadow_core, shadow_guard):
    from .core import composite
    shape = eligible.shape
    reference = source_rgba[..., :3].astype(np.float32)
    for layer, old, stat in zip(layers, original, stats):
        layer.rgb = np.rint(np.clip(layer.rgb, 0, 255))
        # Undefined transparent RGB is not training evidence or an edit target.
        layer.rgb[layer.alpha == 0] = old.rgb[layer.alpha == 0]
        np.testing.assert_array_equal(layer.alpha, old.alpha)
        stat.update(exact_reuse_pixels=0, interior_alpha_repaired_pixels=0)
    after, final_alpha = composite(layers, shape)
    np.testing.assert_array_equal(final_alpha, output_alpha)
    before_u8 = np.rint(np.clip(before, 0, 255)).astype(np.uint8)
    after_u8 = np.rint(np.clip(after, 0, 255)).astype(np.uint8)

    def metrics(rgb):
        diff = np.abs(rgb.astype(np.float32) - reference)[eligible]
        return {"mae": float(diff.mean()) if diff.size else None,
                "p95_max_channel_error": float(np.percentile(diff.max(axis=-1), 95)) if diff.size else None,
                "exact_rgb_pixels": int((np.all(rgb == source_rgba[..., :3], axis=-1) & eligible).sum())}

    # Accidental RGB equality is a metric, NOT a claim of pixel borrowing.
    report = {
        "algorithm_version": "6.4.3-color-only",
        "mode": "color_only",
        "color_mapping": "robust diagonal affine + color-conditioned residual; no spatial RGB borrowing",
        "canvas": [shape[1], shape[0]],
        "reference_opaque_pixels": int((source_rgba[..., 3] == 255).sum()),
        "evaluated_pixels": int(eligible.sum()),
        "before": metrics(before_u8), "after": metrics(after_u8),
        "regions": {}, "verified_exact_reuse_pixels": 0,
        "conflict_pixels": 0, "user_protected_pixels": int(protected.sum()),
        "alpha_preserved": True, "alpha_preserved_relative_to": "normalized_aligned_input",
        "alpha_levels": {"enabled": bool(cfg.alpha_levels), "black": cfg.alpha_black,
                         "white": cfg.alpha_white, "method": "linear_levels",
                         "late_repair_enabled": False,
                         "changed_native_pixels": sum(l.metadata.get("alpha_normalization", {}).get("changed_pixels", 0) for l in layers)},
        "interior_alpha_repaired_pixels": 0,
        "composite_alpha_preserved": True, "contour_alpha_preserved": True,
        "source_alpha_mismatch_pixels": int((np.rint(final_alpha*255).astype(np.uint8) != source_rgba[..., 3]).sum()),
        "removed_shadow_protection": {"method": "exclude detected shadows from color-map fitting; no per-pixel restoration"},
        "layers": stats,
        "limitations": [
            "Reference pixels are fitting samples only; no spatial residual, exact projection, edge solve or contour copying is performed.",
            "Color-only preserves model geometry/detail rather than reference texture, so composite error can exceed reuse mode.",
            "Reliable-sample and removed-shadow classification remain heuristic; hidden colors are inferred, not recovered ground truth.",
            "Alpha normalization is independent; disable Alpha Levels to retain raw coverage byte-for-byte.",
        ],
    }
    false = np.zeros(shape, bool)
    zero = np.zeros(shape, np.float32)
    diagnostics = {
        "before": before_u8, "after": after_u8, "alpha": final_alpha,
        "before_premultiplied": before, "after_premultiplied": after,
        "exact": false, "conflicts": false.copy(), "trust": zero,
        "supports": supports, "background_protection": false.copy(),
        "edge_gamut_conflicts": false.copy(), "edge_exact": false.copy(),
        "interior_alpha_repaired": false.copy(),
        "frequency_edge_gate": zero.copy(), "frequency_edge_changed": false.copy(),
        "removed_shadow_core": shadow_core, "removed_shadow_guard": shadow_guard,
        "removed_shadow_layer_guards": shadow_guards,
    }
    return layers, report, diagnostics
