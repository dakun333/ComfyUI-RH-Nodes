"""Mask-guided Q color recovery from the user's 2026-10-27 Mekajiki snapshot.
Only the fixed-input-Alpha path is included; see SOURCE.json for provenance.
"""
from __future__ import annotations
import torch
import torch.nn.functional as torch_f
from .thin_edge_bypass import detect_thin_edge, is_near_black_background

_BACKGROUND_QUANTIZATION_LEVELS = 31
_BACKGROUND_QUANTIZATION_RADIX = _BACKGROUND_QUANTIZATION_LEVELS + 1
_DARK_BACKGROUND_MAX_CODE = 64
_BACKGROUND_MIN_SUPPORT_PERCENT = 20

class MekajikiUnMult:
    @staticmethod
    def _split_image(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 4:
            raise ValueError(
                f"IMAGE must be BHWC, received shape {tuple(image.shape)}"
            )
        if image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
            raise ValueError("IMAGE batch and spatial dimensions must be nonempty")

        channels = int(image.shape[-1])
        work = torch.nan_to_num(
            image.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        if channels == 1:
            rgb = work.repeat(1, 1, 1, 3)
            alpha = torch.ones_like(work[..., 0])
        elif channels == 2:
            rgb = work[..., :1].repeat(1, 1, 1, 3)
            alpha = work[..., 1]
        elif channels == 3:
            rgb = work
            alpha = torch.ones_like(work[..., 0])
        elif channels == 4:
            rgb = work[..., :3]
            alpha = work[..., 3]
        else:
            raise ValueError(
                f"IMAGE must have 1, 2, 3, or 4 channels, received {channels}"
            )
        return rgb, alpha

    @staticmethod
    def _gamut_safe_unpremultiply(
        observed_rgb: torch.Tensor,
        premultiplied_rgb: torch.Tensor,
        divisor: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen Q: clip negatives and jointly scale positive overflow.

        A negative channel does not revert the entire pixel to observed RGB.
        No background-direction rollback or neighborhood recoloring is applied.
        """
        raw = premultiplied_rgb / divisor.unsqueeze(-1)
        nonnegative = raw.clamp_min(0.0)
        peak = torch.amax(nonnegative, dim=-1, keepdim=True)
        protected = nonnegative / peak.clamp_min(1.0)
        return torch.where(
            active.unsqueeze(-1), protected, torch.zeros_like(protected)
        )


class MekajikiMaskedUnMult:
    _ZERO_ALPHA_EPS = 0.5 / 255.0
    _SAFE_ALPHA = 1.0 / 255.0
    _CONSTANT_TOLERANCE = 1.5 / 255.0
    _CONSTANT_SUPPORT = 0.985
    _LOCAL_CONSTANT_AGREEMENT = 0.80
    _MIN_BACKGROUND_SEED_PIXELS = 16
    _ABUNDANT_BACKGROUND_SEED_PIXELS = 256
    _SEED_COLOR_TOLERANCE = 4.0 / 255.0
    _SEED_COLOR_SUPPORT = 0.80

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")

        if mask.ndim == 2:
            work = mask.unsqueeze(0)
        elif mask.ndim == 3:
            work = mask
        elif mask.ndim == 4:
            channel_last = int(mask.shape[-1]) == 1
            channel_first = int(mask.shape[1]) == 1
            if channel_first and tuple(mask.shape[2:]) == image_size:
                work = mask[:, 0]
            elif channel_last and tuple(mask.shape[1:3]) == image_size:
                work = mask[..., 0]
            elif channel_first and not channel_last:
                work = mask[:, 0]
            elif channel_last and not channel_first:
                work = mask[..., 0]
            else:
                raise ValueError(
                    "Ambiguous four-dimensional MASK; use canonical BHW, B1HW, "
                    "or BHWC with one channel matching the image dimensions"
                )
        else:
            raise ValueError(
                "MASK must be HW, BHW, BHWC with one channel, or B1HW; "
                f"received shape {tuple(mask.shape)}"
            )
        if work.shape[0] < 1 or work.shape[-2] < 1 or work.shape[-1] < 1:
            raise ValueError("MASK batch and spatial dimensions must be nonempty")

        return torch.nan_to_num(
            work.to(dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

    @classmethod
    def _validate_background_seeds(
        cls,
        candidate: torch.Tensor,
        rgb: torch.Tensor | None,
        *,
        require_color_consistency: bool,
    ) -> torch.Tensor:
        """Reject tiny isolated mask holes before using them as background."""

        if int(torch.count_nonzero(candidate).item()) < cls._MIN_BACKGROUND_SEED_PIXELS:
            return torch.zeros_like(candidate, dtype=torch.bool)

        neighbors = (
            torch_f.avg_pool2d(
                candidate.to(dtype=torch.float32)[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0]
            * 9.0
        )
        coherent = candidate & (neighbors >= 3.0)
        coherent_count = int(torch.count_nonzero(coherent).item())
        if coherent_count < cls._MIN_BACKGROUND_SEED_PIXELS:
            return torch.zeros_like(candidate, dtype=torch.bool)

        if rgb is None:
            return coherent
        if (
            not require_color_consistency
            and coherent_count >= cls._ABUNDANT_BACKGROUND_SEED_PIXELS
        ):
            return coherent

        values = cls._sample_values(rgb[coherent])
        median = values.median(dim=0).values
        close = (
            torch.amax(torch.abs(values - median), dim=-1)
            <= cls._SEED_COLOR_TOLERANCE
        )
        if float(close.to(dtype=torch.float32).mean().item()) >= cls._SEED_COLOR_SUPPORT:
            return coherent
        return torch.zeros_like(candidate, dtype=torch.bool)

    @classmethod
    def _background_seeds(
        cls, alpha: torch.Tensor, rgb: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select credible pixels where the background is directly observable."""

        exact = alpha <= cls._ZERO_ALPHA_EPS
        known = cls._validate_background_seeds(
            exact,
            rgb,
            require_color_consistency=False,
        )
        if bool(torch.any(known)):
            return known

        low = torch.quantile(alpha.reshape(-1), 0.01)
        if float(low.item()) <= (8.0 / 255.0):
            soft = alpha <= (low + (0.25 / 255.0))
            known = cls._validate_background_seeds(
                soft,
                rgb,
                require_color_consistency=True,
            )
            if bool(torch.any(known)):
                return known

        return torch.zeros_like(alpha, dtype=torch.bool)

    @staticmethod
    def _sample_values(values: torch.Tensor, limit: int = 262_144) -> torch.Tensor:
        if int(values.shape[0]) <= limit:
            return values
        step = (int(values.shape[0]) + limit - 1) // limit
        return values[::step]

    @classmethod
    def _darkest_supported_color(
        cls, values: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the darkest quantized color with at least 20% support."""
        limit = 262_144
        if int(values.shape[0]) > limit:
            step = (int(values.shape[0]) + limit - 1) // limit
            values = values[::step]
            if weights is not None:
                weights = weights[::step]
        levels = _BACKGROUND_QUANTIZATION_LEVELS
        radix = _BACKGROUND_QUANTIZATION_RADIX
        quantized = torch.floor(values.clamp(0.0, 1.0) * levels + 0.5).to(torch.int64)
        codes = (quantized[:, 0] * radix + quantized[:, 1]) * radix + quantized[:, 2]
        unique_codes, inverse = torch.unique(codes, sorted=True, return_inverse=True)
        if weights is None:
            support = torch.bincount(
                inverse, minlength=int(unique_codes.shape[0])
            ).to(dtype=values.dtype)
        else:
            support = torch.zeros(
                int(unique_codes.shape[0]), dtype=values.dtype, device=values.device
            )
            support.scatter_add_(0, inverse, weights.to(dtype=values.dtype))
        minimum_support = support.sum() * (_BACKGROUND_MIN_SUPPORT_PERCENT / 100.0)
        eligible = support >= minimum_support
        if bool(torch.any(eligible).item()):
            red = unique_codes // (radix * radix)
            green = (unique_codes // radix) % radix
            blue = unique_codes % radix
            darkness = 3 * levels - (red + green + blue)
            darkest = torch.amax(darkness[eligible])
            finalists = eligible & (darkness == darkest)
            finalist_support = torch.where(
                finalists, support, torch.full_like(support, -1.0)
            )
            winning_cluster = torch.argmax(finalist_support)
        else:
            winning_cluster = torch.argmax(support)
        selected = values[inverse == winning_cluster]
        return selected.median(dim=0).values.clamp(0.0, 1.0)

    @classmethod
    def _stabilize_near_black_background(
        cls,
        values: torch.Tensor,
        candidate: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        code = torch.floor(candidate.clamp(0.0, 1.0) * 255.0 + 0.5)
        if int(torch.amax(code).item()) > _DARK_BACKGROUND_MAX_CODE:
            return candidate
        return cls._darkest_supported_color(values, weights)

    @staticmethod
    def _dilate_square(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Fast square dilation using separable horizontal/vertical max pools."""

        if radius <= 0:
            return mask
        kernel = radius * 2 + 1
        dilated = torch_f.max_pool2d(
            mask, kernel_size=(1, kernel), stride=1, padding=(0, radius)
        )
        return torch_f.max_pool2d(
            dilated, kernel_size=(kernel, 1), stride=1, padding=(radius, 0)
        )

    @classmethod
    def _near_mask_background(
        cls,
        rgb: torch.Tensor,
        known: torch.Tensor,
        constant_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, float] | None:
        """Average the closest visible background ring outside alpha support."""

        height, width, _ = rgb.shape
        scale = min(1.0, 512.0 / max(height, width))
        small_height = max(1, int(round(height * scale)))
        small_width = max(1, int(round(width * scale)))

        colors = rgb.permute(2, 0, 1).unsqueeze(0)
        weights = known.to(dtype=rgb.dtype)[None, None]
        if (small_height, small_width) != (height, width):
            small_weights = torch_f.interpolate(
                weights, size=(small_height, small_width), mode="area"
            )
            small_colors = torch_f.interpolate(
                colors * weights,
                size=(small_height, small_width),
                mode="area",
            ) / small_weights.clamp_min(1.0e-8)
        else:
            small_weights = weights
            small_colors = colors

        small_known = small_weights >= 0.999
        small_foreground = (~small_known).to(dtype=rgb.dtype)
        inner_radius = max(1, int(round(4.0 * scale)))
        inner = cls._dilate_square(small_foreground, inner_radius) > 0.0

        for outer_pixels in (32, 64, 128):
            outer_radius = max(inner_radius + 1, int(round(outer_pixels * scale)))
            outer = cls._dilate_square(small_foreground, outer_radius) > 0.0
            ring = small_known & outer & (~inner)
            if int(torch.count_nonzero(ring).item()) < 64:
                continue
            ring_weight = ring.to(dtype=rgb.dtype) * small_weights
            total_weight = ring_weight.sum()
            background = (
                (small_colors * ring_weight).sum(dim=(0, 2, 3)) / total_weight
            )
            ring_values = small_colors[0].permute(1, 2, 0)[ring[0, 0]]
            ring_weights = small_weights[0, 0][ring[0, 0]]
            background = cls._stabilize_near_black_background(
                ring_values, background, ring_weights
            )
            agrees = (
                torch.amax(
                    torch.abs(
                        small_colors - constant_reference.view(1, 3, 1, 1)
                    ),
                    dim=1,
                    keepdim=True,
                )
                <= cls._CONSTANT_TOLERANCE
            )
            agreement = float(
                ((agrees.to(dtype=rgb.dtype) * ring_weight).sum() / total_weight)
                .item()
            )
            return background, agreement

        return None

    @classmethod
    def _estimate_uniform_background(
        cls, rgb: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor | None:
        known = cls._background_seeds(alpha, rgb)
        if not bool(torch.any(known)):
            return None

        all_values = rgb[known]
        sampled = cls._sample_values(all_values)
        median = sampled.median(dim=0).values
        close = (
            torch.amax(torch.abs(sampled - median), dim=-1)
            <= cls._CONSTANT_TOLERANCE
        )

        constant_support = float(close.to(dtype=torch.float32).mean().item())
        nearby = cls._near_mask_background(rgb, known, median)
        if nearby is not None:
            local_background, local_constant_agreement = nearby
            if (
                constant_support >= cls._CONSTANT_SUPPORT
                and local_constant_agreement >= cls._LOCAL_CONSTANT_AGREEMENT
            ):
                return median
            return local_background

        if constant_support >= cls._CONSTANT_SUPPORT:
            return median

        fallback = all_values.mean(dim=0)
        return cls._stabilize_near_black_background(all_values, fallback)

    def apply(self, image: torch.Tensor, mask: torch.Tensor):
        rgb, _ = MekajikiUnMult._split_image(image)
        height, width = (int(rgb.shape[1]), int(rgb.shape[2]))
        alpha = self._normalize_mask(mask, (height, width)).to(device=rgb.device)
        if tuple(alpha.shape[-2:]) != (height, width):
            alpha = torch_f.interpolate(alpha.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)[:, 0]
        image_batch = int(rgb.shape[0])
        mask_batch = int(alpha.shape[0])
        if image_batch != mask_batch:
            if image_batch == 1:
                rgb = rgb.expand(mask_batch, -1, -1, -1)
            elif mask_batch == 1:
                alpha = alpha.expand(image_batch, -1, -1)
            else:
                raise ValueError(f'IMAGE and MASK batches must match, or one batch must equal 1; received {image_batch} and {mask_batch}')
        recovered = []
        for sample_rgb, sample_alpha in zip(rgb, alpha):
            background = self._estimate_uniform_background(sample_rgb, sample_alpha)
            if background is not None and (not is_near_black_background(background)) and detect_thin_edge(sample_alpha).thin_edge_dominant:
                recovered.append(sample_rgb)
                continue
            if background is None:
                straight = torch.where((sample_alpha > 0.0).unsqueeze(-1), sample_rgb, torch.zeros_like(sample_rgb))
            else:
                background = background.view(1, 1, 3)
                premultiplied = sample_rgb - background * (1.0 - sample_alpha.unsqueeze(-1))
                divisor = sample_alpha.clamp_min(self._SAFE_ALPHA)
                straight = MekajikiUnMult._gamut_safe_unpremultiply(sample_rgb, premultiplied, divisor, sample_alpha > 0.0)
            recovered.append(straight)
        output_rgb = torch.stack(recovered, dim=0)
        rgba = torch.cat((output_rgb, alpha.unsqueeze(-1)), dim=-1)
        return (rgba, alpha)
