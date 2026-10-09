"""Automatic and mask-guided UnMult nodes for ComfyUI IMAGE tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_f


_DITHER_GENERATOR = torch.Generator(device="cpu")
_DITHER_GENERATOR.manual_seed(0x554E4D55)
_DITHER_TILE = (
    torch.rand((256, 256), generator=_DITHER_GENERATOR, dtype=torch.float32)
    - torch.rand((256, 256), generator=_DITHER_GENERATOR, dtype=torch.float32)
)
_BACKGROUND_QUANTIZATION_LEVELS = 31
_BACKGROUND_QUANTIZATION_RADIX = _BACKGROUND_QUANTIZATION_LEVELS + 1
_BACKGROUND_MIN_SUPPORT_PERCENT = 20
_BACKGROUND_HIGH_CONFIDENCE_PERCENT = 30
_BACKGROUND_SIDE_SUPPORT_PERCENT = 5
_DARK_BACKGROUND_MAX_CODE = 64
_GAMUT_NEGATIVE_TOLERANCE = 2.0 / 255.0
_GAMUT_NEGATIVE_STRAIGHT_TOLERANCE = 0.25


class MekajikiUnMult:
    """Construct straight alpha from an image on a constant background.

    With automatic background recovery disabled this follows classic black-
    background UnMult behavior: ``m=max(R,G,B); RGB/=m; A*=m`` for ``0<m<1``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_threshold": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "8-bit alpha black point. 0 disables cleanup. Values "
                            "above it are remapped continuously toward 255=opaque."
                        ),
                    },
                ),
                "dither": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "0 disables dither. Adds deterministic triangular RGB "
                            "dither up to one 8-bit code value before later saving."
                        ),
                    },
                ),
                "auto_background": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Cluster joint RGB border samples. A dark candidate "
                            "needs at least 20% total support plus multi-side "
                            "coverage; 30% is high confidence. Otherwise the most "
                            "populous color wins. Near-black mattes use light-only "
                            "alpha evidence to suppress amplified black noise. "
                            "White glow artwork receives automatic brightness "
                            "compensation."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("transparent_rgba", "alpha")
    FUNCTION = "apply"
    CATEGORY = "image/matting"
    DESCRIPTION = (
        "Single-image UnMult with automatic constant-background recovery and "
        "adaptive compensation for white-background glow artwork."
    )

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
        """Use common RGB scaling for overflow; preserve severe matte mismatches."""
        raw = premultiplied_rgb / divisor.unsqueeze(-1)
        nonnegative = raw.clamp_min(0.0)
        peak = torch.amax(nonnegative, dim=-1, keepdim=True)
        protected = nonnegative / peak.clamp_min(1.0)
        severe_negative = torch.any(
            (premultiplied_rgb < -_GAMUT_NEGATIVE_TOLERANCE)
            & (raw < -_GAMUT_NEGATIVE_STRAIGHT_TOLERANCE),
            dim=-1,
            keepdim=True,
        )
        protected = torch.where(severe_negative, observed_rgb, protected)
        return torch.where(
            active.unsqueeze(-1), protected, torch.zeros_like(protected)
        )

    @staticmethod
    def _estimate_background(rgb: torch.Tensor) -> torch.Tensor:
        """Return the darkest sufficiently populated border cluster per image."""

        batch, height, width, _ = rgb.shape
        if height < 1 or width < 1:
            raise ValueError("Cannot estimate a background from an empty IMAGE")

        border = max(1, min(32, (min(height, width) + 49) // 50))
        strips = [
            rgb[:, :border, :, :].reshape(batch, -1, 3),
            rgb[:, max(height - border, 0) :, :, :].reshape(batch, -1, 3),
        ]
        if height > 2 * border:
            middle = rgb[:, border : height - border, :, :]
            strips.extend(
                (
                    middle[:, :, :border, :].reshape(batch, -1, 3),
                    middle[:, :, max(width - border, 0) :, :].reshape(
                        batch, -1, 3
                    ),
                )
            )

        side_lengths = [int(strip.shape[1]) for strip in strips]
        samples = torch.cat(strips, dim=1)
        levels = _BACKGROUND_QUANTIZATION_LEVELS
        radix = _BACKGROUND_QUANTIZATION_RADIX
        backgrounds = []
        for image_samples in samples:
            quantized = torch.floor(
                image_samples.clamp(0.0, 1.0) * levels + 0.5
            ).to(torch.int64)
            codes = (
                (quantized[:, 0] * radix + quantized[:, 1]) * radix
                + quantized[:, 2]
            )
            unique_codes, inverse, counts = torch.unique(
                codes,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )

            red = unique_codes // (radix * radix)
            green = (unique_codes // radix) % radix
            blue = unique_codes % radix
            darkness = 3 * levels - (red + green + blue)
            sample_count = int(image_samples.shape[0])
            minimum_population = (
                sample_count * _BACKGROUND_MIN_SUPPORT_PERCENT + 99
            ) // 100
            high_confidence_population = (
                sample_count * _BACKGROUND_HIGH_CONFIDENCE_PERCENT + 99
            ) // 100

            side_counts = []
            offset = 0
            cluster_count = int(unique_codes.shape[0])
            for side_length in side_lengths:
                side_inverse = inverse[offset : offset + side_length]
                side_counts.append(
                    torch.bincount(side_inverse, minlength=cluster_count)
                )
                offset += side_length
            side_counts = torch.stack(side_counts, dim=0)
            side_lengths_tensor = counts.new_tensor(side_lengths)[:, None]
            meaningful_sides = (
                side_counts * 100
                >= side_lengths_tensor * _BACKGROUND_SIDE_SUPPORT_PERCENT
            )
            meaningful_side_count = torch.sum(meaningful_sides, dim=0)
            opposite_sides = meaningful_sides[0] & meaningful_sides[1]
            if meaningful_sides.shape[0] == 4:
                opposite_sides = opposite_sides | (
                    meaningful_sides[2] & meaningful_sides[3]
                )
            strong_low_confidence_coverage = opposite_sides | (
                meaningful_side_count >= 3
            )
            high_confidence_coverage = meaningful_side_count >= 2
            high_confidence = counts >= high_confidence_population
            spatially_supported = torch.where(
                high_confidence,
                high_confidence_coverage,
                strong_low_confidence_coverage,
            )
            eligible = (counts >= minimum_population) & spatially_supported

            if bool(torch.any(eligible).item()):
                scores = darkness * (sample_count + 1) + counts.to(torch.int64)
                scores = torch.where(
                    eligible, scores, torch.full_like(scores, -1)
                )
                winning_cluster = torch.argmax(scores)
            else:
                winning_cluster = torch.argmax(counts)

            selected = image_samples[inverse == winning_cluster]
            backgrounds.append(selected.median(dim=0).values.clamp(0.0, 1.0))

        return torch.stack(backgrounds, dim=0)

    @staticmethod
    def _dither(height: int, width: int, device: torch.device) -> torch.Tensor:
        repeat_y = (height + 255) // 256
        repeat_x = (width + 255) // 256
        return _DITHER_TILE.repeat(repeat_y, repeat_x)[:height, :width].to(device)

    def apply(
        self,
        image: torch.Tensor,
        black_threshold: int = 0,
        dither: float = 0.0,
        auto_background: bool = True,
    ):
        rgb, input_alpha = self._split_image(image)
        if bool(auto_background):
            estimated_background = self._estimate_background(rgb)
        else:
            estimated_background = torch.zeros(
                (int(rgb.shape[0]), 3), dtype=rgb.dtype, device=rgb.device
            )
        background = estimated_background[:, None, None, :]

        epsilon = torch.finfo(rgb.dtype).eps
        delta = rgb - background
        toward_white = delta / (1.0 - background).clamp_min(epsilon)
        toward_black = -delta / background.clamp_min(epsilon)
        two_sided_alpha = torch.where(delta >= 0.0, toward_white, toward_black)
        if bool(auto_background):
            background_code = torch.floor(
                estimated_background.clamp(0.0, 1.0) * 255.0 + 0.5
            ).to(torch.int64)
            light_only_background = (
                torch.amax(background_code, dim=-1) <= _DARK_BACKGROUND_MAX_CODE
            )
        else:
            light_only_background = torch.zeros(
                int(rgb.shape[0]), dtype=torch.bool, device=rgb.device
            )

        light_only_alpha = toward_white.clamp_min(0.0)
        channel_alpha = torch.where(
            light_only_background[:, None, None, None],
            light_only_alpha,
            two_sided_alpha,
        )
        matte_alpha = torch.amax(channel_alpha, dim=-1).clamp(0.0, 1.0)

        if bool(auto_background):
            gains = []
            for sample_alpha in matte_alpha:
                positive_alpha = sample_alpha[sample_alpha > 0.0]
                if positive_alpha.numel() > 0:
                    median_alpha = positive_alpha.median()
                    gain = (1.0 + 1.55 * torch.sqrt(median_alpha)).clamp(
                        1.0, 2.5
                    )
                else:
                    gain = sample_alpha.new_tensor(1.0)
                gains.append(gain)
            if gains:
                gain = torch.stack(gains)[:, None, None]
                white_background = torch.all(
                    estimated_background >= (254.5 / 255.0), dim=-1
                )[:, None, None]
                lifted_alpha = (matte_alpha * gain).clamp(0.0, 1.0)
                matte_alpha = torch.where(
                    white_background, lifted_alpha, matte_alpha
                )

        divisor = torch.where(
            matte_alpha > 0.0, matte_alpha, torch.ones_like(matte_alpha)
        )
        premultiplied_rgb = rgb - background * (
            1.0 - matte_alpha.unsqueeze(-1)
        )
        output_rgb = self._gamut_safe_unpremultiply(
            rgb, premultiplied_rgb, divisor, matte_alpha > 0.0
        )

        threshold_code = min(max(int(black_threshold), 0), 255)
        if threshold_code == 0:
            alpha_factor = matte_alpha
        elif threshold_code >= 255:
            alpha_factor = torch.zeros_like(matte_alpha)
        else:
            threshold = threshold_code / 255.0
            alpha_factor = ((matte_alpha - threshold) / (1.0 - threshold)).clamp(
                0.0, 1.0
            )
        output_alpha = input_alpha * alpha_factor
        discarded = alpha_factor <= 0.0

        output_rgb = torch.where(
            discarded.unsqueeze(-1), torch.zeros_like(output_rgb), output_rgb
        )
        output_alpha = torch.where(
            discarded, torch.zeros_like(output_alpha), output_alpha
        )

        dither_strength = min(max(float(dither), 0.0), 1.0)
        if dither_strength > 0.0:
            noise = self._dither(
                int(output_rgb.shape[1]),
                int(output_rgb.shape[2]),
                output_rgb.device,
            )
            output_rgb = output_rgb + (
                noise.unsqueeze(0).unsqueeze(-1)
                * (dither_strength / 255.0)
            )
            output_rgb = output_rgb.clamp(0.0, 1.0)
            output_rgb = torch.where(
                (output_alpha > 0.0).unsqueeze(-1),
                output_rgb,
                torch.zeros_like(output_rgb),
            )

        rgba = torch.cat((output_rgb, output_alpha.unsqueeze(-1)), dim=-1)
        return (rgba, output_alpha)


class MekajikiMaskedUnMult:
    """Construct straight RGB from a flattened image and a known opacity mask."""

    _ZERO_ALPHA_EPS = 0.5 / 255.0
    _SAFE_ALPHA = 1.0 / 255.0
    _CONSTANT_TOLERANCE = 1.5 / 255.0
    _CONSTANT_SUPPORT = 0.985
    _LOCAL_CONSTANT_AGREEMENT = 0.80
    _MIN_BACKGROUND_SEED_PIXELS = 16
    _ABUNDANT_BACKGROUND_SEED_PIXELS = 256
    _SEED_COLOR_TOLERANCE = 4.0 / 255.0
    _SEED_COLOR_SUPPORT = 0.80

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Flattened source image. RGB is treated as the observed "
                            "composite C; any existing image alpha is ignored."
                        ),
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Known foreground opacity: 0 is transparent and 1 is "
                            "opaque. Preserved unless allow_alpha_adjustment is on; "
                            "then only necessary overflow pixels may increase."
                        ),
                    },
                ),
            },
            "optional": {
                "allow_alpha_adjustment": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Off: Method 1 preserves alpha. On: gray-anchor overflow "
                            "recovery with a local alpha increase of at most 43/255 "
                            "(below 17 percentage points). White backgrounds may "
                            "look darker; matte noise may be amplified."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("transparent_rgba", "alpha")
    FUNCTION = "apply"
    CATEGORY = "image/matting"
    DESCRIPTION = (
        "Use a known opacity mask and one representative background color to "
        "brighten flattened translucent content without spatial reconstruction seams."
    )

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

    def apply(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        allow_alpha_adjustment: bool = False,
    ):
        rgb, _ = MekajikiUnMult._split_image(image)
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        alpha = self._normalize_mask(mask, (height, width)).to(device=rgb.device)
        if tuple(alpha.shape[-2:]) != (height, width):
            alpha = torch_f.interpolate(
                alpha.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        image_batch = int(rgb.shape[0])
        mask_batch = int(alpha.shape[0])
        if image_batch != mask_batch:
            if image_batch == 1:
                rgb = rgb.expand(mask_batch, -1, -1, -1)
            elif mask_batch == 1:
                alpha = alpha.expand(image_batch, -1, -1)
            else:
                raise ValueError(
                    "IMAGE and MASK batches must match, or one batch must equal 1; "
                    f"received {image_batch} and {mask_batch}"
                )

        if allow_alpha_adjustment:
            from .unmult_alpha_recovery import recover_gray17

        recovered = []
        adjusted_alphas = [] if allow_alpha_adjustment else None
        for sample_rgb, sample_alpha in zip(rgb, alpha):
            background = self._estimate_uniform_background(sample_rgb, sample_alpha)
            if background is None:
                straight = torch.where(
                    (sample_alpha > 0.0).unsqueeze(-1),
                    sample_rgb,
                    torch.zeros_like(sample_rgb),
                )
            else:
                background = background.view(1, 1, 3)
                premultiplied = sample_rgb - background * (
                    1.0 - sample_alpha.unsqueeze(-1)
                )
                divisor = sample_alpha.clamp_min(self._SAFE_ALPHA)
                straight = MekajikiUnMult._gamut_safe_unpremultiply(
                    sample_rgb, premultiplied, divisor, sample_alpha > 0.0
                )
            if allow_alpha_adjustment:
                if background is not None:
                    straight, sample_alpha = recover_gray17(
                        premultiplied, sample_alpha, straight
                    )
                adjusted_alphas.append(sample_alpha)
            recovered.append(straight)

        output_rgb = torch.stack(recovered, dim=0)
        if allow_alpha_adjustment:
            alpha = torch.stack(adjusted_alphas, dim=0)
        rgba = torch.cat((output_rgb, alpha.unsqueeze(-1)), dim=-1)
        return (rgba, alpha)


NODE_CLASS_MAPPINGS = {
    "MekajikiUnMult": MekajikiUnMult,
    "MekajikiMaskedUnMult": MekajikiMaskedUnMult,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MekajikiUnMult": "UnMult (Auto Background)",
    "MekajikiMaskedUnMult": "Mask-Guided UnMult (Uniform Background)",
}
