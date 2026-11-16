"""Local AE-UnMult-derived glow kernel: frozen Auto Q math, no Mask-Guided code.

Computational methods copied from the user's local 2026-10-27 snapshot.
See LIGHT_EFFECTS_SOURCE.md. Operates in the input numeric color space, NOT
implicitly in linear light. No .aex binary, model weights, or private imports.
"""
from __future__ import annotations

import torch

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



class LightEffectsUnmultKernel:
    """Unchanged light-only Auto Q computation; this class is not registered."""

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

