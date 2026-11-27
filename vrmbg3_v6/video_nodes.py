"""Native ComfyUI VIDEO input and alpha-preserving MOV export, separate from model math."""
from __future__ import annotations
import json, math, subprocess, time, wave, shutil
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from .nodes import check_images, VRMBG3V6Matting


def _validate_file_timestamps(video):
    # VideoFromComponents is already an explicit CFR frame batch. For file-backed
    # native VIDEO, do not let get_components silently flatten a VFR timeline.
    from comfy_api.latest import InputImpl
    if not isinstance(video, InputImpl.VideoFromFile):
        return
    source = video.get_stream_source()
    if hasattr(source, "seek"): source.seek(0)
    start, duration = video.get_active_trim_window()
    last = None
    count = 0
    dimensions = None
    with av.open(source) as container:
        stream = container.streams.video[0]
        if not stream.average_rate: raise ValueError("Video has no valid frame rate")
        delta = 1 / float(stream.average_rate)
        for frame in container.decode(stream):
            mm.throw_exception_if_processing_interrupted()
            if frame.pts is None: raise ValueError("Video frame has no timestamp")
            stamp = float(frame.pts * frame.time_base)
            if stamp < start: continue
            if duration and stamp >= start + duration: break
            if last is not None and abs(stamp - last - delta) > 1e-4:
                raise ValueError("Variable frame rate detected; convert to constant fps before matting")
            last = stamp
            count += 1
            fw, fh = frame.width, frame.height
            if int(round(frame.rotation / 90)) % 2: fw, fh = fh, fw
            if dimensions is not None and dimensions != (fw, fh): raise ValueError("Changing video frame dimensions unsupported")
            dimensions = (fw, fh)
        if not count: raise ValueError("Empty video slice")
        return {"frames": count, "width": dimensions[0], "height": dimensions[1], "fps": Fraction(stream.average_rate)}


def _decode_mapped_file(video, scan, *, shared_rgba=False):
    # Match native VideoFromFile RGB format/range, but never stack all decoded
    # frames twice in RAM. No temporal segmentation occurs in model inference.
    from .batch_memory import mapped_tensor
    from comfy_api.latest import Types
    n, w, h = scan["frames"], scan["width"], scan["height"]
    root = Path(folder_paths.get_temp_directory()) / "vrmbg3_v6_batches"
    root.mkdir(parents=True, exist_ok=True)
    required = n * w * h * (7 if shared_rgba else 10) * 4
    free = shutil.disk_usage(root).free
    if free < required + (2 << 30):
        raise ValueError(f"VRMBG3 native FP32 mapped batches need {required/(1<<30):.2f}GiB temp space plus 2GiB reserve; available {free/(1<<30):.2f}GiB")
    rgba = mapped_tensor((n, h, w, 4)) if shared_rgba else None
    images = rgba[..., :3] if shared_rgba else mapped_tensor((n, h, w, 3))
    source = video.get_stream_source()
    if hasattr(source, "seek"): source.seek(0)
    start, duration = video.get_active_trim_window()
    index = 0
    with av.open(source) as c:
        for frame in c.decode(video=0):
            mm.throw_exception_if_processing_interrupted()
            stamp = float(frame.pts * frame.time_base)
            if stamp < start: continue
            if duration and stamp >= start + duration: break
            if frame.format.name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8"):
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255
            else:
                arr = frame.to_ndarray(format="gbrpf32le")
            if frame.rotation != 0:
                arr = np.rot90(arr, k=int(round(frame.rotation // 90)), axes=(0, 1)).copy()
            if index >= n or arr.shape != (h, w, 3): raise ValueError("Video changed between scan and decode")
            np.copyto(images[index].numpy(), arr)
            index += 1
    if index != n: raise ValueError("Video frame count changed between scan and decode")
    audio = None
    source = video.get_stream_source()
    if hasattr(source, "seek"): source.seek(0)
    with av.open(source) as c:
        if c.streams.audio:
            stream = c.streams.audio[-1]
            if n / float(scan["fps"]) > 600:
                raise ValueError("Audio batch longer than 10 minutes; use Video Slice")
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            chunks = []
            first = False
            done = False
            for packet in c.demux(stream):
                mm.throw_exception_if_processing_interrupted()
                for decoded in packet.decode():
                    for frame in resampler.resample(decoded):
                        if duration and frame.time > start + duration:
                            done = True; break
                        if not first:
                            skip = max(0, int((start - frame.pts * stream.time_base) * stream.sample_rate))
                            if skip < frame.samples:
                                first = True; chunks.append(frame.to_ndarray()[..., skip:])
                        else: chunks.append(frame.to_ndarray())
                    if done: break
                if done: break
            if chunks:
                waveform = np.concatenate(chunks, axis=1)
                if duration: waveform = waveform[..., :int(duration * stream.sample_rate)]
                audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": int(stream.sample_rate)}
    components = Types.VideoComponents(images=images, frame_rate=scan["fps"], audio=audio)
    return (components, rgba) if shared_rgba else components


class VRMBG3V6MattingVideo:
    @classmethod
    def INPUT_TYPES(cls):
        fields = VRMBG3V6Matting.INPUT_TYPES()["required"].copy()
        fields.pop("images")
        fields = {"model": fields.pop("model"), "video": ("VIDEO",), **fields}
        return {"required": fields, "optional": VRMBG3V6Matting.INPUT_TYPES()["optional"]}
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "FLOAT", "AUDIO")
    RETURN_NAMES = ("rgba", "alpha", "checker_preview", "run_info", "fps", "audio")
    FUNCTION = "infer_video"
    CATEGORY = "RH-Nodes/matting"

    def infer_video(self, model, video, max_pixels=1048576, precision="bf16", temporal=True, offload_after=True, alpha_shrink=0, alpha_feather=0., enable_alpha_unmult=False, color_recovery_method="multilevel"):
        if not hasattr(video, "get_components"): raise ValueError("Connect native ComfyUI Load Video VIDEO output")
        from comfy_api.latest import InputImpl
        begin = time.perf_counter()
        scan = _validate_file_timestamps(video)
        scan_seconds = time.perf_counter() - begin
        if scan:
            n, w, h = scan["frames"], scan["width"], scan["height"]
        else:
            w, h = video.get_dimensions()
            n = video.get_frame_count()
        # Above 1GiB of RGB, use bounded decode + file-backed native-size outputs.
        # The old 4GiB cutoff incorrectly rejected a 4-second 1440p/48fps clip.
        file_backed = n * w * h * 12 > (1 << 30)
        decode_start = time.perf_counter()
        rgba_buffer = None
        if file_backed and isinstance(video, InputImpl.VideoFromFile):
            components, rgba_buffer = _decode_mapped_file(video, scan, shared_rgba=True)
        else:
            components = video.get_components()
        decode_seconds = time.perf_counter() - decode_start
        images = components.images
        fps = float(components.frame_rate)
        if not math.isfinite(fps) or fps <= 0: raise ValueError("Invalid VIDEO frame rate")
        rgba, alpha, checker, info = VRMBG3V6Matting().infer(model, images, max_pixels, precision, temporal, offload_after, _file_backed=file_backed, _rgba_buffer=rgba_buffer, alpha_shrink=alpha_shrink, alpha_feather=alpha_feather, enable_alpha_unmult=enable_alpha_unmult, color_recovery_method=color_recovery_method)
        details = json.loads(info)
        details.update({"input_type": "native ComfyUI VIDEO", "fps": fps, "source_audio": components.audio is not None,
                        "cpu_batch_storage": "file-backed FP32 / bounded frame decode" if file_backed else "RAM FP32",
                        "rgb_batch_GiB": images.numel() * 4 / (1 << 30), "all_frames_one_temporal_sequence": True,
                        "RGB_RGBA_shared_storage": rgba_buffer is not None,
                        "VIDEO_timings_seconds": {"scan": scan_seconds, "decode_and_allocate": decode_seconds,
                                                  "total": time.perf_counter() - begin}})
        return rgba, alpha, checker, json.dumps(details, ensure_ascii=False), fps, components.audio


def _audio_wav(audio, path, frames, rate):
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Expected ComfyUI AUDIO waveform/sample_rate")
    x = audio["waveform"]
    sr = int(audio["sample_rate"])
    if sr < 8000 or sr > 384000: raise ValueError("Unsupported AUDIO sample rate")
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 8 or x.shape[2] < 1:
        raise ValueError("Expected one AUDIO batch [1,channels,samples]")
    if not torch.isfinite(x).all(): raise ValueError("AUDIO has nonfinite values")
    x = x[0].detach().cpu().float()
    samples = round(Fraction(frames, 1) / rate * sr)
    original = x.shape[-1]
    if original < samples: x = torch.nn.functional.pad(x, (0, samples-original))
    else: x = x[:, :samples]
    clipped = int((x.abs() > 1).sum())
    pcm = np.rint(x.clamp(-1, 1).numpy().T.astype(np.float64) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(x.shape[0]); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())
    return {"sample_rate": sr, "channels": x.shape[0], "source_samples": original, "saved_samples": samples,
            "tail_adjustment_samples": samples-original, "clipped_values": clipped,
            "mov_codec": "pcm_s16le", "preview_codec": "aac"}


class VRMBG3SaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"rgba": ("IMAGE",), "alpha": ("MASK",), "fps": ("FLOAT", {"default": 24., "min": .1, "max": 240.}),
               "filename_prefix": ("STRING", {"default": "VRMBG3/localhalo381_ep4"}), "save_alpha16": ("BOOLEAN", {"default": True})},
               "optional": {"run_info": ("STRING", {"default": ""}), "audio": ("AUDIO",),
                            "mov_codec": (["prores_4444", "qtrle_lossless"], {"default": "prores_4444"})},
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "save"
    CATEGORY = "RH-Nodes/matting"
    OUTPUT_NODE = True

    def save(self, rgba, alpha, fps=24., filename_prefix="VRMBG3/localhalo381_ep4", save_alpha16=True,
             run_info="", prompt=None, extra_pnginfo=None, audio=None, mov_codec="prores_4444"):
        validation_start = time.perf_counter()
        check_images(rgba)
        n, h, w, channels = rgba.shape
        if channels != 4 or alpha.shape != (n, h, w): raise ValueError("RGBA/foreground alpha shape mismatch")
        shared_alpha = (rgba.device == alpha.device and rgba[..., 3].data_ptr() == alpha.data_ptr()
                        and rgba[..., 3].stride() == alpha.stride())
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            # RGBA finite checking above includes this exact alpha storage view.
            if shared_alpha:
                if alpha[i].min() < 0 or alpha[i].max() > 1: raise ValueError("Invalid alpha")
            else:
                if not torch.isfinite(alpha[i]).all() or alpha[i].min() < 0 or alpha[i].max() > 1: raise ValueError("Invalid alpha")
                if not torch.allclose(rgba[i, ..., 3].cpu(), alpha[i].cpu(), atol=1e-6, rtol=0): raise ValueError("RGBA alpha differs from MASK")
        validation_seconds = time.perf_counter() - validation_start
        if not math.isfinite(fps) or fps <= 0: raise ValueError("fps must be finite and positive")
        if mov_codec not in ("prores_4444", "qtrle_lossless"): raise ValueError("Unsupported MOV codec")
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        rate = Fraction(str(fps)).limit_denominator(1000000)
        rate_string = f"{rate.numerator}/{rate.denominator}"
        out, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), w, h)
        while True:
            output = Path(out) / f"{name}_{counter:05d}_vrmbg3"
            try: output.mkdir(exist_ok=False); break
            except FileExistsError: counter += 1
        vw, vh = w+w%2, h+h%2
        audio_path = output / "._audio.wav"
        audio_info = None
        writers = []
        start = time.perf_counter()
        class Writer:
            def __init__(self, filename, pix, ww, hh, codec, audio_codec=None):
                self.path = output / filename
                self.err = open(str(self.path)+".encode.log", "wb")
                args = [ff, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{ww}x{hh}", "-r", rate_string, "-i", "pipe:0"]
                if audio_info and audio_codec:
                    args += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", audio_codec]
                    if audio_codec == "aac": args += ["-b:a", "192k"]
                else: args += ["-an"]
                self.p = subprocess.Popen([*args, *codec, str(self.path)], stdin=subprocess.PIPE, stderr=self.err)
                writers.append(self)
            def write(self, data): self.p.stdin.write(np.ascontiguousarray(data).tobytes())
            def close(self):
                if not self.p.stdin.closed: self.p.stdin.close()
                rc = self.p.wait(timeout=120); self.err.close()
                if rc: raise RuntimeError(f"FFmpeg export failed ({rc}): {self.path}; see its encode.log")
        paths = {"transparent": "transparent.mov", "checker": "checker.mp4"}
        success = False
        try:
            if audio is not None: audio_info = _audio_wav(audio, audio_path, n, rate)
            if mov_codec == "prores_4444":
                mov = Writer(paths["transparent"], "rgba64le", w, h,
                    ["-c:v", "prores_ks", "-profile:v", "4", "-alpha_bits", "16", "-pix_fmt", "yuva444p10le", "-threads", "4", "-movflags", "+faststart"], "pcm_s16le")
            else:
                mov = Writer(paths["transparent"], "rgba", w, h,
                    ["-c:v", "qtrle", "-pix_fmt", "argb", "-movflags", "+faststart"], "pcm_s16le")
            check = Writer(paths["checker"], "rgb24", vw, vh,
                ["-c:v", "libx264", "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "4", "-movflags", "+faststart"], "aac")
            aw = None
            if save_alpha16:
                paths["alpha16"] = "alpha16.mkv"
                aw = Writer(paths["alpha16"], "gray16le", w, h, ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", "-threads", "2"])
            yy, xx = np.indices((h,w))
            bg = np.repeat(np.where(((yy//32+xx//32)%2)[..., None], 210, 170).astype(np.float32), 3, axis=2)
            progress = ProgressBar(n)
            for i in range(n):
                mm.throw_exception_if_processing_interrupted()
                rgb = rgba[i,...,:3].detach().cpu().numpy(); a = alpha[i].detach().cpu().numpy()
                if mov_codec == "prores_4444":
                    mov.write(np.rint(np.dstack([rgb, a])*65535).astype("<u2"))
                else:
                    mov.write(np.rint(np.dstack([rgb, a])*255).astype(np.uint8))
                composite = np.clip(rgb*255*a[..., None]+bg*(1-a[..., None]),0,255).astype(np.uint8)
                check.write(np.pad(composite, ((0,vh-h),(0,vw-w),(0,0)), mode="edge"))
                if aw: aw.write(np.rint(a*65535).astype("<u2"))
                progress.update(1)
            for writer in writers: writer.close()
            success = True
        except Exception as error:
            (output/"FAILED.json").write_text(json.dumps({"error": str(error), "complete": False}), encoding="utf8")
            raise
        finally:
            if not success:
                for writer in writers:
                    try:
                        if not writer.p.stdin.closed: writer.p.stdin.close()
                        writer.p.wait(timeout=30)
                    except (BrokenPipeError, subprocess.TimeoutExpired):
                        # These are only the FFmpeg child processes created by this save call.
                        writer.p.terminate(); writer.p.wait(timeout=10)
                    finally: writer.err.close()
            if audio_path.exists(): audio_path.unlink()
        metadata = {"model_info": run_info, "frames": n, "fps": fps, "rate": rate_string,
                    "native_size": [w,h], "preview_size": [vw,vh], "container": "mov", "mov_codec": mov_codec,
                    "mov_alpha": "straight alpha; prores 4444 via yuva444p10le (not float-lossless)" if mov_codec=="prores_4444" else "straight alpha; lossless RGBA at 8-bit quantization",
                    "alpha16_divisor": 65535, "alpha_semantics": "foreground=1, no minmax/threshold",
                    "audio": audio_info if audio_info else "silent (no AUDIO input)", "prompt": prompt, "extra_pnginfo": extra_pnginfo,
                    "seconds": time.perf_counter()-start, "validation_seconds": validation_seconds, "complete": True, "outputs": {k: str(output/v) for k,v in paths.items()}}
        (output/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf8")
        preview = {"filename": paths["checker"], "subfolder": str(Path(subfolder) / output.name).replace("\\", "/"), "type": "output", "format": "video/mp4",
                   "mov_filename": paths["transparent"], "width": vw, "height": vh}
        # Dedicated DOM preview extension: browser plays MP4, while MOV retains alpha for editors.
        return {"ui": {"vrmbg3_preview": [preview]}, "result": (json.dumps(metadata["outputs"], ensure_ascii=False),)}
