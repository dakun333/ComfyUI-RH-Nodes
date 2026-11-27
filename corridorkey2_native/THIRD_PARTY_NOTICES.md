# Third-Party Notices

CorridorKey v2 is licensed under the Corridor Key Licence (see `LICENSE`).
It depends on the third-party components below, which are not owned by
Corridor Digital LLC and remain under their own licenses.

Upstream foundation weights are fetched from their original publishers by the
node asset preparation flow. This node can convert the runtime model states to
a local `models/checkpoints/corridorkey2_full.safetensors`. The combined file does
not change any component license; do not treat all weights as MIT licensed.
Original weight directories are not needed when loading the combined file.

## C-RADIOv4-SO400M

- Publisher: NVIDIA Corporation
- Source: https://huggingface.co/nvidia/C-RADIOv4-SO400M
- License: [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)
- Use: frozen visual feature extractor. Code is included in
  `runtime/third_party/C-RADIO/`; source weights are prepared in
  `runtime/vfms/cradio/` and runtime state is included in the combined checkpoint.

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

## MoGe-2 ViT-B

- Publisher: Microsoft Corporation
- Source: https://github.com/microsoft/MoGe, weights at https://huggingface.co/Ruicheng/moge-2-vitb-normal
- License: MIT (code and weights)
- Use: frozen geometry feature extractor. The model code is vendored in
  `runtime/third_party/MoGe/` with its license at `runtime/third_party/MoGe/LICENSE`;
  source weights are prepared in `runtime/vfms/moge/` and the complete runtime
  model state is included in the combined checkpoint.

The vendored MoGe code includes DINOv2 (Meta Platforms, Inc.), licensed under
the Apache License 2.0, also reproduced in `runtime/third_party/MoGe/LICENSE`.

## RVM (Recurrent Video MAE), small

- Publisher: Google DeepMind
- Source: https://github.com/google-deepmind/representations4d
- License: Apache License 2.0 (code), Creative Commons Attribution 4.0 (weights)
- Use: frozen recurrent video feature extractor. `corridorkey/rvm.py` is a
  PyTorch port of the reference architecture at `runtime/corridorkey/rvm.py`;
  source weights are prepared in `runtime/vfms/rvm/` and all runtime-loaded
  model state is included in the combined checkpoint. This is Recurrent Video
  MAE, not Robust Video Matting.


## utils3d source fallback

Full source from EasternJournalist/utils3d commit 83bfa5bda008803f6521ce9547f94023b8afeddd.
Used only when the host package is absent. License: vendor/UTILS3D_LICENSE; provenance: vendor/UTILS3D_SOURCE.json.
