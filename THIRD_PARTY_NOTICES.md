# Third-Party Notices

## Y367 去纹（1.9.0）

`y367_destripe/` 来自贡献者的 Y367-FAST 2026-11-18 本地文件快照。贡献者已确认算法、封装及 CNN 权重均为原创并授权按根 MIT 许可证公开发布。模型随包提供，SHA-256：`32c4375a718762bec6352c12f320d3f760a04cc03a8b39ba46f2d09967870268`；原文件哈希见 `source_snapshot.json`。Numba（BSD-2-Clause）、llvmlite（BSD-style / LLVM）、threadpoolctl（BSD-3-Clause）及 NumPy/SciPy/OpenCV/Pillow 保留各自许可证，不作为本仓库源码授权。

This repository contains MIT-licensed project source code and the separately
licensed CorridorKey2 module described below. It declares external
Python packages in `requirements.txt`. Those packages are installed separately
and retain their own licenses.

## PyMaxflow

`OPR_RestoreOriginalPixels` method `B` uses PyMaxflow 1.3.2 for graph-cut
segmentation. PyMaxflow is not vendored in this repository. It is distributed
separately under the GNU General Public License version 3 (GPL-3.0), according
to its package metadata and tagged source distribution. Its C++ graph-cut core
credits Vladimir Kolmogorov and related authors in the upstream project.

- Project: https://github.com/pmneila/PyMaxflow
- Version: 1.3.2
- License: GNU General Public License version 3
- Tagged source: https://github.com/pmneila/PyMaxflow/tree/v1.3.2

All other dependencies likewise remain subject to their respective licenses.

## CorridorKey2 与 Pure Light Matting（1.8.0）

`corridorkey2/` 包含 CorridorKey 2.0 上游代码（固定提交 `485f5e9ab759ead6138cc990776184b59d6ea971`）及本地桥接封装。贡献者确认已获得本次 RH-nodes 集成和重新分发授权；本仓库保留其 **Corridor Key Licence（CC BY-NC-SA 4.0 加附加条款）**，不将其改为 MIT。详见 `corridorkey2/LICENSE`、`corridorkey2/runtime/LICENSE` 和各自 THIRD_PARTY_NOTICES.md。仓库根 MIT 许可不覆盖这些代码、模型或第三方组件。

随附 MoGe 代码为 MIT，内含 DINOv2 为 Apache-2.0，原文保留于 `corridorkey2/runtime/third_party/MoGe/LICENSE`。C-RADIO 使用 NVIDIA Open Model License；RVM 代码 Apache-2.0、权重 CC BY 4.0。模型通过固定官方地址按 manifest 下载，不随仓库分发。全部来源见 CorridorKey2 第三方声明。

`pure_light_matting/` 为贡献者确认有权发布的原创 Auto Q / UnMult 光效算法和封装，授权按本仓库 MIT 发布，不包含 AE 二进制或 CorridorKey 源码。torch、NumPy、ComfyUI 及 PyAV 仍按各自许可提供。
