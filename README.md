# ComfyUI-RH-Nodes

统一维护的 ComfyUI 图像节点集合。当前包含参考图颜色还原、漫画轮廓检测与 BBOX 掩码参考图；后续节点统一在此仓库接入、测试、发布。

## 安装

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/dakun333/ComfyUI-RH-Nodes.git
cd ComfyUI-RH-Nodes
python -m pip install -r requirements.txt
```

重启 ComfyUI 后，在节点搜索中输入下面的任一名称。ComfyUI 自带 PyTorch，因此依赖文件不重复安装 `torch`。

## 节点一览

| 节点 | 分类 | 用途 |
| --- | --- | --- |
| `Reference Color Restore (Occlusion Seam)` | `image/color correction` | 将 AI 编辑图的颜色还原到对齐的参考图；局部处理物体移除产生的遮挡接缝。 |
| `Reference Color Restore (Occlusion Seam Advanced) V0` | `image/color correction` | 保持原 Advanced 节点 ID 与 0805 的最近可信像素/Voronoi 接缝传播行为。 |
| `Reference Color Restore (Occlusion Seam Advanced) V0.8` | `image/color correction` | 使用 V1 的可信区域拟合与连续接缝场，但关闭结构化 unchanged 掩码清理，便于直接对比。 |
| `Reference Color Restore (Occlusion Seam Advanced) V1` | `image/color correction` | 完整的可信仿射颜色拟合、结构清理、连续接缝场与校正边缘热力图。 |
| `🎨 漫画轮廓检测 (Comic Outline)` | `🎨 漫画轮廓` | 纯 OpenCV 的漫画/动漫轮廓检测：亮度边缘、LAB 颜色边缘和前景外轮廓融合，无模型下载。 |
| `BBOX: Mask → 1M Reference Image` | `BBOX Tools` | 将原图 + 黑白掩码转为约 1M 像素、16 对齐的红框参考图和 BBOX 裁切。 |

## Reference Color Restore（遮挡接缝）

将 AI 编辑后的图像接到 `ai_image`，原始且构图对齐的图像接到 `reference_image`。默认节点会输出：

- `corrected_image`：颜色还原后的结果；
- `foreground_mask`、`safe_unchanged_mask`：分析掩码；
- `occlusion_core_mask`、`occlusion_zone_mask`、`occlusion_seam_mask`：局部遮挡接缝诊断掩码；
- `report`：每个批次的分析摘要。

两张图必须对齐。尺寸不同默认会报错；只有构图和宽高比仍然对齐时，才打开 `resize_reference`。高级节点中的 `occlusion_residual_blur` 和 `occlusion_bridge_width` 只影响保守检测出的遮挡区域，设为 `0` 可关闭对应局部处理。

### Advanced 版本选择

- **V0**：注册 ID 仍为 `CCROcclusionColorRestoreAdvanced`，用于保持既有 Advanced 工作流不变。
- **V0.8**：保留 V1 的连续加权残差场、热力图和可选 handoff 控制，仅关闭 `structural_unchanged_cleanup`。
- **V1**：默认开启可信区域仿射拟合和多尺度结构清理。比其他颜色还原节点多一个 `generated_correction_edge_heatmap` IMAGE 输出；它显示最终校正场的锐利边缘，红色表示每像素至少约 2 个 RGB 级别的校正梯度，是检查接缝的候选信号而非单独的缺陷判据。

V1 的 `component_internal_handoff` 与 `seam_handoff_smoothing` 默认关闭，便于直接比较连续场结果；只有需要研究独立 unchanged 岛或局部连续场交接时才开启。

## Comic Outline Detect（漫画轮廓）

输入 `图像`，输出黑线白底的 `轮廓图`。节点支持 IMAGE 批次，逐张处理并保持批次大小。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `line_width` | 3 | 轮廓线宽，范围 1–8。 |
| `edge_percentile` | 90 | LAB 颜色边缘的阈值分位数；提高可减少细碎线条，降低可保留更多细节。 |
| `alpha_thr` | 24 | 有透明通道时，超过此值的像素视为前景。 |
| `bg_threshold` | 18 | 无透明通道时，前景与边界背景的颜色距离阈值。 |
| `add_outer_contour` | 开启 | 叠加前景外轮廓。 |
| `min_edge_area` | 自动 | 最小连通边缘面积；`-1` 根据图像尺寸自动计算。 |
| `invert` | 关闭 | 输出白线黑底，便于特殊合成。 |

调参建议：线条碎或噪点多时，提高 `edge_percentile`（92–95）或增大 `min_edge_area`；细节不足时降低到 84–88；透明 PNG 出现白边时，尝试把 `alpha_thr` 降到 10–20。

## BBOX: Mask → 1M Reference Image（BBOX 掩码参考图）

输入 `image`（原图）和 `bbox_mask`（同尺寸黑白掩码 IMAGE），输出：

- `reference_image`：缩放到约 1M 像素（宽高均为 16 的倍数）的原图，BBOX 区域用外白内红框标注；框内像素与缩放后原图完全一致。
- `bbox_crop`：红框内部的精确裁切，宽高为 16 的倍数，可无损贴回 `reference_image` 对应位置。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `mask_threshold` | 0.5 | 掩码中 ≥ 此值的像素视为白色前景，围成 BBOX。 |
| `target_area` | 1048576 | 目标总像素面积（约 1M）。 |
| `allow_upscale` | 开启 | 允许小图放大到目标面积；关闭则只缩不放大。 |

注意：`bbox_mask` 必须接 **IMAGE** 输出（非 MASK 输出），且与 `image` 像素对齐、尺寸完全一致。节点支持 IMAGE 批次；若批次内各样本产生不同裁切尺寸会报错，需逐张处理。仅依赖 PyTorch，无额外第三方包。
## 维护与新增节点

接入后续节点时，请遵循 [docs/ADDING_NODES.md](docs/ADDING_NODES.md)；该流程也已安装为本机 Codex 技能 `comfyui-node-integration`。仓库会保留节点 ID，避免破坏已有工作流。

## 来源与许可

`Comic Outline Detect` 整合自 [dakun333/ComfyUI-ComicOutline](https://github.com/dakun333/ComfyUI-ComicOutline) 的已发布实现（核对提交 `baf6d0f`）。参考图颜色还原代码保留了原始 MIT 许可声明，详见 [LICENSE](LICENSE)。
