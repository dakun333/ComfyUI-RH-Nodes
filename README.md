# ComfyUI-RH-Nodes

统一维护的 ComfyUI 图像节点集合。当前包含参考图颜色还原与漫画轮廓检测；后续节点统一在此仓库接入、测试、发布。

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
| `Reference Color Restore (Occlusion Seam Advanced)` | `image/color correction` | 暴露完整的颜色拟合、结构保护、接缝与遮挡区参数。 |
| `🎨 漫画轮廓检测 (Comic Outline)` | `🎨 漫画轮廓` | 纯 OpenCV 的漫画/动漫轮廓检测：亮度边缘、LAB 颜色边缘和前景外轮廓融合，无模型下载。 |

## Reference Color Restore（遮挡接缝）

将 AI 编辑后的图像接到 `ai_image`，原始且构图对齐的图像接到 `reference_image`。默认节点会输出：

- `corrected_image`：颜色还原后的结果；
- `foreground_mask`、`safe_unchanged_mask`：分析掩码；
- `occlusion_core_mask`、`occlusion_zone_mask`、`occlusion_seam_mask`：局部遮挡接缝诊断掩码；
- `report`：每个批次的分析摘要。

两张图必须对齐。尺寸不同默认会报错；只有构图和宽高比仍然对齐时，才打开 `resize_reference`。高级节点中的 `occlusion_residual_blur` 和 `occlusion_bridge_width` 只影响保守检测出的遮挡区域，设为 `0` 可关闭对应局部处理。

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

## 维护与新增节点

接入后续节点时，请遵循 [docs/ADDING_NODES.md](docs/ADDING_NODES.md)；该流程也已安装为本机 Codex 技能 `comfyui-node-integration`。仓库会保留节点 ID，避免破坏已有工作流。

## 来源与许可

`Comic Outline Detect` 整合自 [dakun333/ComfyUI-ComicOutline](https://github.com/dakun333/ComfyUI-ComicOutline) 的已发布实现（核对提交 `baf6d0f`）。参考图颜色还原代码保留了原始 MIT 许可声明，详见 [LICENSE](LICENSE)。
