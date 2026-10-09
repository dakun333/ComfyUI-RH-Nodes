# ComfyUI-RH-Nodes

统一维护的 ComfyUI 图像节点集合。当前包含参考图颜色还原、原像素恢复、UnMult 抠图、鲁棒遮罩校色、漫画轮廓检测与 BBOX 参考/还原工具；后续节点统一在此仓库接入、测试、发布。

## 安装

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/dakun333/ComfyUI-RH-Nodes.git
cd ComfyUI-RH-Nodes
# 必须使用启动 ComfyUI 的同一个 Python；标准 venv 安装通常是：
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windows Portable 版通常使用 `..\..\python_embeded\python.exe -m pip install -r requirements.txt`。如果你的目录布局不同，请用实际启动 ComfyUI 的 Python 解释器执行安装，不要直接使用另一个系统 Python。

重启 ComfyUI 后，在节点搜索中输入下面的任一名称。ComfyUI 自带 PyTorch，因此依赖文件不重复安装 `torch`。

从原来的独立目录迁移时，请先移除或停用 `ComfyUI-Original-Pixel-Restore`、`ComfyUI-Robust-Masked-Color-Match` 和 `ComfyUI-Mekajiki-UnMult`，避免 ComfyUI 同时加载重复节点 ID。保留本仓库这一份即可。

## 节点一览

| 节点 | 分类 | 用途 |
| --- | --- | --- |
| `Reference Color Restore (Occlusion Seam)` | `image/color correction` | 将 AI 编辑图的颜色还原到对齐的参考图；局部处理物体移除产生的遮挡接缝。 |
| `Reference Color Restore (Occlusion Seam Advanced) V0` | `image/color correction` | 保持原 Advanced 节点 ID 与 0805 的最近可信像素/Voronoi 接缝传播行为。 |
| `Reference Color Restore (Occlusion Seam Advanced) V0.8` | `image/color correction` | 使用 V1 的可信区域拟合与连续接缝场，但关闭结构化 unchanged 掩码清理，便于直接对比。 |
| `Reference Color Restore (Occlusion Seam Advanced) V1` | `image/color correction` | 完整的可信仿射颜色拟合、结构清理、连续接缝场与校正边缘热力图。 |
| `🎨 漫画轮廓检测 (Comic Outline)` | `🎨 漫画轮廓` | 纯 OpenCV 的漫画/动漫轮廓检测：亮度边缘、LAB 颜色边缘和前景外轮廓融合，无模型下载。 |
| `BBOX: Mask → 1M Reference Image` | `BBOX Tools` | 将原图 + 黑白掩码转为约 1M 像素、16 对齐的红框参考图、裁切与元数据。 |
| `BBOX: Restore Crop to Canvas` | `BBOX Tools` | 将编辑后的 BBOX 裁切贴回画布，可选撤销 16 对齐拉伸。 |
| `Robust Masked Color Match / 鲁棒遮罩校色` | `image/color/robust_match` | 只用排除遮罩外的对应背景像素拟合鲁棒 RGB 仿射变换。 |
| `Core-Preserving Feather Mask / 保核外羽化` | `mask/robust_match` | 保持物体核心为 1，只在安全边距外侧羽化。 |
| `OPR · 原像素恢复 / Restore Pixels (B/A)` | `image/Original Pixel Restore` | 在原图网格上定位编辑并限制融合范围，遮罩外精确保留原图张量值。 |
| `OPR · 大物体删除遮罩 / Large Object Mask` | `image/Original Pixel Restore` | 分离大物体编辑候选、拟合排除、融合和复核遮罩。 |
| `OPR · 仅遮罩合成 / Restricted Composite` | `image/Original Pixel Restore` | 只按融合遮罩合成，不隐式配准或校色。 |
| `OPR · 加载图像＋ICC / Load Image` | `image/Original Pixel Restore` | 加载 8 位图像或 16 位 RGB/灰度 PNG，并单独传递 ICC。 |
| `OPR · 高精度保存＋ICC / Save Precision PNG` | `image/Original Pixel Restore` | 保存 16 位 RGB PNG 或仅在 support 内抖动的 8 位 PNG。 |
| `OPR · 旧版8位保存 / Legacy Save PNG` | `image/Original Pixel Restore` | 兼容旧工作流的 8 位就近舍入保存。 |
| `OPR · 九步诊断图 / Diagnostic Steps` | `image/Original Pixel Restore` | 将已捕获中间结果渲染为九步诊断图。 |
| `UnMult (Auto Background)` | `image/matting` | 从黑底或自动估计的近似纯色底图构造 straight-alpha RGBA。 |
| `Mask-Guided UnMult (Uniform Background)` | `image/matting` | 使用已知前景不透明度遮罩和统一底色近似，反算半透明区域 RGB。 |

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

## BBOX Tools（BBOX 掩码参考图 + 还原）

### BBOX: Mask → 1M Reference Image

输入 `image`（原图）和 `bbox_mask`（同尺寸黑白掩码 IMAGE），输出 5 个端口：

- `reference_image`（IMAGE）：缩放到约 1M 像素、宽高为 16 倍数的红框参考图。
- `bbox_crop`（IMAGE）：红框内部精确裁切，可无损贴回。
- `crop_info`（BBOX_CROP_INFO）：裁切坐标元数据，接给还原节点。
- `crop_info_json`（STRING）：同一元数据的可读 JSON 文本。
- `stretch_info`（BBOX_STRETCH_INFO）：缩放/对齐元数据，可选接给还原节点。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `mask_threshold` | 0.5 | 掩码中 ≥ 此值的像素视为白色前景，围成 BBOX。 |
| `target_area` | 1048576 | 目标总像素面积（约 1M）。 |
| `allow_upscale` | 开启 | 允许小图放大到目标面积。 |
| `alignment_mode` | `stretch` | `stretch`：直接拉伸到 16 倍数；`pad_gray`：保持宽高比缩到 logical 尺寸后在右/下补 #808080 灰至 16 倍数，不拉伸真实内容。 |

### BBOX: Restore Crop to Canvas

将编辑后的 `bbox_crop` 贴回画布，并可选撤销 16 对齐拉伸。

| 输入 | 类型 | 说明 |
| --- | --- | --- |
| `crop` | IMAGE | 编辑后的裁切图，像素原样贴回不缩放。 |
| `crop_info` | BBOX_CROP_INFO | 来自参考图节点。 |
| `background` | 下拉 | 画布背景：`alpha`/`black`/`white`/`red`/`green`/`blue`/`yellow`/`custom`。 |
| `custom_color` | STRING | `background=custom` 时的十六进制颜色，如 `#FF8800`。 |
| `stretch_info`（可选） | BBOX_STRETCH_INFO | 接入后撤销 16 对齐：`stretch` 模式反拉伸至 logical 尺寸；`pad_gray` 模式仅裁去灰边，零重采样。 |

输出：`restored_image`（IMAGE）、`alpha_mask`（MASK）、`working_image_16`（IMAGE）、`working_alpha_mask`（MASK）。

注意：`bbox_mask` 必须接 **IMAGE** 输出（非 MASK），且与 `image` 尺寸完全一致。节点支持 IMAGE 批次；若批次内裁切尺寸不同会报错。缩放使用 Pillow Lanczos。

## Robust Masked Color Match（鲁棒遮罩校色）

`reference` 接希望保持颜色的原图，`target` 接尚未合成回原背景的 AI 编辑图，`exclude_mask` 中白色表示不参与颜色拟合的编辑区域。节点将同一个校色变换应用到完整 `target`，然后由下游节点决定合成范围。

- `reference` 和 `target` 必须尺寸相同、内容位置对应；节点不自动配准图像。
- `mask_resize=nearest` 只处理遮罩尺寸差异，不能修复平移、裁切或构图差异。
- `exclude_expand` 默认 32；若遮罩已经由 OPR Large Object Mask 外扩，设为 0。
- `fit_region_not_blend_mask` 是拟合区域诊断图，不是最终合成遮罩。
- 支持 `target` 批次；`reference` 和 `exclude_mask` 可使用单帧广播或与其批次相同。
- JPEG 等不含透明通道的黑白遮罩，应把 `Load Image.IMAGE` 转为 `MASK`；不要误用其通常来自 alpha 的 `Load Image.MASK` 输出。

`Core-Preserving Feather Mask` 使用 `core_expand` 保持物体与安全边距严格为 1，并在 `feather_width` 范围内向外平滑过渡到 0。全局 RGB 变换不能修复几何错位、局部光照变化、阴影或 AI 重画纹理。

## Original Pixel Restore（原像素恢复）

### 细小编辑与水印

`OPR · 原像素恢复` 将 `original` 和 `edited` 对齐到原图网格，检测候选编辑区域，并只在 `support_mask` 内写入结果。`edited` 可以单帧广播；输出批次跟随 `original`。

- `method=B`：全局配准、受限局部光流、鲁棒校色和图割；需要 PyMaxflow。
- `method=A`：只使用全局配准的基线方法，不执行局部光流和图割。
- `high` / `low`：提高阈值更保守，降低阈值可覆盖较弱细节但会增加误选；必须满足 `0 < low <= high`。
- `padding` / `feather`：控制核心外扩及过渡；严格遵守人工 `edit_mask` 时均设为 0。
- `capture_steps=false`：大图或批次处理时可减少中间数组占用。
- 自动差异遮罩不是语义分割真值。AI 意外重画、阴影和配准误差仍需检查。

### 大物体路径

大面积不透明物体推荐拆分处理：

```text
original ─► Robust.reference ─────────────────────┐
                                                  ├─ OPR · 仅遮罩合成 ─► restored
aligned_edited ─► Robust.target ─► corrected ─────┤
blend_mask ───────────────────────────────────────┘

exclude_mask ─► Robust Masked Color Match.exclude_mask
```

将 `OPR · 大物体删除遮罩` 的 `aligned_edited` 接到 Robust 节点的 `target`，`exclude_mask` 接到同名输入，并把 Robust 的 `exclude_expand` 设为 0。最后只用 `blend_mask` 决定覆盖范围。`include_mask` 可强制纳入区域，`protect_mask` 可强制保留原图且优先级最高。

### ICC 与保存

`OPR · 加载图像＋ICC` 不进行色彩空间转换，只保留解码样本并单独传递 ICC。不要在一个批次中混用不同色彩空间。它接受 SDR 8 位 RGB/灰度图及 16 位 RGB/灰度 PNG，拒绝透明、动画/多页、CMYK、HDR 和高位深 TIFF。`OPR · 高精度保存＋ICC` 支持 `16bit`、`8bit_dither` 和 `8bit_round`；8 位输入不会因此恢复已经丢失的精度。

保存节点默认遵循 ComfyUI 行为，将 prompt、workflow 和诊断信息写入 PNG 元数据。对外分享图片前，如不希望暴露工作流或提示词，请使用 ComfyUI 的禁用元数据选项。

OPR 为 CPU 管线。全分辨率光流、图割、Poisson 和诊断缓存可能占用较多 RAM，单张图内部的长耗时 OpenCV 或求解器调用不会立即响应中断。

## UnMult（纯色底反算透明）

这两个节点输出四通道 straight-alpha `IMAGE` 和单独的 `MASK`。内部 ID 仍为 `MekajikiUnMult` 与 `MekajikiMaskedUnMult`，用于兼容已有工作流；界面使用通用名称。本实现是非官方兼容实现，与同名第三方产品或其权利人不存在隶属、授权或背书关系。

### UnMult (Auto Background)

输入一张 SDR `IMAGE`，支持批次及 1/2/3/4 通道。`auto_background=true` 时，每个批次项目分别从图像边缘聚类估算一个近似纯色底；关闭时使用经典黑底 UnMult 规则。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `black_threshold` | 0 | Alpha 连续黑点，范围 0–255；255 会得到全透明结果。 |
| `dither` | 0 | RGB 固定可复现抖动，范围 0–1；不改变 Alpha。 |
| `auto_background` | 开启 | 估计纯色背景；渐变、复杂边缘或主体贴边时建议关闭或改用遮罩引导节点。 |

单张展平图不能唯一确定原始前景 RGB 与 Alpha。节点构造的是满足纯色底假设的实用表示，不保证逐像素恢复原透明素材。近黑底会抑制向下噪声提供 Alpha 证据；这也可能遗漏比底色更暗的真实前景。计算直接发生在 ComfyUI 张量数值空间，最好与原合成使用相同工作色彩空间。

### Mask-Guided UnMult (Uniform Background)

`image` 是展平图，`mask` 必须表示**前景不透明度**：`0=透明，1=不透明`。节点从可信透明区域和遮罩外近邻环带估计一个统一背景颜色，并按 `F=(C-(1-A)B)/A` 构造 RGB。遮罩会在需要时双线性缩放，单帧图像或遮罩均可向另一方批次广播。

当前默认采用方法一（2026-09-30）：RGB 溢出时对所有非负通道使用同一倍率缩小，避免逐通道裁切改变通道比例；严重负值不相容时回退输入 RGB。遮罩引导节点对近黑候选底色采用至少占 20% 样本权重的最暗联合 RGB 色簇，减少残留颜色污染。

`allow_alpha_adjustment` 为可选布尔输入，默认关闭；旧 API 请求省略它仍按关闭执行。开启时采用 gray17（2026-10-09）路线，仅对必要的正溢出像素增加局部 Alpha，上限为 `43/255`（约 16.86 个绝对百分点，并非乘以 1.17）。输出仍为普通有界 straight RGBA，`alpha` 与第四通道一致；合成时应使用输出 Alpha，而非重新挂原 Mask。该方法以编码 RGB `#383838` 为匹配锚点，不能保证所有底色无损一致，白底可能变深，底色噪声或弱渐变也可能被放大。关闭可回到默认方法一。

- ComfyUI 内置 `Load Image.MASK` 对带 Alpha 的 PNG 通常输出反相 Alpha，直接连接本节点前应先反相。本节点输出的 `alpha` 是直接不透明度，可直接连接另一个 Mask-Guided UnMult；接 `Join Image with Alpha`、Porter-Duff 等按反相 Alpha 解释 MASK 的节点前通常也要先反相。四通道 `transparent_rgba` 可直接交给支持 RGBA 的保存节点。
- 对渐变、纹理或遮罩两侧不同背景，只能消除统一底色的近似分量。
- 完全没有可信透明/近透明区域时，底色不可观测；节点安全保留 `Alpha>0` 内的输入 RGB，只附加遮罩。
- 遮罩轮廓、强度或配准错误会留下底色残影，节点不会自行修改遮罩语义。
- 四通道 `IMAGE` 并非所有 ComfyUI 节点都接受。接 RGB-only 节点时，请先拆分/移除 Alpha，并使用单独的 `alpha` 端口参与合成。

## 验证

使用 ComfyUI 的 Python 并在仓库目录运行：

```powershell
& "C:\path\to\ComfyUI\.venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

当前集成测试使用合成图像和最小 ComfyUI API 桩，不依赖个人素材或固定本机输出目录。测试覆盖根注册、单帧/批次、方法 B、UnMult 黑底/纯色底/遮罩反算、鲁棒校色、遮罩羽化、大物体遮罩、受限合成及 16 位保存核心路径。实际加载验证仍应在 ComfyUI 中执行。

真实 ComfyUI 导入及加载/恢复/16 位保存冒烟测试会使用系统临时目录，不启动服务或写入用户输出目录：

```powershell
& "C:\path\to\ComfyUI\.venv\Scripts\python.exe" tests\run_comfyui_integration.py --comfyui "C:\path\to\ComfyUI"
```

## 维护与新增节点

接入后续节点时，请遵循 [docs/ADDING_NODES.md](docs/ADDING_NODES.md)；该流程也已安装为本机 Codex 技能 `comfyui-node-integration`。仓库会保留节点 ID，避免破坏已有工作流。

## 来源与许可

`Comic Outline Detect` 整合自 [dakun333/ComfyUI-ComicOutline](https://github.com/dakun333/ComfyUI-ComicOutline) 的已发布实现（核对提交 `baf6d0f`）。Original Pixel Restore 与 Robust Masked Color Match 由本次提交者确认为原创代码，并授权并入本仓库按 MIT 许可证发布；首次公开版本以加入本仓库的提交为来源快照。项目许可详见 [LICENSE](LICENSE)。

UnMult Python 实现由本次提交者确认为独立原创，仅参考公开数学行为与公式，不包含从 After Effects AEX 提取、反编译或复制的实现代码，并授权并入本仓库按 MIT 许可证发布。首次公开版本以加入本仓库的提交为来源快照。历史兼容节点 ID 中的名称仅用于已有工作流识别。

`method=B` 依赖单独安装的 GPL 软件包 PyMaxflow 1.3.2；仓库不包含其源码或二进制文件。依赖许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
