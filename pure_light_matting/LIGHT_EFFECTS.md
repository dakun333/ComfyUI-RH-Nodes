# 纯光效视频去背节点（AE UnMult 衍生）

节点包：`ComfyUI-PureLightMatting`。类别：`video/matting`。
显示名：**纯光效去背**。
节点 ID：`CorridorKey2LightEffectsMatting`。

这是用户本地 Mekajiki UnMult / Auto Background 的 **Auto Q 无 Mask 光效分支**，不是 CorridorKey 深度模型，不加载任何权重。只复制必要的 PyTorch 数学核心，运行不依赖原工作目录、AE 或 .aex 二进制。不导入、不依赖、也不调用 `ComfyUI-CorridorKey2`。保留历史节点ID只用于旧工作流兼容。

## 适用范围

- **纯光效**：辉光、火花、烟花、能量光束、无实体的发光特效；推荐空间近似均匀的纯黑/近黑底。
- 自动底色也可以处理均匀白色/纯色底的光效，但单一背景下的 RGB 与 Alpha 无法唯一分离；白底补偿仍是统计估计，不是原始透明素材的保证恢复。
- **不用于**人物、木块、实心物体、普通烟雾、包含暗色实体的火焰实拍或复杂场景。它没有语义识别，黑底下会把黑色实体和背景一起移除。实体抠像请用原 CorridorKey 节点/合适的模型。
- 全画面光效、边缘被光效遮满、渐变/变化背景会令自动底色误估；不能用关闭自动估底来把任意非黑底当黑底。

## 输入和视频接线

新增原生 `video` 输入及 `output_mode=images/video` 互斥输出，详见 `VIDEO_IO_MODES.md`。可直接连接“加载视频”的VIDEO输出；images/video只接一个。下文是IMAGE路线说明。

输入 `images` 是 ComfyUI 的有序 IMAGE 帧批次 `[N,H,W,C]`，支持1/2/3/4通道；普通视频是RGB。不需要 Mask。已有RGBA的Alpha按原算法相乘；ComfyUI LoadImage的独立MASK不会被自动接入。

`LoadVideo → GetVideoComponents.images →（可选 ImageFromBatch）→ 本节点.images`

也可连接现有视频加载节点的 IMAGE 输出。不改尺寸、不改帧顺序；只做逐帧计算，没有光流、时间模型、跨执行状态或自动镜头检测。**帧率/音频不是光效数学算法输入**，需要从原视频组件另接到保存/组装视频节点。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `auto_background` | True | 每帧从四边约2%区域做联合RGB聚类估计单一底色，再反算RGB。暗候选要求20%支持及多边分布；近黑底只用正差估Alpha。纯黑源可关闭，固定B=0并避免估计跳变。 |
| `black_threshold` | 0 | 0–255的Alpha黑点，连续Levels映射，非二值裁切。255全透明；提高会损失弱光。 |
| `dither` | 0.0 | 0–1固定三角RGB抖动，最大约1/255，用于后续8bit保存；不改Alpha、不做去噪/去块。 |
| `processing_device` | auto | 使用ComfyUI当前设备；cpu不占GPU。无模型加载/卸载，不改变宿主线程数或私有依赖。 |

保留来源节点的float32与像素安全行为：NaN→0，+Inf→1，-Inf→0，输入超出0–1会裁切；不是HDR输出节点。

### 输出

1. `foreground_rgb` IMAGE：反算后的 **straight/未预乘 RGB**，不是直接输出带底色的原图。
2. `alpha` MASK：0透明，1不透明。不是LoadImage那种反向MASK。
3. `rgba` IMAGE：straight四通道，可连接SaveImage保存RGBA PNG。
4. `checker_preview` IMAGE：按本算法的数值色域合成的棋盘预览，本身不透明。
5. `report` STRING：算法/范围、设备、帧数、耗时、参数及Alpha统计JSON。

### 色彩与不可逆信息

沿用原Auto Q算法的**输入数值色域**（ComfyUI通常为sRGB编码值），不自动转线性光：

`C = A*F + (1-A)*B`

关闭自动估底时B=0，`A=max(R,G,B)`，`F=RGB/A`（零Alpha处清零）。因此弱光会转为更亮的straight RGB与更低的Alpha，源数值色域下合成回黑底可还原观测颜色（阈值/抖动关闭）。不能把straight RGB误当预乘RGB再直接叠加。

Auto Q处理负RGB时只裁掉负通道，正通道溢出时统一缩放；**没有整像素退回带背景原RGB的硬回退**。近黑保护和色域裁切使含噪背景/负通道的重新合成不保证逐像素等于原图。黑点、抖动、白底估计也会改变结果。

本节点棋盘预览明确使用同一数值色域，**不是原CorridorKey节点的线性光棋盘合成**。若下游在linear-light做alpha-over，弱光外观可能不同；这不是对丢失的物理透明度的唯一恢复。

### 内存与透明视频

逐帧计算限制临时设备占用，但四组输出仍需整个批次的CPU内存，约44字节/像素，不含输入及临时数据。100帧1920×1080仅输出约8.5GiB；长视频建议分批16–32帧。每帧独立，所以相同帧在不同分批方式下算法结果一致；自动底色仍可能随帧变化而闪动。

普通H.264 MP4没有Alpha。`rgba → SaveImage` 得到透明PNG序列；透明视频需选择确实支持Alpha的编码/保存节点。示例中的MP4仅用于**棋盘预览**，不是透明视频。来源帧率可以另接到CreateVideo.fps；视频片段截取时音频需自行同步。

## 示例与验证

- `example_workflows/light_effects_image_workflow.json`
- `example_workflows/light_effects_video_workflow.json`（默认只取16帧）
- `api_examples/light_effects_image_api_prompt.json`
- `api_examples/light_effects_video_api_prompt.json`

示例仅使用本机ComfyUI内置节点，无VHS依赖。加载后选自己的光效素材。不要把示例图片用于普通实体。

测试：

`python -m unittest discover -s tests -p "test*.py" -v`

来源与数学版本见 `LIGHT_EFFECTS_SOURCE.md`；本机实测见 `TEST_RESULTS.md`。安装后需要重启ComfyUI才能在现有进程里加载新注册；集成过程中不主动重启用户服务。


原生VIDEO→VIDEO模式只保留一份RGBA及源fps/audio；原生VIDEO→images模式输出上述4组帧。VIDEO输入的元信息自动传递；手动fps仅IMAGE→VIDEO生效，视频保存格式/透明度限制见 `VIDEO_IO_MODES.md`。
