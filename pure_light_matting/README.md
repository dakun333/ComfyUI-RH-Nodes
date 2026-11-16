# ComfyUI-PureLightMatting

**纯光效去背**：独立、轻量的 AE UnMult 衍生 Auto Q 节点，支持图片和 ComfyUI 原生 VIDEO。

- 无模型权重、无需下载、无需Mask或AE。
- 不包含/依赖 CorridorKey、私有 Python 依赖或独立worker。
- 使用已有 ComfyUI 的 torch/numpy，CPU/GPU均可运行，不改变全局依赖。
- 节点显示名：**纯光效去背**；类别：`RH-Nodes/matting`。
- 内部ID继续使用 `CorridorKey2LightEffectsMatting`，仅为旧工作流兼容，与原AI节点包无运行依赖。

## 安装

安装 RH-nodes 后本模块从 `pure_light_matting/` 注册，重启 ComfyUI即可。不需要下载权重或执行私有依赖安装脚本。VIDEO功能要求支持原生VIDEO API的ComfyUI；源包记录的验证环境为ComfyUI0.24.0。

本次从 `ComfyUI-CorridorKey2` 移出光效注册，因此两包同时安装不会重复注册。不要另装包含同一历史节点ID的旧版本包。

## 用途

只用于**纯光效**：烟花、火花、辉光、能量光束等，优先均匀黑底/近黑底。不是人物、木块、实心物体或普通烟雾的语义抠像。自动估底需要边缘能看到足够的均匀背景；单一背景无法唯一恢复真实Alpha，白底补偿是估计。

## 输入和输出

`images`（IMAGE帧批次）或 `video`（原生VIDEO）**二选一**。可直接接“加载视频”，不必先拆帧。VIDEO自动采用源帧率/音频；IMAGE→VIDEO才使用fps控件。

| output_mode | 有效输出 | 不生成 |
|---|---|---|
| images（默认） | foreground_rgb / alpha / rgba / checker_preview / report | video |
| video | video / report | 整批RGB / Mask / RGBA图片 / 棋盘副本 |

video模式只保留一份RGBA视频缓冲，未选分支的下游静默跳过；不会额外生成棋盘或再计算一遍算法。VIDEO创建时不编码落盘。保留源帧率、音频、元数据及Alpha；Alpha是RGBA存储的视图。

RGB是**straight/未预乘**，不是带黑底原图直接套Mask。算法和棋盘合成保留源数值色域（通常为sRGB编码值），不自动改线性光。Alpha为0透明/1不透明。完整数学与参数见 `LIGHT_EFFECTS.md`。

参数：auto_background默认开启；纯黑源可关闭固定B=0。black_threshold默认0；提高会损失弱光。dither默认0。processing_device=auto/cpu。

## 保存透明视频

images模式：rgba接SaveImage可保存RGBA PNG。

video模式：`本节点.video → GetVideoComponents.images → SaveWEBM(codec=vp9)`，fps同时连接。**必须选vp9**；本机内置SaveWEBM没有音频输入，该保存示例输出静音，但VIDEO对象保留了音频。

不要直接用MP4/H264保存透明结果；本机内置SaveVideo只提供该路线，适配器会明确拒绝丢Alpha。支持Alpha的VIDEO保存器可保存对象的MOV/qtrle或WebM/vp9及音频。详见 `VIDEO_IO_MODES.md`。

## 示例

- `example_workflows/light_effects_image_workflow.json`
- `example_workflows/light_effects_video_workflow.json`（拆帧、取前16帧的旧兼容示例）
- `example_workflows/light_effects_native_video_images_workflow.json`（直接VIDEO输入，输出图片）
- `example_workflows/light_effects_native_video_video_workflow.json`（直接VIDEO输入，输出透明视频）

对应API请求在api_examples。加载工作流后选择自己的纯光效素材。新包可单独传Git；没有权重/依赖二进制要上传。长视频仍需解码帧与输出内存：images约44字节/像素，video约16字节/像素，不含输入/临时数据。

## 测试和来源

在 RH-nodes 根目录运行：`python -m unittest discover -s tests -p "test_*.py" -v`。原始源码快照哈希见 `source_snapshot.json`。

来源见 `LIGHT_EFFECTS_SOURCE.md` 和 `THIRD_PARTY_NOTICES.md`。贡献者已确认原创及公开发布权，授权按本仓库 MIT 发布；未随包包含AE二进制或CorridorKey代码。
