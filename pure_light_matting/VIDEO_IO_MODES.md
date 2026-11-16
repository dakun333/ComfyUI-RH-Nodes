# 纯光效去背原生 VIDEO 输入与互斥输出（2026-11-13）

适用节点：独立 `ComfyUI-PureLightMatting` 包的 `CorridorKey2LightEffectsMatting`（兼容历史节点ID）。节点ID不变，原有输出0–4位置不变，新增第5号（第六个）`video`输出。

## 输入

- `images`：原 IMAGE 帧批次，现在是可选端口，保留图片／已有视频帧工作流。
- `video`：新增 ComfyUI 原生 VIDEO，可直接连接内置“加载视频 / LoadVideo”。无需在输入端接 GetVideoComponents。
- **只连接一个**：两个都接或两个都不接会在解码/推理前明确报错，不偷偷选择其中一路。
- VIDEO输入自动采用源视频的帧率（保留Fraction，例如30000/1001）与音频/元数据，手动fps不覆盖源帧率。IMAGE→VIDEO时才使用手动fps（默认24）。
- 纯光效UnMult将输入Alpha相乘。原生VIDEO若分离RGB/Alpha，光效节点会正确合并且不会重复乘Alpha。

## `output_mode`：每次只生成一种

| 模式 | 有效输出 | 不生成的输出 |
|---|---|---|
| `images`（默认） | foreground_rgb、alpha、rgba、checker_preview、report | video |
| `video` | video、report | foreground_rgb、alpha、rgba、checker_preview |

两模式都只做**一次抠像**。video模式只创建一份float32 RGBA缓冲和原生VIDEO对象：不另外创建RGB批次、Mask批次、棋盘批次，也不调用棋盘合成，不存在AI模型推理。images模式不创建VIDEO、不调用视频编码。

未选输出返回ComfyUI的 `ExecutionBlocker(None)`：保留端口类型/位置，让旧图兼容；相连的下游保存节点会静默跳过，不输出黑图、不报无效视频。切换模式后，报告仍然可用。

注意“视频模式”是**输出类型**，不是自动落盘编码。节点创建VIDEO不编码文件；下游保存节点决定编码格式。抠像内部仍然需要帧：VIDEO输入会解码当前完整视频片段，算法仍需中间RGB/Alpha。它不是无内存的流式解码器。

输出张量（不含输入、临时数据、下游编码）：images约44字节/像素；video约16字节/像素，约减少64%。

## VIDEO携带的内容

- `get_components().images`：straight RGBA（保留源数值色域，通常为sRGB编码值），不是棋盘或黑底合成片。
- `get_components().alpha`：同一RGBA存储的Alpha视图，0透明、1不透明，不额外复制。
- 原帧率、原音频对象、元数据；VIDEO类型是原生VideoInput/VideoFromComponents兼容对象。
- GetVideoComponents在本机输出RGBA IMAGE，可继续做支持RGBA的保存/合成。此提取只是引用输出缓冲，不会再运行抠像。

本机自带VideoFromComponents.get_components会丢弃独立alpha/metadata，原SaveVideo也不处理RGBA，所以用了节点内小型子类保留这些信息。没有修改ComfyUI本体、全局类型、依赖版本或原算法。

## 透明视频保存：不要直接用MP4/H264

你这版ComfyUI的内置SaveVideo仅提供auto/MP4和auto/H264，无法保存透明通道。为避免静默丢Alpha，新VIDEO向该路径明确报错；不能把一个黑底MP4冒充透明视频。

本机已验证的纯内置接线：

`LoadVideo → 抠像节点.video（output_mode=video）→ GetVideoComponents.images → SaveWEBM(codec=vp9)`

将GetVideoComponents.fps接到SaveWEBM.fps。**必须选vp9**，本机SaveWEBM的av1分支不保存Alpha。示例已经接好。

内置SaveWEBM没有音频输入：示例保存的WebM是静音，虽然抠像VIDEO对象保留了音频。若需要同时保存音频，使用支持RGBA+音频的VIDEO保存节点，或调用对象的 `save_to("result.mov")` / `save_to("result.webm")`。本包适配器支持MOV/qtrle（RGBA无损，音频AAC）及WebM/vp9（透明，音频Opus）；这是下游显式保存才执行的功能，不是抠像模式额外自动编码。保存编码时float32会量化8bit，WebM YUV色彩可能有变化。

WebM常规FFmpeg VP9解码器可能忽略Alpha；验证时需强制libvpx-vp9解码。MOV/qtrle的透明度已做逐值量化后往返验证。不同播放器显示黑背景不意味着文件没有透明度。

## 直接加载的示例

- 光效 → images：`example_workflows/light_effects_native_video_images_workflow.json`
- 光效 → video：`example_workflows/light_effects_native_video_video_workflow.json`

对应API示例在api_examples目录。示例默认文件名是本机测试夹的 `fireworks.mp4`；加载后请选择你自己的素材，不会自动复制用户输入文件到Comfy目录。旧IMAGE示例也已更新新控件/端口，并默认images模式。

本包独立包含光效节点、算法、VIDEO适配器和示例；没有AI权重、worker或私有依赖，不导入其他自定义节点。用户当前8188服务不自动重启；文件修改后需重启ComfyUI或重新加载节点，旧画布如不出现新控件请删除并重新添加节点。
