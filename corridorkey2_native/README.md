# ComfyUI-CorridorKey2-Native

独立的宿主依赖版；节点名 **纯色底去背（宿主依赖版）**，ID `CorridorKey2Native`。
使用 ComfyUI 现有 Python 环境，不创建私有依赖目录。

## 安装和使用

1. 安装 RH-nodes；模块位于 `corridorkey2_native/`，分类 `RH-Nodes/matting`。额外依赖已合并到根 `requirements.txt`：
   **`timm`、`opencv-python`（OpenCV）**，不锁版本。
   使用支持安装节点 requirements 的管理器补齐缺失依赖；若手动安装，只需在
   **启动 ComfyUI 的同一个 Python 环境**中执行 `python -m pip install -r requirements.txt`。
   已满足要求的包不需要重装；不要加 `--upgrade`。
2. 将 **`corridorkey2_full.safetensors`** 放进 ComfyUI 的 `models/checkpoints`。
   本仓库和 wheel 不提供该权重文件，也不使用 Git LFS；请从自己的合法模型副本单独复制。
3. 重启 ComfyUI，添加本节点，在 `checkpoint` 下拉框选择该文件。
4. 连接 IMAGE 或 VIDEO（二选一），选择 `output_mode=images/video`。
   旧工作流省略 `checkpoint` 时仍默认使用该文件名。

模型文件放在 ComfyUI 注册的 checkpoints 目录中，可通过 `extra_model_paths.yaml`
使用其他模型目录；下拉框允许相对的子目录文件名，不接受任意绝对路径。
**此文件不是 SD/Flux checkpoint，普通 CheckpointLoader 不支持；只能由本抠像节点加载。**

## 单文件模型

`corridorkey2_full.safetensors` 包含：

- CorridorKey Stage 1、Stage 2；
- C-RADIO、完整 MoGe-2、运行时使用的完整 RVM（Recurrent Video MAE）；
- 三个特征模型的 mean/std，以及重建网络所需的配置。

共 **1720 个张量、4,938,667,848 字节（约 4.94 GB / 4.60 GiB）**。
保留原运行时的精度和值，没有 FP16 转换、量化、蒸馏或裁剪 MoGe。
RVM 原始 NPZ 中原推理代码不会加载的预训练辅助张量不写入文件；原始文件仍保留。
**checkpoint 只打包模型参数/配置，不打包 Python 库；完整网络代码仍随节点分发。**

文件存在时，不再读取 `runtime/weights`、`runtime/vfms` 或下载原始权重。
因此在另一台机器部署时，只需节点源码、所需依赖和这个 checkpoint，
不需要复制原来的多文件权重目录。本机这些旧目录暂时保留，不自动删除。

若默认 checkpoint 不存在，节点会沿用原有校验下载流程准备完整原始模型，
然后在 CPU 上生成并逐张量验证单文件 checkpoint；这是用户首次执行节点时触发的步骤，
不是导入节点就下载。首次准备还需要原权重和转换临时文件的额外磁盘空间/内存，
之后运行只加载单文件。这次验证的是已有原权重转换及独立运行，未测试全新环境的完整下载。

需要手动转换时，可在 ComfyUI 的 Python 中运行本目录的 `convert_checkpoint.py`。
默认输出到 `models/checkpoints`，已有合法文件会复用；`--output` 可指定目标，
`--overwrite` 才允许替换已有文件。

单文件主要方便搬运和管理，**不等于减少模型计算量、显存或大幅加速推理**。
模型/权重继续受 `LICENSE` 和 `THIRD_PARTY_NOTICES.md` 中各自许可约束，合并不改变许可。

## 依赖策略

- 节点执行和 `install.py` 只检查依赖，不运行 pip、不创建 `python_deps`。
  requirements 的安装由常规安装流程负责，而不是推理时偷偷改环境。
- 推理使用启动 ComfyUI 的同一个 Python 和其 site-packages，在一次性 worker 中执行。
- 不按上游 NumPy/Transformers 版本范围拒绝环境，先实际导入、加载和推理。
- 已有 `utils3d` 时直接尝试使用，不比较版本、不升级/降级、不替换。
  已有包导入/API 不兼容时报告原始错误。
- 只有找不到 `utils3d` 时使用 `vendor/utils3d` 完整源码，不覆盖宿主包。
- torch、torchvision、numpy、transformers、huggingface_hub、safetensors、einops、scipy
  复用宿主环境；缺失或不兼容时明确报错。

可选的只读检查（不是安装步骤）：

```powershell
& "C:\path\to\ComfyUI\.venv\Scripts\python.exe" -I "C:\path\to\ComfyUI\custom_nodes\ComfyUI-RH-Nodes\corridorkey2_native\worker.py" --check-deps
```

## Alpha 收缩与羽化

- `alpha_shrink`：向内收缩的像素半径，整数 0–64。使用椭圆形灰度腐蚀，
  保留半透明覆盖度，不把 Alpha 二值化。0 为关闭。
- `alpha_feather`：高斯羽化的 sigma，0–64，可用小数。数值越大边缘越柔和，0 为关闭。
- 按最终输出分辨率计算，不随 `model_scale` 改变像素单位。
  固定顺序为 **先收缩，再羽化**；只改 Alpha，不改模型预测的 straight RGB。
- 同时作用于 `alpha`、`rgba`、棋盘预览和 VIDEO 中的透明度。
  VIDEO 模式仍不额外生成图片批次或棋盘预览；不增加第二次模型推理。
- 默认 `0 / 0`，像素输出保持原样；旧工作流省略新参数时仍默认关闭。
  示例工作流已补齐两个新控件，report 中的 `alpha_postprocess` 记录实际参数。
- 可先试 `alpha_shrink=1`、`alpha_feather=0.5`；细碎木屑、头发、光效或运动模糊区域
  不宜过度收缩。羽化可能把覆盖度扩散到边缘外侧，不是前景颜色修复。

## VIDEO 与输出

两种模式只计算选中的输出，VIDEO 保留源帧率/音频。
RGBA 是预测的干净前景颜色加 Alpha，不是原图套 mask。
透明视频保存及示例见 `VIDEO_IO_MODES.md`；不要用不支持透明度的 H264 MP4 冒充透明视频。

## 本机验证（2026-11-26）

以下为源插件记录，RH接入验证另见仓库根README。本次只调整分类、注册、CLI默认目录查找及打包；模型计算保持源快照。公开集成和再分发授权已由贡献者确认，上游许可证及第三方声明完整保留。

- ComfyUI 0.45.0 / Python 3.12.10 / RTX 3060 12GB。
- 直接复用 NumPy 2.4.4、Transformers 5.11.0、timm 1.0.27、OpenCV 4.13.0。
- 47 项单元测试通过，包括 Alpha 默认零改动/灰度收缩/羽化/两种输出一致性、
  checkpoint 校验/路径边界/已有文件不访问原权重，
  以及示例工作流的节点 ID、checkpoint 与接线校验。
- 转换时验证全部 1720 个张量的值与 dtype。
- 没有原始 `weights` / `vfms` 目录的测试副本完成真实 GPU 推理和 ComfyUI 队列：
  VIDEO → images（4 帧 RGBA PNG），VIDEO → video（透明 VP9 WebM），未选输出分支跳过。
- 4 帧、320×320、`model_scale=0.5` 的 float32 RGBA 与转换前的**同一宿主版**逐像素完全一致。
  这不代表不同依赖环境间也逐像素一致。
- 本次未安装、升级或降级宿主依赖；已有 utils3d 优先使用策略保持不变。
- 这证明当前环境可运行，不保证任意库版本、全新下载流程或所有长视频都已验证。
