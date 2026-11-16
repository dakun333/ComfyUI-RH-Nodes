# RH-nodes CorridorKey2 安装

安装 RH-nodes，重启 ComfyUI，搜索“纯色底去背”（`CorridorKey2Matting`），分类 `RH-Nodes/matting`。

第一次实际执行自动在此模块的 `python_deps/` 安装固定私有依赖，并按 `assets_manifest.json` 的官方固定 URL、大小和 SHA-256 下载约 5.08GB 模型到 `runtime/`，完成后继续推理。ComfyUI 启动扫描不安装依赖、不下载模型。此模块必须可写。

需要 Windows x64、ComfyUI 使用的 CPython 3.12、可用的 NVIDIA CUDA torch/torchvision、pip、Git 和网络；不会替换宿主 Python、torch、NumPy 或 Transformers。非此环境不能使用自动安装版，不影响其他 RH 节点族加载。

IMAGE 和 VIDEO 二选一；可选 hint_mask 为直接不透明度（1前景），接 Load Image MASK 时打开 invert_hint_mask。images/video 输出模式互斥；VIDEO 使用源帧率、音频和元数据。纯光效建议使用“纯光效去背”。

依赖和模型目录以当前 `corridorkey2/` 为根，不复用独立插件目录。日志在 `corridorkey2/logs/`；帧临时数据默认系统临时盘，可使用 temp_directory 指定空间充足的目录。

可选设置：`CORRIDORKEY2_AUTO_INSTALL_DEPS=0` 禁用自动依赖安装；`CORRIDORKEY2_PIP_INDEX_URL` 设置私有依赖镜像，权重仍从原发布者下载。

手动备用（使用启动 ComfyUI 的同一个 Python）：

```powershell
& "C:\path\to\ComfyUI\.venv\Scripts\python.exe" "C:\path\to\ComfyUI\custom_nodes\ComfyUI-RH-Nodes\corridorkey2\install.py"
```

`--check` 仅检查；`--download-models` 主动下载，配合 `--verify` 强制 SHA-256 校验。无需把 isolated_requirements.txt 安装到宿主环境。依赖安装、下载失败可修复后重试，完整日志在 logs。

透明 PNG 用 rgba → Save Image；透明 VIDEO 用 video → GetVideoComponents → SaveWEBM（vp9）。MP4/H264 不保留 Alpha。附带工作流不附带媒体，请在加载节点中选择自己的素材。详细说明见 VIDEO_IO_MODES.md。

CorridorKey 代码和模型仍使用 Corridor Key Licence 及其附加条款；不属于仓库 MIT 授权，见 LICENSE 和 THIRD_PARTY_NOTICES.md。
