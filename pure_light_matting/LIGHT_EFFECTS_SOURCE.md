# Pure Light Matting 来源

本模块取自贡献者的 `ComfyUI-PureLightMatting` 2026-11-16 文件快照；原文件 SHA-256 见 `source_snapshot.json`。

光效核心为原创 Auto Q / UnMult 数学实现，保留 2026-10-15 算法及 2026-10-27 快照行为。采用 `LightEffectsUnmultKernel`，包含图像拆分、底色估计、联合色域缩放、确定性抖动和 Alpha 计算；此次接入不修改数学核心。

贡献者已确认其为有权公开发布的原创代码，并授权按 RH-nodes 的 MIT 许可证发布。未包含 AE 二进制、AE 宿主接口、CorridorKey 软件或神经模型权重。历史 ID `CorridorKey2LightEffectsMatting` 仅保留工作流兼容，节点显示名为“纯光效去背”。

此次接入更改分类为 `RH-Nodes/matting`，增加根注册、打包、文档和测试。
