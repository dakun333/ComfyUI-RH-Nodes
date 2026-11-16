# CorridorKey2（RH-nodes 首次执行自动安装版）

CorridorKey **2.0** 节点 **纯色底去背**，分类 `RH-Nodes/matting`。支持IMAGE/原生VIDEO输入和互斥images/video输出。

**解压 → 重启ComfyUI → 第一次执行：自动安装私有依赖、自动下载缺失权重、继续推理。正常安装不需要手动Python命令。**
ComfyUI启动/节点扫描时不运行pip；依赖验证通过后后续执行跳过安装。

详细步骤见 **[INSTALL.zh-CN.md](INSTALL.zh-CN.md)**。

- 干净源码包：不带权重、python_deps、日志或缓存。模型资源约5.08GB，首次执行按manifest固定官方URL/大小/SHA256下载。
- 前提：Windows x64、ComfyUI自己的CPython3.12、NVIDIA CUDA、pip/Git和网络；不自动升级Python/torch，不承诺CPU/Mac/Python3.13+。
- 安全隔离：宿主只导入轻量桥；安装/模型在隔离worker中，私有依赖只写节点python_deps，不修改宿主NumPy/Transformers/torch。
- 初始化锁：自动/手动安装共用OS锁，锁内检查固定依赖版本/真实导入，防止并发重复安装；验证后原子切换目录。
- 进度：依赖安装/验证阶段写job状态，完整pip与推理日志在logs；环境/网络失败会明确报错，可修复后重试。
- 节点ID：CorridorKey2Matting；接口不变。纯光效去背仍是独立包，不在此处注册。
- 首次测试：导入example_workflows/install_test_images.json或install_test_video.json，并选择自己的素材。
- 可选：CORRIDORKEY2_PIP_INDEX_URL指定pip镜像；CORRIDORKEY2_AUTO_INSTALL_DEPS=0禁用自动pip。手动install.py仍保留作备用。

VIDEO/透明保存和内存说明见VIDEO_IO_MODES.md；API示例见api_examples。
普通H.264/MP4不能保存透明Alpha；附带video工作流保存VP9 WebM。
模型结构/数学公式未改；本次只加入首次执行依赖引导。

RH 接入保留源码快照（source_snapshot.json），仅调整分类、安装文档及脚本目录查找，根注册保持节点族失败隔离。贡献者确认已获本次集成和重新分发授权；原 Corridor Key Licence、上游及第三方署名完整保留，不改为 MIT。

许可原文见LICENSE/runtime/LICENSE，第三方声明见THIRD_PARTY_NOTICES.md。
上游固定代码提交：485f5e9ab759ead6138cc990776184b59d6ea971。
