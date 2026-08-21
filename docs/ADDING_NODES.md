# 接入新 ComfyUI 节点

本仓库是 `dakun333` 的统一节点集合。收到一个本地节点目录和“整合进入 ComfyUI 节点项目”的要求时，按下面的顺序执行；不要为了接入新节点改动或删除已有节点的 ID、输入名、输出名和默认行为。

## 1. 接入前检查

1. 阅读源项目的 README、安装说明、许可证与依赖；记录来源仓库和确认的提交。
2. 查看源项目的 Git 状态、远端和追踪分支。若是复制目录，先确认没有未提交的用户修改。
3. 检查源码中是否硬编码了本机路径、模型、密钥或网络地址。密钥绝不提交。
4. 把依赖和许可证与本仓库作兼容性核对；不确定的许可证或所有权必须先向用户确认。

## 2. 实现规则

1. 将一个节点族放在清晰的独立模块中，例如 `new_feature.py`；避免与 `nodes.py`、`algorithm.py` 的现有导入冲突。
2. 在根 `__init__.py` 中注册唯一的 `NODE_CLASS_MAPPINGS` 键和人类可读的 `NODE_DISPLAY_NAME_MAPPINGS` 名称。为每个节点族独立捕获导入失败，确保一个可选依赖故障不阻止其他节点加载。
3. 使用 ComfyUI 的 `IMAGE`（BHWC、0–1 float）和 `MASK` 约定。新 IMAGE 节点应明确批次行为；若只支持单图，先报出清晰错误。
4. 保持既有节点 ID、端口名、参数名、默认值、返回顺序不变，保障已有工作流可恢复。
5. 合并最小必要依赖到 `requirements.txt`。不要在其中重复声明 ComfyUI 自带的 `torch`，不要固定本机 Python 或模型路径。
6. 更新 `pyproject.toml` 的版本、说明、GitHub Repository 和 Comfy 元数据；仓库 URL 必须与实际仓库一致。

## 3. 文档与测试

1. 在 README 增加节点用途、节点搜索名、输入输出、参数范围、调参建议、安装方法和功能限制。
2. 为非原创整合代码保留来源和许可证信息。
3. 运行语法编译、根 `__init__.py` 动态导入、注册表检查，以及每个新节点最小输入的功能冒烟测试。涉及图像节点时至少测一张典型图；支持批次时同时测两张图。
4. 若依赖或 ComfyUI 版本限制无法本机验证，在 README 与提交说明中明确标记，不得伪称已验证。

## 4. 发布清单

1. 确认 `.gitignore` 排除 `__pycache__`、虚拟环境、构建物与 `.env`。
2. 检查 `git status --short` 和 `git diff --check`，确认没有密钥、缓存、绝对本机路径或无关文件。
3. 在 PowerShell 中执行 GitHub 操作；优先使用 HTTPS 和 `gh auth git-credential`。推送失败时先保留本地提交，再排查网络或凭据，禁止 force push。
4. 推送后核对远端默认分支、README、文件清单和最新提交 SHA；必要时创建带验证说明的 Release。

## 本仓库当前节点族

| 模块 | 注册 ID |
| --- | --- |
| `nodes.py` | `CCROcclusionColorRestore`、`CCROcclusionColorRestoreAdvanced`、`CCROcclusionColorRestoreAdvancedV08`、`CCROcclusionColorRestoreAdvancedV1` |
| `comic_outline.py` | `ComicOutlineDetect` |
| `bbox_mask_reference.py` | `BBoxMaskToReferenceImage`、`BBoxRestoreCropToCanvas` |
