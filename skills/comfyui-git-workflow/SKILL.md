---
name: comfyui-git-workflow
description: "在 dakun333/ComfyUI-RH-Nodes 中提交改动的固定流程：分支命名、提交信息、PR 记录、作者自查合并与责任边界。"
---

# 提交与协作流程

目标只有一个：让每一笔改动可追溯，让分支之间不互相踩踏。**流程不判定节点质量。**

## 责任边界

| 角色 | 负责 | 不负责 |
| --- | --- | --- |
| 提交者 | 自己代码的正确性、自测、文档与兼容性；合并自己的 PR | 别人的节点与工作流 |
| 流程（分支/PR/冒烟检查） | 改动可追溯、分支隔离、记录当时的检查结果 | 判定节点是否可用 |
| 维护者 | 仓库设置、依赖与许可决策、分支保护 | 逐个审代码 |

## 步骤

1. 同步并开分支。分支名用 `人名/类型-简述`，例如 `1hew/fix-unmult-alpha`：
   ```powershell
   git switch main
   git pull --ff-only
   git switch -c <人名>/<fix|feat|chore|docs>-<简述>
   ```
   一次改动开一个分支，用完即删；不要用一个常驻分支长期累积改动。
2. 一个分支只做一件事，并且从最新 `main` 拉出。分支存活越久，合并时冲突越大，也越容易在解冲突时覆盖掉别人已合入的改动。不得改动其他节点家族的 ID、输入名、输出名、默认值（技术规范见 `docs/ADDING_NODES.md`）。
3. 本地自测：`python -m pytest tests -q`。
4. 提交：`feat|fix|chore|docs: 一句话说明`，正文写影响范围与验证结果。
5. 推送：`git push -u origin <分支>`。**修改 `.github/workflows/` 时**需要 `workflow` scope 的令牌，或走 SSH 远端（`git push ssh://git@github.com/dakun333/ComfyUI-RH-Nodes.git <分支>`）。
6. 开 PR，按模板记录：改了什么、为什么、影响面、自查结果、回滚方式。不需要请求他人审核。
7. 自查后自行合并（使用 merge commit），删除远端和本地分支（`git branch -d <分支>`），回拉 `main`。

## 约定

- 改动走 PR，便于追溯；确需直接推送 `main` 时只推极小改动，并且不要对 `main` 强制推送
- 不在一个 PR 里混装无关改动
- 不在 PR 里声称没有实际运行过的验证
- 分支不留存：合并后删除远端与本地分支，`main` 始终是唯一长期存在的分支

## 说明

冒烟检查（`.github/workflows/tests.yml`）只提供记录，不作为合并门槛；它跑挂了不代表改动无效，也不阻止合并。节点能用不能用，由作者判断并负责。
