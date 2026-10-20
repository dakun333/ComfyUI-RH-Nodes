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

1. 同步并开分支。分支名用 `<类型>/<简述>`（Conventional Branch 的 `<type>/<description>`），例如 `fix/unmult-alpha`：
   ```powershell
   git switch main
   git pull --ff-only
   git switch -c <feat|fix|chore|docs|refactor|test|perf|ci>/<简述>
   ```
   类型说明改动性质，简述用连字符小写英文；有 Issue 编号时写成 `fix/12-unmult-alpha`。**不要把人名放进分支名**：PR 列表和提交元数据已经标明作者，分支名应当回答「改什么」，而不是「谁改的」。
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

## 已知坑（操作层面）

1. **推送含 `.github/workflows/` 的提交**：HTTPS 令牌缺 `workflow` scope 时会被拒绝（`refusing to allow a Personal Access Token ... without workflow scope`）。改用 SSH 远端推送（`git push ssh://git@github.com/dakun333/ComfyUI-RH-Nodes.git <分支>`），或给令牌补上该 scope。
2. **用 REST 合并 PR 可能持续 404**：PR 刚创建、或冒烟刚跑完时，`PUT /repos/{owner}/{repo}/pulls/{n}/merge` 可能连续返回 404，但 PR 自身是 `mergeable: clean`。改走 GraphQL 的 `mergePullRequest` 通常一次成功，或直接在网页点合并。
3. **仓库没有分支保护**：`git push origin main` 会被接受，但这样没有 PR 记录，等于绕开追溯。推送前先 `git fetch --prune`，避免本地残留已删除的远端分支引用。
4. **目录是 junction**：真实仓库在 `D:\1hew\ComfyUI-RH-Nodes`，`custom_nodes\ComfyUI-RH-Nodes` 只是目录联接。改完代码要重启 ComfyUI 才生效，且只存在一份文件。

## 说明

冒烟检查（`.github/workflows/tests.yml`）只提供记录，不作为合并门槛；它跑挂了不代表改动无效，也不阻止合并。节点能用不能用，由作者判断并负责。
