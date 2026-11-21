<div align="center">

# 🚀 CodeSwarm-Agent

### *让多个 Agent 在同一个代码仓库里分工，交付经得起检查的改动*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/编排-LangGraph-167b70)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/license/mit)

</div>

---

## 📖 项目背景

拿到一项开发任务时，你可能说得清想改什么，却还叫不出相关函数的名字。CodeSwarm-Agent 可以从这段描述出发，找到相关代码，安排调查和修改，再把选中的补丁和验证结果交给你。

你也可以沿着执行轨迹回看 Manager 为什么分派任务，Worker 看过哪些代码，哪次测试影响了最终选择。任务中断后，可以从已保存的进度继续。

---

## 🎯 它能帮你做什么

- **让独立调查同时推进。** 简单任务由 Manager 直接处理，可拆分的调查交给 Worker 并行执行。
- **在需要时找回原文。** 通过代码检索缩小范围，把长日志和旧上下文原文保存在可回读的 Artifact 中。
- **拿到可以复查的改动。** 用补丁、测试和必要的独立审查支持任务完成判定。

---

## 💡 解决方法

| 模块 | 职责 | 核心技术 |
| --- | --- | --- |
| 🎛️ Manager V2 | 验收规划、派发与重新路由 | LangGraph、规则约束与模型决策 |
| 🔍 仓库检索 | 定位与任务相关的代码 | 路径与符号、BM25、AST 关系、可选 Dense、RRF |
| 🗜️ 上下文管理 | 控制窗口增长、保留原始证据 | 临时召回清理、阈值摘要、Artifact 按需回读 |
| 🧪 执行与验收 | 隔离修改、集成补丁与测试 | Git worktree、执行后端、按风险触发独立审查 |
| 🔭 恢复与观测 | 保存进度并追踪模型和工具调用 | SQLite Checkpoint、本地 JSONL、可选 LangSmith |

### 执行流程

```text
任务描述 + 目标 Git 仓库
    ↓
Manager V2 制定验收计划，选择直接执行或分派 Worker
    ↓
按需检索代码 → 在隔离工作区调查与修改 → 保存证据 Artifact
    ↓
集成候选补丁 → 执行验证 → 按风险触发独立审查
    ↓
返回选中的补丁 Artifact ID 与验证结果
```

---

## ✨ 项目亮点

### 🧩 1. 根据证据调整 Manager–Worker 协作

Manager 在发现新证据或执行遇阻时重新判断下一步。独立的只读调查可以并行，对已有代码的修改按顺序集成。LangGraph 保存任务状态和已确认的 Worker 结果，可从运行目录继续任务。

### 🔎 2. 知道要改什么，就能开始找代码

给出文件名或函数名时，Agent 优先查找对应位置。如果手头只有任务描述，它会从关键词和代码结构入手，也可以启用 Dense 检索补充语义线索。检索器校验候选片段是否仍与当前工作区一致，再用 RRF 合并各路排名，让有限的上下文优先留给相关代码。

### 📦 3. 上下文可以精简，关键证据随时回查

长工具输出的正文保存为 Artifact，活跃上下文保留有限预览与引用。已消费的临时召回正文可从窗口清理。达到摘要阈值时，先归档本次摘要实际使用的旧交互内容及已有摘要，再调用摘要器。需要核对细节时，可按 Artifact ID 回读原文。

### ✅ 4. 看得到改了什么，也查得到怎样验收

Agent 在独立 Git worktree 中修改代码，先集成候选补丁，再运行配置的验收测试。风险规则可要求独立审查。输出中的 `selected_patch_artifact_id` 指向选中的补丁。

### 🔭 5. 顺着调用轨迹看懂 Agent 的决策

本地 JSONL 记录路由、模型调用、工具调用、检索和摘要事件。开启 LangSmith 后，可在网页中查看模型和工具调用的父子关系、耗时与可用的 Token 统计。

---

## 🚀 快速开始

### 环境准备

需要 Python 3.12、Git 和 uv。以下是源码就位后在项目根目录运行的 Windows PowerShell 示例。

```powershell
uv sync --locked --no-dev --extra test --extra observability --extra windows-guarded
Copy-Item trae_config.yaml.example trae_config.yaml
Copy-Item .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY`，在 `trae_config.yaml` 中选择模型。若要启用 LangSmith，在 `.env` 中设置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY`。

### 执行仓库任务

目标目录需要是已有初始提交的 Git 仓库。下例使用 `--trust-repository`，请仅对信任的仓库使用此选项。

```powershell
uv run --locked --no-sync codeswarm-cli "Fix the reported regression" `
  --working-dir D:/path/to/repository `
  --execution-backend windows-guarded --trust-repository `
  --test-command "python -m pytest -q"
```

- **恢复任务**。使用 `--resume-run <运行目录>`，保持相同的执行后端与模型配置。
- **容器执行**。使用 `--execution-backend docker --sandbox-image <镜像仓库>@sha256:<摘要>`。构建配方位于 `docker/Dockerfile`，需传入按摘要固定版本的 Python 3.12 Linux 基础镜像 `BASE_IMAGE`。

### 运行测试

基础测试使用临时 Git 仓库、预设模型响应与模拟网络端点，无需配置模型 API Key。

```powershell
$env:LANGSMITH_TRACING = 'false'
uv run --locked --no-sync pytest
```

---

## 📁 仓库结构

```text
CodeSwarm-Agent/
├── codeswarm_agent/
│   ├── manager_graph.py     # Manager V2 执行图
│   ├── routing.py           # 任务路由
│   ├── langgraph_backend.py # 状态持久化与恢复
│   ├── context/             # 代码检索与上下文管理
│   ├── runtime/             # 工具调用与执行后端
│   ├── artifacts.py         # 证据与补丁 Artifact
│   └── telemetry.py         # 轨迹记录与 LangSmith 导出
├── trae_agent/              # 上游模型与工具适配
├── configs/                 # 配置
├── docker/                  # 容器镜像配方
├── tests/codeswarm_agent/    # 基础与集成测试
└── trae_config.yaml.example # CLI 配置模板
```

---

## 🛠️ 技术栈

| 层级 | 技术 |
| --- | --- |
| 语言与依赖管理 | Python 3.12、uv |
| 编排与持久化 | LangGraph、SQLite |
| 执行与工具 | Git worktree、Windows 受控执行、Docker、MCP |
| 可观测性 | JSONL、OpenTelemetry、LangSmith |
| 测试 | pytest |

---

## 📄 开源许可与致谢

本项目沿用 MIT 许可，并保留上游版权声明。模型与工具层基于 [ByteDance Trae Agent](https://github.com/bytedance/trae-agent)。
