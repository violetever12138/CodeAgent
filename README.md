<div align="center">

# 🚀 CodeSwarm-Agent

### *从仓库任务到可验证的补丁，保留检索证据与执行轨迹*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/编排-LangGraph-167b70)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/license/mit)

</div>

---

## 📖 项目背景

仓库级代码修改既要找对位置，也要给出可以复查的补丁和测试结果。一次任务可能跨越多个文件，产生大量检索结果和测试日志，还可能在调试中途被打断。

CodeSwarm-Agent 在 Trae Agent 的模型与工具基础上，用 LangGraph 将任务调度、代码定位、补丁集成和验证串成一条可恢复的执行流程。

---

## 🎯 设计目标

- **按任务风险分工**。简单任务由 Manager 直接处理，可拆分的调查交给 Worker 并行执行。
- **按需获取证据**。通过代码检索缩小范围，把长日志和旧上下文原文保存在可回读的 Artifact 中。
- **以验证结果交付**。用补丁、测试和必要的独立审查支持任务完成判定。

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

### 🔎 2. 从任务描述定位代码

优先匹配明确给出的路径和符号。只有自然语言描述时，可通过 BM25 和代码结构提供候选，并按需启用 Dense 检索。候选先经过当前工作区的版本及文件校验，再按 RRF 融合排序，选中的片段在预算内进入模型上下文。

### 📦 3. 压缩前归档摘要源，长输出按需回读

长工具输出的正文保存为 Artifact，活跃上下文保留有限预览与引用。已消费的临时召回正文可从窗口清理。达到摘要阈值时，先归档本次摘要实际使用的旧交互内容及已有摘要，再调用摘要器。需要核对细节时，可按 Artifact ID 回读原文。

### ✅ 4. 用补丁、测试和审查决定是否完成

Agent 在独立 Git worktree 中修改代码，先集成候选补丁，再运行配置的验收测试。风险规则可要求独立审查。输出中的 `selected_patch_artifact_id` 指向选中的补丁。

### 🔭 5. 保留可检查的模型与工具调用轨迹

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
