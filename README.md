<div align="center">

# 🚀 CodeSwarm-Agent

### *面向仓库级开发任务的多智能体协作、代码检索与补丁验证*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/编排-LangGraph-167b70)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/license/mit)

</div>

---

## 📖 项目背景

仓库级开发任务通常跨越多个模块。随着 Agent 从代码定位进入修改与验证，三个问题直接影响最终补丁的质量。

1. **协作边界**。调用关系和测试覆盖可以并行调查，多名 Worker 同时修改已有文件却可能引入冲突。
2. **上下文预算**。源码片段、工具输出和测试日志持续累积。过早丢弃原文，又会让后续判断失去依据。
3. **完成判定**。Worker 报告完成时，候选补丁仍需在集成后的代码上接受验证。

CodeSwarm-Agent 基于 Trae Agent 和 LangGraph 构建，由 Manager V2 组织调查、修改和验收，并将选中的补丁与验证证据关联。

---

## 🎯 实验动机

> 哪些工作值得并行？哪些证据必须保留？满足什么条件，Manager 才能接受一个补丁？

项目围绕这三个问题设计执行策略。Manager 根据任务边界和已有证据调整分工，检索器根据当前工作区版本筛选代码，上下文管理器保存可回读的原文。补丁进入集成阶段后，由测试结果和风险规则共同约束完成判定。

---

## 💡 解决方法

### 五大模块协同工作

| 模块 | 职责 | 核心技术 |
| --- | --- | --- |
| 🎛️ Manager V2 | 验收规划、派发与重新路由 | LangGraph、规则约束与模型决策 |
| 🔍 仓库检索 | 定位与任务相关的代码 | 路径与符号、BM25、AST 关系、可选 Dense、RRF |
| 🗜️ 上下文管理 | 控制窗口增长、保留原始证据 | 临时召回清理、阈值摘要、Artifact 按需回读 |
| 🧪 执行与验收 | 隔离修改、集成补丁与测试 | Git worktree、执行后端、按风险触发独立审查 |
| 🔭 恢复与观测 | 保存进度并追踪模型和工具调用 | SQLite Checkpoint、本地 JSONL、可选 LangSmith |

### 数据流全景

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

## ✨ 项目精彩之处

### 🎛️ 1. Manager V2 自适应编排

- **动态路由**。Manager 先制定验收计划，再决定直接执行或委派 Worker。工具失败或新证据出现时，可重新评估分工。
- **读写分离调度**。独立的只读调查并行运行，对已有代码的修改串行推进，减少补丁集成时的冲突。

### 🔍 2. 面向当前工作区的混合代码检索

- **多路召回**。已知路径或符号时优先精确定位。对于自然语言任务描述，BM25 提供词项匹配，符号和 AST 关系补充结构线索，也可启用 Dense 语义召回。
- **版本约束**。增量索引跟踪文件变更，候选片段经工作区版本校验后参与 RRF 排名融合。检索器按上下文预算选取最终片段。

### 🗜️ 3. 可追溯的上下文压缩

- **工具输出外置**。长输出以 Artifact 保存，模型窗口保留预览和引用。后续步骤可按 ID 回读正文。
- **摘要前归档**。已消费的临时召回正文可先行清理。达到摘要阈值后，系统将本次摘要使用的旧交互和已有摘要归档，再调用摘要器。这为核对摘要遗漏保留了原始依据。

### 🧪 4. 基于测试证据的补丁验收

- **隔离执行**。Worker 在独立 Git worktree 中修改代码。集成阶段对候选补丁运行配置的验收测试，风险规则可进一步触发独立审查。
- **交付产物**。`selected_patch_artifact_id` 标识最终选中的补丁，便于将实际改动与对应的验证结果一并复查。

### 🔭 5. 断点恢复与执行轨迹

- **状态恢复**。LangGraph Checkpoint 持久化 Manager 状态与已确认的 Worker 结果，中断后可从运行目录继续执行。
- **轨迹分析**。本地 JSONL 记录路由、检索、摘要和模型工具调用。启用 LangSmith 后，可按调用树查看父子关系、耗时及可用的 Token 统计。

### 📊 6. 发布版本验证

| 验证项目 | 本地结果 |
| --- | --- |
| Windows 自动化测试 | 191 项通过，1 项可选 ANN 基准跳过 |
| 独立 Python 3.12 环境 | 7 项针对性测试通过，成功构建 wheel |
| Docker 后端 | 已提供镜像构建配方，容器运行待验证 |

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

## 🗺️ Roadmap

- [x] Manager V2 路由与任务恢复
- [x] 版本感知的代码检索与摘要源归档
- [x] 补丁验收与 LangSmith 调用追踪
- [ ] Docker 后端端到端验证

---

## 🤝 贡献

欢迎通过 Issue 或 Pull Request 讨论协作策略、代码检索和补丁验证。报告执行问题时，请附上能够复现问题的任务描述、执行后端和测试命令。

---

## 📄 License

本项目沿用 MIT 许可，并保留上游版权声明。模型与工具层基于 [ByteDance Trae Agent](https://github.com/bytedance/trae-agent)。
