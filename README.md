<div align="center">

# 🚀 CodeAgent

### *面向仓库级开发任务的多智能体协作、代码修改与验证*

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/编排-LangGraph-167b70)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/license/mit)

</div>

---

## 📖 项目背景

仓库级开发任务通常跨越多个模块。随着 Agent 从代码定位进入修改与验证，三个问题直接影响代码交付。

1. **协作边界** 任务过于复杂时单agent上下文容量不足，多名 Worker 同时修改已有文件却可能引入冲突。
2. **上下文预算** 源码片段、工具输出和测试日志持续累积导致上下文不足，过早丢弃原文，又会让后续判断失去依据。
3. **断点恢复** 当任务因为不可抗因素中断，系统能够从断点恢复任务进程。

CodeAgent 基于 Trae Agent 和 LangGraph 构建，由 Manager V2 组织调查、修改和验收，交付修改后的代码及对应的验证结果。

---

## 🎯 实验动机

> 哪些工作值得并行？哪些证据必须保留？修改后的代码应满足哪些验收条件？

项目围绕这三个问题设计执行策略。Manager 根据任务边界和已有证据调整分工，检索器根据当前工作区版本筛选代码，上下文管理器保存可回读的原文。集成各项代码修改后，Manager 根据测试结果和风险规则判定任务是否完成。

---

## 💡 解决方法

### 五大模块协同工作

| 模块 | 职责 | 核心技术 |
| --- | --- | --- |
| 🎛️ Manager V2 | 验收规划、派发与重新路由 | LangGraph、规则约束与模型决策 |
| 🔍 仓库检索 | 定位与任务相关的代码 | 路径与符号、BM25、AST 关系、可选 Dense、RRF |
| 🗜️ 上下文管理 | 控制窗口增长、保留原始证据 | 临时召回清理、阈值摘要、Artifact 按需回读 |
| 🧪 执行与验收 | 隔离修改、集成代码改动与测试 | Git worktree、执行后端、按风险触发独立审查 |
| 🔭 恢复与观测 | 保存进度并追踪模型和工具调用 | SQLite Checkpoint、LangSmith |

### 数据流全景

```text
任务描述 + 目标仓库
    ↓
Manager V2 制定验收计划，选择直接执行或分派 Worker
    ↓
按需检索代码 → 在隔离工作区调查与修改 → 保存证据 Artifact
    ↓
集成代码改动 → 运行验收测试 → 按风险触发独立审查
    ↓
交付修改后的代码与验证结果
```

---

## ✨ 项目精彩之处

### 🎛️ 1. Manager V2 自适应编排

- **动态路由** Manager 先制定验收计划，再决定直接执行或委派 Worker。工具失败或新证据出现时，可重新评估分工。
- **读写分离调度** 独立的只读调查可并行运行，对已有不可独立拆分的代码修改串行推进，减少合并代码改动时的冲突。

### 🔍 2. 面向当前工作区的混合代码检索

- **多路召回** 已知路径或符号时优先精确定位。对于自然语言任务描述，BM25 提供词项匹配，符号和 AST 关系补充结构线索，同时支持 Dense 语义召回作为补充。
- **版本约束** 增量索引跟踪文件变更，候选片段经工作区版本校验后参与 RRF 排名融合。检索器按上下文预算选取最终片段。

### 🗜️ 3. 可追溯的上下文压缩

- **工具输出外置** 长输出以 Artifact 保存，模型窗口保留预览和引用。后续步骤可按 ID 回读正文。
- **摘要前归档** 已消费的临时召回正文可先行清理。达到摘要阈值后，系统将本次摘要使用的旧交互和已有摘要归档，再调用摘要器。这为核对摘要遗漏保留了原始依据。

### 🧪 4. 基于测试证据的代码验收

- **隔离执行** Worker 在独立 Git worktree 中修改代码。集成后对修改后的代码运行配置的验收测试，风险规则可进一步触发独立审查。
- **交付产物** 交付工作区中修改后的代码，并附上验收结果，便于核对改动和已执行的测试。

### 🔭 5. 断点恢复与执行轨迹

- **状态恢复** LangGraph Checkpoint 持久化 Manager 状态与已确认的 Worker 结果，中断后可从运行目录继续执行。
- **轨迹分析** 通过 LangSmith 的调用树查看模型和工具调用的父子关系、耗时及可用的 Token 统计，定位长耗时调用和执行异常。

### 📊 6. 发布版本验证

| 验证项目 | 本地结果 |
| --- | --- |
| Windows 自动化测试 | 191 项通过，1 项可选 ANN 基准跳过 |
| 独立 Python 3.12 环境 | 7 项针对性测试通过，成功构建 wheel |
| Docker 后端 | 提供镜像构建配方 |

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


---

## 📁 仓库结构

```text
CodeAgent/
├── codeagent/
│   ├── manager_graph.py     # Manager V2 执行图
│   ├── routing.py           # 任务路由
│   ├── langgraph_backend.py # 状态持久化与恢复
│   ├── context/             # 代码检索与上下文管理
│   ├── runtime/             # 工具调用与执行后端
│   ├── artifacts.py         # 工具输出与摘要源归档
│   └── telemetry.py         # 轨迹记录与 LangSmith 导出
├── trae_agent/              # 上游模型与工具适配
├── configs/                 # 配置
├── docker/                  # 容器镜像配方
├── tests/codeagent/    # 基础与集成测试
└── trae_config.yaml.example # CLI 配置模板
```

---

## 🛠️ 技术栈

| 层级 | 技术 |
| --- | --- |
| 语言与依赖管理 | Python 3.12、uv |
| 编排与持久化 | LangGraph、SQLite |
| 执行与工具 | Git worktree、Windows 受控执行、Docker、MCP |
| 可观测性 | OpenTelemetry、LangSmith |
| 测试 | pytest |

---


## 📄 License

本项目沿用 MIT 许可，并保留上游版权声明。模型与工具层基于 [ByteDance Trae Agent](https://github.com/bytedance/trae-agent)。
