"""Hierarchical multi-agent orchestration built on Trae Agent workers."""

from codeswarm_agent.orchestrator import CodeSwarmOrchestrator
from codeswarm_agent.protocol import Budget, TaskOutcome, TaskRequest
from codeswarm_agent.worker import TraeWorkerFactory

__all__ = [
    "Budget",
    "CodeSwarmOrchestrator",
    "TaskOutcome",
    "TaskRequest",
    "TraeWorkerFactory",
]
