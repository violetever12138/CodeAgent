"""Command-line entry point for CodeSwarm orchestration."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import warnings
from pathlib import Path

from dotenv import dotenv_values

from codeagent.context import ContextSettings
from codeagent.context.dense import OpenAIEmbeddingProvider
from codeagent.context.qwen_embedding import QwenEmbeddingProvider
from codeagent.context.reranker import HybridReranker
from codeagent.orchestrator import CodeSwarmOrchestrator
from codeagent.protocol import Budget, TaskRequest, to_jsonable
from codeagent.safety import DockerBackend, SandboxProfile, WindowsGuardedBackend
from codeagent.worker import TraeWorkerFactory
from trae_agent.utils.config import Config


def _load_project_dotenv(config_path: Path) -> None:
    """Load non-empty local values without replacing explicit process settings."""
    for name, value in dotenv_values(config_path.parent / ".env").items():
        if value and not os.environ.get(name):
            os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeswarm-cli")
    parser.add_argument("issue", nargs="?", help="Repository issue or engineering task")
    parser.add_argument("--working-dir")
    parser.add_argument("--config-file", default="trae_config.yaml")
    parser.add_argument("--base-commit")
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--acceptance", action="append", default=[])
    parser.add_argument(
        "--max-work-units",
        type=int,
        default=3,
        help="Maximum number of tasks in one Manager assignment",
    )
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--max-wall-time", type=float, default=1800.0)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--run-root")
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--context-output-reserve", type=int, default=4096)
    parser.add_argument("--summary-trigger-ratio", type=float)
    parser.add_argument("--summary-target-ratio", type=float)
    parser.add_argument("--summary-max-tokens", type=int)
    parser.add_argument("--embedding-model", help="Optional OpenAI-compatible embedding model")
    parser.add_argument("--embedding-base-url", help="Optional OpenAI-compatible embedding endpoint")
    parser.add_argument("--embedding-provider", choices=("none", "qwen", "openai"))
    parser.add_argument("--embedding-device", help="Local Qwen device, e.g. cpu or cuda:0")
    parser.add_argument("--embedding-revision", help="Fixed local Qwen model commit")
    parser.add_argument("--embedding-max-tokens", type=int)
    parser.add_argument("--trace-project", help="LangSmith project for CodeSwarm traces")
    parser.add_argument("--no-tracing", action="store_true")
    parser.add_argument("--architecture-mode", choices=("manager_v2",), default="manager_v2")
    parser.add_argument("--dense-backend", choices=("brute_force", "usearch"))
    parser.add_argument("--reranker-mode", choices=("feature", "hybrid"), default="feature")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument(
        "--reranker-revision",
        default="2cfc18c9415c912f9d8155881c133215df768a70",
    )
    parser.add_argument("--reranker-top-n", type=int, default=16)
    parser.add_argument("--reranker-timeout", type=float, default=4.0)
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument(
        "--resume-run",
        help="Resume an existing LangGraph run directory from its latest checkpoint",
    )
    parser.add_argument("--cancel-run", help="Request cancellation of an existing run directory")
    parser.add_argument("--cancel-reason", default="cancelled by user")
    parser.add_argument(
        "--approval-decision",
        choices=("accept", "replan", "terminate", "route_single"),
    )
    parser.add_argument("--approval-comment")
    parser.add_argument(
        "--execution-backend", choices=("auto", "windows-guarded", "docker"), default="auto",
    )
    parser.add_argument(
        "--trust-repository", action="store_true",
        help="Required for guarded host execution; declares this repository trusted",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--sandbox-image",
        help="Docker image pinned as name@sha256:digest; enables the hardened profile",
    )
    execution.add_argument(
        "--allow-unsafe-local-execution",
        action="store_true",
        help="Explicitly allow tools on the host; intended only for trusted local repositories",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--multi-agent", action="store_true")
    mode.add_argument("--single-agent", action="store_true")
    return parser


async def _run(args: argparse.Namespace):
    if args.cancel_run:
        marker = CodeSwarmOrchestrator.cancel(args.cancel_run, args.cancel_reason)
        return {"status": "cancellation_requested", "marker": str(marker)}
    config_path = Path(args.config_file).resolve()
    # Resolve secrets next to the selected config file so the CLI also works when
    # invoked outside the repository root or from an installed console script.
    # Some launchers inject empty variables; treat those as unset while retaining
    # the usual precedence of a non-empty process environment over the local file.
    _load_project_dotenv(config_path)
    config = Config.create(config_file=str(config_path)).resolve_config_values()
    if args.trace_project:
        config.codeswarm.tracing["project"] = args.trace_project
    if args.no_tracing:
        config.codeswarm.tracing["enabled"] = False
    if args.allow_unsafe_local_execution:
        warnings.warn(
            "--allow-unsafe-local-execution is deprecated; use "
            "--execution-backend windows-guarded --trust-repository",
            FutureWarning,
            stacklevel=2,
        )
        args.trust_repository = True
    execution_backend_name = args.execution_backend
    if execution_backend_name == "auto":
        execution_backend_name = "docker" if args.sandbox_image else "windows-guarded"
    if execution_backend_name == "windows-guarded" and not args.trust_repository:
        raise ValueError("the Windows guarded backend requires --trust-repository")
    if execution_backend_name == "windows-guarded" and args.sandbox_image:
        raise ValueError("--sandbox-image requires --execution-backend docker")
    if execution_backend_name == "docker" and not args.sandbox_image:
        raise ValueError("the Docker backend requires --sandbox-image pinned by digest")
    guarded_backend = None
    if execution_backend_name == "windows-guarded":
        guarded_backend = WindowsGuardedBackend(
            timeout_seconds=min(300.0, args.max_wall_time)
        )
    elif execution_backend_name == "docker":
        guarded_backend = DockerBackend(args.sandbox_image, SandboxProfile())
    if args.reranker_mode == "hybrid":
        HybridReranker(
            mode="hybrid", model=args.reranker_model,
            revision=args.reranker_revision,
        ).preload()
    embedding_provider = None
    embedding_mode = args.embedding_provider or ("openai" if args.embedding_model else config.codeswarm.embedding_provider)
    if embedding_mode == "qwen":
        embedding_provider = QwenEmbeddingProvider(
            revision=args.embedding_revision or config.codeswarm.embedding_revision,
            device=args.embedding_device or config.codeswarm.embedding_device,
            max_tokens=args.embedding_max_tokens or config.codeswarm.embedding_max_tokens,
        )
        if args.embedding_model and args.embedding_model != embedding_provider.model_id:
            raise ValueError("qwen provider uses Qwen/Qwen3-Embedding-0.6B")
    elif embedding_mode == "openai":
        if not args.embedding_model:
            raise ValueError("OpenAI-compatible embeddings require --embedding-model")
        primary = config.trae_agent.model.model_provider if config.trae_agent else None
        api_key = os.environ.get("OPENAI_API_KEY") or (
            primary.api_key if primary and primary.provider == "openai" else None
        )
        if not api_key:
            raise ValueError("--embedding-model requires OPENAI_API_KEY or an OpenAI primary provider")
        embedding_provider = OpenAIEmbeddingProvider(
            args.embedding_model,
            api_key,
            args.embedding_base_url or (primary.base_url if primary and primary.provider == "openai"
                                        else None),
        )
    orchestrator = CodeSwarmOrchestrator(
        TraeWorkerFactory(
            config,
            ContextSettings(
                context_window_tokens=args.context_window,
                output_reserve_tokens=args.context_output_reserve,
                summary_trigger_ratio=args.summary_trigger_ratio if args.summary_trigger_ratio is not None else config.codeswarm.summary_trigger_ratio,
                summary_target_ratio=args.summary_target_ratio if args.summary_target_ratio is not None else config.codeswarm.summary_target_ratio,
                summary_max_tokens=args.summary_max_tokens if args.summary_max_tokens is not None else config.codeswarm.summary_max_tokens,
                reranker_mode=args.reranker_mode,
                reranker_model=args.reranker_model,
                reranker_revision=args.reranker_revision,
                reranker_top_n=args.reranker_top_n,
                reranker_timeout=args.reranker_timeout,
                dense_backend=args.dense_backend or config.codeswarm.dense_backend,
            ),
            embedding_provider,
            sandbox_image=args.sandbox_image,
            sandbox_profile=SandboxProfile() if args.sandbox_image else None,
            execution_backend=guarded_backend,
        ),
        run_root=args.run_root,
        keep_workspaces=args.keep_workspaces,
        architecture_mode=args.architecture_mode,
    )
    if args.resume_run:
        if args.issue or args.working_dir:
            raise ValueError("issue and --working-dir must be omitted when resuming a run")
        return await orchestrator.resume(
            args.resume_run,
            approval_decision=args.approval_decision,
            approval_comment=args.approval_comment,
        )
    if not args.issue or not args.working_dir:
        raise ValueError("issue and --working-dir are required for a new run")
    if args.approval_decision or args.approval_comment:
        raise ValueError("approval options require --resume-run")
    forced_mode = True if args.multi_agent else False if args.single_agent else None
    budget = Budget(
        max_parallel=args.max_parallel,
        max_workers=args.max_workers,
        max_retries_per_task=args.max_retries,
        max_replans=args.max_replans,
        max_work_units=args.max_work_units,
        max_wall_time_seconds=args.max_wall_time,
        max_total_tokens=args.max_total_tokens,
        max_tool_calls=args.max_tool_calls,
    )
    request = TaskRequest(
        issue=args.issue,
        repository=str(Path(args.working_dir).resolve()),
        base_commit=args.base_commit,
        acceptance_criteria=tuple(args.acceptance),
        test_commands=tuple(args.test_command),
        force_multi_agent=forced_mode,
        max_work_units=args.max_work_units,
        budget=budget,
    )
    return await orchestrator.run(request)


def main() -> None:
    args = build_parser().parse_args()
    outcome = asyncio.run(_run(args))
    print(json.dumps(to_jsonable(outcome), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
