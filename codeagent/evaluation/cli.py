"""Aggregate independent harness JSONL without invoking the Agent again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codeagent.evaluation.layered import (
    ValidationLevel,
    ValidationStatus,
    evaluate_layer,
    profile_from_dict,
)
from codeagent.evaluation.manifest import ManifestStore
from codeagent.evaluation.metrics import EvaluationResult
from codeagent.evaluation.report import write_report, write_validation_report


def _load_results(path: str | Path) -> tuple[EvaluationResult, ...]:
    return tuple(
        EvaluationResult(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="codeswarm-eval")
    parser.add_argument("results", help="Independent harness result JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--level", choices=[level.value for level in ValidationLevel])
    parser.add_argument("--baseline", help="Paired baseline result JSONL for L1/L2")
    parser.add_argument("--manifest", help="Checksummed EvaluationManifest for budget gates")
    parser.add_argument(
        "--baseline-manifest", help="Checksummed baseline manifest for control parity"
    )
    parser.add_argument("--profile", help="Optional versioned validation profile JSON")
    parser.add_argument(
        "--enforce", action="store_true",
        help="Exit with code 2 unless the selected layer passes",
    )
    args = parser.parse_args()
    rows = _load_results(args.results)
    metrics = write_report(rows, args.output)
    output: dict[str, object] = {"metrics": metrics}
    if args.level:
        baseline = _load_results(args.baseline) if args.baseline else None
        manifest = ManifestStore(args.manifest).load() if args.manifest else None
        baseline_manifest = (
            ManifestStore(args.baseline_manifest).load()
            if args.baseline_manifest else None
        )
        profile = (
            profile_from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
            if args.profile else None
        )
        validation = evaluate_layer(
            args.level,
            rows,
            baseline=baseline,
            manifest=manifest,
            baseline_manifest=baseline_manifest,
            profile=profile,
        )
        write_validation_report(validation, args.output)
        output["validation"] = validation.to_dict()
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if args.enforce and validation.status != ValidationStatus.PASS:
            raise SystemExit(2)
        return
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
