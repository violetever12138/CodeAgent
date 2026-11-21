"""Leakage-resistant evaluation manifests, results, metrics, and reports."""

from codeagent.evaluation.layered import (
    DEFAULT_PROFILES,
    LayerValidationResult,
    ValidationLevel,
    ValidationStatus,
    evaluate_layer,
)
from codeagent.evaluation.manifest import EvaluationManifest, ManifestStore
from codeagent.evaluation.metrics import EvaluationResult, aggregate, compare_paired

__all__ = [
    "DEFAULT_PROFILES",
    "EvaluationManifest",
    "EvaluationResult",
    "LayerValidationResult",
    "ManifestStore",
    "ValidationLevel",
    "ValidationStatus",
    "aggregate",
    "compare_paired",
    "evaluate_layer",
]
