"""Consistent best-effort secret redaction for persisted traces and tool artifacts."""

from __future__ import annotations

import json
import re
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|auth[_-]?token|access[_-]?token|"
    r"refresh[_-]?token|password|passwd|secret|cookie|private[_-]?key)($|[_-])",
    re.I,
)
_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"(?i)(\b(?:[a-z][a-z0-9_-]*[_-])?(?:api[_-]?key|authorization|auth[_-]?token|access[_-]?token|"
        r"refresh[_-]?token|password|passwd|secret|cookie)\b[\"']?\s*[:=]\s*)"
        r"(\[REDACTED\]|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;\}\]]+)"
    ),
)


def redact_text(value: str) -> str:
    if value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                clean = redact_value(parsed)
                return json.dumps(clean, ensure_ascii=False) if clean != parsed else value
        except (ValueError, RecursionError):
            pass
    redacted = _TEXT_PATTERNS[0].sub(r"\1" + REDACTED, value)
    redacted = _TEXT_PATTERNS[1].sub(REDACTED, redacted)
    return _TEXT_PATTERNS[2].sub(r"\1" + REDACTED, redacted)


def redact_value(value: Any, *, key: str | None = None) -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(item_key): redact_value(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
