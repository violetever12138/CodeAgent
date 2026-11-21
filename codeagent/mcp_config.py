"""Bind explicitly configured local services to this Worker, without a shell."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


def bind_mcp_configs(configs, workspace, artifact_root):
    values = {
        "CODESWARM_PYTHON": sys.executable,
        "CODESWARM_PACKAGE_ROOT": str(Path(__file__).resolve().parents[1]),
        "CODESWARM_WORKSPACE": str(Path(workspace).resolve()),
        "CODESWARM_ARTIFACT_ROOT": str(Path(artifact_root).resolve()),
    }

    def expand(value):
        if value is None:
            return None
        for key, replacement in values.items():
            value = value.replace("${" + key + "}", replacement)
        return value

    return {name: replace(config, command=expand(config.command),
                          args=[expand(arg) for arg in config.args] if config.args is not None else None,
                          cwd=expand(config.cwd), env={**{key: expand(value) for key, value in (config.env or {}).items()},
                              "CODESWARM_WORKSPACE": values["CODESWARM_WORKSPACE"],
                              "CODESWARM_ARTIFACT_ROOT": values["CODESWARM_ARTIFACT_ROOT"]})
            for name, config in (configs or {}).items()}
