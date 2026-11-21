"""Print redacted local capability information for CodeSwarm execution."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess


def capabilities() -> dict[str, object]:
    docker_cli = shutil.which("docker") is not None
    docker_ready = False
    docker_error = None if docker_cli else "docker CLI unavailable"
    if docker_cli:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .ServerVersion}}"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            docker_ready = result.returncode == 0
            if not docker_ready:
                docker_error = "daemon unavailable or access denied"
        except (OSError, subprocess.TimeoutExpired):
            docker_error = "docker capability probe failed"
    return {
        "schema_version": 1,
        "os": {"system": platform.system(), "release": platform.release(),
               "machine": platform.machine()},
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "execution_backends": {
            "windows-guarded": {
                "available": os.name == "nt" and importlib.util.find_spec("win32job") is not None,
                "strength": "process_guarded_trusted",
                "filesystem_isolation": False,
                "network_isolation": False,
            },
            "docker": {
                "cli_available": docker_cli,
                "daemon_ready": docker_ready,
                "error": docker_error,
                "strength": "container_inspected" if docker_ready else "not_run",
            },
        },
        "context_ml": {
            "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
        },
        "secrets": "redacted",
    }


def main() -> None:
    print(json.dumps(capabilities(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
