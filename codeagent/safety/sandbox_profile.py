"""Validated Docker hardening profile used when CodeSwarm sandbox mode is enabled."""

from __future__ import annotations

import re
from dataclasses import dataclass

_DIGEST_PINNED_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class SandboxProfile:
    name: str = "repo_task_v1"
    user: str = "10001:10001"
    network_disabled: bool = True
    read_only_rootfs: bool = True
    memory_mb: int = 4096
    cpus: float = 2.0
    pids_limit: int = 256
    tmpfs_mb: int = 1024
    require_image_digest: bool = True

    def __post_init__(self) -> None:
        if min(self.memory_mb, self.pids_limit, self.tmpfs_mb) <= 0 or self.cpus <= 0:
            raise ValueError("sandbox resource limits must be positive")
        if not self.user or self.user in {"0", "0:0", "root"}:
            raise ValueError("sandbox must use a non-root user")

    def validate_image(self, image: str) -> None:
        if self.require_image_digest and not _DIGEST_PINNED_IMAGE.fullmatch(image):
            raise ValueError("sandbox image must be pinned by sha256 digest")

    def docker_options(self) -> dict[str, object]:
        return {
            "user": self.user,
            "network_disabled": self.network_disabled,
            "read_only": self.read_only_rootfs,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "mem_limit": f"{self.memory_mb}m",
            "nano_cpus": int(self.cpus * 1_000_000_000),
            "pids_limit": self.pids_limit,
            "tmpfs": {
                "/tmp": f"rw,noexec,nosuid,nodev,size={self.tmpfs_mb}m",
            },
            "mount_tools_read_only": True,
        }
