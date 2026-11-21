"""Cooperative cancellation shared by orchestration, remote I/O, and sandboxes."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, TypeVar

T = TypeVar("T")


class RunCancelled(asyncio.CancelledError):
    """Cancellation carrying a stable user-visible reason."""

    def __init__(self, reason: str = "run cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class CancellationToken:
    """One-shot cancellation token that can also follow a persisted marker file."""

    marker_path: Path | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)
    _reason: str = field(default="run cancelled", init=False)

    @property
    def cancelled(self) -> bool:
        self._sync_marker()
        return self._event.is_set()

    @property
    def reason(self) -> str:
        self._sync_marker()
        return self._reason

    def cancel(self, reason: str = "run cancelled", *, persist: bool = True) -> None:
        if self._event.is_set():
            return
        self._reason = reason.strip() or "run cancelled"
        self._event.set()
        if persist and self.marker_path is not None:
            self.marker_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.marker_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "reason": self._reason,
                        "cancelled_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.marker_path)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled(self.reason)

    async def wait(self) -> str:
        while not self._event.is_set():
            self._sync_marker()
            if self._event.is_set():
                break
            try:
                await asyncio.wait_for(self._event.wait(), timeout=0.2)
            except TimeoutError:
                continue
        return self._reason

    async def run(self, operation: Awaitable[T]) -> T:
        """Race an operation against cancellation and reclaim the losing task."""

        self.raise_if_cancelled()
        work = asyncio.ensure_future(operation)
        watcher = asyncio.create_task(self.wait())
        try:
            done, _ = await asyncio.wait(
                {work, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await work
                raise RunCancelled(watcher.result())
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            return await work
        except RunCancelled:
            raise
        except asyncio.CancelledError as exc:
            # Ctrl+C, LangGraph cancellation and a cancelled provider request all
            # converge on the same durable run-level protocol.
            self.cancel("operation cancelled")
            work.cancel()
            watcher.cancel()
            await asyncio.gather(work, watcher, return_exceptions=True)
            raise RunCancelled(self.reason) from exc

    def _sync_marker(self) -> None:
        if self._event.is_set() or self.marker_path is None or not self.marker_path.exists():
            return
        try:
            payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
            reason = str(payload.get("reason") or "run cancelled")
        except (OSError, ValueError, TypeError):
            reason = "run cancelled"
        self.cancel(reason, persist=False)


def request_run_cancellation(run_directory: str | Path, reason: str) -> Path:
    """Persist a cancellation request consumable by a live or resumed run."""

    marker = Path(run_directory).resolve() / "cancel.json"
    token = CancellationToken(marker)
    token.cancel(reason)
    return marker
