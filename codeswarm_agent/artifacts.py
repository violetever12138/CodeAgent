"""Immutable artifact blobs backed by a transactional SQLite catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import statistics
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    kind: str
    path: str
    size: int
    sha256: str
    created_at: str | None = None
    producer_task: str | None = None
    logical_key: str | None = None
    supersedes_id: str | None = None


class ArtifactStore:
    """Content-addressed blobs with O(1) catalog updates and logical lineage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.sqlite3"
        self._lock = RLock()
        self._insert_latencies_ms: deque[float] = deque(maxlen=1000)
        self._connection = sqlite3.connect(
            self.catalog_path, timeout=30, check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._initialize()
        self._migrate_legacy_index()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    producer_task TEXT
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    logical_key TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
                    supersedes_id TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact_insert_metrics (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    latency_ms REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_kind_idx ON artifacts(kind);
                """
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(ARTIFACT_SCHEMA_VERSION),),
            )

    def _migrate_legacy_index(self) -> None:
        legacy = self.root / "index.json"
        if not legacy.exists():
            return
        with self._lock:
            count = self._connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            if count:
                return
            entries = json.loads(legacy.read_text(encoding="utf-8"))
            now = datetime.now(timezone.utc).isoformat()
            with self._connection:
                for entry in entries:
                    path = Path(str(entry["path"])).resolve()
                    if self.root not in path.parents or path.stem != entry["id"]:
                        raise ValueError(f"artifact path escapes store: {entry['id']}")
                    self._connection.execute(
                        """INSERT OR IGNORE INTO artifacts
                           (id, kind, filename, size, sha256, created_at, producer_task)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry["id"], entry["kind"], path.name, int(entry["size"]),
                            entry["sha256"], entry.get("created_at") or now,
                            entry.get("producer_task"),
                        ),
                    )
            migrated = legacy.with_name("index.legacy.json")
            if not migrated.exists():
                legacy.replace(migrated)
                migrated.chmod(stat.S_IREAD)

    def put_text(
        self,
        kind: str,
        content: str,
        suffix: str = ".txt",
        *,
        producer_task: str | None = None,
        logical_key: str | None = None,
    ) -> ArtifactRecord:
        started = time.perf_counter()
        payload = content.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        safe_kind = re.sub(r"[^a-zA-Z0-9_.-]", "-", kind).strip("-") or "artifact"
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        artifact_id = f"{safe_kind}-{digest[:16]}"
        filename = f"{artifact_id}{safe_suffix}"
        path = self.root / filename
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if not path.exists():
                temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{id(self)}.tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
            supersedes_id = None
            with self._connection:
                self._connection.execute(
                    """INSERT OR IGNORE INTO artifacts
                       (id, kind, filename, size, sha256, created_at, producer_task)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (artifact_id, kind, filename, len(payload), digest, now, producer_task),
                )
                if logical_key:
                    previous = self._connection.execute(
                        "SELECT artifact_id FROM aliases WHERE logical_key = ?", (logical_key,),
                    ).fetchone()
                    supersedes_id = previous[0] if previous and previous[0] != artifact_id else None
                    self._connection.execute(
                        """INSERT INTO aliases(logical_key, artifact_id, supersedes_id, updated_at)
                           VALUES (?, ?, ?, ?)
                           ON CONFLICT(logical_key) DO UPDATE SET
                             artifact_id=excluded.artifact_id,
                             supersedes_id=excluded.supersedes_id,
                             updated_at=excluded.updated_at""",
                        (logical_key, artifact_id, supersedes_id, now),
                    )
            record = self.get(artifact_id, logical_key=logical_key)
            latency = (time.perf_counter() - started) * 1000
            self._insert_latencies_ms.append(latency)
            with self._connection:
                self._connection.execute(
                    "INSERT INTO artifact_insert_metrics(latency_ms) VALUES (?)", (latency,)
                )
            return record

    def get(self, artifact_id: str, *, logical_key: str | None = None) -> ArtifactRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown artifact: {artifact_id}")
            alias = None
            if logical_key:
                alias = self._connection.execute(
                    "SELECT * FROM aliases WHERE logical_key = ? AND artifact_id = ?",
                    (logical_key, artifact_id),
                ).fetchone()
            return self._record(row, alias)

    def latest(self, logical_key: str) -> ArtifactRecord:
        with self._lock:
            alias = self._connection.execute(
                "SELECT * FROM aliases WHERE logical_key = ?", (logical_key,),
            ).fetchone()
            if alias is None:
                raise KeyError(f"unknown artifact logical key: {logical_key}")
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (alias["artifact_id"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"artifact alias is dangling: {logical_key}")
            return self._record(row, alias)

    def _record(self, row: sqlite3.Row, alias: sqlite3.Row | None = None) -> ArtifactRecord:
        record = ArtifactRecord(
            id=row["id"], kind=row["kind"], path=str(self.root / row["filename"]),
            size=int(row["size"]), sha256=row["sha256"], created_at=row["created_at"],
            producer_task=row["producer_task"],
            logical_key=alias["logical_key"] if alias is not None else None,
            supersedes_id=alias["supersedes_id"] if alias is not None else None,
        )
        return self._validate_record(record)

    def read_text(self, artifact_id: str, max_chars: int | None = None) -> str:
        record = self.get(artifact_id)
        payload = Path(record.path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError(f"artifact content hash mismatch: {artifact_id}")
        content = payload.decode("utf-8")
        if max_chars is not None and len(content) > max_chars:
            return content[:max_chars] + "\n<artifact clipped>"
        return content

    def read_lines(self, artifact_id: str, start_line: int, end_line: int) -> str:
        if start_line < 1 or end_line < start_line or end_line - start_line > 200:
            raise ValueError("artifact line range must contain 1-201 lines")
        selected = []
        for number, line in enumerate(self.read_text(artifact_id).splitlines(), 1):
            if number > end_line:
                break
            if number >= start_line:
                selected.append(f"{number:6}\t{line.rstrip()}")
        return "\n".join(selected)

    def _validate_record(self, record: ArtifactRecord) -> ArtifactRecord:
        path = Path(record.path).resolve()
        if self.root not in path.parents or path.stem != record.id:
            raise ValueError(f"artifact path escapes store: {record.id}")
        return record

    def records(self) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM artifacts ORDER BY created_at, id"
            ).fetchall()
            return tuple(self._record(row) for row in rows)

    def metrics(self) -> dict[str, int | float | None]:
        with self._lock:
            artifact_count = self._connection.execute(
                "SELECT COUNT(*) FROM artifacts"
            ).fetchone()[0]
            alias_count = self._connection.execute(
                "SELECT COUNT(*) FROM aliases"
            ).fetchone()[0]
            latency_rows = self._connection.execute(
                """SELECT latency_ms FROM artifact_insert_metrics
                   ORDER BY sequence DESC LIMIT 1000"""
            ).fetchall()
        latencies = [float(row[0]) for row in latency_rows]
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_count": int(artifact_count),
            "alias_count": int(alias_count),
            "full_catalog_rewrites": 0,
            "artifact_insert_latency_median_ms": (
                statistics.median(latencies) if latencies else None
            ),
        }

    def close(self) -> None:
        with self._lock:
            self._connection.close()
