"""Persistent repository index with per-file incremental overlays."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from threading import RLock
from typing import Iterable

from codeswarm_agent.context.protocol import CodeChunk

INDEX_SCHEMA_VERSION = 2

_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".go": "go", ".rs": "rust", ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".txt": "text", ".sh": "bash",
}
_SYMBOL_TYPES = {
    "function_definition", "async_function_definition", "class_definition",
    "method_definition", "function_declaration", "method_declaration",
    "class_declaration", "interface_declaration", "function_item",
}
_SKIP_PARTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", "dist", "build", "vendor", ".codeswarm-runs",
}


class RepositoryIndexStore:
    """SQLite persistence for snapshots and content-addressed chunk records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    file_hash TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    PRIMARY KEY(file_hash, relative_path)
                );
                CREATE TABLE IF NOT EXISTS files (
                    workspace_key TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    tombstone INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(workspace_key, base_commit, relative_path)
                );
                CREATE TABLE IF NOT EXISTS base_files (
                    base_commit TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    PRIMARY KEY(base_commit, relative_path)
                );
                """
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(INDEX_SCHEMA_VERSION),),
            )

    @staticmethod
    def _decode(payload: str) -> tuple[CodeChunk, ...]:
        return tuple(CodeChunk(**item) for item in json.loads(payload))

    @staticmethod
    def _encode(chunks: tuple[CodeChunk, ...]) -> str:
        return json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False)

    def load(
        self, workspace_key: str, base_commit: str,
    ) -> dict[str, tuple[str, tuple[CodeChunk, ...]]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT relative_path, file_hash, chunks_json, tombstone FROM files
                   WHERE workspace_key = ? AND base_commit = ?""",
                (workspace_key, base_commit),
            ).fetchall()
        return {
            row["relative_path"]: (row["file_hash"], self._decode(row["chunks_json"]))
            for row in rows if not row["tombstone"]
        }

    def has_workspace_snapshot(self, workspace_key: str, base_commit: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM files WHERE workspace_key=? AND base_commit=? LIMIT 1",
                (workspace_key, base_commit),
            ).fetchone()
        return row is not None

    def tombstones(self, workspace_key: str, base_commit: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT relative_path FROM files
                   WHERE workspace_key=? AND base_commit=? AND tombstone=1""",
                (workspace_key, base_commit),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def load_base(self, base_commit: str) -> dict[str, tuple[str, tuple[CodeChunk, ...]]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT relative_path, file_hash, chunks_json FROM base_files WHERE base_commit=?",
                (base_commit,),
            ).fetchall()
        return {
            row["relative_path"]: (row["file_hash"], self._decode(row["chunks_json"]))
            for row in rows
        }

    def save_base(
        self, base_commit: str, files: dict[str, tuple[str, tuple[CodeChunk, ...]]],
    ) -> None:
        with self._lock, self._connection:
            for path, (file_hash, chunks) in files.items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO base_files VALUES (?, ?, ?, ?)",
                    (base_commit, path, file_hash, self._encode(chunks)),
                )

    def cached(self, relative_path: str, file_hash: str) -> tuple[CodeChunk, ...] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT chunks_json FROM chunk_cache WHERE file_hash = ? AND relative_path = ?",
                (file_hash, relative_path),
            ).fetchone()
        return self._decode(row[0]) if row else None

    def update(
        self,
        workspace_key: str,
        base_commit: str,
        upserts: dict[str, tuple[str, tuple[CodeChunk, ...]]],
        deleted: Iterable[str],
    ) -> int:
        tombstones = tuple(dict.fromkeys(deleted))
        with self._lock, self._connection:
            for path, (file_hash, chunks) in upserts.items():
                payload = self._encode(chunks)
                self._connection.execute(
                    "INSERT OR IGNORE INTO chunk_cache VALUES (?, ?, ?)",
                    (file_hash, path, payload),
                )
                self._connection.execute(
                    """INSERT INTO files VALUES (?, ?, ?, ?, ?, 0)
                       ON CONFLICT(workspace_key, base_commit, relative_path) DO UPDATE SET
                         file_hash=excluded.file_hash,
                         chunks_json=excluded.chunks_json,
                         tombstone=0""",
                    (workspace_key, base_commit, path, file_hash, payload),
                )
            for path in tombstones:
                self._connection.execute(
                    """INSERT INTO files VALUES (?, ?, ?, '', '[]', 1)
                       ON CONFLICT(workspace_key, base_commit, relative_path) DO UPDATE SET
                         file_hash='', chunks_json='[]', tombstone=1""",
                    (workspace_key, base_commit, path),
                )
        return len(upserts) + len(tombstones)


class RepositoryIndex:
    def __init__(
        self,
        root: str | Path,
        trace_root: str | Path,
        *,
        max_file_bytes: int = 250_000,
        max_files: int = 20_000,
        store_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.trace_root = Path(trace_root).resolve()
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.base_commit = self._git_output("rev-parse", "HEAD").strip() or "non-git"
        self.workspace_key = hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        shared_store = store_path or (self.trace_root.parent / "repository-index.sqlite3")
        self.store = RepositoryIndexStore(shared_store)
        has_workspace = self.store.has_workspace_snapshot(self.workspace_key, self.base_commit)
        workspace_files = self.store.load(self.workspace_key, self.base_commit)
        base_files = self.store.load_base(self.base_commit)
        self._files = dict(base_files)
        self._files.update(workspace_files)
        for path in self.store.tombstones(self.workspace_key, self.base_commit):
            self._files.pop(path, None)
        self._loaded_shared_base = False
        if not has_workspace and base_files:
            self._loaded_shared_base = True
        self.repository_version = self.base_commit
        self.chunks: tuple[CodeChunk, ...] = ()
        self.metrics = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "scanned_files": 0,
            "rechunked_files": 0,
            "updated_rows": 0,
            "full_rebuilds": 0,
        }
        self._publish_chunks()

    def refresh(self, paths: Iterable[str] | None = None) -> tuple[str, ...]:
        if paths is None and self._loaded_shared_base:
            paths = self.discover_changed_paths()
            self._loaded_shared_base = False
        if paths is None:
            selected = {path.relative_to(self.root).as_posix(): path for path in self._paths()}
            deleted = set(self._files) - set(selected)
            self.metrics["full_rebuilds"] += 1
        else:
            selected: dict[str, Path] = {}
            deleted: set[str] = set()
            for raw in dict.fromkeys(paths):
                normalized = self._safe_relative(raw)
                if normalized is None:
                    continue
                candidate = (self.root / normalized).resolve()
                if not candidate.exists():
                    if normalized in self._files:
                        deleted.add(normalized)
                    continue
                if self._is_indexable(candidate):
                    selected[normalized] = candidate

        changed: list[str] = []
        upserts: dict[str, tuple[str, tuple[CodeChunk, ...]]] = {}
        for relative, path in selected.items():
            self.metrics["scanned_files"] += 1
            try:
                content = path.read_bytes()
            except OSError:
                continue
            file_hash = hashlib.sha256(content).hexdigest()
            previous = self._files.get(relative)
            if previous and previous[0] == file_hash:
                continue
            try:
                source = content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            chunks = self.store.cached(relative, file_hash)
            if chunks is None:
                chunks = tuple(self._chunk(relative, source, file_hash))
                self.metrics["rechunked_files"] += 1
            self._files[relative] = (file_hash, chunks)
            upserts[relative] = (file_hash, chunks)
            changed.append(relative)
        for relative in sorted(deleted):
            self._files.pop(relative, None)
            changed.append(relative)
        self.metrics["updated_rows"] += self.store.update(
            self.workspace_key, self.base_commit, upserts, deleted,
        )
        self._publish_chunks()
        self._write_trace(changed)
        if paths is None and not self.discover_changed_paths():
            self.store.save_base(self.base_commit, self._files)
        return tuple(dict.fromkeys(changed))

    def discover_changed_paths(self) -> tuple[str, ...]:
        """Discover dirty/untracked/renamed paths without reading source contents."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                cwd=self.root, capture_output=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return tuple(self._files)
        if result.returncode != 0:
            return tuple(self._files)
        parts = result.stdout.decode("utf-8", "replace").split("\0")
        changed: list[str] = []
        index = 0
        while index < len(parts):
            entry = parts[index]
            index += 1
            if not entry:
                continue
            status = entry[:2]
            changed.append(entry[3:].replace("\\", "/"))
            if ("R" in status or "C" in status) and index < len(parts) and parts[index]:
                changed.append(parts[index].replace("\\", "/"))
                index += 1
        return tuple(dict.fromkeys(changed))

    def export_chunks(self, path: str | Path | None = None) -> Path:
        """Create the legacy JSONL debug export only when explicitly requested."""
        target = Path(path).resolve() if path else self.trace_root / "chunks.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for chunk in self.chunks:
                stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        return target

    def file_hash(self, relative_path: str) -> str | None:
        entry = self._files.get(relative_path)
        return entry[0] if entry else None

    def _publish_chunks(self) -> None:
        digest = hashlib.sha256()
        for path, (file_hash, _) in sorted(self._files.items()):
            digest.update(f"{path}\0{file_hash}\0".encode())
        self.repository_version = f"{self.base_commit}:{digest.hexdigest()[:20]}"
        self.chunks = tuple(
            replace(chunk, repository_version=self.repository_version)
            for path in sorted(self._files) for chunk in self._files[path][1]
        )

    def _paths(self) -> list[Path]:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=self.root, capture_output=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            names = [name for name in result.stdout.decode("utf-8", "replace").split("\0") if name]
        else:
            names = [str(path.relative_to(self.root)) for path in self.root.rglob("*")]
        paths: list[Path] = []
        for name in sorted(set(names)):
            normalized = self._safe_relative(name)
            if normalized is None:
                continue
            path = (self.root / normalized).resolve()
            if self._is_indexable(path):
                paths.append(path)
            if len(paths) >= self.max_files:
                break
        return paths

    def _safe_relative(self, value: str) -> str | None:
        relative = Path(value.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or set(relative.parts) & _SKIP_PARTS:
            return None
        return relative.as_posix()

    def _is_indexable(self, path: Path) -> bool:
        try:
            return (
                self.root in path.parents and not path.is_symlink() and path.is_file()
                and path.suffix.lower() in _LANGUAGES
                and path.stat().st_size <= self.max_file_bytes
            )
        except OSError:
            return False

    def _chunk(self, path: str, source: str, file_hash: str) -> list[CodeChunk]:
        lines = source.splitlines()
        language = _LANGUAGES[Path(path).suffix.lower()]
        spans: list[tuple[int, int, str | None, str]] = []
        structural: dict[str, str] = {}
        if language == "python":
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                tree = None
            if tree is not None:
                imports: set[str] = set()
                calls: set[str] = set()
                references: set[str] = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module)
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            calls.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            calls.add(node.func.attr)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        references.add(node.id)
                structural = {
                    "imports": ",".join(sorted(imports)),
                    "calls": ",".join(sorted(calls)),
                    "references": ",".join(sorted(references)),
                    "test_relation": "test" if "test" in Path(path).stem.lower() else "source",
                }

                def visit(nodes, parent: str = "") -> None:
                    for node in nodes:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            symbol = f"{parent}.{node.name}" if parent else node.name
                            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                            end = node.end_lineno or node.lineno
                            if isinstance(node, ast.ClassDef):
                                spans.append((start, min(end, start + 19), symbol, "class"))
                                visit(node.body, symbol)
                            else:
                                spans.append((start, end, symbol, "symbol"))
                visit(tree.body)
        elif language not in {"markdown", "yaml", "toml", "json", "text", "bash"}:
            try:
                from tree_sitter_languages import get_parser

                parser = get_parser(language)
                syntax = parser.parse(source.encode("utf-8"))
                stack = [syntax.root_node]
                while stack:
                    node = stack.pop()
                    if node.type in _SYMBOL_TYPES:
                        name_node = node.child_by_field_name("name")
                        symbol = name_node.text.decode("utf-8", "replace") if name_node else None
                        if symbol:
                            spans.append((node.start_point[0] + 1, node.end_point[0] + 1,
                                          symbol, "symbol"))
                    stack.extend(reversed(node.children))
            except (ImportError, LookupError, TypeError, ValueError):
                pass
        if lines:
            if spans:
                spans.insert(0, (1, min(len(lines), 60), None, "file"))
            else:
                spans.append((1, len(lines), None, "file"))
        chunks: list[CodeChunk] = []
        for start, end, symbol, kind in spans:
            if end < start:
                continue
            cursor = start
            while cursor <= end:
                stop = min(cursor + 79, end)
                text = "\n".join(lines[cursor - 1:stop])
                if text.strip():
                    chunk_id = hashlib.sha256(
                        f"{path}:{symbol}:{cursor}:{stop}:{file_hash}".encode()
                    ).hexdigest()[:24]
                    chunks.append(CodeChunk(
                        id=chunk_id, path=path, language=language, symbol=symbol,
                        start_line=cursor, end_line=stop, text=text,
                        file_hash=file_hash, repository_version="", kind=kind,
                        metadata=structural,
                    ))
                if stop == end:
                    break
                cursor = stop - 7
        if not spans and source.strip():
            chunks.append(CodeChunk(
                id=hashlib.sha256(f"{path}:{file_hash}".encode()).hexdigest()[:24],
                path=path, language=language, symbol=None, start_line=1,
                end_line=max(1, len(lines)), text=source[:8000], file_hash=file_hash,
                repository_version="", kind="file", metadata=structural,
            ))
        return chunks

    def _write_trace(self, changed: list[str]) -> None:
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "base_commit": self.base_commit,
            "repository_version": self.repository_version,
            "files": {path: value[0] for path, value in sorted(self._files.items())},
            "changed_files": changed,
            "chunk_count": len(self.chunks),
            "chunk_strategy": "ast-symbol-structure-v2",
            "metrics": self.metrics,
            "chunks_exported": False,
        }
        (self.trace_root / "index_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _git_output(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=self.root, capture_output=True,
                text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout if result.returncode == 0 else ""
