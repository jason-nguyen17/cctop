"""Read Codex-provider threads owned by Oh My Pi, without reading credentials.

OMP is a Bun application with its own session format, separate from native
Codex. Writable open files establish liveness; records describe the last observed
activity, with UNKNOWN used when no durable evidence establishes runtime state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from .codex import CodexSession, _integer, _run
from .status import SessionStatus


def _installed_script(args: list[str]) -> bool:
    """Recognize the installed OMP entrypoint, not text elsewhere in argv."""
    if not args:
        return False
    script = args[1] if args[0] == "run" and len(args) > 1 else args[0]
    path = Path(script)
    return (
        path.name == "cli.js"
        and path.parent.name == "dist"
        and (
            path.parent.parent.name == "pi-coding-agent"
            and path.parent.parent.parent.name == "@oh-my-pi"
        )
    )


def _omp_processes() -> list[int]:
    processes = []
    if sys.platform.startswith("linux"):
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid() or (entry / "exe").readlink().name != "bun":
                    continue
                name = (entry / "comm").read_text().strip()
                args = [
                    os.fsdecode(arg) for arg in (entry / "cmdline").read_bytes().split(b"\0") if arg
                ]
            except OSError:
                continue
            if name == "omp" or _installed_script(args[1:]):
                processes.append(int(entry.name))
    else:
        candidates = set()
        for line in _run(["ps", "-Ao", "pid=,uid=,comm="]).splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                if int(parts[1]) == os.getuid() and Path(parts[2]).name == "bun":
                    candidates.add(int(parts[0]))
        if candidates:
            pids = ",".join(str(pid) for pid in sorted(candidates))
            for line in _run(["ps", "-p", pids, "-o", "pid=,args="]).splitlines():
                parts = line.split(None, 1)
                if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) not in candidates:
                    continue
                try:
                    args = shlex.split(parts[1])
                except ValueError:
                    continue
                if args and (Path(args[0]).name == "omp" or _installed_script(args[1:])):
                    processes.append(int(parts[0]))
    return processes


def _open_sessions(pid: int, root: Path) -> list[Path]:
    names = []
    if sys.platform.startswith("linux"):
        try:
            descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
        except OSError:
            return []
        for descriptor in descriptors:
            try:
                target = descriptor.readlink()
                flags = Path(f"/proc/{pid}/fdinfo/{descriptor.name}").read_text()
                access = next(
                    (
                        int(line.split()[1], 8)
                        for line in flags.splitlines()
                        if line.startswith("flags:")
                    ),
                    0,
                )
            except (OSError, ValueError, IndexError):
                continue
            if access & os.O_ACCMODE:
                names.append(target)
    else:
        writable = False
        for line in _run(["lsof", "-a", "-p", str(pid), "-Fpfan"]).splitlines():
            if line.startswith("f"):
                writable = False
            elif line.startswith("a"):
                writable = line[1:] in ("w", "u")
            elif line.startswith("n") and writable:
                names.append(Path(line[1:]))
    paths = set()
    for name in names:
        if name.suffix != ".jsonl":
            continue
        try:
            path = name.resolve(strict=True)
        except OSError:
            continue
        if path.is_relative_to(root):
            paths.add(path)
    return sorted(paths)


@dataclass
class _SessionState:
    session_id: str = ""
    cwd: str = ""
    title: str = ""
    provider: str = ""
    model: str | None = None
    total_tokens: int = 0
    context_used: int = 0
    context_window: int = 0
    last_activity: datetime | None = None
    status: SessionStatus = SessionStatus.UNKNOWN
    pending: dict[str, str] = field(default_factory=dict)
    identity: tuple[int, int] = (0, 0)
    offset: int = 0
    mtime_ns: int = 0
    partial: bytes = b""
    skipping_line: bool = False


_lock = RLock()
_cache: OrderedDict[Path, _SessionState] = OrderedDict()
_WINDOW_CACHE: OrderedDict[Path, tuple[tuple[int, ...], dict[str, int]]] = OrderedDict()
_CHUNK = 256 * 1024
_MAX_LINE = 4 * 1024 * 1024


def _pending_status(state: _SessionState) -> SessionStatus:
    if any(name == "ask" for name in state.pending.values()):
        return SessionStatus.WAITING_PERMISSION
    return SessionStatus.GENERATING if state.pending else SessionStatus.UNKNOWN


def _apply_record(state: _SessionState, line: bytes) -> None:
    try:
        record = json.loads(line)
    except (ValueError, UnicodeError):
        return
    if not isinstance(record, dict):
        return
    kind = record.get("type")
    if kind in ("title", "title_change"):
        title = record.get("title")
        if isinstance(title, str):
            state.title = title
    elif kind == "session":
        for key, attribute in (("id", "session_id"), ("cwd", "cwd"), ("title", "title")):
            if isinstance(record.get(key), str):
                setattr(state, attribute, record[key])
    elif kind == "model_change" and record.get("role", "default") == "default":
        model = record.get("model")
        if isinstance(model, str) and "/" in model:
            state.provider, state.model = model.split("/", 1)
    elif kind == "custom" and record.get("customType") == "tool_execution_start":
        data = record.get("data")
        if isinstance(data, dict) and isinstance(data.get("toolCallId"), str):
            call_id = data["toolCallId"]
            name = data.get("toolName")
            if isinstance(name, str):
                state.pending[call_id] = name
            else:
                state.pending.setdefault(call_id, "")
            state.status = _pending_status(state)
    elif kind == "custom" and record.get("customType") == "session_exit":
        data = record.get("data")
        if isinstance(data, dict) and data.get("kind") in (
            "normal",
            "fatal",
            "process_exit",
            "signal",
        ):
            state.pending.clear()
            state.status = SessionStatus.IDLE
    elif kind == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return
        role = message.get("role")
        if role == "assistant":
            if isinstance(message.get("provider"), str):
                state.provider = message["provider"]
            if isinstance(message.get("model"), str):
                state.model = message["model"]
            usage = message.get("usage")
            if isinstance(usage, dict):
                component_total = sum(
                    _integer(usage.get(key), 0)
                    for key in ("input", "cacheRead", "cacheWrite", "output")
                )
                state.total_tokens += _integer(usage.get("totalTokens"), component_total)
                state.context_used = sum(
                    _integer(usage.get(key), 0) for key in ("input", "cacheRead", "cacheWrite")
                )
            snapshot = message.get("contextSnapshot")
            if isinstance(snapshot, dict):
                prompt = _integer(snapshot.get("promptTokens"), state.context_used)
                removed = _integer(snapshot.get("historyRewriteTokensRemoved"), 0)
                state.context_used = max(0, prompt - removed)
            reason = message.get("stopReason")
            if reason in ("stop", "aborted", "error"):
                state.pending.clear()
                state.status = SessionStatus.IDLE
            elif reason == "toolUse":
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "toolCall"
                            and isinstance(block.get("id"), str)
                        ):
                            name = block.get("name")
                            state.pending[block["id"]] = name if isinstance(name, str) else ""
                state.status = _pending_status(state)
            else:
                state.status = SessionStatus.UNKNOWN
        elif role == "toolResult":
            call_id = message.get("toolCallId")
            if isinstance(call_id, str):
                state.pending.pop(call_id, None)
            state.status = _pending_status(state)
        elif role == "user":
            state.status = SessionStatus.UNKNOWN
    else:
        # Do not retain custom payloads, credential pins, or message content.
        return
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            state.last_activity = (
                parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
            )
        except ValueError:
            pass


def _session_state(path: Path) -> _SessionState:
    with _lock:
        try:
            stat = path.stat()
        except OSError:
            _cache.pop(path, None)
            return _SessionState()
        identity = (stat.st_dev, stat.st_ino)
        state = _cache.get(path)
        if (
            state is None
            or state.identity != identity
            or stat.st_size < state.offset
            or (stat.st_size == state.offset and stat.st_mtime_ns != state.mtime_ns)
        ):
            state = _SessionState(identity=identity)
        _cache[path] = state
        _cache.move_to_end(path)
        while len(_cache) > 128:
            _cache.popitem(last=False)
        try:
            with path.open("rb") as handle:
                handle.seek(state.offset)
                while chunk := handle.read(_CHUNK):
                    state.offset += len(chunk)
                    lines = (state.partial + chunk).split(b"\n")
                    state.partial = lines.pop()
                    for line in lines:
                        if not state.skipping_line and len(line) <= _MAX_LINE:
                            _apply_record(state, line)
                        state.skipping_line = False
                    if len(state.partial) > _MAX_LINE:
                        state.partial = b""
                        state.skipping_line = True
                state.mtime_ns = stat.st_mtime_ns
        except OSError:
            pass
        return state


def _model_windows(config_dir: Path) -> dict[str, int]:
    """Only public model metadata from OMP's cache, opened strictly read-only."""
    path = config_dir / "models.db"
    try:
        stat = path.stat()
    except OSError:
        return {}
    # WAL commits can change model metadata without touching the main database.
    try:
        wal = path.with_name(path.name + "-wal").stat()
        wal_signature = (wal.st_ino, wal.st_mtime_ns, wal.st_size)
    except OSError:
        wal_signature = (0, 0, 0)
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size, *wal_signature)
    cached = _WINDOW_CACHE.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    windows = {}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1) as db:
            rows = db.execute(
                "SELECT models FROM model_cache WHERE provider_id LIKE ? ORDER BY updated_at DESC",
                ("openai-codex:%",),
            )
            for (raw,) in rows:
                models = json.loads(raw)
                if not isinstance(models, list):
                    continue
                for model in models:
                    if not isinstance(model, dict) or model.get("provider") != "openai-codex":
                        continue
                    model_id = model.get("id")
                    if isinstance(model_id, str) and model_id not in windows:
                        windows[model_id] = _integer(model.get("contextWindow"), 0)
    except (sqlite3.Error, ValueError, TypeError):
        pass
    _WINDOW_CACHE[path] = (signature, windows)
    _WINDOW_CACHE.move_to_end(path)
    while len(_WINDOW_CACHE) > 8:
        _WINDOW_CACHE.popitem(last=False)
    return windows


def discover_sessions(config_dir: Path) -> list[CodexSession]:
    """OMP Codex-provider threads with actual writable session-file ownership."""
    with _lock:
        config_dir = config_dir.expanduser().resolve()
        root = (config_dir / "sessions").resolve()
        live: dict[Path, int] = {}
        for pid in _omp_processes():
            for path in _open_sessions(pid, root):
                path = path.resolve()
                if path.is_relative_to(root):
                    live.setdefault(path, pid)
        windows = _model_windows(config_dir) if live else {}
        sessions = []
        for path, pid in live.items():
            state = _session_state(path)
            if not state.session_id or state.provider != "openai-codex":
                continue
            # Child headers may share identity; each owned file is a distinct thread.
            identity = hashlib.sha256(str(path).encode()).hexdigest()[:24]
            name = state.title or Path(state.cwd).name or state.session_id[:8]
            sessions.append(
                CodexSession(
                    pid,
                    f"omp:{identity}",
                    name,
                    state.cwd,
                    state.model,
                    state.total_tokens,
                    state.context_used,
                    windows.get(state.model or "", state.context_window),
                    state.last_activity,
                    state.status,
                )
            )
        for path in list(_cache):
            if path.is_relative_to(root) and path not in live:
                del _cache[path]
        return sessions
