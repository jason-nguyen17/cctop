"""Read live Codex threads from executable processes and their open rollouts.

A Desktop app-server can own many threads. Only open rollout handles (or an
explicit interactive CLI resume ID) establish ownership; cwd and file age do
not. Rollouts are read incrementally and lifecycle events determine activity.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock

from .status import SessionStatus

CODEX_DIR = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_ROLLOUT = re.compile(rf"rollout-.*-({_UUID})\.jsonl$")


@dataclass(frozen=True)
class CodexSession:
    pid: int
    session_id: str
    name: str
    cwd: str
    model: str | None
    total_tokens: int
    context_used: int
    context_window: int
    last_activity: datetime | None
    status: SessionStatus = SessionStatus.UNKNOWN


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _codex_processes() -> list[tuple[int, str]]:
    """Native Codex executables owned by this user, including app-servers.

    Node launchers are deliberately omitted: the native child owns the rollout.
    Checking executable identity avoids matching scripts with Codex in their args.
    """
    processes = []
    if sys.platform.startswith("linux"):
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                executable = (entry / "exe").readlink()
                if executable.name.removesuffix(" (deleted)") != "codex":
                    continue
                args = (entry / "cmdline").read_bytes().split(b"\0")
                command = shlex.join(arg.decode(errors="replace") for arg in args if arg)
            except OSError:
                continue
            processes.append((int(entry.name), command))
    else:
        # macOS truncates non-final ps columns, even with -ww. Keep the
        # executable last, preserving full paths (including spaces).
        candidates = set()
        for line in _run(["ps", "-Ao", "pid=,uid=,comm="]).splitlines():
            parts = line.split(None, 2)
            if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            if int(parts[1]) == os.getuid() and Path(parts[2]).name == "codex":
                candidates.add(int(parts[0]))
        if candidates:
            pids = ",".join(str(pid) for pid in sorted(candidates))
            for line in _run(["ps", "-p", pids, "-o", "pid=,args="]).splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in candidates:
                    processes.append((int(parts[0]), parts[1]))
    return processes


def _interactive_cli(command: str) -> bool:
    try:
        args = shlex.split(command)[1:]
    except ValueError:
        return False
    # These modes do not represent an interactive thread without an open handle.
    return not any(
        arg
        in {
            "app-server",
            "exec",
            "e",
            "exec-server",
            "mcp-server",
            "mcp",
            "login",
            "logout",
            "completion",
            "debug",
            "sandbox",
            "apply",
            "review",
        }
        for arg in args
    )


def _process_config_dir(pid: int) -> Path | None:
    """Read only CODEX_HOME from a Linux process; never expose its environment."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for entry in entries:
        if entry.startswith(b"CODEX_HOME="):
            return Path(os.fsdecode(entry[len(b"CODEX_HOME=") :])).expanduser().resolve()
    return (Path.home() / ".codex").resolve()


def _lsof_cwd(pid: int) -> str | None:
    if sys.platform.startswith("linux"):
        try:
            return str(Path(f"/proc/{pid}/cwd").readlink())
        except OSError:
            return None
    for line in _run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"]).splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def _open_rollouts(pid: int, root: Path) -> list[Path]:
    """Writable rollout handles beneath this account; read-only viewers do not own threads."""
    candidates: list[tuple[bool, str]] = []
    if sys.platform.startswith("linux"):
        try:
            descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
        except OSError:
            return []
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                flags = Path(f"/proc/{pid}/fdinfo/{descriptor.name}").read_text()
                match = re.search(r"^flags:\s+([0-7]+)$", flags, re.MULTILINE)
                writable = bool(match and int(match.group(1), 8) & os.O_ACCMODE)
            except OSError:
                continue
            if writable:
                candidates.append((True, target))
    else:
        # -F records make filenames independent of human column widths.
        writable = False
        for line in _run(["lsof", "-a", "-p", str(pid), "-Fpfan"]).splitlines():
            if line.startswith("f"):
                writable = False
            elif line.startswith("a"):
                writable = line[1:] in ("w", "u")
            elif line.startswith("n") and writable:
                candidates.append((True, line[1:]))
    paths = []
    seen = set()
    for _, target in sorted(candidates, key=lambda item: not item[0]):
        path = Path(target)
        if not _ROLLOUT.fullmatch(path.name):
            continue
        try:
            path = path.resolve(strict=True)
        except OSError:
            continue
        if path.is_relative_to(root) and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _rollout_for_id(session_id: str, config_dir: Path | None = None) -> Path | None:
    root = (config_dir or CODEX_DIR).resolve() / "sessions"
    return next(
        (
            path.resolve()
            for path in root.glob(f"**/rollout-*-{session_id}.jsonl")
            if path.resolve().is_relative_to(root)
        ),
        None,
    )


def _read_session_meta(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            record = json.loads(handle.readline(512_000))
    except (OSError, ValueError):
        return None
    if isinstance(record, dict) and record.get("type") == "session_meta":
        payload = record.get("payload")
        return payload if isinstance(payload, dict) else None
    return None


def _index_names(config_dir: Path | None = None) -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        with ((config_dir or CODEX_DIR) / "session_index.jsonl").open(
            encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                session_id, name = record.get("id"), record.get("thread_name")
                if isinstance(session_id, str) and isinstance(name, str):
                    names[session_id] = name
    except OSError:
        pass
    return names


@dataclass
class _RolloutState:
    meta: dict | None = None
    total_tokens: int = 0
    context_used: int = 0
    context_window: int = 0
    model: str | None = None
    last_activity: datetime | None = None
    status: SessionStatus = SessionStatus.UNKNOWN
    pending_input: set[str] = field(default_factory=set)
    identity: tuple[int, int] = (0, 0)
    offset: int = 0
    mtime_ns: int = 0
    partial: bytes = b""
    skipping_line: bool = False


_rollout_cache: OrderedDict[Path, _RolloutState] = OrderedDict()
_rollout_lock = RLock()
_CACHE_SIZE = 128
_CHUNK_SIZE = 256 * 1024
_MAX_LINE = 4 * 1024 * 1024


def _integer(value: object, previous: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return previous


def _apply_record(state: _RolloutState, line: bytes) -> None:
    try:
        record = json.loads(line)
    except (ValueError, UnicodeError):
        return
    if not isinstance(record, dict):
        return
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return
    kind = record.get("type")
    if kind == "session_meta":
        state.meta = payload
    elif kind == "turn_context" and isinstance(payload.get("model"), str):
        state.model = payload["model"]
    elif kind == "event_msg":
        event = payload.get("type")
        if event in ("task_started", "turn_started"):
            state.pending_input.clear()
            state.status = SessionStatus.GENERATING
        elif event in (
            "task_complete",
            "turn_complete",
            "turn_completed",
            "turn_aborted",
            "task_aborted",
        ):
            state.pending_input.clear()
            state.status = SessionStatus.IDLE
        elif event == "token_count":
            info = payload.get("info")
            if isinstance(info, dict):
                total = info.get("total_token_usage")
                if isinstance(total, dict):
                    state.total_tokens = _integer(total.get("total_tokens"), state.total_tokens)
                last = info.get("last_token_usage")
                if isinstance(last, dict):
                    # Input includes cache; output includes reasoning. Count each once.
                    fallback = state.context_used
                    if isinstance(last.get("input_tokens"), int):
                        fallback = _integer(last.get("input_tokens"), 0) + _integer(
                            last.get("output_tokens"), 0
                        )
                    state.context_used = _integer(last.get("total_tokens"), fallback)
                state.context_window = _integer(
                    info.get("model_context_window"), state.context_window
                )
    elif kind == "response_item":
        call_id = payload.get("call_id")
        if isinstance(call_id, str):
            if payload.get("type") == "function_call" and payload.get("name") in (
                "request_user_input",
                "functions.request_user_input",
            ):
                state.pending_input.add(call_id)
                state.status = SessionStatus.WAITING_PERMISSION
            elif payload.get("type") == "function_call_output" and call_id in state.pending_input:
                state.pending_input.remove(call_id)
                if not state.pending_input:
                    state.status = SessionStatus.GENERATING
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str):
        try:
            state.last_activity = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            pass


def _rollout_state(path: Path) -> _RolloutState:
    """Cache parsed state, retain partial appends, and reset on rotation/truncation."""
    with _rollout_lock:
        try:
            stat = path.stat()
        except OSError:
            _rollout_cache.pop(path, None)
            return _RolloutState()
        identity = (stat.st_dev, stat.st_ino)
        state = _rollout_cache.get(path)
        if (
            state is None
            or state.identity != identity
            or stat.st_size < state.offset
            or (stat.st_size == state.offset and stat.st_mtime_ns != state.mtime_ns)
        ):
            state = _RolloutState(identity=identity)
        _rollout_cache[path] = state
        _rollout_cache.move_to_end(path)
        while len(_rollout_cache) > _CACHE_SIZE:
            _rollout_cache.popitem(last=False)
        try:
            with path.open("rb") as handle:
                handle.seek(state.offset)
                while chunk := handle.read(_CHUNK_SIZE):
                    state.offset += len(chunk)
                    pieces = (state.partial + chunk).split(b"\n")
                    state.partial = pieces.pop()
                    for line in pieces:
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


def _parse_rollout(
    path: Path,
) -> tuple[dict | None, int, int, int, str | None, datetime | None]:
    """Return the established public metadata/token tuple."""
    with _rollout_lock:
        state = _rollout_state(path)
        return (
            state.meta,
            state.total_tokens,
            state.context_used,
            state.context_window,
            state.model,
            state.last_activity,
        )


def discover_sessions(config_dir: Path | None = None) -> list[CodexSession]:
    """Actual open Codex threads for one account, never newest-cwd history."""
    with _rollout_lock:
        config_dir = (config_dir or CODEX_DIR).expanduser().resolve()
        root = config_dir / "sessions"
        sessions: dict[str, CodexSession] = {}
        names: dict[str, str] | None = None
        active_paths: set[Path] = set()
        for pid, command in _codex_processes():
            rollouts = [
                path for path in _open_rollouts(pid, root) if path.resolve().is_relative_to(root)
            ]
            interactive = _interactive_cli(command)
            process_root = _process_config_dir(pid)
            owns_config = process_root == config_dir or (
                process_root is None and config_dir == CODEX_DIR.resolve()
            )
            if not rollouts and interactive and owns_config:
                match = re.search(rf"\bresume\s+({_UUID})(?:\s|$)", command)
                rollout = _rollout_for_id(match.group(1), config_dir) if match else None
                if rollout:
                    rollouts = [rollout]
            if not rollouts and (not interactive or not owns_config):
                continue
            if names is None:
                names = _index_names(config_dir)
            active_paths.update(rollouts)
            candidates: list[Path | None] = list(rollouts) if rollouts else [None]
            for path in candidates:
                state = _rollout_state(path) if path else _RolloutState()
                meta = state.meta or {}
                file_id = _ROLLOUT.fullmatch(path.name) if path else None
                session_id = meta.get("id") if isinstance(meta.get("id"), str) else None
                session_id = session_id or (file_id.group(1) if file_id else f"pid-{pid}")
                if session_id in sessions:
                    continue
                cwd = meta.get("cwd") if isinstance(meta.get("cwd"), str) else None
                cwd = cwd or _lsof_cwd(pid) or ""
                name = names.get(session_id) or (Path(cwd).name if cwd else "") or session_id[:8]
                sessions[session_id] = CodexSession(
                    pid,
                    session_id,
                    name,
                    cwd,
                    state.model,
                    state.total_tokens,
                    state.context_used,
                    state.context_window,
                    state.last_activity,
                    state.status,
                )
        for path in list(_rollout_cache):
            if path.is_relative_to(root) and path not in active_paths:
                del _rollout_cache[path]
        return list(sessions.values())
