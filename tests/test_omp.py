"""OMP-hosted Codex sessions use their own files, accounting, and lifecycle."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from rich.console import Console

from cctop import collect
from cctop.cli import _print_view, _session_status_label
from cctop.collect import Account
from cctop.monitor import FleetMonitor
from cctop.status import SessionStatus

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def _line(kind: str, **fields) -> str:
    return json.dumps({"type": kind, "timestamp": "2020-01-01T00:00:00Z", **fields}) + "\n"


def _message(provider="openai-codex", model="gpt-6.1-sol", stop="stop", **fields) -> str:
    return _line(
        "message",
        message={
            "role": "assistant",
            "provider": provider,
            "model": model,
            "stopReason": stop,
            **fields,
        },
    )


def _session(root: Path, name: str, events: str = "") -> Path:
    path = root / "sessions" / "--work-shared--" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _line("title", title=name)
        + _line("session", version=3, id="shared-session-id", cwd="/work/shared")
        + events
    )
    return path


def _process(monkeypatch, paths):
    from cctop import omp

    monkeypatch.setattr(omp, "_omp_processes", lambda: [42])
    monkeypatch.setattr(
        omp,
        "_open_sessions",
        lambda pid, root: [path for path in paths if path.is_relative_to(root)],
    )
    return omp


def test_discover_accounts_includes_omp_agent_sessions(tmp_path, monkeypatch) -> None:
    root = tmp_path / ".omp" / "agent"
    (root / "sessions").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert Account("omp", root, "omp") in collect.discover_accounts()


def test_live_files_keep_children_separate_and_use_latest_provider(tmp_path, monkeypatch) -> None:
    parent = _session(tmp_path, "parent", _message())
    child = _session(tmp_path, "child", _message())
    switched_away = _session(
        tmp_path,
        "switched-away",
        _message() + _line("model_change", model="anthropic/claude-opus", role="default"),
    )
    switched_back = _session(
        tmp_path,
        "switched-back",
        _message("anthropic", "claude-opus")
        + _line("model_change", model="openai-codex/gpt-6.1-sol", role="default"),
    )
    _session(tmp_path, "closed-history", _message())
    foreign = _session(tmp_path / "other-account", "foreign", _message())
    omp = _process(monkeypatch, [parent, child, switched_away, switched_back, foreign])

    sessions = omp.discover_sessions(tmp_path)

    assert {session.name for session in sessions} == {"parent", "child", "switched-back"}
    assert len({session.session_id for session in sessions}) == 3
    assert all(session.session_id.startswith("omp:") for session in sessions)
    assert {session.pid for session in sessions} == {42}
    assert {session.model for session in sessions} == {"gpt-6.1-sol"}


def test_monitor_routes_omp_without_native_quota_or_auth_and_renders(tmp_path, monkeypatch) -> None:
    path = _session(
        tmp_path,
        "current",
        _message(
            stop="toolUse",
            content=[
                {
                    "type": "toolCall",
                    "id": "call-one",
                    "name": "bash",
                    "arguments": {"command": "true"},
                }
            ],
        ),
    )
    _process(monkeypatch, [path])
    account = Account("omp", tmp_path, "omp")

    def forbidden(*args, **kwargs):
        raise AssertionError("OMP must not query another runtime's quota or credentials")

    from cctop import authctl, codex_usage, usage

    monkeypatch.setattr(authctl, "ensure_fresh", forbidden)
    monkeypatch.setattr(authctl, "read_expiry", forbidden)
    monkeypatch.setattr(codex_usage, "fetch_account_limits", forbidden)
    monkeypatch.setattr(usage, "fetch_account_limits", forbidden)
    monitor = FleetMonitor([account])
    limits = monitor.poll_limits(NOW)
    snapshot = monitor.snapshot(NOW)

    assert len(limits) == 1 and limits[0].source == "none" and not limits[0].windows
    assert limits[0].error
    assert len(snapshot.sessions) == 1
    state = snapshot.sessions[0]
    assert state.session.kind == "omp" and state.account == "omp"
    assert state.status is SessionStatus.GENERATING
    assert _session_status_label(state) == "working"
    output = StringIO()
    _print_view(snapshot, Console(file=output, width=140, color_system=None))
    assert "working" in output.getvalue() and "current" in output.getvalue()
    from textual.widgets import DataTable

    from cctop.app import CctopApp

    # Exercise the actual TUI renderer without starting its background pollers.
    monkeypatch.setattr(CctopApp, "_tick_sessions", lambda self: None)
    monkeypatch.setattr(CctopApp, "_tick_stats", lambda self: None)
    monkeypatch.setattr(CctopApp, "action_refresh_limits", lambda self: None)

    async def render_table():
        app = CctopApp([account])
        async with app.run_test():
            app._render_sessions(snapshot.sessions, NOW)
            table = app.query_one("#sessions", DataTable)
            assert "status" not in {key.value for key in table.columns}
            return table.get_cell(state.session.session_id, "name").plain

    assert asyncio.run(render_table()) == "current"
    with path.open("a") as handle:
        handle.write(_message(stop="stop"))
    assert monitor.snapshot(NOW + timedelta(seconds=6)).sessions[0].status is SessionStatus.IDLE


def test_message_accounting_and_unknown_context_window(tmp_path, monkeypatch) -> None:
    path = _session(
        tmp_path,
        "usage",
        _message(
            usage={
                "input": 10,
                "output": 3,
                "cacheRead": 20,
                "cacheWrite": 2,
                "totalTokens": 35,
                "reasoningTokens": 1,
            },
            contextSnapshot={"promptTokens": 72, "nonMessageTokens": 5, "compactionEpoch": 0},
        )
        + _message(
            usage={"input": 4, "output": 2, "cacheRead": 6, "cacheWrite": 0, "totalTokens": 12},
            contextSnapshot={"promptTokens": 99, "nonMessageTokens": 5, "compactionEpoch": 0},
        ),
    )
    omp = _process(monkeypatch, [path])

    session = omp.discover_sessions(tmp_path)[0]
    state = collect.build_snapshot(
        [Account("omp", tmp_path, "omp")], NOW, with_limits=False
    ).sessions[0]

    assert session.total_tokens == 47
    assert session.context_used == 99
    assert state.totals.total_tokens == 47
    assert state.context is None
    with sqlite3.connect(tmp_path / "models.db") as database:
        database.execute(
            "CREATE TABLE model_cache "
            "(provider_id TEXT PRIMARY KEY, models TEXT, updated_at INTEGER)"
        )
        database.execute(
            "INSERT INTO model_cache VALUES (?, ?, ?)",
            (
                "openai-codex:0.159.0",
                json.dumps(
                    [{"id": "gpt-6.1-sol", "provider": "openai-codex", "contextWindow": 400000}]
                ),
                100,
            ),
        )
    with path.open("a") as handle:
        handle.write(
            _message(
                usage={"input": 7, "cacheRead": 8, "cacheWrite": 1, "output": 2, "totalTokens": 18},
                contextSnapshot={"promptTokens": 100, "historyRewriteTokensRemoved": 7},
            )
        )
    context = (
        collect.build_snapshot([Account("omp", tmp_path, "omp")], NOW, with_limits=False)
        .sessions[0]
        .context
    )
    assert context is not None and context.used_tokens == 93 and context.window_tokens == 400000
    with closing(sqlite3.connect(tmp_path / "models.db")) as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA wal_autocheckpoint=0")
        assert omp.discover_sessions(tmp_path)[0].context_window == 400000
        before = (tmp_path / "models.db").stat()
        database.execute(
            "UPDATE model_cache SET models = ?, updated_at = ?",
            (
                json.dumps(
                    [{"id": "gpt-6.1-sol", "provider": "openai-codex", "contextWindow": 420000}]
                ),
                101,
            ),
        )
        database.commit()
        after = (tmp_path / "models.db").stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        assert omp.discover_sessions(tmp_path)[0].context_window == 420000
    with path.open("a") as handle:
        handle.write(
            _message(
                usage={"input": 7, "cacheRead": 8, "cacheWrite": 1, "output": 2, "totalTokens": 18}
            )
        )
    assert omp.discover_sessions(tmp_path)[0].context_used == 16


def test_pending_tool_lifecycle_and_malformed_partial_records(tmp_path, monkeypatch) -> None:
    path = _session(
        tmp_path,
        "lifecycle",
        _message(
            stop="toolUse",
            content=[{"type": "toolCall", "id": "one", "name": "bash", "arguments": {}}],
        ),
    )
    omp = _process(monkeypatch, [path])

    def status():
        return omp.discover_sessions(tmp_path)[0].status

    assert status() is SessionStatus.GENERATING
    with path.open("a") as handle:
        handle.write("null\n[]\n{malformed}\n" + _line("message", message=None))
        handle.write(_line("message", message={"role": "toolResult", "toolCallId": "other"}))
    assert status() is SessionStatus.GENERATING
    completed = _line("message", message={"role": "toolResult", "toolCallId": "one"})
    with path.open("a") as handle:
        handle.write(completed[:25])
    assert status() is SessionStatus.GENERATING
    with path.open("a") as handle:
        handle.write(completed[25:])
    assert status() is SessionStatus.UNKNOWN
    with path.open("a") as handle:
        handle.write(_message(stop="aborted"))
    assert status() is SessionStatus.IDLE
    with path.open("a") as handle:
        handle.write(_line("message", message={"role": "user", "content": []}))
    assert status() is SessionStatus.UNKNOWN

    with path.open("a") as handle:
        handle.write(
            _message(
                stop="toolUse",
                content=[{"type": "toolCall", "id": "question", "name": "ask", "arguments": {}}],
            )
        )
        handle.write(
            _line(
                "custom",
                customType="tool_execution_start",
                data={
                    "toolCallId": "question",
                    "toolName": "ask",
                    "startedAt": "2020-01-01T00:00:00Z",
                },
            )
        )
    assert status() is SessionStatus.WAITING_PERMISSION
    state = collect.build_snapshot(
        [Account("omp", tmp_path, "omp")], NOW, with_limits=False
    ).sessions[0]
    assert _session_status_label(state) == "waiting"
    with path.open("a") as handle:
        handle.write(_line("message", message={"role": "toolResult", "toolCallId": "question"}))
    assert status() is SessionStatus.UNKNOWN
