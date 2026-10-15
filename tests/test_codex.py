"""Live Codex discovery and rollout contracts, using generated session files."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cctop import codex
from cctop.collect import Account, codex_session_states
from cctop.status import SessionStatus

FIRST = "11111111-1111-4111-8111-111111111111"
SECOND = "22222222-2222-4222-8222-222222222222"
HISTORY = "33333333-3333-4333-8333-333333333333"
OLD = "2020-01-01T00:00:00Z"
NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def _record(kind: str, payload, timestamp: str = OLD) -> str:
    return json.dumps({"type": kind, "payload": payload, "timestamp": timestamp}) + "\n"


def _event(kind: str, **payload) -> str:
    return _record("event_msg", {"type": kind, **payload})


def _rollout(root: Path, session_id: str, events: str = "") -> Path:
    path = root / "sessions" / "2026" / "10" / "15" / f"rollout-test-{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _record(
            "session_meta", {"id": session_id, "cwd": "/work/shared", "model_provider": "openai"}
        )
        + events
    )
    return path


def _processes(monkeypatch, root: Path, processes, opened, roots=None) -> None:
    """Replace only OS process/FD inspection; discovery and parsing stay real."""
    monkeypatch.setattr(codex, "CODEX_DIR", root)
    monkeypatch.setattr(codex, "_codex_processes", lambda: processes)
    monkeypatch.setattr(codex, "_lsof_cwd", lambda pid: "/work/shared")
    monkeypatch.setattr(
        codex,
        "_open_rollouts",
        lambda pid, config_dir: [
            path for path in opened.get(pid, []) if path.is_relative_to(config_dir)
        ],
        raising=False,
    )
    monkeypatch.setattr(
        codex, "_process_config_dir", lambda pid: (roots or {}).get(pid, root), raising=False
    )


def test_app_server_reports_each_open_rollout_without_cwd_history(tmp_path, monkeypatch) -> None:
    first = _rollout(tmp_path, FIRST, _event("task_started", turn_id="one"))
    second = _rollout(tmp_path, SECOND, _event("task_complete", turn_id="two"))
    _rollout(tmp_path, HISTORY, _event("task_complete", turn_id="old"))
    (tmp_path / "session_index.jsonl").write_text(
        json.dumps({"id": FIRST, "thread_name": "First task"}) + "\n"
    )
    _processes(monkeypatch, tmp_path, [(42, "/usr/bin/codex app-server")], {42: [first, second]})

    sessions = {session.session_id: session for session in codex.discover_sessions()}

    assert set(sessions) == {FIRST, SECOND}
    assert {session.pid for session in sessions.values()} == {42}
    assert sessions[FIRST].name == "First task"
    assert sessions[FIRST].status is SessionStatus.GENERATING
    assert sessions[SECOND].status is SessionStatus.IDLE


def test_cli_resume_keeps_exact_id_and_deduplicates_wrapper(tmp_path, monkeypatch) -> None:
    _rollout(tmp_path, FIRST, _event("task_started", turn_id="one"))
    _rollout(tmp_path, HISTORY, _event("task_complete", turn_id="old"))
    _processes(
        monkeypatch,
        tmp_path,
        [(41, f"node /usr/bin/codex resume {FIRST}"), (42, f"/usr/bin/codex resume {FIRST}")],
        {},
    )

    sessions = codex.discover_sessions()

    assert [session.session_id for session in sessions] == [FIRST]


def test_account_roots_keep_sessions_and_collector_status_isolated(tmp_path, monkeypatch) -> None:
    primary, other = tmp_path / "primary", tmp_path / "other"
    first = _rollout(primary, FIRST, _event("task_started", turn_id="one"))
    second = _rollout(other, SECOND, _event("task_complete", turn_id="two"))
    _processes(
        monkeypatch,
        primary,
        [(41, "/usr/bin/codex app-server"), (42, "/usr/bin/codex app-server")],
        {41: [first], 42: [second]},
        {41: primary, 42: other},
    )

    active = codex_session_states(Account("primary", primary, "codex"), NOW)
    idle = codex_session_states(Account("other", other, "codex"), NOW)

    assert [(state.account, state.session.session_id, state.status) for state in active] == [
        ("primary", FIRST, SessionStatus.GENERATING)
    ]
    assert [(state.account, state.session.session_id, state.status) for state in idle] == [
        ("other", SECOND, SessionStatus.IDLE)
    ]


def test_task_lifecycle_uses_events_instead_of_timestamp_age(tmp_path, monkeypatch) -> None:
    path = _rollout(tmp_path, FIRST, _event("task_started", turn_id="one"))
    _processes(monkeypatch, tmp_path, [(42, f"/usr/bin/codex resume {FIRST}")], {42: [path]})
    account = Account("codex", tmp_path, "codex")

    def status():
        return codex_session_states(account, NOW)[0].status

    assert status() is SessionStatus.GENERATING
    with path.open("a") as handle:
        handle.write(
            _record("event_msg", {"type": "task_complete", "turn_id": "one"}, NOW.isoformat())
        )
    assert status() is SessionStatus.IDLE
    with path.open("a") as handle:
        handle.write(_event("task_started", turn_id="two"))
        handle.write(
            _record("event_msg", {"type": "turn_aborted", "turn_id": "two"}, NOW.isoformat())
        )
    assert status() is SessionStatus.IDLE


def test_only_unanswered_user_input_call_means_waiting(tmp_path, monkeypatch) -> None:
    path = _rollout(tmp_path, FIRST, _event("task_started", turn_id="one"))
    _processes(monkeypatch, tmp_path, [(42, f"/usr/bin/codex resume {FIRST}")], {42: [path]})

    def append(payload):
        with path.open("a") as handle:
            handle.write(_record("response_item", payload))
        return codex.discover_sessions()[0].status

    assert (
        append({"type": "function_call", "name": "exec_command", "call_id": "shell"})
        is SessionStatus.GENERATING
    )
    assert (
        append({"type": "function_call", "name": "request_user_input", "call_id": "question"})
        is SessionStatus.WAITING_PERMISSION
    )
    assert (
        append({"type": "function_call_output", "call_id": "shell", "output": "done"})
        is SessionStatus.WAITING_PERMISSION
    )
    assert (
        append({"type": "function_call_output", "call_id": "question", "output": "answer"})
        is SessionStatus.GENERATING
    )


def test_turn_model_and_token_usage_do_not_double_count_cached_input(tmp_path, monkeypatch) -> None:
    path = _rollout(
        tmp_path,
        FIRST,
        _record("turn_context", {"model": "gpt-5.3-codex"})
        + _event(
            "token_count",
            info={
                "total_token_usage": {
                    "input_tokens": 1000,
                    "cached_input_tokens": 800,
                    "output_tokens": 50,
                    "total_tokens": 1050,
                },
                "last_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 5,
                    "total_tokens": 105,
                },
                "model_context_window": 400000,
            },
        ),
    )
    _processes(monkeypatch, tmp_path, [(42, f"/usr/bin/codex resume {FIRST}")], {42: [path]})

    session = codex.discover_sessions()[0]

    assert session.model == "gpt-5.3-codex"
    assert session.total_tokens == 1050
    assert session.context_used == 105
    assert session.context_window == 400000
    path.write_text(
        _record("session_meta", {"id": FIRST, "cwd": "/work/shared", "model_provider": "openai"})
    )
    assert codex.discover_sessions()[0].model is None


def test_malformed_records_partial_append_and_truncation_keep_valid_state(
    tmp_path, monkeypatch
) -> None:
    path = _rollout(tmp_path, FIRST, _event("task_started", turn_id="one"))
    with path.open("a") as handle:
        handle.write('null\n[]\n{"type":"event_msg","payload":null}\n')
        handle.write('{"type":"event_msg","payload":{"type":"token_count","info":[]}}\n')
        handle.write("{malformed}\n")
        handle.write(_event("task_complete", turn_id="one")[:35])
    _processes(monkeypatch, tmp_path, [(42, f"/usr/bin/codex resume {FIRST}")], {42: [path]})

    assert codex.discover_sessions()[0].status is SessionStatus.GENERATING
    with path.open("a") as handle:
        handle.write(_event("task_complete", turn_id="one")[35:])
    assert codex.discover_sessions()[0].status is SessionStatus.IDLE
    path.write_text(
        _record("session_meta", {"id": FIRST, "cwd": "/work/shared"})
        + _event("task_started", turn_id="two")
    )
    assert codex.discover_sessions()[0].status is SessionStatus.GENERATING


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process FD boundary")
def test_open_rollouts_require_writable_handles_and_selected_root(tmp_path) -> None:
    current = _rollout(tmp_path / "account", FIRST)
    history = _rollout(tmp_path / "account", HISTORY)
    other = _rollout(tmp_path / "other", SECOND)

    with current.open("r+"), history.open("r"), other.open("a"):
        assert codex._open_rollouts(os.getpid(), tmp_path / "account" / "sessions") == [current]


def test_macos_process_identity_preserves_executable_paths_with_spaces(monkeypatch) -> None:
    native = "/Applications/ChatGPT Atlas.app/Contents/Resources/codex"
    uid = os.getuid()
    command = f"'{native}' app-server"

    def run(args):
        if args == ["ps", "-Ao", "pid=,uid=,comm="]:
            return (
                f"42 {uid} {native}\n"
                f"43 {uid} /usr/bin/python\n"
                f"44 {uid + 1} /usr/bin/codex\n"
                f"45 {uid} /usr/bin/codex-code-mode-host\n"
            )
        if args == ["ps", "-p", "42", "-o", "pid=,args="]:
            return f"42 {command}\n"
        # macOS truncates comm when followed by an args column.
        if args == ["ps", "-Ao", "pid=,uid=,comm=,args="]:
            return f"42 {uid} /Applications/Ch {command}\n43 {uid} python codex.py\n"
        raise AssertionError(f"Unexpected process inspection: {args}")

    monkeypatch.setattr(codex.sys, "platform", "darwin")
    monkeypatch.setattr(codex, "_run", run)

    assert codex._codex_processes() == [(42, command)]
