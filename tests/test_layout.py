"""Dashboard space follows sessions, including at small terminal sizes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from textual.geometry import Region
from textual.widgets import DataTable, Static

from cctop.app import CctopApp
from cctop.collect import Account
from cctop.models import (
    AccountLimits,
    ContextWindow,
    LimitWindow,
    Session,
    SessionState,
    UsageTotals,
)
from cctop.stats import FleetStats
from cctop.stats_screen import StatsScreen
from cctop.status import SessionStatus

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


def _dashboard_fixture(root: Path):
    accounts = [
        Account("alpha", root / "claude"),
        Account("beta", root / "codex", "codex"),
        Account("remote-work", root / "omp", "omp"),
    ]
    windows = [
        LimitWindow("session", "5h", 40, NOW, "normal", True),
        LimitWindow("weekly_all", "week", 65, NOW, "normal", False),
    ]
    limits = [
        AccountLimits("alpha", "max", windows, "api", NOW),
        AccountLimits("beta", "pro", windows, "api", NOW),
        AccountLimits(
            "remote-work",
            None,
            [],
            "none",
            None,
            error="OMP Codex sessions; account quota unavailable",
        ),
    ]
    states = []
    for index in range(8):
        session = Session(
            1000 + index,
            f"session-{index}",
            "/work/项目/" + "very-long-directory/" * 15,
            f"Task {index} " + "项目任务🚀" * 30,
            "",
            "omp",
            "3",
            NOW,
            NOW,
            NOW,
        )
        states.append(
            SessionState(
                session,
                "remote-work",
                True,
                SessionStatus.GENERATING,
                "gpt-6.1-sol",
                ContextWindow(100000, 400000),
                UsageTotals(2000, 1000, 0, 8000, None),
                NOW,
            )
        )
    return accounts, limits, states


def _disable_services(monkeypatch):
    from cctop import stats_screen

    monkeypatch.setattr(CctopApp, "_tick_sessions", lambda self: None)
    monkeypatch.setattr(CctopApp, "_tick_limits", lambda self: None)
    monkeypatch.setattr(CctopApp, "action_refresh_limits", lambda self: None)
    monkeypatch.setattr(
        stats_screen,
        "build_provider_stats",
        lambda accounts: [("Claude", FleetStats()), ("Codex", FleetStats())],
    )


def _visible_text(widget) -> str:
    return "\n".join(
        strip.text
        for strip in widget.render_lines(Region(0, 0, widget.size.width, widget.size.height))
    )


def test_sessions_remain_readable_through_resize_selection_and_statistics(tmp_path, monkeypatch):
    _disable_services(monkeypatch)
    accounts, limits, states = _dashboard_fixture(tmp_path)

    async def check():
        app = CctopApp(accounts)
        app.monitor.limits = limits
        async with app.run_test(size=(120, 40)) as pilot:
            app._render_limits(limits, NOW)
            app._render_sessions(states, NOW)
            await pilot.pause()
            table = app.query_one("#sessions", DataTable)
            table.focus()
            table.move_cursor(row=2)
            await pilot.pause()
            selected = states[2].session.session_id
            assert app._selected_session_id == selected
            for width, height in ((120, 40), (80, 24)):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                sessions = app.query_one("#sessions-box")
                stats = app.query_one("#stats-box")
                detail = app.query_one("#detail", Static)
                assert sessions.region.height >= stats.region.height * 2
                assert table.region.height >= 5
                assert table.max_scroll_x == 0
                assert detail.region.height <= 4
                assert states[2].account in _visible_text(detail)
                assert detail.region.bottom < sessions.region.bottom
                assert (
                    len([line for line in _visible_text(detail).splitlines() if line.strip()]) >= 2
                )
                assert app._selected_session_id == selected
                assert table.get_row_at(table.cursor_row) == table.get_row(selected)
            await pilot.press("s")
            assert isinstance(app.screen, StatsScreen)
            await pilot.press("escape")
            assert not isinstance(app.screen, StatsScreen)
            assert app._selected_session_id == selected

    asyncio.run(check())


def test_quota_cards_share_rows_and_source_only_account_note_stays_compact(tmp_path, monkeypatch):
    _disable_services(monkeypatch)
    accounts, limits, states = _dashboard_fixture(tmp_path)

    async def check():
        app = CctopApp(accounts)
        app.monitor.limits = limits
        async with app.run_test(size=(120, 40)) as pilot:
            app._render_sessions(states, NOW)
            for width, height in ((120, 40), (80, 24)):
                await pilot.resize_terminal(width, height)
                app._render_limits(limits, NOW)
                await pilot.pause()
                usage = app.query_one("#usage", Static)
                lines = _visible_text(usage).splitlines()
                headers = [line for line in lines if "alpha" in line and "beta" in line]
                assert len(headers) == 1
                offset = headers[0].index("beta") - headers[0].index("alpha")
                assert abs(offset - usage.size.width / 2) <= 5
                assert sum("remote-work" in line for line in lines) == 1
                assert usage.region.height <= 5
            error_limits = [
                limits[0],
                AccountLimits("beta", None, [], "none", None, error="needs /login"),
                limits[2],
            ]
            app._render_limits(error_limits, NOW)
            await pilot.pause()
            assert "needs /login" in _visible_text(app.query_one("#usage", Static))

    asyncio.run(check())
