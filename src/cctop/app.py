"""The live Textual TUI.

Layout: a prominent USAGE panel at the top (the two accounts side by side, each
window a wide bar), the SESSIONS table below as the denser ledger, and a footer
with fleet totals, the limits-fetch age, and key hints.

Palette: teal accent (#20B2AA) on bar fills, section borders, and the active-
window marker; monochrome terminal otherwise. Usage bars stay teal even when
maxed (fullness carries the signal, not color); red is reserved for a blocked
or dead session. This mirrors the house teal-on-mono look (to be matched
exactly to sleuth-tui later).

Cost safety: the session table refreshes every second from local files only
(free), while the usage-limits panel is refetched on a slower throttle (GET
/api/oauth/usage, also free) with a visible "N ago" stamp and a key to refresh
now. Both polls run in thread workers so neither blocks the UI.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from rich.console import Group
from rich.table import Table as RichTable
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.timer import Timer
from textual.widgets import DataTable, Rule, Static

from .cli import (
    _format_age,
    _format_context,
    _format_cost,
    _format_model,
    _format_reset,
    _format_tokens,
    _session_status_label,
    _short_cwd,
    _tier_label,
)
from .collect import Account
from .models import AccountLimits, LimitWindow, SessionState
from .monitor import FleetMonitor
from .stats import FleetStats

TEAL = "#20B2AA"
DIM = "grey37"
MUTED = "grey50"
BAR_WIDTH = 22

_STATUS_STYLE = {
    "idle": TEAL,
    "shell": "white",
    "generating": TEAL,
    "waiting_permission": "red",
    "stale": MUTED,
    "dead": "red",
    "unknown": MUTED,
}

_COLUMNS = ("acct", "name", "status", "model", "cwd", "ctx", "tokens", "cost", "age")


def _fit(text: str, width: int) -> str:
    """Fit terminal cells, including wide Unicode characters."""
    rendered = Text(text)
    rendered.truncate(max(0, width), overflow="ellipsis", pad=True)
    return rendered.plain


def _cell(value: str, width: int, style: str = "") -> Text:
    rendered = Text(value, style=style, no_wrap=True, overflow="ellipsis")
    rendered.truncate(max(0, width), overflow="ellipsis")
    return rendered


def _bar(fraction: float, width: int) -> Text:
    """A teal fill over a grey track. Fullness carries the signal, not color, so
    a maxed window reads as a full teal bar rather than switching to red."""
    filled = max(0, min(width, int(round(fraction * width))))
    bar = Text()
    bar.append("█" * filled, style=TEAL)
    bar.append("░" * (width - filled), style=DIM)
    return bar


def _gauge_line(window: LimitWindow, now: datetime, bar_width: int, label_width: int) -> Text:
    line = Text()
    line.append(_fit(window.label, label_width) + " ", style="default")
    if not window.has_data:
        # A window with no reading yet: an empty track and a muted dash, so it
        # reads as "not started" rather than a real 0%.
        line.append("░" * bar_width, style=DIM)
        line.append("   —", style=MUTED)
        return line
    line.append(_bar(window.used_fraction, bar_width))
    line.append(f" {window.percent:>3.0f}%", style=TEAL)
    line.append(" ●" if window.is_active else "  ", style=TEAL)
    line.append(f" {_format_reset(window.resets_at, now)}", style=MUTED)
    return line


def _account_block(
    account: AccountLimits, now: datetime, bar_width: int, label_width: int
) -> Group:
    header = Text()
    header.append(account.account, style=f"bold {TEAL}")
    tier = _tier_label(account.tier)
    if tier:
        header.append("   ", style="default")
        header.append(tier, style=MUTED)

    lines = [header]
    if account.source != "api":
        lines.append(Text(account.error or "no limit data", style=MUTED))
    else:
        lines.extend(_gauge_line(window, now, bar_width, label_width) for window in account.windows)
    return Group(*lines)


def _render_usage(
    limits: list[AccountLimits],
    now: datetime,
    width: int | None = None,
    source_only_names: set[str] | None = None,
) -> RichTable:
    """Equal-width quota cards with a separate compact source-only notice."""
    width = max(20, width or 120)
    source_only_names = source_only_names or set()
    sources = [
        account
        for account in limits
        if account.account in source_only_names
        and account.source == "none"
        and account.error == "OMP Codex sessions; account quota unavailable"
    ]
    quotas = [account for account in limits if account not in sources]
    count = len(quotas)
    per_row = max(1, min(count, width // 34))
    gap = 3
    column = max(1, (width - (per_row - 1) * gap) // per_row)
    label_width = min(11, max(6, column // 4))
    bar_width = max(4, min(BAR_WIDTH, column - label_width - 17))
    cards = RichTable.grid(padding=(0, gap), expand=True)
    for _ in range(per_row):
        cards.add_column(ratio=1, width=column)
    for start in range(0, count, per_row):
        chunk = quotas[start : start + per_row]
        cells: list = [_account_block(account, now, bar_width, label_width) for account in chunk]
        cells += [Text("")] * (per_row - len(cells))
        cards.add_row(*cells)
    grid = RichTable.grid(expand=True)
    grid.add_column(ratio=1)
    if quotas:
        grid.add_row(cards)
    if sources:
        names = ", ".join(account.account for account in sources)
        note = Text.assemble((names, TEAL), (" · OMP sessions only; quota unavailable", MUTED))
        note.no_wrap = True
        note.overflow = "ellipsis"
        grid.add_row(note)
    return grid


def _format_next_refresh(fetched: datetime | None, interval: float, now: datetime) -> str:
    """Countdown to the next usage refresh, e.g. "next in 2m14s" or "refreshing"."""
    if fetched is None:
        return "refreshing"
    remaining = interval - (now - fetched).total_seconds()
    if remaining <= 1:
        return "refreshing"
    if remaining < 60:
        return f"next in {int(remaining)}s"
    minutes, seconds = divmod(int(remaining), 60)
    return f"next in {minutes}m{seconds:02d}s"


class CctopApp(App):
    """nvtop-style live view of Claude Code sessions and usage limits."""

    # Pin the terminal tab title so it doesn't flicker as the view refreshes.
    TITLE = "cctop"
    SUB_TITLE = ""

    CSS = f"""
    Screen {{ background: $background; }}
    #topbar {{ height: 1; padding: 0 1; }}
    #usage-box {{
        border: round {TEAL};
        border-title-color: {TEAL};
        height: auto;
        padding: 0 1;
    }}
    #sessions-box {{
        border: round {TEAL};
        border-title-color: {TEAL};
        height: 1fr;
        padding: 0 1;
    }}
    #sessions-box DataTable {{ height: 1fr; }}
    #sessions-box Rule {{ color: {TEAL} 30%; margin: 0; height: 1; }}
    #detail {{ height: 3; }}
    #stats-box {{
        border: round {TEAL};
        border-title-color: {TEAL};
        height: auto;
        padding: 0 1;
    }}
    #footer {{ height: 1; padding: 0 1; color: #808080; }}
    DataTable {{ background: transparent; }}
    DataTable > .datatable--header {{ color: {TEAL}; text-style: bold; background: transparent; }}
    DataTable > .datatable--cursor {{ background: {TEAL} 25%; }}
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_limits", "Refresh limits"),
        ("R", "refresh_token", "Refresh token"),
        ("a", "add_account", "Add account"),
        ("s", "stats", "Stats"),
        ("slash", "search", "Search"),
        ("comma", "settings", "Settings"),
    ]

    def __init__(
        self,
        accounts: list[Account],
        limits_interval: float = 180.0,
        heatmap_weeks: int = 26,
        auto_refresh_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = FleetMonitor(accounts, auto_refresh_tokens=auto_refresh_tokens)
        self._limits_interval = limits_interval
        self._heatmap_weeks = heatmap_weeks
        self._auto_refresh_tokens = auto_refresh_tokens
        self._limits_timer: Timer | None = None
        self._states_by_id: dict[str, SessionState] = {}
        self._selected_session_id: str | None = None
        self._provider_stats: list[tuple[str, FleetStats]] = []

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with Container(id="usage-box"):
            yield Static(id="usage")
        with Container(id="sessions-box"):
            yield DataTable(id="sessions", cursor_type="row")
            yield Rule(line_style="dashed")
            yield Static(id="detail")
        with Container(id="stats-box"):
            yield Static(id="stats")
        yield Static(id="footer")

    def on_mount(self) -> None:
        self.query_one("#usage-box", Container).border_title = "USAGE"
        self.query_one("#sessions-box", Container).border_title = "SESSIONS"
        self.query_one("#stats-box", Container).border_title = "STATS"

        table = self.query_one("#sessions", DataTable)
        for column in _COLUMNS:
            table.add_column(column, key=column)

        self.query_one("#usage", Static).update(Text("fetching limits...", style=MUTED))
        self.query_one("#stats", Static).update(Text("loading stats...", style=MUTED))
        self.set_interval(1.0, self._tick_sessions)
        self._limits_timer = self.set_interval(self._limits_interval, self._tick_limits)
        self.set_interval(300.0, self._tick_stats)
        self._tick_sessions()
        self.action_refresh_limits()
        self._tick_stats()

    @work(thread=True, exclusive=True, group="stats")
    def _tick_stats(self) -> None:
        from .stats_screen import build_provider_stats

        provider_stats = build_provider_stats(self.monitor.accounts)
        self.call_from_thread(self._render_stats, provider_stats)

    def _render_stats(self, provider_stats: list[tuple[str, FleetStats]]) -> None:
        self._provider_stats = provider_stats
        lines = []
        width = max(1, self.query_one("#stats", Static).size.width or self.size.width - 4)
        for label, stats in provider_stats:
            if not stats.days and not stats.total_messages:
                summary = (
                    f" · {stats.total_sessions:,} sessions · no daily activity recorded"
                    if stats.total_sessions
                    else " · no activity recorded"
                )
                line = Text.assemble((label, f"bold {TEAL}"), (summary, MUTED))
            else:
                line = Text.assemble(
                    (label, f"bold {TEAL}"),
                    (
                        f"  {stats.total_sessions:,} sessions · {stats.total_messages:,} messages"
                        f" · {_format_tokens(sum(stats.model_tokens.values()))} tokens",
                        "default",
                    ),
                )
            lines.append(_cell(line.plain, width, MUTED) if line.cell_len > width else line)
        content = (
            Group(*lines)
            if lines
            else Text("No activity history · s opens statistics", style=MUTED)
        )
        self.query_one("#stats", Static).update(content)

    # -- session polling (fast, free, file-only) -------------------------------

    @work(thread=True, exclusive=True, group="sessions")
    def _tick_sessions(self) -> None:
        now = datetime.now(timezone.utc)
        states = self.monitor.poll_sessions(now)
        self.call_from_thread(self._render_sessions, states, now)

    def _render_sessions(self, states: list[SessionState], now: datetime) -> None:
        self._render_topbar(now)
        self._states_by_id = {s.session.session_id: s for s in states}

        table = self.query_one("#sessions", DataTable)
        available = max(1, self.size.width - 6)
        columns = list(_COLUMNS if available >= 105 else _COLUMNS[:-2])
        fixed = {"acct": 5, "status": 7, "model": 12, "ctx": 6, "tokens": 7, "cost": 7, "age": 5}
        flexible = max(
            2,
            available
            - 2 * len(columns)
            - sum(fixed.values())
            + (fixed["cost"] + fixed["age"] if "cost" not in columns else 0),
        )
        widths = dict(fixed, name=max(1, flexible * 45 // 100))
        widths["cwd"] = max(1, flexible - widths["name"])
        selected = self._selected_session_id
        if selected not in self._states_by_id:
            selected = states[0].session.session_id if states else None
        with table.prevent(DataTable.RowHighlighted):
            table.clear(columns=True)
            for column in columns:
                table.add_column(column, key=column, width=widths[column])
            selected_index = 0
            for index, state in enumerate(states):
                session_id = state.session.session_id
                style = _STATUS_STYLE.get(state.status.value, "white")
                values = {
                    "acct": state.account,
                    "name": state.session.name or session_id[:8],
                    "status": _session_status_label(state),
                    "model": _format_model(state.model),
                    "cwd": _short_cwd(state.session.cwd),
                    "ctx": _format_context(state),
                    "tokens": _format_tokens(state.totals.total_tokens),
                    "cost": _format_cost(state.totals.cost_usd),
                    "age": _format_age(state.last_activity, now),
                }
                table.add_row(
                    *[
                        _cell(values[column], widths[column], style if column == "status" else "")
                        for column in columns
                    ],
                    key=session_id,
                )
                if session_id == selected:
                    selected_index = index
            if states:
                table.move_cursor(row=selected_index, animate=False)
        self._selected_session_id = selected
        self._render_detail(now)
        self._render_footer(states, now)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._selected_session_id = event.row_key.value
        self._render_detail(datetime.now(timezone.utc))

    def _render_detail(self, now: datetime) -> None:
        selected = self._selected_session_id
        state = self._states_by_id.get(selected) if selected is not None else None
        widget = self.query_one("#detail", Static)
        if state is None:
            widget.update(Text("", style=MUTED))
            widget.tooltip = None
            return

        session = state.session
        totals = state.totals
        ctx = state.context
        width = max(1, self.size.width - 4)
        context = (
            f"{_format_tokens(ctx.used_tokens)}/{_format_tokens(ctx.window_tokens)}"
            if ctx is not None
            else "—"
        )
        summary = (
            f"ctx {context} · {_format_tokens(totals.total_tokens)} tokens · "
            f"{_format_cost(totals.cost_usd)} · {_format_age(state.last_activity, now)} ago"
            f" · pid {session.pid}"
        )
        if width >= 120:
            summary += (
                f" · in {_format_tokens(totals.input_tokens)}"
                f" out {_format_tokens(totals.output_tokens)}"
                f" cache {_format_tokens(totals.cache_read_tokens)}"
                f"/{_format_tokens(totals.cache_creation_tokens)}"
            )
        model = state.model or "unknown model"
        version = f" · v{session.version}" if session.version else ""
        identity = f"{model}{version} · "
        if Text(identity + summary).cell_len <= width:
            summary = identity + summary
        name = session.name or session.session_id[:8]
        widget.tooltip = Text(
            f"{state.account} · {model}{version}\n{name}\n{session.cwd}",
        )
        widget.update(
            Group(
                _cell(f"{state.account} · {name}", width, f"bold {TEAL}"),
                _cell(_short_cwd(session.cwd), width, MUTED),
                _cell(summary, width, MUTED),
            )
        )

    def _render_topbar(self, now: datetime) -> None:
        names = "  ".join(account.name for account in self.monitor.accounts)
        clock = now.astimezone().strftime("%H:%M:%S")
        bar = RichTable.grid(expand=True)
        bar.add_column(justify="left")
        bar.add_column(justify="right")
        left = Text.assemble(
            ("cctop", f"bold {TEAL}"),
            ("     ", "default"),
            (names, MUTED),
        )
        bar.add_row(left, Text(clock, style=MUTED))
        self.query_one("#topbar", Static).update(bar)

    def _render_footer(self, states: list[SessionState], now: datetime) -> None:
        total_tokens = sum(s.totals.total_tokens for s in states)
        total_cost = sum(s.totals.cost_usd or 0.0 for s in states)
        fetched = self.monitor.limits_fetched_at
        age = f"updated {_format_age(fetched, now)} ago" if fetched else "no data yet"
        nxt = _format_next_refresh(fetched, self._limits_interval, now)
        width = max(1, self.size.width - 2)
        if width >= 145:
            hints = "/ search · r refresh · R token · a add · s stats · , settings · q quit"
            summary = (
                f"{len(states)} sessions · {_format_tokens(total_tokens)} tokens · "
                f"${total_cost:,.0f}   usage {age} · {nxt}"
            )
        else:
            hints = "/ search · r refresh · s stats · q quit"
            summary = f"{len(states)} sessions · {_format_tokens(total_tokens)} tokens"
        footer = RichTable.grid(expand=True)
        footer.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
        footer.add_column(justify="right", width=Text(hints).cell_len, no_wrap=True)
        footer.add_row(Text(summary, style=MUTED), Text(hints, style=MUTED))
        self.query_one("#footer", Static).update(footer)

    # -- limits polling (slow, free, networked) --------------------------------

    @work(thread=True, exclusive=True, group="limits")
    def _tick_limits(self) -> None:
        now = datetime.now(timezone.utc)
        limits = self.monitor.poll_limits(now)
        self.call_from_thread(self._render_limits, limits, now)

    def action_stats(self) -> None:
        from .stats_screen import StatsScreen

        self.push_screen(StatsScreen(self.monitor.accounts, date.today()))

    def action_search(self) -> None:
        from .search_screen import SearchScreen

        self.push_screen(SearchScreen(self.monitor.accounts))

    def action_settings(self) -> None:
        from . import config as config_module
        from .collect import discover_accounts
        from .settings_screen import SettingsScreen, build_rows

        rows = build_rows(discover_accounts(), config_module.load_config())
        self.push_screen(
            SettingsScreen(
                self._limits_interval,
                self._heatmap_weeks,
                rows,
                auto_refresh_tokens=self._auto_refresh_tokens,
            )
        )

    def apply_settings(self) -> None:
        """Reload the config file and apply it live (called by the settings screen).

        Updates the refresh interval (rescheduling the timer + countdown), the
        heatmap range, and the visible account set, with no restart.
        """
        from datetime import timedelta

        from . import config as config_module
        from .collect import resolve_accounts

        config = config_module.load_config()
        self._limits_interval = config.limits_refresh_seconds(180.0)
        self._heatmap_weeks = config.heatmap_weeks(26)
        self._auto_refresh_tokens = config.auto_refresh_tokens(True)
        self.monitor.limits_interval = timedelta(seconds=self._limits_interval)
        self.monitor.auto_refresh_tokens = self._auto_refresh_tokens
        self.monitor.accounts = resolve_accounts(config)

        self._reschedule_limits_timer()
        self.action_refresh_limits()
        self._tick_sessions()
        self._tick_stats()

    def action_add_account(self) -> None:
        """Provision an account and sign in, without leaving cctop.

        Suspends the dashboard to hand the terminal to two optional prompts
        (clone config from an existing account; set a shell alias) and the
        interactive login, then rediscovers accounts so the new one shows
        immediately. Everything is explicit and additive: nothing is aliased or
        cloned unless typed in, the clone is an allowlist (never credentials or
        state), and the login is scoped to the one new config dir.
        """
        from pathlib import Path

        from rich.console import Console

        from . import manage
        from .cli import resolve_config_dir, reusable_logged_out_dir, run_login
        from .collect import discover_accounts

        home = Path.home()
        console = Console()

        with self.suspend():
            claude = [a.name for a in discover_accounts() if a.provider == "claude"]
            console.print("[bold]Add a Claude account[/bold]")
            console.print(f"existing: {'  '.join(claude)}\n")

            source_spec = console.input(
                "Clone config (CLAUDE.md, settings, ...) from which account? [blank=none]: "
            ).strip()
            alias = console.input("Shell alias to set? [blank=none]: ").strip()

            plan = manage.plan_add(home, alias or None, reuse_dir=reusable_logged_out_dir(home))
            source_dir = resolve_config_dir(source_spec) if source_spec else None
            if source_spec and (source_dir is None or source_dir == plan.config_dir):
                console.print(f"[red]skipping clone: '{source_spec}' not usable[/red]")
                source_dir = None

            console.print(f"\nconfig dir {plan.config_dir}")
            for action in manage.ensure_config_dir(plan.config_dir):
                console.print(f"  + {action}")
            if source_dir is not None:
                for action in manage.copy_config(source_dir, plan.config_dir):
                    console.print(f"  + {action}")
            if alias:
                for action in manage.set_alias(plan):
                    console.print(f"  + {action}")
                console.print(f"[grey50]run `source ~/.zshrc` to use `{alias}` here[/grey50]")

            signed_in = run_login(plan.config_dir, console)
            console.input("\nPress Enter to return to cctop...")

        self.monitor.accounts = discover_accounts()
        self._tick_sessions()
        self.action_refresh_limits()
        state = "signed in" if signed_in else "set up (not signed in)"
        self.notify(f"{plan.config_dir.name} {state}", title="add account", timeout=6)

    def action_refresh_limits(self) -> None:
        self._force_limits()

    def action_refresh_token(self) -> None:
        """Ask the owner binary to renew each Claude account's expired token.

        cctop writes no credential: the delegated `claude mcp list` run does
        the refresh. Runs off the UI thread since it shells out per account.
        """
        self.notify("Refreshing tokens via the claude binary...", timeout=3)
        self._refresh_token()

    @work(thread=True, exclusive=True, group="token")
    def _refresh_token(self) -> None:
        now = datetime.now(timezone.utc)
        results, limits = self.monitor.refresh_credentials(now)

        summary = "   ".join(f"{r.account}: {r.message}" for r in results)
        summary = summary or "no Claude accounts to refresh"
        any_failed = any(not r.ok for r in results)

        self.call_from_thread(self._render_limits, limits, now)
        self.call_from_thread(
            lambda: self.notify(
                summary,
                title="token refresh",
                severity="warning" if any_failed else "information",
                timeout=6,
            )
        )

    @work(thread=True, exclusive=True, group="limits")
    def _force_limits(self) -> None:
        now = datetime.now(timezone.utc)
        limits = self.monitor.force_refresh_limits(now)
        self.call_from_thread(self._render_limits, limits, now)
        # Reset the periodic timer so the next auto-refresh is a full interval
        # away, keeping the footer countdown honest after a manual refresh.
        self.call_from_thread(self._reschedule_limits_timer)

    def _reschedule_limits_timer(self) -> None:
        if self._limits_timer is not None:
            self._limits_timer.stop()
        self._limits_timer = self.set_interval(self._limits_interval, self._tick_limits)

    def _render_limits(self, limits: list[AccountLimits], now: datetime) -> None:
        usage = self.query_one("#usage", Static)
        width = max(1, self.size.width - 4)
        source_only = {
            account.name for account in self.monitor.accounts if account.provider == "omp"
        }
        content = (
            _render_usage(limits, now, width, source_only)
            if limits
            else Text("no accounts", style=MUTED)
        )
        usage.update(content)

        for notice in self.monitor.pop_auth_notices():
            if "/login" in notice:
                self.notify(notice, title="token refresh", severity="warning", timeout=6)
            else:
                self.notify(notice, title="token refresh", timeout=6)

    def on_resize(self, event) -> None:
        self.call_after_refresh(self._resize_content)

    def _resize_content(self) -> None:
        now = datetime.now(timezone.utc)
        if self.monitor.limits:
            self._render_limits(self.monitor.limits, now)
        self._render_sessions(list(self._states_by_id.values()), now)
        self._render_stats(self._provider_stats)
