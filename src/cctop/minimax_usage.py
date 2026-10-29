"""Live MiniMax Token Plan quota, as `mmx quota show` reads it.

The MiniMax CLI (mmx) fetches Token Plan quotas from GET
{base}/v1/token_plan/remains (discovered from mmx's own bundle). It is a plain
GET, so it consumes no quota, and returns per-model 5-hour and weekly windows
as remaining percentages.

The credential is read at runtime from ~/.mmx/config.json (an API key or mmx's
OAuth access token) and used only to authenticate to MiniMax's own API; it is
never logged or persisted. mmx is not a coding agent, so this provider has
limits only: sessions of agents configured with a MiniMax key appear under
those agents' own accounts.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .codex_usage import _window_label
from .models import AccountLimits, LimitWindow

MMX_DIR = Path.home() / ".mmx"
_BASE_URLS = {"global": "https://api.minimax.io", "cn": "https://api.minimax.cn"}
_REMAINS_PATH = "/v1/token_plan/remains"


def _read_auth(config_dir: Path) -> tuple[str, str] | None:
    """The (token, base_url) mmx would use, or None when it has no credential."""
    try:
        record = json.loads((config_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        record = {}
    if not isinstance(record, dict):
        record = {}
    oauth = record.get("oauth") if isinstance(record.get("oauth"), dict) else {}

    token = record.get("api_key") or oauth.get("access_token") or os.environ.get("MINIMAX_API_KEY")
    if not isinstance(token, str) or not token.strip():
        return None

    region = record.get("region") or oauth.get("region") or "global"
    base = (
        os.environ.get("MINIMAX_BASE_URL")
        or record.get("base_url")
        or oauth.get("resource_url")
        or _BASE_URLS.get(region, _BASE_URLS["global"])
    )
    return token.strip(), str(base).rstrip("/")


def _not_in_plan(model: dict) -> bool:
    """mmx's own test for a model the plan does not include (shown as "—")."""
    return (
        model.get("current_interval_total_count") == 0
        and model.get("current_weekly_total_count") == 0
        and model.get("current_interval_status") == 3
        and model.get("current_weekly_status") == 3
    )


def _window(
    kind: str, label: str, remaining: object, start_ms: object, end_ms: object
) -> LimitWindow | None:
    if not isinstance(remaining, (int, float)):
        return None
    percent = max(0.0, min(100.0, 100.0 - float(remaining)))
    resets_at = None
    if isinstance(end_ms, (int, float)) and end_ms > 0:
        resets_at = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    if (
        isinstance(start_ms, (int, float))
        and isinstance(end_ms, (int, float))
        and end_ms > start_ms
    ):
        duration = _window_label(int((end_ms - start_ms) // 1000))
        label = f"{label} {duration}".strip()
    return LimitWindow(
        kind=kind,
        label=label,
        percent=percent,
        resets_at=resets_at,
        severity="warning" if percent >= 100 else "normal",
        is_active=False,
        has_data=resets_at is not None,
    )


def _windows(payload: dict) -> list[LimitWindow]:
    windows: list[LimitWindow] = []
    models = [m for m in payload.get("model_remains") or [] if isinstance(m, dict)]
    for model in models:
        if _not_in_plan(model):
            continue
        # The plan-wide pool is named "general"; label only model-scoped pools.
        name = model.get("model_name") or ""
        prefix = "" if name in ("", "general") else str(name)
        for window in (
            _window(
                "session",
                prefix,
                model.get("current_interval_remaining_percent"),
                model.get("start_time"),
                model.get("end_time"),
            ),
            _window(
                "weekly",
                prefix,
                model.get("current_weekly_remaining_percent"),
                model.get("weekly_start_time"),
                model.get("weekly_end_time"),
            ),
        ):
            if window is not None:
                windows.append(window)

    # Mark the most-consumed window as the active (binding) one.
    if windows:
        peak = max(range(len(windows)), key=lambda i: windows[i].percent)
        windows[peak] = LimitWindow(
            kind=windows[peak].kind,
            label=windows[peak].label,
            percent=windows[peak].percent,
            resets_at=windows[peak].resets_at,
            severity=windows[peak].severity,
            is_active=True,
            has_data=windows[peak].has_data,
        )
    return windows


def fetch_account_limits(account: str = "mm-0", config_dir: Path = MMX_DIR) -> AccountLimits:
    """Fetch and parse the MiniMax Token Plan's live quota (free GET)."""
    auth = _read_auth(config_dir)
    if auth is None:
        return AccountLimits(
            account, None, [], "none", None, error="no minimax key - run mmx auth login"
        )
    token, base = auth
    if token.startswith("sk-api-"):
        return AccountLimits(
            account, None, [], "none", None, error="pay-as-you-go key; no Token Plan quota"
        )

    request = urllib.request.Request(base + _REMAINS_PATH, method="GET")
    request.add_header("authorization", f"Bearer {token}")
    request.add_header("accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return AccountLimits(
                account,
                None,
                [],
                "none",
                None,
                error="key rejected - run mmx auth login",
                auth_expired=True,
            )
        if error.code == 429 or error.code >= 500:
            retry_after: float | None = None
            raw_retry = error.headers.get("retry-after")
            if raw_retry is not None:
                try:
                    retry_after = float(raw_retry)
                except ValueError:
                    retry_after = None
            label = "rate limited (429)" if error.code == 429 else f"HTTP {error.code}"
            return AccountLimits(
                account,
                None,
                [],
                "none",
                None,
                error=f"minimax quota: {label}",
                retriable=True,
                retry_after=retry_after,
            )
        return AccountLimits(
            account, None, [], "none", None, error=f"minimax quota: HTTP {error.code}"
        )
    except (urllib.error.URLError, TimeoutError) as error:
        return AccountLimits(
            account, None, [], "none", None, error=f"minimax quota: {error}", retriable=True
        )
    except json.JSONDecodeError as error:
        return AccountLimits(account, None, [], "none", None, error=f"minimax quota: {error}")

    status = payload.get("base_resp") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or (isinstance(status, dict) and status.get("status_code")):
        message = status.get("status_msg") if isinstance(status, dict) else "unexpected response"
        return AccountLimits(account, None, [], "none", None, error=f"minimax quota: {message}")

    return AccountLimits(
        account=account,
        tier="token plan",
        windows=_windows(payload),
        source="api",
        fetched_at=datetime.now(timezone.utc),
    )
