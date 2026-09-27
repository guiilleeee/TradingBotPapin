"""Outbound alerts, delivered as Telegram messages from the bot to one chat.

The bot calls the Telegram Bot API's `sendMessage` over plain HTTPS. It only
ever sends; it never reads replies or waits on anyone.

Configuration (environment, like every other secret):
  TELEGRAM_BOT_TOKEN  the token @BotFather issued. Whoever holds it controls
                      the bot, so it is redacted everywhere (secrets_redaction).
  TELEGRAM_CHAT_ID    the numeric id of the one chat that receives alerts.

Every public alert function is fire-and-forget: it never raises, so an alert
failure can never take a trading cycle down with it. With either variable unset
(or malformed), every alert is a silent no-op. Text passes through
secrets_redaction.sanitize before leaving the process, and is sent as plain text
(no parse_mode).

Web push is NOT implemented: the dashboard is a static page with nowhere to
store subscriptions. `_send_web_push` stays a documented no-op.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from secrets_redaction import sanitize as _sanitize

TELEGRAM_API = "https://api.telegram.org"
HTTP_TIMEOUT = 15.0
# Telegram rejects messages over 4096 characters.
MAX_MESSAGE_CHARS = 3500

_BOT_TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
_CHAT_ID_RE = re.compile(r"^-?\d{1,20}$")


# ------------------------------------------------------------------- telegram


def telegram_settings() -> tuple[Optional[str], Optional[str]]:
    """(bot token, chat id) from the environment."""
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None
    return token, chat_id


def credentials_valid(token: Optional[str], chat_id: Optional[str]) -> bool:
    return bool(token and chat_id and _BOT_TOKEN_RE.match(token) and _CHAT_ID_RE.match(chat_id))


def telegram_configured() -> bool:
    """Both variables set *and* well-formed. A malformed pair is treated as unset."""
    token, chat_id = telegram_settings()
    if token is None or chat_id is None:
        return False
    if not credentials_valid(token, chat_id):
        logging.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is malformed; refusing to use them")
        return False
    return True


def telegram_call(method: str, payload: Dict[str, Any]) -> Any:
    """One Bot API call; returns its `result`. Raises on any failure, including
    an HTTP 200 whose body is not `{"ok": true}`."""
    token, _ = telegram_settings()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    resp = requests.post(f"{TELEGRAM_API}/bot{token}/{method}", json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict) or body.get("ok") is not True:
        description = body.get("description") if isinstance(body, dict) else body
        raise RuntimeError(_sanitize(f"Telegram {method} failed: {description}"))
    return body.get("result")


def _send_telegram(subject: str, message: str, config: Optional[dict] = None) -> None:
    _, chat_id = telegram_settings()
    if not chat_id:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")
    text = f"{_sanitize(subject)[:200]}\n\n{_sanitize(message or '')}".strip()
    telegram_call("sendMessage", {
        "chat_id": chat_id,
        "text": text[:MAX_MESSAGE_CHARS],
        "disable_web_page_preview": True,
    })


def _send_web_push(subject: str, message: str, config: Optional[dict] = None) -> None:
    """Not implemented -- see the module docstring."""
    return None


def send_alert(subject: str, message: str, config: Optional[dict] = None) -> None:
    """Dispatch to whichever channels `config` enables. Never raises."""
    config = config or {}

    web_push_config = config.get("web_push", {})
    if isinstance(web_push_config, dict) and web_push_config.get("enabled", False):
        try:
            _send_web_push(subject, message, web_push_config)
        except Exception as e:  # noqa: BLE001
            logging.error(f"Web push failed: {_sanitize(str(e))}")

    telegram_config = config.get("telegram", {})
    if isinstance(telegram_config, dict) and telegram_config.get("enabled", False):
        try:
            _send_telegram(subject, message, telegram_config)
        except Exception as e:  # noqa: BLE001
            logging.error(f"Telegram push failed: {_sanitize(str(e))}")


def _notify(line: str) -> None:
    """Every typed alert below lands here: one line to Telegram if configured."""
    if not telegram_configured():
        return
    line = " ".join(str(line).split())[:300]
    try:
        _send_telegram(line, "")
    except Exception as e:  # noqa: BLE001 - an alert must never break a cycle
        logging.error(f"Telegram push failed: {_sanitize(str(e))}")


def _prefix(is_live: bool) -> str:
    return "" if is_live else "[SIM] "


def _fmt_qty(qty: Optional[float]) -> str:
    if qty is None:
        return ""
    return f"{float(qty):.8f}".rstrip("0").rstrip(".") + " "


def _fmt_usd(price: float) -> str:
    price = float(price)
    return f"${price:,.2f}" if abs(price) >= 1 else f"${price:.6g}"


def _first_line(text: str) -> str:
    return (str(text or "").strip().splitlines() or [""])[0]


# ------------------------------------------------------------- typed alerts
# One short line per event, no model reasoning: "BUY 10 AAPL @ $182.30".


def send_trade_alert(
    is_live: bool, symbol: str, action: str, qty: Optional[float], price: float
) -> None:
    _notify(f"{_prefix(is_live)}{action.upper()} {_fmt_qty(qty)}{symbol} @ {_fmt_usd(price)}")


def send_auto_close_alert(is_live: bool, symbol: str, qty: Optional[float], price: float) -> None:
    send_trade_alert(is_live, symbol, "sell", qty, price)


def send_circuit_breaker_alert(is_live: bool, today_loss_pct: float, threshold_pct: float) -> None:
    _notify(f"{_prefix(is_live)}CIRCUIT BREAKER {today_loss_pct:.2f}% (limit -{abs(threshold_pct):.2f}%)")


def send_cycle_failure_alert(is_live: bool, summary: str) -> None:
    _notify(f"{_prefix(is_live)}CYCLE FAILED: {_first_line(summary)}")


def send_screening_complete_alert(is_live: bool, symbols: List[str], error: Exception = None) -> None:
    _notify(f"{_prefix(is_live)}SCREENING DONE: {len(symbols)} symbols")


def send_screening_failure_alert(is_live: bool, reason: str) -> None:
    _notify(f"{_prefix(is_live)}SCREENING FAILED: {_first_line(reason)}")


def send_volume_wake_alert(symbol: str, price: float, trigger_reasons: List[str], wake_action: str) -> None:
    _notify(f"WAKE-UP {wake_action} {symbol} @ {_fmt_usd(price)}")
