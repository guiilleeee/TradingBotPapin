"""Outbound alerts, delivered as phone push notifications through ntfy.

ntfy (https://ntfy.sh, or a self-hosted server) is a plain HTTP pub/sub service
with Android and iOS apps: the bot POSTs a message to a topic, the app
subscribed to that topic shows it as a native push notification. No account,
no inbound endpoint on the bot's side -- which is what a short-lived GitHub
Actions job or a systemd timer can actually use. approval.py builds its
approve/reject round-trip on the same two calls (`ntfy_publish`, `ntfy_poll`).

Configuration (environment, like every other secret):
  NTFY_TOPIC   required; the topic the phone app subscribes to. On the public
               ntfy.sh server the topic name IS the password -- use a long
               random one (e.g. `openssl rand -hex 16`).
  NTFY_SERVER  optional, default https://ntfy.sh
  NTFY_TOKEN   optional access token, for a protected/self-hosted server.

Every public alert function is fire-and-forget: it never raises, so an alert
failure can never take a trading cycle down with it. With NTFY_TOPIC unset,
every alert is a silent no-op. Text passes through secrets_redaction.sanitize
before leaving the process.

Web push is NOT implemented: the dashboard is a static page with nowhere to
store subscriptions. `_send_web_push` stays a documented no-op.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from secrets_redaction import sanitize as _sanitize

DEFAULT_NTFY_SERVER = "https://ntfy.sh"
HTTP_TIMEOUT = 15.0
# ntfy.sh rejects message bodies over 4096 bytes (they become attachments).
MAX_MESSAGE_CHARS = 3500


# ----------------------------------------------------------------------- ntfy


def ntfy_settings() -> tuple[str, Optional[str], Optional[str]]:
    """(server, topic, token) from the environment."""
    server = (os.environ.get("NTFY_SERVER") or DEFAULT_NTFY_SERVER).rstrip("/")
    return server, os.environ.get("NTFY_TOPIC") or None, os.environ.get("NTFY_TOKEN") or None


# On the public ntfy.sh server the topic name is the only secret, so a short or
# guessable one would let anyone read trade proposals and answer approvals.
# `tb-` + `openssl rand -hex 16` is 35 characters; anything under 24, or with
# characters ntfy doesn't allow in a topic, is refused outright.
_STRONG_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{24,64}$")


def topic_is_strong(topic: Optional[str]) -> bool:
    return bool(topic) and bool(_STRONG_TOPIC_RE.match(topic))


def ntfy_configured() -> bool:
    """A topic is set *and* strong enough to use. A weak one is treated as unset."""
    topic = ntfy_settings()[1]
    if topic is None:
        return False
    if not topic_is_strong(topic):
        logging.error("NTFY_TOPIC is too short or has invalid characters; refusing to use it "
                      "(generate one with: echo tb-$(openssl rand -hex 16))")
        return False
    return True


def _auth_headers() -> Dict[str, str]:
    token = ntfy_settings()[2]
    return {"Authorization": f"Bearer {token}"} if token else {}


def ntfy_publish(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST one JSON message (payload must name its `topic`). Raises on failure."""
    server, _, _ = ntfy_settings()
    resp = requests.post(server, json=payload, headers=_auth_headers(), timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def ntfy_poll(topic: str, since: int) -> List[Dict[str, Any]]:
    """Every cached message on `topic` published at/after unix time `since`.

    ntfy answers with newline-delimited JSON; a line that doesn't parse is
    skipped rather than trusted. Raises on a transport failure.
    """
    server, _, _ = ntfy_settings()
    resp = requests.get(
        f"{server}/{topic}/json",
        params={"poll": "1", "since": str(int(since))},
        headers=_auth_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    messages = []
    for line in resp.text.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("event") == "message":
            messages.append(item)
    return messages


def _send_ntfy(subject: str, message: str, config: Optional[dict] = None, priority: int = 3) -> None:
    _, topic, _ = ntfy_settings()
    if not topic:
        raise RuntimeError("NTFY_TOPIC is not set")
    ntfy_publish({
        "topic": topic,
        "title": _sanitize(subject)[:200],
        "message": _sanitize(message or subject)[:MAX_MESSAGE_CHARS],
        "priority": priority,
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

    ntfy_config = config.get("ntfy", {})
    if isinstance(ntfy_config, dict) and ntfy_config.get("enabled", False):
        try:
            _send_ntfy(subject, message, ntfy_config)
        except Exception as e:  # noqa: BLE001
            logging.error(f"ntfy push failed: {_sanitize(str(e))}")


def _notify(subject: str, message: str, priority: int = 3) -> None:
    """Every typed alert below lands here: ntfy if configured, else nothing."""
    if not ntfy_configured():
        return
    try:
        _send_ntfy(subject, message, priority=priority)
    except Exception as e:  # noqa: BLE001 - an alert must never break a cycle
        logging.error(f"ntfy push failed: {_sanitize(str(e))}")


def _mode(is_live: bool) -> str:
    return "REAL" if is_live else "SIMULACIO"


# ------------------------------------------------------------- typed alerts


def send_trade_alert(
    is_live: bool,
    symbol: str,
    action: str,
    size_pct: float,
    price: float,
    confidence: float,
    reasoning: str,
) -> None:
    _notify(
        f"[{_mode(is_live)}] {action.upper()} {symbol}",
        f"Preu {price:.6g} | mida {size_pct:.2f}% | conf {confidence:.2f}\n\n{reasoning}",
    )


def send_circuit_breaker_alert(is_live: bool, today_loss_pct: float, threshold_pct: float) -> None:
    _notify(
        f"[{_mode(is_live)}] Circuit breaker activat",
        f"Perdua realitzada d'avui {today_loss_pct:.2f}% (limit -{abs(threshold_pct):.2f}%). "
        "No s'obriran noves operacions fins dema (UTC).",
        priority=4,
    )


def send_auto_close_alert(is_live: bool, symbol: str, reason: str, pnl: float) -> None:
    _notify(f"[{_mode(is_live)}] Tancament automatic {symbol}", f"P&L {pnl:+.2f} USD\n\n{reason}")


def send_cycle_failure_alert(is_live: bool, summary: str) -> None:
    _notify(f"[{_mode(is_live)}] Cicle fallit", summary, priority=4)


def send_approval_outcome_alert(is_live: bool, symbol: str, action: str, outcome: str) -> None:
    _notify(f"[{_mode(is_live)}] {action.upper()} {symbol}: {outcome}", f"Ordre no enviada ({outcome}).")


def send_screening_complete_alert(is_live: bool, symbols: List[str], error: Exception = None) -> None:
    _notify(f"[{_mode(is_live)}] Cribratge setmanal", f"{len(symbols)} accions: {', '.join(symbols)}")


def send_screening_failure_alert(is_live: bool, reason: str) -> None:
    _notify(f"[{_mode(is_live)}] Cribratge setmanal fallit", reason, priority=4)


def send_volume_wake_alert(symbol: str, price: float, trigger_reasons: List[str], wake_action: str) -> None:
    _notify(f"Wake-up {wake_action} {symbol}", f"Preu {price:.6g}: {', '.join(trigger_reasons)}")
