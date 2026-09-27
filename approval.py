"""Human-in-the-loop trade approval via phone push notification (config: approval_mode).

When `approval_mode: true`, every live, model-driven order is held until you tap
**Aprovar** or **Rebutjar** on a push notification. No answer within
`approval_timeout_seconds` (default 600) means no trade.

Transport: ntfy (see notifications.py). The round-trip, end to end:

  1. The bot publishes the proposal to NTFY_TOPIC with two action buttons. The
     ntfy app on the phone shows it as a normal push notification (lock screen
     included) with "Aprovar" / "Rebutjar" underneath.
  2. Tapping a button makes the *phone* send an HTTP POST, body
     "<request id>:approve" or "<request id>:reject", to the reply topic
     NTFY_TOPIC + "-reply". The notification then clears itself.
  3. Meanwhile the bot polls that reply topic every few seconds for a message
     carrying this request's one-off id.

Nothing here needs an inbound endpoint on the bot's side, so it works the same
from a GitHub Actions job and from a VPS timer.

Who can approve: whoever can publish to the reply topic *and* has seen this
request's random id, which only ever appears in the notification itself. On the
public ntfy.sh that means whoever knows NTFY_TOPIC -- treat it like a password
(long and random), or use a server/topic protected with NTFY_TOKEN.

Fail-safe in every direction -- all of these resolve to "not approved":
  * approval mode on but NTFY_TOPIC unset             -> "unavailable"
  * the proposal can't be published (ntfy down, 4xx)  -> "unavailable"
  * a poll fails (network, 5xx)                       -> keep polling to the deadline
  * a malformed reply line, a reply for another id,
    or any body other than exactly "<id>:approve"      -> ignored
  * no valid tap by the deadline                      -> "timeout"
  * an approve and a reject both seen for this id     -> "rejected"
There is no path where a failure here places an order.

Deliberately NOT gated: the sweep's automatic stop-loss/take-profit exits
(main.sweep_open_positions). Waiting ten minutes on a stop that has already been
crossed would defeat the stop.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import notifications
from secrets_redaction import sanitize

DEFAULT_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 5.0
# Allow for a little clock skew between this machine and the ntfy server when
# asking for replies "since" the moment the request went out.
SINCE_SKEW_SECONDS = 10

APPROVE = "approve"
REJECT = "reject"


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    # "approved" | "rejected" | "timeout" | "unavailable"
    outcome: str
    detail: str = ""


def approval_enabled(config: Mapping[str, Any]) -> bool:
    """Strict, like live_execution: only a literal boolean true turns this on."""
    return (config or {}).get("approval_mode") is True


def approval_timeout_seconds(config: Mapping[str, Any]) -> float:
    try:
        value = float((config or {}).get("approval_timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def reply_topic(topic: str) -> str:
    return f"{topic}-reply"


def format_request(
    symbol: str,
    action: str,
    size_pct: float,
    price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    confidence: float,
    reasoning: str,
    equity: float,
    timeout_seconds: float,
) -> str:
    notional = equity * size_pct / 100.0
    lines = [f"Preu actual: {price:.6g}"]
    if action == "buy":
        lines.append(f"Mida: {size_pct:.2f}% del patrimoni (~${notional:,.2f})")
    if stop_loss is not None and take_profit is not None:
        lines.append(f"Stop-loss: {stop_loss:.6g} | Take-profit: {take_profit:.6g}")
    lines.append(f"Confianca: {confidence:.2f}")
    lines.append("")
    lines.append(reasoning or "(sense raonament)")
    lines.append("")
    lines.append(f"Sense resposta en {int(timeout_seconds // 60)} min = no s'executa.")
    return sanitize("\n".join(lines))[: notifications.MAX_MESSAGE_CHARS]


def build_notification(
    topic: str, server: str, token: Optional[str], request_id: str, title: str, message: str
) -> Dict[str, Any]:
    """The ntfy publish payload: a max-priority push with two HTTP action buttons."""
    url = f"{server}/{reply_topic(topic)}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def button(label: str, verdict: str) -> Dict[str, Any]:
        action: Dict[str, Any] = {
            "action": "http",
            "label": label,
            "url": url,
            "method": "POST",
            "body": f"{request_id}:{verdict}",
            "clear": True,
        }
        if headers:
            action["headers"] = headers
        return action

    return {
        "topic": topic,
        "title": title,
        "message": message,
        "priority": 5,
        "tags": ["warning"],
        "actions": [button("Aprovar", APPROVE), button("Rebutjar", REJECT)],
    }


def request_approval(
    *,
    symbol: str,
    action: str,
    size_pct: float,
    price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    confidence: float,
    reasoning: str,
    equity: float,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    publish: Optional[Callable[[Dict[str, Any]], Any]] = None,
    poll: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ApprovalDecision:
    """Send the proposal, then block until a matching tap or the deadline."""
    publish = publish or notifications.ntfy_publish
    poll = poll or notifications.ntfy_poll
    server, topic, token = notifications.ntfy_settings()
    if not topic:
        return ApprovalDecision(
            False, "unavailable", "approval_mode is on but NTFY_TOPIC is not set"
        )
    if not notifications.topic_is_strong(topic):
        return ApprovalDecision(
            False, "unavailable",
            "NTFY_TOPIC is too short or has invalid characters -- anyone could guess it "
            "and approve trades; generate one with: echo tb-$(openssl rand -hex 16)",
        )

    request_id = secrets.token_hex(8)
    approve_body = f"{request_id}:{APPROVE}"
    reject_body = f"{request_id}:{REJECT}"
    title = sanitize(f"APROVACIO: {action.upper()} {symbol}")
    message = format_request(
        symbol, action, size_pct, price, stop_loss, take_profit,
        confidence, reasoning, equity, timeout_seconds,
    )

    since = int(wall_clock()) - SINCE_SKEW_SECONDS
    try:
        publish(build_notification(topic, server, token, request_id, title, message))
    except Exception as exc:  # noqa: BLE001
        return ApprovalDecision(
            False, "unavailable",
            sanitize(f"could not publish the approval request: {type(exc).__name__}: {exc}"),
        )

    deadline = clock() + float(timeout_seconds)
    while clock() < deadline:
        try:
            replies = poll(reply_topic(topic), since) or []
        except Exception:  # noqa: BLE001 - transient; keep polling until the deadline
            replies = []

        bodies = {
            str(r.get("message", "")).strip()
            for r in replies
            if isinstance(r, dict)
        }
        if reject_body in bodies:
            _close_out(publish, topic, title, "REBUTJADA")
            return ApprovalDecision(False, "rejected", "rejected from the phone")
        if approve_body in bodies:
            _close_out(publish, topic, title, "APROVADA")
            return ApprovalDecision(True, "approved", "approved from the phone")

        sleep(min(POLL_INTERVAL_SECONDS, max(deadline - clock(), 0.0)))

    _close_out(publish, topic, title, "CADUCADA (no s'executa)")
    return ApprovalDecision(False, "timeout", f"no answer within {int(timeout_seconds)}s")


def _close_out(publish: Callable[[Dict[str, Any]], Any], topic: str, title: str, verdict: str) -> None:
    """A short follow-up push so the phone shows how the request ended. Best effort."""
    try:
        publish({"topic": topic, "title": f"{title}: {verdict}", "message": verdict, "priority": 3})
    except Exception:  # noqa: BLE001
        pass
