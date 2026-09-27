"""Human-in-the-loop trade approval via Telegram (config: approval_mode).

When `approval_mode: true`, every live, model-driven order is held until you tap
**Aprovar** or **Rebutjar** under a Telegram message. No answer within
`approval_timeout_seconds` (default 600) means no trade.

Transport: the Telegram Bot API (see notifications.py). The round-trip:

  1. The bot sends the proposal to TELEGRAM_CHAT_ID with an inline keyboard of
     two buttons, whose callback data is "<request id>:approve" / ":reject".
  2. Tapping a button makes Telegram queue a `callback_query` update for the bot.
  3. Meanwhile the bot polls `getUpdates` every few seconds for a callback that
     carries this request's one-off id, from the configured chat.

Nothing here needs an inbound endpoint on the bot's side, so it works the same
from a GitHub Actions job and from a VPS timer. It does need that nothing else
consumes the bot's updates: a webhook on the bot makes getUpdates fail (409), and
a second process polling at the same time can swallow a tap. Both end in a
timeout -- no trade -- never in a wrong approval.

Who can approve: a tap counts only if it comes from a message in
TELEGRAM_CHAT_ID (and, for a private chat, from that same user) and carries this
request's random id, which only ever appears in the proposal's own buttons.

Fail-safe in every direction -- all of these resolve to "not approved":
  * approval mode on but token / chat id unset or malformed -> "unavailable"
  * the proposal can't be sent (Telegram down, 4xx, ok=false) -> "unavailable"
  * a poll fails (network, 5xx, 409 webhook conflict)       -> keep polling to the deadline
  * a malformed update, a tap for another id or chat, or
    any callback data other than exactly "<id>:approve"     -> ignored
  * no valid tap by the deadline                            -> "timeout"
  * an approve and a reject both seen for this id           -> "rejected"
There is no path where a failure here places an order.

Deliberately NOT gated: the sweep's automatic stop-loss/take-profit exits
(main.sweep_open_positions). Waiting ten minutes on a stop that has already been
crossed would defeat the stop.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import notifications
from secrets_redaction import sanitize

DEFAULT_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 5.0

APPROVE = "approve"
REJECT = "reject"

TelegramCall = Callable[[str, Dict[str, Any]], Any]


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
    return sanitize("\n".join(lines))


def build_message(chat_id: str, request_id: str, title: str, message: str) -> Dict[str, Any]:
    """The sendMessage payload: plain text plus two inline buttons."""
    text = f"{title}\n\n{message}"[: notifications.MAX_MESSAGE_CHARS]
    return {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "Aprovar", "callback_data": f"{request_id}:{APPROVE}"},
                {"text": "Rebutjar", "callback_data": f"{request_id}:{REJECT}"},
            ]]
        },
    }


def _verdict(update: Any, chat_id: str, request_id: str) -> Optional[str]:
    """APPROVE / REJECT if `update` is a tap on this request from the right chat."""
    if not isinstance(update, dict):
        return None
    query = update.get("callback_query")
    if not isinstance(query, dict):
        return None
    message = query.get("message")
    chat = message.get("chat") if isinstance(message, dict) else None
    if not isinstance(chat, dict) or str(chat.get("id")) != chat_id:
        return None
    if not chat_id.startswith("-"):
        sender = query.get("from")
        if not isinstance(sender, dict) or str(sender.get("id")) != chat_id:
            return None
    data = query.get("data")
    if data == f"{request_id}:{APPROVE}":
        return APPROVE
    if data == f"{request_id}:{REJECT}":
        return REJECT
    return None


def _next_offset(updates: list, offset: Optional[int]) -> Optional[int]:
    """Confirm everything seen so far, so the next poll only returns newer updates."""
    for update in updates:
        update_id = update.get("update_id") if isinstance(update, dict) else None
        if isinstance(update_id, int) and not isinstance(update_id, bool):
            offset = max(offset or 0, update_id + 1)
    return offset


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
    call: Optional[TelegramCall] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ApprovalDecision:
    """Send the proposal, then block until a matching tap or the deadline."""
    call = call or notifications.telegram_call
    token, chat_id = notifications.telegram_settings()
    if not token or not chat_id:
        return ApprovalDecision(
            False, "unavailable",
            "approval_mode is on but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID is not set",
        )
    if not notifications.credentials_valid(token, chat_id):
        return ApprovalDecision(
            False, "unavailable", "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is malformed"
        )

    request_id = secrets.token_hex(8)
    title = sanitize(f"APROVACIO: {action.upper()} {symbol}")
    message = format_request(
        symbol, action, size_pct, price, stop_loss, take_profit,
        confidence, reasoning, equity, timeout_seconds,
    )

    try:
        sent = call("sendMessage", build_message(chat_id, request_id, title, message))
    except Exception as exc:  # noqa: BLE001
        return ApprovalDecision(
            False, "unavailable",
            sanitize(f"could not send the approval request: {type(exc).__name__}: {exc}"),
        )
    message_id = sent.get("message_id") if isinstance(sent, dict) else None

    offset: Optional[int] = None
    deadline = clock() + float(timeout_seconds)
    while clock() < deadline:
        payload: Dict[str, Any] = {"timeout": 0, "allowed_updates": ["callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        try:
            updates = call("getUpdates", payload)
        except Exception:  # noqa: BLE001 - transient; keep polling until the deadline
            updates = None
        updates = updates if isinstance(updates, list) else []
        offset = _next_offset(updates, offset)

        verdicts = set()
        for update in updates:
            verdict = _verdict(update, chat_id, request_id)
            if verdict:
                verdicts.add(verdict)
                _answer(call, update["callback_query"].get("id"))
        if REJECT in verdicts:
            _close_out(call, chat_id, message_id, title, "REBUTJADA")
            return ApprovalDecision(False, "rejected", "rejected from Telegram")
        if APPROVE in verdicts:
            _close_out(call, chat_id, message_id, title, "APROVADA")
            return ApprovalDecision(True, "approved", "approved from Telegram")

        sleep(min(POLL_INTERVAL_SECONDS, max(deadline - clock(), 0.0)))

    _close_out(call, chat_id, message_id, title, "CADUCADA (no s'executa)")
    return ApprovalDecision(False, "timeout", f"no answer within {int(timeout_seconds)}s")


def _answer(call: TelegramCall, callback_query_id: Any) -> None:
    """Stops the button's loading spinner. Best effort."""
    if not callback_query_id:
        return
    try:
        call("answerCallbackQuery", {"callback_query_id": callback_query_id})
    except Exception:  # noqa: BLE001
        pass


def _close_out(call: TelegramCall, chat_id: str, message_id: Any, title: str, verdict: str) -> None:
    """Remove the buttons and post how the request ended. Best effort."""
    if message_id is not None:
        try:
            call("editMessageReplyMarkup", {
                "chat_id": chat_id, "message_id": message_id, "reply_markup": {"inline_keyboard": []},
            })
        except Exception:  # noqa: BLE001
            pass
    try:
        call("sendMessage", {"chat_id": chat_id, "text": f"{title}: {verdict}"})
    except Exception:  # noqa: BLE001
        pass
