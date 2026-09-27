"""Web Push to the dashboard (phone/desktop browser), alongside Telegram.

How the pieces fit (no inbound endpoint anywhere, GitHub Pages stays static):

  * `python web_push.py setup` (once, on the VPS) makes two key pairs:
      - VAPID (P-256): signs every push. Private half -> VAPID_PRIVATE_KEY in .env.
      - RSA-2048: the page encrypts each browser subscription with the public
        half, so the public repo only ever stores ciphertext. Private half ->
        PUSH_SUBSCRIPTION_KEY in .env.
    Both public halves go to docs/push_config.json, which the page reads.
  * The dashboard's bell subscribes the browser, encrypts the subscription
    (RSA-OAEP-SHA256 wrapping an AES-256-GCM key) and sends it through the same
    GitHub-token dispatch as its other buttons. push_subscriptions.yml runs
    `python web_push.py store`, the only writer of docs/push_subscriptions.json.
  * notifications._notify calls `send_to_all` with the same one-line alert as
    Telegram. Subscriptions a push service reports gone (404/410) are remembered
    in push_dead.json (local, gitignored) and skipped from then on.

Sending never raises: an alert must never break a trading cycle.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from secrets_redaction import sanitize

SUBSCRIPTIONS_PATH = os.path.join("docs", "push_subscriptions.json")
CONFIG_PATH = os.path.join("docs", "push_config.json")
DEAD_PATH = "push_dead.json"
# VAPID's `sub` claim: "mailto:<VAPID_ADMIN_EMAIL>", or this origin if unset.
# pywebpush only accepts a mailto: address or a bare https:// origin -- a URL
# with a path (e.g. the dashboard's /TradingBotPapin/) is rejected outright.
DEFAULT_SUBJECT = "https://guiilleeee.github.io"
_EMAIL_RE = re.compile(r"^[^@\s:/]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")
MAX_SUBSCRIPTIONS = 20
PUSH_TTL_SECONDS = 3600
HTTP_TIMEOUT = 10.0

# Decrypted endpoints must belong to a real browser push service, so a crafted
# subscription can never make the VPS send requests to an arbitrary host.
ALLOWED_PUSH_HOSTS = (
    "fcm.googleapis.com",
    "updates.push.services.mozilla.com",
    "push.services.mozilla.com",
    "web.push.apple.com",
    "notify.windows.com",
    "push.apple.com",
)

_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------- keys


def generate_keys() -> Dict[str, str]:
    vapid = ec.generate_private_key(ec.SECP256R1())
    vapid_private = _b64url(vapid.private_numbers().private_value.to_bytes(32, "big"))
    vapid_public = _b64url(vapid.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))

    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sub_private = base64.b64encode(rsa_key.private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())).decode()
    sub_public = base64.b64encode(rsa_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
    return {
        "vapid_private_key": vapid_private,
        "vapid_public_key": vapid_public,
        "subscription_private_key": sub_private,
        "subscription_public_key": sub_public,
    }


# ------------------------------------------------------------- encryption

_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def subscription_id(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode()).hexdigest()


def encrypt_subscription(subscription: Dict[str, Any], public_key_b64: str) -> Dict[str, Any]:
    """What the dashboard does in the browser (WebCrypto), mirrored for tests."""
    public_key = serialization.load_der_public_key(base64.b64decode(public_key_b64))
    aes_key = AESGCM.generate_key(bit_length=256)
    iv = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(iv, json.dumps(subscription).encode(), None)
    return {
        "id": subscription_id(subscription["endpoint"]),
        "v": 1,
        "k": base64.b64encode(public_key.encrypt(aes_key, _OAEP)).decode(),
        "iv": base64.b64encode(iv).decode(),
        "ct": base64.b64encode(ciphertext).decode(),
    }


def decrypt_subscription(record: Dict[str, Any], private_key_b64: str) -> Dict[str, Any]:
    private_key = serialization.load_der_private_key(base64.b64decode(private_key_b64), password=None)
    aes_key = private_key.decrypt(base64.b64decode(record["k"]), _OAEP)
    plaintext = AESGCM(aes_key).decrypt(base64.b64decode(record["iv"]), base64.b64decode(record["ct"]), None)
    subscription = json.loads(plaintext)
    if not endpoint_allowed(subscription.get("endpoint")):
        raise ValueError("endpoint is not a known browser push service")
    keys = subscription.get("keys")
    if not isinstance(keys, dict) or not keys.get("p256dh") or not keys.get("auth"):
        raise ValueError("subscription has no keys")
    return {"endpoint": subscription["endpoint"], "keys": {"p256dh": keys["p256dh"], "auth": keys["auth"]}}


def endpoint_allowed(endpoint: Any) -> bool:
    if not isinstance(endpoint, str):
        return False
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == h or host.endswith("." + h) for h in ALLOWED_PUSH_HOSTS)


# ----------------------------------------------------------- the store


def load_store(path: str = SUBSCRIPTIONS_PATH) -> List[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            subs = json.load(handle).get("subscriptions", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [s for s in subs if isinstance(s, dict)] if isinstance(subs, list) else []


def _valid_b64(value: Any, max_len: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= max_len and bool(_B64_RE.match(value))


def validate_record(payload: Any) -> Dict[str, Any]:
    """Shape and size checks only -- the content is ciphertext by design."""
    if not isinstance(payload, dict):
        raise ValueError("payload is not an object")
    if not isinstance(payload.get("id"), str) or not _ID_RE.match(payload["id"]):
        raise ValueError("bad id")
    if payload.get("v") != 1:
        raise ValueError("unsupported version")
    if not _valid_b64(payload.get("k"), 700) or not _valid_b64(payload.get("iv"), 24) \
            or not _valid_b64(payload.get("ct"), 4000):
        raise ValueError("bad ciphertext fields")
    return {k: payload[k] for k in ("id", "v", "k", "iv", "ct")}


def apply_change(subs: List[Dict[str, Any]], action: str, payload: Any) -> List[Dict[str, Any]]:
    if action == "push_unsubscribe":
        sub_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(sub_id, str) or not _ID_RE.match(sub_id):
            raise ValueError("bad id")
        return [s for s in subs if s.get("id") != sub_id]
    if action == "push_subscribe":
        record = validate_record(payload)
        record["added_at"] = _utc_now_iso()
        kept = [s for s in subs if s.get("id") != record["id"]]
        if len(kept) >= MAX_SUBSCRIPTIONS:
            raise ValueError(f"already {MAX_SUBSCRIPTIONS} subscriptions; unsubscribe an old device first")
        return kept + [record]
    raise ValueError(f"unknown action {action!r}")


def write_store(subs: List[Dict[str, Any]], path: str = SUBSCRIPTIONS_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"subscriptions": subs}, handle, indent=2)


# --------------------------------------------------------------- sending


def vapid_subject() -> str:
    """The `sub` claim: mailto: the admin email if it's set and well-formed."""
    email = (os.environ.get("VAPID_ADMIN_EMAIL") or "").strip()
    if email.lower().startswith("mailto:"):
        email = email[len("mailto:"):]
    if email and _EMAIL_RE.match(email):
        return f"mailto:{email}"
    if email:
        logging.error("VAPID_ADMIN_EMAIL is not a valid email address; using the default subject")
    return DEFAULT_SUBJECT


def settings() -> tuple[Optional[str], Optional[str], str]:
    vapid = (os.environ.get("VAPID_PRIVATE_KEY") or "").strip() or None
    sub_key = (os.environ.get("PUSH_SUBSCRIPTION_KEY") or "").strip() or None
    return vapid, sub_key, vapid_subject()


def configured() -> bool:
    vapid, sub_key, _ = settings()
    return bool(vapid and sub_key)


def _load_dead(path: str) -> set:
    try:
        with open(path, encoding="utf-8") as handle:
            dead = json.load(handle)
    except (OSError, ValueError):
        return set()
    return set(dead) if isinstance(dead, list) else set()


def _save_dead(dead: set, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sorted(dead), handle)
    except OSError:
        pass


def send_to_all(
    title: str,
    body: str,
    store_path: str = SUBSCRIPTIONS_PATH,
    dead_path: str = DEAD_PATH,
    sender: Any = None,
) -> int:
    """Push to every stored subscription. Returns how many were delivered."""
    if not configured():
        return 0
    vapid, sub_key, subject = settings()
    if sender is None:
        from pywebpush import webpush as sender  # imported lazily: only the VPS sends

    dead = _load_dead(dead_path)
    newly_dead = set()
    delivered = 0
    data = json.dumps({"title": title, "body": body})
    for record in load_store(store_path):
        sub_id = record.get("id")
        if sub_id in dead:
            continue
        try:
            subscription = decrypt_subscription(record, sub_key)
        except Exception as exc:  # noqa: BLE001 - one bad record never blocks the rest
            logging.error(f"Web push: skipping an unreadable subscription ({type(exc).__name__})")
            continue
        try:
            sender(
                subscription_info=subscription, data=data, vapid_private_key=vapid,
                vapid_claims={"sub": subject}, ttl=PUSH_TTL_SECONDS, timeout=HTTP_TIMEOUT,
            )
            delivered += 1
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (404, 410):
                newly_dead.add(sub_id)
            else:
                logging.error(f"Web push failed: {sanitize(type(exc).__name__ + ': ' + str(exc))[:300]}")
    if newly_dead:
        _save_dead(dead | newly_dead, dead_path)
    return delivered


# ------------------------------------------------------------------- CLI


def _cmd_setup(args: argparse.Namespace) -> int:
    keys = generate_keys()
    os.makedirs(os.path.dirname(args.config) or ".", exist_ok=True)
    with open(args.config, "w", encoding="utf-8") as handle:
        json.dump({
            "vapid_public_key": keys["vapid_public_key"],
            "subscription_public_key": keys["subscription_public_key"],
        }, handle, indent=2)
    print(f"Wrote the public keys to {args.config} (commit and push it).\n")
    print("Add these two lines to /opt/tradingbot/.env (never commit them):\n")
    print(f"VAPID_PRIVATE_KEY={keys['vapid_private_key']}")
    print(f"PUSH_SUBSCRIPTION_KEY={keys['subscription_private_key']}")
    return 0


def _cmd_store(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(os.environ.get("PAYLOAD") or "null")
        subs = apply_change(load_store(args.path), args.action, payload)
    except (ValueError, TypeError) as exc:
        print(f"::error::rejected: {exc}", file=sys.stderr)
        return 1
    write_store(subs, args.path)
    print(f"{args.action}: {len(subs)} subscription(s) stored.")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    if not configured():
        print("VAPID_PRIVATE_KEY / PUSH_SUBSCRIPTION_KEY are not set.", file=sys.stderr)
        return 1
    sent = send_to_all("TradingBot Papin", "Prova de notificacio")
    print(f"Delivered to {sent} of {len(load_store())} subscription(s).")
    return 0 if sent else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Web Push for the TradingBot dashboard.")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="generate keys (once, on the VPS)")
    setup.add_argument("--config", default=CONFIG_PATH)
    store = sub.add_parser("store", help="apply a subscribe/unsubscribe (push_subscriptions.yml)")
    store.add_argument("--action", required=True, choices=["push_subscribe", "push_unsubscribe"])
    store.add_argument("--path", default=SUBSCRIPTIONS_PATH)
    sub.add_parser("test", help="send a test notification to every subscription")
    args = parser.parse_args(argv)
    return {"setup": _cmd_setup, "store": _cmd_store, "test": _cmd_test}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
