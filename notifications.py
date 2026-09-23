import json
import logging
from typing import List, Dict, Any

from secrets_redaction import sanitize as _sanitize

def send_alert(subject: str, message: str, config: dict = None) -> None:
    if config is None:
        config = {}
    
    # Try Web Push first
    web_push_config = config.get("web_push", {})
    if web_push_config and web_push_config.get("enabled", False):
        try:
            # Fake web push logic for now (mocked in tests)
            _send_web_push(subject, message, web_push_config)
        except Exception as e:
            logging.error(f"Web push failed: {e}")
            # Do not break cycle
    
    # Fallback to Telegram if enabled
    telegram_config = config.get("telegram", {})
    if telegram_config and telegram_config.get("enabled", False):
        try:
            _send_telegram(subject, message, telegram_config)
        except Exception as e:
            logging.error(f"Telegram push failed: {e}")

def _send_web_push(subject: str, message: str, config: dict) -> None:
    pass

def _send_telegram(subject: str, message: str, config: dict) -> None:
    pass

def send_screening_complete_alert(is_live: bool, symbols: List[str], error: Exception = None) -> None:
    pass

def send_screening_failure_alert(is_live: bool, reason: str) -> None:
    pass

def send_volume_wake_alert(symbol: str, price: float, trigger_reasons: List[str], wake_action: str) -> None:
    pass
