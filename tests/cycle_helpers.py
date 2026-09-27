"""Shared stubs for tests that drive main._process_symbol end to end, offline."""

import pandas as pd

import data_fetcher
import execution
from mode import ModeSettings
from models import SignalOutput, TechnicalIndicators

INDICATORS = TechnicalIndicators(
    rsi_14=55.0,
    sma_20=100.0,
    sma_50=98.0,
    price_change_pct=1.0,
    volume_change_pct=10.0,
    trend_slope=0.2,
    sma_20_vs_50_pct=2.0,
    price_vs_sma_20_pct=0.5,
)


def settings(is_live=False, min_confidence=0.5):
    return ModeSettings(is_live=is_live, system_prompt="test prompt", min_confidence=min_confidence)


def stub_market(monkeypatch, prices):
    """`prices` maps symbol -> price, or symbol -> list of prices returned in turn
    (the approval gate re-fetches the price after an approval)."""
    queues = {s: (list(p) if isinstance(p, list) else [p]) for s, p in prices.items()}

    def fake_fetch(symbol, period=None, interval="1d"):
        queue = queues[symbol]
        price = queue.pop(0) if len(queue) > 1 else queue[0]
        return pd.DataFrame({"Close": [price], "Volume": [1.0]})

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(data_fetcher, "compute_indicators", lambda df: INDICATORS)
    monkeypatch.setattr(data_fetcher, "fetch_headlines", lambda symbol: [])


def buy_signal(symbol, price, stop_pct=0.01, reward_risk=3.0, confidence=0.9):
    stop = price * (1 - stop_pct)
    take = price + (price - stop) * reward_risk
    return SignalOutput(
        symbol=symbol,
        action="buy",
        confidence=confidence,
        position_size_pct=5.0,
        stop_loss_price=stop,
        take_profit_price=take,
        reasoning="prova",
    )


def record_executions(monkeypatch):
    """Replace execution.execute_trade with a recorder that returns a dry-run fill."""
    from models import ExecutionResult

    calls = []

    def fake_execute(signal, current_price, live_equity, is_live, existing_position=None):
        calls.append({"signal": signal, "price": current_price, "is_live": is_live,
                      "existing_position": existing_position})
        return ExecutionResult(status="dry_run", message="[test]", qty=1.0, fill_price=current_price)

    monkeypatch.setattr(execution, "execute_trade", fake_execute)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: None)
    return calls
