import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger import BotLogger  # noqa: E402  (needs the sys.path line above)

REAL_DB = ROOT / "trading_bot.db"


@pytest.fixture(autouse=True)
def no_real_telegram(monkeypatch):
    """The VPS runs pytest with its real .env loaded; without this, any cycle test
    that fires an alert would message the real chat. Tests about Telegram set
    their own fake credentials."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Same for Web Push: without these, web_push.send_to_all is a no-op.
    monkeypatch.delenv("VAPID_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("PUSH_SUBSCRIPTION_KEY", raising=False)
    monkeypatch.delenv("VAPID_ADMIN_EMAIL", raising=False)


@pytest.fixture(autouse=True)
def market_open_by_default(monkeypatch):
    """Pin NYSE to "open" unless a test says otherwise.

    Without this, every buy test depended on the wall clock: run outside
    9:30-16:00 ET (or on a weekend) and a dozen execution tests failed on the
    market-hours gate -- and trading_bot.yml runs pytest before every cycle, so
    an off-hours cycle never got to trade at all. A test about closed-market
    behaviour patches this back itself.
    """
    import execution
    import volume_watch  # imports _is_market_open by name, so patch its copy too

    monkeypatch.setattr(execution, "_is_market_open", lambda: True)
    monkeypatch.setattr(volume_watch, "_is_market_open", lambda: True)


@pytest.fixture(autouse=True)
def no_network_for_cycle_side_data(monkeypatch):
    """Every full cycle now fetches QQQ (benchmark) and a universe-wide price
    download (funnel). Neither may reach the network from a test; a test about
    either patches in its own fixture data.
    """
    import benchmark
    import funnel

    monkeypatch.setattr(benchmark, "fetch_benchmark_price", lambda symbol="QQQ": None)
    monkeypatch.setattr(funnel, "fetch_funnel_data", lambda symbols: {})

    import data_fetcher

    # Earnings dates come from yfinance; a test about them passes its own.
    monkeypatch.setattr(data_fetcher, "fetch_days_to_earnings", lambda symbol, today=None: None)


@pytest.fixture
def tmp_logger(tmp_path):
    """A BotLogger on a throwaway SQLite file.

    The assertion is not decoration: a test that reached the real trading_bot.db
    would rewrite live position state.
    """
    db_path = tmp_path / "test_trading_bot.db"
    assert db_path.resolve() != REAL_DB.resolve()
    assert str(tmp_path) in str(db_path)
    return BotLogger(str(db_path))


@pytest.fixture
def read_signals(tmp_logger):
    """Return every signals row as a dict with the JSON blobs already decoded."""

    def _read():
        conn = sqlite3.connect(tmp_logger.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
        finally:
            conn.close()

        out = []
        for row in rows:
            item = dict(row)
            for key in ("signal_input", "raw_output", "final_signal", "execution_result"):
                item[key] = json.loads(item[key]) if item[key] else None
            out.append(item)
        return out

    return _read
