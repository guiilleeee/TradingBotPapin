from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from data_fetcher import compute_indicators, market_day_elapsed_fraction


def _make_sample_df(n_bars=60, last_vol=1000.0, prev_vol=1000.0, last_date=None):
    dates = pd.date_range(end=last_date or "2026-06-01", periods=n_bars, freq="B")
    close = np.linspace(100.0, 150.0, n_bars)
    vol = np.full(n_bars, 1000.0)
    vol[-2] = prev_vol
    vol[-1] = last_vol
    return pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": vol,
    }, index=dates)


def test_market_day_elapsed_fraction_open_and_closed():
    # Wednesday 10:45 AM ET (75 mins into 390 min day = 75/390 = 0.1923)
    t_open = datetime(2026, 6, 3, 10, 45, tzinfo=ZoneInfo("America/New_York"))
    frac = market_day_elapsed_fraction(t_open)
    assert 0.19 < frac < 0.20

    # Wednesday 8:00 AM ET (pre-market -> closed)
    t_early = datetime(2026, 6, 3, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    assert market_day_elapsed_fraction(t_early) == 1.0

    # Wednesday 5:00 PM ET (after-hours -> closed)
    t_late = datetime(2026, 6, 3, 17, 0, tzinfo=ZoneInfo("America/New_York"))
    assert market_day_elapsed_fraction(t_late) == 1.0

    # Saturday (weekend -> closed)
    t_weekend = datetime(2026, 6, 6, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert market_day_elapsed_fraction(t_weekend) == 1.0


def test_compute_indicators_projects_intraday_volume():
    # Wednesday 10:45 AM ET (approx 19.23% of day elapsed)
    now_et = datetime(2026, 6, 3, 10, 45, tzinfo=ZoneInfo("America/New_York"))

    # 200 volume in ~19.2% of the day projects to ~1040, compared to yesterday's 1000
    df = _make_sample_df(n_bars=60, last_vol=200.0, prev_vol=1000.0, last_date="2026-06-03")

    ind = compute_indicators(df, now=now_et)
    # Without projection, 200 / 1000 would be -80.0%
    # With projection, it should be positive (~ +4.0%)
    assert ind.volume_change_pct > 0.0
    assert abs(ind.volume_change_pct - 4.0) < 2.0


def test_compute_indicators_normal_when_market_closed():
    # Wednesday 8:00 PM ET (after market close)
    now_et = datetime(2026, 6, 3, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    df = _make_sample_df(n_bars=60, last_vol=1200.0, prev_vol=1000.0, last_date="2026-06-03")

    ind = compute_indicators(df, now=now_et)
    # Market closed: last_vol is complete, so 1200 / 1000 = +20%
    assert abs(ind.volume_change_pct - 20.0) < 0.1
