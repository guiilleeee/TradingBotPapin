"""Market data and derived indicators.

Everything here is read-only and failure-tolerant in one specific direction:
a headline fetch may fail silently, price data may not.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from models import TechnicalIndicators

# 120, not 60. yfinance has interpreted the "Nd" period both ways across versions
# -- as N calendar days (where equities trade only ~5/7 of them, so 60d yields
# roughly 42 bars, fewer than the 50 SMA-50 needs, which is what crashed the
# indicator step) and, as of 1.x, as N returned bars. 120 is safe under either
# reading: ~85 bars on the calendar interpretation, 120 on the bar interpretation,
# both comfortably clear of the 51-bar floor. The compute_indicators guard below
# is what actually enforces it; this is just a sensible default.
#
# The lookback is deliberately identical for equities and crypto. Crypto trades
# 7 days a week and was never affected, but a longer window costs nothing and one
# code path beats a branch that can only ever be wrong for one side.
DEFAULT_PERIOD = "120d"

RSI_PERIOD = 14
ATR_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50

# Headlines older than this are dropped: a week-old story is already in the price,
# and without a date the model can't tell it from this morning's.
HEADLINE_MAX_AGE_HOURS = 72

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={q}&region=US&lang=en-US"
# Yahoo answers 429 Too Many Requests to a bare requests/urllib User-Agent, on the
# very first call. Without this header the feed never returns anything and the
# try/except below quietly hands the model an empty headline list forever.
_HEADLINE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}


def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV bars for `symbol`.

    The dropna on Close is not optional: yfinance routinely returns a trailing row
    for the current, still-incomplete session whose Close is NaN. Left in place it
    poisons the last-bar reads below.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError(f"{symbol}: yfinance returned no bars for period={period} interval={interval}")

    if "Close" not in df.columns:
        raise ValueError(f"{symbol}: yfinance response has no Close column (got {list(df.columns)})")

    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"{symbol}: every returned bar had a NaN Close")

    return df


def _wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """RSI with Wilder's smoothing, seeded from the first `period` changes.

    Written out rather than pulled from TA-Lib -- one function is not worth a
    native dependency that has to be built on every CI runner.
    """
    delta = close.diff().dropna()
    if len(delta) < period:
        raise ValueError(
            f"RSI-{period} needs at least {period + 1} bars, got {len(delta) + 1}"
        )

    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Seed: simple mean of the first `period` gains/losses.
    avg_gain = float(gains.iloc[:period].mean())
    avg_loss = float(losses.iloc[:period].mean())

    # Then smooth forward across the remainder of the series.
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> Optional[float]:
    """Average True Range with Wilder's smoothing, or None without High/Low data.

    Same seeding as _wilder_rsi: a simple mean of the first `period` true ranges,
    then smoothed forward.
    """
    if not {"High", "Low", "Close"} <= set(df.columns):
        return None
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = df["Close"].astype(float).shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1).iloc[1:].dropna()
    if len(true_range) < period:
        return None
    atr = float(true_range.iloc[:period].mean())
    for value in true_range.iloc[period:]:
        atr = (atr * (period - 1) + float(value)) / period
    return atr if np.isfinite(atr) else None


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    sma_short: int = SMA_SHORT,
    sma_long: int = SMA_LONG,
) -> TechnicalIndicators:
    """Derive the indicator bundle from OHLCV bars.

    The length guard is explicit and up front. Without it a short frame produces
    NaN SMAs that surface much later as an opaque Pydantic validation error on
    `sma_50` -- which tells you nothing about the real cause.
    """
    required = sma_long + 1
    if len(df) < required:
        raise ValueError(
            f"Not enough bars to compute indicators: have {len(df)}, need {required} "
            f"(SMA-{sma_long} plus one prior bar for the change columns); "
            f"short by {required - len(df)}. Widen the fetch period."
        )

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series([0.0] * len(df))

    sma_s = float(close.rolling(sma_short).mean().iloc[-1])
    sma_l = float(close.rolling(sma_long).mean().iloc[-1])

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    price_change_pct = ((last_close / prev_close) - 1.0) * 100.0 if prev_close else 0.0

    last_vol = float(volume.iloc[-1])
    prev_vol = float(volume.iloc[-2])
    volume_change_pct = ((last_vol / prev_vol) - 1.0) * 100.0 if prev_vol else 0.0

    # Trend slope: linear regression of closing prices over the last `sma_short` bars,
    # normalised to percent-per-day to be comparable across price levels.
    slope_window = min(sma_short, len(close))
    recent = close.iloc[-slope_window:].values.astype(float)
    x = np.arange(slope_window, dtype=float)
    if slope_window >= 2 and np.std(x) > 0:
        slope = np.polyfit(x, recent, 1)[0]  # units per bar (day)
        trend_slope = (slope / recent[-1]) * 100.0 if recent[-1] != 0 else 0.0
    else:
        trend_slope = 0.0

    # SMA cross-over: how far SMA-20 is above/below SMA-50, as a percentage.
    sma_20_vs_50_pct = ((sma_s - sma_l) / sma_l) * 100.0 if sma_l != 0 else 0.0

    # Price vs short MA: how far the current price is from SMA-20.
    price_vs_sma_20_pct = ((last_close - sma_s) / sma_s) * 100.0 if sma_s != 0 else 0.0

    return TechnicalIndicators(
        rsi_14=_wilder_rsi(close, rsi_period),
        sma_20=sma_s,
        sma_50=sma_l,
        price_change_pct=price_change_pct,
        volume_change_pct=volume_change_pct,
        trend_slope=trend_slope,
        sma_20_vs_50_pct=sma_20_vs_50_pct,
        price_vs_sma_20_pct=price_vs_sma_20_pct,
        atr_14=_wilder_atr(df),
    )


def latest_price(df: pd.DataFrame) -> float:
    """Close of the last complete bar."""
    return float(df["Close"].astype(float).iloc[-1])


def fetch_headlines(
    symbol: str,
    limit: int = 5,
    timeout: float = 10.0,
    max_age_hours: float = HEADLINE_MAX_AGE_HOURS,
    now: Optional[datetime] = None,
) -> List[str]:
    """Recent Yahoo Finance headlines for `symbol`, newest `max_age_hours` only.

    Each one is prefixed with its age ("[5h ago] ...") so the model can weigh
    fresh news over stale. An item without a parseable pubDate is dropped:
    its freshness can't be checked.

    Never raises. A news outage is not a reason to skip a trading decision, so the
    caller gets an empty list and the model is told there are no headlines.
    """
    try:
        resp = requests.get(
            _YAHOO_RSS.format(q=symbol), timeout=timeout, headers=_HEADLINE_HEADERS
        )
        resp.raise_for_status()
        return parse_headlines(resp.content, limit=limit, max_age_hours=max_age_hours, now=now)
    except Exception:
        return []


def parse_headlines(
    xml_bytes: bytes,
    limit: int = 5,
    max_age_hours: float = HEADLINE_MAX_AGE_HOURS,
    now: Optional[datetime] = None,
) -> List[str]:
    """The RSS parsing half of fetch_headlines, separate so it tests offline."""
    now = now or datetime.now(timezone.utc)
    root = ET.fromstring(xml_bytes)
    titles: List[str] = []
    for item in root.iter("item"):
        title: Optional[ET.Element] = item.find("title")
        published: Optional[ET.Element] = item.find("pubDate")
        if title is None or not title.text or published is None or not published.text:
            continue
        try:
            when = parsedate_to_datetime(published.text.strip())
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age_hours = (now - when).total_seconds() / 3600.0
        if age_hours > max_age_hours:
            continue
        titles.append(f"[{max(int(age_hours), 0)}h ago] {title.text.strip()}")
        if len(titles) >= limit:
            break
    return titles


# ------------------------------------------------------------------- earnings


def fetch_days_to_earnings(symbol: str, today: Optional[date] = None) -> Optional[int]:
    """Calendar days until the next earnings report, or None if unknown.

    Never raises: a missing date just means the earnings rule can't fire.
    """
    try:
        calendar = yf.Ticker(symbol).calendar
    except Exception:
        return None
    return days_to_earnings_from_calendar(calendar, today)


def days_to_earnings_from_calendar(calendar: Any, today: Optional[date] = None) -> Optional[int]:
    """Parse yfinance's `Ticker.calendar` (a dict in yfinance 1.x; a DataFrame in
    older versions) into days until the nearest earnings date not in the past."""
    today = today or datetime.now(timezone.utc).date()
    raw: Any = None
    try:
        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date")
        elif isinstance(calendar, pd.DataFrame) and "Earnings Date" in calendar.index:
            raw = list(calendar.loc["Earnings Date"].values)
    except Exception:
        return None
    if raw is None:
        return None
    candidates: Iterable[Any] = raw if isinstance(raw, (list, tuple)) else [raw]

    upcoming = []
    for value in candidates:
        try:
            day = pd.Timestamp(value).date()
        except (TypeError, ValueError):
            continue
        if day >= today:
            upcoming.append((day - today).days)
    return min(upcoming) if upcoming else None
