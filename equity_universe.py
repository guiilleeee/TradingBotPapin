"""Equity universe (Nasdaq-50) and screening signal (yfinance).

This module replaces the legacy S&P 500 + Nasdaq-100 logic. The target universe is
now strictly the top 50 non-financial Nasdaq-100 constituents by market cap.

FMP provides sector and market cap data on its constituent endpoints, so the
pipeline is:
1. Fetch Nasdaq-100 constituents from FMP.
2. Filter out financials (Financial Services sector).
3. Sort by market cap descending and take the top 50.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Sequence, Set

import pandas as pd
import requests
import yfinance as yf

from secrets_redaction import sanitize

FMP_BASE_URL = "https://financialmodelingprep.com"
HTTP_TIMEOUT = 30.0
INTER_CALL_DELAY_SECONDS = 0.4

NASDAQ_100_MAX_PLAUSIBLE_SIZE = 160

# We need the top 50 non-financial constituents
TARGET_UNIVERSE_SIZE = 50

# Minimum volume floor for equities (defense in depth).
MIN_EQUITY_VOLUME = 100_000.0

VOLUME_WEIGHT = 0.6
MOMENTUM_WEIGHT = 0.4


class FMPError(RuntimeError):
    """Raised for a hard FMP failure the caller should know about (missing key)."""


def _api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise FMPError(
            "FMP_API_KEY is not set. Sign up for a free key at "
            "financialmodelingprep.com and set it as the FMP_API_KEY environment "
            "variable / GitHub secret."
        )
    return key


def _get(path: str, params: dict | None = None) -> Any:
    query = dict(params or {})
    query["apikey"] = _api_key()
    try:
        resp = requests.get(f"{FMP_BASE_URL}{path}", params=query, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise FMPError(f"FMP {path}: {data['Error Message']}")
    except Exception as exc:
        raise FMPError(sanitize(f"{type(exc).__name__}: {exc}")) from None
    time.sleep(INTER_CALL_DELAY_SECONDS)
    return data


def fetch_nasdaq50() -> List[str]:
    """Fetch Nasdaq-100, filter financials, sort by market cap, take top 50."""
    for path in ("/stable/nasdaq-constituent", "/api/v3/nasdaq_constituent"):
        try:
            data = _get(path)
            if not isinstance(data, list) or not data:
                continue

            if len(data) > NASDAQ_100_MAX_PLAUSIBLE_SIZE:
                return []

            # Filter out financial services
            non_financials = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                sector = str(row.get("sector", "")).lower()
                if "financial" in sector:
                    continue
                # FMP might return market_cap or marketCap depending on the endpoint schema.
                # Default to 0 if missing so it falls to the bottom of the sort.
                mcap = row.get("marketCap") or row.get("market_cap") or 0.0
                non_financials.append({"symbol": sym, "mcap": float(mcap)})
            
            # Sort by market cap descending and take top 50
            non_financials.sort(key=lambda x: x["mcap"], reverse=True)
            return [x["symbol"] for x in non_financials[:TARGET_UNIVERSE_SIZE]]
        except Exception:
            continue

    return []


def build_equity_universe() -> Set[str]:
    """Nasdaq-50 universe.

    Never empty by construction unless the fetch fails, in which case the caller
    falls back to whatever symbols.yaml or config.yaml already has.
    """
    return set(fetch_nasdaq50())


_MIN_TRADING_DAYS_FOR_MOMENTUM = 2
PRICE_DATA_FETCH_PERIOD = "5d"


def fetch_universe_price_data(symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    symbols = list(symbols)
    if not symbols:
        return {}

    try:
        df = yf.download(
            symbols,
            period=PRICE_DATA_FETCH_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    present = {c[0] for c in df.columns} if isinstance(df.columns, pd.MultiIndex) else set()
    
    # If there's only one symbol, yf.download doesn't use a MultiIndex
    if len(symbols) == 1 and not isinstance(df.columns, pd.MultiIndex):
        present = {symbols[0]}
        df = pd.concat([df], keys=symbols, axis=1)

    out: Dict[str, Dict[str, float]] = {}
    for symbol in symbols:
        if symbol not in present:
            continue
        try:
            close = df[symbol]["Close"].dropna()
            volume = df[symbol]["Volume"].dropna()
        except KeyError:
            continue
        if len(close) < _MIN_TRADING_DAYS_FOR_MOMENTUM or volume.empty:
            continue

        prev_close = float(close.iloc[-2])
        if prev_close == 0:
            continue

        out[symbol] = {
            "price_change_pct": (float(close.iloc[-1]) / prev_close - 1.0) * 100.0,
            "volume": float(volume.iloc[-1]),
        }

    return out


def percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    unique_sorted = sorted(set(values.values()))
    if len(unique_sorted) == 1:
        return {symbol: 1.0 for symbol in values}
    rank_of_value = {v: i / (len(unique_sorted) - 1) for i, v in enumerate(unique_sorted)}
    return {symbol: rank_of_value[v] for symbol, v in values.items()}


def score_equities(
    universe: Set[str], price_data: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    volume_by_symbol: Dict[str, float] = {}
    momentum_by_symbol: Dict[str, float] = {}
    for symbol, data in price_data.items():
        if symbol not in universe:
            continue
        volume = data.get("volume")
        if isinstance(volume, (int, float)) and volume >= MIN_EQUITY_VOLUME:
            volume_by_symbol[symbol] = float(volume)
        pct = data.get("price_change_pct")
        if isinstance(pct, (int, float)):
            momentum_by_symbol[symbol] = abs(float(pct))

    volume_pct = percentile_ranks(volume_by_symbol)
    momentum_pct = percentile_ranks(momentum_by_symbol)

    results = []
    for symbol in sorted(universe):
        v = volume_pct.get(symbol, 0.0)
        m = momentum_pct.get(symbol, 0.0)
        results.append(
            {
                "symbol": symbol,
                "score": VOLUME_WEIGHT * v + MOMENTUM_WEIGHT * m,
                "volume": volume_by_symbol.get(symbol),
                "momentum_pct": momentum_by_symbol.get(symbol),
                "has_signal": symbol in volume_by_symbol or symbol in momentum_by_symbol,
            }
        )

    results.sort(key=lambda r: (r["score"], r["symbol"]), reverse=True)
    return results


def select_top_equities(scored: Sequence[Dict[str, Any]], count: int) -> List[str]:
    with_signal = [r for r in scored if r["has_signal"]]
    without_signal = [r for r in scored if not r["has_signal"]]
    ordered = with_signal + without_signal
    return [r["symbol"] for r in ordered[:count]]
