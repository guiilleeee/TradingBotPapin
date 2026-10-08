"""Equity universe (S&P 500 top 25 by market cap) and screening signal (yfinance).

Pipeline, run weekly by screening.py:
1. Fetch S&P 500 constituents from FMP, falling back to Wikipedia's public
   constituent table (no key) when FMP's endpoints are restricted on the
   current plan.
2. Filter out financials (Financials / Financial Services sector).
3. Rank by market cap and take the top 25. The whole non-financial list is the
   "pool" funnel.prefilter scans each cycle for breakouts outside those 25.

Market cap: neither source carries it, so every constituent without a positive
market cap gets one from yfinance (in parallel -- ~500 lookups), and if too few
constituents end up with a real market cap the whole fetch returns [] so
screening.py leaves last week's symbols.yaml untouched rather than publishing an
arbitrary list.
"""

from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Set

import pandas as pd
import requests
import yfinance as yf

from secrets_redaction import sanitize

FMP_BASE_URL = "https://financialmodelingprep.com"
HTTP_TIMEOUT = 30.0
INTER_CALL_DELAY_SECONDS = 0.4
MARKET_CAP_LOOKUP_WORKERS = 8

SP500_MAX_PLAUSIBLE_SIZE = 560
SP500_MIN_PLAUSIBLE_SIZE = 450

# No-key fallback for S&P 500 membership and GICS sector. Wikipedia rejects
# requests without a User-Agent.
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0 (TradingBotPapin weekly screen)"}

# The top 25 non-financial S&P 500 constituents by market cap.
TARGET_UNIVERSE_SIZE = 25

# Secondary share class -> primary. Dropped when the primary is also a constituent.
SECONDARY_SHARE_CLASSES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}

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


def _yfinance_market_cap(symbol: str) -> float:
    """Market cap from yfinance, 0.0 on any failure."""
    try:
        # Item access, not .get(): FastInfo.get("market_cap") returns None on
        # yfinance 1.7 even though info["market_cap"] is populated.
        return float(yf.Ticker(symbol).fast_info["market_cap"] or 0.0)
    except Exception:
        return 0.0


def rank_by_market_cap(
    rows: List[Dict[str, Any]],
    size: int,
    market_cap_lookup=None,
    keep_all: bool = False,
) -> List[str]:
    """Top `size` symbols by market cap, filling gaps from `market_cap_lookup`.

    Returns [] unless at least `size` rows end up with a positive market cap -- an
    arbitrary list must never be published as "the largest companies".
    `keep_all` returns every ranked symbol instead of only the first `size`.
    """
    # One slot per company: a second share class would double the exposure to
    # one issuer while crowding out the 25th-largest company.
    present = {r["symbol"] for r in rows}
    rows = [
        r for r in rows
        if not (r["symbol"] in SECONDARY_SHARE_CLASSES
                and SECONDARY_SHARE_CLASSES[r["symbol"]] in present)
    ]
    missing = [row for row in rows if not row["mcap"] or row["mcap"] <= 0]
    if missing:
        market_cap_lookup = market_cap_lookup or _yfinance_market_cap
        with ThreadPoolExecutor(max_workers=MARKET_CAP_LOOKUP_WORKERS) as executor:
            caps = executor.map(lambda r: market_cap_lookup(r["symbol"]), missing)
            for row, mcap in zip(missing, caps):
                row["mcap"] = mcap
    ranked = sorted((r for r in rows if r["mcap"] > 0), key=lambda r: r["mcap"], reverse=True)
    if len(ranked) < size:
        return []
    return [r["symbol"] for r in (ranked if keep_all else ranked[:size])]


def _normalize_symbol(raw: Any) -> str:
    # BRK.B / BRK/B -> BRK-B, the form yfinance and the rest of the pipeline use.
    if raw is None or pd.isna(raw):  # an empty table cell reads as NaN
        return ""
    return str(raw).strip().upper().replace(".", "-").replace("/", "-")


def _check_plausible_size(source: str, count: int) -> None:
    if not SP500_MIN_PLAUSIBLE_SIZE <= count <= SP500_MAX_PLAUSIBLE_SIZE:
        raise RuntimeError(
            f"{source}: {count} constituents, expected "
            f"{SP500_MIN_PLAUSIBLE_SIZE}-{SP500_MAX_PLAUSIBLE_SIZE} -- not the S&P 500"
        )


def _fmp_sp500_rows(path: str) -> List[Dict[str, Any]]:
    """Non-financial S&P 500 rows ({symbol, mcap}) from one FMP endpoint.

    Raises on any failure (HTTP error, empty or implausible list) so the caller
    can log the reason and move on to the next source.
    """
    data = _get(path)
    if not isinstance(data, list) or not data:
        raise FMPError(f"FMP {path}: empty or non-list response")
    _check_plausible_size(f"FMP {path}", len(data))

    non_financials = []
    for row in data:
        if not isinstance(row, dict):
            continue
        sym = _normalize_symbol(row.get("symbol"))
        if not sym:
            continue
        sector = str(row.get("sector", "")).lower()
        if "financial" in sector:
            continue
        # FMP might return market_cap or marketCap depending on the endpoint schema.
        # Default to 0 if missing so rank_by_market_cap fills it from yfinance.
        mcap = row.get("marketCap") or row.get("market_cap") or 0.0
        try:
            mcap = float(mcap)
        except (TypeError, ValueError):
            mcap = 0.0
        non_financials.append({"symbol": sym, "mcap": mcap})
    return non_financials


def _fetch_sp500_from_wikipedia() -> List[Dict[str, Any]]:
    """Non-financial S&P 500 rows ({symbol, mcap}) from Wikipedia's constituent
    table -- no key. Market cap is left at 0 for rank_by_market_cap to fill.

    Raises on a bad response, a missing column, or a list whose size is not
    plausibly the S&P 500 -- never returns a silently wrong universe.
    """
    resp = requests.get(SP500_WIKIPEDIA_URL, headers=WIKIPEDIA_HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    try:
        # flavor pinned: unpinned, a page with no table makes pandas fall back to
        # bs4/html5lib, and an ImportError there would skip the clean RuntimeError.
        table = pd.read_html(io.StringIO(resp.text), match="Symbol", flavor="lxml")[0]
        symbols = table["Symbol"]
        sectors = table["GICS Sector"]
    except (ValueError, KeyError, IndexError, ImportError) as exc:
        raise RuntimeError(f"wikipedia: unexpected page shape ({type(exc).__name__}: {exc})") from None

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for raw_symbol, sector in zip(symbols, sectors):
        sym = _normalize_symbol(raw_symbol)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        if "financial" in str(sector).lower():
            continue
        out.append({"symbol": sym, "mcap": 0.0})

    # Size-check the whole listing, before financials were dropped.
    _check_plausible_size("wikipedia", len(seen))
    return out


def fetch_sp500_top(size: int = TARGET_UNIVERSE_SIZE, keep_all: bool = False) -> List[str]:
    """Fetch the S&P 500, filter financials, rank by market cap, take the top `size`
    (or, with `keep_all`, every ranked constituent -- still [] below `size`).

    Sources in order: FMP's two constituent endpoints, then Wikipedia's table.
    Every failed source is printed with its reason (for journalctl) before moving
    on; never raises -- [] if all of them fail.
    """
    sources = [
        (path, lambda path=path: _fmp_sp500_rows(path))
        for path in ("/stable/sp500-constituent", "/api/v3/sp500_constituent")
    ]
    sources.append(("wikipedia", _fetch_sp500_from_wikipedia))

    for name, fetch_rows in sources:
        try:
            top = rank_by_market_cap(fetch_rows(), size, keep_all=keep_all)
        except Exception as exc:
            print(sanitize(f"  sp500 constituents: {name} failed: {exc}"))
            continue
        if top:
            return top
        print(f"  sp500 constituents: {name} failed: fewer than {size} constituents with a market cap")

    print("  sp500 constituents: every source failed, returning []")
    return []


def build_equity_universe() -> List[str]:
    """Top-25 S&P 500 universe, largest market cap first.

    A list, not a set: the order is meaningful (the funnel breaks score ties by
    it). Empty only if the fetch fails, in which case the caller falls back to
    whatever symbols.yaml or config.yaml already has.
    """
    return fetch_sp500_top(TARGET_UNIVERSE_SIZE)


def build_equity_pool() -> List[str]:
    """Every non-financial S&P 500 constituent, largest market cap first.

    The top TARGET_UNIVERSE_SIZE of it are the week's universe; the whole list is
    what funnel.prefilter scans each scheduled cycle for breakouts outside it.
    Same all-or-nothing rule: [] if fewer than TARGET_UNIVERSE_SIZE rank.
    """
    return fetch_sp500_top(TARGET_UNIVERSE_SIZE, keep_all=True)


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
