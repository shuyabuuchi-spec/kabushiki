"""Yahoo Finance market data fetcher and TSE trading session helpers."""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta
from typing import Iterable
import logging
import requests
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# Fixed Japanese market closures (month, day). TSE is closed Dec 31 - Jan 3.
FIXED_TSE_HOLIDAYS = {
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 11),
    (2, 23),
    (4, 29),
    (5, 3),
    (5, 4),
    (5, 5),
    (8, 11),
    (11, 3),
    (11, 23),
    (12, 31),
}

# Yahoo Finance interval mapping & max lookback days per native interval
YAHOO_INTERVAL_SPEC = {
    "5m": ("5m", 58),
    "10m": ("5m", 58),   # Resampled cleanly from real 5m bars
    "15m": ("15m", 58),
    "30m": ("30m", 58),
    "60m": ("60m", 700),
    "1d": ("1d", 1800),
}


def now_jst() -> datetime:
    return datetime.now(JST)


def is_tse_trading_day(dt: datetime | None = None) -> bool:
    """Return True if dt (in JST) is a weekday and not a fixed TSE market holiday."""
    d = (dt or now_jst()).astimezone(JST)
    if d.weekday() >= 5:
        return False
    if (d.month, d.day) in FIXED_TSE_HOLIDAYS:
        return False
    return True


def is_tse_market_open(dt: datetime | None = None, include_buffer_minutes: int = 5) -> bool:
    """Return True if TSE is currently in session (09:00-11:30 or 12:30-15:30 JST).

    Note: TSE extended closing time to 15:30 JST in November 2024.
    """
    d = (dt or now_jst()).astimezone(JST)
    if not is_tse_trading_day(d):
        return False
    t = d.time()
    morning_open = time(9, 0)
    morning_close = time(11, 30)
    afternoon_open = time(12, 30)
    afternoon_close = time(15, min(59, 30 + include_buffer_minutes))
    return (morning_open <= t <= morning_close) or (afternoon_open <= t <= afternoon_close)


def tse_market_status(dt: datetime | None = None) -> dict[str, object]:
    """Return human-readable TSE market status in Japanese."""
    d = (dt or now_jst()).astimezone(JST)
    if not is_tse_trading_day(d):
        return {
            "is_open": False,
            "session": "休場（土日・祝日）",
            "jst_time": d.isoformat(),
        }
    t = d.time()
    if time(9, 0) <= t <= time(11, 30):
        session = "前場 取引中 (09:00-11:30)"
        is_open = True
    elif time(11, 30) < t < time(12, 30):
        session = "昼休み (11:30-12:30)"
        is_open = False
    elif time(12, 30) <= t <= time(15, 30):
        session = "後場 取引中 (12:30-15:30)"
        is_open = True
    elif t < time(9, 0):
        session = "寄り付き前 (09:00開始)"
        is_open = False
    else:
        session = "大引け後 (15:30終了)"
        is_open = False
    return {
        "is_open": is_open,
        "session": session,
        "jst_time": d.isoformat(),
    }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance DataFrame to tz-aware JST OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    rename_map = {}
    for col in out.columns:
        low = str(col).lower()
        if low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"
    out = out.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in required):
        return pd.DataFrame(columns=required)

    out = out[required].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Close"])

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone.utc).tz_convert(JST)
    else:
        out.index = out.index.tz_convert(JST)

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Resample 5m bars to 10m (or any target interval) cleanly."""
    if df.empty or target_interval != "10m":
        return df

    resampled = (
        df.resample("10min", label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "Close"])
    )
    return resampled


def _fetch_via_yahoo_chart_api(symbol: str, native_interval: str, period_days: int) -> pd.DataFrame:
    """Fallback fetcher directly querying Yahoo Finance v8 Chart JSON API."""
    range_str = f"{min(max(period_days, 1), 59)}d"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": native_interval,
        "range": range_str,
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; KabushikiSim/1.0)"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("chart", {}).get("result") or []
    if not results:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    res = results[0]
    timestamps = res.get("timestamp") or []
    quotes = (res.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(JST)
    df = pd.DataFrame(
        {
            "Open": quotes.get("open", []),
            "High": quotes.get("high", []),
            "Low": quotes.get("low", []),
            "Close": quotes.get("close", []),
            "Volume": quotes.get("volume", []),
        },
        index=idx,
    )
    return _normalize_ohlcv(df)


class MarketDataProvider:
    """Fetches real Japanese stock OHLCV bars via Yahoo Finance."""

    def __init__(self, prefer_yfinance: bool = True):
        self.prefer_yfinance = prefer_yfinance and (yf is not None)

    def fetch_symbol(
        self,
        symbol: str,
        interval: str = "15m",
        period_days: int = 10,
    ) -> pd.DataFrame:
        native_interval, max_days = YAHOO_INTERVAL_SPEC.get(interval, ("15m", 58))
        days = min(max(period_days, 1), max_days)
        period_str = f"{days}d"

        df = pd.DataFrame()
        if self.prefer_yfinance:
            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(period=period_str, interval=native_interval, auto_adjust=False)
                df = _normalize_ohlcv(raw)
            except Exception as exc:  # pragma: no cover
                logger.warning("yfinance fetch failed for %s: %s; trying chart API", symbol, exc)

        if df.empty:
            try:
                df = _fetch_via_yahoo_chart_api(symbol, native_interval, days)
            except Exception as exc:  # pragma: no cover
                logger.warning("Yahoo chart API fetch failed for %s: %s", symbol, exc)
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        if interval == "10m":
            df = resample_ohlcv(df, "10m")

        return df

    def fetch_many(
        self,
        symbols: Iterable[str],
        interval: str = "15m",
        period_days: int = 10,
    ) -> dict[str, pd.DataFrame]:
        symbol_list = list(dict.fromkeys(symbols))
        if not symbol_list:
            return {}

        native_interval, max_days = YAHOO_INTERVAL_SPEC.get(interval, ("15m", 58))
        days = min(max(period_days, 1), max_days)
        result: dict[str, pd.DataFrame] = {}

        if self.prefer_yfinance and len(symbol_list) > 1:
            try:
                batch = yf.download(
                    tickers=" ".join(symbol_list),
                    period=f"{days}d",
                    interval=native_interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                if batch is not None and not batch.empty:
                    if isinstance(batch.columns, pd.MultiIndex):
                        top_levels = set(batch.columns.get_level_values(0))
                        for sym in symbol_list:
                            if sym in top_levels:
                                sub = _normalize_ohlcv(batch[sym])
                                if interval == "10m":
                                    sub = resample_ohlcv(sub, "10m")
                                if not sub.empty:
                                    result[sym] = sub
                    elif len(symbol_list) == 1:
                        sub = _normalize_ohlcv(batch)
                        if interval == "10m":
                            sub = resample_ohlcv(sub, "10m")
                        if not sub.empty:
                            result[symbol_list[0]] = sub
            except Exception as exc:  # pragma: no cover
                logger.warning("Batch yfinance download failed (%s); falling back per symbol", exc)

        # Fetch any missing symbols individually
        for sym in symbol_list:
            if sym not in result or result[sym].empty:
                df = self.fetch_symbol(sym, interval=interval, period_days=days)
                if not df.empty:
                    result[sym] = df

        return result
