"""External non-Yahoo historical market & macro data collectors (FRED, Frankfurter ECB, optional J-Quants)."""

from __future__ import annotations

from datetime import datetime, timedelta
import io
import logging
import os
from typing import Any
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Official St. Louis Fed (FRED) public CSV series (free, no API key required)
FRED_MACRO_SERIES = {
    "FRED_USDJPY": ("DEXJPUS", "ドル円為替 (FRED DEXJPUS)"),
    "FRED_VIX": ("VIXCLS", "米国VIX恐怖指数 (FRED VIXCLS)"),
    "FRED_US10Y": ("DGS10", "米国10年債利回り (FRED DGS10)"),
    "FRED_NIKKEI": ("NIKKEI225", "日経225日足 (FRED NIKKEI225)"),
}


def fetch_fred_series(series_code: str, lookback_years: int = 5) -> pd.DataFrame:
    """Fetch daily historical macro data from St. Louis Fed (FRED) official public CSV."""
    cutoff = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}&cosd={cutoff}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "KabushikiML/2.0"})
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    if raw.empty or len(raw.columns) < 2:
        return pd.DataFrame(columns=["date", "value"])

    date_col, val_col = raw.columns[0], raw.columns[1]
    df = raw.rename(columns={date_col: "date", val_col: "value"})[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    return df


def fetch_frankfurter_usdjpy(lookback_years: int = 4) -> pd.DataFrame:
    """Fetch historical daily USD/JPY rates from Frankfurter (official ECB open API, no key)."""
    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    url = f"https://api.frankfurter.dev/v1/{start_date}..?base=USD&symbols=JPY"
    resp = requests.get(url, timeout=15, headers={"User-Agent": "KabushikiML/2.0"})
    resp.raise_for_status()
    rates = resp.json().get("rates") or {}
    rows = []
    for d, obj in sorted(rates.items()):
        val = obj.get("JPY")
        if val is not None:
            rows.append({"date": str(d), "value": float(val)})
    return pd.DataFrame(rows, columns=["date", "value"])


def fetch_all_external_macro(lookback_years: int = 5) -> dict[str, dict[str, Any]]:
    """Fetch all non-Yahoo external macro series (FRED + Frankfurter fallback).

    Returns {series_id: {"source": str, "label": str, "df": DataFrame[date, value]}}.
    """
    collected: dict[str, dict[str, Any]] = {}

    for series_id, (fred_code, label) in FRED_MACRO_SERIES.items():
        try:
            df = fetch_fred_series(fred_code, lookback_years=lookback_years)
            if not df.empty:
                collected[series_id] = {
                    "source": "FRED (米連銀公式)",
                    "label": label,
                    "df": df,
                }
        except Exception as exc:  # pragma: no cover
            logger.warning("FRED fetch failed for %s: %s", fred_code, exc)

    # Frankfurter ECB fallback / supplement for USD/JPY
    if "FRED_USDJPY" not in collected or collected["FRED_USDJPY"]["df"].empty:
        try:
            fx_df = fetch_frankfurter_usdjpy(lookback_years=min(lookback_years, 4))
            if not fx_df.empty:
                collected["FRED_USDJPY"] = {
                    "source": "Frankfurter (ECB公式)",
                    "label": "ドル円為替 (ECB Frankfurter)",
                    "df": fx_df,
                }
        except Exception as exc:  # pragma: no cover
            logger.warning("Frankfurter fetch failed: %s", exc)

    return collected


def build_daily_macro_frame(macro_series_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a strictly causal daily macro feature table indexed by YYYY-MM-DD date string.

    Callers MUST shift by +1 trading day (or match strictly on `date < bar_date`) when joining
    onto intraday TSE bars so zero future daily closes leak into intraday predictions.
    """
    if not macro_series_map:
        return pd.DataFrame()

    cols: dict[str, pd.Series] = {}
    for key, df in macro_series_map.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        cols[key] = s

    if not cols:
        return pd.DataFrame()

    merged = pd.DataFrame(cols).sort_index().ffill()
    out = pd.DataFrame(index=merged.index)

    if "FRED_USDJPY" in merged.columns:
        out["usdjpy_chg5"] = merged["FRED_USDJPY"].pct_change(5).fillna(0.0)
    else:
        out["usdjpy_chg5"] = 0.0

    if "FRED_VIX" in merged.columns:
        out["vix_level"] = merged["FRED_VIX"].fillna(20.0)
        out["vix_chg5"] = merged["FRED_VIX"].pct_change(5).fillna(0.0)
    else:
        out["vix_level"] = 20.0
        out["vix_chg5"] = 0.0

    if "FRED_US10Y" in merged.columns:
        out["us10y_level"] = merged["FRED_US10Y"].fillna(4.0)
        out["us10y_chg5"] = merged["FRED_US10Y"].diff(5).fillna(0.0)
    else:
        out["us10y_level"] = 4.0
        out["us10y_chg5"] = 0.0

    out.index = pd.to_datetime(out.index)
    return out.sort_index()
