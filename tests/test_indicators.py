"""Unit tests for technical indicators."""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from kabushiki.indicators import compute_indicators, extract_latest_snapshot
from kabushiki.market_data import JST


def _make_ohlcv(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    start = datetime(2026, 11, 20, 9, 0, tzinfo=JST)
    idx = [start + timedelta(minutes=15 * i) for i in range(len(prices))]
    vols = volumes or [10000.0] * len(prices)
    rows = []
    for p, v in zip(prices, vols):
        rows.append(
            {
                "Open": p * 0.999,
                "High": p * 1.003,
                "Low": p * 0.997,
                "Close": p,
                "Volume": v,
            }
        )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_compute_indicators_uptrend():
    prices = [1000.0 + i * 5.0 for i in range(40)]
    df = _make_ohlcv(prices)
    ind = compute_indicators(df)
    snap = extract_latest_snapshot(ind)

    assert snap is not None
    assert snap.ema5 > snap.ema20
    assert snap.rsi14 > 70.0
    assert snap.macd_hist >= 0.0
    assert snap.atr14 > 0.0
    assert snap.vwap > 0.0
    assert snap.close == prices[-1]


def test_compute_indicators_flat_series_no_nan():
    prices = [500.0] * 30
    df = _make_ohlcv(prices, volumes=[0.0] * 30)
    ind = compute_indicators(df)
    snap = extract_latest_snapshot(ind)

    assert snap is not None
    assert not np.isnan(snap.rsi14)
    assert snap.rsi14 == 50.0
    assert snap.bb_pct_b == 0.5
    assert snap.vwap == 500.0
