"""Technical indicator calculations for intraday and daily Japanese stock bars."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class IndicatorSnapshot:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema5: float
    ema20: float
    sma60: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_pct_b: float
    atr14: float
    vwap: float
    volume_ratio: float
    roc4: float
    roc12: float
    prev_close: float
    change_pct: float
    ema5_prev: float
    ema20_prev: float


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on an OHLCV DataFrame indexed by timestamp.

    Expected columns: Open, High, Low, Close, Volume.
    Returns a new DataFrame with indicator columns attached.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float).fillna(0.0)

    # Moving averages
    out["EMA5"] = close.ewm(span=5, adjust=False).mean()
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()
    out["SMA60"] = close.rolling(window=60, min_periods=1).mean()

    # RSI(14) using Wilder-style EMA smoothing
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 and avg_gain > 0 -> RSI = 100; when both == 0 -> RSI = 50
    rsi = rsi.where(avg_loss > 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    out["RSI14"] = rsi.fillna(50.0)

    # MACD (12, 26, 9)
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # Bollinger Bands (20, 2.0)
    bb_mid = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    out["BB_Mid"] = bb_mid
    out["BB_Upper"] = bb_mid + 2.0 * bb_std
    out["BB_Lower"] = bb_mid - 2.0 * bb_std
    bb_width = (out["BB_Upper"] - out["BB_Lower"]).replace(0.0, np.nan)
    out["BB_PctB"] = ((close - out["BB_Lower"]) / bb_width).fillna(0.5)

    # ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1.0 / 14.0, min_periods=1, adjust=False).mean().fillna(0.0)

    # Session-reset VWAP (grouped by trading date)
    typical_price = (high + low + close) / 3.0
    if isinstance(out.index, pd.DatetimeIndex):
        dates = out.index.date
        tp_vol = typical_price * volume
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
        out["VWAP"] = (cum_tp_vol / cum_vol.replace(0.0, np.nan)).fillna(close)
    else:
        cum_vol = volume.cumsum()
        out["VWAP"] = ((typical_price * volume).cumsum() / cum_vol.replace(0.0, np.nan)).fillna(close)

    # Volume Ratio vs 20-bar rolling mean
    vol_ma20 = volume.rolling(window=20, min_periods=1).mean()
    out["Volume_Ratio"] = (volume / vol_ma20.replace(0.0, np.nan)).fillna(1.0)

    # Rate of Change (momentum)
    out["ROC4"] = close.pct_change(periods=4).fillna(0.0)
    out["ROC12"] = close.pct_change(periods=12).fillna(0.0)

    return out


def extract_latest_snapshot(ind_df: pd.DataFrame) -> IndicatorSnapshot | None:
    """Extract the latest bar's IndicatorSnapshot from an indicator DataFrame."""
    if ind_df is None or ind_df.empty:
        return None

    last = ind_df.iloc[-1]
    prev = ind_df.iloc[-2] if len(ind_df) >= 2 else last
    ts = ind_df.index[-1]
    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

    close_val = float(last["Close"])
    prev_close_val = float(prev["Close"])
    change_pct = ((close_val - prev_close_val) / prev_close_val) if prev_close_val else 0.0

    return IndicatorSnapshot(
        timestamp=ts_str,
        open=float(last["Open"]),
        high=float(last["High"]),
        low=float(last["Low"]),
        close=close_val,
        volume=float(last["Volume"]),
        ema5=float(last["EMA5"]),
        ema20=float(last["EMA20"]),
        sma60=float(last["SMA60"]),
        rsi14=float(last["RSI14"]),
        macd=float(last["MACD"]),
        macd_signal=float(last["MACD_Signal"]),
        macd_hist=float(last["MACD_Hist"]),
        macd_hist_prev=float(prev["MACD_Hist"]),
        bb_upper=float(last["BB_Upper"]),
        bb_mid=float(last["BB_Mid"]),
        bb_lower=float(last["BB_Lower"]),
        bb_pct_b=float(last["BB_PctB"]),
        atr14=float(last["ATR14"]),
        vwap=float(last["VWAP"]),
        volume_ratio=float(last["Volume_Ratio"]),
        roc4=float(last["ROC4"]),
        roc12=float(last["ROC12"]),
        prev_close=prev_close_val,
        change_pct=change_pct,
        ema5_prev=float(prev["EMA5"]),
        ema20_prev=float(prev["EMA20"]),
    )
