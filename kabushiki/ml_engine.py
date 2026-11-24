"""Strict zero-lookahead Walk-Forward Machine Learning engine for Japanese equities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd

from .advisor import StockRecommendation, _round_jpy
from .config import AppConfig, StockInfo
from .indicators import IndicatorSnapshot

FEATURE_COLS = [
    "RSI14",
    "BB_PctB",
    "Volume_Ratio",
    "ROC4",
    "ROC12",
    "dist_vwap",
    "dist_ema20",
    "dist_sma60",
    "ema_spread",
    "atr_pct",
    "candle_body",
    "rel_str_4",
    "rel_str_12",
    "bm_above_ema20",
    "bm_roc4",
    "bm_roc12",
    "hour",
]

FEATURE_LABELS_JA = {
    "RSI14": "RSI(14)",
    "BB_PctB": "ボリンジャー%B",
    "Volume_Ratio": "出来高倍率",
    "ROC4": "短期騰落率(4本)",
    "ROC12": "中期騰落率(12本)",
    "dist_vwap": "VWAP乖離率",
    "dist_ema20": "EMA20乖離率",
    "dist_sma60": "SMA60乖離率",
    "ema_spread": "EMA5-EMA20幅",
    "atr_pct": "ボラティリティ(ATR%)",
    "candle_body": "ローソク足実体",
    "rel_str_4": "対日経相対強度(4本)",
    "rel_str_12": "対日経相対強度(12本)",
    "bm_above_ema20": "日経トレンド位置",
    "bm_roc4": "日経短期騰落(4本)",
    "bm_roc12": "日経中期騰落(12本)",
    "hour": "取引時間帯",
}


def attach_ml_features(
    ind_df: pd.DataFrame,
    bm_ind_df: pd.DataFrame | None = None,
    horizon_bars: int = 8,
) -> pd.DataFrame:
    """Attach strictly causal ML features at bar t and forward training target Open[t+1] -> Close[t+H].

    Note: The forward target column `fwd_ret` is ONLY used for historical bars whose
    `t_train + horizon_bars` is strictly earlier than the prediction timestamp `t_now`.
    """
    if ind_df is None or ind_df.empty:
        return pd.DataFrame()

    df = ind_df.copy()
    if bm_ind_df is not None and not bm_ind_df.empty:
        bm_aligned = bm_ind_df.reindex(df.index, method="ffill")
        df["bm_above_ema20"] = (bm_aligned["Close"] > bm_aligned["EMA20"]).astype(float).fillna(0.5)
        df["bm_roc4"] = bm_aligned["ROC4"].fillna(0.0)
        df["bm_roc12"] = bm_aligned["ROC12"].fillna(0.0)
    else:
        df["bm_above_ema20"] = 0.5
        df["bm_roc4"] = 0.0
        df["bm_roc12"] = 0.0

    close = df["Close"].astype(float)
    open_p = df["Open"].astype(float)
    df["rel_str_4"] = df["ROC4"] - df["bm_roc4"]
    df["rel_str_12"] = df["ROC12"] - df["bm_roc12"]
    df["dist_vwap"] = (close - df["VWAP"]) / df["VWAP"].replace(0.0, np.nan)
    df["dist_ema20"] = (close - df["EMA20"]) / df["EMA20"].replace(0.0, np.nan)
    df["dist_sma60"] = (close - df["SMA60"]) / df["SMA60"].replace(0.0, np.nan)
    df["ema_spread"] = (df["EMA5"] - df["EMA20"]) / df["EMA20"].replace(0.0, np.nan)
    df["atr_pct"] = df["ATR14"] / close.replace(0.0, np.nan)
    df["candle_body"] = (close - open_p) / open_p.replace(0.0, np.nan)
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour + df.index.minute / 60.0
    else:
        df["hour"] = 12.0

    # Forward label strictly for past training rows: buy at t+1 Open, value at t+H Close
    next_open = open_p.shift(-1)
    df["fwd_ret"] = (close.shift(-horizon_bars) / next_open.replace(0.0, np.nan)) - 1.0
    return df


@dataclass
class MLModelDiagnostics:
    model_type: str = "Walk-Forward Ridge (L2正則化)"
    interval: str = "30m"
    horizon_bars: int = 8
    train_samples: int = 0
    oos_samples: int = 0
    oos_ic: float = 0.0
    oos_hit_rate_pct: float = 0.0
    top_features: list[dict[str, Any]] | None = None
    trained_up_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WalkForwardRidgeModel:
    """Fast, strictly causal L2-regularized Walk-Forward Ridge Regression model (pure NumPy)."""

    def __init__(self, l2_alpha: float = 600.0, horizon_bars: int = 8, min_samples: int = 120):
        self.l2_alpha = float(l2_alpha)
        self.horizon_bars = int(horizon_bars)
        self.min_samples = int(min_samples)
        self.weights: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.train_samples: int = 0
        self.trained_up_to: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.mu is not None and self.sd is not None

    def fit_from_panel(
        self,
        panel_df: pd.DataFrame,
        cutoff_ts: pd.Timestamp,
    ) -> bool:
        """Fit strictly on rows with index <= cutoff_ts (guaranteeing zero future label leakage)."""
        if panel_df is None or panel_df.empty:
            return False
        tr = panel_df.loc[:cutoff_ts]
        if len(tr) < self.min_samples:
            return False

        X = tr[FEATURE_COLS].values.astype(float)
        y = tr["fwd_ret"].values.astype(float)
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        if len(X) < self.min_samples:
            return False

        # Winsorize extreme outlier returns (+/- 12%) for robust linear weights
        y = np.clip(y, -0.12, 0.12)

        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xn = np.clip((X - self.mu) / self.sd, -5.0, 5.0)
        Xn_b = np.hstack([Xn, np.ones((len(Xn), 1))])

        # Scale L2 regularization mildly with sample size for stability as data grows
        eff_alpha = self.l2_alpha * max(1.0, len(X) / 4000.0)
        reg = np.eye(Xn_b.shape[1]) * eff_alpha
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(Xn_b.T @ Xn_b + reg, Xn_b.T @ y)
        self.train_samples = int(len(X))
        self.trained_up_to = cutoff_ts.isoformat() if hasattr(cutoff_ts, "isoformat") else str(cutoff_ts)
        return True

    def predict_row(self, row: pd.Series) -> float | None:
        if not self.is_fitted:
            return None
        vals = row[FEATURE_COLS].values.astype(float)
        if not np.isfinite(vals).all():
            return None
        xn = np.clip((vals - self.mu) / self.sd, -5.0, 5.0)
        xn_b = np.append(xn, 1.0)
        return float(xn_b @ self.weights)

    def top_feature_weights(self, k: int = 5) -> list[dict[str, Any]]:
        if not self.is_fitted:
            return []
        coef = self.weights[:-1]
        order = np.argsort(np.abs(coef))[::-1][:k]
        out = []
        for idx in order:
            fname = FEATURE_COLS[int(idx)]
            wval = float(coef[int(idx)])
            out.append(
                {
                    "feature": fname,
                    "label": FEATURE_LABELS_JA.get(fname, fname),
                    "weight": round(wval * 100.0, 4),
                    "direction": "上昇寄与" if wval > 0 else "逆張り(押し目)寄与",
                }
            )
        return out


def build_training_panel(
    feat_map: dict[str, pd.DataFrame],
    warmup_bars: int = 40,
) -> pd.DataFrame:
    """Combine per-symbol feature DataFrames into a time-sorted panel for fast walk-forward fitting."""
    frames = []
    for sym, df in feat_map.items():
        if df is None or len(df) <= warmup_bars:
            continue
        sub = df.iloc[warmup_bars:][FEATURE_COLS + ["fwd_ret"]].copy()
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS + ["fwd_ret"])
    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    return panel.sort_index()


def evaluate_walkforward_oos(
    panel_df: pd.DataFrame,
    interval: str,
    horizon_bars: int = 8,
    min_pred_return: float = 0.008,
    l2_alpha: float = 600.0,
) -> MLModelDiagnostics:
    """Run a quick strictly causal walk-forward evaluation over the accumulated panel to report honest OOS accuracy."""
    if panel_df is None or panel_df.empty:
        return MLModelDiagnostics(interval=interval, horizon_bars=horizon_bars)

    clean = panel_df.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    unique_ts = sorted(clean.index.unique())
    if len(unique_ts) < 60:
        return MLModelDiagnostics(
            interval=interval,
            horizon_bars=horizon_bars,
            train_samples=len(clean),
        )

    model = WalkForwardRidgeModel(l2_alpha=l2_alpha, horizon_bars=horizon_bars, min_samples=120)
    start_i = max(40, len(unique_ts) // 3)
    retrain_step = max(10, (len(unique_ts) - start_i) // 12)

    preds: list[float] = []
    actuals: list[float] = []

    for i in range(start_i, len(unique_ts)):
        t_now = unique_ts[i]
        cutoff_idx = i - horizon_bars - 1
        if cutoff_idx < 20:
            continue
        if (not model.is_fitted) or ((i - start_i) % retrain_step == 0):
            model.fit_from_panel(clean, unique_ts[cutoff_idx])
        if not model.is_fitted:
            continue

        cur = clean.loc[clean.index == t_now]
        X_cur = cur[FEATURE_COLS].values.astype(float)
        y_cur = cur["fwd_ret"].values.astype(float)
        Xn = np.clip((X_cur - model.mu) / model.sd, -5.0, 5.0)
        Xn_b = np.hstack([Xn, np.ones((len(Xn), 1))])
        p_cur = Xn_b @ model.weights
        preds.extend(p_cur.tolist())
        actuals.extend(y_cur.tolist())

    # Fit latest production weights strictly on all bars whose labels have already matured
    latest_cutoff_idx = max(0, len(unique_ts) - horizon_bars - 1)
    model.fit_from_panel(clean, unique_ts[latest_cutoff_idx])

    oos_ic = 0.0
    hit_rate = 0.0
    if len(preds) >= 20:
        p_arr = np.array(preds, dtype=float)
        a_arr = np.array(actuals, dtype=float)
        if p_arr.std() > 1e-9 and a_arr.std() > 1e-9:
            oos_ic = float(np.corrcoef(p_arr, a_arr)[0, 1])
        hi = a_arr[p_arr >= min_pred_return]
        if len(hi) >= 3:
            hit_rate = float((hi > 0.0).mean() * 100.0)
        else:
            hit_rate = float(((p_arr > 0) == (a_arr > 0)).mean() * 100.0)

    return MLModelDiagnostics(
        model_type="Walk-Forward Ridge (L2正則化)",
        interval=interval,
        horizon_bars=horizon_bars,
        train_samples=model.train_samples,
        oos_samples=len(preds),
        oos_ic=round(oos_ic, 4),
        oos_hit_rate_pct=round(hit_rate, 1),
        top_features=model.top_feature_weights(5),
        trained_up_to=model.trained_up_to,
    )


def ml_prediction_to_recommendation(
    stock: StockInfo,
    snap: IndicatorSnapshot,
    pred_ret: float,
    cfg: AppConfig,
    min_pred_return: float = 0.008,
) -> StockRecommendation:
    """Convert a strictly causal ML expected-return forecast into a StockRecommendation."""
    round_trip_cost = 2.0 * (cfg.fee_rate + cfg.slippage_bps / 10000.0)
    pred_pct = pred_ret * 100.0
    net_edge_pct = (pred_ret - round_trip_cost) * 100.0

    # Map expected return into intuitive 0-100 score (min_pred_return -> ~72, 0% -> 50, -0.8% -> 28)
    scale = max(min_pred_return, 0.004)
    score = 50.0 + (pred_ret / scale) * 22.0
    score = max(0.0, min(100.0, round(score, 1)))

    if pred_ret >= min_pred_return * 1.35:
        action = "STRONG_BUY"
        action_label = "ML強い買い"
    elif pred_ret >= min_pred_return:
        action = "BUY"
        action_label = "ML買い推奨"
    elif pred_ret <= -min_pred_return:
        action = "SELL"
        action_label = "ML売り警戒"
    else:
        action = "HOLD"
        action_label = "見送り(コスト圏)"

    reasons = [
        f"ML予測期待リターン {pred_pct:+.2f}% (コスト差引 {net_edge_pct:+.2f}%)",
    ]
    dist_ema20_pct = ((snap.close - snap.ema20) / snap.ema20 * 100.0) if snap.ema20 else 0.0
    if dist_ema20_pct <= -0.5:
        reasons.append(f"EMA20下方乖離からの押し目反発期待 ({dist_ema20_pct:+.1f}%)")
    elif dist_ema20_pct >= 1.2:
        reasons.append(f"EMA20上方乖離・反落注意 ({dist_ema20_pct:+.1f}%)")
    if snap.rsi14 <= 38.0:
        reasons.append(f"RSI売られ過ぎ圏 (RSI={snap.rsi14:.1f})")
    elif snap.rsi14 >= 70.0:
        reasons.append(f"RSI過熱圏 (RSI={snap.rsi14:.1f})")
    if snap.volume_ratio >= 1.4:
        reasons.append(f"出来高活発 ({snap.volume_ratio:.1f}倍)")

    entry = snap.close
    target_price = _round_jpy(entry * (1.0 + max(cfg.take_profit_pct, 0.025)))
    stop_loss_price = _round_jpy(max(1.0, entry * (1.0 - max(cfg.stop_loss_pct, 0.015))))
    risk = max(entry - stop_loss_price, 0.01)
    reward = max(target_price - entry, 0.0)

    rec = StockRecommendation(
        symbol=stock.symbol,
        name=stock.name,
        sector=stock.sector,
        timestamp=snap.timestamp,
        price=_round_jpy(entry),
        change_pct=round(snap.change_pct * 100.0, 2),
        score=score,
        action=action,
        action_label=action_label,
        reasons=reasons[:4],
        suggested_entry=_round_jpy(entry),
        target_price=target_price,
        stop_loss_price=stop_loss_price,
        risk_reward=round(reward / risk, 2),
        rsi14=round(snap.rsi14, 1),
        macd_hist=round(snap.macd_hist, 4),
        vwap=_round_jpy(snap.vwap),
        volume_ratio=round(snap.volume_ratio, 2),
        bb_pct_b=round(snap.bb_pct_b, 2),
        atr14=round(snap.atr14, 2),
    )
    return rec
