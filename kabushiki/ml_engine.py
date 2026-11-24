"""Strict zero-lookahead Walk-Forward Ensemble ML engine (L2 Ridge + HistGradientBoosting + FRED Macro)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None

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
    # External FRED / ECB macro features (strictly lagged to prior trading day -> zero lookahead)
    "usdjpy_chg5",
    "vix_level",
    "vix_chg5",
    "us10y_level",
    "us10y_chg5",
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
    "usdjpy_chg5": "ドル円5日変化(FRED)",
    "vix_level": "米国VIX恐怖指数(FRED)",
    "vix_chg5": "VIX5日変化(FRED)",
    "us10y_level": "米10年債利回り(FRED)",
    "us10y_chg5": "米10年債5日変化(FRED)",
}

MACRO_DEFAULTS = {
    "usdjpy_chg5": 0.0,
    "vix_level": 20.0,
    "vix_chg5": 0.0,
    "us10y_level": 4.0,
    "us10y_chg5": 0.0,
}


def attach_ml_features(
    ind_df: pd.DataFrame,
    bm_ind_df: pd.DataFrame | None = None,
    macro_daily_df: pd.DataFrame | None = None,
    horizon_bars: int = 8,
) -> pd.DataFrame:
    """Attach strictly causal 22 ML features at bar t and forward training target Open[t+1] -> Close[t+H].

    External macro daily features (FRED USD/JPY, VIX, US10Y) are strictly shifted so an intraday
    bar on date D only ever sees macro values published strictly BEFORE date D (zero lookahead).
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

    # Strictly causal prior-day join for FRED/ECB macro features
    if macro_daily_df is not None and not macro_daily_df.empty and isinstance(df.index, pd.DatetimeIndex):
        # Bar date minus 1 day guarantees strictly prior completed macro day
        bar_prev_dates = pd.to_datetime(df.index.tz_localize(None).normalize()) - pd.Timedelta(days=1)
        macro_sorted = macro_daily_df.sort_index()
        pos = macro_sorted.index.searchsorted(bar_prev_dates, side="right") - 1
        for col, default_val in MACRO_DEFAULTS.items():
            if col in macro_sorted.columns:
                col_vals = macro_sorted[col].values
                aligned = np.where(pos >= 0, col_vals[np.clip(pos, 0, len(col_vals) - 1)], default_val)
                df[col] = pd.Series(aligned, index=df.index).fillna(default_val)
            else:
                df[col] = default_val
    else:
        for col, default_val in MACRO_DEFAULTS.items():
            df[col] = default_val

    next_open = open_p.shift(-1)
    df["fwd_ret"] = (close.shift(-horizon_bars) / next_open.replace(0.0, np.nan)) - 1.0
    return df


@dataclass
class MLModelDiagnostics:
    model_type: str = "Walk-Forward アンサンブル (L2 Ridge 55% + HistGBDT 45%)"
    interval: str = "30m"
    horizon_bars: int = 8
    feature_count: int = len(FEATURE_COLS)
    data_sources: str = "Yahoo日中足 + FREDマクロ(ドル円/VIX/米10年債)"
    train_samples: int = 0
    oos_samples: int = 0
    oos_ic: float = 0.0
    ridge_ic: float = 0.0
    gbdt_ic: float = 0.0
    oos_hit_rate_pct: float = 0.0
    oos_rmse: float = 0.0
    top_features: list[dict[str, Any]] = field(default_factory=list)
    learning_curve: list[dict[str, Any]] = field(default_factory=list)
    trained_up_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WalkForwardEnsembleModel:
    """Strictly causal Walk-Forward Ensemble: L2 Ridge Regression + HistGradientBoostingRegressor."""

    def __init__(
        self,
        l2_alpha: float = 600.0,
        horizon_bars: int = 8,
        min_samples: int = 120,
        use_gbdt: bool = True,
        ridge_weight: float = 0.55,
    ):
        self.l2_alpha = float(l2_alpha)
        self.horizon_bars = int(horizon_bars)
        self.min_samples = int(min_samples)
        self.use_gbdt = bool(use_gbdt and HistGradientBoostingRegressor is not None)
        self.ridge_weight = float(ridge_weight)
        self.weights: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.gbdt: Any = None
        self.train_samples: int = 0
        self.trained_up_to: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.mu is not None and self.sd is not None

    @property
    def model_label(self) -> str:
        if self.gbdt is not None:
            return "Walk-Forward アンサンブル (L2 Ridge 55% + HistGBDT 45%)"
        return "Walk-Forward L2 Ridge回帰"

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

        y = np.clip(y, -0.12, 0.12)

        # 1. L2-regularized Ridge Regression (stable linear component)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xn = np.clip((X - self.mu) / self.sd, -5.0, 5.0)
        Xn_b = np.hstack([Xn, np.ones((len(Xn), 1))])

        eff_alpha = self.l2_alpha * max(1.0, len(X) / 4000.0)
        reg = np.eye(Xn_b.shape[1]) * eff_alpha
        reg[-1, -1] = 0.0
        self.weights = np.linalg.solve(Xn_b.T @ Xn_b + reg, Xn_b.T @ y)

        # 2. Shallow L2-regularized HistGradientBoostingRegressor (non-linear interaction component)
        self.gbdt = None
        if self.use_gbdt and len(X) >= max(250, self.min_samples):
            try:
                gbdt = HistGradientBoostingRegressor(
                    max_iter=55,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_leaf=max(25, min(80, len(X) // 40)),
                    l2_regularization=25.0,
                    max_bins=64,
                    early_stopping=False,
                    random_state=42,
                )
                gbdt.fit(Xn, y)
                self.gbdt = gbdt
            except Exception:
                self.gbdt = None

        self.train_samples = int(len(X))
        self.trained_up_to = cutoff_ts.isoformat() if hasattr(cutoff_ts, "isoformat") else str(cutoff_ts)
        return True

    def predict_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (ensemble_pred, ridge_pred, gbdt_pred_or_None)."""
        Xn = np.clip((X - self.mu) / self.sd, -5.0, 5.0)
        Xn_b = np.hstack([Xn, np.ones((len(Xn), 1))])
        p_ridge = Xn_b @ self.weights
        if self.gbdt is not None:
            p_gbdt = self.gbdt.predict(Xn)
            p_ens = self.ridge_weight * p_ridge + (1.0 - self.ridge_weight) * p_gbdt
            return p_ens, p_ridge, p_gbdt
        return p_ridge, p_ridge, None

    def predict_row(self, row: pd.Series) -> float | None:
        if not self.is_fitted:
            return None
        vals = row[FEATURE_COLS].values.astype(float)
        if not np.isfinite(vals).all():
            return None
        p_ens, _, _ = self.predict_matrix(vals.reshape(1, -1))
        return float(p_ens[0])

    def top_feature_weights(self, k: int = 6) -> list[dict[str, Any]]:
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


# Backward-compatible alias
WalkForwardRidgeModel = WalkForwardEnsembleModel


def build_training_panel(
    feat_map: dict[str, pd.DataFrame],
    warmup_bars: int = 40,
    extra_prior_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine per-symbol feature DataFrames (+ optional long-history 60m panel) into a time-sorted panel."""
    frames = []
    if extra_prior_panel is not None and not extra_prior_panel.empty:
        frames.append(extra_prior_panel[FEATURE_COLS + ["fwd_ret"]].copy())
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
    data_sources: str = "Yahoo日中足 + FREDマクロ(ドル円/VIX/米10年債)",
) -> MLModelDiagnostics:
    """Run strictly causal walk-forward evaluation and produce a step-by-step learning_curve."""
    if panel_df is None or panel_df.empty:
        return MLModelDiagnostics(interval=interval, horizon_bars=horizon_bars, data_sources=data_sources)

    clean = panel_df.dropna(subset=FEATURE_COLS + ["fwd_ret"])
    unique_ts = sorted(clean.index.unique())
    if len(unique_ts) < 60:
        return MLModelDiagnostics(
            interval=interval,
            horizon_bars=horizon_bars,
            data_sources=data_sources,
            train_samples=len(clean),
        )

    model = WalkForwardEnsembleModel(l2_alpha=l2_alpha, horizon_bars=horizon_bars, min_samples=120)
    start_i = max(40, len(unique_ts) // 3)
    num_folds = min(10, max(4, (len(unique_ts) - start_i) // 12))
    retrain_step = max(8, (len(unique_ts) - start_i) // num_folds)

    preds_ens: list[float] = []
    preds_ridge: list[float] = []
    preds_gbdt: list[float] = []
    actuals: list[float] = []

    learning_curve: list[dict[str, Any]] = []
    fold_preds: list[float] = []
    fold_actuals: list[float] = []
    fold_step = 0

    for i in range(start_i, len(unique_ts)):
        t_now = unique_ts[i]
        cutoff_idx = i - horizon_bars - 1
        if cutoff_idx < 20:
            continue

        is_retrain = (not model.is_fitted) or ((i - start_i) % retrain_step == 0)
        if is_retrain:
            if fold_preds:
                fa = np.array(fold_actuals, dtype=float)
                fp = np.array(fold_preds, dtype=float)
                f_ic = float(np.corrcoef(fp, fa)[0, 1]) if (fp.std() > 1e-9 and fa.std() > 1e-9) else 0.0
                hi = fa[fp >= min_pred_return]
                f_hit = float((hi > 0.0).mean() * 100.0) if len(hi) >= 3 else float(((fp > 0) == (fa > 0)).mean() * 100.0)
                f_rmse = float(np.sqrt(np.mean((fp - fa) ** 2)) * 100.0)
                fold_step += 1
                learning_curve.append(
                    {
                        "fold": fold_step,
                        "timestamp": t_now.isoformat() if hasattr(t_now, "isoformat") else str(t_now),
                        "train_samples": model.train_samples,
                        "oos_ic": round(f_ic, 4),
                        "hit_rate_pct": round(f_hit, 1),
                        "rmse_pct": round(f_rmse, 3),
                    }
                )
                fold_preds = []
                fold_actuals = []
            model.fit_from_panel(clean, unique_ts[cutoff_idx])

        if not model.is_fitted:
            continue

        cur = clean.loc[clean.index == t_now]
        X_cur = cur[FEATURE_COLS].values.astype(float)
        y_cur = cur["fwd_ret"].values.astype(float)
        p_e, p_r, p_g = model.predict_matrix(X_cur)
        preds_ens.extend(p_e.tolist())
        preds_ridge.extend(p_r.tolist())
        if p_g is not None:
            preds_gbdt.extend(p_g.tolist())
        actuals.extend(y_cur.tolist())
        fold_preds.extend(p_e.tolist())
        fold_actuals.extend(y_cur.tolist())

    # Record final fold
    if fold_preds and model.is_fitted:
        fa = np.array(fold_actuals, dtype=float)
        fp = np.array(fold_preds, dtype=float)
        f_ic = float(np.corrcoef(fp, fa)[0, 1]) if (fp.std() > 1e-9 and fa.std() > 1e-9) else 0.0
        hi = fa[fp >= min_pred_return]
        f_hit = float((hi > 0.0).mean() * 100.0) if len(hi) >= 3 else float(((fp > 0) == (fa > 0)).mean() * 100.0)
        f_rmse = float(np.sqrt(np.mean((fp - fa) ** 2)) * 100.0)
        fold_step += 1
        last_t = unique_ts[-1]
        learning_curve.append(
            {
                "fold": fold_step,
                "timestamp": last_t.isoformat() if hasattr(last_t, "isoformat") else str(last_t),
                "train_samples": model.train_samples,
                "oos_ic": round(f_ic, 4),
                "hit_rate_pct": round(f_hit, 1),
                "rmse_pct": round(f_rmse, 3),
            }
        )

    # Fit latest production weights on all matured historical samples
    latest_cutoff_idx = max(0, len(unique_ts) - horizon_bars - 1)
    model.fit_from_panel(clean, unique_ts[latest_cutoff_idx])

    oos_ic = ridge_ic = gbdt_ic = hit_rate = oos_rmse = 0.0
    if len(preds_ens) >= 20:
        pe = np.array(preds_ens, dtype=float)
        pr = np.array(preds_ridge, dtype=float)
        aa = np.array(actuals, dtype=float)
        if pe.std() > 1e-9 and aa.std() > 1e-9:
            oos_ic = float(np.corrcoef(pe, aa)[0, 1])
        if pr.std() > 1e-9 and aa.std() > 1e-9:
            ridge_ic = float(np.corrcoef(pr, aa)[0, 1])
        if len(preds_gbdt) == len(actuals):
            pg = np.array(preds_gbdt, dtype=float)
            if pg.std() > 1e-9 and aa.std() > 1e-9:
                gbdt_ic = float(np.corrcoef(pg, aa)[0, 1])
        oos_rmse = float(np.sqrt(np.mean((pe - aa) ** 2)) * 100.0)
        hi = aa[pe >= min_pred_return]
        if len(hi) >= 3:
            hit_rate = float((hi > 0.0).mean() * 100.0)
        else:
            hit_rate = float(((pe > 0) == (aa > 0)).mean() * 100.0)

    return MLModelDiagnostics(
        model_type=model.model_label,
        interval=interval,
        horizon_bars=horizon_bars,
        feature_count=len(FEATURE_COLS),
        data_sources=data_sources,
        train_samples=model.train_samples,
        oos_samples=len(preds_ens),
        oos_ic=round(oos_ic, 4),
        ridge_ic=round(ridge_ic, 4),
        gbdt_ic=round(gbdt_ic, 4),
        oos_hit_rate_pct=round(hit_rate, 1),
        oos_rmse=round(oos_rmse, 3),
        top_features=model.top_feature_weights(6),
        learning_curve=learning_curve,
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
        f"MLアンサンブル予測 {pred_pct:+.2f}% (コスト差引 {net_edge_pct:+.2f}%)",
    ]
    dist_ema20_pct = ((snap.close - snap.ema20) / snap.ema20 * 100.0) if snap.ema20 else 0.0
    if dist_ema20_pct <= -0.5:
        reasons.append(f"EMA20下方乖離・押し目反発期待 ({dist_ema20_pct:+.1f}%)")
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

    return StockRecommendation(
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
