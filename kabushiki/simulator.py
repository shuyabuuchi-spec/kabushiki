"""Strict zero-lookahead paper-trading simulator with Walk-Forward ML, persistent data collection, fees, and ~20% tax."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any
import math
import pandas as pd

from .advisor import (
    score_stock,
    score_stock_mean_reversion,
    evaluate_sell_timing,
    StockRecommendation,
    SellEvaluation,
    _round_jpy,
)
from .config import AppConfig, SimulationPattern, MODE_UNIT_SIZE, MODE_LABELS, StockInfo
from .external_data import fetch_all_external_macro, build_daily_macro_frame
from .indicators import compute_indicators, extract_latest_snapshot
from .market_data import JST, MarketDataProvider, now_jst
from .ml_engine import (
    WalkForwardEnsembleModel,
    WalkForwardRidgeModel,
    attach_ml_features,
    build_training_panel,
    evaluate_walkforward_oos,
    ml_prediction_to_recommendation,
)
from .storage import Storage

INTERVAL_MINUTES = {
    "5m": 5,
    "10m": 10,
    "15m": 15,
    "30m": 30,
    "60m": 60,
}


def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    factor = slippage_bps / 10000.0
    if side == "BUY":
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def _compute_order_quantity(
    price: float,
    cash: float,
    total_equity: float,
    unit_size: int,
    max_allocation_pct: float,
    fee_rate: float,
    slippage_bps: float,
) -> int:
    """Compute tradable share quantity respecting unit_size (1 or 100), fees, and cash."""
    exec_price = _apply_slippage(price, "BUY", slippage_bps)
    if exec_price <= 0:
        return 0
    unit_cost = exec_price * unit_size * (1.0 + fee_rate)
    if unit_cost > cash:
        return 0

    budget = min(cash, total_equity * max_allocation_pct)
    if budget < unit_cost:
        if unit_size >= 100 and cash >= unit_cost:
            return unit_size
        return 0

    lots = int(math.floor(budget / unit_cost))
    return max(0, lots * unit_size)


def _apply_realized_tax(
    net_realized_pnl: float,
    cumulative_realized_pnl: float,
    total_tax_paid: float,
    tax_rate: float,
) -> tuple[float, float, float]:
    """Japanese Specific Account (特定口座・源泉徴収あり) tax withholding & loss-offset refund."""
    new_cum = cumulative_realized_pnl + net_realized_pnl
    required_tax = max(0.0, new_cum * tax_rate)
    tax_delta = required_tax - total_tax_paid
    return new_cum, required_tax, tax_delta


def _compute_after_tax_equity(
    cash: float,
    positions: dict[str, dict[str, Any]],
    cumulative_realized_pnl: float,
    total_tax_paid: float,
    fee_rate: float,
    tax_rate: float,
) -> tuple[float, float, float, float]:
    """Return (holdings_value, unrealized_pnl_net_of_fees, est_unrealized_tax, after_tax_equity)."""
    holdings_value = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
    unrealized_pnl = sum(
        (float(p["last_price"]) - float(p["avg_cost"])) * int(p["quantity"])
        - float(p.get("buy_fee", 0.0))
        - float(p["last_price"]) * int(p["quantity"]) * fee_rate
        for p in positions.values()
    )
    est_exit_fees = holdings_value * fee_rate
    combined_taxable = cumulative_realized_pnl + unrealized_pnl
    est_total_tax = max(0.0, combined_taxable * tax_rate)
    est_unrealized_tax = max(0.0, est_total_tax - total_tax_paid)
    after_tax_equity = cash + holdings_value - est_exit_fees - est_unrealized_tax
    return holdings_value, unrealized_pnl, est_unrealized_tax, after_tax_equity


def _check_preplaced_exit_during_bar(
    pos: dict[str, Any],
    bar_open: float,
    bar_high: float,
    bar_low: float,
    strategy_cfg: AppConfig,
) -> tuple[float, str, bool] | None:
    """Check pre-placed brokerage stop-loss / trailing / take-profit strictly without intrabar lookahead.

    Returns (trigger_price, reason, is_stop_loss) or None.
    """
    entry_price = float(pos["avg_cost"])
    prior_high = float(pos["highest_price"])
    stop_loss = float(pos["stop_loss_price"])
    target_price = float(pos["target_price"])

    peak_gain_pct = ((prior_high - entry_price) / entry_price) if entry_price > 0 else 0.0
    trailing_stop: float | None = None
    if peak_gain_pct >= strategy_cfg.trailing_activation_pct:
        trailing_stop = _round_jpy(prior_high * (1.0 - strategy_cfg.trailing_stop_pct))

    if bar_low <= stop_loss:
        trigger_px = min(bar_open, stop_loss)
        return trigger_px, f"逆指値損切り約定 (設定¥{stop_loss:,.1f} / 安値¥{bar_low:,.1f})", True

    if trailing_stop is not None and bar_low <= trailing_stop:
        trigger_px = min(bar_open, trailing_stop)
        return trigger_px, f"トレーリング逆指値約定 (設定¥{trailing_stop:,.1f} / 高値¥{prior_high:,.1f})", False

    if bar_high >= target_price:
        trigger_px = max(bar_open, target_price)
        return trigger_px, f"指値利確約定 (目標¥{target_price:,.1f} / 高値¥{bar_high:,.1f})", False

    return None


def _merge_ohlcv_frames(older: pd.DataFrame | None, newer: pd.DataFrame | None) -> pd.DataFrame:
    if older is None or older.empty:
        return newer.copy() if (newer is not None and not newer.empty) else pd.DataFrame()
    if newer is None or newer.empty:
        return older.copy()
    merged = pd.concat([older, newer])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


class TradingSimulator:
    """Multi-pattern zero-lookahead simulator with Walk-Forward ML & persistent ML data collection."""

    def __init__(
        self,
        cfg: AppConfig,
        storage: Storage,
        provider: MarketDataProvider | None = None,
    ):
        self.cfg = cfg
        self.storage = storage
        self.provider = provider or MarketDataProvider()

    def _fetch_and_accumulate_bars(
        self,
        symbols: list[str],
        interval: str,
        period_days: int,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        include_benchmark: bool = True,
        persist: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Fetch live Yahoo Finance bars, persist them into SQLite `ml_bars`, and merge with accumulated history."""
        req_symbols = list(dict.fromkeys(symbols))
        fetch_syms = list(req_symbols)
        if include_benchmark and "^N225" not in fetch_syms:
            fetch_syms.append("^N225")

        if preloaded_data is not None:
            fresh_map = {s: df for s, df in preloaded_data.items() if df is not None and not df.empty}
        else:
            fresh_map = self.provider.fetch_many(fetch_syms, interval=interval, period_days=period_days)
            if persist and fresh_map:
                try:
                    self.storage.save_ml_bars(interval, fresh_map)
                except Exception:
                    pass

        stored_map = self.storage.load_ml_bars(interval, fetch_syms) if (persist and preloaded_data is None) else {}

        merged_map: dict[str, pd.DataFrame] = {}
        for sym in req_symbols:
            combined = _merge_ohlcv_frames(stored_map.get(sym), fresh_map.get(sym))
            if not combined.empty:
                merged_map[sym] = combined

        bm_df = _merge_ohlcv_frames(stored_map.get("^N225"), fresh_map.get("^N225"))
        return merged_map, bm_df

    def _ensure_macro_daily_frame(self, force_refresh: bool = False, allow_network: bool = True) -> pd.DataFrame:
        """Load persisted FRED/ECB macro series from SQLite, refreshing from network when empty or requested."""
        stored = self.storage.load_ml_macro()
        is_real_provider = isinstance(self.provider, MarketDataProvider)
        should_fetch = allow_network and is_real_provider and (force_refresh or not stored)
        if should_fetch:
            try:
                fresh = fetch_all_external_macro(lookback_years=5)
                if fresh:
                    self.storage.save_ml_macro(fresh)
                    stored = self.storage.load_ml_macro()
            except Exception:
                pass
        return build_daily_macro_frame(stored)

    def _load_long_history_60m_panel(
        self,
        symbols: list[str],
        macro_daily_df: pd.DataFrame,
        horizon_60m: int = 4,
    ) -> pd.DataFrame:
        """Build a 2-year 60m prior training panel from SQLite ml_bars (4-hour horizon matching 30m H=8)."""
        stored_60m = self.storage.load_ml_bars("60m", [*symbols, "^N225"])
        if not stored_60m:
            return pd.DataFrame()
        bm_60m = stored_60m.get("^N225")
        bm_ind = compute_indicators(bm_60m) if (bm_60m is not None and not bm_60m.empty) else pd.DataFrame()
        feat_60m: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = stored_60m.get(sym)
            if df is not None and len(df) > 60:
                feat_60m[sym] = attach_ml_features(
                    compute_indicators(df),
                    bm_ind,
                    macro_daily_df=macro_daily_df,
                    horizon_bars=horizon_60m,
                )
        return build_training_panel(feat_60m, warmup_bars=40)

    def _prepare_feature_frames(
        self,
        watchlist: list[StockInfo],
        interval: str,
        period_days: int,
        horizon_bars: int = 8,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        benchmark_df: pd.DataFrame | None = None,
        persist: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        symbols = [s.symbol for s in watchlist]
        raw_map, fetched_bm = self._fetch_and_accumulate_bars(
            symbols=symbols,
            interval=interval,
            period_days=period_days,
            preloaded_data=preloaded_data,
            include_benchmark=(benchmark_df is None),
            persist=persist,
        )
        bm_raw = benchmark_df if (benchmark_df is not None and not benchmark_df.empty) else fetched_bm
        bm_ind = compute_indicators(bm_raw) if (bm_raw is not None and not bm_raw.empty) else pd.DataFrame()
        macro_daily_df = self._ensure_macro_daily_frame(allow_network=(preloaded_data is None and persist))

        feat_map: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = raw_map.get(sym)
            if df is not None and not df.empty:
                ind = compute_indicators(df)
                feat_map[sym] = attach_ml_features(
                    ind,
                    bm_ind,
                    macro_daily_df=macro_daily_df,
                    horizon_bars=horizon_bars,
                )
        return feat_map, bm_raw, macro_daily_df

    def collect_ml_data(
        self,
        intervals: tuple[str, ...] = ("15m", "30m", "60m"),
        period_days: int = 55,
    ) -> dict[str, Any]:
        """Collect & persist multi-source data (Yahoo 15m/30m + Yahoo 60m 2-year + FRED/ECB 5-year macro) & train."""
        all_syms = sorted(set(self.cfg.symbol_meta_map().keys()))
        # 1. External non-Yahoo macro data (FRED: USD/JPY, VIX, US10Y, Nikkei + ECB Frankfurter fallback)
        macro_daily_df = self._ensure_macro_daily_frame(force_refresh=True, allow_network=True)

        # 2. Fetch 60m long history (up to ~2 years = 700 days, capped at 1,500 bars/symbol in DB)
        if "60m" in intervals:
            raw_60m, bm_60m = self._fetch_and_accumulate_bars(
                symbols=all_syms,
                interval="60m",
                period_days=650,
                include_benchmark=True,
                persist=True,
            )
            if raw_60m:
                self.storage.save_ml_bars("60m", {**raw_60m, **({"^N225": bm_60m} if not bm_60m.empty else {})}, max_bars_per_symbol=1500)

        prior_60m_panel = self._load_long_history_60m_panel(all_syms, macro_daily_df, horizon_60m=4)
        sources_desc = "Yahoo 15m/30m + Yahoo 60m(長期) + FRED米連銀(ドル円/VIX/米10年債)"

        collected: dict[str, Any] = {}
        for iv in intervals:
            if iv == "60m":
                raw_map = self.storage.load_ml_bars("60m", all_syms)
                bm_raw = self.storage.load_ml_bars("60m", ["^N225"]).get("^N225", pd.DataFrame())
            else:
                raw_map, bm_raw = self._fetch_and_accumulate_bars(
                    symbols=all_syms,
                    interval=iv,
                    period_days=period_days,
                    include_benchmark=True,
                    persist=True,
                )
            bm_ind = compute_indicators(bm_raw) if (bm_raw is not None and not bm_raw.empty) else pd.DataFrame()
            horizon = 12 if iv == "15m" else (4 if iv == "60m" else 8)
            feat_map = {
                sym: attach_ml_features(
                    compute_indicators(df),
                    bm_ind,
                    macro_daily_df=macro_daily_df,
                    horizon_bars=horizon,
                )
                for sym, df in raw_map.items()
                if not df.empty
            }
            use_prior = prior_60m_panel if (iv == "30m" and not prior_60m_panel.empty) else None
            panel = build_training_panel(feat_map, extra_prior_panel=use_prior)
            diag = evaluate_walkforward_oos(
                panel,
                interval=iv,
                horizon_bars=horizon,
                data_sources=sources_desc,
            )
            self.storage.save_ml_diagnostics(iv, diag.to_dict(), record_history=True)
            collected[iv] = {
                "symbols": len(raw_map),
                "diagnostics": diag.to_dict(),
            }
        self.storage.export_dashboard_snapshot(self.cfg)
        return {"collected": collected, "ml_status": self.storage.get_ml_status()}

    def _resolve_pattern(self, mode_or_pattern_id: str) -> SimulationPattern:
        p = self.cfg.get_pattern(mode_or_pattern_id)
        if p is not None:
            return p
        unit_size = MODE_UNIT_SIZE.get(mode_or_pattern_id, 1)
        return SimulationPattern(
            id=mode_or_pattern_id,
            name=MODE_LABELS.get(mode_or_pattern_id, mode_or_pattern_id),
            strategy_type="trend",
            initial_capital=self.cfg.initial_capital,
            unit_size=unit_size,
            watchlist_key="lot100" if unit_size >= 100 else "lot1",
            interval=self.cfg.interval,
            max_positions=self.cfg.max_positions,
            max_allocation_pct=self.cfg.max_allocation_pct,
            fee_rate=self.cfg.fee_rate,
            tax_rate=self.cfg.tax_rate,
            slippage_bps=self.cfg.slippage_bps,
            buy_score_threshold=self.cfg.buy_score_threshold,
            strong_buy_threshold=self.cfg.strong_buy_threshold,
            sell_score_threshold=self.cfg.sell_score_threshold,
            take_profit_pct=self.cfg.take_profit_pct,
            stop_loss_pct=self.cfg.stop_loss_pct,
            trailing_activation_pct=self.cfg.trailing_activation_pct,
            trailing_stop_pct=self.cfg.trailing_stop_pct,
            close_before_eod=self.cfg.close_before_eod,
        )

    def _score_universe_at_bar(
        self,
        pattern: SimulationPattern,
        strategy_cfg: AppConfig,
        watchlist: list[StockInfo],
        feat_map: dict[str, pd.DataFrame],
        bar_ts: pd.Timestamp,
        ml_model: WalkForwardRidgeModel | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[StockRecommendation], dict[str, StockRecommendation], dict[str, float]]:
        """Score every stock strictly using data <= bar_ts according to pattern.strategy_type."""
        exact_bars: dict[str, Any] = {}
        latest_snaps: dict[str, Any] = {}
        recommendations: list[StockRecommendation] = []
        rec_by_sym: dict[str, StockRecommendation] = {}
        ml_preds: dict[str, float] = {}

        for stock in watchlist:
            ind_df = feat_map.get(stock.symbol)
            if ind_df is None or ind_df.empty:
                continue
            pos_idx = int(ind_df.index.searchsorted(bar_ts, side="right")) - 1
            if pos_idx < 1:
                continue
            row = ind_df.iloc[pos_idx]
            if ind_df.index[pos_idx] == bar_ts:
                exact_bars[stock.symbol] = row
            snap = extract_latest_snapshot(ind_df.iloc[pos_idx - 1 : pos_idx + 1])
            if snap is None:
                continue

            if pattern.strategy_type == "ml_walkforward" and ml_model is not None and ml_model.is_fitted:
                pred = ml_model.predict_row(row)
                if pred is not None:
                    ml_preds[stock.symbol] = pred
                    rec = ml_prediction_to_recommendation(
                        stock=stock,
                        snap=snap,
                        pred_ret=pred,
                        cfg=strategy_cfg,
                        min_pred_return=pattern.ml_min_pred_return,
                    )
                else:
                    rec = score_stock_mean_reversion(stock, snap, strategy_cfg)
            elif pattern.strategy_type == "mean_reversion":
                rec = score_stock_mean_reversion(stock, snap, strategy_cfg)
            else:
                rec = score_stock(stock, snap, strategy_cfg)

            recommendations.append(rec)
            latest_snaps[stock.symbol] = snap
            rec_by_sym[stock.symbol] = rec

        recommendations.sort(key=lambda r: r.score, reverse=True)
        return exact_bars, latest_snaps, recommendations, rec_by_sym, ml_preds

    def _should_queue_sell_for_strategy(
        self,
        pattern: SimulationPattern,
        pos: dict[str, Any],
        ev: SellEvaluation,
        ml_pred: float | None,
    ) -> tuple[bool, str]:
        """Determine whether to queue a next-bar Open SELL order for a held position."""
        bars_held = int(pos.get("bars_held", 0))
        if pattern.max_hold_bars > 0 and bars_held >= pattern.max_hold_bars:
            return True, f"最大保有期間到達 ({bars_held}本経過・資金回転決済)"

        if pattern.strategy_type == "ml_walkforward":
            # Avoid panic-selling on small noisy dips; rely on pre-placed SL/TP/Trailing + strong negative ML reversal
            if ml_pred is not None and ml_pred <= -max(pattern.ml_min_pred_return, 0.008):
                return True, f"ML予測が大幅マイナス転換 ({ml_pred*100:+.2f}%)"
            if pattern.close_before_eod and "大引け前" in ev.primary_reason:
                return True, ev.primary_reason
            return False, ev.primary_reason

        if pattern.strategy_type == "mean_reversion":
            # Mean-reversion: exit on overheated bounce or EOD, NOT on temporary dip
            if ev.should_sell and ("過熱" in ev.primary_reason or "利確" in ev.primary_reason or "トレーリング" in ev.primary_reason or "大引け前" in ev.primary_reason):
                return True, ev.primary_reason
            return False, ev.primary_reason

        # Default trend strategy
        return ev.should_sell, ev.primary_reason

    def _execute_bar(
        self,
        portfolio_id: str,
        pattern: SimulationPattern,
        strategy_cfg: AppConfig,
        bar_ts: pd.Timestamp,
        watchlist: list[StockInfo],
        feat_map: dict[str, pd.DataFrame],
        ml_model: WalkForwardRidgeModel | None = None,
    ) -> dict[str, Any]:
        """Process one bar strictly chronologically with zero lookahead."""
        portfolio = self.storage.ensure_portfolio(
            portfolio_id,
            initial_capital=strategy_cfg.initial_capital,
            unit_size=pattern.unit_size,
        )
        cash = float(portfolio["cash"])
        realized_pnl_total = float(portfolio["realized_pnl"])
        total_fees = float(portfolio["total_fees"])
        total_tax_paid = float(portfolio.get("total_tax", 0.0))
        resolved_unit = int(portfolio["unit_size"])
        init_cap = float(portfolio["initial_capital"])

        positions: dict[str, dict[str, Any]] = {p["symbol"]: p for p in self.storage.get_positions(portfolio_id)}
        pending_orders = self.storage.get_pending_orders(portfolio_id)
        cooldowns = self.storage.get_cooldowns(portfolio_id)

        bar_ts_iso = bar_ts.isoformat()
        bar_dt = bar_ts.to_pydatetime()
        bar_minutes = INTERVAL_MINUTES.get(pattern.interval, 15)

        exact_bars, latest_snaps, recommendations, rec_by_sym, ml_preds = self._score_universe_at_bar(
            pattern=pattern,
            strategy_cfg=strategy_cfg,
            watchlist=watchlist,
            feat_map=feat_map,
            bar_ts=bar_ts,
            ml_model=ml_model,
        )
        executed_trades: list[dict[str, Any]] = []

        # === PHASE A: Execute pre-orders proposed BEFORE bar_ts at bar_ts OPEN ===
        pending_sells = [o for o in pending_orders if o["side"] == "SELL" and str(o["signal_time"]) < bar_ts_iso]
        pending_buys = [o for o in pending_orders if o["side"] == "BUY" and str(o["signal_time"]) < bar_ts_iso]
        pending_buys.sort(key=lambda o: float(o.get("score", 0.0)), reverse=True)

        for order in pending_sells:
            sym = order["symbol"]
            if sym not in positions:
                continue
            bar_row = exact_bars.get(sym)
            if bar_row is None:
                continue
            pos = positions[sym]
            exec_price = _apply_slippage(float(bar_row["Open"]), "SELL", strategy_cfg.slippage_bps)
            qty = int(pos["quantity"])
            gross = exec_price * qty
            sell_fee = gross * strategy_cfg.fee_rate
            buy_fee = float(pos.get("buy_fee", 0.0))
            net_pnl = (exec_price - float(pos["avg_cost"])) * qty - sell_fee - buy_fee

            realized_pnl_total, total_tax_paid, tax_delta = _apply_realized_tax(
                net_pnl, realized_pnl_total, total_tax_paid, strategy_cfg.tax_rate
            )
            cash += gross - sell_fee - tax_delta
            total_fees += sell_fee

            self.storage.record_trade(
                mode=portfolio_id,
                timestamp=bar_ts_iso,
                signal_time=str(order["signal_time"]),
                symbol=sym,
                name=pos["name"],
                side="SELL",
                quantity=qty,
                price=round(exec_price, 2),
                gross_amount=round(gross, 2),
                fee=round(sell_fee, 2),
                tax=round(tax_delta, 2),
                realized_pnl=round(net_pnl, 2),
                reason=f"[事前売り提案→始値約定] {order['reason']}",
            )
            self.storage.delete_position(portfolio_id, sym)
            del positions[sym]
            executed_trades.append(
                {
                    "timestamp": bar_ts_iso,
                    "signal_time": str(order["signal_time"]),
                    "symbol": sym,
                    "name": pos["name"],
                    "side": "SELL",
                    "quantity": qty,
                    "price": round(exec_price, 2),
                    "fee": round(sell_fee, 2),
                    "tax": round(tax_delta, 2),
                    "realized_pnl": round(net_pnl, 2),
                    "reason": f"[事前売り提案→始値約定] {order['reason']}",
                }
            )

        holdings_val_open = sum(
            float(exact_bars[s]["Open"]) * int(p["quantity"])
            if s in exact_bars
            else float(p["last_price"]) * int(p["quantity"])
            for s, p in positions.items()
        )
        total_equity_open = cash + holdings_val_open

        for order in pending_buys:
            if len(positions) >= strategy_cfg.max_positions:
                break
            sym = order["symbol"]
            if sym in positions:
                continue
            if cooldowns.get(sym, "") > bar_ts_iso:
                continue
            bar_row = exact_bars.get(sym)
            if bar_row is None:
                continue

            open_px = float(bar_row["Open"])
            qty = _compute_order_quantity(
                price=open_px,
                cash=cash,
                total_equity=total_equity_open,
                unit_size=resolved_unit,
                max_allocation_pct=strategy_cfg.max_allocation_pct,
                fee_rate=strategy_cfg.fee_rate,
                slippage_bps=strategy_cfg.slippage_bps,
            )
            if qty <= 0:
                continue

            exec_price = _apply_slippage(open_px, "BUY", strategy_cfg.slippage_bps)
            gross = exec_price * qty
            buy_fee = gross * strategy_cfg.fee_rate
            total_cost = gross + buy_fee
            if total_cost > cash:
                continue

            cash -= total_cost
            total_fees += buy_fee

            sig_px = float(order.get("signal_price") or exec_price)
            tp_ratio = (float(order["target_price"]) / sig_px) if sig_px > 0 and float(order.get("target_price", 0)) > 0 else (1.0 + strategy_cfg.take_profit_pct)
            sl_ratio = (float(order["stop_loss_price"]) / sig_px) if sig_px > 0 and float(order.get("stop_loss_price", 0)) > 0 else (1.0 - strategy_cfg.stop_loss_pct)

            new_pos = {
                "symbol": sym,
                "name": order["name"],
                "quantity": qty,
                "avg_cost": round(exec_price, 2),
                "buy_fee": round(buy_fee, 2),
                "highest_price": round(exec_price, 2),
                "target_price": _round_jpy(exec_price * max(1.005, tp_ratio)),
                "stop_loss_price": _round_jpy(max(1.0, exec_price * min(0.995, sl_ratio))),
                "entry_time": bar_ts_iso,
                "last_price": open_px,
                "bars_held": 0,
            }
            positions[sym] = new_pos
            self.storage.upsert_position(portfolio_id, new_pos)
            self.storage.record_trade(
                mode=portfolio_id,
                timestamp=bar_ts_iso,
                signal_time=str(order["signal_time"]),
                symbol=sym,
                name=order["name"],
                side="BUY",
                quantity=qty,
                price=round(exec_price, 2),
                gross_amount=round(gross, 2),
                fee=round(buy_fee, 2),
                tax=0.0,
                realized_pnl=0.0,
                reason=f"[事前買い予測→始値約定] {order['reason']}",
            )
            executed_trades.append(
                {
                    "timestamp": bar_ts_iso,
                    "signal_time": str(order["signal_time"]),
                    "symbol": sym,
                    "name": order["name"],
                    "side": "BUY",
                    "quantity": qty,
                    "price": round(exec_price, 2),
                    "fee": round(buy_fee, 2),
                    "tax": 0.0,
                    "realized_pnl": 0.0,
                    "reason": f"[事前買い予測→始値約定] {order['reason']}",
                }
            )

        # === PHASE B: Check pre-placed stop/trailing/take-profit during bar_ts ===
        for sym in list(positions.keys()):
            pos = positions[sym]
            bar_row = exact_bars.get(sym)
            if bar_row is None:
                continue
            b_open = float(bar_row["Open"])
            b_high = float(bar_row["High"])
            b_low = float(bar_row["Low"])
            b_close = float(bar_row["Close"])

            hit = _check_preplaced_exit_during_bar(pos, b_open, b_high, b_low, strategy_cfg)
            if hit is not None:
                trigger_px, hit_reason, is_sl = hit
                exec_price = _apply_slippage(trigger_px, "SELL", strategy_cfg.slippage_bps)
                qty = int(pos["quantity"])
                gross = exec_price * qty
                sell_fee = gross * strategy_cfg.fee_rate
                buy_fee = float(pos.get("buy_fee", 0.0))
                net_pnl = (exec_price - float(pos["avg_cost"])) * qty - sell_fee - buy_fee

                realized_pnl_total, total_tax_paid, tax_delta = _apply_realized_tax(
                    net_pnl, realized_pnl_total, total_tax_paid, strategy_cfg.tax_rate
                )
                cash += gross - sell_fee - tax_delta
                total_fees += sell_fee

                self.storage.record_trade(
                    mode=portfolio_id,
                    timestamp=bar_ts_iso,
                    signal_time=str(pos["entry_time"]),
                    symbol=sym,
                    name=pos["name"],
                    side="SELL",
                    quantity=qty,
                    price=round(exec_price, 2),
                    gross_amount=round(gross, 2),
                    fee=round(sell_fee, 2),
                    tax=round(tax_delta, 2),
                    realized_pnl=round(net_pnl, 2),
                    reason=hit_reason,
                )
                self.storage.delete_position(portfolio_id, sym)
                del positions[sym]
                if is_sl and pattern.cooldown_bars > 0:
                    until_iso = (bar_dt + timedelta(minutes=bar_minutes * pattern.cooldown_bars)).isoformat()
                    cooldowns[sym] = until_iso
                    self.storage.set_cooldown(portfolio_id, sym, until_iso)
                executed_trades.append(
                    {
                        "timestamp": bar_ts_iso,
                        "signal_time": str(pos["entry_time"]),
                        "symbol": sym,
                        "name": pos["name"],
                        "side": "SELL",
                        "quantity": qty,
                        "price": round(exec_price, 2),
                        "fee": round(sell_fee, 2),
                        "tax": round(tax_delta, 2),
                        "realized_pnl": round(net_pnl, 2),
                        "reason": hit_reason,
                    }
                )
            else:
                pos["last_price"] = b_close
                pos["highest_price"] = max(float(pos["highest_price"]), b_high, b_close)
                pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
                self.storage.upsert_position(portfolio_id, pos)

        # === PHASE C: Bar t CLOSE -> Queue Pre-Order Proposals for Next Bar (t+1) ===
        sell_evaluations: list[SellEvaluation] = []
        next_pending: list[dict[str, Any]] = []

        for sym, pos in positions.items():
            snap = latest_snaps.get(sym)
            rec = rec_by_sym.get(sym)
            if snap is None or rec is None:
                continue
            ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_dt)
            should_q, q_reason = self._should_queue_sell_for_strategy(pattern, pos, ev, ml_preds.get(sym))
            if should_q and not ev.should_sell:
                ev.should_sell = True
                ev.urgency = max(ev.urgency, 80.0)
                ev.timing_label = "次回始値で売却予定"
                ev.primary_reason = q_reason
            sell_evaluations.append(ev)
            if should_q:
                next_pending.append(
                    {
                        "symbol": sym,
                        "name": pos["name"],
                        "side": "SELL",
                        "signal_time": bar_ts_iso,
                        "signal_price": float(snap.close),
                        "score": rec.score,
                        "target_price": ev.target_price,
                        "stop_loss_price": ev.stop_loss_price,
                        "reason": q_reason,
                    }
                )

        queued_sell_syms = {o["symbol"] for o in next_pending if o["side"] == "SELL"}
        slots_to_propose = max(0, strategy_cfg.max_positions - (len(positions) - len(queued_sell_syms)))
        proposed_buys = 0
        est_cash = cash + sum(float(positions[s]["last_price"]) * int(positions[s]["quantity"]) for s in queued_sell_syms)

        for rec in recommendations:
            if proposed_buys >= slots_to_propose:
                break
            if rec.symbol in positions or rec.action not in ("STRONG_BUY", "BUY"):
                continue
            if cooldowns.get(rec.symbol, "") > bar_ts_iso:
                continue
            snap = latest_snaps.get(rec.symbol)
            if snap is None:
                continue
            unit_cost = float(snap.close) * resolved_unit * (1.0 + strategy_cfg.fee_rate)
            if unit_cost > est_cash:
                continue
            next_pending.append(
                {
                    "symbol": rec.symbol,
                    "name": rec.name,
                    "side": "BUY",
                    "signal_time": bar_ts_iso,
                    "signal_price": float(snap.close),
                    "score": rec.score,
                    "target_price": rec.target_price,
                    "stop_loss_price": rec.stop_loss_price,
                    "reason": f"{rec.action_label} (スコア {rec.score:.1f}): " + " / ".join(rec.reasons[:2]),
                }
            )
            proposed_buys += 1

        self.storage.replace_pending_orders(portfolio_id, next_pending)

        # === PHASE D: Record After-Tax Mark-to-Market Equity ===
        holdings_value, unrealized_pnl, est_unr_tax, after_tax_equity = _compute_after_tax_equity(
            cash=cash,
            positions=positions,
            cumulative_realized_pnl=realized_pnl_total,
            total_tax_paid=total_tax_paid,
            fee_rate=strategy_cfg.fee_rate,
            tax_rate=strategy_cfg.tax_rate,
        )
        return_pct = ((after_tax_equity - init_cap) / init_cap * 100.0) if init_cap > 0 else 0.0

        self.storage.update_portfolio(
            mode=portfolio_id,
            cash=round(cash, 2),
            realized_pnl=round(realized_pnl_total, 2),
            total_fees=round(total_fees, 2),
            total_tax=round(total_tax_paid, 2),
            last_bar_ts=bar_ts_iso,
        )
        self.storage.record_equity_point(
            mode=portfolio_id,
            timestamp=bar_ts_iso,
            cash=cash,
            holdings_value=holdings_value,
            total_equity=after_tax_equity,
            return_pct=return_pct,
            open_positions=len(positions),
        )

        sell_evaluations.sort(key=lambda e: e.urgency, reverse=True)
        self.storage.save_latest_signals(
            mode=portfolio_id,
            recommendations=[r.to_dict() for r in recommendations],
            sell_evaluations=[e.to_dict() for e in sell_evaluations],
            updated_at=now_jst().isoformat(),
        )

        return {
            "pattern_id": portfolio_id,
            "mode": portfolio_id,
            "bar_ts": bar_ts_iso,
            "total_equity": round(after_tax_equity, 1),
            "return_pct": round(return_pct, 2),
            "total_fees": round(total_fees, 1),
            "total_tax": round(total_tax_paid + est_unr_tax, 1),
            "trades_executed": executed_trades,
            "pending_orders_count": len(next_pending),
            "open_positions": len(positions),
            "recommendation_count": len(recommendations),
        }

    def run_catchup_pattern(
        self,
        pattern: SimulationPattern,
        interval: str | None = None,
        lookback_days: int = 7,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        benchmark_df: pd.DataFrame | None = None,
        export_snapshot: bool = True,
        portfolio_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Replay all unprocessed bars for one pattern while accumulating ML training data."""
        use_interval = interval or pattern.interval
        watchlist = self.cfg.get_watchlist(pattern.watchlist_key)
        strategy_cfg = pattern.to_strategy_config(self.cfg.watchlists)
        pid = portfolio_id_override or pattern.id

        portfolio = self.storage.ensure_portfolio(
            pid,
            initial_capital=pattern.initial_capital,
            unit_size=pattern.unit_size,
        )
        last_bar_ts_str = portfolio.get("last_bar_ts")

        # For ML patterns, fetch deep history (up to 55d) if stored bars aren't populated yet
        stored_stats = self.storage.get_ml_status()
        iv_bars = next((x["bars"] for x in stored_stats["by_interval"] if x["interval"] == use_interval), 0)
        fetch_days = 55 if (pattern.strategy_type == "ml_walkforward" and iv_bars < 1500 and preloaded_data is None) else max(lookback_days, 7)

        feat_map, _, macro_daily_df = self._prepare_feature_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=fetch_days,
            horizon_bars=pattern.ml_horizon_bars,
            preloaded_data=preloaded_data,
            benchmark_df=benchmark_df,
            persist=(preloaded_data is None),
        )
        if not feat_map:
            return {
                "pattern_id": pid,
                "name": pattern.name,
                "mode": pid,
                "bars_processed": 0,
                "trades_executed": 0,
                "message": "市場データを取得できませんでした。",
            }

        all_timestamps: set[pd.Timestamp] = set()
        for df in feat_map.values():
            all_timestamps.update(df.index)
        sorted_ts = sorted(all_timestamps)
        ts_index = {t: idx for idx, t in enumerate(sorted_ts)}

        if last_bar_ts_str:
            last_ts = pd.Timestamp(last_bar_ts_str)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize(JST)
            else:
                last_ts = last_ts.tz_convert(JST)
            target_ts = [ts for ts in sorted_ts if ts > last_ts]
        else:
            cutoff = sorted_ts[-1] - timedelta(days=1)
            warmup_idx = min(20, max(0, len(sorted_ts) - 1))
            target_ts = [ts for ts in sorted_ts[warmup_idx:] if ts >= cutoff]
            if not target_ts:
                target_ts = [sorted_ts[-1]]

        ml_model: WalkForwardEnsembleModel | None = None
        panel_df: pd.DataFrame | None = None
        if pattern.strategy_type == "ml_walkforward":
            prior_60m = (
                self._load_long_history_60m_panel([s.symbol for s in watchlist], macro_daily_df, horizon_60m=4)
                if use_interval == "30m"
                else None
            )
            panel_df = build_training_panel(feat_map, extra_prior_panel=prior_60m)
            ml_model = WalkForwardEnsembleModel(horizon_bars=pattern.ml_horizon_bars)
            sources_desc = "Yahoo日中足 + Yahoo 60m(長期) + FRED米連銀(ドル円/VIX/米10年債)" if (prior_60m is not None and not prior_60m.empty) else "Yahoo日中足 + FRED米連銀(ドル円/VIX/米10年債)"
            diag = evaluate_walkforward_oos(
                panel_df,
                interval=use_interval,
                horizon_bars=pattern.ml_horizon_bars,
                min_pred_return=pattern.ml_min_pred_return,
                data_sources=sources_desc,
            )
            self.storage.save_ml_diagnostics(use_interval, diag.to_dict(), record_history=True)

        total_trades = 0
        last_res: dict[str, Any] = {}
        retrain_counter = 0
        for ts in target_ts:
            if ml_model is not None and panel_df is not None:
                idx = ts_index[ts]
                cutoff_idx = idx - pattern.ml_horizon_bars - 1
                if cutoff_idx >= 20 and ((not ml_model.is_fitted) or retrain_counter % 15 == 0):
                    ml_model.fit_from_panel(panel_df, sorted_ts[cutoff_idx])
                retrain_counter += 1

            last_res = self._execute_bar(pid, pattern, strategy_cfg, ts, watchlist, feat_map, ml_model=ml_model)
            total_trades += len(last_res.get("trades_executed", []))

        if not target_ts and sorted_ts:
            if ml_model is not None and panel_df is not None and len(sorted_ts) > pattern.ml_horizon_bars + 25:
                ml_model.fit_from_panel(panel_df, sorted_ts[-pattern.ml_horizon_bars - 1])
            self._refresh_signals_only(pid, pattern, strategy_cfg, sorted_ts[-1], watchlist, feat_map, ml_model=ml_model)

        if export_snapshot:
            self.storage.export_dashboard_snapshot(self.cfg)

        return {
            "pattern_id": pid,
            "name": pattern.name,
            "mode": pid,
            "strategy_type": pattern.strategy_type,
            "interval": use_interval,
            "initial_capital": pattern.initial_capital,
            "unit_size": pattern.unit_size,
            "bars_processed": len(target_ts),
            "trades_executed": total_trades,
            "latest_bar": target_ts[-1].isoformat() if target_ts else last_bar_ts_str,
            "total_equity": last_res.get("total_equity"),
            "return_pct": last_res.get("return_pct"),
            "total_fees": last_res.get("total_fees"),
            "total_tax": last_res.get("total_tax"),
            "pending_orders_count": last_res.get("pending_orders_count"),
        }

    def run_catchup_all_patterns(
        self,
        lookback_days: int = 7,
        pattern_ids: list[str] | None = None,
        override_interval: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run catch-up across all enabled patterns in parallel, batch-fetching & persisting ML data once per interval."""
        if pattern_ids:
            targets = [p for p in self.cfg.patterns if p.id in set(pattern_ids)]
        else:
            targets = self.cfg.get_enabled_patterns()

        if not targets:
            return []

        interval_symbols: dict[str, set[str]] = defaultdict(set)
        interval_needs_ml: dict[str, bool] = defaultdict(bool)
        for p in targets:
            iv = override_interval or p.interval
            for s in self.cfg.get_watchlist(p.watchlist_key):
                interval_symbols[iv].add(s.symbol)
            if p.strategy_type == "ml_walkforward":
                interval_needs_ml[iv] = True

        stored_stats = self.storage.get_ml_status()
        stored_counts = {x["interval"]: x["bars"] for x in stored_stats["by_interval"]}

        # Auto-bootstrap external FRED macro data (5y) and 60m long history (~1-2y) if not yet persisted
        if any(interval_needs_ml.values()) and isinstance(self.provider, MarketDataProvider):
            self._ensure_macro_daily_frame(force_refresh=(stored_stats.get("macro_total_rows", 0) == 0), allow_network=True)
            if stored_counts.get("60m", 0) < 1000:
                all_ml_syms = sorted({s for syms in interval_symbols.values() for s in syms})
                raw_60m, bm_60m = self._fetch_and_accumulate_bars(
                    symbols=all_ml_syms,
                    interval="60m",
                    period_days=650,
                    include_benchmark=True,
                    persist=False,
                )
                if raw_60m:
                    to_save = {**raw_60m}
                    if bm_60m is not None and not bm_60m.empty:
                        to_save["^N225"] = bm_60m
                    self.storage.save_ml_bars("60m", to_save, max_bars_per_symbol=1500)

        merged_by_interval: dict[str, dict[str, pd.DataFrame]] = {}
        bm_by_interval: dict[str, pd.DataFrame] = {}

        for iv, syms in interval_symbols.items():
            # Bootstrap 55 days on first ML run; otherwise incremental 10-day fetch merged with SQLite history!
            fetch_days = 55 if (interval_needs_ml[iv] and stored_counts.get(iv, 0) < 1500) else max(lookback_days, 10)
            merged_map, bm_df = self._fetch_and_accumulate_bars(
                symbols=sorted(syms),
                interval=iv,
                period_days=fetch_days,
                include_benchmark=True,
                persist=True,
            )
            merged_by_interval[iv] = merged_map
            bm_by_interval[iv] = bm_df

        results: list[dict[str, Any]] = []
        for p in targets:
            iv = override_interval or p.interval
            res = self.run_catchup_pattern(
                pattern=p,
                interval=iv,
                lookback_days=lookback_days,
                preloaded_data=merged_by_interval.get(iv),
                benchmark_df=bm_by_interval.get(iv),
                export_snapshot=False,
            )
            results.append(res)

        self.storage.export_dashboard_snapshot(self.cfg)
        return results

    def run_catchup(
        self,
        mode: str = "lot1",
        interval: str | None = None,
        lookback_days: int = 7,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible single mode/pattern catch-up."""
        pattern = self._resolve_pattern(mode)
        pid_override = mode if mode in ("lot1", "lot100") and mode != pattern.id else pattern.id
        return self.run_catchup_pattern(
            pattern=pattern,
            interval=interval,
            lookback_days=lookback_days,
            preloaded_data=preloaded_data,
            export_snapshot=True,
            portfolio_id_override=pid_override,
        )

    def run_step(
        self,
        mode: str = "lot1",
        interval: str | None = None,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Process just the latest bar for a mode or pattern."""
        pattern = self._resolve_pattern(mode)
        pid = mode if mode in ("lot1", "lot100") and mode != pattern.id else pattern.id
        use_interval = interval or pattern.interval
        watchlist = self.cfg.get_watchlist(pattern.watchlist_key)
        strategy_cfg = pattern.to_strategy_config(self.cfg.watchlists)
        portfolio = self.storage.ensure_portfolio(
            pid,
            initial_capital=pattern.initial_capital,
            unit_size=pattern.unit_size,
        )
        last_bar_ts_str = portfolio.get("last_bar_ts")

        feat_map, _, _ = self._prepare_feature_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=14,
            horizon_bars=pattern.ml_horizon_bars,
            preloaded_data=preloaded_data,
            persist=(preloaded_data is None),
        )
        if not feat_map:
            return {"pattern_id": pid, "mode": pid, "bars_processed": 0, "message": "市場データを取得できませんでした。"}

        all_ts = sorted({ts for df in feat_map.values() for ts in df.index})
        latest_ts = all_ts[-1]

        ml_model: WalkForwardRidgeModel | None = None
        if pattern.strategy_type == "ml_walkforward" and len(all_ts) > pattern.ml_horizon_bars + 25:
            panel_df = build_training_panel(feat_map)
            ml_model = WalkForwardRidgeModel(horizon_bars=pattern.ml_horizon_bars)
            ml_model.fit_from_panel(panel_df, all_ts[-pattern.ml_horizon_bars - 1])

        if last_bar_ts_str:
            prev_ts = pd.Timestamp(last_bar_ts_str)
            if prev_ts.tzinfo is None:
                prev_ts = prev_ts.tz_localize(JST)
            else:
                prev_ts = prev_ts.tz_convert(JST)
            if latest_ts <= prev_ts:
                self._refresh_signals_only(pid, pattern, strategy_cfg, latest_ts, watchlist, feat_map, ml_model=ml_model)
                self.storage.export_dashboard_snapshot(self.cfg)
                return {
                    "pattern_id": pid,
                    "mode": pid,
                    "bars_processed": 0,
                    "latest_bar": latest_ts.isoformat(),
                    "message": "最新バーは処理済みです（推奨シグナルのみ更新）。",
                }

        res = self._execute_bar(pid, pattern, strategy_cfg, latest_ts, watchlist, feat_map, ml_model=ml_model)
        self.storage.export_dashboard_snapshot(self.cfg)
        return {
            "pattern_id": pid,
            "mode": pid,
            "bars_processed": 1,
            "trades_executed": len(res["trades_executed"]),
            "latest_bar": latest_ts.isoformat(),
            "total_equity": res["total_equity"],
            "return_pct": res["return_pct"],
        }

    def _refresh_signals_only(
        self,
        portfolio_id: str,
        pattern: SimulationPattern,
        strategy_cfg: AppConfig,
        bar_ts: pd.Timestamp,
        watchlist: list[StockInfo],
        feat_map: dict[str, pd.DataFrame],
        ml_model: WalkForwardRidgeModel | None = None,
    ) -> None:
        positions = self.storage.get_positions(portfolio_id)
        pos_by_sym = {p["symbol"]: p for p in positions}
        _, latest_snaps, recs, rec_by_sym, _ = self._score_universe_at_bar(
            pattern=pattern,
            strategy_cfg=strategy_cfg,
            watchlist=watchlist,
            feat_map=feat_map,
            bar_ts=bar_ts,
            ml_model=ml_model,
        )
        sell_evals: list[SellEvaluation] = []
        for sym, pos in pos_by_sym.items():
            snap = latest_snaps.get(sym)
            rec = rec_by_sym.get(sym)
            if snap is None or rec is None:
                continue
            pos["last_price"] = float(snap.close)
            ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_ts.to_pydatetime())
            sell_evals.append(ev)
        sell_evals.sort(key=lambda e: e.urgency, reverse=True)
        self.storage.save_latest_signals(
            mode=portfolio_id,
            recommendations=[r.to_dict() for r in recs],
            sell_evaluations=[e.to_dict() for e in sell_evals],
            updated_at=now_jst().isoformat(),
        )

    def run_backtest_pattern(
        self,
        pattern: SimulationPattern,
        days: int = 10,
        override_interval: str | None = None,
        override_capital: float | None = None,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        benchmark_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run an isolated strict zero-lookahead backtest (including Walk-Forward ML, fees, and 20.315% tax)."""
        init_cap = float(override_capital if override_capital is not None else pattern.initial_capital)
        use_interval = override_interval or pattern.interval
        unit_size = pattern.unit_size
        watchlist = self.cfg.get_watchlist(pattern.watchlist_key)
        strategy_cfg = pattern.to_strategy_config(self.cfg.watchlists)

        # ML needs extra historical warmup bars before day 1 for strictly causal walk-forward training
        extra_warmup_days = 25 if pattern.strategy_type == "ml_walkforward" else 5
        fetch_days = min(58, max(days + extra_warmup_days, 12))
        feat_map, bm_raw, macro_daily_df = self._prepare_feature_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=fetch_days,
            horizon_bars=pattern.ml_horizon_bars,
            preloaded_data=preloaded_data,
            benchmark_df=benchmark_df,
            persist=(preloaded_data is None),
        )
        if not feat_map:
            return {"error": "市場データを取得できませんでした。ネットワーク接続を確認してください。"}

        all_timestamps: set[pd.Timestamp] = set()
        for df in feat_map.values():
            all_timestamps.update(df.index)
        sorted_ts = sorted(all_timestamps)
        if len(sorted_ts) < 5:
            return {"error": "バックテストに十分なバー数がありません。"}

        ts_to_global_idx = {ts: i for i, ts in enumerate(sorted_ts)}
        end_ts = sorted_ts[-1]
        start_cutoff = end_ts - timedelta(days=days)
        min_warmup = 60 if pattern.strategy_type == "ml_walkforward" else 26
        warmup_start = min(min_warmup, max(2, len(sorted_ts) // 4))
        sim_timestamps = [ts for ts in sorted_ts[warmup_start:] if ts >= start_cutoff]
        if not sim_timestamps:
            sim_timestamps = sorted_ts[warmup_start:]

        ml_model: WalkForwardEnsembleModel | None = None
        panel_df: pd.DataFrame | None = None
        if pattern.strategy_type == "ml_walkforward":
            prior_60m = (
                self._load_long_history_60m_panel([s.symbol for s in watchlist], macro_daily_df, horizon_60m=4)
                if (use_interval == "30m" and preloaded_data is None)
                else None
            )
            panel_df = build_training_panel(feat_map, extra_prior_panel=prior_60m)
            ml_model = WalkForwardEnsembleModel(horizon_bars=pattern.ml_horizon_bars)

        cash = init_cap
        realized_pnl_total = 0.0
        total_fees = 0.0
        total_tax_paid = 0.0
        positions: dict[str, dict[str, Any]] = {}
        pending_orders: list[dict[str, Any]] = []
        cooldown_until_idx: dict[str, int] = {}
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []

        sym_ts_pos: dict[str, dict[pd.Timestamp, int]] = {
            sym: {ts: idx for idx, ts in enumerate(df.index)}
            for sym, df in feat_map.items()
        }
        last_known_idx: dict[str, int] = {}

        for step_i, bar_ts in enumerate(sim_timestamps):
            bar_ts_iso = bar_ts.isoformat()
            bar_dt = bar_ts.to_pydatetime()
            global_i = ts_to_global_idx[bar_ts]

            # Retrain Walk-Forward ML strictly on bars whose forward label is already known before bar_ts
            if ml_model is not None and panel_df is not None:
                cutoff_idx = global_i - pattern.ml_horizon_bars - 1
                if cutoff_idx >= 25 and ((not ml_model.is_fitted) or step_i % 15 == 0):
                    ml_model.fit_from_panel(panel_df, sorted_ts[cutoff_idx])

            exact_bars: dict[str, Any] = {}
            recs: list[StockRecommendation] = []
            snaps: dict[str, Any] = {}
            rec_by_sym: dict[str, StockRecommendation] = {}
            ml_preds: dict[str, float] = {}

            for stock in watchlist:
                ind_df = feat_map.get(stock.symbol)
                if ind_df is None or ind_df.empty:
                    continue
                idx = sym_ts_pos[stock.symbol].get(bar_ts)
                if idx is not None:
                    last_known_idx[stock.symbol] = idx
                    exact_bars[stock.symbol] = ind_df.iloc[idx]
                else:
                    idx = last_known_idx.get(stock.symbol)
                if idx is None or idx < 1:
                    continue

                row = ind_df.iloc[idx]
                snap = extract_latest_snapshot(ind_df.iloc[max(0, idx - 1) : idx + 1])
                if snap is None:
                    continue

                if pattern.strategy_type == "ml_walkforward" and ml_model is not None and ml_model.is_fitted:
                    pred = ml_model.predict_row(row)
                    if pred is not None:
                        ml_preds[stock.symbol] = pred
                        rec = ml_prediction_to_recommendation(
                            stock, snap, pred, strategy_cfg, min_pred_return=pattern.ml_min_pred_return
                        )
                    else:
                        rec = score_stock_mean_reversion(stock, snap, strategy_cfg)
                elif pattern.strategy_type == "mean_reversion":
                    rec = score_stock_mean_reversion(stock, snap, strategy_cfg)
                else:
                    rec = score_stock(stock, snap, strategy_cfg)

                recs.append(rec)
                snaps[stock.symbol] = snap
                rec_by_sym[stock.symbol] = rec

            recs.sort(key=lambda r: r.score, reverse=True)

            # === PHASE A: Execute pre-orders from previous bar at bar_ts OPEN ===
            pending_sells = [o for o in pending_orders if o["side"] == "SELL" and str(o["signal_time"]) < bar_ts_iso]
            pending_buys = [o for o in pending_orders if o["side"] == "BUY" and str(o["signal_time"]) < bar_ts_iso]
            pending_buys.sort(key=lambda o: float(o.get("score", 0.0)), reverse=True)

            for order in pending_sells:
                sym = order["symbol"]
                if sym not in positions or sym not in exact_bars:
                    continue
                pos = positions[sym]
                exec_price = _apply_slippage(float(exact_bars[sym]["Open"]), "SELL", strategy_cfg.slippage_bps)
                qty = int(pos["quantity"])
                gross = exec_price * qty
                sell_fee = gross * strategy_cfg.fee_rate
                buy_fee = float(pos.get("buy_fee", 0.0))
                net_pnl = (exec_price - float(pos["avg_cost"])) * qty - sell_fee - buy_fee

                realized_pnl_total, total_tax_paid, tax_delta = _apply_realized_tax(
                    net_pnl, realized_pnl_total, total_tax_paid, strategy_cfg.tax_rate
                )
                cash += gross - sell_fee - tax_delta
                total_fees += sell_fee
                trades.append(
                    {
                        "timestamp": bar_ts_iso,
                        "signal_time": str(order["signal_time"]),
                        "symbol": sym,
                        "name": pos["name"],
                        "side": "SELL",
                        "quantity": qty,
                        "price": round(exec_price, 2),
                        "fee": round(sell_fee, 2),
                        "tax": round(tax_delta, 2),
                        "realized_pnl": round(net_pnl, 2),
                        "reason": f"[事前売り提案→始値約定] {order['reason']}",
                    }
                )
                del positions[sym]

            holdings_val_open = sum(
                float(exact_bars[s]["Open"]) * int(p["quantity"])
                if s in exact_bars
                else float(p["last_price"]) * int(p["quantity"])
                for s, p in positions.items()
            )
            total_eq_open = cash + holdings_val_open

            for order in pending_buys:
                if len(positions) >= strategy_cfg.max_positions:
                    break
                sym = order["symbol"]
                if sym in positions or sym not in exact_bars:
                    continue
                if cooldown_until_idx.get(sym, -1) > step_i:
                    continue
                open_px = float(exact_bars[sym]["Open"])
                qty = _compute_order_quantity(
                    price=open_px,
                    cash=cash,
                    total_equity=total_eq_open,
                    unit_size=unit_size,
                    max_allocation_pct=strategy_cfg.max_allocation_pct,
                    fee_rate=strategy_cfg.fee_rate,
                    slippage_bps=strategy_cfg.slippage_bps,
                )
                if qty <= 0:
                    continue
                exec_price = _apply_slippage(open_px, "BUY", strategy_cfg.slippage_bps)
                gross = exec_price * qty
                buy_fee = gross * strategy_cfg.fee_rate
                if gross + buy_fee > cash:
                    continue
                cash -= gross + buy_fee
                total_fees += buy_fee

                sig_px = float(order.get("signal_price") or exec_price)
                tp_ratio = (float(order["target_price"]) / sig_px) if sig_px > 0 and float(order.get("target_price", 0)) > 0 else (1.0 + strategy_cfg.take_profit_pct)
                sl_ratio = (float(order["stop_loss_price"]) / sig_px) if sig_px > 0 and float(order.get("stop_loss_price", 0)) > 0 else (1.0 - strategy_cfg.stop_loss_pct)
                positions[sym] = {
                    "symbol": sym,
                    "name": order["name"],
                    "quantity": qty,
                    "avg_cost": round(exec_price, 2),
                    "buy_fee": round(buy_fee, 2),
                    "highest_price": round(exec_price, 2),
                    "target_price": _round_jpy(exec_price * max(1.005, tp_ratio)),
                    "stop_loss_price": _round_jpy(max(1.0, exec_price * min(0.995, sl_ratio))),
                    "entry_time": bar_ts_iso,
                    "last_price": open_px,
                    "bars_held": 0,
                }
                trades.append(
                    {
                        "timestamp": bar_ts_iso,
                        "signal_time": str(order["signal_time"]),
                        "symbol": sym,
                        "name": order["name"],
                        "side": "BUY",
                        "quantity": qty,
                        "price": round(exec_price, 2),
                        "fee": round(buy_fee, 2),
                        "tax": 0.0,
                        "realized_pnl": 0.0,
                        "reason": f"[事前買い予測→始値約定] {order['reason']}",
                    }
                )

            # === PHASE B: Pre-placed stop/trailing/take-profit during bar_ts ===
            for sym in list(positions.keys()):
                if sym not in exact_bars:
                    continue
                pos = positions[sym]
                bar_row = exact_bars[sym]
                b_open, b_high, b_low, b_close = (
                    float(bar_row["Open"]),
                    float(bar_row["High"]),
                    float(bar_row["Low"]),
                    float(bar_row["Close"]),
                )

                hit = _check_preplaced_exit_during_bar(pos, b_open, b_high, b_low, strategy_cfg)
                if hit is not None:
                    trigger_px, hit_reason, is_sl = hit
                    exec_price = _apply_slippage(trigger_px, "SELL", strategy_cfg.slippage_bps)
                    qty = int(pos["quantity"])
                    gross = exec_price * qty
                    sell_fee = gross * strategy_cfg.fee_rate
                    buy_fee = float(pos.get("buy_fee", 0.0))
                    net_pnl = (exec_price - float(pos["avg_cost"])) * qty - sell_fee - buy_fee

                    realized_pnl_total, total_tax_paid, tax_delta = _apply_realized_tax(
                        net_pnl, realized_pnl_total, total_tax_paid, strategy_cfg.tax_rate
                    )
                    cash += gross - sell_fee - tax_delta
                    total_fees += sell_fee
                    trades.append(
                        {
                            "timestamp": bar_ts_iso,
                            "signal_time": str(pos["entry_time"]),
                            "symbol": sym,
                            "name": pos["name"],
                            "side": "SELL",
                            "quantity": qty,
                            "price": round(exec_price, 2),
                            "fee": round(sell_fee, 2),
                            "tax": round(tax_delta, 2),
                            "realized_pnl": round(net_pnl, 2),
                            "reason": hit_reason,
                        }
                    )
                    if is_sl and pattern.cooldown_bars > 0:
                        cooldown_until_idx[sym] = step_i + pattern.cooldown_bars
                    del positions[sym]
                else:
                    pos["last_price"] = b_close
                    pos["highest_price"] = max(float(pos["highest_price"]), b_high, b_close)
                    pos["bars_held"] = int(pos.get("bars_held", 0)) + 1

            # === PHASE C: Bar t CLOSE -> Queue pre-orders for next bar (t+1) ===
            next_pending: list[dict[str, Any]] = []
            for sym, pos in positions.items():
                snap = snaps.get(sym)
                rec = rec_by_sym.get(sym)
                if snap is None or rec is None:
                    continue
                ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_dt)
                should_q, q_reason = self._should_queue_sell_for_strategy(pattern, pos, ev, ml_preds.get(sym))
                if should_q:
                    next_pending.append(
                        {
                            "symbol": sym,
                            "name": pos["name"],
                            "side": "SELL",
                            "signal_time": bar_ts_iso,
                            "signal_price": float(snap.close),
                            "score": rec.score,
                            "target_price": ev.target_price,
                            "stop_loss_price": ev.stop_loss_price,
                            "reason": q_reason,
                        }
                    )

            queued_sell_syms = {o["symbol"] for o in next_pending if o["side"] == "SELL"}
            slots = max(0, strategy_cfg.max_positions - (len(positions) - len(queued_sell_syms)))
            proposed_buys = 0
            est_cash = cash + sum(float(positions[s]["last_price"]) * int(positions[s]["quantity"]) for s in queued_sell_syms)
            for rec in recs:
                if proposed_buys >= slots:
                    break
                if rec.symbol in positions or rec.action not in ("STRONG_BUY", "BUY"):
                    continue
                if cooldown_until_idx.get(rec.symbol, -1) > step_i:
                    continue
                snap = snaps.get(rec.symbol)
                if snap is None:
                    continue
                unit_cost = float(snap.close) * unit_size * (1.0 + strategy_cfg.fee_rate)
                if unit_cost > est_cash:
                    continue
                next_pending.append(
                    {
                        "symbol": rec.symbol,
                        "name": rec.name,
                        "side": "BUY",
                        "signal_time": bar_ts_iso,
                        "signal_price": float(snap.close),
                        "score": rec.score,
                        "target_price": rec.target_price,
                        "stop_loss_price": rec.stop_loss_price,
                        "reason": f"{rec.action_label} (スコア {rec.score:.1f}): " + " / ".join(rec.reasons[:2]),
                    }
                )
                proposed_buys += 1

            pending_orders = next_pending

            # === PHASE D: Mark-to-market After-Tax Equity ===
            holdings_val, unrealized_pnl, est_unr_tax, after_tax_eq = _compute_after_tax_equity(
                cash=cash,
                positions=positions,
                cumulative_realized_pnl=realized_pnl_total,
                total_tax_paid=total_tax_paid,
                fee_rate=strategy_cfg.fee_rate,
                tax_rate=strategy_cfg.tax_rate,
            )
            ret_pct = (after_tax_eq - init_cap) / init_cap * 100.0
            equity_curve.append(
                {
                    "timestamp": bar_ts_iso,
                    "total_equity": round(after_tax_eq, 1),
                    "cash": round(cash, 1),
                    "holdings_value": round(holdings_val, 1),
                    "return_pct": round(ret_pct, 2),
                    "open_positions": len(positions),
                }
            )

        final_holdings, unrealized_pnl, est_unr_tax, final_after_tax_equity = _compute_after_tax_equity(
            cash=cash,
            positions=positions,
            cumulative_realized_pnl=realized_pnl_total,
            total_tax_paid=total_tax_paid,
            fee_rate=strategy_cfg.fee_rate,
            tax_rate=strategy_cfg.tax_rate,
        )
        total_tax_all = total_tax_paid + est_unr_tax
        net_pnl = final_after_tax_equity - init_cap
        pre_tax_equity = cash + final_holdings + total_tax_paid
        gross_pnl_before_tax = pre_tax_equity - init_cap
        return_pct = (net_pnl / init_cap * 100.0) if init_cap > 0 else 0.0

        sell_trades = [t for t in trades if t["side"] == "SELL"]
        wins = [t for t in sell_trades if t["realized_pnl"] > 0]
        losses = [t for t in sell_trades if t["realized_pnl"] < 0]
        win_rate = (len(wins) / len(sell_trades) * 100.0) if sell_trades else 0.0
        gross_profit = sum(t["realized_pnl"] for t in wins)
        gross_loss = abs(sum(t["realized_pnl"] for t in losses))
        profit_factor = (
            round(gross_profit / gross_loss, 2)
            if gross_loss > 0
            else (round(gross_profit, 2) if gross_profit > 0 else 0.0)
        )

        peak = init_cap
        max_dd_pct = 0.0
        for pt in equity_curve:
            if pt["total_equity"] > peak:
                peak = pt["total_equity"]
            dd = (peak - pt["total_equity"]) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

        benchmark_return_pct = None
        try:
            if bm_raw is not None and not bm_raw.empty and sim_timestamps:
                bm_sliced = bm_raw.loc[sim_timestamps[0] : sim_timestamps[-1]]
                if len(bm_sliced) >= 2:
                    bm_start = float(bm_sliced["Close"].iloc[0])
                    bm_end = float(bm_sliced["Close"].iloc[-1])
                    if bm_start > 0:
                        benchmark_return_pct = round((bm_end - bm_start) / bm_start * 100.0, 2)
        except Exception:
            benchmark_return_pct = None

        return {
            "pattern_id": pattern.id,
            "pattern_name": pattern.name,
            "strategy_type": pattern.strategy_type,
            "mode": pattern.id,
            "mode_label": pattern.name,
            "unit_size": unit_size,
            "interval": use_interval,
            "days": days,
            "execution_policy": "事前予測(t足) → 次足(t+1)始値約定 / 手数料＋税控除",
            "bars_simulated": len(sim_timestamps),
            "start_time": sim_timestamps[0].isoformat(),
            "end_time": sim_timestamps[-1].isoformat(),
            "symbols_loaded": len(feat_map),
            "initial_capital": round(init_cap, 1),
            "final_equity": round(final_after_tax_equity, 1),
            "pre_tax_equity": round(pre_tax_equity, 1),
            "cash": round(cash, 1),
            "holdings_value": round(final_holdings, 1),
            "net_pnl": round(net_pnl, 1),
            "gross_pnl_before_tax": round(gross_pnl_before_tax, 1),
            "return_pct": round(return_pct, 2),
            "realized_pnl": round(realized_pnl_total, 1),
            "unrealized_pnl": round(unrealized_pnl, 1),
            "total_fees": round(total_fees, 1),
            "total_tax": round(total_tax_all, 1),
            "fee_rate": strategy_cfg.fee_rate,
            "tax_rate": strategy_cfg.tax_rate,
            "total_trades": len(trades),
            "closed_trades": len(sell_trades),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "benchmark_nikkei225_pct": benchmark_return_pct,
            "ml_train_samples": ml_model.train_samples if ml_model else 0,
            "open_positions": list(positions.values()),
            "pending_orders": pending_orders,
            "equity_curve": equity_curve,
            "trades": list(reversed(trades)),
        }

    def run_backtest_compare(
        self,
        patterns: list[SimulationPattern] | None = None,
        days: int = 10,
        override_capital: float | None = None,
        override_interval: str | None = None,
    ) -> dict[str, Any]:
        """Run parallel zero-lookahead real-data backtests across multiple patterns (including ML)."""
        targets = patterns if patterns is not None else self.cfg.get_enabled_patterns()
        if not targets:
            return {"error": "有効なシミュレーションパターンがありません。"}

        interval_symbols: dict[str, set[str]] = defaultdict(set)
        interval_needs_ml: dict[str, bool] = defaultdict(bool)
        for p in targets:
            iv = override_interval or p.interval
            for s in self.cfg.get_watchlist(p.watchlist_key):
                interval_symbols[iv].add(s.symbol)
            if p.strategy_type == "ml_walkforward":
                interval_needs_ml[iv] = True

        fetched_by_interval: dict[str, dict[str, pd.DataFrame]] = {}
        benchmarks_by_interval: dict[str, pd.DataFrame] = {}
        for iv, syms in interval_symbols.items():
            extra = 25 if interval_needs_ml[iv] else 5
            fetch_days = min(58, max(days + extra, 12))
            raw_map, bm_df = self._fetch_and_accumulate_bars(
                symbols=sorted(syms),
                interval=iv,
                period_days=fetch_days,
                include_benchmark=True,
                persist=True,
            )
            fetched_by_interval[iv] = raw_map
            benchmarks_by_interval[iv] = bm_df

        results: list[dict[str, Any]] = []
        for p in targets:
            iv = override_interval or p.interval
            res = self.run_backtest_pattern(
                pattern=p,
                days=days,
                override_interval=iv,
                override_capital=override_capital,
                preloaded_data=fetched_by_interval.get(iv),
                benchmark_df=benchmarks_by_interval.get(iv),
            )
            if "error" not in res:
                results.append(res)

        if not results:
            return {"error": "比較バックテストの市場データを取得できませんでした。"}

        results.sort(key=lambda r: float(r["return_pct"]), reverse=True)
        return {
            "compare_mode": True,
            "days": days,
            "pattern_count": len(results),
            "best_pattern": results[0]["pattern_name"],
            "best_return_pct": results[0]["return_pct"],
            "results": results,
        }

    def run_backtest(
        self,
        mode: str = "lot1",
        interval: str = "15m",
        days: int = 10,
        initial_capital: float | None = None,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        benchmark_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible single mode/pattern backtest."""
        pattern = self._resolve_pattern(mode)
        return self.run_backtest_pattern(
            pattern=pattern,
            days=days,
            override_interval=interval,
            override_capital=initial_capital,
            preloaded_data=preloaded_data,
            benchmark_df=benchmark_df,
        )
