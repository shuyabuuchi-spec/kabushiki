"""Paper-trading simulator, PC-OFF catch-up engine, and parallel multi-pattern backtester."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any
import math
import pandas as pd

from .advisor import score_stock, evaluate_sell_timing, StockRecommendation, SellEvaluation
from .config import AppConfig, SimulationPattern, MODE_UNIT_SIZE, MODE_LABELS, StockInfo
from .indicators import compute_indicators, extract_latest_snapshot
from .market_data import JST, MarketDataProvider, now_jst
from .storage import Storage


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
    """Compute tradable share quantity respecting unit_size (1 or 100) and cash."""
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


class TradingSimulator:
    """Multi-pattern paper-trading simulator with PC-OFF catch-up and parallel backtesting."""

    def __init__(
        self,
        cfg: AppConfig,
        storage: Storage,
        provider: MarketDataProvider | None = None,
    ):
        self.cfg = cfg
        self.storage = storage
        self.provider = provider or MarketDataProvider()

    def _prepare_indicator_frames(
        self,
        watchlist: list[StockInfo],
        interval: str,
        period_days: int,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        symbols = [s.symbol for s in watchlist]
        raw_map = (
            preloaded_data
            if preloaded_data is not None
            else self.provider.fetch_many(symbols, interval=interval, period_days=period_days)
        )
        ind_map: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = raw_map.get(sym)
            if df is not None and not df.empty:
                ind_map[sym] = compute_indicators(df)
        return ind_map

    def _resolve_pattern(self, mode_or_pattern_id: str) -> SimulationPattern:
        p = self.cfg.get_pattern(mode_or_pattern_id)
        if p is not None:
            return p
        unit_size = MODE_UNIT_SIZE.get(mode_or_pattern_id, 1)
        return SimulationPattern(
            id=mode_or_pattern_id,
            name=MODE_LABELS.get(mode_or_pattern_id, mode_or_pattern_id),
            initial_capital=self.cfg.initial_capital,
            unit_size=unit_size,
            watchlist_key="lot100" if unit_size >= 100 else "lot1",
            interval=self.cfg.interval,
            max_positions=self.cfg.max_positions,
            max_allocation_pct=self.cfg.max_allocation_pct,
            fee_rate=self.cfg.fee_rate,
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

    def _execute_bar(
        self,
        portfolio_id: str,
        strategy_cfg: AppConfig,
        unit_size: int,
        bar_ts: pd.Timestamp,
        watchlist: list[StockInfo],
        ind_map: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Process a single bar timestamp chronologically for one pattern portfolio."""
        portfolio = self.storage.ensure_portfolio(
            portfolio_id,
            initial_capital=strategy_cfg.initial_capital,
            unit_size=unit_size,
        )
        cash = float(portfolio["cash"])
        realized_pnl_total = float(portfolio["realized_pnl"])
        total_fees = float(portfolio["total_fees"])
        resolved_unit = int(portfolio["unit_size"])
        init_cap = float(portfolio["initial_capital"])

        positions_list = self.storage.get_positions(portfolio_id)
        positions: dict[str, dict[str, Any]] = {p["symbol"]: p for p in positions_list}

        bar_ts_iso = bar_ts.isoformat()
        bar_dt = bar_ts.to_pydatetime()

        recommendations: list[StockRecommendation] = []
        latest_snaps: dict[str, Any] = {}
        rec_by_sym: dict[str, StockRecommendation] = {}

        # 1. Score every stock using data up to bar_ts (strictly no lookahead!)
        for stock in watchlist:
            ind_df = ind_map.get(stock.symbol)
            if ind_df is None or ind_df.empty:
                continue
            pos_idx = int(ind_df.index.searchsorted(bar_ts, side="right")) - 1
            if pos_idx < 1:
                continue
            sliced = ind_df.iloc[pos_idx - 1 : pos_idx + 1]
            snap = extract_latest_snapshot(sliced)
            if snap is None:
                continue
            rec = score_stock(stock, snap, strategy_cfg)
            recommendations.append(rec)
            latest_snaps[stock.symbol] = snap
            rec_by_sym[stock.symbol] = rec

        recommendations.sort(key=lambda r: r.score, reverse=True)

        executed_trades: list[dict[str, Any]] = []
        sell_evaluations: list[SellEvaluation] = []

        # 2. Evaluate held positions & execute sells first
        for sym in list(positions.keys()):
            pos = positions[sym]
            snap = latest_snaps.get(sym)
            rec = rec_by_sym.get(sym)
            if snap is None or rec is None:
                continue

            pos["last_price"] = float(snap.close)
            pos["highest_price"] = max(float(pos["highest_price"]), float(snap.high), float(snap.close))
            ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_dt)

            if ev.should_sell:
                exec_price = _apply_slippage(float(snap.close), "SELL", strategy_cfg.slippage_bps)
                qty = int(pos["quantity"])
                gross = exec_price * qty
                fee = gross * strategy_cfg.fee_rate
                net_proceeds = gross - fee
                pnl = (exec_price - float(pos["avg_cost"])) * qty - fee

                cash += net_proceeds
                realized_pnl_total += pnl
                total_fees += fee

                self.storage.record_trade(
                    mode=portfolio_id,
                    timestamp=bar_ts_iso,
                    symbol=sym,
                    name=pos["name"],
                    side="SELL",
                    quantity=qty,
                    price=round(exec_price, 2),
                    gross_amount=round(gross, 2),
                    fee=round(fee, 2),
                    realized_pnl=round(pnl, 2),
                    reason=ev.primary_reason,
                )
                self.storage.delete_position(portfolio_id, sym)
                del positions[sym]
                executed_trades.append(
                    {
                        "timestamp": bar_ts_iso,
                        "symbol": sym,
                        "name": pos["name"],
                        "side": "SELL",
                        "quantity": qty,
                        "price": round(exec_price, 2),
                        "realized_pnl": round(pnl, 2),
                        "reason": ev.primary_reason,
                    }
                )
            else:
                self.storage.upsert_position(portfolio_id, pos)
                sell_evaluations.append(ev)

        # 3. Evaluate buy candidates
        holdings_val_pre_buy = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
        total_equity_pre_buy = cash + holdings_val_pre_buy

        for rec in recommendations:
            if len(positions) >= strategy_cfg.max_positions:
                break
            if rec.symbol in positions:
                continue
            if rec.action not in ("STRONG_BUY", "BUY"):
                continue

            snap = latest_snaps[rec.symbol]
            qty = _compute_order_quantity(
                price=float(snap.close),
                cash=cash,
                total_equity=total_equity_pre_buy,
                unit_size=resolved_unit,
                max_allocation_pct=strategy_cfg.max_allocation_pct,
                fee_rate=strategy_cfg.fee_rate,
                slippage_bps=strategy_cfg.slippage_bps,
            )
            if qty <= 0:
                continue

            exec_price = _apply_slippage(float(snap.close), "BUY", strategy_cfg.slippage_bps)
            gross = exec_price * qty
            fee = gross * strategy_cfg.fee_rate
            total_cost = gross + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            total_fees += fee
            reason = f"{rec.action_label} (スコア {rec.score:.1f}): " + " / ".join(rec.reasons[:2])

            new_pos = {
                "symbol": rec.symbol,
                "name": rec.name,
                "quantity": qty,
                "avg_cost": round(exec_price, 2),
                "highest_price": round(exec_price, 2),
                "target_price": rec.target_price,
                "stop_loss_price": rec.stop_loss_price,
                "entry_time": bar_ts_iso,
                "last_price": float(snap.close),
            }
            positions[rec.symbol] = new_pos
            self.storage.upsert_position(portfolio_id, new_pos)
            self.storage.record_trade(
                mode=portfolio_id,
                timestamp=bar_ts_iso,
                symbol=rec.symbol,
                name=rec.name,
                side="BUY",
                quantity=qty,
                price=round(exec_price, 2),
                gross_amount=round(gross, 2),
                fee=round(fee, 2),
                realized_pnl=0.0,
                reason=reason,
            )
            executed_trades.append(
                {
                    "timestamp": bar_ts_iso,
                    "symbol": rec.symbol,
                    "name": rec.name,
                    "side": "BUY",
                    "quantity": qty,
                    "price": round(exec_price, 2),
                    "realized_pnl": 0.0,
                    "reason": reason,
                }
            )
            ev_new = evaluate_sell_timing(new_pos, rec, snap, strategy_cfg, bar_time=bar_dt)
            sell_evaluations.append(ev_new)

        # 4. Mark-to-market & persist equity curve point
        holdings_value = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
        total_equity = cash + holdings_value
        return_pct = ((total_equity - init_cap) / init_cap * 100.0) if init_cap > 0 else 0.0

        self.storage.update_portfolio(
            mode=portfolio_id,
            cash=round(cash, 2),
            realized_pnl=round(realized_pnl_total, 2),
            total_fees=round(total_fees, 2),
            last_bar_ts=bar_ts_iso,
        )
        self.storage.record_equity_point(
            mode=portfolio_id,
            timestamp=bar_ts_iso,
            cash=cash,
            holdings_value=holdings_value,
            total_equity=total_equity,
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
            "total_equity": round(total_equity, 1),
            "return_pct": round(return_pct, 2),
            "trades_executed": executed_trades,
            "open_positions": len(positions),
            "recommendation_count": len(recommendations),
        }

    def run_catchup_pattern(
        self,
        pattern: SimulationPattern,
        interval: str | None = None,
        lookback_days: int = 7,
        preloaded_data: dict[str, pd.DataFrame] | None = None,
        export_snapshot: bool = True,
        portfolio_id_override: str | None = None,
    ) -> dict[str, Any]:
        """Replay all unprocessed real-market bars for a specific SimulationPattern."""
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

        ind_map = self._prepare_indicator_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=max(lookback_days, 7),
            preloaded_data=preloaded_data,
        )
        if not ind_map:
            return {
                "pattern_id": pid,
                "name": pattern.name,
                "mode": pid,
                "bars_processed": 0,
                "trades_executed": 0,
                "message": "市場データを取得できませんでした。",
            }

        all_timestamps: set[pd.Timestamp] = set()
        for df in ind_map.values():
            all_timestamps.update(df.index)
        sorted_ts = sorted(all_timestamps)

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

        total_trades = 0
        last_res: dict[str, Any] = {}
        for ts in target_ts:
            last_res = self._execute_bar(pid, strategy_cfg, pattern.unit_size, ts, watchlist, ind_map)
            total_trades += len(last_res.get("trades_executed", []))

        if not target_ts and sorted_ts:
            self._refresh_signals_only(pid, strategy_cfg, sorted_ts[-1], watchlist, ind_map)

        if export_snapshot:
            self.storage.export_dashboard_snapshot(self.cfg)

        return {
            "pattern_id": pid,
            "name": pattern.name,
            "mode": pid,
            "interval": use_interval,
            "initial_capital": pattern.initial_capital,
            "unit_size": pattern.unit_size,
            "bars_processed": len(target_ts),
            "trades_executed": total_trades,
            "latest_bar": target_ts[-1].isoformat() if target_ts else last_bar_ts_str,
            "total_equity": last_res.get("total_equity"),
            "return_pct": last_res.get("return_pct"),
        }

    def run_catchup_all_patterns(
        self,
        lookback_days: int = 7,
        pattern_ids: list[str] | None = None,
        override_interval: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run catch-up across multiple patterns in parallel, batch-fetching Yahoo Finance once per interval."""
        if pattern_ids:
            targets = [p for p in self.cfg.patterns if p.id in set(pattern_ids)]
        else:
            targets = self.cfg.get_enabled_patterns()

        if not targets:
            return []

        # Group required symbols by interval so Yahoo Finance is only queried once per interval
        interval_symbols: dict[str, set[str]] = defaultdict(set)
        for p in targets:
            iv = override_interval or p.interval
            for s in self.cfg.get_watchlist(p.watchlist_key):
                interval_symbols[iv].add(s.symbol)

        fetched_by_interval: dict[str, dict[str, pd.DataFrame]] = {}
        fetch_days = max(lookback_days, 7)
        for iv, syms in interval_symbols.items():
            fetched_by_interval[iv] = self.provider.fetch_many(
                sorted(syms),
                interval=iv,
                period_days=fetch_days,
            )

        results: list[dict[str, Any]] = []
        for p in targets:
            iv = override_interval or p.interval
            res = self.run_catchup_pattern(
                pattern=p,
                interval=iv,
                lookback_days=lookback_days,
                preloaded_data=fetched_by_interval.get(iv),
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

        ind_map = self._prepare_indicator_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=7,
            preloaded_data=preloaded_data,
        )
        if not ind_map:
            return {"pattern_id": pid, "mode": pid, "bars_processed": 0, "message": "市場データを取得できませんでした。"}

        latest_ts = max(df.index[-1] for df in ind_map.values() if not df.empty)
        if last_bar_ts_str:
            prev_ts = pd.Timestamp(last_bar_ts_str)
            if prev_ts.tzinfo is None:
                prev_ts = prev_ts.tz_localize(JST)
            else:
                prev_ts = prev_ts.tz_convert(JST)
            if latest_ts <= prev_ts:
                self._refresh_signals_only(pid, strategy_cfg, latest_ts, watchlist, ind_map)
                self.storage.export_dashboard_snapshot(self.cfg)
                return {
                    "pattern_id": pid,
                    "mode": pid,
                    "bars_processed": 0,
                    "latest_bar": latest_ts.isoformat(),
                    "message": "最新バーは処理済みです（推奨シグナルのみ更新）。",
                }

        res = self._execute_bar(pid, strategy_cfg, pattern.unit_size, latest_ts, watchlist, ind_map)
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
        strategy_cfg: AppConfig,
        bar_ts: pd.Timestamp,
        watchlist: list[StockInfo],
        ind_map: dict[str, pd.DataFrame],
    ) -> None:
        positions = self.storage.get_positions(portfolio_id)
        pos_by_sym = {p["symbol"]: p for p in positions}
        recs: list[StockRecommendation] = []
        sell_evals: list[SellEvaluation] = []
        for stock in watchlist:
            ind_df = ind_map.get(stock.symbol)
            if ind_df is None or ind_df.empty:
                continue
            pos_idx = int(ind_df.index.searchsorted(bar_ts, side="right")) - 1
            if pos_idx < 1:
                continue
            snap = extract_latest_snapshot(ind_df.iloc[pos_idx - 1 : pos_idx + 1])
            if snap is None:
                continue
            rec = score_stock(stock, snap, strategy_cfg)
            recs.append(rec)
            if stock.symbol in pos_by_sym:
                pos = pos_by_sym[stock.symbol]
                pos["last_price"] = float(snap.close)
                ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_ts.to_pydatetime())
                sell_evals.append(ev)
        recs.sort(key=lambda r: r.score, reverse=True)
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
        """Run an isolated in-memory backtest for a specific SimulationPattern."""
        init_cap = float(override_capital if override_capital is not None else pattern.initial_capital)
        use_interval = override_interval or pattern.interval
        unit_size = pattern.unit_size
        watchlist = self.cfg.get_watchlist(pattern.watchlist_key)
        strategy_cfg = pattern.to_strategy_config(self.cfg.watchlists)

        fetch_days = min(58, max(days + 5, 10))
        ind_map = self._prepare_indicator_frames(
            watchlist=watchlist,
            interval=use_interval,
            period_days=fetch_days,
            preloaded_data=preloaded_data,
        )
        if not ind_map:
            return {"error": "市場データを取得できませんでした。ネットワーク接続を確認してください。"}

        all_timestamps: set[pd.Timestamp] = set()
        for df in ind_map.values():
            all_timestamps.update(df.index)
        sorted_ts = sorted(all_timestamps)
        if len(sorted_ts) < 5:
            return {"error": "バックテストに十分なバー数がありません。"}

        end_ts = sorted_ts[-1]
        start_cutoff = end_ts - timedelta(days=days)
        warmup_start = min(26, max(2, len(sorted_ts) // 5))
        sim_timestamps = [ts for ts in sorted_ts[warmup_start:] if ts >= start_cutoff]
        if not sim_timestamps:
            sim_timestamps = sorted_ts[warmup_start:]

        cash = init_cap
        realized_pnl_total = 0.0
        total_fees = 0.0
        positions: dict[str, dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []

        sym_ts_pos: dict[str, dict[pd.Timestamp, int]] = {
            sym: {ts: idx for idx, ts in enumerate(df.index)}
            for sym, df in ind_map.items()
        }
        last_known_idx: dict[str, int] = {}

        for bar_ts in sim_timestamps:
            bar_ts_iso = bar_ts.isoformat()
            bar_dt = bar_ts.to_pydatetime()
            recs: list[StockRecommendation] = []
            snaps: dict[str, Any] = {}
            rec_by_sym: dict[str, StockRecommendation] = {}

            for stock in watchlist:
                ind_df = ind_map.get(stock.symbol)
                if ind_df is None or ind_df.empty:
                    continue
                idx = sym_ts_pos[stock.symbol].get(bar_ts)
                if idx is not None:
                    last_known_idx[stock.symbol] = idx
                else:
                    idx = last_known_idx.get(stock.symbol)
                if idx is None or idx < 1:
                    continue

                snap = extract_latest_snapshot(ind_df.iloc[max(0, idx - 1) : idx + 1])
                if snap is None:
                    continue
                rec = score_stock(stock, snap, strategy_cfg)
                recs.append(rec)
                snaps[stock.symbol] = snap
                rec_by_sym[stock.symbol] = rec

            recs.sort(key=lambda r: r.score, reverse=True)

            # Evaluate sells
            for sym in list(positions.keys()):
                pos = positions[sym]
                snap = snaps.get(sym)
                rec = rec_by_sym.get(sym)
                if snap is None or rec is None:
                    continue
                pos["last_price"] = float(snap.close)
                pos["highest_price"] = max(float(pos["highest_price"]), float(snap.high), float(snap.close))
                ev = evaluate_sell_timing(pos, rec, snap, strategy_cfg, bar_time=bar_dt)
                if ev.should_sell:
                    exec_price = _apply_slippage(float(snap.close), "SELL", strategy_cfg.slippage_bps)
                    qty = int(pos["quantity"])
                    gross = exec_price * qty
                    fee = gross * strategy_cfg.fee_rate
                    pnl = (exec_price - float(pos["avg_cost"])) * qty - fee
                    cash += gross - fee
                    realized_pnl_total += pnl
                    total_fees += fee
                    trades.append(
                        {
                            "timestamp": bar_ts_iso,
                            "symbol": sym,
                            "name": pos["name"],
                            "side": "SELL",
                            "quantity": qty,
                            "price": round(exec_price, 2),
                            "fee": round(fee, 2),
                            "realized_pnl": round(pnl, 2),
                            "reason": ev.primary_reason,
                        }
                    )
                    del positions[sym]

            # Evaluate buys
            holdings_val_pre = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
            total_eq_pre = cash + holdings_val_pre
            for rec in recs:
                if len(positions) >= strategy_cfg.max_positions:
                    break
                if rec.symbol in positions or rec.action not in ("STRONG_BUY", "BUY"):
                    continue
                snap = snaps[rec.symbol]
                qty = _compute_order_quantity(
                    price=float(snap.close),
                    cash=cash,
                    total_equity=total_eq_pre,
                    unit_size=unit_size,
                    max_allocation_pct=strategy_cfg.max_allocation_pct,
                    fee_rate=strategy_cfg.fee_rate,
                    slippage_bps=strategy_cfg.slippage_bps,
                )
                if qty <= 0:
                    continue
                exec_price = _apply_slippage(float(snap.close), "BUY", strategy_cfg.slippage_bps)
                gross = exec_price * qty
                fee = gross * strategy_cfg.fee_rate
                if gross + fee > cash:
                    continue
                cash -= gross + fee
                total_fees += fee
                positions[rec.symbol] = {
                    "symbol": rec.symbol,
                    "name": rec.name,
                    "quantity": qty,
                    "avg_cost": round(exec_price, 2),
                    "highest_price": round(exec_price, 2),
                    "target_price": rec.target_price,
                    "stop_loss_price": rec.stop_loss_price,
                    "entry_time": bar_ts_iso,
                    "last_price": float(snap.close),
                }
                trades.append(
                    {
                        "timestamp": bar_ts_iso,
                        "symbol": rec.symbol,
                        "name": rec.name,
                        "side": "BUY",
                        "quantity": qty,
                        "price": round(exec_price, 2),
                        "fee": round(fee, 2),
                        "realized_pnl": 0.0,
                        "reason": f"{rec.action_label} (スコア {rec.score:.1f}): " + " / ".join(rec.reasons[:2]),
                    }
                )

            holdings_val = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
            total_eq = cash + holdings_val
            ret_pct = (total_eq - init_cap) / init_cap * 100.0
            equity_curve.append(
                {
                    "timestamp": bar_ts_iso,
                    "total_equity": round(total_eq, 1),
                    "cash": round(cash, 1),
                    "holdings_value": round(holdings_val, 1),
                    "return_pct": round(ret_pct, 2),
                    "open_positions": len(positions),
                }
            )

        final_holdings = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions.values())
        final_equity = cash + final_holdings
        unrealized_pnl = sum(
            (float(p["last_price"]) - float(p["avg_cost"])) * int(p["quantity"])
            for p in positions.values()
        )
        net_pnl = final_equity - init_cap
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
            bm = benchmark_df
            if bm is None and preloaded_data is None:
                bm = self.provider.fetch_symbol("^N225", interval=use_interval, period_days=fetch_days)
            if bm is not None and not bm.empty and sim_timestamps:
                bm_sliced = bm.loc[sim_timestamps[0] : sim_timestamps[-1]]
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
            "mode": pattern.id,
            "mode_label": pattern.name,
            "unit_size": unit_size,
            "interval": use_interval,
            "days": days,
            "bars_simulated": len(sim_timestamps),
            "start_time": sim_timestamps[0].isoformat(),
            "end_time": sim_timestamps[-1].isoformat(),
            "symbols_loaded": len(ind_map),
            "initial_capital": round(init_cap, 1),
            "final_equity": round(final_equity, 1),
            "cash": round(cash, 1),
            "holdings_value": round(final_holdings, 1),
            "net_pnl": round(net_pnl, 1),
            "return_pct": round(return_pct, 2),
            "realized_pnl": round(realized_pnl_total, 1),
            "unrealized_pnl": round(unrealized_pnl, 1),
            "total_fees": round(total_fees, 1),
            "total_trades": len(trades),
            "closed_trades": len(sell_trades),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "benchmark_nikkei225_pct": benchmark_return_pct,
            "open_positions": list(positions.values()),
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
        """Run parallel real-data backtests across multiple patterns and rank them side-by-side."""
        targets = patterns if patterns is not None else self.cfg.get_enabled_patterns()
        if not targets:
            return {"error": "有効なシミュレーションパターンがありません。"}

        fetch_days = min(58, max(days + 5, 10))
        interval_symbols: dict[str, set[str]] = defaultdict(set)
        for p in targets:
            iv = override_interval or p.interval
            for s in self.cfg.get_watchlist(p.watchlist_key):
                interval_symbols[iv].add(s.symbol)

        fetched_by_interval: dict[str, dict[str, pd.DataFrame]] = {}
        benchmarks_by_interval: dict[str, pd.DataFrame] = {}
        for iv, syms in interval_symbols.items():
            fetched_by_interval[iv] = self.provider.fetch_many(
                sorted(syms),
                interval=iv,
                period_days=fetch_days,
            )
            try:
                benchmarks_by_interval[iv] = self.provider.fetch_symbol("^N225", interval=iv, period_days=fetch_days)
            except Exception:
                benchmarks_by_interval[iv] = pd.DataFrame()

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
