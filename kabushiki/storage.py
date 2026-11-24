"""SQLite persistence with pending pre-orders, commissions, 20% tax tracking, and snapshot export."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

from .config import (
    AppConfig,
    SimulationPattern,
    DEFAULT_DB_PATH,
    DEFAULT_SNAPSHOT_PATH,
    WEB_SNAPSHOT_PATH,
    MODE_LABELS,
    MODE_UNIT_SIZE,
)
from .market_data import JST, now_jst, tse_market_status


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS portfolios (
    mode TEXT PRIMARY KEY,
    unit_size INTEGER NOT NULL,
    initial_capital REAL NOT NULL,
    cash REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    total_fees REAL NOT NULL DEFAULT 0.0,
    total_tax REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_bar_ts TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost REAL NOT NULL,
    buy_fee REAL NOT NULL DEFAULT 0.0,
    highest_price REAL NOT NULL,
    target_price REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    last_price REAL NOT NULL,
    PRIMARY KEY (mode, symbol)
);

CREATE TABLE IF NOT EXISTS pending_orders (
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    side TEXT NOT NULL,
    signal_time TEXT NOT NULL,
    signal_price REAL NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    target_price REAL NOT NULL DEFAULT 0.0,
    stop_loss_price REAL NOT NULL DEFAULT 0.0,
    reason TEXT NOT NULL,
    PRIMARY KEY (mode, symbol, side)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    signal_time TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    gross_amount REAL NOT NULL,
    fee REAL NOT NULL,
    tax REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_mode_ts ON trades (mode, timestamp DESC);

CREATE TABLE IF NOT EXISTS equity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cash REAL NOT NULL,
    holdings_value REAL NOT NULL,
    total_equity REAL NOT NULL,
    return_pct REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    UNIQUE (mode, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_equity_mode_ts ON equity_history (mode, timestamp ASC);

CREATE TABLE IF NOT EXISTS latest_snapshots (
    mode TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    recommendations_json TEXT NOT NULL,
    sell_evaluations_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_bars (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, interval, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ml_bars_iv_sym_ts ON ml_bars (interval, symbol, timestamp ASC);

CREATE TABLE IF NOT EXISTS ml_model_status (
    interval TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS symbol_cooldowns (
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    until_ts TEXT NOT NULL,
    PRIMARY KEY (mode, symbol)
);
"""


class Storage:
    def __init__(self, db_path: Path | None = None, snapshot_paths: list[Path] | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if snapshot_paths is not None:
            self.default_snapshot_paths = snapshot_paths
        elif db_path is not None and db_path.resolve() != DEFAULT_DB_PATH.resolve():
            self.default_snapshot_paths = [self.db_path.parent / "dashboard_snapshot.json"]
        else:
            self.default_snapshot_paths = [DEFAULT_SNAPSHOT_PATH, WEB_SNAPSHOT_PATH]
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # Safe schema migrations for existing DB files
            self._ensure_column(conn, "portfolios", "total_tax", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "positions", "buy_fee", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "positions", "bars_held", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "trades", "signal_time", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "trades", "tax", "REAL NOT NULL DEFAULT 0.0")

    def ensure_portfolio(
        self,
        mode: str,
        initial_capital: float = 100000.0,
        unit_size: int | None = None,
    ) -> dict[str, Any]:
        resolved_unit = int(unit_size) if unit_size is not None else MODE_UNIT_SIZE.get(mode, 1)
        now = now_jst().isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM portfolios WHERE mode = ?", (mode,)).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO portfolios (mode, unit_size, initial_capital, cash, realized_pnl, total_fees, total_tax, created_at, updated_at, last_bar_ts)
                    VALUES (?, ?, ?, ?, 0.0, 0.0, 0.0, ?, ?, NULL)
                    """,
                    (mode, resolved_unit, float(initial_capital), float(initial_capital), now, now),
                )
                row = conn.execute("SELECT * FROM portfolios WHERE mode = ?", (mode,)).fetchone()
            else:
                if unit_size is not None and int(row["unit_size"]) != resolved_unit:
                    conn.execute(
                        "UPDATE portfolios SET unit_size = ?, updated_at = ? WHERE mode = ?",
                        (resolved_unit, now, mode),
                    )
                    row = conn.execute("SELECT * FROM portfolios WHERE mode = ?", (mode,)).fetchone()
            return dict(row)

    def reset_portfolio(
        self,
        mode: str,
        initial_capital: float = 100000.0,
        unit_size: int | None = None,
    ) -> dict[str, Any]:
        resolved_unit = int(unit_size) if unit_size is not None else MODE_UNIT_SIZE.get(mode, 1)
        now = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM pending_orders WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM symbol_cooldowns WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM trades WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM equity_history WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM latest_snapshots WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM portfolios WHERE mode = ?", (mode,))
            conn.execute(
                """
                INSERT INTO portfolios (mode, unit_size, initial_capital, cash, realized_pnl, total_fees, total_tax, created_at, updated_at, last_bar_ts)
                VALUES (?, ?, ?, ?, 0.0, 0.0, 0.0, ?, ?, NULL)
                """,
                (mode, resolved_unit, float(initial_capital), float(initial_capital), now, now),
            )
            row = conn.execute("SELECT * FROM portfolios WHERE mode = ?", (mode,)).fetchone()
            return dict(row)

    def delete_portfolio(self, mode: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM pending_orders WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM symbol_cooldowns WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM trades WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM equity_history WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM latest_snapshots WHERE mode = ?", (mode,))
            conn.execute("DELETE FROM portfolios WHERE mode = ?", (mode,))

    def get_portfolio(
        self,
        mode: str,
        initial_capital: float = 100000.0,
        unit_size: int | None = None,
    ) -> dict[str, Any]:
        return self.ensure_portfolio(mode, initial_capital, unit_size=unit_size)

    def update_portfolio(
        self,
        mode: str,
        cash: float,
        realized_pnl: float,
        total_fees: float,
        total_tax: float = 0.0,
        last_bar_ts: str | None = None,
    ) -> None:
        now = now_jst().isoformat()
        with self._connect() as conn:
            if last_bar_ts is not None:
                conn.execute(
                    """
                    UPDATE portfolios
                    SET cash = ?, realized_pnl = ?, total_fees = ?, total_tax = ?, updated_at = ?, last_bar_ts = ?
                    WHERE mode = ?
                    """,
                    (cash, realized_pnl, total_fees, total_tax, now, last_bar_ts, mode),
                )
            else:
                conn.execute(
                    """
                    UPDATE portfolios
                    SET cash = ?, realized_pnl = ?, total_fees = ?, total_tax = ?, updated_at = ?
                    WHERE mode = ?
                    """,
                    (cash, realized_pnl, total_fees, total_tax, now, mode),
                )

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE mode = ? ORDER BY entry_time ASC",
                (mode,),
            ).fetchall()
            return [dict(r) for r in rows]

    def upsert_position(self, mode: str, pos: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO positions (mode, symbol, name, quantity, avg_cost, buy_fee, highest_price, target_price, stop_loss_price, entry_time, last_price, bars_held)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mode, symbol) DO UPDATE SET
                    name = excluded.name,
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    buy_fee = excluded.buy_fee,
                    highest_price = excluded.highest_price,
                    target_price = excluded.target_price,
                    stop_loss_price = excluded.stop_loss_price,
                    entry_time = excluded.entry_time,
                    last_price = excluded.last_price,
                    bars_held = excluded.bars_held
                """,
                (
                    mode,
                    pos["symbol"],
                    pos["name"],
                    int(pos["quantity"]),
                    float(pos["avg_cost"]),
                    float(pos.get("buy_fee", 0.0)),
                    float(pos["highest_price"]),
                    float(pos["target_price"]),
                    float(pos["stop_loss_price"]),
                    str(pos["entry_time"]),
                    float(pos["last_price"]),
                    int(pos.get("bars_held", 0)),
                ),
            )

    def delete_position(self, mode: str, symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE mode = ? AND symbol = ?", (mode, symbol))

    def get_pending_orders(self, mode: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_orders WHERE mode = ? ORDER BY side DESC, score DESC",
                (mode,),
            ).fetchall()
            return [dict(r) for r in rows]

    def replace_pending_orders(self, mode: str, orders: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_orders WHERE mode = ?", (mode,))
            for o in orders:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pending_orders (mode, symbol, name, side, signal_time, signal_price, score, target_price, stop_loss_price, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mode,
                        o["symbol"],
                        o["name"],
                        o["side"],
                        str(o["signal_time"]),
                        float(o["signal_price"]),
                        float(o.get("score", 0.0)),
                        float(o.get("target_price", 0.0)),
                        float(o.get("stop_loss_price", 0.0)),
                        str(o["reason"]),
                    ),
                )

    def record_trade(
        self,
        mode: str,
        timestamp: str,
        symbol: str,
        name: str,
        side: str,
        quantity: int,
        price: float,
        gross_amount: float,
        fee: float,
        realized_pnl: float,
        reason: str,
        signal_time: str = "",
        tax: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (mode, timestamp, signal_time, symbol, name, side, quantity, price, gross_amount, fee, tax, realized_pnl, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    timestamp,
                    signal_time,
                    symbol,
                    name,
                    side,
                    int(quantity),
                    float(price),
                    float(gross_amount),
                    float(fee),
                    float(tax),
                    float(realized_pnl),
                    reason,
                ),
            )

    def get_trades(self, mode: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE mode = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
                (mode, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_equity_point(
        self,
        mode: str,
        timestamp: str,
        cash: float,
        holdings_value: float,
        total_equity: float,
        return_pct: float,
        open_positions: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO equity_history (mode, timestamp, cash, holdings_value, total_equity, return_pct, open_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mode, timestamp) DO UPDATE SET
                    cash = excluded.cash,
                    holdings_value = excluded.holdings_value,
                    total_equity = excluded.total_equity,
                    return_pct = excluded.return_pct,
                    open_positions = excluded.open_positions
                """,
                (
                    mode,
                    timestamp,
                    round(cash, 2),
                    round(holdings_value, 2),
                    round(total_equity, 2),
                    round(return_pct, 4),
                    int(open_positions),
                ),
            )

    def get_equity_history(self, mode: str, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, cash, holdings_value, total_equity, return_pct, open_positions
                FROM (
                    SELECT * FROM equity_history WHERE mode = ? ORDER BY timestamp DESC LIMIT ?
                )
                ORDER BY timestamp ASC
                """,
                (mode, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_latest_signals(
        self,
        mode: str,
        recommendations: list[dict[str, Any]],
        sell_evaluations: list[dict[str, Any]],
        updated_at: str | None = None,
    ) -> None:
        ts = updated_at or now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO latest_snapshots (mode, updated_at, recommendations_json, sell_evaluations_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(mode) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    recommendations_json = excluded.recommendations_json,
                    sell_evaluations_json = excluded.sell_evaluations_json
                """,
                (
                    mode,
                    ts,
                    json.dumps(recommendations, ensure_ascii=False),
                    json.dumps(sell_evaluations, ensure_ascii=False),
                ),
            )

    def get_latest_signals(self, mode: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM latest_snapshots WHERE mode = ?",
                (mode,),
            ).fetchone()
            if row is None:
                return {"updated_at": None, "recommendations": [], "sell_evaluations": []}
            return {
                "updated_at": row["updated_at"],
                "recommendations": json.loads(row["recommendations_json"]),
                "sell_evaluations": json.loads(row["sell_evaluations_json"]),
            }

    # --- Cooldown Persistence ---
    def get_cooldowns(self, mode: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, until_ts FROM symbol_cooldowns WHERE mode = ?",
                (mode,),
            ).fetchall()
            return {str(r["symbol"]): str(r["until_ts"]) for r in rows}

    def set_cooldown(self, mode: str, symbol: str, until_ts: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_cooldowns (mode, symbol, until_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(mode, symbol) DO UPDATE SET until_ts = excluded.until_ts
                """,
                (mode, symbol, until_ts),
            )

    # --- Persistent ML Training Data Store (accumulates beyond Yahoo Finance 58-day window!) ---
    def save_ml_bars(
        self,
        interval: str,
        symbol_frames: dict[str, pd.DataFrame],
        max_bars_per_symbol: int = 3000,
    ) -> int:
        """Upsert real OHLCV bars into SQLite so ML training data grows continuously over months."""
        if not symbol_frames:
            return 0
        total_upserted = 0
        with self._connect() as conn:
            for sym, df in symbol_frames.items():
                if df is None or df.empty:
                    continue
                records = []
                for ts, row in df.iterrows():
                    ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    records.append(
                        (
                            sym,
                            interval,
                            ts_str,
                            float(row["Open"]),
                            float(row["High"]),
                            float(row["Low"]),
                            float(row["Close"]),
                            float(row["Volume"]),
                        )
                    )
                if not records:
                    continue
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO ml_bars (symbol, interval, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                total_upserted += len(records)
                # Keep database compact (~1+ year of 30m bars per symbol)
                conn.execute(
                    """
                    DELETE FROM ml_bars
                    WHERE symbol = ? AND interval = ? AND timestamp NOT IN (
                        SELECT timestamp FROM ml_bars
                        WHERE symbol = ? AND interval = ?
                        ORDER BY timestamp DESC LIMIT ?
                    )
                    """,
                    (sym, interval, sym, interval, int(max_bars_per_symbol)),
                )
        return total_upserted

    def load_ml_bars(self, interval: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Load accumulated OHLCV history from SQLite normalized to JST DatetimeIndex."""
        result: dict[str, pd.DataFrame] = {}
        with self._connect() as conn:
            if symbols:
                placeholders = ",".join("?" for _ in symbols)
                rows = conn.execute(
                    f"""
                    SELECT symbol, timestamp, open, high, low, close, volume
                    FROM ml_bars
                    WHERE interval = ? AND symbol IN ({placeholders})
                    ORDER BY symbol ASC, timestamp ASC
                    """,
                    [interval, *symbols],
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT symbol, timestamp, open, high, low, close, volume
                    FROM ml_bars
                    WHERE interval = ?
                    ORDER BY symbol ASC, timestamp ASC
                    """,
                    (interval,),
                ).fetchall()

        by_sym: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_sym.setdefault(r["symbol"], []).append(dict(r))

        for sym, items in by_sym.items():
            idx = pd.to_datetime([it["timestamp"] for it in items], utc=True).tz_convert(JST)
            df = pd.DataFrame(
                {
                    "Open": [it["open"] for it in items],
                    "High": [it["high"] for it in items],
                    "Low": [it["low"] for it in items],
                    "Close": [it["close"] for it in items],
                    "Volume": [it["volume"] for it in items],
                },
                index=idx,
            )
            result[sym] = df[~df.index.duplicated(keep="last")].sort_index()
        return result

    def save_ml_diagnostics(self, interval: str, diagnostics: dict[str, Any]) -> None:
        now = now_jst().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ml_model_status (interval, updated_at, diagnostics_json)
                VALUES (?, ?, ?)
                ON CONFLICT(interval) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    diagnostics_json = excluded.diagnostics_json
                """,
                (interval, now, json.dumps(diagnostics, ensure_ascii=False)),
            )

    def get_ml_status(self) -> dict[str, Any]:
        """Return accumulated ML dataset statistics and latest Walk-Forward OOS diagnostics."""
        with self._connect() as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS cnt, COUNT(DISTINCT symbol) AS syms, MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM ml_bars"
            ).fetchone()
            iv_rows = conn.execute(
                """
                SELECT interval, COUNT(*) AS bars, COUNT(DISTINCT symbol) AS symbols,
                       MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
                FROM ml_bars
                GROUP BY interval
                ORDER BY interval ASC
                """
            ).fetchall()
            diag_rows = conn.execute("SELECT interval, updated_at, diagnostics_json FROM ml_model_status").fetchall()

        models: dict[str, Any] = {}
        for r in diag_rows:
            d = json.loads(r["diagnostics_json"])
            d["updated_at"] = r["updated_at"]
            models[str(r["interval"])] = d

        return {
            "total_bars": int(total_row["cnt"] or 0),
            "unique_symbols": int(total_row["syms"] or 0),
            "earliest_bar": total_row["min_ts"],
            "latest_bar": total_row["max_ts"],
            "by_interval": [dict(r) for r in iv_rows],
            "models": models,
        }

    def build_pattern_summary(self, pattern: SimulationPattern) -> dict[str, Any]:
        summary = self.build_mode_summary(
            mode=pattern.id,
            initial_capital=pattern.initial_capital,
            unit_size=pattern.unit_size,
            label=pattern.name,
            tax_rate=pattern.tax_rate,
            fee_rate=pattern.fee_rate,
        )
        summary["pattern_id"] = pattern.id
        summary["pattern"] = pattern.to_dict()
        summary["enabled"] = pattern.enabled
        summary["interval"] = pattern.interval
        summary["watchlist_key"] = pattern.watchlist_key
        summary["description"] = pattern.description
        return summary

    def build_mode_summary(
        self,
        mode: str,
        initial_capital: float = 100000.0,
        unit_size: int | None = None,
        label: str | None = None,
        tax_rate: float = 0.20315,
        fee_rate: float = 0.001,
    ) -> dict[str, Any]:
        portfolio = self.get_portfolio(mode, initial_capital, unit_size=unit_size)
        positions = self.get_positions(mode)
        pending_orders = self.get_pending_orders(mode)
        trades = self.get_trades(mode, limit=150)
        equity_history = self.get_equity_history(mode, limit=600)
        signals = self.get_latest_signals(mode)

        holdings_value = sum(float(p["last_price"]) * int(p["quantity"]) for p in positions)
        unrealized_pnl = sum(
            (float(p["last_price"]) - float(p["avg_cost"])) * int(p["quantity"])
            - float(p.get("buy_fee", 0.0))
            - float(p["last_price"]) * int(p["quantity"]) * fee_rate
            for p in positions
        )
        # Cash already reflects fees paid and realized tax withheld/refunded
        total_equity = float(portfolio["cash"]) + holdings_value
        total_tax_paid = float(portfolio.get("total_tax", 0.0))
        realized_pnl = float(portfolio["realized_pnl"])

        # Estimated additional tax if all current holdings were liquidated now
        combined_taxable = realized_pnl + unrealized_pnl
        estimated_total_tax = max(0.0, combined_taxable * tax_rate)
        estimated_unrealized_tax = max(0.0, estimated_total_tax - total_tax_paid)
        # Also deduct estimated sell fee on current holdings for strict net liquidation value
        est_exit_fees = holdings_value * fee_rate
        after_tax_equity = total_equity - est_exit_fees - estimated_unrealized_tax

        init_cap = float(portfolio["initial_capital"])
        net_pnl = after_tax_equity - init_cap
        return_pct = (net_pnl / init_cap * 100.0) if init_cap > 0 else 0.0
        gross_pnl_before_tax = (total_equity + total_tax_paid) - init_cap

        sell_trades = [t for t in trades if t["side"] == "SELL"]
        wins = [t for t in sell_trades if float(t["realized_pnl"]) > 0]
        losses = [t for t in sell_trades if float(t["realized_pnl"]) < 0]
        win_rate = (len(wins) / len(sell_trades) * 100.0) if sell_trades else 0.0
        gross_profit = sum(float(t["realized_pnl"]) for t in wins)
        gross_loss = abs(sum(float(t["realized_pnl"]) for t in losses))
        profit_factor = (
            round(gross_profit / gross_loss, 2)
            if gross_loss > 0
            else (round(gross_profit, 2) if gross_profit > 0 else 0.0)
        )

        peak = init_cap
        max_dd_pct = 0.0
        for pt in equity_history:
            eq = float(pt["total_equity"])
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        return {
            "mode": mode,
            "mode_label": label or MODE_LABELS.get(mode, mode),
            "unit_size": int(portfolio["unit_size"]),
            "initial_capital": round(init_cap, 1),
            "cash": round(float(portfolio["cash"]), 1),
            "holdings_value": round(holdings_value, 1),
            "total_equity": round(after_tax_equity, 1),
            "pre_tax_equity": round(total_equity + total_tax_paid, 1),
            "net_pnl": round(net_pnl, 1),
            "gross_pnl_before_tax": round(gross_pnl_before_tax, 1),
            "return_pct": round(return_pct, 2),
            "realized_pnl": round(realized_pnl, 1),
            "unrealized_pnl": round(unrealized_pnl, 1),
            "total_fees": round(float(portfolio["total_fees"]), 1),
            "total_tax": round(total_tax_paid + estimated_unrealized_tax, 1),
            "realized_tax_paid": round(total_tax_paid, 1),
            "tax_rate": tax_rate,
            "fee_rate": fee_rate,
            "trade_count": len(trades),
            "closed_trades": len(sell_trades),
            "win_rate": round(win_rate, 1),
            "profit_factor": profit_factor,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "last_bar_ts": portfolio["last_bar_ts"],
            "updated_at": signals["updated_at"] or portfolio["updated_at"],
            "positions": positions,
            "pending_orders": pending_orders,
            "sell_evaluations": signals["sell_evaluations"],
            "recommendations": signals["recommendations"],
            "trades": trades,
            "equity_history": equity_history,
        }

    def export_dashboard_snapshot(
        self,
        cfg: AppConfig,
        output_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Export full multi-pattern snapshot JSON for live UI & GitHub Pages."""
        patterns_by_id: dict[str, Any] = {}
        pattern_order: list[str] = []
        for pattern in cfg.patterns:
            patterns_by_id[pattern.id] = self.build_pattern_summary(pattern)
            pattern_order.append(pattern.id)

        first_lot1 = next((p.id for p in cfg.patterns if p.unit_size == 1), pattern_order[0] if pattern_order else None)
        first_lot100 = next((p.id for p in cfg.patterns if p.unit_size >= 100), first_lot1)
        modes_compat: dict[str, Any] = {}
        if first_lot1 and first_lot1 in patterns_by_id:
            modes_compat["lot1"] = patterns_by_id[first_lot1]
        if first_lot100 and first_lot100 in patterns_by_id:
            modes_compat["lot100"] = patterns_by_id[first_lot100]

        snapshot = {
            "generated_at": now_jst().isoformat(),
            "market_status": tse_market_status(),
            "execution_policy": "事前予測(t足確定) → 次足(t+1)始値約定 / 手数料＋譲渡益課税(約20.315%)控除 / Walk-Forward ML継続学習",
            "ml_status": self.get_ml_status(),
            "config": {
                "initial_capital": cfg.initial_capital,
                "interval": cfg.interval,
                "default_mode": cfg.default_mode,
                "max_positions": cfg.max_positions,
                "max_allocation_pct": cfg.max_allocation_pct,
                "fee_rate": cfg.fee_rate,
                "tax_rate": cfg.tax_rate,
                "slippage_bps": cfg.slippage_bps,
                "take_profit_pct": cfg.take_profit_pct,
                "stop_loss_pct": cfg.stop_loss_pct,
                "trailing_activation_pct": cfg.trailing_activation_pct,
                "trailing_stop_pct": cfg.trailing_stop_pct,
                "close_before_eod": cfg.close_before_eod,
            },
            "pattern_order": pattern_order,
            "patterns": patterns_by_id,
            "modes": modes_compat,
        }
        paths = output_paths if output_paths is not None else self.default_snapshot_paths
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
        return snapshot
