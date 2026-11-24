"""Unit tests for the 100k JPY simulator, PC-OFF catch-up, and backtester."""

from datetime import datetime, timedelta
from pathlib import Path
import math
import pandas as pd

from kabushiki.config import AppConfig
from kabushiki.market_data import JST
from kabushiki.simulator import TradingSimulator
from kabushiki.storage import Storage


def _synthetic_wave_bars(base_price: float, n_bars: int = 80, phase: float = 0.0) -> pd.DataFrame:
    """Create deterministic oscillating intraday bars that trigger both buys and sells."""
    start = datetime(2026, 11, 16, 9, 0, tzinfo=JST)
    rows = []
    idx = []
    for i in range(n_bars):
        # Smooth upward-drifting wave with clear momentum cycles
        cycle = math.sin((i + phase) * 0.28)
        drift = i * 0.0008
        close = base_price * (1.0 + drift + 0.035 * cycle)
        prev = base_price * (1.0 + max(0, i - 1) * 0.0008 + 0.035 * math.sin((max(0, i - 1) + phase) * 0.28))
        open_p = prev
        high = max(open_p, close) * 1.003
        low = min(open_p, close) * 0.997
        vol = 20000.0 if close > open_p else 10000.0
        rows.append({"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol})
        idx.append(start + timedelta(minutes=15 * i))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def _make_test_config() -> AppConfig:
    return AppConfig(
        initial_capital=100000.0,
        interval="15m",
        max_positions=2,
        max_allocation_pct=0.45,
        fee_rate=0.0,
        slippage_bps=5.0,
        watchlists={
            "lot1": [
                {"symbol": "7203.T", "name": "トヨタ自動車", "sector": "輸送用機器"},
                {"symbol": "6758.T", "name": "ソニーグループ", "sector": "電気機器"},
            ],
            "lot100": [
                {"symbol": "9432.T", "name": "NTT", "sector": "情報・通信"},
                {"symbol": "4689.T", "name": "LINEヤフー", "sector": "情報・通信"},
            ],
        },
    )


def test_backtest_lot1_and_lot100_respect_unit_sizes(tmp_path: Path):
    cfg = _make_test_config()
    storage = Storage(db_path=tmp_path / "sim.db")
    sim = TradingSimulator(cfg, storage)

    data_lot1 = {
        "7203.T": _synthetic_wave_bars(2500.0, n_bars=90, phase=0.0),
        "6758.T": _synthetic_wave_bars(3200.0, n_bars=90, phase=4.0),
    }
    res1 = sim.run_backtest(mode="lot1", interval="15m", days=10, initial_capital=100000.0, preloaded_data=data_lot1)
    assert "error" not in res1
    assert res1["initial_capital"] == 100000.0
    assert res1["bars_simulated"] > 20
    assert res1["total_trades"] > 0
    assert len(res1["equity_curve"]) == res1["bars_simulated"]

    # 100-share mode with sub-1000 JPY stocks
    data_lot100 = {
        "9432.T": _synthetic_wave_bars(160.0, n_bars=90, phase=0.0),
        "4689.T": _synthetic_wave_bars(420.0, n_bars=90, phase=3.0),
    }
    res100 = sim.run_backtest(mode="lot100", interval="15m", days=10, initial_capital=100000.0, preloaded_data=data_lot100)
    assert "error" not in res100
    assert res100["unit_size"] == 100
    assert res100["total_trades"] > 0
    for tr in res100["trades"]:
        assert tr["quantity"] % 100 == 0


def test_pc_off_catchup_replays_only_missed_bars(tmp_path: Path):
    cfg = _make_test_config()
    storage = Storage(db_path=tmp_path / "catchup.db")
    sim = TradingSimulator(cfg, storage)

    full_data = {
        "7203.T": _synthetic_wave_bars(2500.0, n_bars=60, phase=0.0),
        "6758.T": _synthetic_wave_bars(3200.0, n_bars=60, phase=3.0),
    }
    first_half = {k: v.iloc[:40] for k, v in full_data.items()}

    # First run while PC is ON
    r1 = sim.run_catchup(mode="lot1", interval="15m", preloaded_data=first_half)
    assert r1["bars_processed"] >= 1
    first_count = len(storage.get_equity_history("lot1"))

    # PC was OFF for 20 bars (5 hours). Next catchup must replay exact 20 missed bars.
    r2 = sim.run_catchup(mode="lot1", interval="15m", preloaded_data=full_data)
    assert r2["bars_processed"] == 20
    second_count = len(storage.get_equity_history("lot1"))
    assert second_count == first_count + 20

    # Re-running catchup with same data should process 0 bars (idempotent!)
    r3 = sim.run_catchup(mode="lot1", interval="15m", preloaded_data=full_data)
    assert r3["bars_processed"] == 0


def test_parallel_compare_backtest_with_custom_capitals(tmp_path: Path):
    from kabushiki.config import SimulationPattern

    cfg = _make_test_config()
    cfg.patterns = [
        SimulationPattern(
            id="p_100k_15m",
            name="10万円・15分足",
            initial_capital=100000.0,
            unit_size=1,
            watchlist_key="lot1",
            interval="15m",
            buy_score_threshold=60.0,
        ),
        SimulationPattern(
            id="p_500k_15m",
            name="50万円・15分足・慎重",
            initial_capital=500000.0,
            unit_size=1,
            watchlist_key="lot1",
            interval="15m",
            buy_score_threshold=70.0,
        ),
    ]
    storage = Storage(db_path=tmp_path / "parallel.db")

    class _Provider:
        def fetch_symbol(self, symbol, interval="15m", period_days=10):
            return _synthetic_wave_bars(2500.0, n_bars=80, phase=1.0)

        def fetch_many(self, symbols, interval="15m", period_days=10):
            return {s: _synthetic_wave_bars(2500.0, n_bars=80, phase=float(i * 3)) for i, s in enumerate(symbols)}

    sim = TradingSimulator(cfg, storage, provider=_Provider())
    res = sim.run_backtest_compare(days=7)
    assert res["compare_mode"] is True
    assert res["pattern_count"] == 2
    caps = {r["pattern_id"]: r["initial_capital"] for r in res["results"]}
    assert caps["p_100k_15m"] == 100000.0
    assert caps["p_500k_15m"] == 500000.0

