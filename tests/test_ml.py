"""Unit tests for Walk-Forward ML engine, persistent ML bar accumulation, and mean-reversion scoring."""

from pathlib import Path
from kabushiki.advisor import score_stock_mean_reversion
from kabushiki.config import AppConfig, SimulationPattern, StockInfo
from kabushiki.indicators import compute_indicators, extract_latest_snapshot
from kabushiki.ml_engine import (
    WalkForwardRidgeModel,
    attach_ml_features,
    build_training_panel,
    evaluate_walkforward_oos,
)
from kabushiki.simulator import TradingSimulator
from kabushiki.storage import Storage
from tests.test_simulator import _synthetic_wave_bars


def test_ml_bar_persistence_and_walkforward_model(tmp_path: Path):
    storage = Storage(db_path=tmp_path / "ml.db")
    bars_a = _synthetic_wave_bars(2000.0, n_bars=120, phase=0.0)
    bars_b = _synthetic_wave_bars(3000.0, n_bars=120, phase=2.5)
    bm = _synthetic_wave_bars(38000.0, n_bars=120, phase=1.0)

    # 1. Save bars into ml_bars and reload (verifies deduplication & persistence)
    n_saved = storage.save_ml_bars("30m", {"7203.T": bars_a, "6758.T": bars_b, "^N225": bm})
    assert n_saved == 360
    # Re-saving overlapping bars must not duplicate
    storage.save_ml_bars("30m", {"7203.T": bars_a.iloc[-40:]})
    loaded = storage.load_ml_bars("30m", ["7203.T", "6758.T", "^N225"])
    assert len(loaded["7203.T"]) == 120
    assert len(loaded["^N225"]) == 120

    stats = storage.get_ml_status()
    assert stats["total_bars"] == 360
    assert stats["unique_symbols"] == 3

    # 2. Build causal features & train Walk-Forward Ridge strictly before cutoff
    bm_ind = compute_indicators(loaded["^N225"])
    feat_map = {
        sym: attach_ml_features(compute_indicators(loaded[sym]), bm_ind, horizon_bars=8)
        for sym in ("7203.T", "6758.T")
    }
    panel = build_training_panel(feat_map, warmup_bars=25)
    assert len(panel) > 140

    model = WalkForwardRidgeModel(horizon_bars=8, min_samples=60)
    cutoff = sorted(panel.index.unique())[-15]
    assert model.fit_from_panel(panel, cutoff) is True
    assert model.is_fitted is True
    assert len(model.top_feature_weights(5)) == 5

    pred = model.predict_row(feat_map["7203.T"].iloc[-1])
    assert pred is not None
    assert -0.5 < pred < 0.5

    diag = evaluate_walkforward_oos(panel, interval="30m", horizon_bars=8)
    assert diag.train_samples >= 60


def test_ml_walkforward_and_mean_reversion_backtest_zero_lookahead(tmp_path: Path):
    cfg = AppConfig(
        watchlists={
            "lot1": [
                {"symbol": "7203.T", "name": "トヨタ", "sector": "輸送用機器"},
                {"symbol": "6758.T", "name": "ソニー", "sector": "電気機器"},
            ]
        },
        patterns=[
            SimulationPattern(
                id="p_ml_test",
                name="MLテスト",
                strategy_type="ml_walkforward",
                initial_capital=100000.0,
                interval="30m",
                ml_min_pred_return=0.002,
                ml_horizon_bars=6,
                max_hold_bars=10,
                cooldown_bars=3,
            ),
            SimulationPattern(
                id="p_rev_test",
                name="押し目テスト",
                strategy_type="mean_reversion",
                initial_capital=100000.0,
                interval="30m",
                buy_score_threshold=60.0,
                cooldown_bars=3,
            ),
        ],
    )
    storage = Storage(db_path=tmp_path / "ml_sim.db")
    sim = TradingSimulator(cfg, storage)
    preloaded = {
        "7203.T": _synthetic_wave_bars(2200.0, n_bars=150, phase=0.0),
        "6758.T": _synthetic_wave_bars(3100.0, n_bars=150, phase=3.0),
    }
    bm = _synthetic_wave_bars(38000.0, n_bars=150, phase=1.0)

    res_ml = sim.run_backtest_pattern(cfg.patterns[0], days=10, preloaded_data=preloaded, benchmark_df=bm)
    assert "error" not in res_ml
    assert res_ml["strategy_type"] == "ml_walkforward"
    assert res_ml["ml_train_samples"] > 0
    for tr in res_ml["trades"]:
        if tr["side"] == "BUY":
            assert tr["signal_time"] < tr["timestamp"]

    res_rev = sim.run_backtest_pattern(cfg.patterns[1], days=10, preloaded_data=preloaded, benchmark_df=bm)
    assert "error" not in res_rev
    assert res_rev["strategy_type"] == "mean_reversion"

    # Also test mean-reversion scorer directly on a dip snapshot
    ind = compute_indicators(preloaded["7203.T"])
    snap = extract_latest_snapshot(ind)
    rec = score_stock_mean_reversion(StockInfo("7203.T", "トヨタ"), snap, cfg)
    assert 0.0 <= rec.score <= 100.0
