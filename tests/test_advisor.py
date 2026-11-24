"""Unit tests for recommendation scoring and sell-timing estimator."""

from datetime import datetime
from kabushiki.advisor import score_stock, evaluate_sell_timing
from kabushiki.config import AppConfig, StockInfo
from kabushiki.indicators import IndicatorSnapshot
from kabushiki.market_data import JST


def _sample_snap(**overrides) -> IndicatorSnapshot:
    base = dict(
        timestamp="2026-11-24T10:30:00+09:00",
        open=1000.0,
        high=1025.0,
        low=998.0,
        close=1020.0,
        volume=50000.0,
        ema5=1012.0,
        ema20=1005.0,
        sma60=990.0,
        rsi14=56.0,
        macd=4.5,
        macd_signal=3.0,
        macd_hist=1.5,
        macd_hist_prev=-0.2,
        bb_upper=1040.0,
        bb_mid=1005.0,
        bb_lower=970.0,
        bb_pct_b=0.71,
        atr14=12.0,
        vwap=1008.0,
        volume_ratio=1.8,
        roc4=0.018,
        roc12=0.03,
        prev_close=1005.0,
        change_pct=0.0149,
        ema5_prev=1004.0,
        ema20_prev=1005.0,
    )
    base.update(overrides)
    return IndicatorSnapshot(**base)


def test_score_stock_strong_buy_on_golden_cross_and_volume():
    cfg = AppConfig()
    stock = StockInfo(symbol="7203.T", name="トヨタ自動車", sector="輸送用機器")
    snap = _sample_snap()

    rec = score_stock(stock, snap, cfg)
    assert rec.score >= cfg.strong_buy_threshold
    assert rec.action == "STRONG_BUY"
    assert rec.target_price > rec.price
    assert rec.stop_loss_price < rec.price
    assert rec.risk_reward > 1.0
    assert len(rec.reasons) >= 2


def test_sell_timing_stop_loss_and_trailing_stop():
    cfg = AppConfig(take_profit_pct=0.035, stop_loss_pct=0.02, trailing_activation_pct=0.015, trailing_stop_pct=0.01)
    stock = StockInfo(symbol="9432.T", name="NTT", sector="情報・通信")

    # Case 1: Stop-loss hit
    snap_sl = _sample_snap(close=975.0, open=990.0, high=990.0, low=970.0, ema5=985.0, ema20=995.0, macd_hist=-1.0)
    rec_sl = score_stock(stock, snap_sl, cfg)
    pos = {
        "symbol": "9432.T",
        "name": "NTT",
        "quantity": 100,
        "avg_cost": 1000.0,
        "highest_price": 1005.0,
        "target_price": 1035.0,
        "stop_loss_price": 980.0,
    }
    ev_sl = evaluate_sell_timing(pos, rec_sl, snap_sl, cfg, bar_time=datetime(2026, 11, 24, 13, 0, tzinfo=JST))
    assert ev_sl.should_sell is True
    assert ev_sl.urgency >= 90.0
    assert "損切り" in ev_sl.primary_reason

    # Case 2: Trailing stop hit after peaking at 1030 (+3%) and dropping to 1015
    snap_tr = _sample_snap(close=1015.0, open=1022.0, high=1022.0, low=1014.0)
    rec_tr = score_stock(stock, snap_tr, cfg)
    pos_tr = {
        "symbol": "9432.T",
        "name": "NTT",
        "quantity": 100,
        "avg_cost": 1000.0,
        "highest_price": 1030.0,
        "target_price": 1045.0,
        "stop_loss_price": 980.0,
    }
    ev_tr = evaluate_sell_timing(pos_tr, rec_tr, snap_tr, cfg, bar_time=datetime(2026, 11, 24, 14, 0, tzinfo=JST))
    assert ev_tr.should_sell is True
    assert ev_tr.trailing_stop_price is not None
    assert "トレーリングストップ" in ev_tr.primary_reason
