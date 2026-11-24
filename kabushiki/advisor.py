"""Investment recommendation and sell-timing estimation engine."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, time
from typing import Any

from .config import AppConfig, StockInfo
from .indicators import IndicatorSnapshot
from .market_data import JST


@dataclass
class StockRecommendation:
    symbol: str
    name: str
    sector: str
    timestamp: str
    price: float
    change_pct: float
    score: float
    action: str  # STRONG_BUY, BUY, HOLD, SELL
    action_label: str
    reasons: list[str]
    suggested_entry: float
    target_price: float
    stop_loss_price: float
    risk_reward: float
    rsi14: float
    macd_hist: float
    vwap: float
    volume_ratio: float
    bb_pct_b: float
    atr14: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SellEvaluation:
    symbol: str
    name: str
    quantity: int
    entry_price: float
    current_price: float
    highest_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    should_sell: bool
    urgency: float  # 0-100
    timing_label: str  # 今すぐ売却推奨, 利確検討, 損切り警戒, 保有継続
    primary_reason: str
    reasons: list[str]
    target_price: float
    stop_loss_price: float
    trailing_stop_price: float | None
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_jpy(val: float) -> float:
    if val >= 10000:
        return round(val, 0)
    if val >= 1000:
        return round(val, 1)
    return round(val, 1)


def score_stock(
    stock: StockInfo,
    snap: IndicatorSnapshot,
    cfg: AppConfig,
) -> StockRecommendation:
    """Evaluate a Japanese stock's buy/hold/sell attractiveness (0-100 score)."""
    score = 50.0
    reasons: list[str] = []

    # 1. Trend & Moving Averages (EMA5 vs EMA20, Price vs VWAP) -> up to +/- 22 pts
    golden_cross = snap.ema5 > snap.ema20 and snap.ema5_prev <= snap.ema20_prev
    dead_cross = snap.ema5 < snap.ema20 and snap.ema5_prev >= snap.ema20_prev

    if golden_cross:
        score += 14.0
        reasons.append("短期ゴールデンクロス発生 (EMA5 > EMA20)")
    elif snap.ema5 > snap.ema20:
        score += 8.0
        reasons.append("短期上昇トレンド維持 (EMA5 > EMA20)")
    elif dead_cross:
        score -= 14.0
        reasons.append("短期デッドクロス発生 (EMA5 < EMA20)")
    else:
        score -= 7.0
        reasons.append("短期下降トレンド (EMA5 < EMA20)")

    if snap.close > snap.vwap * 1.001:
        score += 7.0
        reasons.append("VWAP上抜けで日中買い優勢")
    elif snap.close < snap.vwap * 0.999:
        score -= 6.0
        reasons.append("VWAP下回りで日中売り優勢")

    # Medium-term trend filter (SMA60) to avoid catching falling knives
    if snap.close > snap.sma60 * 1.002 and snap.ema20 >= snap.ema20_prev:
        score += 7.0
        reasons.append("中期トレンド上昇 (SMA60上抜け)")
    elif snap.close < snap.sma60 * 0.995 and snap.ema20 < snap.ema20_prev:
        score -= 9.0
        reasons.append("中期下降トレンド下 (SMA60割れ)")

    # 2. MACD Momentum -> up to +/- 16 pts
    macd_turning_up = snap.macd_hist > 0 and snap.macd_hist_prev <= 0
    macd_turning_down = snap.macd_hist < 0 and snap.macd_hist_prev >= 0
    if macd_turning_up:
        score += 13.0
        reasons.append("MACDヒストグラム陽転")
    elif snap.macd_hist > 0 and snap.macd_hist > snap.macd_hist_prev:
        score += 8.0
        reasons.append("MACD上昇モメンタム拡大")
    elif macd_turning_down:
        score -= 13.0
        reasons.append("MACDヒストグラム陰転")
    elif snap.macd_hist < 0 and snap.macd_hist < snap.macd_hist_prev:
        score -= 8.0
        reasons.append("MACD下落モメンタム拡大")

    # 3. RSI(14) Mean-reversion + healthy momentum -> up to +/- 16 pts
    if 25.0 <= snap.rsi14 <= 38.0 and snap.close >= snap.open:
        score += 13.0
        reasons.append(f"RSI売られ過ぎ圏からの反発気配 (RSI={snap.rsi14:.1f})")
    elif 45.0 <= snap.rsi14 <= 64.0:
        score += 6.0
        reasons.append(f"RSI健全な上昇圏 (RSI={snap.rsi14:.1f})")
    elif snap.rsi14 >= 75.0:
        score -= 12.0
        reasons.append(f"RSI過熱圏・反落警戒 (RSI={snap.rsi14:.1f})")
    elif snap.rsi14 < 25.0:
        score += 4.0
        reasons.append(f"RSI極端な売られ過ぎ (RSI={snap.rsi14:.1f})")

    # 4. Bollinger Band Position -> up to +/- 10 pts
    if snap.bb_pct_b <= 0.22 and snap.close >= snap.open:
        score += 9.0
        reasons.append("ボリンジャー下限付近で押し目反発")
    elif 0.50 <= snap.bb_pct_b <= 0.82:
        score += 4.0
    elif snap.bb_pct_b >= 0.98:
        score -= 8.0
        reasons.append("ボリンジャー上限超過・高値掴み注意")

    # 5. Volume Surge Confirmation -> up to +/- 12 pts
    bullish_candle = snap.close > snap.open
    if snap.volume_ratio >= 1.5 and bullish_candle:
        score += 11.0
        reasons.append(f"出来高急増を伴う陽線 ({snap.volume_ratio:.1f}倍)")
    elif snap.volume_ratio >= 1.5 and not bullish_candle:
        score -= 9.0
        reasons.append(f"出来高急増を伴う陰線 ({snap.volume_ratio:.1f}倍)")
    elif snap.volume_ratio >= 1.2 and bullish_candle:
        score += 5.0
        reasons.append(f"出来高増加 ({snap.volume_ratio:.1f}倍)")

    # 6. Short-term ROC momentum -> up to +/- 6 pts
    if 0.003 <= snap.roc4 <= 0.035:
        score += 5.0
    elif snap.roc4 <= -0.015:
        score -= 5.0

    score = max(0.0, min(100.0, round(score, 1)))

    if score >= cfg.strong_buy_threshold:
        action = "STRONG_BUY"
        action_label = "強い買い推奨"
    elif score >= cfg.buy_score_threshold:
        action = "BUY"
        action_label = "買い推奨"
    elif score <= cfg.sell_score_threshold:
        action = "SELL"
        action_label = "売り警戒"
    else:
        action = "HOLD"
        action_label = "様子見"

    # Compute ATR-adapted take-profit and stop-loss prices
    entry = snap.close
    atr_target_dist = max(snap.atr14 * 2.2, entry * cfg.take_profit_pct)
    atr_stop_dist = max(snap.atr14 * 1.4, entry * cfg.stop_loss_pct)
    # Clamp sensible bounds for intraday/swing trading
    atr_target_dist = min(atr_target_dist, entry * max(cfg.take_profit_pct * 2.0, 0.08))
    atr_stop_dist = min(atr_stop_dist, entry * max(cfg.stop_loss_pct * 1.8, 0.05))

    target_price = _round_jpy(entry + atr_target_dist)
    stop_loss_price = _round_jpy(max(1.0, entry - atr_stop_dist))
    risk = max(entry - stop_loss_price, 0.01)
    reward = max(target_price - entry, 0.0)
    risk_reward = round(reward / risk, 2)

    if not reasons:
        reasons.append("方向感待ち（中立圏）")

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
        risk_reward=risk_reward,
        rsi14=round(snap.rsi14, 1),
        macd_hist=round(snap.macd_hist, 4),
        vwap=_round_jpy(snap.vwap),
        volume_ratio=round(snap.volume_ratio, 2),
        bb_pct_b=round(snap.bb_pct_b, 2),
        atr14=round(snap.atr14, 2),
    )


def evaluate_sell_timing(
    position: dict[str, Any],
    rec: StockRecommendation,
    snap: IndicatorSnapshot,
    cfg: AppConfig,
    bar_time: datetime | None = None,
) -> SellEvaluation:
    """Estimate optimal sell timing for a currently held position."""
    entry_price = float(position["avg_cost"])
    quantity = int(position["quantity"])
    current_price = float(snap.close)
    highest_price = max(float(position.get("highest_price") or entry_price), current_price, float(snap.high))

    target_price = float(position.get("target_price") or _round_jpy(entry_price * (1.0 + cfg.take_profit_pct)))
    stop_loss_price = float(position.get("stop_loss_price") or _round_jpy(entry_price * (1.0 - cfg.stop_loss_pct)))

    unrealized_pnl = (current_price - entry_price) * quantity
    unrealized_pnl_pct = ((current_price - entry_price) / entry_price) if entry_price > 0 else 0.0
    peak_gain_pct = ((highest_price - entry_price) / entry_price) if entry_price > 0 else 0.0

    trailing_stop_price: float | None = None
    if peak_gain_pct >= cfg.trailing_activation_pct:
        trailing_stop_price = _round_jpy(highest_price * (1.0 - cfg.trailing_stop_pct))

    reasons: list[str] = []
    should_sell = False
    urgency = 15.0
    primary_reason = "上昇余地あり・保有継続"

    # 1. Stop-loss triggered
    if current_price <= stop_loss_price or unrealized_pnl_pct <= -cfg.stop_loss_pct:
        should_sell = True
        urgency = max(urgency, 96.0)
        msg = f"損切りライン到達 (現在¥{current_price:,.1f} <= 損切り¥{stop_loss_price:,.1f} / {unrealized_pnl_pct*100:+.2f}%)"
        reasons.append(msg)
        primary_reason = msg

    # 2. Trailing stop triggered (lock in gains after peak)
    if trailing_stop_price is not None and current_price <= trailing_stop_price:
        should_sell = True
        urgency = max(urgency, 92.0)
        msg = f"トレーリングストップ発動 (高値¥{highest_price:,.1f}から反落・利益確定)"
        reasons.append(msg)
        if not primary_reason.startswith("損切り"):
            primary_reason = msg

    # 3. Take-profit target reached
    if current_price >= target_price or unrealized_pnl_pct >= cfg.take_profit_pct:
        should_sell = True
        urgency = max(urgency, 88.0)
        msg = f"利確目標到達 (現在¥{current_price:,.1f} >= 目標¥{target_price:,.1f} / {unrealized_pnl_pct*100:+.2f}%)"
        reasons.append(msg)
        if urgency == 88.0:
            primary_reason = msg

    # 4. Technical reversal / overheated exhaustion (with anti-whipsaw confirmation)
    held_minutes = 999.0
    if bar_time is not None and position.get("entry_time"):
        try:
            et = datetime.fromisoformat(str(position["entry_time"]))
            if et.tzinfo is None:
                et = et.replace(tzinfo=JST)
            held_minutes = max(0.0, (bar_time.astimezone(JST) - et.astimezone(JST)).total_seconds() / 60.0)
        except Exception:
            held_minutes = 999.0

    severe_breakdown = rec.score <= max(20.0, cfg.sell_score_threshold - 6.0)
    confirmed_reversal = rec.score <= cfg.sell_score_threshold and held_minutes >= 30.0
    if severe_breakdown or confirmed_reversal:
        should_sell = True
        urgency = max(urgency, 78.0)
        msg = f"テクニカル悪化による売りシグナル (スコア {rec.score:.1f} <= {cfg.sell_score_threshold:.0f})"
        reasons.append(msg)
        if len(reasons) == 1:
            primary_reason = msg
    elif snap.rsi14 >= 78.0 and unrealized_pnl_pct > 0.008 and snap.close < snap.open:
        should_sell = True
        urgency = max(urgency, 76.0)
        msg = f"高値圏RSI過熱＋陰線出現で利確推奨 (RSI={snap.rsi14:.1f})"
        reasons.append(msg)
        if len(reasons) == 1:
            primary_reason = msg

    # 5. Optional EOD day-trade exit (15:10+ JST)
    if cfg.close_before_eod and bar_time is not None:
        bt_jst = bar_time.astimezone(JST)
        if bt_jst.time() >= time(15, 10):
            should_sell = True
            urgency = max(urgency, 82.0)
            msg = "大引け前デイトレ決済ルール (15:10以降)"
            reasons.append(msg)
            if len(reasons) == 1:
                primary_reason = msg

    # Warning levels when holding
    if not should_sell:
        if unrealized_pnl_pct <= -(cfg.stop_loss_pct * 0.65):
            urgency = 58.0
            msg = f"損切りライン接近中 (残り{(current_price - stop_loss_price):,.1f}円)"
            reasons.append(msg)
            primary_reason = msg
        elif unrealized_pnl_pct >= cfg.take_profit_pct * 0.7:
            urgency = 55.0
            msg = f"利確目標に接近中 (目標¥{target_price:,.1f}まであと¥{max(0.0, target_price - current_price):,.1f})"
            reasons.append(msg)
            primary_reason = msg
        elif rec.score < 48.0:
            urgency = 45.0
            msg = f"モメンタムやや鈍化 (スコア {rec.score:.1f})"
            reasons.append(msg)
            primary_reason = msg
        else:
            urgency = max(10.0, round(60.0 - rec.score * 0.5, 1))
            reasons.append(f"トレンド維持 (スコア {rec.score:.1f} / 利確目標¥{target_price:,.1f} / 損切り¥{stop_loss_price:,.1f})")
            primary_reason = reasons[0]

    if should_sell:
        timing_label = "今すぐ売却推奨"
    elif urgency >= 55.0 and unrealized_pnl_pct > 0:
        timing_label = "利確検討"
    elif urgency >= 55.0 and unrealized_pnl_pct <= 0:
        timing_label = "損切り警戒"
    else:
        timing_label = "保有継続"

    return SellEvaluation(
        symbol=str(position["symbol"]),
        name=str(position.get("name") or rec.name),
        quantity=quantity,
        entry_price=_round_jpy(entry_price),
        current_price=_round_jpy(current_price),
        highest_price=_round_jpy(highest_price),
        unrealized_pnl=round(unrealized_pnl, 1),
        unrealized_pnl_pct=round(unrealized_pnl_pct * 100.0, 2),
        should_sell=should_sell,
        urgency=round(urgency, 1),
        timing_label=timing_label,
        primary_reason=primary_reason,
        reasons=reasons,
        target_price=_round_jpy(target_price),
        stop_loss_price=_round_jpy(stop_loss_price),
        trailing_stop_price=trailing_stop_price,
        score=rec.score,
    )
