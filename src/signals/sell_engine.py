from __future__ import annotations
import numpy as np, pandas as pd
from core.types import BuyParams
from typing import Any, Dict
from indicators.basic import (
    _rsi_series_wilder, _slope_pct_per_day, _donchian_prev_low, _donch_breakout_strength,
    _atr_from_ohlc, _sigmoid, _bb_percent_b_and_bw
)

def compute_sell_signal(
    row: pd.Series,
    dates: pd.Series,
    close: np.ndarray,
    sma200: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    params: BuyParams
) -> Dict[str, Any]:
    """
    Build a SELL score by blending: RSI overbought/rollover, trend down, Donchian breakdown strength,
    Bollinger %B near the upper band, exhaustion, and distribution/flow.
    """
    last = float(close[-1])

    # --- RSI (value + rollover)
    rsi_val = None
    for c in ("momentum__rsi", "rsi"):
        if c in row and pd.notna(row[c]):
            rsi_val = float(row[c]); break
    rsi_series = _rsi_series_wilder(close, 14)
    if rsi_val is None or not np.isfinite(rsi_val):
        rsi_val = float(rsi_series[-1]) if len(rsi_series) else float("nan")
    rsi_rollover = False
    if len(rsi_series) >= 3 and np.isfinite(rsi_series[-1]) and np.isfinite(rsi_series[-2]):
        rsi_rollover = (rsi_series[-2] > rsi_series[-1])

    # --- Trend down (EMA/SMA crosses + slopes)
    ema_fast = pd.Series(close, dtype="float64").ewm(span=params.ema_fast_span, adjust=False).mean().to_numpy()
    sma_mid  = pd.Series(close, dtype="float64").rolling(params.sma_mid_window, min_periods=5).mean().to_numpy()
    ema_slope = _slope_pct_per_day(ema_fast, min(params.ema_fast_span*2, len(ema_fast)))
    sma_slope = _slope_pct_per_day(sma_mid,  min(params.sma_mid_window, len(sma_mid)))
    ema_cross_down = (np.isfinite(ema_fast[-1]) and last < float(ema_fast[-1]))
    sma_cross_down = (np.isfinite(sma_mid[-1])  and last < float(sma_mid[-1]))

    # --- Donchian (prior N-low) + ATR strength
    prev_low = _donchian_prev_low(low, params.donch_lookback_sell)
    atr = _atr_from_ohlc(high, low, close, period=14)
    _, donch_sell_raw = _donch_breakout_strength(last, np.nan, prev_low, atr)
    s_donch = _sigmoid(donch_sell_raw / 0.5)

    # --- Bollinger %B (near upper band = more sell)
    percent_b, bw = _bb_percent_b_and_bw(close, params.bb_window, params.bb_k)
    s_bbands = float(percent_b) if np.isfinite(percent_b) else 0.0  # high near upper band
    s_bbands = float(np.clip(s_bbands, 0.0, 1.0))

    # --- Exhaustion (touching upper band) – keep as a binary “spike” feature
    s = pd.Series(close, dtype="float64")
    bb_ma = s.rolling(20, min_periods=20).mean()
    bb_sd = s.rolling(20, min_periods=20).std(ddof=0)
    bb_up = (bb_ma + 2.0 * bb_sd).to_numpy()
    exhaustion = (np.isfinite(bb_up[-1]) and last > float(bb_up[-1]))

    # --- Flow out (distribution): down candle + volume spike
    vol = float(row.get("vol", row.get("volume__vol", np.nan)))
    vol_avg20 = float(row.get("vol_avg20", row.get("volume__vol_avg20", np.nan)))
    vol_ratio = (vol / vol_avg20) if (np.isfinite(vol) and np.isfinite(vol_avg20) and vol_avg20 > 0) else np.nan
    down_candle = (len(open_) and last < float(open_[-1]))
    flow_out = (down_candle and np.isfinite(vol_ratio) and (vol_ratio >= params.vol_ratio_min))

    # --- Gap-down vs prior low
    gap_down = (len(open_) >= 2 and len(low) >= 2 and float(open_[-1]) < float(low[-2]) * (1.0 - params.gap_down_min_pct/100.0))
    gap_bonus = 0.05 if gap_down else 0.0

    # ---- Components in [0,1]
    s_rsi = 0.0
    if np.isfinite(rsi_val):
        core = (float(rsi_val) - params.rsi_overbought_min) / 5.0
        s_rsi = _sigmoid(core) * (1.25 if rsi_rollover else 1.0)
        s_rsi = float(np.clip(s_rsi, 0.0, 1.0))

    s_trend_down = 0.0
    s_trend_down += 0.4 * (1.0 if ema_cross_down else 0.0)
    s_trend_down += 0.2 * (1.0 if sma_cross_down else 0.0)
    s_trend_down += 0.2 * _sigmoid(-ema_slope * 500.0)
    s_trend_down += 0.2 * _sigmoid(-sma_slope * 1500.0)

    s_breakdown = 1.0 if (np.isfinite(prev_low) and last <= prev_low) else 0.0
    s_exhaustion = 1.0 if exhaustion else 0.0
    s_flow_out = 1.0 if flow_out else 0.0

    sell_comp = (
        params.w_rsi_sell      * s_rsi +
        params.w_trend_down    * s_trend_down +
        params.w_donchian_sell * s_donch +
        params.w_bbands_sell   * s_bbands +
        params.w_breakdown     * s_breakdown +
        params.w_exhaustion    * s_exhaustion +
        params.w_flow_out      * s_flow_out +
        gap_bonus
    )
    sell_comp = float(np.clip(sell_comp, 0.0, 1.0))
    sell = (sell_comp >= params.sell_threshold)

    reasons = []
    if s_breakdown >= 1.0: reasons.append("Breakdown below prior Donchian low")
    if ema_cross_down: reasons.append(f"Below EMA{params.ema_fast_span}")
    if sma_cross_down: reasons.append(f"Below SMA{params.sma_mid_window}")
    if np.isfinite(rsi_val) and rsi_val >= params.rsi_overbought_min: reasons.append(f"RSI≥{int(params.rsi_overbought_min)}")
    if rsi_rollover: reasons.append("RSI rolling over")
    if exhaustion: reasons.append("Bollinger upper exhaustion")
    if flow_out: reasons.append("Distribution day (volume spike on down candle)")
    if gap_down: reasons.append("Gap-down vs prior low")

    return {
        "sell": bool(sell),
        "score": sell_comp,
        "components": {
            "rsi": s_rsi, "trend_down": s_trend_down, "bbands": s_bbands,
            "donchian": s_donch, "breakdown_flag": s_breakdown,
            "exhaustion": s_exhaustion, "flow_out": s_flow_out, "gap_bonus": gap_bonus
        },
        "features": {
            "RSI": rsi_val, "percent_b": percent_b, "bb_bandwidth": bw,
            "ema_cross_down": ema_cross_down, "sma_cross_down": sma_cross_down,
            "prev_donch_low": prev_low, "gap_down": gap_down
        },
        "reasons": reasons
    }