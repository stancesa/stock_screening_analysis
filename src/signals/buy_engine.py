from __future__ import annotations
import numpy as np, pandas as pd
from core.types import BuyParams
from typing import Any, Dict
from indicators.basic import (
    _rsi_wilder, _slope_pct_per_day, _donchian_prev_high, _donch_breakout_strength, _atr_from_ohlc,
    _sigmoid, _bb_percent_b_and_bw, _boll_lower
)

def compute_buy_signal(
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
    Build a BUY score by blending: RSI, Trend, Value, Flow, Bollinger %B, and Donchian breakout strength.
    The composite is a weighted sum (clipped to [0,1]). A BUY fires when composite ≥ composite_threshold,
    then guardrails (liquidity, min price, valid stop) must pass.
    """
    last = float(close[-1])
    sma200_last = float(sma200[-1]) if len(sma200) else float("nan")

    # --- RSI (point or recomputed)
    rsi_val = None
    for c in ("momentum__rsi", "rsi"):
        if c in row and pd.notna(row[c]):
            rsi_val = float(row[c]); break
    if rsi_val is None or not np.isfinite(rsi_val):
        rsi_val = _rsi_wilder(close, 14)

    # --- Value vs SMA200
    dev_pct = None
    if np.isfinite(sma200_last) and sma200_last > 0:
        dev_pct = (last / sma200_last - 1.0) * 100.0

    # --- Trend (price + SMA200 slopes + regime)
    trend_price_slope = _slope_pct_per_day(close, params.sma200_window)   # %/day
    trend_sma_slope   = _slope_pct_per_day(sma200, params.sma200_window)  # %/day
    above_sma = (np.isfinite(sma200_last) and last > sma200_last)

    # --- Volume "flow"
    vol = float(row.get("vol", row.get("volume__vol", np.nan)))
    vol_avg20 = float(row.get("vol_avg20", row.get("volume__vol_avg20", np.nan)))
    vol_ratio = (vol / vol_avg20) if (np.isfinite(vol) and np.isfinite(vol_avg20) and vol_avg20 > 0) else np.nan

    # --- Donchian (prior N-high) + ATR strength
    prev_high = _donchian_prev_high(high, params.donch_lookback)
    atr = _atr_from_ohlc(high, low, close, period=14)
    donch_buy_raw, _ = _donch_breakout_strength(last, prev_high, np.nan, atr)
    s_donch = _sigmoid(donch_buy_raw / 0.5)  # ~0.5*ATR over prior high -> noticeable score

    # (Legacy) simple breakout boolean and gap-up bonus
    breakout = (np.isfinite(prev_high) and last >= prev_high)
    gap_up = (len(open_) >= 2 and len(high) >= 2 and float(open_[-1]) > float(high[-2]) * (1.0 + params.gap_min_pct/100.0))
    gap_bonus = 0.05 if gap_up else 0.0

    # --- Bollinger %B and BandWidth
    percent_b, bw = _bb_percent_b_and_bw(close, params.bb_window, params.bb_k)
    # For BUY, prefer being near/below lower band → high score when %B is small
    s_bbands = 1.0 - float(percent_b) if np.isfinite(percent_b) else 0.0
    s_bbands = float(np.clip(s_bbands, 0.0, 1.0))

    # --- Individual component scores in [0,1]
    # RSI: high score when RSI is low (oversold turning up), centered at rsi_buy_max
    k_rsi = 5.0
    s_rsi = _sigmoid((params.rsi_buy_max - float(rsi_val)) / k_rsi) if np.isfinite(rsi_val) else 0.0

    # Value: below SMA200 favored; center at negative dev %
    k_val = 2.5
    if dev_pct is None or not np.isfinite(dev_pct):
        s_value = 0.0
    else:
        s_value = _sigmoid(((-dev_pct) - abs(params.value_center_dev_pct)) / k_val)

    # Trend blend: regime + slopes
    s_trend = 0.0
    s_trend += 0.5 * (1.0 if above_sma else 0.0)
    s_trend += 0.25 * _sigmoid(trend_price_slope * 500.0)
    s_trend += 0.25 * _sigmoid(trend_sma_slope * 1500.0)

    # Flow (volume spike vs avg)
    s_flow = 0.0 if not np.isfinite(vol_ratio) else _sigmoid((vol_ratio - params.vol_ratio_min) / 0.4)

    # Legacy breakout flag (optional small weight)
    s_breakout = 1.0 if breakout else 0.0

    # --- Composite score
    comp = (
        params.w_rsi       * s_rsi +
        params.w_trend     * s_trend +
        params.w_value     * s_value +
        params.w_flow      * s_flow +
        params.w_bbands    * s_bbands +
        params.w_donchian  * s_donch +
        params.w_breakout  * s_breakout +
        gap_bonus
    )
    comp = float(np.clip(comp, 0.0, 1.0))

    # --- Stops (same as before)
    atr_stop = last - params.atr_mult * atr if np.isfinite(atr) else float("nan")
    pct_stop = last * (1.0 - params.stop_pct / 100.0)
    band_stop = _boll_lower(close, window=20, k=2.0)
    swing_stop = float(np.min(low[-10:])) if len(low) >= 10 else float("nan")
    candidates = [s for s in (atr_stop, pct_stop, band_stop, swing_stop) if np.isfinite(s) and s < last]
    recommended_stop = max(candidates) if candidates else np.nan
    if not np.isfinite(recommended_stop):
        recommended_stop = pct_stop if np.isfinite(pct_stop) else np.nan

    basis = None
    if np.isfinite(recommended_stop):
        basis = (
            "atr"   if np.isclose(recommended_stop, atr_stop,   atol=1e-6) else
            "band"  if np.isclose(recommended_stop, band_stop,  atol=1e-6) else
            "swing" if np.isclose(recommended_stop, swing_stop, atol=1e-6) else
            "percent"
        )

    if np.isfinite(recommended_stop):
        risk_amt = last - recommended_stop
        target = last + params.reward_R * risk_amt
        rr = (target - last) / risk_amt if risk_amt > 0 else np.nan
    else:
        target, rr = np.nan, np.nan

    # Sizing & guards
    risk_dollars = params.portfolio_value * (params.risk_per_trade_pct / 100.0)
    shares_by_stop = int(np.floor(risk_dollars / max(last - recommended_stop, 1e-9))) if np.isfinite(recommended_stop) else 0
    adv_dollars = (vol_avg20 * last) if np.isfinite(vol_avg20) else np.nan
    liq_ok = (not np.isfinite(adv_dollars)) or (adv_dollars >= params.min_adv_dollars)
    price_ok = last >= params.min_price
    guards_ok = liq_ok and price_ok and np.isfinite(recommended_stop) and (last > recommended_stop)

    reasons = []
    if not liq_ok: reasons.append("Liquidity below minimum ADV$")
    if not price_ok: reasons.append(f"Price < ${params.min_price:.2f}")
    if not np.isfinite(recommended_stop): reasons.append("No valid stop")
    if last <= recommended_stop: reasons.append("Stop >= last")

    return {
        "buy": bool(comp >= params.composite_threshold and guards_ok),
        "score": comp,
        "components": {
            "rsi": s_rsi, "trend": s_trend, "value": s_value, "flow": s_flow,
            "bbands": s_bbands, "donchian": s_donch, "breakout_flag": s_breakout, "gap_bonus": gap_bonus
        },
        "features": {
            "RSI": rsi_val, "dev_pct_vs_SMA200": dev_pct, "vol_ratio": vol_ratio,
            "percent_b": percent_b, "bb_bandwidth": bw, "donch_prev_high": prev_high,
        },
        "stop": float(recommended_stop) if np.isfinite(recommended_stop) else None,
        "stop_basis": basis,
        "target": float(target) if np.isfinite(target) else None,
        "rr_multiple": float(rr) if np.isfinite(rr) else None,
        "shares": int(max(shares_by_stop, 0)) if guards_ok else 0,
        "guards_ok": guards_ok,
        "guard_reasons": reasons,
        "adv_dollars": float(adv_dollars) if np.isfinite(adv_dollars) else None
    }