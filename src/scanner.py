import datetime as dt
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from .config import ScannerConfig
from .utils import read_list, fetch_history, market_risk_ok
from .indicators import (
    rsi, macd, bullish_macd_cross,
    ema, atr, donchian, swing_low, bollinger
)
from .news_fundamentals import get_news_and_sentiment, get_fundamentals


# ---------- helpers ----------

def _to_float(x) -> float:
    """Return a scalar float from pandas/NumPy without FutureWarnings."""
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return float("nan")
        x = x.iloc[-1]
    elif isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return float("nan")
        x = x[-1]
    try:
        # prefer .item() when available (NumPy scalar)
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")


# ---------- main ----------

def technicals(df: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, Any]:
    """
    Compute signals & many entry/stop helpers (NaN-safe, scalar outputs).
    """
    # Tunables with safe defaults
    atr_period = getattr(cfg, "atr_period", 14)
    atr_multiple = getattr(cfg, "atr_multiple", 1.5)
    ema_fast_span = getattr(cfg, "ema_fast_span", 21)
    ema_slow_span = getattr(cfg, "ema_slow_span", 50)
    boll_period = getattr(cfg, "boll_period", 20)
    boll_std = getattr(cfg, "boll_std", 2.0)
    donchian_lb = getattr(cfg, "donchian_lookback", 20)
    rsi_oversold = getattr(cfg, "rsi_oversold", 35)

    close = df["AdjClose"]
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    # Core indicators
    rsi_series = rsi(close, cfg.rsi_period)
    macd_line, signal_line, _ = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    sma200_series = close.rolling(cfg.sma_long).mean()
    vol_avg20_series = df["Volume"].rolling(cfg.vol_avg_period).mean()

    # Additional indicators
    sma20_series = close.rolling(20).mean()
    sma50_series = close.rolling(getattr(cfg, "sma_mid", 50)).mean()
    ema_fast_series = ema(close, ema_fast_span)
    ema_slow_series = ema(close, ema_slow_span)
    atr_series = atr(df, period=atr_period)
    donch_hi, donch_lo = donchian(df, lookback=donchian_lb)
    bb_ma, bb_up, bb_lo, bb_pb, bb_bw = bollinger(close, period=boll_period, num_std=boll_std)

    # Scalars
    last_close = _to_float(latest["Close"])
    adj_close_val = _to_float(latest["AdjClose"])
    prev_close = _to_float(prev["Close"])
    volume = _to_float(latest["Volume"])
    vol_avg20 = _to_float(vol_avg20_series.iloc[-1])

    rsi_val = _to_float(rsi_series.iloc[-1])
    rsi_prev = _to_float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else float("nan")

    macd_val = _to_float(macd_line.iloc[-1])
    macd_sig = _to_float(signal_line.iloc[-1])

    sma200 = _to_float(sma200_series.iloc[-1])
    sma50 = _to_float(sma50_series.iloc[-1])
    sma20 = _to_float(sma20_series.iloc[-1])

    ema_fast = _to_float(ema_fast_series.iloc[-1])
    ema_slow = _to_float(ema_slow_series.iloc[-1])

    atr_val = _to_float(atr_series.iloc[-1])
    donch_hi_v = _to_float(donch_hi.iloc[-1])
    donch_lo_v = _to_float(donch_lo.iloc[-1])

    bb_ma_v = _to_float(bb_ma.iloc[-1])
    bb_up_v = _to_float(bb_up.iloc[-1])
    bb_lo_v = _to_float(bb_lo.iloc[-1])
    bb_pb_v = _to_float(bb_pb.iloc[-1])
    bb_bw_v = _to_float(bb_bw.iloc[-1])

    # 52-week distances
    hi_52 = _to_float(df["High"].rolling(252).max().iloc[-1])
    lo_52 = _to_float(df["Low"].rolling(252).min().iloc[-1])
    pct_from_52w_high = (adj_close_val / hi_52 - 1.0) * 100 if np.isfinite(adj_close_val) and np.isfinite(hi_52) and hi_52 > 0 else None
    pct_to_52w_low = (adj_close_val / lo_52 - 1.0) * 100 if np.isfinite(adj_close_val) and np.isfinite(lo_52) and lo_52 > 0 else None

    # Entry gates
    rsi_ok = (np.isfinite(rsi_val) and rsi_val <= rsi_oversold)

    macd_cross_ok = bullish_macd_cross(macd_line, signal_line, getattr(cfg, "macd_cross_lookback", 5))

    sma_dev_pct = None
    sma_ok = False
    if np.isfinite(adj_close_val) and np.isfinite(sma200) and sma200 != 0:
        sma_dev = (adj_close_val - sma200) / sma200
        sma_dev_pct = round(100.0 * sma_dev, 2)
        sma_ok = (sma_dev <= -getattr(cfg, "sma_deviation_req", 0.10))

    up_day = (np.isfinite(prev_close) and np.isfinite(last_close) and last_close > prev_close)
    vol_spike_ok = False
    if up_day and np.isfinite(volume) and np.isfinite(vol_avg20) and vol_avg20 > 0:
        vol_spike_ok = (volume >= getattr(cfg, "volume_spike_mult", 1.5) * vol_avg20)

    # Extra entry context
    ema_fast_cross_up = (np.isfinite(last_close) and np.isfinite(ema_fast) and last_close > ema_fast)
    ema50_reclaim = (np.isfinite(last_close) and np.isfinite(ema_slow) and last_close > ema_slow)
    above_boll_mid = (np.isfinite(last_close) and np.isfinite(bb_ma_v) and last_close >= bb_ma_v)
    donchian_breakout = (np.isfinite(last_close) and np.isfinite(donch_hi_v) and last_close >= donch_hi_v)
    rsi_hook_up = (np.isfinite(rsi_val) and np.isfinite(rsi_prev) and rsi_prev <= rsi_oversold and rsi_val > rsi_prev)

    # Gap up today?
    gap_up = (np.isfinite(prev_close) and np.isfinite(_to_float(df.iloc[-1]["Open"])) and _to_float(df.iloc[-1]["Open"]) > prev_close)

    # Stops (candidates)
    atr_stop = last_close - getattr(cfg, "atr_multiple", 1.5) * atr_val if np.isfinite(last_close) and np.isfinite(atr_val) else float("nan")
    swing_lv = swing_low(df, left=2, right=2)
    swing_stop = swing_lv if np.isfinite(swing_lv) else float("nan")
    band_stop = bb_lo_v if np.isfinite(bb_lo_v) else float("nan")
    pct_stop = last_close * (1 - getattr(cfg, "stop_loss_pct", 0.1)) if np.isfinite(last_close) else float("nan")

    candidates = [s for s in (atr_stop, swing_stop, band_stop, pct_stop) if np.isfinite(s) and s < last_close]
    recommended_stop = max(candidates) if candidates else None
    stop_basis = None
    if recommended_stop is not None:
        if abs(recommended_stop - (atr_stop if np.isfinite(atr_stop) else -1)) < 1e-9:
            stop_basis = "atr"
        elif abs(recommended_stop - (swing_stop if np.isfinite(swing_stop) else -1)) < 1e-9:
            stop_basis = "swing"
        elif abs(recommended_stop - (band_stop if np.isfinite(band_stop) else -1)) < 1e-9:
            stop_basis = "band"
        else:
            stop_basis = "percent"

    score = (
        int(rsi_ok)
        + int(macd_cross_ok)
        + int(sma_ok)
        + int(vol_spike_ok)
        + int(ema_fast_cross_up)
        + int(ema50_reclaim)
        + int(above_boll_mid)
        + int(donchian_breakout)
        + int(rsi_hook_up)
    )

    return {
        # core metrics
        "rsi": round(rsi_val, 2) if np.isfinite(rsi_val) else None,
        "macd": round(macd_val, 4) if np.isfinite(macd_val) else None,
        "macd_signal": round(macd_sig, 4) if np.isfinite(macd_sig) else None,
        "sma200": round(sma200, 2) if np.isfinite(sma200) else None,
        "sma50": round(sma50, 2) if np.isfinite(sma50) else None,
        "sma20": round(sma20, 2) if np.isfinite(sma20) else None,
        "ema_fast": round(ema_fast, 2) if np.isfinite(ema_fast) else None,
        "ema_slow": round(ema_slow, 2) if np.isfinite(ema_slow) else None,

        # distances
        "sma_dev_pct": sma_dev_pct,
        "pct_from_52w_high": round(pct_from_52w_high, 2) if pct_from_52w_high is not None else None,
        "pct_to_52w_low": round(pct_to_52w_low, 2) if pct_to_52w_low is not None else None,

        # volume, bands, donchian
        "vol": int(volume) if np.isfinite(volume) else None,
        "vol_avg20": int(vol_avg20) if np.isfinite(vol_avg20) else None,
        "bb_ma": round(bb_ma_v, 2) if np.isfinite(bb_ma_v) else None,
        "bb_up": round(bb_up_v, 2) if np.isfinite(bb_up_v) else None,
        "bb_lo": round(bb_lo_v, 2) if np.isfinite(bb_lo_v) else None,
        "bb_percent_b": round(bb_pb_v, 3) if np.isfinite(bb_pb_v) else None,
        "bb_bandwidth": round(bb_bw_v, 3) if np.isfinite(bb_bw_v) else None,
        "donchian_hi": round(donch_hi_v, 2) if np.isfinite(donch_hi_v) else None,
        "donchian_lo": round(donch_lo_v, 2) if np.isfinite(donch_lo_v) else None,

        # binary gates
        "rsi_ok": bool(rsi_ok),
        "macd_cross_ok": bool(macd_cross_ok),
        "sma_dev_ok": bool(sma_ok),
        "vol_spike_ok": bool(vol_spike_ok),
        "ema_fast_cross_up": bool(ema_fast_cross_up),
        "ema50_reclaim": bool(ema50_reclaim),
        "above_boll_mid": bool(above_boll_mid),
        "donchian_breakout": bool(donchian_breakout),
        "rsi_hook_up": bool(rsi_hook_up),
        "gap_up": bool(gap_up),

        # stops
        "atr": round(atr_val, 3) if np.isfinite(atr_val) else None,
        "atr_stop": round(atr_stop, 2) if np.isfinite(atr_stop) else None,
        "swing_low": round(swing_lv, 2) if np.isfinite(swing_lv) else None,
        "band_stop": round(band_stop, 2) if np.isfinite(band_stop) else None,
        "pct_stop": round(pct_stop, 2) if np.isfinite(pct_stop) else None,
        "recommended_stop": round(recommended_stop, 2) if recommended_stop is not None else None,
        "recommended_stop_basis": stop_basis,

        # score & series
        "signals_score": int(score),
        "entry_price": last_close if np.isfinite(last_close) else None,
        "close_series": df["Close"].tail(260).squeeze().tolist(),
        "sma200_series": sma200_series.tail(260).bfill().squeeze().tolist(),
        "dates_series": [d.strftime("%Y-%m-%d") for d in df.tail(260).index],
    }


def analyze_ticker(ticker: str, cfg: ScannerConfig) -> Optional[Dict[str, Any]]:
    end = dt.date.today()
    start = end - dt.timedelta(days=cfg.lookback_days)
    df = fetch_history(ticker, start, end)
    if df.empty or len(df) < cfg.sma_long + 5:
        return None

    tech = technicals(df, cfg)
    entry_price = _to_float(tech.get("entry_price"))
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    stop_price = round(entry_price * (1 - cfg.stop_loss_pct), 2)
    target_price = round(entry_price * (1 + cfg.target_gain_pct), 2)

    per_trade_risk = cfg.portfolio_value * cfg.risk_per_trade_pct
    stop_distance = entry_price - stop_price
    size_by_risk = (per_trade_risk / stop_distance) if stop_distance > 0 else 0.0

    max_margin_dollars = cfg.portfolio_value * cfg.margin_ltv_cap
    alloc_cap = max_margin_dollars / max(1, cfg.max_open_margin_positions)
    shares_by_alloc = alloc_cap / entry_price
    shares_suggested = int(max(0, min(size_by_risk, shares_by_alloc)))

    sn = get_news_and_sentiment(ticker)
    fn = get_fundamentals(ticker)

    sentiment_val = sn.get("sentiment")
    sentiment_score = 0.0
    if isinstance(sentiment_val, (int, float)) and np.isfinite(sentiment_val):
        sentiment_score = 1.0 + float(sentiment_val)

    catalyst_score = 1.0 if sn.get("catalysts") else 0.0

    fundamentals_score = 0.0
    pe = fn.get("pe_ttm")
    fpe = fn.get("pe_fwd")
    if isinstance(pe, (int, float)) and np.isfinite(pe) and 4 <= pe <= 35:
        fundamentals_score += 0.5
    if isinstance(fpe, (int, float)) and np.isfinite(fpe) and 4 <= fpe <= 30:
        fundamentals_score += 0.5
    dy = fn.get("div_yield_pct")
    if isinstance(dy, (int, float)) and np.isfinite(dy) and dy >= 3:
        fundamentals_score += 0.25
    de = fn.get("debt_to_equity")
    if isinstance(de, (int, float)) and np.isfinite(de) and de <= 2.0:
        fundamentals_score += 0.25

    composite = float(tech["signals_score"]) + sentiment_score + catalyst_score + fundamentals_score

    return {
        "ticker": ticker,
        "date": dt.date.today().isoformat(),
        "last": round(entry_price, 2),
        "rsi": tech["rsi"],
        "macd": tech["macd"],
        "macd_signal": tech["macd_signal"],
        "sma200": tech["sma200"],
        "sma_dev_pct": tech["sma_dev_pct"],
        "vol": tech["vol"],
        "vol_avg20": tech["vol_avg20"],
        "rsi_ok": tech["rsi_ok"],
        "macd_cross_ok": tech["macd_cross_ok"],
        "sma_dev_ok": tech["sma_dev_ok"],
        "vol_spike_ok": tech["vol_spike_ok"],
        "signals_score": int(tech["signals_score"]),
        "stop_price": stop_price,
        "target_price": target_price,
        "trailing_trigger_gain": cfg.trailing_trigger_gain,
        "trailing_pct": cfg.trailing_pct,
        "max_hold_days": cfg.max_hold_days,
        "suggested_shares": shares_suggested,
        "risk_dollars": round(per_trade_risk, 2),
        "alloc_cap_dollars": round(alloc_cap, 2),
        "headline_sentiment": round(float(sentiment_val), 3) if isinstance(sentiment_val, (int, float)) and np.isfinite(sentiment_val) else None,
        "catalysts": sn.get("catalysts", ""),
        "pe_ttm": fn.get("pe_ttm"),
        "pe_fwd": fn.get("pe_fwd"),
        "div_yield_pct": fn.get("div_yield_pct"),
        "payout_ratio": fn.get("payout_ratio"),
        "debt_to_equity": fn.get("debt_to_equity"),
        "composite_score": round(composite, 2),
        "close_series": tech["close_series"],
        "sma200_series": tech["sma200_series"],
        "dates_series": tech["dates_series"],
    }


def run_scan(universe_file: str, holdings_file: str, cfg: ScannerConfig, out_csv: str = "data/weekly_margin_report.csv"):
    if cfg.vix_filter_enabled and not market_risk_ok(cfg.vix_threshold, True):
        print(f"Market risk elevated (VIX > {cfg.vix_threshold}). Skipping new entries.")
        return None

    universe = read_list(universe_file)
    holdings = set(read_list(holdings_file))
    if not universe:
        print("No tickers found in", universe_file)
        return None

    records = []
    for t in universe:
        try:
            rec = analyze_ticker(t, cfg)
            if rec is not None:
                rec["owned"] = (t in holdings)
                if rec["owned"]:
                    rec["signals_score"] += cfg.holdings_boost_score
                    rec["composite_score"] = round(rec["composite_score"] + cfg.holdings_boost_score, 2)
                records.append(rec)
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not records:
        print("No candidates.")
        return None

    df = pd.DataFrame(records).sort_values(
        by=["composite_score", "signals_score", "rsi"],
        ascending=[False, False, True]
    )
    df_out = df.drop(columns=["close_series", "sma200_series", "dates_series"], errors="ignore")
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df_out)} candidates.")
    return df
