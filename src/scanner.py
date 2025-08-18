import os
import json
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Union

import numpy as np
import pandas as pd
import logging
from logging import LoggerAdapter

from collections import OrderedDict

from .config import ScannerConfig
from .utils import read_list, fetch_history, market_risk_ok
from .indicators import (
    rsi, macd, bullish_macd_cross,
    ema, atr, donchian, swing_low, bollinger
)
from .news_fundamentals import get_news_and_sentiment, get_fundamentals

SECTION_ORDER = [
    "meta",
    "price_context",
    "trend",
    "momentum",
    "volume",
    "bands_breakouts",
    "stops_and_risk",
    "signals_and_scores",
    "position_sizing",
    "fundamentals_and_sentiment",
    "series",
]

# ---------------- logging ----------------

def setup_logging():
    """Idempotent logging setup with ticker-aware adapter."""
    if getattr(setup_logging, "_configured", False):
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = (
            '%(asctime)s %(levelname)s '
            '[%(name)s] '
            '%(message)s'
        )
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(level)
    setup_logging._configured = True

setup_logging()
_logger = logging.getLogger(__name__)

class TickerLogger(LoggerAdapter):
    def process(self, msg, kwargs):
        prefix = ""
        if self.extra and self.extra.get("ticker"):
            prefix = f"[{self.extra['ticker']}] "
        return prefix + str(msg), kwargs

def get_log(ticker: Optional[str] = None) -> TickerLogger:
    return TickerLogger(_logger, {"ticker": ticker})

def log_value(log: TickerLogger, name: str, val, level=logging.DEBUG, sample_n: int = 3):
    """Log type/shape/preview for debugging mysterious values."""
    info = {"name": name, "type": type(val).__name__}
    try:
        if isinstance(val, pd.DataFrame):
            info["shape"] = list(val.shape)
            info["columns"] = list(map(str, val.columns[:10]))
        elif isinstance(val, pd.Series):
            info["shape"] = [len(val)]
            info["head"] = val.head(sample_n).tolist()
        elif isinstance(val, np.ndarray):
            info["shape"] = list(val.shape)
            info["sample"] = val.ravel()[:sample_n].tolist()
        else:
            info["value"] = val if isinstance(val, (int, float, str, bool, type(None))) else repr(val)
    except Exception as e:
        info["error"] = f"failed_introspect: {e}"
    log.log(level, "VALUE %s", json.dumps(info, default=str))

# ---------- helpers ----------

def _to_float(x) -> float:
    """Return a scalar float from pandas/NumPy without FutureWarnings."""
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return float("nan")
        x = x.iloc[-1]

    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return float("nan")
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return float(x.item())
        if len(x) == 1:
            return float(x[0])
        x = x[-1]

    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float("nan")


def safe_round(x, ndigits=2):
    """Round but tolerate numpy arrays/series."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return round(float(x.item()), ndigits)
        if x.size == 0:
            return None
        xr = np.round(x.astype(float), ndigits)
        return float(xr.ravel()[-1])
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return None
        return safe_round(x.iloc[-1], ndigits)
    try:
        return round(float(x), ndigits)
    except Exception:
        return None

def _to_1d_list(obj):
    if isinstance(obj, pd.Series):
        return obj.to_numpy().ravel().tolist()
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].to_numpy().ravel().tolist()
        return obj.to_numpy().ravel().tolist()
    if isinstance(obj, np.ndarray):
        return obj.ravel().tolist()
    try:
        return list(obj)
    except Exception:
        return []

def _pct_slope(series: pd.Series, window: int = 50):
    s = series.dropna()
    if len(s) < 2:
        return None
    y = s.iloc[-window:].to_numpy()
    x = np.arange(len(y))
    m = np.polyfit(x, y, 1)[0]
    if y[-1] == 0 or not np.isfinite(y[-1]):
        return None
    return (m / y[-1]) * 100.0

def _iter_file_lines(paths: Iterable[Union[str, Path]]) -> Iterable[str]:
    """Yield lines from one or more files. Ignores missing files with a warning."""
    for p in paths:
        p = Path(p)
        if not p.exists():
            get_log().warning("File not found (skipping): %s", p)
            continue
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

def _parse_list_lines(lines: Iterable[str]) -> List[str]:
    """
    Parse lines into items:
      - trims whitespace
      - removes full-line comments (# ... )
      - removes inline comments (e.g. AAPL  # watchlist)
      - skips empty lines
      - preserves order while dropping duplicates
    """
    items = []
    seen = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        # strip inline comments
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # optional: allow comma-separated on a single line
        for tok in [t.strip() for t in line.split(",")]:
            if not tok:
                continue
            # normalize tickers (optional): upper-case
            tok_norm = tok.upper()
            if tok_norm not in seen:
                seen.add(tok_norm)
                items.append(tok_norm)
    return items

def read_merged_list(files: Union[str, Path, Iterable[Union[str, Path]]]) -> List[str]:
    """
    Read and coalesce one or many files into a single ordered list.
    Accepts a single path or an iterable of paths.
    """
    if isinstance(files, (str, Path)):
        files = [files]
    return _parse_list_lines(_iter_file_lines(files))

# ---------- builders ----------

def build_structured_record(
    ticker, entry_price, stop_price, target_price, risk_reward,
    per_trade_risk, alloc_cap, shares_suggested, cfg,
    sn, fn, tech, composite, sentiment_val
):
    log = get_log(ticker)
    # Basic context
    log.debug("Building structured record")
    log_value(log, "entry_price", entry_price)
    log_value(log, "stop_price", stop_price)
    log_value(log, "target_price", target_price)
    log_value(log, "risk_reward", risk_reward)
    log_value(log, "composite", composite)

    def _num(x):
        return isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)

    de  = fn.get("debt_to_equity")
    peg = fn.get("peg_ratio")
    rg  = fn.get("revenue_growth_qoq")
    eg  = fn.get("eps_growth_qoq")
    npm = fn.get("net_profit_margin")
    fcfm= fn.get("fcf_margin")

    fundamentals_analysis = OrderedDict({
        "de_ok":        (_num(de)  and de  <= 2.0) or None,
        "peg_ok":       (_num(peg) and 0.5 <= peg <= 2.0) or None,
        "rev_growth_pos": (_num(rg) and rg > 0) or None,
        "rev_growth_strong": (_num(rg) and rg > 5) or None,
        "eps_growth_pos": (_num(eg) and eg > 0) or None,
        "eps_growth_strong": (_num(eg) and eg > 5) or None,
        "net_margin_healthy": (_num(npm) and npm > 10) or None,
        "fcf_margin_healthy": (_num(fcfm) and fcfm > 5) or None,
    })

    rec = {
        "meta": OrderedDict({
            "ticker": ticker,
            "date": dt.date.today().isoformat(),
            "last": safe_round(_to_float(entry_price), 2),
        }),
        "price_context": OrderedDict({
            "pct_from_52w_high": tech["pct_from_52w_high"],
            "pct_to_52w_low": tech["pct_to_52w_low"],
            "sma_dev_pct": tech["sma_dev_pct"],
        }),
        "trend": OrderedDict({
            "sma200": tech["sma200"],
            "sma200_slope_pct":tech["sma200_slope_pct"],
            "sma200_trending_up": tech["sma200_trending_up"],
            "sma50": tech["sma50"],
            "sma20": tech["sma20"],
            "ema_fast": tech["ema_fast"],
            "ema_slow": tech["ema_slow"],
            "ema_fast_cross_up": tech["ema_fast_cross_up"],
            "ema50_reclaim": tech["ema50_reclaim"],
            "above_boll_mid": tech["above_boll_mid"],
        }),
        "momentum": OrderedDict({
            "rsi": tech["rsi"],
            "rsi_ok": tech["rsi_ok"],
            "rsi_hook_up": tech["rsi_hook_up"],
            "macd": tech["macd"],
            "macd_signal": tech["macd_signal"],
            "macd_cross_ok": tech["macd_cross_ok"],
            "gap_up": tech["gap_up"],
        }),
        "volume": OrderedDict({
            "vol": tech["vol"],
            "vol_avg20": tech["vol_avg20"],
            "vol_spike_ok": tech["vol_spike_ok"],
        }),
        "bands_breakouts": OrderedDict({
            "bb_ma": tech["bb_ma"],
            "bb_up": tech["bb_up"],
            "bb_lo": tech["bb_lo"],
            "bb_percent_b": tech["bb_percent_b"],
            "bb_bandwidth": tech["bb_bandwidth"],
            "donchian_hi": tech["donchian_hi"],
            "donchian_lo": tech["donchian_lo"],
            "donchian_breakout": tech["donchian_breakout"],
        }),
        "stops_and_risk": OrderedDict({
            "atr": tech["atr"],
            "atr_stop": tech["atr_stop"],
            "swing_low": tech["swing_low"],
            "band_stop": tech["band_stop"],
            "pct_stop": tech["pct_stop"],
            "recommended_stop": tech["recommended_stop"],
            "recommended_stop_basis": tech["recommended_stop_basis"],
            "planned_stop_price": stop_price,
            "planned_target_price": target_price,
            "trailing_trigger_gain": cfg.trailing_trigger_gain,
            "trailing_pct": cfg.trailing_pct,
            "max_hold_days": cfg.max_hold_days,
            "risk_reward": _to_float(risk_reward) if risk_reward is not None else None,
        }),
        "signals_and_scores": OrderedDict({
            "signals_score": int(tech["signals_score"]),
            "composite_score": safe_round(_to_float(composite), 2),
        }),
        "position_sizing": OrderedDict({
            "suggested_shares": int(shares_suggested),
            "risk_dollars": safe_round(_to_float(per_trade_risk), 2),
            "alloc_cap_dollars": safe_round(_to_float(alloc_cap), 2),
        }),
        "fundamentals_and_sentiment": OrderedDict({
            "headline_sentiment": (
                safe_round(_to_float(sentiment_val), 3)
                if isinstance(sentiment_val, (int, float, np.floating, np.integer)) and np.isfinite(sentiment_val)
                else None
            ),
            "catalysts": sn.get("catalysts", ""),
            "pe_ttm": fn.get("pe_ttm"),
            "pe_fwd": fn.get("pe_fwd"),
            "div_yield_pct": fn.get("div_yield_pct"),
            "payout_ratio": fn.get("payout_ratio"),
            "debt_to_equity": fn.get("debt_to_equity"),
            "revenue_growth_qoq": fn.get("revenue_growth_qoq"),
            "eps_growth_qoq": fn.get("eps_growth_qoq"),
            "net_profit_margin": fn.get("net_profit_margin"),
            "fcf_margin": fn.get("fcf_margin"),
            "peg_ratio": fn.get("peg_ratio"),
            "analysis": fundamentals_analysis,
        }),
        "series": OrderedDict({
            "close_series": _to_1d_list(tech.get("close_series", [])),
            "sma200_series": _to_1d_list(tech.get("sma200_series", [])),
            "dates_series": tech.get("dates_series", []),
        }),
    }
    log.debug("Structured record built")
    return OrderedDict((sec, rec[sec]) for sec in SECTION_ORDER if sec in rec)

def flatten_sections(structured: dict, sep="__") -> dict:
    return OrderedDict(
        (f"{section}{sep}{k}", v)
        for section in SECTION_ORDER
        if section in structured
        for k, v in structured[section].items()
    )

# ---------- main ----------

def technicals(df: pd.DataFrame, cfg: ScannerConfig) -> Dict[str, Any]:
    log = get_log(getattr(cfg, "current_ticker", None))
    log.info("Computing technicals")
    try:
        log_value(log, "df.info", {"shape": list(df.shape), "cols": list(map(str, df.columns))}, level=logging.INFO)

        # Ensure "Close" is a Series, not a DataFrame
        close_obj = df["Close"]
        if isinstance(close_obj, pd.DataFrame):
            log.warning('"Close" is a DataFrame; taking first column')
            close = close_obj.iloc[:, 0]
        else:
            close = close_obj

        # Ensure "AdjClose" column naming is consistent
        if "AdjClose" in df:
            adj_close = df["AdjClose"]
        elif "Adj Close" in df:
            log.warning('Renaming "Adj Close" -> "AdjClose"')
            df = df.rename(columns={"Adj Close": "AdjClose"})
            adj_close = df["AdjClose"]
        else:
            log.warning('Missing AdjClose; falling back to Close for adj_close')
            adj_close = close

        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) >= 2 else latest

        # Core indicators
        rsi_series = rsi(close, cfg.rsi_period)
        macd_line, signal_line, _ = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        sma200_series = close.rolling(cfg.sma_long, min_periods=1).mean()
        vol_avg20_series = df["Volume"].rolling(cfg.vol_avg_period).mean()

        # Additional indicators
        sma20_series = close.rolling(20).mean()
        sma50_series = close.rolling(getattr(cfg, "sma_mid", 50)).mean()
        ema_fast_series = ema(close, getattr(cfg, "ema_fast_span", 21))
        ema_slow_series = ema(close, getattr(cfg, "ema_slow_span", 50))
        atr_series = atr(df, period=getattr(cfg, "atr_period", 14))
        donch_hi, donch_lo = donchian(df, lookback=getattr(cfg, "donchian_lookback", 20))
        bb_ma, bb_up, bb_lo, bb_pb, bb_bw = bollinger(close, period=getattr(cfg, "boll_period", 20), num_std=getattr(cfg, "boll_std", 2.0))

        # Scalars
        last_price = _to_float(latest.get("AdjClose", latest.get("Close")))
        prev_price = _to_float(prev.get("AdjClose", prev.get("Close")))
        volume = _to_float(latest["Volume"])
        vol_avg20 = _to_float(vol_avg20_series.iloc[-1])
        open_price = _to_float(df.iloc[-1]["Open"])

        rsi_val = _to_float(rsi_series.iloc[-1])
        rsi_prev = _to_float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else float("nan")

        macd_val = _to_float(macd_line.iloc[-1])
        macd_sig = _to_float(signal_line.iloc[-1])

        sma200 = _to_float(sma200_series.iloc[-1])
        sma200_slope_pct = _pct_slope(sma200_series, window=50)
        sma200_trending_up = (sma200_slope_pct is not None) and (sma200_slope_pct > 0)
        sma50 = _to_float(sma50_series.iloc[-1])
        sma20 = _to_float(sma20_series.iloc[-1])

        ema_fast = _to_float(ema_fast_series.iloc[-1])
        ema_slow = _to_float(ema_slow_series.iloc[-1])

        atr_val = _to_float(atr_series.iloc[-1])
        atr_pct = (atr_val / last_price) * 100.0 if np.isfinite(atr_val) and np.isfinite(last_price) and last_price > 0 else None
        donch_hi_v = _to_float(donch_hi.iloc[-1])
        donch_lo_v = _to_float(donch_lo.iloc[-1])
        donch_hi_prev = _to_float(donch_hi.shift(1).iloc[-1])
        donch_lo_prev = _to_float(donch_lo.shift(1).iloc[-1])

        bb_ma_v = _to_float(bb_ma.iloc[-1])
        bb_up_v = _to_float(bb_up.iloc[-1])
        bb_lo_v = _to_float(bb_lo.iloc[-1])
        bb_pb_v = _to_float(bb_pb.iloc[-1])
        bb_bw_v = _to_float(bb_bw.iloc[-1])

        # 52-week
        hi_52_ser = df["High"].rolling(252, min_periods=252).max()
        lo_52_ser = df["Low"].rolling(252, min_periods=252).min()
        hi_52 = _to_float(hi_52_ser.iloc[-1])
        lo_52 = _to_float(lo_52_ser.iloc[-1])
        pct_from_52w_high = (last_price / hi_52 - 1.0) * 100 if np.isfinite(last_price) and np.isfinite(hi_52) and hi_52 > 0 else None
        pct_to_52w_low = (last_price / lo_52 - 1.0) * 100 if np.isfinite(last_price) and np.isfinite(lo_52) and lo_52 > 0 else None

        # Entry gates
        rsi_ok = (np.isfinite(rsi_val) and rsi_val <= getattr(cfg, "rsi_oversold", 35))
        macd_cross_ok = bullish_macd_cross(macd_line, signal_line, getattr(cfg, "macd_cross_lookback", 5))

        sma_dev_pct = None
        sma_ok = False
        if np.isfinite(last_price) and np.isfinite(sma200) and sma200 != 0:
            sma_dev = (last_price - sma200) / sma200
            sma_dev_pct = safe_round(100.0 * sma_dev, 2)
            sma_ok = (sma_dev <= -getattr(cfg, "sma_deviation_req", 0.10))

        up_day = (np.isfinite(prev_price) and np.isfinite(last_price) and last_price > prev_price)
        vol_spike_ok = False
        if up_day and np.isfinite(volume) and np.isfinite(vol_avg20) and vol_avg20 > 0:
            vol_spike_ok = (volume >= getattr(cfg, "volume_spike_mult", 1.5) * vol_avg20)

        # Context
        ema_fast_cross_up = (np.isfinite(last_price) and np.isfinite(ema_fast) and last_price > ema_fast)
        ema50_reclaim = (np.isfinite(last_price) and np.isfinite(ema_slow) and last_price > ema_slow)
        above_boll_mid = (np.isfinite(last_price) and np.isfinite(bb_ma_v) and last_price >= bb_ma_v)
        donchian_breakout = (np.isfinite(last_price) and np.isfinite(donch_hi_prev) and last_price >= donch_hi_prev)
        rsi_hook_up = (np.isfinite(rsi_val) and np.isfinite(rsi_prev) and rsi_prev <= getattr(cfg, "rsi_oversold", 35) and rsi_val > rsi_prev)

        # Gap up today?
        gap_up = (np.isfinite(prev_price) and np.isfinite(open_price) and open_price > prev_price)

        atr_stop = last_price - getattr(cfg, "atr_multiple", 1.5) * atr_val if np.isfinite(last_price) and np.isfinite(atr_val) else float("nan")
        swing_lv = swing_low(df, left=2, right=2)
        swing_stop = swing_lv if np.isfinite(swing_lv) else float("nan")
        band_stop = bb_lo_v if np.isfinite(bb_lo_v) else float("nan")
        pct_stop = last_price * (1 - getattr(cfg, "stop_loss_pct", 0.1)) if np.isfinite(last_price) else float("nan")

        candidates = [s for s in (atr_stop, swing_stop, band_stop, pct_stop) if np.isfinite(s) and s < last_price]
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

        # Loud diagnostics on tricky values
        for nm, vv in [
            ("last_price", last_price),
            ("sma200", sma200),
            ("rsi_val", rsi_val),
            ("macd_val", macd_val),
            ("atr_val", atr_val),
            ("bb_ma_v", bb_ma_v),
            ("donch_hi_v", donch_hi_v),
        ]:
            log_value(log, nm, vv)

        result = {
            # core metrics
            "rsi": safe_round(rsi_val, 2) if np.isfinite(rsi_val) else None,
            "macd": safe_round(macd_val, 4) if np.isfinite(macd_val) else None,
            "macd_signal": safe_round(macd_sig, 4) if np.isfinite(macd_sig) else None,
            "sma200": safe_round(sma200, 2) if np.isfinite(sma200) else None,
            "sma50": safe_round(sma50, 2) if np.isfinite(sma50) else None,
            "sma20": safe_round(sma20, 2) if np.isfinite(sma20) else None,
            "sma200_slope_pct": safe_round(sma200_slope_pct, 4) if sma200_slope_pct is not None else None,
            "sma200_trending_up": bool(sma200_trending_up),
            "ema_fast": safe_round(ema_fast, 2) if np.isfinite(ema_fast) else None,
            "ema_slow": safe_round(ema_slow, 2) if np.isfinite(ema_slow) else None,

            # distances
            "sma_dev_pct": sma_dev_pct,
            "pct_from_52w_high": safe_round(pct_from_52w_high, 2) if pct_from_52w_high is not None else None,
            "pct_to_52w_low": safe_round(pct_to_52w_low, 2) if pct_to_52w_low is not None else None,

            # volume, bands, donchian
            "vol": int(volume) if np.isfinite(volume) else None,
            "vol_avg20": int(vol_avg20) if np.isfinite(vol_avg20) else None,
            "bb_ma": safe_round(bb_ma_v, 2) if np.isfinite(bb_ma_v) else None,
            "bb_up": safe_round(bb_up_v, 2) if np.isfinite(bb_up_v) else None,
            "bb_lo": safe_round(bb_lo_v, 2) if np.isfinite(bb_lo_v) else None,
            "bb_percent_b": safe_round(bb_pb_v, 3) if np.isfinite(bb_pb_v) else None,
            "bb_bandwidth": safe_round(bb_bw_v, 3) if np.isfinite(bb_bw_v) else None,
            "donchian_hi": safe_round(donch_hi_v, 2) if np.isfinite(donch_hi_v) else None,
            "donchian_lo": safe_round(donch_lo_v, 2) if np.isfinite(donch_lo_v) else None,

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
            "atr": safe_round(atr_val, 3) if np.isfinite(atr_val) else None,
            "atr_stop": safe_round(atr_stop, 2) if np.isfinite(atr_stop) else None,
            "atr_pct": safe_round(atr_pct, 2) if atr_pct is not None else None,
            "swing_low": safe_round(swing_lv, 2) if np.isfinite(swing_lv) else None,
            "band_stop": safe_round(band_stop, 2) if np.isfinite(band_stop) else None,
            "pct_stop": safe_round(pct_stop, 2) if np.isfinite(pct_stop) else None,
            "recommended_stop": safe_round(recommended_stop, 2) if recommended_stop is not None else None,
            "recommended_stop_basis": stop_basis,

            # score & series
            "signals_score": int(score),
            "entry_price": last_price if np.isfinite(last_price) else None,

            # series (safe -> lists)
            "close_series": _to_1d_list(close.tail(260)),
            "sma200_series": _to_1d_list(sma200_series.tail(260)),
            "dates_series": [d.strftime("%Y-%m-%d") for d in df.tail(260).index],
        }

        # Final sanity logs to catch ndarray/DF leaks before returning
        for nm in ["close_series", "sma200_series", "dates_series"]:
            log_value(log, nm, result[nm])

        log.info("Technicals computed OK")
        return result

    except Exception:
        log.exception("technicals() failed")
        raise

def analyze_ticker(ticker: str, cfg: ScannerConfig) -> Optional[Dict[str, Any]]:
    cfg.current_ticker = ticker  # for logs inside technicals()
    log = get_log(ticker)
    log.info("Analyze ticker start")
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=cfg.lookback_days)
        df = fetch_history(ticker, start, end)
        if df is None or df.empty or len(df) < cfg.sma_long + 5:
            log.warning("No data or insufficient rows: empty=%s len=%s", getattr(df, "empty", None), len(df) if df is not None else None)
            return None

        log_value(log, "history_df", {"shape": list(df.shape), "cols": list(map(str, df.columns))}, level=logging.INFO)

        tech = technicals(df, cfg)
        log_value(log, "tech_signals_score", tech.get("signals_score"), level=logging.INFO)
        entry_price = _to_float(tech.get("entry_price"))
        log_value(log, "entry_price", entry_price, level=logging.INFO)
        if not np.isfinite(entry_price) or entry_price <= 0:
            log.warning("Invalid entry_price")
            return None

        stop_price = safe_round(entry_price * (1 - cfg.stop_loss_pct), 2)
        target_price = safe_round(entry_price * (1 + cfg.target_gain_pct), 2)

        risk = entry_price - stop_price if (np.isfinite(entry_price) and np.isfinite(stop_price)) else float("nan")
        reward = target_price - entry_price if (np.isfinite(target_price) and np.isfinite(entry_price)) else float("nan")
        rr = (reward / risk) if (np.isfinite(reward) and np.isfinite(risk) and risk > 0) else None

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
        if isinstance(sentiment_val, (int, float, np.floating, np.integer)) and np.isfinite(sentiment_val):
            sentiment_score = 1.0 + _to_float(sentiment_val)

        catalyst_score = 1.0 if sn.get("catalysts") else 0.0

        fundamentals_score = 0.0
        def _bump(x, lo, hi, pts):
            return pts if isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x) and lo <= x <= hi else 0.0

        fundamentals_score += _bump(fn.get("pe_ttm"), 4, 35, 0.5)
        fundamentals_score += _bump(fn.get("pe_fwd"), 4, 30, 0.5)
        dy = fn.get("div_yield_pct")
        if isinstance(dy, (int, float, np.floating, np.integer)) and np.isfinite(dy) and dy >= 3:
            fundamentals_score += 0.25
        de = fn.get("debt_to_equity")
        if isinstance(de, (int, float, np.floating, np.integer)) and np.isfinite(de) and de <= 2.0:
            fundamentals_score += 0.25
        peg = fn.get("peg_ratio")
        if isinstance(peg, (int, float, np.floating, np.integer)) and np.isfinite(peg) and 0.5 <= peg <= 2.0:
            fundamentals_score += 0.5
        rg = fn.get("revenue_growth_qoq")
        if isinstance(rg, (int, float, np.floating, np.integer)) and np.isfinite(rg) and rg > 5:
            fundamentals_score += 0.25
        eg = fn.get("eps_growth_qoq")
        if isinstance(eg, (int, float, np.floating, np.integer)) and np.isfinite(eg) and eg > 5:
            fundamentals_score += 0.25
        npm = fn.get("net_profit_margin")
        if isinstance(npm, (int, float, np.floating, np.integer)) and np.isfinite(npm) and npm > 10:
            fundamentals_score += 0.25
        fcfm = fn.get("fcf_margin")
        if isinstance(fcfm, (int, float, np.floating, np.integer)) and np.isfinite(fcfm) and fcfm > 5:
            fundamentals_score += 0.25

        composite = _to_float(tech["signals_score"]) + sentiment_score + catalyst_score + fundamentals_score
        log_value(log, "composite", composite, level=logging.INFO)

        structured = build_structured_record(
            ticker=ticker,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=rr,
            per_trade_risk=per_trade_risk,
            alloc_cap=alloc_cap,
            shares_suggested=shares_suggested,
            cfg=cfg,
            sn=sn,
            fn=fn,
            tech=tech,
            composite=composite,
            sentiment_val=sentiment_val,
        )

        flat = flatten_sections(structured, sep="__")
        log.info("Analyze ticker done")
        return flat

    except Exception:
        log.exception("analyze_ticker() failed")
        raise

def run_scan(
    universe_files: Union[str, Path, Iterable[Union[str, Path]]],
    holdings_files: Union[str, Path, Iterable[Union[str, Path]]],
    cfg: ScannerConfig,
    out_csv: str = "data/weekly_margin_report.csv",
):
    log = get_log()
    log.info("Run scan start")
    try:
        if cfg.vix_filter_enabled and not market_risk_ok(cfg.vix_threshold, True):
            log.warning("Market risk elevated (VIX > %s). Skipping new entries.", cfg.vix_threshold)
            return None

        universe = read_merged_list(universe_files)
        holdings = set(read_merged_list(holdings_files))

        if not universe:
            log.warning("No tickers found in Ticker Lists%s", universe_files)
            return None
        
        if not holdings:
            log.warning("No tickers found in Holdings %s", holdings_files)
            return None

        records = []
        for t in universe:
            tlog = get_log(t)
            try:
                rec = analyze_ticker(t, cfg)
                if rec is None:
                    tlog.info("No candidate")
                    continue

                rec["owned"] = (t in holdings)

                sig_keys = ["signals_score", "signals_and_scores__signals_score"]
                comp_keys = ["composite_score", "signals_and_scores__composite_score"]

                if rec["owned"]:
                    for k in sig_keys:
                        if k in rec and rec[k] is not None:
                            rec[k] += cfg.holdings_boost_score
                            break
                    for k in comp_keys:
                        if k in rec and rec[k] is not None:
                            rec[k] = safe_round(_to_float(rec[k]) + cfg.holdings_boost_score, 2)
                            break

                records.append(rec)
                tlog.info("Added candidate")
            except Exception:
                tlog.exception("Failed processing ticker")
                # continue to next ticker

        if not records:
            log.info("No candidates.")
            return None

        df = pd.DataFrame(records)

        def first_present(cols):
            for c in cols:
                if c in df.columns:
                    return c
            return None

        comp_col = first_present(["composite_score", "signals_and_scores__composite_score"])
        sig_col  = first_present(["signals_score", "signals_and_scores__signals_score"])
        rsi_col  = first_present(["rsi", "momentum__rsi"])

        sort_cols = [c for c in [comp_col, sig_col, rsi_col] if c]
        sort_asc  = [False, False, True][:len(sort_cols)]

        if sort_cols:
            log.info("Sorting by %s", sort_cols)
            df = df.sort_values(by=sort_cols, ascending=sort_asc)

        series_cols = [c for c in df.columns if c.startswith("series__")]
        # series_cols += ["close_series", "sma200_series", "dates_series"]
        # df_out = df.drop(columns=[c for c in series_cols if c in df.columns], errors="ignore")

        df.to_csv(out_csv, index=False)
        log.info("Wrote %s with %d candidates.", out_csv, len(df))
        return df

    except Exception:
        log.exception("run_scan() failed")
        raise
