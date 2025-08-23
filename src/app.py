# app.py
from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================
# Constants / paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # project-root/
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "weekly_margin_report.csv"

# Make console output UTF-8 friendly (best effort)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# =========================
# Profiles (save/load filters)
# =========================
PROFILES_DIR = PROJECT_ROOT / ".app_state"
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

def save_profile(name: str, data: dict):
    fp = PROFILES_DIR / f"{name}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_profile(name: str) -> Optional[dict]:
    fp = PROFILES_DIR / f"{name}.json"
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def list_profiles() -> list[str]:
    return sorted([p.stem for p in PROFILES_DIR.glob("*.json")])

# ===== Plot Theme =====
THEME_FILE = PROFILES_DIR / "plot_theme.json"
DEFAULT_THEME = {
    "close":   "#1f77b4",
    "sma200":  "#2ca02c",
    "overlay": "#9467bd",
    "stop":    "#d62728",
    "target":  "#17becf",
    "proj_mid": "#ff7f0e",
    "risk_band": "#8dd3c7",
    "proj_band": "#ffbb78",
}

def load_theme() -> dict:
    if THEME_FILE.exists():
        try:
            return json.load(open(THEME_FILE, "r", encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_THEME.copy()

def save_theme(theme: dict):
    try:
        json.dump(theme, open(THEME_FILE, "w", encoding="utf-8"), indent=2)
    except Exception:
        pass

# one in-memory copy for this session; survives chart switches
if "plot_theme" not in st.session_state:
    st.session_state.plot_theme = load_theme()

th = st.session_state.plot_theme

# =========================
# Technical overlay registry
# =========================
@dataclass
class OverlayResult:
    traces: List[go.Scatter] | None = None
    shapes: List[dict] | None = None
    annotations: List[dict] | None = None

OverlayFn = Callable[[pd.Series, pd.Series, pd.Series, pd.Series, dict], OverlayResult]
# signature: (x_datetime, close, sma200, full_row, params) -> OverlayResult

def _sma(y: pd.Series, window: int) -> pd.Series:
    return pd.Series(y, index=range(len(y))).rolling(window, min_periods=1).mean()

def _ema(y: pd.Series, span: int) -> pd.Series:
    return pd.Series(y, index=range(len(y))).ewm(span=span, adjust=False).mean()

def ov_sma(x, close, sma200, row, params) -> OverlayResult:
    w = int(params.get("window", 20))
    y = _sma(close, w)
    return OverlayResult(traces=[go.Scatter(x=x, y=y, mode="lines", name=f"SMA{w}")])

def ov_ema(x, close, sma200, row, params) -> OverlayResult:
    span = int(params.get("span", 21))
    y = _ema(close, span)
    return OverlayResult(traces=[go.Scatter(x=x, y=y, mode="lines", name=f"EMA{span}")])

def ov_bbands(x, close, sma200, row, params) -> OverlayResult:
    w = int(params.get("window", 20))
    k = float(params.get("std", 2.0))
    s = pd.Series(close, index=range(len(close)))
    ma = s.rolling(w, min_periods=1).mean()
    sd = s.rolling(w, min_periods=1).std(ddof=0)
    upper, lower = ma + k * sd, ma - k * sd
    return OverlayResult(traces=[
        go.Scatter(x=x, y=ma,    mode="lines", name=f"BB MA{w}"),
        go.Scatter(x=x, y=upper, mode="lines", name=f"BB Upper ({k}σ)"),
        go.Scatter(x=x, y=lower, mode="lines", name=f"BB Lower ({k}σ)"),
    ])

# Name -> {fn, default params, schema for UI (type/hint/range)}
TECHNICALS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "SMA":    {"fn": ov_sma,    "params": {"window": 20},        "schema": {"window": ("int", 2, 400)}},
    "EMA":    {"fn": ov_ema,    "params": {"span": 21},          "schema": {"span": ("int", 2, 400)}},
    "BBands": {"fn": ov_bbands, "params": {"window": 20, "std": 2.0}, "schema": {"window": ("int", 2, 400), "std": ("float", 0.5, 4.0)}},
}

# =========================
# Helpers
# =========================
# Specific per-control help (multi-line tooltips). Hover the "?" next to each widget.
HELP = {
    # ===== BUY: thresholds & knobs =====
    "composite_threshold":
        "BUY will only trigger if the weighted composite ≥ this value.\n"
        "Engine: comp = Σ(weight_i × score_i) + gap_bonus.\n"
        "Higher = pickier/fewer buys; Lower = more frequent buys.",

    "rsi_buy_max":
        "RSI(14, Wilder). BUY score uses s_rsi = sigmoid((rsi_buy_max − RSI)/5).\n"
        "Lower RSI → higher BUY score until this cap. Raise to accept higher-RSI entries.",

    "vol_ratio_min":
        "Volume spike filter using vol/vol_avg20.\n"
        "BUY flow score ≈ sigmoid((vol_ratio − vol_ratio_min)/0.4).\n"
        "Higher threshold = only strong accumulation days contribute.",

    "donch_lookback":
        "Donchian lookback N used for prior high in BUY and prior low in SELL (yesterday’s channel).\n"
        "BUY breakout strength uses (last − prev_N_high)/ATR.",

    "gap_min_pct":
        "Extra BUY bonus if today’s open > yesterday’s high by ≥ this %.\n"
        "Used to reward momentum breakouts. Set 0 to disable.",

    "atr_mult":
        "Engine stop = last − ATR(14) × multiple (when ATR is available).\n"
        "Higher = wider stop (lower position size, more room).",

    "stop_pct_ui":
        "Fallback stop % if ATR is unavailable: stop = last × (1 − %/100).\n"
        "Only used when ATR cannot be computed.",

    "value_center":
        "Value component compares last vs SMA200.\n"
        "Centered at this % discount to SMA200 (negative favors buys below SMA200).",

    "sma_window":
        "Trend slopes window (days). Price & SMA200 slopes are normalized and fed through a sigmoid.\n"
        "Bigger = smoother/slower trend; smaller = snappier/more reactive.",

    "use_engine_stop":
        "If ON, the chart overlays the engine’s computed stop/target (ATR/swing/band/% fallback)\n"
        "instead of the manual risk band above.",

    "bb_window":
        "Bollinger Bands MA window for %B/bandwidth calculations (typical: 20).",

    "bb_k":
        "Bollinger band width in standard deviations (k·σ). Typical: 2.0.\n"
        "Larger k widens bands (slower %B).",

    # ===== BUY: weights (all 0–1, linear blend) =====
    "w_rsi":
        "Weight of RSI component in BUY. Oversold (RSI low) increases score up to rsi_buy_max.",

    "w_trend":
        "Weight of trend/regime component (price & SMA200 slopes + above/below SMA200).",

    "w_value":
        "Weight of value vs SMA200 (discount favored around 'value_center').",

    "w_flow":
        "Weight of accumulation flow (vol/avg20). Requires vol & 20-day average volume.",

    "w_bbands":
        "Weight of Bollinger %B. BUY prefers being near/below lower band → higher score when %B is small.",

    "w_donch":
        "Weight of Donchian breakout strength.\n"
        "Strength = sigmoid(((last − prev_N_high)/ATR) / 0.5).",

    "w_break":
        "Weight of legacy boolean breakout flag (last ≥ prev_N_high).\n"
        "Use small/zero to avoid double-counting with Donchian.",

    # ===== SELL: thresholds & knobs =====
    "sell_threshold":
        "SELL will trigger if the SELL composite ≥ this value.\n"
        "Higher = pickier/fewer sells; Lower = more frequent trims/exits.",

    "rsi_overbought_min":
        "SELL RSI component uses s_rsi ≈ sigmoid((RSI − threshold)/5).\n"
        "Also boosts slightly if RSI is rolling over. Lower this to make SELL react sooner.",

    "donch_lookback_sell":
        "Donchian lookback N for prior low in SELL (yesterday’s channel).\n"
        "SELL breakout strength uses (prev_N_low − last)/ATR.",

    "ema_fast_span":
        "Fast EMA span for SELL trend checks (cross-under + slope). Smaller = quicker to flip down.",

    "sma_mid_window":
        "Mid SMA window for SELL trend checks (cross-under + slope).",

    "gap_down_min_pct":
        "Extra SELL bonus if today’s open < yesterday’s low by ≥ this % (gap-down).",

    # ===== SELL: weights (all 0–1) =====
    "w_rsi_sell":
        "Weight of RSI overbought/rollover in SELL.",

    "w_trend_down":
        "Weight of down-trend signals (below EMA/SMA + negative slopes).",

    "w_breakdown":
        "Weight of simple breakdown flag (last ≤ prior N-day low).",

    "w_exhaustion":
        "Weight of Bollinger upper-band exhaustion (last > upper band).",

    "w_flow_out":
        "Weight of distribution: down candle with volume spike (vol/avg20 ≥ threshold).",

    "w_bbands_sell":
        "Weight of %B for SELL (near upper band = higher SELL score).",

    "w_donch_sell":
        "Weight of Donchian SELL strength.\n"
        "Strength = sigmoid(((prev_N_low − last)/ATR) / 0.5).",
}

def h(k: str) -> str:  # tiny helper to keep widget lines tidy
    return HELP[k]

def _to_list(x: Any) -> Optional[list]:
    """Coerce cell into list if possible (handles str, list, Series, ndarray)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, pd.Series):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                return list(v)
        except Exception:
            if "," in s:
                return [p.strip() for p in s.split(",")]
        return None
    try:
        return list(x)
    except Exception:
        return None

def _resolve(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cs = set(cols)
    for c in candidates:
        if c in cs:
            return c
    return None

def _get_series_lists(
    row: pd.Series,
    dates_col: str,
    close_col: str,
    sma200_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
):
    dates = _to_list(row[dates_col])
    close = _to_list(row[close_col])
    sma   = _to_list(row[sma200_col])
    open_ = _to_list(row[open_col])
    high  = _to_list(row[high_col])
    low   = _to_list(row[low_col])

    # basic checks
    if not dates or not close or not sma or not open_ or not high or not low:
        return None, None, None, None, None, None

    n = min(len(dates), len(close), len(sma), len(open_), len(high), len(low))
    if n == 0:
        return None, None, None, None, None, None

    x = pd.to_datetime(dates[:n], errors="coerce")
    mask = x.notna()

    return (
        x[mask],
        np.asarray(close[:n], dtype=float)[mask],
        np.asarray(sma[:n], dtype=float)[mask],
        np.asarray(open_[:n], dtype=float)[mask],
        np.asarray(high[:n], dtype=float)[mask],
        np.asarray(low[:n], dtype=float)[mask],
    )

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _coerce_boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

@st.cache_data(show_spinner=False)

def _read_path_cached(p: Path) -> dict:
    """Return {'df': DataFrame, 'sheets': [names], 'picked': name or None}."""
    ext = p.suffix.lower()
    if ext == ".csv":
        return {"df": pd.read_csv(p), "sheets": None, "picked": None}
    if ext in {".xlsx", ".xls"}:
        xls = pd.read_excel(p, sheet_name=None)
        names = list(xls.keys())
        # default to first sheet; UI can re-pick
        return {"df": xls[names[0]], "sheets": names, "picked": names[0]}
    raise ValueError(f"Unsupported file type: {ext}")

def _read_any_table(uploaded_file: Optional[io.BytesIO], path_text: str, prefer_output: bool) -> tuple[pd.DataFrame, str, Optional[Tuple[List[str], str]]]:
    """
    Load a DataFrame from:
      1) DEFAULT_OUTPUT (if prefer_output True and exists),
      2) uploaded_file (.csv/.xlsx/.xls),
      3) explicit path.
    Returns (df, description, sheets_info) where sheets_info is (sheet_names, current_pick) or None.
    """
    # 1) Latest generated
    if prefer_output and DEFAULT_OUTPUT.exists():
        res = _read_path_cached(DEFAULT_OUTPUT)
        return res["df"], f"Loaded generated file: {DEFAULT_OUTPUT.as_posix()}", (res["sheets"], res["picked"])

    # 2) Uploaded
    if uploaded_file is not None:
        name = getattr(uploaded_file, "name", "uploaded")
        ext = Path(name).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(uploaded_file), f"Loaded uploaded CSV: {name}", None
        if ext in {".xlsx", ".xls"}:
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            names = list(xls.keys())
            # pick via UI
            sheet = st.sidebar.selectbox("Excel sheet (uploaded)", names, index=0)
            return xls[sheet], f"Loaded uploaded Excel: {name} (sheet: {sheet})", (names, sheet)
        st.error(f"Unsupported file type: {ext}. Use .csv or .xlsx")
        st.stop()

    # 3) Path
    p = Path(path_text)
    if p.exists():
        res = _read_path_cached(p)
        sheets = res["sheets"]
        picked = res["picked"]
        if sheets:
            sheet = st.sidebar.selectbox(f"Excel sheet ({p.name})", sheets, index=sheets.index(picked) if picked in sheets else 0)
            if sheet != picked:
                # re-read to return the selected sheet
                xls = pd.read_excel(p, sheet_name=None)
                return xls[sheet], f"Loaded from path: {p.as_posix()} (sheet: {sheet})", (sheets, sheet)
        return res["df"], f"Loaded from path: {p.as_posix()}", (sheets, picked)
    st.info("Upload a CSV/XLSX, pick 'Latest generated', or provide a valid path.")
    st.stop()

def _run_main_and_reload() -> tuple[bool, str]:
    """Runs `python main.py` from project root and returns (success, logs_text)."""
    cmd = ["python", "main.py"]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        success = (proc.returncode == 0)
        logs = (proc.stdout or "") + ("\n" + (proc.stderr or ""))
        return success, logs
    except Exception as e:
        return False, f"Failed to run {' '.join(cmd)}: {e}"

OPS = {
    "==":   lambda s, v: s == v,
    "!=":   lambda s, v: s != v,
    ">":    lambda s, v: s > v,
    ">=":   lambda s, v: s >= v,
    "<":    lambda s, v: s < v,
    "<=":   lambda s, v: s <= v,
    "contains":     lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    "not contains": lambda s, v: ~s.astype(str).str.contains(str(v), case=False, na=False),
    "isna":  lambda s, v: s.isna(),
    "notna": lambda s, v: s.notna(),
    # Optional sugar for booleans:
    "is true":  lambda s, v: s.astype("boolean") == True,   # noqa: E712
    "is false": lambda s, v: s.astype("boolean") == False,  # noqa: E712
}

TRUE_TOKENS  = {"true", "1", "yes", "y", "t"}
FALSE_TOKENS = {"false", "0", "no", "n", "f"}

def _anti(Z, need_even=True):
    # If antithetic requested and Z has shape (T, N//2), mirror to (T, N)
    return np.concatenate([Z, -Z], axis=1) if need_even else Z

def _annual_drift_from_sma(
    sma_series: np.ndarray,
    lookback: int = 200,
    min_points: int = 60,
) -> float:
    """
    Estimate annualized drift from the slope of log(SMA) via OLS.
    Returns mu_annual (e.g., 0.08 = +8%/yr). Falls back to 0.0 if not enough data.
    """
    s = pd.Series(sma_series, dtype="float64").dropna()
    if len(s) < max(min_points, 5):
        return 0.0
    s = s.iloc[-min(lookback, len(s)):]
    y = np.log(s.values)
    x = np.arange(len(y), dtype="float64")
    # robust to constant series / nans
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=np.nanmedian(y))
    if not np.isfinite(x).all():
        return 0.0
    # slope per trading day
    try:
        slope_per_day = np.polyfit(x, y, 1)[0]
    except Exception:
        return 0.0
    # annualize (252 trading days)
    mu_annual = float(slope_per_day * 252.0)
    return mu_annual

def _ann_sigma_from_estimator(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    open_: Optional[np.ndarray] = None,
    mode: str = "EWMA",
    lam: float = 0.94,
    window: int = 252,
) -> float:
    """
    Return annualized volatility (sigma) using close-only or OHLC estimators.

    Modes (case-insensitive):
      Close-only: "EWMA", "ROLLING", "MAD"
      OHLC-based: "PARKINSON", "GK" (Garman–Klass), "RS" (Rogers–Satchell), "YANGZHANG" (a.k.a. "YZ")

    Notes:
      - Applies 'window' to the most recent data.
      - Annualizes with sqrt(252).
      - Falls back safely to close-only EWMA if OHLC not available/valid.
    """
    # --- prepare close & close-to-close returns
    c_full = pd.Series(close, dtype="float64").dropna()
    if len(c_full) < 2:
        return 0.0

    if window and len(c_full) > window:
        c_full = c_full[-window:]

    r_cc = np.log(c_full).diff().dropna().values
    if len(r_cc) == 0:
        return 0.0

    m = (mode or "EWMA").strip().lower()

    def ann_from_daily_var(var_daily: float) -> float:
        var_daily = float(max(var_daily, 0.0))
        return float(np.sqrt(var_daily) * np.sqrt(252.0))

    # ---- Close-only estimators ----
    if m == "ewma":
        var = pd.Series(r_cc).ewm(alpha=(1 - lam), adjust=False, min_periods=10).var().iloc[-1]
        if not np.isfinite(var) or var <= 0:
            var = np.var(r_cc, ddof=1)
        return ann_from_daily_var(var)

    if m == "rolling":
        rr = pd.Series(r_cc)
        var = rr.rolling(20, min_periods=10).var().iloc[-1]
        if not np.isfinite(var) or var <= 0:
            var = np.var(r_cc, ddof=1)
        return ann_from_daily_var(var)

    if m == "mad":
        mabs = float(np.mean(np.abs(r_cc)))
        var = (np.pi / 2.0) * (mabs ** 2)
        return ann_from_daily_var(var)

    # ---- OHLC-based estimators (need O/H/L) ----
    need_ohlc = m in {"parkinson", "gk", "garman-klass", "garmanklass", "rs", "rogers-satchell", "yangzhang", "yang-zhang", "yz"}
    have_ohlc = (open_ is not None) and (high is not None) and (low is not None)

    if need_ohlc and have_ohlc:
        # Align arrays to a common recent length and clean
        o = pd.Series(np.asarray(open_, dtype="float64")).dropna()
        h = pd.Series(np.asarray(high, dtype="float64")).dropna()
        l = pd.Series(np.asarray(low,  dtype="float64")).dropna()
        c = c_full.copy()

        n = min(len(o), len(h), len(l), len(c))
        if n >= 2:
            o = o.iloc[-n:].to_numpy()
            h = h.iloc[-n:].to_numpy()
            l = l.iloc[-n:].to_numpy()
            c = c.iloc[-n:].to_numpy()

            mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (o > 0) & (h > 0) & (l > 0) & (c > 0)
            o, h, l, c = o[mask], h[mask], l[mask], c[mask]
            n = len(c)

            if n >= 2:
                hl = np.log(h / l)             # intraday range
                co = np.log(c / o)             # open->close
                c_prev = np.roll(c, 1)
                overnight = np.log(o / c_prev) # close_prev->open (overnight gap)

                # drop first element to align deltas
                hl = hl[1:]; co = co[1:]; overnight = overnight[1:]
                o1 = o[1:]; h1 = h[1:]; l1 = l[1:]; c1 = c[1:]

                if m == "parkinson":
                    var = float(np.mean(hl**2) / (4.0 * np.log(2.0)))
                    return ann_from_daily_var(var)

                if m in {"gk", "garman-klass", "garmanklass"}:
                    var = float(np.mean(0.5 * (hl**2) - (2.0 * np.log(2.0) - 1.0) * (co**2)))
                    return ann_from_daily_var(var)

                if m in {"rs", "rogers-satchell"}:
                    ho = np.log(h1 / o1); hc = np.log(h1 / c1)
                    lo = np.log(l1 / o1); lc = np.log(l1 / c1)
                    rs = ho * hc + lo * lc
                    var = float(np.mean(rs))
                    return ann_from_daily_var(var)

                if m in {"yangzhang", "yang-zhang", "yz"}:
                    # Yang–Zhang decomposition
                    n_yz = len(c1)
                    if n_yz >= 2:
                        # sample variances for components
                        sigma_o2 = float(np.var(overnight, ddof=1)) if len(overnight) > 1 else float(np.var(overnight))
                        sigma_c2 = float(np.var(co, ddof=1))        if len(co) > 1        else float(np.var(co))
                        ho = np.log(h1 / o1); hc = np.log(h1 / c1)
                        lo = np.log(l1 / o1); lc = np.log(l1 / c1)
                        rs = ho * hc + lo * lc
                        sigma_rs2 = float(np.mean(rs))
                        # k weight
                        k = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1)) if n_yz > 1 else 0.34
                        var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
                        return ann_from_daily_var(var)
                # If we fell through (unknown OHLC mode), drop to fallback below.

    # ---- Fallbacks ----
    # If an OHLC mode was requested but OHLC not available/invalid -> EWMA on closes
    rr = pd.Series(r_cc)
    var = rr.ewm(alpha=(1 - lam), adjust=False, min_periods=10).var().iloc[-1]
    if not np.isfinite(var) or var <= 0:
        var = np.var(r_cc, ddof=1)
    return ann_from_daily_var(var)



def _project_next_month(
    y_close: np.ndarray,
    start_date: pd.Timestamp,
    horizon_days: int = 22,
    sims: int = 2000,
    pct_low: float = 10.0,
    pct_high: float = 90.0,
    model: str = "EWMA+t",
    seed: Optional[int] = None,
    # tuning knobs
    window: int = 252,
    lam: float = 0.94,
    df_t: int = 5,
    antithetic: bool = False,
    block: int = 5,
    horizon_months: Optional[int] = None,
    vol_mode: str = "EWMA",                 # "EWMA","ROLLING","MAD","PARKINSON","GK","RS","YANGZHANG"
    y_high: Optional[np.ndarray] = None,
    y_low: Optional[np.ndarray] = None,
    y_open: Optional[np.ndarray] = None,
    stochastic_vol: bool = False,
    # drift from SMA (short/long)
    y_sma_short_for_drift: Optional[np.ndarray] = None,  # e.g., SMA20
    y_sma_long_for_drift: Optional[np.ndarray]  = None,  # e.g., SMA200
    use_sma_drift: bool = True,
    sma_short_weight: float = 0.4,   # weight on SMA20 slope/level
    sma_long_weight: float  = 0.6,   # weight on SMA200 slope/level
    risk_free_ann: float = 0.015,
    drift_cap_ann: float = 0.40,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project price paths and return (future_dates, median, low, high).
    `vol_mode` supports OHLC estimators; pass y_open/y_high/y_low to activate.
    Drift uses a blend of SMA-slope (from price), SMA-level slopes (from SMA20/200),
    plus small historical + risk-free components.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    close = pd.Series(y_close, dtype="float64").clip(lower=1e-9)

    # --- future business dates
    if horizon_months and horizon_months > 0:
        end_dt = start_date + pd.offsets.BMonthEnd(horizon_months)
        future = pd.bdate_range(start=start_date + pd.offsets.BDay(1), end=end_dt)
    else:
        T_days = int(horizon_days if horizon_days and horizon_days > 0 else 22)
        future = pd.bdate_range(start=start_date, periods=T_days + 1)[1:]

    T = len(future)
    if len(close) < 20 or T == 0:
        last = float(close.iloc[-1])
        med = np.full(T, last)
        return future, med, med, med

    # recent log returns (close-to-close)
    r = np.log(close).diff().dropna().values
    r = r[-window:] if len(r) > window else r
    last = float(close.iloc[-1])
    N = int(sims)

    # --- helpers
    def _annual_drift_from_sma_local(sma_series: np.ndarray, lookback: int = 200, min_points: int = 60) -> float:
        """Annualized drift from slope of log(SMA) via OLS."""
        s = pd.Series(sma_series, dtype="float64").dropna()
        if len(s) < max(min_points, 5):
            return 0.0
        s = s.iloc[-min(lookback, len(s)):]
        y = np.log(s.values)
        x = np.arange(len(y), dtype="float64")
        try:
            slope_per_day = np.polyfit(x, y, 1)[0]
        except Exception:
            return 0.0
        return float(slope_per_day * 252.0)

    def _sma_slope(series: np.ndarray, window_: int) -> float:
        """Normalized slope (per-day %) of last `window_` bars of price."""
        if series is None or len(series) < window_:
            return 0.0
        y = pd.Series(series[-window_:], dtype="float64")
        if not np.isfinite(y.iloc[-1]) or y.iloc[-1] <= 0:
            return 0.0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope / y.iloc[-1])  # % per day

    # ---- DRIFT: combine SMA20/200 slope signals + SMA level slopes + tiny hist + risk-free
    # 1) price-based SMA slopes (directional)
    sma20_slope = _sma_slope(y_close, window_=20)
    sma200_slope = _sma_slope(y_close, window_=200)
    daily_drift_from_slope = (sma_short_weight * sma20_slope) + (sma_long_weight * sma200_slope)
    ann_drift_from_slope = daily_drift_from_slope * 252.0

    # 2) SMA-level-based log-slope drift (uses provided SMA20/SMA200 series if available)
    mu_short_ann = 0.0
    mu_long_ann  = 0.0
    if use_sma_drift and y_sma_short_for_drift is not None:
        mu_short_ann = _annual_drift_from_sma_local(y_sma_short_for_drift, lookback=min(90, window), min_points=30)
    if use_sma_drift and y_sma_long_for_drift is not None:
        mu_long_ann = _annual_drift_from_sma_local(y_sma_long_for_drift, lookback=min(252, window), min_points=60)

    # normalize weights if one SMA missing
    w_s = float(np.clip(sma_short_weight, 0.0, 1.0))
    w_l = float(np.clip(sma_long_weight,  0.0, 1.0))
    if y_sma_short_for_drift is None and y_sma_long_for_drift is not None:
        w_s, w_l = 0.0, 1.0
    elif y_sma_long_for_drift is None and y_sma_short_for_drift is not None:
        w_s, w_l = 1.0, 0.0
    elif (y_sma_short_for_drift is None) and (y_sma_long_for_drift is None):
        w_s, w_l = 0.0, 0.0

    mu_sma_combo = (w_s * mu_short_ann) + (w_l * mu_long_ann)

    # 3) tiny historical + risk-free baselines
    mu_hist_ann = float(np.mean(r) * 252.0) if len(r) > 5 else 0.0

    # Final blend (weights are heuristics; tune to taste)
    mu_blend_ann = (
        0.50 * float(ann_drift_from_slope) +   # momentum direction from SMA slopes on price
        0.30 * float(mu_sma_combo) +           # regime anchor from SMA level slopes
        0.10 * float(mu_hist_ann) +            # tiny historical
        0.10 * float(risk_free_ann)            # small baseline
    )
    mu_blend_ann = float(np.clip(mu_blend_ann, -drift_cap_ann, drift_cap_ann))
    mu_d = mu_blend_ann * dt

    # --- antithetic helpers
    def _antithetic_norm(T_, N_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_normal((T_, half))
        Z_full = np.concatenate([Z_half, -Z_half], axis=1)
        return Z_full[:, :N_]

    def _antithetic_t(T_, N_, df_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_t(df_, size=(T_, half)) / np.sqrt(df_ / (df_ - 2))
        Z_full = np.concatenate([Z_half, -Z_half], axis=1)
        return Z_full[:, :N_]

    # --- volatility level (uses OHLC if provided)
    sigma_ann = _ann_sigma_from_estimator(
        close=y_close,
        high=y_high, low=y_low, open_=y_open,
        mode=vol_mode, lam=lam, window=window
    )
    sigma_d0 = sigma_ann * np.sqrt(dt)

    # --- simulate
    if model == "GBM" or model == "EWMA+t":
        if model == "GBM":
            Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            if stochastic_vol:
                # mean-reverting random variance around sigma_d0
                rr2 = r**2
                if len(rr2) >= 30:
                    xcv = rr2[:-1]; ycv = rr2[1:]
                    rho = np.corrcoef(xcv, ycv)[0, 1]
                    rho = 0.8 if not np.isfinite(rho) else float(np.clip(rho, 0.0, 0.98))
                else:
                    rho = 0.8
                v0 = sigma_d0**2
                sig = np.empty((T, N))
                sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_d - 0.5 * sig**2) + sig * Z
            else:
                steps = (mu_d - 0.5 * sigma_d0**2) + sigma_d0 * Z
            paths = last * np.exp(np.cumsum(steps, axis=0))

        else:  # "EWMA+t"
            # keep median neutral by default (driftless-t). Use mu_t = mu_d if you want trend here too.
            mu_t = 0.0
            Z = _antithetic_t(T, N, df_t) if antithetic else (rng.standard_t(df_t, size=(T, N)) / np.sqrt(df_t / (df_t - 2)))
            if stochastic_vol:
                rho = 0.85
                v0 = sigma_d0**2
                sig = np.empty((T, N))
                sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_t - 0.5 * sig**2) + sig * Z
            else:
                steps = (mu_t - 0.5 * sigma_d0**2) + sigma_d0 * Z
            paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Bootstrap":
        def block_bootstrap(returns, T_, sims_, block_=5):
            returns = np.asarray(returns)
            if len(returns) == 0:
                return np.zeros((T_, sims_))
            if block_ <= 1 or len(returns) < block_:
                idx = rng.integers(0, len(returns), size=(T_, sims_))
                return returns[idx]
            R = np.empty((T_, sims_))
            max_start = max(0, len(returns) - block_)
            for j in range(sims_):
                path = []
                while len(path) < T_:
                    start = rng.integers(0, max_start + 1)
                    path.extend(returns[start:start+block_].tolist())
                R[:, j] = path[:T_]
            return R

        steps = block_bootstrap(r, T, N, block=int(block))
        paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Jump":
        Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
        sigma_d = np.std(r, ddof=1) * np.sqrt(dt) if len(r) > 1 else sigma_d0
        mu_j, sig_j, lam_j = -0.02, 0.06, 3.0
        Njump = rng.poisson(lam=(lam_j * dt), size=(T, N))
        J = rng.normal(mu_j, sig_j, size=(T, N)) * Njump
        steps = (0.0 - 0.5 * sigma_d**2) + sigma_d * Z + J
        paths = last * np.exp(np.cumsum(steps, axis=0))

    else:
        # fallback to GBM (propagate all OHLC + drift knobs, incl. SMA20/200 combo)
        return _project_next_month(
            y_close, start_date,
            horizon_days=horizon_days, sims=sims,
            pct_low=pct_low, pct_high=pct_high,
            model="GBM", seed=seed,
            window=window, lam=lam, df_t=df_t, antithetic=antithetic, block=block,
            horizon_months=horizon_months,
            vol_mode=vol_mode, y_high=y_high, y_low=y_low, y_open=y_open,
            stochastic_vol=stochastic_vol,
            y_sma_short_for_drift=y_sma_short_for_drift,
            y_sma_long_for_drift=y_sma_long_for_drift,
            use_sma_drift=use_sma_drift,
            sma_short_weight=sma_short_weight,
            sma_long_weight=sma_long_weight,
            risk_free_ann=risk_free_ann,
            drift_cap_ann=drift_cap_ann,
        )

    # summarize
    low  = np.nanpercentile(paths, pct_low, axis=1)
    high = np.nanpercentile(paths, pct_high, axis=1)
    med  = np.nanpercentile(paths, 50, axis=1)
    return future, med, low, high


def _sma_slope(series: np.ndarray, window: int = 20) -> float:
    """Return slope (per day) of last SMA window using linear regression."""
    if series is None or len(series) < window:
        return 0.0
    y = pd.Series(series[-window:], dtype="float64")
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)  # slope of regression line
    return slope / y.iloc[-1]  # normalize by price level -> % drift per day

def _parse_ci_label(lbl: str) -> tuple[float, float]:
    # Accept both "10–90%" (en dash) and "10-90%" (hyphen)
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%?\s*$", lbl)
    if not m:
        return (10.0, 90.0)
    a, b = float(m.group(1)), float(m.group(2))
    a, b = min(a, b), max(a, b)
    a = max(0.0, min(49.999, a))
    b = min(100.0, max(50.001, b))
    return a, b

def _looks_bool_token(val: str) -> Optional[bool]:
    s = str(val).strip().lower()
    if s in TRUE_TOKENS:  return True
    if s in FALSE_TOKENS: return False
    return None

def _coerce_series_for_rule(series: pd.Series, op: str, raw_value: str) -> tuple[pd.Series, Any]:
    """
    Decide how to coerce the series/value based on the operator + provided value.
    Returns (coerced_series, coerced_value).
    """
    # text ops keep strings
    if op in {"contains", "not contains"}:
        return series.astype("string"), raw_value

    # null checks ignore the value
    if op in {"isna", "notna"}:
        return series, None

    # explicit boolean ops
    if op in {"is true", "is false"}:
        return series.astype("boolean"), None

    # if typed value looks boolean, compare as boolean (== / != supported later)
    val_bool = _looks_bool_token(raw_value)
    if val_bool is not None:
        # we'll build a robust boolean series
        ser_bool = series.map(lambda z:
            True  if str(z).strip().lower() in TRUE_TOKENS  else
            False if str(z).strip().lower() in FALSE_TOKENS else np.nan
        ).astype("boolean")
        return ser_bool, val_bool

    # try datetime if series is datetime-like OR value parses as a date
    val_ts = _parse_datetime_value(raw_value) if raw_value is not None else pd.NaT
    if pd.api.types.is_datetime64_any_dtype(series) or (isinstance(val_ts, pd.Timestamp) and pd.notna(val_ts)):
        ser_dt = pd.to_datetime(series, errors="coerce")
        return ser_dt, val_ts

    # otherwise numeric comparison
    return pd.to_numeric(series, errors="coerce"), _safe_cast_number(raw_value)

# =========================
# Buy-signal engine (helpers + core)
# =========================
from dataclasses import dataclass

def _sigmoid(x: float, k: float = 1.0) -> float:
    try:
        return float(1.0 / (1.0 + np.exp(-x / max(k, 1e-9))))
    except Exception:
        return 0.5
    
def _bbands_series(close: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ma, upper, lower) Bollinger Bands arrays for the given close series."""
    s = pd.Series(close, dtype="float64")
    ma = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    up = (ma + k * sd).to_numpy()
    lo = (ma - k * sd).to_numpy()
    return ma.to_numpy(), up, lo

def _bb_percent_b_and_bw(close: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[float, float]:
    """
    Return Bollinger %B and BandWidth at the last bar.
    %B = (Close - Lower)/(Upper - Lower), clipped to [0,1] if bands valid.
    BandWidth = (Upper - Lower)/MA. Returns (np.nan, np.nan) if not enough history.
    """
    ma, up, lo = _bbands_series(close, window, k)
    if len(close) == 0 or not np.isfinite(up[-1]) or not np.isfinite(lo[-1]):
        return np.nan, np.nan
    denom = max(up[-1] - lo[-1], 1e-12)
    percent_b = (float(close[-1]) - lo[-1]) / denom
    if np.isfinite(ma[-1]) and ma[-1] > 0:
        bw = (up[-1] - lo[-1]) / ma[-1]
    else:
        bw = np.nan
    return float(np.clip(percent_b, 0.0, 1.0)), float(bw)

def _donch_breakout_strength(last: float, prev_hi: float, prev_lo: float, atr: float) -> tuple[float, float]:
    """
    Return Donchian breakout strengths for BUY and SELL:
      buy_strength  = max((last - prev_hi)/ATR, 0)  in [0, +)
      sell_strength = max((prev_lo - last)/ATR, 0)  in [0, +)
    These are later squashed to [0,1] with a sigmoid (so ~0.5·ATR above/below starts to count).
    If ATR is not finite, returns (1.0/0.0 based on plain breakout booleans) as a fallback.
    """
    if not np.isfinite(atr) or atr <= 0:
        return (1.0 if (np.isfinite(prev_hi) and last >= prev_hi) else 0.0,
                1.0 if (np.isfinite(prev_lo) and last <= prev_lo) else 0.0)
    return (max((last - prev_hi) / atr, 0.0) if np.isfinite(prev_hi) else 0.0,
            max((prev_lo - last) / atr, 0.0) if np.isfinite(prev_lo) else 0.0)

def _wilder_rma(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA used by RSI/ATR. Returns same-length array with NaNs for warmup."""
    a = pd.Series(arr, dtype="float64")
    rma = a.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return rma.to_numpy()

def _rsi_wilder(close: np.ndarray, period: int = 14) -> float:
    x = pd.Series(close, dtype="float64").dropna().to_numpy()
    if len(x) < period + 1:
        return float("nan")
    diff = np.diff(x)
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    avg_gain = _wilder_rma(up, period)
    avg_loss = _wilder_rma(dn, period)
    rs = np.divide(avg_gain, np.where(avg_loss == 0, np.nan, avg_loss))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi[-1])

def _rsi_series_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    x = pd.Series(close, dtype="float64").dropna().to_numpy()
    if len(x) < period + 1:
        return np.array([], dtype="float64")
    diff = np.diff(x)
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    avg_gain = pd.Series(up).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(dn).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # align back to price length by padding a nan at start
    return np.concatenate([[np.nan], rsi.to_numpy()])

def _atr_from_ohlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    h = pd.Series(high, dtype="float64").to_numpy()
    l = pd.Series(low,  dtype="float64").to_numpy()
    c = pd.Series(close, dtype="float64").to_numpy()
    n = min(len(h), len(l), len(c))
    if n < period + 1:
        return float("nan")
    h, l, c = h[-n:], l[-n:], c[-n:]
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    atr = _wilder_rma(tr, period)
    return float(atr[-1])

def _boll_lower(close: np.ndarray, window: int = 20, k: float = 2.0) -> float:
    s = pd.Series(close, dtype="float64").dropna()
    if len(s) < max(window, 5):
        return float("nan")
    ma = s.rolling(window, min_periods=window).mean().iloc[-1]
    sd = s.rolling(window, min_periods=window).std(ddof=0).iloc[-1]
    return float(ma - k * sd)

def _donchian_prev_high(high: np.ndarray, lookback: int = 20) -> float:
    h = pd.Series(high, dtype="float64").dropna().to_numpy()
    if len(h) < lookback + 1:
        return float("nan")
    # use yesterday's channel
    return float(np.max(h[-(lookback+1):-1]))

def _donchian_prev_low(low: np.ndarray, lookback: int = 20) -> float:
    l = pd.Series(low, dtype="float64").dropna().to_numpy()
    if len(l) < lookback + 1:
        return float("nan")
    return float(np.min(l[-(lookback+1):-1]))


def _slope_pct_per_day(series: np.ndarray, window: int) -> float:
    s = pd.Series(series, dtype="float64").dropna()
    if len(s) < window:
        return 0.0
    y = s.iloc[-window:].to_numpy()
    x = np.arange(len(y), dtype="float64")
    try:
        m = np.polyfit(x, y, 1)[0]
        if y[-1] <= 0 or not np.isfinite(y[-1]):
            return 0.0
        return float(m / y[-1])
    except Exception:
        return 0.0

@dataclass
class BuyParams:
    # BUY weights (all ∈ [0,1]; final score is clipped to [0,1])
    w_rsi: float = 0.20
    w_trend: float = 0.20
    w_value: float = 0.10
    w_flow: float = 0.05
    w_bbands: float = 0.20          # NEW: Bollinger %B (oversold near lower band)
    w_donchian: float = 0.25        # NEW: Donchian breakout (ATR-scaled)
    w_breakout: float = 0.00        # (kept for backward compat, default 0 to avoid double-counting)

    composite_threshold: float = 0.60

    # BUY feature centers / knobs
    rsi_buy_max: float = 45.0
    rsi_floor: float = 20.0
    sma200_window: int = 200
    donch_lookback: int = 20
    gap_min_pct: float = 0.5
    value_center_dev_pct: float = -5.0
    vol_ratio_min: float = 1.50

    # Bollinger settings (shared)
    bb_window: int = 20             
    bb_k: float = 2.0                

    # Stops / sizing
    use_engine_stop: bool = True
    atr_mult: float = 1.5
    stop_pct: float = 10.0
    reward_R: float = 2.0

    portfolio_value: float = 20000.0
    risk_per_trade_pct: float = 0.5
    min_price: float = 1.0
    min_adv_dollars: float = 250_000.0

    # SELL weights
    w_rsi_sell: float = 0.30
    w_trend_down: float = 0.30
    w_breakdown: float = 0.00        
    w_exhaustion: float = 0.10
    w_flow_out: float = 0.05
    w_bbands_sell: float = 0.25      
    w_donchian_sell: float = 0.25    

    sell_threshold: float = 0.60

    # SELL features
    rsi_overbought_min: float = 70.0
    ema_fast_span: int = 21
    sma_mid_window: int = 50
    donch_lookback_sell: int = 20
    gap_down_min_pct: float = 0.5

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


def compute_signal_series_for_row(
    row: pd.Series,
    x: pd.Series,
    close: np.ndarray,
    sma200: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    params: BuyParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute historical BUY/SELL booleans across the visible slice.
    Returns (buy_idx, sell_idx) as integer index arrays, aligned to `close`.
    """
    r_clean = _row_without_point_indicators(row)
    n = len(close)

    # Warmup: enough for Donchian & basic indicators, but NEVER clamp to n-1
    # (We intentionally do NOT require a full SMA200 lookback; trend features
    # will gracefully degrade when history is shorter.)
    warmup = max(22, int(params.donch_lookback) + 1, int(params.donch_lookback_sell) + 1)
    warmup = min(warmup, max(0, n - 3))  # leave plenty of bars to evaluate

    buy_idx, sell_idx = [], []
    for i in range(warmup, n):
        b = compute_buy_signal(r_clean, x[:i+1], close[:i+1], sma200[:i+1], open_[:i+1], high[:i+1], low[:i+1], params)
        s = compute_sell_signal(r_clean, x[:i+1], close[:i+1], sma200[:i+1], open_[:i+1], high[:i+1], low[:i+1], params)
        if b["buy"]:
            buy_idx.append(i)
        if s["sell"]:
            sell_idx.append(i)

    return np.asarray(buy_idx, dtype=int), np.asarray(sell_idx, dtype=int)

def _row_without_point_indicators(row: pd.Series) -> pd.Series:
    """Return a copy of row with point-in-time indicators nulled, so engines recompute series-based values."""
    r = row.copy()
    for c in ("momentum__rsi", "rsi", "vol", "volume__vol", "vol_avg20", "volume__vol_avg20"):
        if c in r:
            r[c] = np.nan
    return r

def run_dca_backtest(
    dates: pd.Series,
    close: np.ndarray,
    buy_idx: np.ndarray,
    sell_idx: np.ndarray,
    *,
    starting_cash: float = 10_000.0,
    buy_pct_first: float = 25.0,
    buy_pct_next: float = 25.0,
    dca_trigger_drop_pct: float = 5.0,
    max_dca_legs: int = 3,
    sell_pct_first: float = 50.0,
    sell_pct_next: float = 50.0,
) -> tuple[np.ndarray, list[dict], float]:
    """
    DCA/backtest on top of signal series.
    Returns (equity_curve, trades, total_return_fraction).
    """
    buy_set  = set(map(int, buy_idx))
    sell_set = set(map(int, sell_idx))

    cash = float(starting_cash)
    shares = 0.0
    last_buy_px = np.nan
    dca_legs = 0
    sold_once_since_last_buy = False

    equity = np.zeros(len(close), dtype=float)
    trades: list[dict] = []

    for i, px in enumerate(close):
        # -- SELL first (priority to risk management)
        if i in sell_set and shares > 0:
            pct = sell_pct_first if not sold_once_since_last_buy else sell_pct_next
            qty = shares * max(min(pct, 100.0), 0.0) / 100.0
            if qty > 0:
                proceeds = qty * px
                cash += proceeds
                shares -= qty
                trades.append({"side":"SELL","i":i,"date":dates[i],"price":float(px),"qty":float(qty),"cash":float(cash),"shares":float(shares)})
                sold_once_since_last_buy = True
                if shares <= 1e-9:
                    shares = 0.0
                    dca_legs = 0
                    last_buy_px = np.nan
                    sold_once_since_last_buy = False

        # -- BUY (DCA)
        if i in buy_set:
            gate = (not np.isfinite(last_buy_px)) or (px <= last_buy_px * (1.0 - dca_trigger_drop_pct / 100.0))
            if dca_legs < max_dca_legs and gate and cash > 0:
                pct = buy_pct_first if not np.isfinite(last_buy_px) else buy_pct_next
                invest = cash * max(min(pct, 100.0), 0.0) / 100.0
                if invest > 0:
                    qty = invest / px
                    cash -= invest
                    shares += qty
                    last_buy_px = float(px)
                    dca_legs += 1
                    sold_once_since_last_buy = False
                    trades.append({"side":"BUY","i":i,"date":dates[i],"price":float(px),"qty":float(qty),"cash":float(cash),"shares":float(shares)})

        equity[i] = cash + shares * px

    total_return = (equity[-1] / starting_cash) - 1.0 if len(equity) else 0.0
    return equity, trades, float(total_return)

def _safe_cast_number(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return np.nan
    
def _infer_col_kind(s: pd.Series) -> str:
    """
    Return one of {'bool','numeric','datetime','string'} based on dtype and values.
    Treats columns with only true/false-like strings as bool.
    """
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # heuristic: many datasets store booleans as strings
    sample = s.dropna().astype(str).str.strip().str.lower()
    if not sample.empty:
        unique_small = set(sample.unique()[:20])
        if unique_small <= (TRUE_TOKENS | FALSE_TOKENS):
            return "bool"
        # quick date sniff: first non-empty parse success => datetime
        try:
            probe = pd.to_datetime(sample.iloc[0], errors="raise")
            if probe is not pd.NaT:  # parsed
                return "datetime"
        except Exception:
            pass
    return "string"

def _parse_datetime_value(val: str):
    """Return a pandas.Timestamp or NaT from a user-entered value."""
    try:
        ts = pd.to_datetime(val, errors="coerce")
    except Exception:
        ts = pd.NaT
    return ts

# =========================
# UI
# =========================
st.set_page_config(page_title="Ticker Evaluator", layout="wide")
st.title("📈 Ticker Evaluator — Interactive Review")
st.caption("Load scanner CSV/XLSX, filter/sort, overlay technicals, and inspect setups. You can also re-run the scan.")

# --- Controls: run main + data source ---
c_r1, c_r2, c_r3 = st.columns([1.2, 1.2, 2])
with c_r1:
    run_now = st.button("▶ Run scan (main.py)")
with c_r2:
    source_choice = st.selectbox("Data source", ["Latest generated", "Upload file", "Path"], index=0)
with c_r3:
    default_path = st.text_input("Path (.csv or .xlsx)", value=DEFAULT_OUTPUT.as_posix())

uploaded = None
if source_choice == "Upload file":
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

# Execute main.py on demand
if run_now:
    with st.spinner("Running main.py..."):
        ok, logs = _run_main_and_reload()
    with st.expander("Run logs", expanded=True):
        st.code(logs or "(no output)")
    if not ok:
        st.error("main.py failed. Check logs above.")
    else:
        st.success(f"Scan completed. Reloading: {DEFAULT_OUTPUT.as_posix()}")

# Load data (after possibly running main)
prefer_output = (source_choice == "Latest generated")
df, loaded_msg, sheets_info = _read_any_table(uploaded, default_path, prefer_output=prefer_output)
st.caption(loaded_msg)

if df.empty:
    st.warning("Table is empty.")
    st.stop()

# --- Column resolution (legacy vs flat)
cols = list(df.columns)
ticker_col = _resolve(cols, ("ticker", "meta__ticker"))
date_col   = _resolve(cols, ("date", "meta__date"))
comp_col   = _resolve(cols, ("composite_score", "signals_and_scores__composite_score"))
sig_col    = _resolve(cols, ("signals_score", "signals_and_scores__signals_score"))
rsi_col    = _resolve(cols, ("rsi", "momentum__rsi"))
owned_col  = _resolve(cols, ("owned", "meta__owned", "position__owned"))

last_col   = _resolve(cols, ("last", "meta__last"))
stop_col   = _resolve(cols, ("stop_price", "stops_and_risk__planned_stop_price"))
tgt_col    = _resolve(cols, ("target_price", "stops_and_risk__planned_target_price"))

dates_series_col  = _resolve(cols, ("dates_series", "series__dates_series"))
close_series_col  = _resolve(cols, ("close_series", "series__close_series"))
sma200_series_col = _resolve(cols, ("sma200_series", "series__sma200_series"))

open_series_col    = _resolve(cols, ("open_series", "series__open_series"))
high_series_col    = _resolve(cols, ("high_series", "series__high_series"))
low_series_col     = _resolve(cols, ("low_series", "series__low_series"))

required_any = [ticker_col, dates_series_col, close_series_col, sma200_series_col, open_series_col, high_series_col, low_series_col]
if any(c is None for c in required_any):
    st.error(
        "Data is missing one or more required series columns: "
        "`ticker`, `dates_series`, `close_series`, `sma200_series`, "
        "`open_series`, `high_series`, `low_series` "
        "(or their `series__*` equivalents)."
        f"Current available columns include {df.columns}"
    )
    st.stop()

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")
min_sig  = st.sidebar.number_input("Min signals_score", value=0, step=1, key="min_sig")
min_comp = st.sidebar.number_input("Min composite_score", value=0.0, step=0.5, format="%.2f", key="min_comp")
rsi_min, rsi_max = st.sidebar.slider("RSI range", 0, 100, (0, 100), key="rsi_range")
owned_only = st.sidebar.checkbox("Owned only", value=False, key="owned_only")
search = st.sidebar.text_input("Search ticker (substring)", value="", key="search")

rsi_min, rsi_max = st.session_state["rsi_range"]

# Coerce types for filters
if sig_col:  df[sig_col]  = _coerce_numeric(df[sig_col])
if comp_col: df[comp_col] = _coerce_numeric(df[comp_col])
if rsi_col:  df[rsi_col]  = _coerce_numeric(df[rsi_col])
if owned_col:
    df[owned_col] = _coerce_boolish(df[owned_col])

mask = pd.Series(True, index=df.index)
if sig_col:
    mask &= df[sig_col] >= min_sig
if comp_col:
    mask &= df[comp_col] >= min_comp
if rsi_col:
    mask &= df[rsi_col].between(rsi_min, rsi_max)
if owned_only and owned_col:
    mask &= df[owned_col].fillna(False)
if search and ticker_col:
    s = search.strip().lower()
    mask &= df[ticker_col].astype(str).str.lower().str.contains(s, na=False)

# --- Custom filters
st.sidebar.divider()
st.sidebar.subheader("Custom filters")

if "custom_rules" not in st.session_state:
    st.session_state["custom_rules"] = []   # list of {"col","op","val"}

cols_all = list(df.columns)
logical = st.sidebar.radio("Combine rules with", ["AND", "OR"], horizontal=True, key="cf_logical")

with st.sidebar.expander("Add rule", expanded=True):
    col = st.selectbox("Column", cols_all, key="cf_col")

    # infer & build operator/value controls
    kind = _infer_col_kind(df[col]) if col else "string"
    if kind == "bool":
        op_choices = ["is true", "is false", "==", "!=", "isna", "notna"]
    elif kind == "numeric":
        op_choices = ["==", "!=", ">", ">=", "<", "<=", "isna", "notna"]
    elif kind == "datetime":
        op_choices = ["==", "!=", ">", ">=", "<", "<=", "isna", "notna"]
    else:
        op_choices = ["contains", "not contains", "==", "!=", "isna", "notna"]

    op = st.selectbox("Operator", op_choices, key="cf_op")

    # value widget
    if op in {"isna", "notna", "is true", "is false"}:
        val = ""
    else:
        if kind == "bool":
            val = st.radio("Value", ["True", "False"], horizontal=True, key="cf_val_bool")
        elif kind == "numeric":
            val = str(st.number_input("Value", value=0.0, step=1.0, key="cf_val_num"))
        elif kind == "datetime":
            val = st.text_input("Value (date/time)", placeholder="e.g. 2024-12-31 15:30", key="cf_val_dt")
        else:
            val = st.text_input("Value", key="cf_val_txt")

    # IMPORTANT: give the button a key and force a rerun after append
    if st.button("Add rule", key="btn_add_rule"):
        # sanitize value based on inferred kind/operator
        try:
            v_clean = val
            if op in {"isna", "notna", "is true", "is false"}:
                v_clean = ""

            elif kind == "bool":
                v_clean = "True" if str(val).strip().lower() in TRUE_TOKENS else "False" if str(val).strip().lower() in FALSE_TOKENS else val

            elif kind == "numeric":
                # accept str but ensure it parses
                _ = float(str(val))  # raises if invalid

            elif kind == "datetime":
                _v = _parse_datetime_value(str(val))
                if pd.isna(_v):
                    raise ValueError("Invalid datetime")

            # only then append
            st.session_state.custom_rules.append({"col": col, "op": op, "val": str(v_clean)})
            st.sidebar.success(f"Added: {col} {op} {v_clean}")
            st.rerun()

        except Exception as e:
            st.sidebar.warning(f"Rule not added: {e}")

# View/manage rules
if st.session_state.custom_rules:
    for i, r in enumerate(st.session_state.custom_rules):
        st.sidebar.write(f"{i+1}. `{r['col']} {r['op']} {r['val']}`")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Clear rules"):
        st.session_state.custom_rules = []
    if c2.button("Remove last") and st.session_state.custom_rules:
        st.session_state.custom_rules.pop()

# Apply custom rules
if st.session_state.custom_rules:
    masks = []
    bad_rules = []
    for idx, r in enumerate(st.session_state.custom_rules, start=1):
        col, op, val = r.get("col"), r.get("op"), r.get("val")
        try:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not in table")
            if op not in OPS:
                raise KeyError(f"Operator '{op}' not supported")

            ser, coerced_val = _coerce_series_for_rule(df[col], op, val)
            # Some ops ignore value; others need it
            if op in {"isna", "notna", "is true", "is false"}:
                rule_mask = OPS[op](ser, None)
            else:
                rule_mask = OPS[op](ser, coerced_val)

            # Ensure boolean mask aligned to df
            rule_mask = pd.Series(rule_mask, index=df.index).fillna(False)
            if rule_mask.dtype != bool:
                rule_mask = rule_mask.astype(bool)

            masks.append(rule_mask)

        except Exception as e:
            bad_rules.append((idx, f"{col} {op} {val}", str(e)))

    if bad_rules:
        with st.sidebar.expander("⚠ Some rules were skipped", expanded=False):
            for i, txt, err in bad_rules:
                st.write(f"{i}. `{txt}` — {err}")

    if masks:
        comb = masks[0]
        for m in masks[1:]:
            comb = (comb & m) if logical == "AND" else (comb | m)
        mask &= comb


df_view = df[mask].copy()

st.sidebar.divider()
st.sidebar.subheader("Chart colors")

_prev_theme = json.dumps(th, sort_keys=True)

c1, c2 = st.sidebar.columns(2)
with c1:
    th["close"]   = st.color_picker("Close",   th["close"])
    th["sma200"]  = st.color_picker("SMA200",  th["sma200"])
    th["overlay"] = st.color_picker("Overlays", th["overlay"])
    th["risk_band"] = st.color_picker("Risk band fill", th.get("risk_band", "#8dd3c7"))
with c2:
    th["stop"]     = st.color_picker("Stop",     th["stop"])
    th["target"]   = st.color_picker("Target",   th["target"])
    th["proj_mid"] = st.color_picker("Proj mid", th["proj_mid"])
    th["proj_band"] = st.color_picker("Proj band fill", th["proj_band"])

_new_theme = json.dumps(th, sort_keys=True)
if _new_theme != _prev_theme:
    save_theme(th)
    st.sidebar.caption("✅ Theme saved")

# =========================
# Sort & table
# =========================
st.sidebar.header("Sort")
sort_candidates = [c for c in [comp_col, sig_col, rsi_col, last_col, date_col] if c]
sort_choice = st.sidebar.selectbox("Primary sort", sort_candidates, index=0 if sort_candidates else 0, key="sort_choice")
ascending   = st.sidebar.checkbox("Ascending", value=(sort_choice == rsi_col), key="sort_asc")

if sort_choice:
    df_view = df_view.sort_values(sort_choice, ascending=ascending, kind="mergesort")  # stable

st.subheader("Results")

# Column picker for table
default_cols = [c for c in [ticker_col, comp_col, sig_col, rsi_col, last_col, owned_col, date_col] if c]
chosen_cols = st.multiselect("Columns to show", options=list(df_view.columns), default=default_cols, key="table_cols")
if df_view.empty:
    st.info("No rows match the current filters.")
    st.stop()
st.dataframe(df_view[chosen_cols], use_container_width=True, hide_index=True)

# =========================
# Selection & series extraction
# =========================
st.markdown("---")
st.subheader("Details & Chart")

if ticker_col:
    tickers = df_view[ticker_col].astype(str).tolist()
    sel_default = tickers[0] if tickers else None
    sel_ticker = st.selectbox("Select ticker", tickers, index=0 if tickers else None)
    row = df_view[df_view[ticker_col].astype(str) == sel_ticker].iloc[0]
else:
    row = df_view.iloc[0]

x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(row, dates_series_col, close_series_col, sma200_series_col, open_series_col, high_series_col, low_series_col)
if x is None:
    st.warning("Selected row has empty or invalid series.")
    st.stop()

stop_val = float(row.get(stop_col, np.nan)) if stop_col else np.nan
tgt_val  = float(row.get(tgt_col,  np.nan)) if tgt_col  else np.nan
last_val = float(row.get(last_col, np.nan)) if last_col else np.nan

# --- Risk band controls
st.subheader("Risk/Reward")
use_row_stops = st.checkbox("Use row Stop/Target if available", value=True)

# If missing or user prefers synthetic values, compute from risk tolerance
risk_pct = st.number_input("Risk % (per trade)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, help="Percent below current price for stop.")
reward_R = st.number_input("Reward multiple (R)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, help="Target = Last + R * (Last - Stop).")

if not np.isfinite(last_val):
    last_val = float(y_close[-1])  # fallback to latest plotted close

if use_row_stops and np.isfinite(stop_val) and np.isfinite(tgt_val):
    stop_for_band = stop_val
    tgt_for_band  = tgt_val
else:
    stop_for_band = last_val * (1.0 - risk_pct/100.0)
    risk_amount   = last_val - stop_for_band
    tgt_for_band  = last_val + reward_R * risk_amount

# =========================
# Chart options (now that x/y are known)
# =========================
st.subheader("Chart Options")
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1: show_sma    = st.checkbox("Show SMA200", value=True, key="show_sma")
with c2: show_stop   = st.checkbox("Show Stop", value=bool(np.isfinite(stop_val)), key="show_stop")
with c3: show_target = st.checkbox("Show Target", value=bool(np.isfinite(tgt_val)), key="show_target")
with c4: range_days  = st.number_input("Last N days", min_value=10, max_value=len(x), value=min(180, len(x)), step=10, key="range_days")

# Overlays picker + param editors
oc1, oc2 = st.columns([2, 2])
with oc1:
    chosen_overlays = st.multiselect(
        "Overlays",
        options=list(TECHNICALS_REGISTRY.keys()),
        default=["SMA", "EMA"],
        key="overlays",
    )
with oc2:
    # Render simple param UIs for chosen overlays based on their schema
    overlay_params: Dict[str, dict] = {}
    for key in chosen_overlays:
        spec = TECHNICALS_REGISTRY.get(key, {})
        schema = spec.get("schema", {})
        defaults = spec.get("params", {})
        if not schema:
            overlay_params[key] = defaults
            continue
        with st.expander(f"{key} params", expanded=False):
            params = {}
            saved_params = st.session_state.get("overlay_params_loaded", {}).get(key, {})
            for pname, (ptype, lo, hi) in schema.items():
                # prefer saved value, else registry default
                dv = saved_params.get(pname, defaults.get(pname, lo))
                if ptype == "int":
                    params[pname] = st.number_input(f"{key}.{pname}", value=int(dv), min_value=int(lo), max_value=int(hi), step=1)
                elif ptype == "float":
                    params[pname] = st.number_input(f"{key}.{pname}", value=float(dv), min_value=float(lo), max_value=float(hi), step=0.1, format="%.2f")
                else:
                    params[pname] = st.text_input(f"{key}.{pname}", value=str(dv))
            overlay_params[key] = params

# --- Projections UI ---
st.subheader("Projections")
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
with pc1:
    show_projection = st.checkbox("Show projection simulations", value=True)
with pc2:
    proj_band = st.selectbox(
        "Projection band",
        ["10–90%", "5–95%", "20–80%", "25–75%", "Custom"],
        index=0
    )
with pc3:
    proj_sims = st.number_input("Simulations", min_value=100, max_value=20000, value=2000, step=1000)
with pc4:
    model_choice = st.selectbox("Projection model", ["EWMA+t", "GBM", "Bootstrap", "Jump"], index=0)
with pc5:
    proj_months = st.number_input("Months to project", min_value=1, max_value=24, value=1, step=1)

if proj_band == "Custom":
    cc1, cc2 = st.columns(2)
    with cc1:
        pct_low = st.number_input("Lower percentile", min_value=0.0, max_value=49.9, value=10.0, step=0.5)
    with cc2:
        pct_high = st.number_input("Upper percentile", min_value=50.1, max_value=100.0, value=90.0, step=0.5)
else:
    pct_low, pct_high = _parse_ci_label(proj_band)

with st.expander("Advanced projection controls"):
    window = st.number_input("Calibration window (trading days)", min_value=60, max_value=252*5, value=252, step=20, help="History length used to estimate drift/vol/jumps")
    lam = st.slider("EWMA lambda (vol persistence)", 0.80, 0.99, 0.94, 0.01, help="Higher = smoother volatility")
    df_t = st.slider("Student-t degrees of freedom", 3, 15, 5, 1, help="Lower => fatter tails")
    antithetic = st.checkbox("Use antithetic variates (variance reduction)", value=True)
    block = st.number_input("Bootstrap block size", min_value=3, max_value=30, value=5, step=1)
    vol_mode = st.selectbox(
        "Volatility estimator",
        ["YangZhang", "Parkinson", "GK", "RS", "CloseEWMA", "CloseRolling", "MAD"],
        index=0,
        help="OHLC-based estimators (YangZhang/Parkinson/GK/RS) use Open/High/Low/Close."
    )
    if vol_mode in {"YangZhang","Parkinson","GK","RS"} and (y_open is None or y_high is None or y_low is None):
        st.warning(f"{vol_mode} needs Open/High/Low; falling back to CloseEWMA.")
        vol_mode = "CloseEWMA"
    stochastic_vol = st.checkbox("Stochastic volatility (mean-reverting)", value=False)
    use_fixed_seed = st.checkbox("Use fixed seed = 42", value=False)
    if use_fixed_seed:
        seed = 42
    else:
        seed = int(st.number_input("Seed (for reproducibility)", min_value=0, max_value=2**32-1, value=12345, step=1))

# Slice last N
x = x[-range_days:]
y_close = y_close[-range_days:]
y_sma = y_sma[-range_days:]

y_open = y_open[-range_days:]
y_high = y_high[-range_days:]
y_low  = y_low[-range_days:]

y_sma20_for_drift = (
    pd.Series(y_close, dtype="float64")
      .rolling(20, min_periods=10)
      .mean()
      .to_numpy()
)

# =========================
# Plot
# =========================
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y_close, mode="lines", name="Close",
                         line=dict(color=th["close"])))
if show_sma:
    fig.add_trace(go.Scatter(x=x, y=y_sma, mode="lines", name="SMA200",
                             line=dict(color=th["sma200"])))

shapes: List[dict] = []
annotations: List[dict] = []
if show_stop and np.isfinite(stop_for_band):
    shapes.append(dict(type="line", xref="x", yref="y", x0=x.min(), x1=x.max(),
                       y0=stop_for_band, y1=stop_for_band,
                       line=dict(dash="dash", color=th["stop"])))
    annotations.append(dict(x=x.max(), y=stop_for_band, xref="x", yref="y",
                            text=f"Stop {stop_for_band:.2f}", showarrow=False, xanchor="left",
                            font=dict(color=th["stop"])))

if show_target and np.isfinite(tgt_for_band):
    shapes.append(dict(type="line", xref="x", yref="y", x0=x.min(), x1=x.max(),
                       y0=tgt_for_band, y1=tgt_for_band,
                       line=dict(dash="dot", color=th["target"])))
    annotations.append(dict(x=x.max(), y=tgt_for_band, xref="x", yref="y",
                            text=f"Target {tgt_for_band:.2f}", showarrow=False, xanchor="left",
                            font=dict(color=th["target"])))

# Risk band fill between stop and target (only if both finite)
if np.isfinite(stop_for_band) and np.isfinite(tgt_for_band):
    # build two filled traces (down then up) to create a band
    xs = pd.to_datetime([x.min(), x.max()])
    y_low  = [min(stop_for_band, tgt_for_band)] * 2
    y_high = [max(stop_for_band, tgt_for_band)] * 2

    fig.add_trace(go.Scatter(
        x=list(xs)+list(xs[::-1]),
        y=y_high + y_low[::-1],
        fill='toself', mode='lines', name='Risk Band',
        line=dict(width=0),
        fillcolor=th.get("risk_band", th["proj_band"]),
        opacity=0.25,
        hoverinfo="skip",
        showlegend=False,
    ))

# Apply overlays (with per-overlay params from UI)
for key in chosen_overlays:
    entry = TECHNICALS_REGISTRY.get(key)
    if not entry:
        continue
    params = overlay_params.get(key, entry.get("params", {}))
    res = entry["fn"](x, pd.Series(y_close), pd.Series(y_sma), row, params)
    for tr in (res.traces or []):
        # default overlay color if unset
        if not getattr(tr, "line", None) or not getattr(tr.line, "color", None):
            tr.line = dict(color=th["overlay"])
        fig.add_trace(tr)
    if res.shapes:
        shapes.extend(res.shapes)
    if res.annotations:
        annotations.extend(res.annotations)

# Projection (based on the sliced series you’re viewing)
if show_projection:
    last_date = pd.to_datetime(x.iloc[-1]) if hasattr(x, "iloc") else pd.to_datetime(x[-1])
    fut_dates, med, low, high = _project_next_month(
        y_close=y_close,
        start_date=last_date,
        sims=int(proj_sims),
        pct_low=pct_low, pct_high=pct_high,
        model=model_choice,
        seed=seed,
        window=window, lam=lam, df_t=df_t,
        antithetic=antithetic, block=int(block),
        horizon_months=int(proj_months),
        vol_mode=vol_mode,
        y_open=y_open, y_high=y_high, y_low=y_low,
        stochastic_vol=stochastic_vol,
        y_sma_short_for_drift=y_sma20_for_drift,
        y_sma_long_for_drift=y_sma,        
        sma_short_weight=0.9,
        sma_long_weight=0.1,
    )

    # Band fill between low/high
    fig.add_trace(go.Scatter(
        x=list(fut_dates) + list(fut_dates[::-1]),
        y=list(high) + list(low[::-1]),
        fill="toself", mode="lines", line=dict(width=0),
        fillcolor=st.session_state.plot_theme["proj_band"], opacity=0.25,
        name=f"Projection {int(pct_low)}–{int(pct_high)}",
        showlegend=True
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=fut_dates, y=med, mode="lines",
        line=dict(color=st.session_state.plot_theme["proj_mid"], dash="dash"),
        name="Projection median"
    ))

fig.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"{row.get(ticker_col, 'Ticker')} — Close & SMA200",
    shapes=shapes,
    annotations=annotations,
    hovermode="x unified",
)

# =========================
# Buy-signal Engine (UI + call)
# =========================
st.subheader("Buy-Signal Engine")

# Risk profile presets
profile = st.radio("Profile", ["Conservative", "Balanced", "Aggressive"], horizontal=True, index=1)

# Defaults by profile
preset = {
    "Conservative": dict(composite_threshold=0.70, w_rsi=0.20, w_trend=0.35, w_breakout=0.25, w_value=0.15, w_flow=0.05,
                         rsi_buy_max=40.0, vol_ratio_min=1.75, atr_mult=2.0, stop_pct=12.0, reward_R=float(reward_R),
                         # sell
                         sell_threshold=0.65, rsi_overbought_min=72.0, ema_fast_span=21, sma_mid_window=50,
                         donch_lookback_sell=20, gap_down_min_pct=0.5),
    "Balanced": dict(
        composite_threshold=0.60,
        # BUY weights
        w_rsi=0.20, w_trend=0.20, w_value=0.10, w_flow=0.05,
        w_bbands=0.20, w_donchian=0.25, w_breakout=0.00,
        # BUY knobs
        rsi_buy_max=45.0, vol_ratio_min=1.50, atr_mult=1.5, stop_pct=10.0, reward_R=float(reward_R),
        donch_lookback=20, gap_min_pct=0.5,
        # SELL thresholds/knobs
        sell_threshold=0.60, rsi_overbought_min=70.0,
        ema_fast_span=21, sma_mid_window=50, donch_lookback_sell=20, gap_down_min_pct=0.5,
        # SELL weights
        w_rsi_sell=0.30, w_trend_down=0.30, w_breakdown=0.00, w_exhaustion=0.10, w_flow_out=0.05,
        w_bbands_sell=0.25, w_donchian_sell=0.25,
    ),
    "Aggressive":   dict(composite_threshold=0.50, w_rsi=0.30, w_trend=0.20, w_breakout=0.30, w_value=0.10, w_flow=0.10,
                         rsi_buy_max=50.0, vol_ratio_min=1.25, atr_mult=1.2, stop_pct=9.0,  reward_R=float(reward_R),
                         sell_threshold=0.55, rsi_overbought_min=68.0, ema_fast_span=13, sma_mid_window=34,
                         donch_lookback_sell=15, gap_down_min_pct=0.3),
}[profile]

with st.expander("Tune parameters", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.popover("More on BUY thresholds").markdown(
            "- **Composite threshold**: minimum composite to fire BUY.\n"
            "- **RSI buy max**: cap used in `sigmoid((rsi_buy_max − RSI)/5)`.\n"
            "- **Min vol / avg20**: flow gate: `sigmoid((vol/avg20 − min)/0.4)`.\n"
            "- **Donchian lookback**: prior N-day high for breakout & ATR-scaled strength.\n"
            "- **Gap-up min %**: adds small bonus if open ≥ prev high × (1 + %)."
        )
        composite_threshold = st.slider("BUY: Composite threshold", 0.30, 0.90, float(preset["composite_threshold"]), 0.01, help=h("composite_threshold"))
        rsi_buy_max        = st.slider("BUY: RSI buy max", 20, 70, int(preset["rsi_buy_max"]), 1, help=h("rsi_buy_max"))
        vol_ratio_min      = st.slider("Min vol / avg20 (both)", 1.0, 3.0, float(preset["vol_ratio_min"]), 0.05, help=h("vol_ratio_min"))
        donch_lookback     = st.slider("BUY: Donchian lookback", 10, 60, 20, 1, help=h("donch_lookback"))
        gap_min_pct        = st.slider("BUY: Gap-up min % (vs prev high)", 0.0, 3.0, 0.5, 0.1, help=h("gap_min_pct"))

    with c2:
        st.popover("More on Stops, Value & Bollinger").markdown(
            "- **Engine stop selection**: chooses the *highest* valid stop below last from:\n"
            "  - ATR stop = `last − ATR14 × atr_mult`\n"
            "  - % fallback = `last × (1 − stop_pct/100)` (only if ATR unavailable)\n"
            "  - Bollinger lower band (20, k)\n"
            "  - 10-bar swing low\n"
            "- **Value center vs SMA200**: value score favors discounts vs SMA200.\n"
            "  - `dev_pct = (last / SMA200 − 1) × 100`.\n"
            "  - Score ≈ `sigmoid(((-dev_pct) − |value_center|)/2.5)`.\n"
            "  - More negative center → stronger preference for buying below SMA200.\n"
            "- **Trend window (buy)**: window used for normalized price & SMA200 slopes.\n"
            "  - Slopes feed `sigmoid(…×500)` and `sigmoid(…×1500)` inside Trend.\n"
            "- **Bollinger window / k**: drives %B & bandwidth\n"
            "  - Wider bands (higher *k*) → slower %B changes.\n"
            "- **Use engine stop/target**: if ON, chart shows engine’s computed stop/target\n"
            "  (ATR/swing/band/% fallback) instead of manual risk band."
        )
        atr_mult        = st.slider("BUY: ATR stop ×", 0.8, 3.0, float(preset["atr_mult"]), 0.1, help=h("atr_mult"))
        stop_pct_ui     = st.slider("BUY: Fallback % stop", 3.0, 20.0, float(preset["stop_pct"]), 0.5, help=h("stop_pct_ui"))
        value_center    = st.slider("BUY: Value center vs SMA200 (%)", -20.0, 10.0, -5.0, 0.5, help=h("value_center"))
        sma_window      = st.slider("Trend window (days, buy)", 100, 300, 200, 10, help=h("sma_window"))
        use_engine_stop = st.checkbox("Use engine stop/target on chart", value=False, help=h("use_engine_stop"))
        bb_window       = st.slider("Bollinger window", 10, 60, int(preset.get("bb_window", 20)), 1, help=h("bb_window"))
        bb_k            = st.slider("Bollinger k (σ)", 1.0, 3.0, float(preset.get("bb_k", 2.0)), 0.1, help=h("bb_k"))

    with c3:
        st.popover("More on BUY weights & blend").markdown(
            "- **Composite** = Σ(weights × component_scores) + gap_bonus, then clipped to [0,1].\n"
            "- **RSI**: oversold (low RSI) boosts `s_rsi = sigmoid((rsi_buy_max − RSI)/5)`.\n"
            "- **Trend**: 0.5×(above SMA200) + 0.25×sigmoid(price-slope×500) + 0.25×sigmoid(SMA200-slope×1500).\n"
            "- **Value**: favors discounts vs SMA200 using your *value_center*.\n"
            "- **Flow**: `sigmoid((vol/avg20 − vol_ratio_min)/0.4)`; needs vol & 20-day avg vol.\n"
            "- **Bollinger %B**: BUY uses `1 − %B` (near/below lower band → higher score).\n"
            "- **Donchian**: strength = `sigmoid(((last − prev_N_high)/ATR) / 0.5)`.\n"
            "- **Legacy breakout**: boolean (last ≥ prev_N_high). Keep small/0 if using Donchian to avoid double-counting.\n"
            "- Tip: weights don't have to sum to 1, but keeping ~1.0 makes threshold intuition easier."
        )
        w_rsi   = st.slider("Weight (BUY): RSI", 0.0, 1.0, float(preset["w_rsi"]), 0.05, help=h("w_rsi"))
        w_trend = st.slider("Weight (BUY): Trend", 0.0, 1.0, float(preset["w_trend"]), 0.05, help=h("w_trend"))
        w_value = st.slider("Weight (BUY): Value", 0.0, 1.0, float(preset["w_value"]), 0.05, help=h("w_value"))
        w_flow  = st.slider("Weight (BUY): Flow", 0.0, 1.0, float(preset["w_flow"]), 0.05, help=h("w_flow"))
        w_bbands = st.slider("Weight (BUY): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands", 0.20)), 0.05, help=h("w_bbands"))
        w_donch  = st.slider("Weight (BUY): Donchian",     0.0, 1.0, float(preset.get("w_donchian", 0.25)), 0.05, help=h("w_donch"))
        w_break  = st.slider("Weight (BUY): Legacy breakout", 0.0, 1.0, float(preset.get("w_breakout", 0.00)), 0.05, help=h("w_break"))

    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.popover("More on SELL thresholds").markdown(
            "- **SELL composite threshold**: SELL triggers when SELL composite ≥ this.\n"
            "- **RSI overbought min**: `s_rsi ≈ sigmoid((RSI − threshold)/5)`; boosted if RSI rolls over.\n"
            "- **Donchian lookback (SELL)**: prior N-day *low* (yesterday’s channel) used for breakdown checks\n"
            "  and ATR-scaled SELL strength: `sigmoid(((prev_N_low − last)/ATR) / 0.5)`."
        )
        sell_threshold      = st.slider("SELL: Composite threshold", 0.30, 0.90, float(preset["sell_threshold"]), 0.01, help=h("sell_threshold"))
        rsi_overbought_min  = st.slider("SELL: RSI overbought min", 55, 85, int(preset["rsi_overbought_min"]), 1, help=h("rsi_overbought_min"))
        donch_lookback_sell = st.slider("SELL: Donchian lookback", 10, 60, int(preset["donch_lookback_sell"]), 1, help=h("donch_lookback_sell"))

    with s2:
        st.popover("More on SELL trend & gaps").markdown(
            "- **EMA fast span / SMA mid window**: used for cross-under checks and slopes.\n"
           "  - Trend-down score parts: 0.4×(below EMA_fast) + 0.2×(below SMA_mid)\n"
            "    + 0.2×sigmoid(−EMA_slope×500) + 0.2×sigmoid(−SMA_slope×1500).\n"
            "- **Gap-down min %**: extra SELL bonus if open < yesterday’s low by ≥ this %."
        )
        ema_fast_span   = st.slider("SELL: EMA fast span", 5, 55, int(preset["ema_fast_span"]), 1, help=h("ema_fast_span"))
        sma_mid_window  = st.slider("SELL: SMA mid window", 20, 100, int(preset["sma_mid_window"]), 1, help=h("sma_mid_window"))
        gap_down_min_pct = st.slider("SELL: Gap-down min % (vs prev low)", 0.0, 3.0, float(preset["gap_down_min_pct"]), 0.1, help=h("gap_down_min_pct"))

    with s3:
        st.popover("More on SELL weights & components").markdown(
            "- **RSI**: overbought/rollover emphasis.\n"
            "- **Trend down**: cross-under + negative slopes (EMA/SMA).\n"
            "- **Breakdown**: boolean (last ≤ prior N-day low).\n"
            "- **Exhaustion**: boolean (last > Bollinger upper band).\n"
            "- **Flow out**: down candle with vol/avg20 ≥ threshold.\n"
            "- **%B (SELL)**: uses `%B` directly (near upper band → higher SELL score).\n"
            "- **Donchian SELL**: ATR-scaled breakdown strength.\n"
            "- Tip: If you raise **Donchian SELL**, consider lowering **Breakdown** to avoid overweighting the same event."
        )
        w_rsi_sell   = st.slider("Weight (SELL): RSI", 0.0, 1.0, 0.30, 0.05, help=h("w_rsi_sell"))
        w_trend_down = st.slider("Weight (SELL): Trend down", 0.0, 1.0, 0.30, 0.05, help=h("w_trend_down"))
        w_breakdown  = st.slider("Weight (SELL): Breakdown", 0.0, 1.0, 0.25, 0.05, help=h("w_breakdown"))
        w_exhaustion = st.slider("Weight (SELL): Exhaustion", 0.0, 1.0, 0.10, 0.05, help=h("w_exhaustion"))
        w_flow_out   = st.slider("Weight (SELL): Flow out", 0.0, 1.0, 0.05, 0.05, help=h("w_flow_out"))
        w_bbands_sell = st.slider("Weight (SELL): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands_sell", 0.25)), 0.05, help=h("w_bbands_sell"))
        w_donch_sell  = st.slider("Weight (SELL): Donchian",     0.0, 1.0, float(preset.get("w_donchian_sell", 0.25)), 0.05, help=h("w_donch_sell"))

    # sizing & guardrails (shared)
    c4, c5, c6 = st.columns(3)
    with c4:
        portfolio_value = st.number_input("Portfolio ($)", min_value=1000.0, value=20000.0, step=1000.0)
    with c5:
        risk_per_trade_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    with c6:
        min_adv_dollars = st.number_input("Min ADV$ (liquidity)", min_value=0.0, value=250000.0, step=25000.0)

params = BuyParams(
    composite_threshold=float(composite_threshold),
    w_rsi=float(w_rsi), w_trend=float(w_trend), w_breakout=float(w_break), w_value=float(w_value), w_flow=float(w_flow),
    rsi_buy_max=float(rsi_buy_max), rsi_floor=20.0,
    sma200_window=int(sma_window),
    donch_lookback=int(donch_lookback), gap_min_pct=float(gap_min_pct),
    value_center_dev_pct=float(value_center), vol_ratio_min=float(vol_ratio_min),
    use_engine_stop=bool(use_engine_stop), atr_mult=float(atr_mult), stop_pct=float(stop_pct_ui),
    reward_R=float(reward_R),
    portfolio_value=float(portfolio_value), risk_per_trade_pct=float(risk_per_trade_pct),
    min_price=1.0, min_adv_dollars=float(min_adv_dollars),
    w_bbands=float(w_bbands), w_donchian=float(w_donch),
    bb_window=int(bb_window), bb_k=float(bb_k),
    w_bbands_sell=float(w_bbands_sell), w_donchian_sell=float(w_donch_sell),

    # SELL-side knobs
    sell_threshold=float(sell_threshold),
    w_rsi_sell=float(w_rsi_sell), w_trend_down=float(w_trend_down), w_breakdown=float(w_breakdown),
    w_exhaustion=float(w_exhaustion), w_flow_out=float(w_flow_out),
    rsi_overbought_min=float(rsi_overbought_min),
    ema_fast_span=int(ema_fast_span), sma_mid_window=int(sma_mid_window),
    donch_lookback_sell=int(donch_lookback_sell), gap_down_min_pct=float(gap_down_min_pct),
)

# Run engines for the selected row
buy_res  = compute_buy_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
sell_res = compute_sell_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)

# === Historic signals + DCA backtest ===
with st.expander("Backtest & DCA simulator", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        starting_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0)
        buy_pct_first = st.slider("Buy % on first BUY", 0.0, 100.0, 25.0, 5.0)
        buy_pct_next  = st.slider("Buy % on next BUY",  0.0, 100.0, 25.0, 5.0)
    with c2:
        dca_trigger_drop_pct = st.slider("Extra BUY only if price ≤ last buy by (%)", 0.0, 50.0, 5.0, 1.0)
        max_dca_legs = st.slider("Max DCA legs per accumulation", 0, 10, 3, 1)
    with c3:
        sell_pct_first = st.slider("Sell % on first SELL", 0.0, 100.0, 50.0, 5.0)
        sell_pct_next  = st.slider("Sell % on next SELL",  0.0, 100.0, 50.0, 5.0)

    # Compute historical signal series (uses current slice)
    buy_idx, sell_idx = compute_signal_series_for_row(
        row=row, x=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params
    )

    # Markers: show if user ran batch OR simply because we're in this panel
    if st.session_state.get("enable_hist_markers", False) or True:
        if len(buy_idx):
            fig.add_trace(go.Scatter(
                x=x[buy_idx], y=y_close[buy_idx], mode="markers",
                name="BUY (hist)",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                hovertext=["BUY"]*len(buy_idx), hoverinfo="x+y+text",
            ))
        if len(sell_idx):
            fig.add_trace(go.Scatter(
                x=x[sell_idx], y=y_close[sell_idx], mode="markers",
                name="SELL (hist)",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                hovertext=["SELL"]*len(sell_idx), hoverinfo="x+y+text",
            ))

    # DCA backtest
    equity, trades, total_ret = run_dca_backtest(
        dates=x, close=y_close, buy_idx=buy_idx, sell_idx=sell_idx,
        starting_cash=starting_cash,
        buy_pct_first=buy_pct_first, buy_pct_next=buy_pct_next,
        dca_trigger_drop_pct=dca_trigger_drop_pct, max_dca_legs=max_dca_legs,
        sell_pct_first=sell_pct_first, sell_pct_next=sell_pct_next,
    )

    # Add equity curve to secondary Y-axis
    fig.update_layout(
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Strategy equity", overlaying="y", side="right", showgrid=False),
    )
    fig.add_trace(go.Scatter(
        x=x, y=equity, mode="lines", name="Strategy equity", yaxis="y2",
        line=dict(dash="dash")
    ))

    st.metric("DCA backtest return", f"{total_ret*100:.1f}%")
    with st.expander("Executed trades", expanded=False):
        if trades:
            tdf = pd.DataFrame(trades)
            st.dataframe(tdf[["date","side","price","qty","cash","shares"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No trades executed by this configuration.")

# Decide final action
def _decide_action(buy_res, sell_res, buy_thr, sell_thr):
    m_buy  = buy_res["score"]  - float(buy_thr)
    m_sell = sell_res["score"] - float(sell_thr)
    if buy_res["buy"] and not sell_res["sell"]:
        return "BUY", m_buy
    if sell_res["sell"] and not buy_res["buy"]:
        return "SELL", m_sell
    if buy_res["buy"] and sell_res["sell"]:
        return ("BUY", m_buy) if m_buy >= m_sell else ("SELL", m_sell)
    return "HOLD", max(m_buy, m_sell)

action, action_margin = _decide_action(buy_res, sell_res, params.composite_threshold, params.sell_threshold)

# Optionally override the risk band with engine stop/target
if params.use_engine_stop and buy_res["stop"] and buy_res["target"]:
    stop_for_band = buy_res["stop"]
    tgt_for_band = buy_res["target"]

# Quick metrics
e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("Action", action + (" ✅" if action != "HOLD" else ""))
e2.metric("Buy score", f"{buy_res['score']:.2f}")
e3.metric("Sell score", f"{sell_res['score']:.2f}")
e4.metric("Stop", f"{buy_res['stop']:.2f}" if buy_res["stop"] else "—", help=f"Basis: {buy_res.get('stop_basis','')}")
e5.metric("Target", f"{buy_res['target']:.2f}" if buy_res["target"] else "—")

with st.expander("Buy engine details", expanded=False):
    st.write("Components (0–1):", buy_res["components"])
    st.write("Features:", buy_res["features"])
    st.write("Guardrails OK:", buy_res["guards_ok"])
    if buy_res["guard_reasons"]:
        st.warning("Guards triggered: " + "; ".join(buy_res["guard_reasons"]))
    if buy_res.get("adv_dollars") is not None:
        st.caption(f"ADV$: ${buy_res['adv_dollars']:,.0f}")

with st.expander("Sell engine details", expanded=False):
    st.write("Components (0–1):", sell_res["components"])
    st.write("Features:", sell_res["features"])
    if sell_res["reasons"]:
        st.warning("Triggers: " + "; ".join(sell_res["reasons"]))

# Add BUY/SELL markers to the chart
try:
    if action == "BUY" or buy_res["buy"]:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y_close[-1]], mode="markers",
            name="BUY", marker=dict(symbol="triangle-up", size=12, color=th["overlay"])
        ))
        annotations.append(dict(
            x=x[-1], y=y_close[-1], xref="x", yref="y",
            text="BUY", showarrow=True, arrowhead=2, ax=0, ay=-25, font=dict(color=th["overlay"])
        ))
    if action == "SELL" or sell_res["sell"]:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y_close[-1]], mode="markers",
            name="SELL", marker=dict(symbol="triangle-down", size=12, color=th["stop"])
        ))
        annotations.append(dict(
            x=x[-1], y=y_close[-1], xref="x", yref="y",
            text="SELL", showarrow=True, arrowhead=2, ax=0, ay=25, font=dict(color=th["stop"])
        ))
except Exception:
    pass

# Re-attach updated annotations/shapes after adding the markers
fig.update_layout(annotations=annotations, shapes=shapes)

with st.expander("Batch: compute Action for all rows", expanded=False):
    if st.button("Compute actions for table"):
        def _row_action(r: pd.Series):
            sx, sc, ss, so, sh, sl = _get_series_lists(
                r, dates_series_col, close_series_col, sma200_series_col, open_series_col, high_series_col, low_series_col
            )
            if sx is None:
                return pd.Series({
                    "engine_action": "NA",
                    "engine_buy_score": np.nan,
                    "engine_sell_score": np.nan,
                    "engine_stop": np.nan,
                    "engine_target": np.nan,
                    "engine_shares": np.nan,
                })
            b = compute_buy_signal(r, sx, sc, ss, so, sh, sl, params)
            s = compute_sell_signal(r, sx, sc, ss, so, sh, sl, params)
            act, _ = _decide_action(b, s, params.composite_threshold, params.sell_threshold)
            return pd.Series({
                "engine_action": act,
                "engine_buy_score": b["score"],
                "engine_sell_score": s["score"],
                "engine_stop": b["stop"],
                "engine_target": b["target"],
                "engine_shares": b["shares"],
            })

        add = df_view.apply(_row_action, axis=1)
        df_view = pd.concat([df_view, add], axis=1)
        st.success("Computed actions for current filtered rows.")
        show_cols = chosen_cols + [c for c in ["engine_action","engine_buy_score","engine_sell_score","engine_stop","engine_target","engine_shares"] if c not in chosen_cols]
        st.dataframe(df_view[show_cols], use_container_width=True, hide_index=True)

        st.session_state["enable_hist_markers"] = True
        st.rerun()

st.plotly_chart(fig, use_container_width=True)

# =========================
# Metrics & details
# =========================
m1, m2, m3, m4, m5 = st.columns(5)
fmt = lambda v, d=2: "" if pd.isna(v) else f"{float(v):,.{d}f}"
m1.metric("Composite", fmt(row.get(comp_col)))
m2.metric("Signals", fmt(row.get(sig_col), 0))
m3.metric("RSI", fmt(row.get(rsi_col)))
m4.metric("Last", fmt(row.get(last_col)))
m5.metric("Owned", "Yes" if bool(row.get(owned_col, False)) else "No")

with st.expander("Raw row data"):
    st.json({c: (row[c] if pd.notna(row[c]) else None) for c in df_view.columns})

# =========================
# Profiles (save/load)
# =========================
st.sidebar.divider()
st.sidebar.subheader("Filter profiles")

profile_name = st.sidebar.text_input("Profile name", value="default")

# Build the payload from current UI state
def _current_profile_payload() -> dict:
    return {
        # built-in filters
        "min_sig":     int(min_sig),
        "min_comp":    float(min_comp),
        "rsi_min":     int(st.session_state["rsi_range"][0]),
        "rsi_max":     int(st.session_state["rsi_range"][1]),
        "owned_only":  bool(owned_only),
        "search":      search,

        # custom filter builder
        "custom_rules": st.session_state.get("custom_rules", []),
        "logical":      logical,

        # sort + table
        "sort_choice":  st.session_state.get("sort_choice"),
        "sort_asc":     bool(st.session_state.get("sort_asc", True)),
        "table_cols":   st.session_state.get("table_cols", default_cols),

        # chart options
        "show_sma":     bool(st.session_state.get("show_sma", True)),
        "show_stop":    bool(st.session_state.get("show_stop", False)),
        "show_target":  bool(st.session_state.get("show_target", False)),
        "range_days":   int(st.session_state.get("range_days", 180)),

        # overlays + their params as currently set in the UI
        "overlays":     st.session_state.get("overlays", []),
        "overlay_params": overlay_params,  # whatever was built in the UI loop

        # (optional) remember data source choice & path
        "source_choice": source_choice,
        "default_path":  default_path,
    }

colA, colB = st.sidebar.columns(2)
if colA.button("Save profile"):
    payload = _current_profile_payload()
    save_profile(profile_name, payload)
    st.sidebar.success(f"Saved profile '{profile_name}'")

existing = list_profiles()
chosen_profile = st.sidebar.selectbox("Load profile", options=["(select)"] + existing)

def _apply_profile_to_session(prof: dict):
    # restore custom rules first (no widget keys involved)
    st.session_state["custom_rules"] = prof.get("custom_rules", [])

    # restore sort/table/chart/overlays via widget keys
    st.session_state["sort_choice"] = prof.get("sort_choice", st.session_state.get("sort_choice"))
    st.session_state["sort_asc"]    = prof.get("sort_asc", st.session_state.get("sort_asc", True))
    st.session_state["table_cols"]  = prof.get("table_cols", st.session_state.get("table_cols", []))

    st.session_state["show_sma"]    = prof.get("show_sma", st.session_state.get("show_sma", True))
    st.session_state["show_stop"]   = prof.get("show_stop", st.session_state.get("show_stop", False))
    st.session_state["show_target"] = prof.get("show_target", st.session_state.get("show_target", False))
    st.session_state["range_days"]  = prof.get("range_days", st.session_state.get("range_days", 180))

    st.session_state["overlays"]    = prof.get("overlays", st.session_state.get("overlays", []))
    # If overlay_params were saved, stash to session for next render. We’ll read them when building the param UI.
    st.session_state["overlay_params_loaded"] = prof.get("overlay_params", {})

    # (optional) restore data source choice & path
    # These aren’t strictly required, but handy if you want to sticky the same file/workflow
    st.session_state["source_choice"] = prof.get("source_choice", st.session_state.get("source_choice", "Latest generated"))
    st.session_state["default_path"]  = prof.get("default_path", st.session_state.get("default_path", default_path))

    # restore simple numeric/text widgets via query params to keep UI consistent on rerun
    st.session_state["min_sig"]   = prof.get("min_sig", 0)
    st.session_state["min_comp"]  = prof.get("min_comp", 0.0)
    st.session_state["rsi_range"] = (prof.get("rsi_min", 0), prof.get("rsi_max", 100))
    st.session_state["owned_only"]= bool(prof.get("owned_only", 0))
    st.session_state["search"]    = prof.get("search", "")

if st.sidebar.button("Load", disabled=(chosen_profile == "(select)")):
    prof = load_profile(chosen_profile)
    if prof:
        _apply_profile_to_session(prof)
        st.rerun()

# =========================
# Export filtered CSV
# =========================
st.download_button(
    "Download filtered CSV",
    df_view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_tickers.csv",
    mime="text/csv",
)
