# app.py
from __future__ import annotations

import ast
import io
import json
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

def _get_series_lists(row: pd.Series, dates_col: str, close_col: str, sma200_col: str):
    dates = _to_list(row[dates_col])
    close = _to_list(row[close_col])
    sma   = _to_list(row[sma200_col])
    if not dates or not close or not sma:
        return None, None, None
    n = min(len(dates), len(close), len(sma))
    if n == 0:
        return None, None, None
    x = pd.to_datetime(dates[:n], errors="coerce")
    mask = x.notna()
    return x[mask], np.asarray(close[:n], dtype=float)[mask], np.asarray(sma[:n], dtype=float)[mask]

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

def _project_next_month(
    y_close: np.ndarray,
    start_date: pd.Timestamp,
    horizon_days: int = 22,
    sims: int = 400,
    pct_low: float = 10.0,
    pct_high: float = 90.0,
    model: str = "EWMA+t",          # "GBM", "EWMA+t", "Bootstrap", "GARCH(t)", "Jump"
    seed: Optional[int] = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = pd.Series(y_close, dtype="float64").clip(lower=1e-9)
    if len(close) < 20:
        future = pd.bdate_range(start=start_date, periods=horizon_days+1)[1:]
        last = float(close.iloc[-1])
        med = np.full(len(future), last)
        return future, med, med, med

    r = np.log(close).diff().dropna().values
    last = float(close.iloc[-1])
    T, N = horizon_days, int(sims)

    if model == "GBM":
        mu = np.mean(r) * 252.0
        sigma = np.std(r, ddof=1) * np.sqrt(252.0)
        dt = 1/252
        mu_d = 0.25*mu*dt   # shrink drift
        sigma_d = sigma * np.sqrt(dt)
        Z = rng.standard_normal((T, N))
        steps = (mu_d - 0.5*sigma_d**2) + sigma_d*Z
        paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "EWMA+t":
        lam = 0.94
        r_ser = pd.Series(r)

        # Most recent EWMA variance; need min_periods>=2 for a defined variance
        ewma = r_ser.ewm(alpha=(1 - lam), adjust=False, min_periods=2).var()
        ewma_var = float(ewma.iloc[-1])

        # Fallback if EWMA variance is NaN or 0
        if not np.isfinite(ewma_var) or ewma_var <= 0:
            ewma_var = float(np.var(r, ddof=1))

        sigma = np.sqrt(ewma_var) * np.sqrt(252.0)

        dt = 1 / 252.0
        mu_d = 0.0  # keep drift neutral
        sigma_d = sigma * np.sqrt(dt)

        # fat tails via standardized t; variance-normalized to 1
        df_t = 5
        Z = rng.standard_t(df_t, size=(T, N)) / np.sqrt(df_t / (df_t - 2))

        steps = (mu_d - 0.5 * sigma_d**2) + sigma_d * Z
        paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Bootstrap":
        # block bootstrap on returns
        def block_bootstrap(returns, T, sims, block=5):
            R = np.empty((T, sims))
            idx = np.arange(len(returns)-block)
            for j in range(sims):
                path = []
                while len(path) < T:
                    start = rng.choice(idx)
                    path.extend(returns[start:start+block].tolist())
                R[:, j] = path[:T]
            return R
        steps = block_bootstrap(r, T, N, block=5)
        paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Jump":
        # GBM + rare jumps
        mu = np.mean(r) * 252.0
        sigma = np.std(r, ddof=1) * np.sqrt(252.0)
        dt = 1/252
        mu_d = 0.0
        sigma_d = sigma * np.sqrt(dt)
        Z = rng.standard_normal((T, N))
        lambda_jump, mu_J, sigma_J = 3.0, -0.02, 0.06
        Njump = rng.poisson(lam=lambda_jump*dt, size=(T, N))
        J = rng.normal(mu_J, sigma_J, size=(T, N)) * Njump
        steps = (mu_d - 0.5*sigma_d**2) + sigma_d*Z + J
        paths = last * np.exp(np.cumsum(steps, axis=0))

    else:
        # fallback to GBM
        return _project_next_month(y_close, start_date, horizon_days, sims, pct_low, pct_high, model="GBM", seed=seed)

    # summarize
    low  = np.nanpercentile(paths, pct_low, axis=1)
    high = np.nanpercentile(paths, pct_high, axis=1)
    med  = np.nanpercentile(paths, 50, axis=1)
    future = pd.bdate_range(start=start_date, periods=horizon_days+1)[1:]
    return future, med, low, high


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

required_any = [ticker_col, dates_series_col, close_series_col, sma200_series_col]
if any(c is None for c in required_any):
    st.error(
        "Data is missing one or more required series columns: "
        "`ticker`, `dates_series`, `close_series`, `sma200_series` "
        "(or their `series__*` equivalents)."
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
    st.session_state.custom_rules = []  # [{"col": str, "op": str, "val": str}]

cols_all = list(df.columns)
logical = st.sidebar.radio("Combine rules with", ["AND", "OR"], horizontal=True)

with st.sidebar.expander("Add rule"):
    col = st.selectbox("Column", cols_all, key="cf_col")

    if col:
        kind = _infer_col_kind(df[col])

        # Operator choices by kind
        if kind == "bool":
            op_choices = ["is true", "is false", "==", "!=", "isna", "notna"]
        elif kind == "numeric":
            op_choices = ["==", "!=", ">", ">=", "<", "<=", "isna", "notna"]
        elif kind == "datetime":
            op_choices = ["==", "!=", ">", ">=", "<", "<=", "isna", "notna"]
        else:  # string
            op_choices = ["contains", "not contains", "==", "!=", "isna", "notna"]

        op = st.selectbox("Operator", op_choices, key="cf_op")

        # Value widget based on kind & op
        if op in {"isna", "notna", "is true", "is false"}:
            val = ""  # not used
        else:
            if kind == "bool":
                # Accept a boolean token via radio; will become 'True'/'False' string
                bool_pick = st.radio("Value", ["True", "False"], horizontal=True, key="cf_val_bool")
                val = bool_pick
            elif kind == "numeric":
                val_num = st.number_input("Value", value=0.0, step=1.0, key="cf_val_num")
                val = str(val_num)
            elif kind == "datetime":
                # Allow natural typing; you can also use st.date_input if you only store dates
                val_dt = st.text_input("Value (date/time)", placeholder="e.g. 2024-12-31 or 2024-12-31 15:30", key="cf_val_dt")
                val = val_dt
            else:
                val = st.text_input("Value", key="cf_val_txt")

        if st.button("Add rule"):
            st.session_state.custom_rules.append({"col": col, "op": op, "val": val})

st.sidebar.divider()
st.sidebar.subheader("Chart colors")
th = st.session_state.plot_theme
c1, c2 = st.sidebar.columns(2)
with c1:
    th["close"]   = st.color_picker("Close",   th["close"])
    th["sma200"]  = st.color_picker("SMA200",  th["sma200"])
    th["overlay"] = st.color_picker("Overlays", th["overlay"])
with c2:
    th["stop"]     = st.color_picker("Stop",     th["stop"])
    th["target"]   = st.color_picker("Target",   th["target"])
    th["proj_mid"] = st.color_picker("Proj mid", th["proj_mid"])
th["proj_band"] = st.color_picker("Proj band fill", th["proj_band"])
th["risk_band"] = st.color_picker("Risk band fill", th.get("risk_band", "#8dd3c7"))

if st.sidebar.button("Save colors"):
    save_theme(th)
    st.sidebar.success("Saved chart colors.")

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
    for r in st.session_state.custom_rules:
        col, op, val = r["col"], r["op"], r["val"]
        if col not in df.columns or op not in OPS:
            continue

        ser, coerced_val = _coerce_series_for_rule(df[col], op, val)
        rule_mask = OPS[op](ser, coerced_val)
        masks.append(rule_mask.reindex(df.index).fillna(False))

    if masks:
        comb = masks[0]
        for m in masks[1:]:
            comb = (comb & m) if logical == "AND" else (comb | m)
        mask &= comb

df_view = df[mask].copy()

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

x, y_close, y_sma = _get_series_lists(row, dates_series_col, close_series_col, sma200_series_col)
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

st.subheader("Projections")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    show_projection = st.checkbox("Show 1-month projection", value=True)
with pc2:
    proj_band = st.selectbox("Projection band", ["10–90%", "5–95%"], index=0)
with pc3:
    proj_sims = st.number_input("Simulations", min_value=100, max_value=2000, value=400, step=100)
with pc4:
    model_choice = st.selectbox("Projection model", ["EWMA+t", "GBM", "Bootstrap", "Jump"], index=0)

seed_input = st.number_input("Random seed (optional)", value=0, step=1, key="proj_seed")
seed = None if seed_input == 0 else int(seed_input)

pct_low, pct_high = (10.0, 90.0) if proj_band == "10–90%" else (5.0, 95.0)

# Slice last N
x = x[-range_days:]
y_close = y_close[-range_days:]
y_sma = y_sma[-range_days:]

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
        y_close=np.asarray(y_close, dtype="float64"),
        start_date=last_date,
        horizon_days=22,
        sims=int(proj_sims),
        pct_low=pct_low,
        pct_high=pct_high,
        model=model_choice,   # <-- important
        seed=seed,            # <-- important
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
