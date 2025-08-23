from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.paths import PROJECT_ROOT, DEFAULT_OUTPUT
from core.io import DEFAULT_THEME,load_theme, save_theme, _run_main_and_reload, _read_path_cached, save_profile, load_profile, list_profiles
from core.utils import _resolve, _get_series_lists, _coerce_numeric, _coerce_boolish, _parse_ci_label
from core.filters import OPS, TRUE_TOKENS, FALSE_TOKENS, _infer_col_kind, _coerce_series_for_rule, _parse_datetime_value
from core.helptext import h
from core.types import BuyParams

from viz.overlays import TECHNICALS_REGISTRY

from signals.buy_engine import compute_buy_signal
from signals.sell_engine import compute_sell_signal
from signals.series import compute_signal_series_for_row
from signals.decision import _decide_action
from signals.sweeps import param_grid

from sim.backtest import run_dca_backtest
from sim.projection import _project_next_month
from sim.metrics import compute_metrics


# ────────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ticker Evaluator", layout="wide")

# Make console output UTF-8 friendly (best effort)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if "plot_theme" not in st.session_state:
    # merge persisted theme onto defaults
    st.session_state.plot_theme = {**DEFAULT_THEME, **load_theme()}
else:
    st.session_state.plot_theme = {**DEFAULT_THEME, **st.session_state.plot_theme}
th = st.session_state.plot_theme

DEFAULT_SETTINGS = {
    "chart": {
        "show_sma": True,
        "show_stop": True,
        "show_target": True,
        "range_days": 180,
        "hovermode": "x unified",  # "x unified" | "closest" | "x"
        "plot_height": 480,
        "template":"plotly_white",
    },
    "projections": {
        "enabled": True,
        "band": "10–90%",          # "10–90%" | "5–95%" | "20–80%" | "25–75%" | "Custom"
        "sims": 2000,
        "model": "EWMA+t",         # "EWMA+t" | "GBM" | "Bootstrap" | "Jump"
        "months": 1,
        # advanced defaults
        "window": 252,
        "lam": 0.94,
        "df_t": 5,
        "antithetic": True,
        "block": 5,
        "vol_mode": "YangZhang",   # "YangZhang","Parkinson","GK","RS","CloseEWMA","CloseRolling","MAD"
        "stochastic_vol": False,
        "seed_mode": "custom",     # "fixed" | "custom"
        "seed": 12345,
    },
    "overlays": {
        "defaults": ["SMA", "EMA"],
    },
    "data": {
        "source_choice": "Latest generated",        # UI sticky default
        "default_path": DEFAULT_OUTPUT.as_posix(),  # UI sticky default
    },
}

def get_settings() -> dict:
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = deepcopy(DEFAULT_SETTINGS)
    return st.session_state.app_settings

st.title("📈 Ticker Evaluator — Interactive Review")
st.caption("Load scanner CSV/XLSX, filter/sort, overlay technicals, and inspect setups. You can also re-run the scan.")


# ────────────────────────────────────────────────────────────────────────────────
# Small UI helpers 
# ────────────────────────────────────────────────────────────────────────────────
def K(ns: str, name: str) -> str:
    return f"{ns}__{name}"

def ui_run_scan_and_choose_source(default_path: str) -> tuple[Optional[io.BytesIO], str, bool]:
    """Top controls: run main.py and pick the data source."""
    c_r1, c_r2, c_r3 = st.columns([1.2, 1.2, 2])
    settings = get_settings()
    with c_r1:
        run_now = st.button("▶ Run scan (main.py)")
    with c_r2:
        source_choice = st.selectbox(
            "Data source",
            ["Latest generated", "Upload file", "Path"],
            index=["Latest generated","Upload file","Path"].index(settings["data"]["source_choice"])
        )
    with c_r3:
        path_text = st.text_input("Path (.csv or .xlsx)", value=settings["data"]["default_path"])

    settings["data"]["source_choice"] = source_choice
    settings["data"]["default_path"] = path_text

    uploaded = None
    if source_choice == "Upload file":
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if run_now:
        with st.spinner("Running main.py..."):
            ok, logs = _run_main_and_reload()
        with st.expander("Run logs", expanded=True):
            st.code(logs or "(no output)")
        st.success(f"Scan completed. Reloading: {DEFAULT_OUTPUT.as_posix()}") if ok else st.error("main.py failed. Check logs above.")

    prefer_output = (source_choice == "Latest generated")
    return uploaded, path_text, prefer_output

def _inject_chip_css():
    if st.session_state.get("_chip_css_injected_v2"):
        return
    st.session_state["_chip_css_injected_v2"] = True
    st.markdown(
        """
        <style>
        .kpi-row { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-start; margin:6px 0; }
        .kpi-chip {
          display:inline-flex; align-items:center; gap:10px;
          padding:8px 12px; border-radius:999px; font-weight:600; font-size:13px; line-height:1;
          backdrop-filter: blur(6px);
          border:1px solid rgba(255,255,255,.14);
          box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 2px 6px rgba(0,0,0,.25);
          color: var(--chip-fg, #0f172a);
        }
        .kpi-chip .val {
          font-variant-numeric: tabular-nums;
          padding:3px 8px; border-radius:999px;
          background: var(--chip-badge-bg, rgba(255,255,255,.25));
          border:1px solid rgba(255,255,255,.22);
          color: inherit;
        }
        .kpi-chip:hover { transform: translateY(-1px); transition: transform .15s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _is_dark_theme() -> bool:
    """Detect Streamlit dark/black themes."""
    try:
        base = st.get_option("theme.base") or ""
        if str(base).lower() == "dark":
            return True
        bg = st.get_option("theme.backgroundColor") or "#0e1117"
        r, g, b = _hex_to_rgb(bg)
        luminance = (0.299*r + 0.587*g + 0.114*b) / 255
        return luminance < 0.45  # treat near-black as dark
    except Exception:
        return False

def _chip_style_from_base(base_hex: str, force_white_text: bool) -> str:
    r, g, b = _hex_to_rgb(base_hex or "#22c55e")
    top = f"rgba({r},{g},{b}, .30)"
    bot = f"rgba({r},{g},{b}, .14)"
    badge = f"rgba({r},{g},{b}, .18)"
    # If on dark theme, make text white; otherwise pick based on accent luminance
    if force_white_text:
        fg = "#ffffff"
    else:
        luminance = (0.299*r + 0.587*g + 0.114*b) / 255
        fg = "#0f172a" if luminance > 0.7 else "white"
    return (
        f"background: linear-gradient(180deg, {top}, {bot});"
        f"border-color: rgba({r},{g},{b}, .45);"
        f"--chip-badge-bg:{badge};"
        f"--chip-fg:{fg};"
    )

def kpi_chip(
    label: str,
    value: str | float | int,
    base_color: str = "#22c55e",
    icon: str | None = None,
    *,
    scale: float = 1.2,
    force_white_on_dark: bool = True,
):
    """
    Pretty KPI pill with gradient & value badge.

    Args:
      label/value: content
      base_color: hex string for the accent/gradient
      icon: optional emoji (e.g., '📈')
      scale: size multiplier (1.0 normal, 1.2 = 20% larger, etc.)
      force_white_on_dark: if True, sets text to white on dark/black themes
    """
    _inject_chip_css()
    style = _chip_style_from_base(base_color, force_white_text=_is_dark_theme() if force_white_on_dark else False)

    # scale paddings / font sizes
    pad_y = max(6, int(8 * scale))
    pad_x = max(10, int(12 * scale))
    gap = max(8, int(10 * scale))
    fs = max(12, int(13 * scale))
    val_pad_y = max(2, int(3 * scale))
    val_pad_x = max(6, int(8 * scale))
    val_fs = max(11, int(12 * scale))

    icon_html = f"<span style='font-size:{fs}px; line-height:1'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div class="kpi-chip"
             style="{style}
                    padding:{pad_y}px {pad_x}px; gap:{gap}px; font-size:{fs}px;">
            {icon_html}
            <span>{label}</span>
            <span class="val" style="padding:{val_pad_y}px {val_pad_x}px; font-size:{val_fs}px;">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_row(
    chips: list[tuple[str, str | float | int, str, str | None]],
    *,
    scale: float = 1.2,
    force_white_on_dark: bool = True,
):
    """
    Render a responsive row of chips.
    chips = [(label, value, base_color_hex, icon_or_None), ...]
    """
    _inject_chip_css()
    dark = _is_dark_theme() if force_white_on_dark else False

    pad_y = max(6, int(8 * scale))
    pad_x = max(10, int(12 * scale))
    gap = max(8, int(10 * scale))
    fs = max(12, int(13 * scale))
    val_pad_y = max(2, int(3 * scale))
    val_pad_x = max(6, int(8 * scale))
    val_fs = max(11, int(12 * scale))

    html = ['<div class="kpi-row">']
    for label, value, color, icon in chips:
        r, g, b = _hex_to_rgb(color or "#22c55e")
        top = f"rgba({r},{g},{b}, .30)"; bot = f"rgba({r},{g},{b}, .14)"; badge = f"rgba({r},{g},{b}, .18)"
        if dark:
            fg = "#ffffff"
        else:
            lum = (0.299*r + 0.587*g + 0.114*b)/255
            fg = "#0f172a" if lum > 0.7 else "white"
        style = (
            f"background: linear-gradient(180deg, {top}, {bot});"
            f"border-color: rgba({r},{g},{b}, .45);"
            f"--chip-badge-bg:{badge};"
            f"--chip-fg:{fg};"
            f"padding:{pad_y}px {pad_x}px; gap:{gap}px; font-size:{fs}px;"
        )
        icon_html = f"<span style='font-size:{fs}px; line-height:1'>{icon}</span>" if icon else ""
        html.append(
            f'<div class="kpi-chip" style="{style}">{icon_html}'
            f'<span>{label}</span>'
            f'<span class="val" style="padding:{val_pad_y}px {val_pad_x}px; font-size:{val_fs}px;">{value}</span>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

def render_kpis(row: pd.Series, cols: dict):
    fmt = lambda v, d=2: "—" if (v is None or pd.isna(v)) else f"{float(v):,.{d}f}"
    comp = row.get(cols["comp"]); sig = row.get(cols["sig"]); rsi = row.get(cols["rsi"])
    last = row.get(cols["last"]); owned = row.get(cols["owned"], False)

    # simple color heuristics
    bg_comp = th["kpi_good"] if (pd.notna(comp) and float(comp) >= 0.60) else th["kpi_bad"]
    bg_rsi  = th["kpi_good"] if (pd.notna(rsi) and float(rsi) <= 45) else th["kpi_bad"]

    kpi_row([
        ("Composite", f"{row.get(cols['comp'], float('nan')):,.2f}", "#22c55e", "✅"),
        ("Signals",   f"{row.get(cols['sig'], 0):,.0f}",            "#3b82f6", "📊"),
        ("RSI",       f"{row.get(cols['rsi'], float('nan')):,.2f}", "#ef4444", "🧭"),
        ("Last",      f"{row.get(cols['last'], float('nan')):,.2f}", "#a78bfa", "💵"),
        ("Owned",     "Yes" if bool(row.get(cols['owned'], False)) else "No", "#f59e0b", "📦"),
    ], scale=1.35)

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
                xls = pd.read_excel(p, sheet_name=None)
                return xls[sheet], f"Loaded from path: {p.as_posix()} (sheet: {sheet})", (sheets, sheet)
        return res["df"], f"Loaded from path: {p.as_posix()}", (sheets, picked)

    st.info("Upload a CSV/XLSX, pick 'Latest generated', or provide a valid path.")
    st.stop()

def resolve_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    return dict(
        ticker=_resolve(cols, ("ticker", "meta__ticker")),
        date=_resolve(cols, ("date", "meta__date")),
        comp=_resolve(cols, ("composite_score", "signals_and_scores__composite_score")),
        sig=_resolve(cols, ("signals_score", "signals_and_scores__signals_score")),
        rsi=_resolve(cols, ("rsi", "momentum__rsi")),
        owned=_resolve(cols, ("owned", "meta__owned", "position__owned")),
        last=_resolve(cols, ("last", "meta__last")),
        stop=_resolve(cols, ("stop_price", "stops_and_risk__planned_stop_price")),
        tgt=_resolve(cols, ("target_price", "stops_and_risk__planned_target_price")),
        dates_series=_resolve(cols, ("dates_series", "series__dates_series")),
        close_series=_resolve(cols, ("close_series", "series__close_series")),
        sma200_series=_resolve(cols, ("sma200_series", "series__sma200_series")),
        open_series=_resolve(cols, ("open_series", "series__open_series")),
        high_series=_resolve(cols, ("high_series", "series__high_series")),
        low_series=_resolve(cols, ("low_series", "series__low_series")),
    )


def render_sidebar_filters(df: pd.DataFrame, cols: dict) -> tuple[pd.DataFrame, str]:
    """Render all sidebar filtering, custom rules, sort, and return filtered+sorted view and sort column."""
    st.sidebar.header("Filters")
    min_sig  = st.sidebar.number_input("Min signals_score", value=0, step=1, key="min_sig")
    min_comp = st.sidebar.number_input("Min composite_score", value=0.0, step=0.5, format="%.2f", key="min_comp")
    rsi_min, rsi_max = st.sidebar.slider("RSI range", 0, 100, (0, 100), key="rsi_range")
    owned_only = st.sidebar.checkbox("Owned only", value=False, key="owned_only")
    search = st.sidebar.text_input("Search ticker (substring)", value="", key="search")
    rsi_min, rsi_max = st.session_state["rsi_range"]

    # Coercions
    if cols["sig"]:  df[cols["sig"]]  = _coerce_numeric(df[cols["sig"]])
    if cols["comp"]: df[cols["comp"]] = _coerce_numeric(df[cols["comp"]])
    if cols["rsi"]:  df[cols["rsi"]]  = _coerce_numeric(df[cols["rsi"]])
    if cols["owned"]:
        df[cols["owned"]] = _coerce_boolish(df[cols["owned"]])

    mask = pd.Series(True, index=df.index)
    if cols["sig"]:
        mask &= df[cols["sig"]] >= min_sig
    if cols["comp"]:
        mask &= df[cols["comp"]] >= min_comp
    if cols["rsi"]:
        mask &= df[cols["rsi"]].between(rsi_min, rsi_max)
    if owned_only and cols["owned"]:
        mask &= df[cols["owned"]].fillna(False)
    if search and cols["ticker"]:
        s = search.strip().lower()
        mask &= df[cols["ticker"]].astype(str).str.lower().str.contains(s, na=False)

    # Custom filters
    st.sidebar.divider()
    st.sidebar.subheader("Custom filters")
    if "custom_rules" not in st.session_state:
        st.session_state["custom_rules"] = []   # list of {"col","op","val"}

    cols_all = list(df.columns)
    logical = st.sidebar.radio("Combine rules with", ["AND", "OR"], horizontal=True, key="cf_logical")

    with st.sidebar.expander("Add rule", expanded=True):
        col = st.selectbox("Column", cols_all, key="cf_col")
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

        if st.button("Add rule", key="btn_add_rule"):
            try:
                v_clean = val
                if op in {"isna", "notna", "is true", "is false"}:
                    v_clean = ""
                elif kind == "bool":
                    v_clean = "True" if str(val).strip().lower() in TRUE_TOKENS else "False" if str(val).strip().lower() in FALSE_TOKENS else val
                elif kind == "numeric":
                    _ = float(str(val))
                elif kind == "datetime":
                    _v = _parse_datetime_value(str(val))
                    if pd.isna(_v):
                        raise ValueError("Invalid datetime")

                st.session_state.custom_rules.append({"col": col, "op": op, "val": str(v_clean)})
                st.sidebar.success(f"Added: {col} {op} {v_clean}")
                st.rerun()
            except Exception as e:
                st.sidebar.warning(f"Rule not added: {e}")

    if st.session_state.custom_rules:
        for i, r in enumerate(st.session_state.custom_rules):
            st.sidebar.write(f"{i+1}. `{r['col']} {r['op']} {r['val']}`")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Clear rules"):
            st.session_state.custom_rules = []
        if c2.button("Remove last") and st.session_state.custom_rules:
            st.session_state.custom_rules.pop()

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
                if op in {"isna", "notna", "is true", "is false"}:
                    rule_mask = OPS[op](ser, None)
                else:
                    rule_mask = OPS[op](ser, coerced_val)
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

    # Sort
    st.sidebar.divider()
    st.sidebar.header("Sort")
    sort_candidates = [c for c in [cols["comp"], cols["sig"], cols["rsi"], cols["last"], cols["date"]] if c]
    sort_choice = st.sidebar.selectbox("Primary sort", sort_candidates, index=0 if sort_candidates else 0, key="sort_choice")
    ascending   = st.sidebar.checkbox("Ascending", value=(sort_choice == cols["rsi"]), key="sort_asc")

    df_view = df[mask].copy()
    if sort_choice:
        df_view = df_view.sort_values(sort_choice, ascending=ascending, kind="mergesort")

    return df_view, sort_choice or ""


def pick_row_and_series(df_view: pd.DataFrame, cols: dict):
    """Row picker + extract OHLC & SMA series."""
    if df_view.empty:
        st.info("No rows match the current filters.")
        st.stop()

    # Table columns picker (for Review tab)
    default_cols = [c for c in [cols["ticker"], cols["comp"], cols["sig"], cols["rsi"], cols["last"], cols["owned"], cols["date"]] if c]
    chosen_cols = st.multiselect("Columns to show", options=list(df_view.columns), default=default_cols, key="table_cols")

    # Row selection
    if cols["ticker"]:
        tickers = df_view[cols["ticker"]].astype(str).tolist()
        sel_ticker = st.selectbox("Select ticker", tickers, index=0 if tickers else None)
        row = df_view[df_view[cols["ticker"]].astype(str) == sel_ticker].iloc[0]
    else:
        row = df_view.iloc[0]

    # Series extraction
    x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(
        row, cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]
    )
    if x is None:
        st.warning("Selected row has empty or invalid series.")
        st.stop()

    return row, chosen_cols, x, y_close, y_sma, y_open, y_high, y_low


def theme_controls():
    """Theme color pickers (Settings tab)."""
    _prev_theme = json.dumps(th, sort_keys=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        th["close"]   = st.color_picker("Close",   th["close"])
        th["sma200"]  = st.color_picker("SMA200",  th["sma200"])
        th["overlay"] = st.color_picker("Overlays", th["overlay"])
        th["risk_band"] = st.color_picker("Risk band fill", th.get("risk_band", "#8dd3c7"))
        th["buy"]  = st.color_picker("Buy marker", th["buy"])
    with c2:
        th["stop"]     = st.color_picker("Stop",     th["stop"])
        th["target"]   = st.color_picker("Target",   th["target"])
        th["proj_mid"] = st.color_picker("Proj mid", th["proj_mid"])
        th["proj_band"] = st.color_picker("Proj band fill", th["proj_band"])
        th["sell"] = st.color_picker("Sell marker", th["sell"])
    with c3:
        th["equity"] = st.color_picker("Equity line", th["equity"])
    _new_theme = json.dumps(th, sort_keys=True)
    if _new_theme != _prev_theme:
        save_theme(th)
        st.caption("✅ Theme saved")


# ────────────────────────────────────────────────────────────────────────────────
# Data source & load (shared across tabs)
# ────────────────────────────────────────────────────────────────────────────────
uploaded, path_text, prefer_output = ui_run_scan_and_choose_source(DEFAULT_OUTPUT.as_posix())
df, loaded_msg, sheets_info = _read_any_table(uploaded, path_text, prefer_output)
st.caption(loaded_msg)

if df.empty:
    st.warning("Table is empty.")
    st.stop()

# Column resolution
cols = resolve_columns(df)
required_any = [cols["ticker"], cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]]
if any(c is None for c in required_any):
    st.error(
        "Data is missing one or more required series columns: "
        "`ticker`, `dates_series`, `close_series`, `sma200_series`, "
        "`open_series`, `high_series`, `low_series` "
        "(or their `series__*` equivalents)."
        f"Current available columns include {df.columns}"
    )
    st.stop()

# Sidebar: filters/sort (returns filtered view)
df_view, _sort_col = render_sidebar_filters(df, cols)

# ────────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────────
tab_review, tab_signals, tab_backtest, tab_settings, tab_profiles = st.tabs(
    ["📊 Review", "🧠 Signals", "🧪 Backtest", "⚙️ Settings", "💾 Profiles"]
)

# ────────────────────────────────────────────────────────────────────────────────
# 📊 REVIEW TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_review:
    st.subheader("Results")
    # Row & series selection + table (table displayed here)
    row, chosen_cols, x, y_close, y_sma, y_open, y_high, y_low = pick_row_and_series(df_view, cols)

    column_config = {}
    if cols["comp"] and cols["comp"] in df_view.columns:
        column_config[cols["comp"]] = st.column_config.ProgressColumn(
            "Composite", help="0–1", min_value=0.0, max_value=1.0, format="%.2f"
        )
    if cols["sig"] and cols["sig"] in df_view.columns:
        column_config[cols["sig"]] = st.column_config.ProgressColumn(
            "Signals", help="Composite sub-score", min_value=0.0, max_value=1.0, format="%.2f"
        )
    if cols["rsi"] and cols["rsi"] in df_view.columns:
        column_config[cols["rsi"]] = st.column_config.ProgressColumn(
            "RSI", min_value=0, max_value=100, format="%d"
        )
    if cols["owned"] and cols["owned"] in df_view.columns:
        column_config[cols["owned"]] = st.column_config.CheckboxColumn("Owned")

    st.dataframe(
        df_view[chosen_cols],
        use_container_width=True, hide_index=True,
        column_config=column_config
    )

    st.dataframe(df_view[chosen_cols], use_container_width=True, hide_index=True)

    # Details & Risk band
    st.markdown("---")
    st.subheader("Details & Chart")

    # Row-level prices
    stop_val = float(row.get(cols["stop"], np.nan)) if cols["stop"] else np.nan
    tgt_val  = float(row.get(cols["tgt"],  np.nan)) if cols["tgt"]  else np.nan
    last_val = float(row.get(cols["last"], np.nan)) if cols["last"] else np.nan
    if not np.isfinite(last_val):
        last_val = float(y_close[-1])

    # Risk/Reward controls
    st.subheader("Risk/Reward")
    use_row_stops = st.checkbox("Use row Stop/Target if available", value=True)
    risk_pct = st.number_input("Risk % (per trade)", min_value=0.1, max_value=50.0, value=5.0, step=0.5, help="Percent below current price for stop.")
    reward_R = st.number_input("Reward multiple (R)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, help="Target = Last + R * (Last - Stop).")

    if use_row_stops and np.isfinite(stop_val) and np.isfinite(tgt_val):
        stop_for_band = stop_val
        tgt_for_band  = tgt_val
    else:
        stop_for_band = last_val * (1.0 - risk_pct/100.0)
        risk_amount   = last_val - stop_for_band
        tgt_for_band  = last_val + reward_R * risk_amount

    # Chart options
    st.subheader("Chart Options")
    chart_cfg = get_settings()["chart"]
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1: show_sma    = st.checkbox("Show SMA200", value=bool(chart_cfg["show_sma"]), key="show_sma")
    with c2: show_stop   = st.checkbox("Show Stop", value=bool(chart_cfg["show_stop"]), key="show_stop")
    with c3: show_target = st.checkbox("Show Target", value=bool(chart_cfg["show_target"]), key="show_target")
    with c4: range_days  = st.number_input(
        "Last N days",
        min_value=10,
        max_value=len(x),
        value=min(int(chart_cfg["range_days"]), len(x)),
        step=10,
        key="range_days",
    )
    # Slice last N
    x_plot = x[-range_days:]
    y_close_plot = y_close[-range_days:]
    y_sma_plot   = y_sma[-range_days:]
    y_open_plot  = y_open[-range_days:]
    y_high_plot  = y_high[-range_days:]
    y_low_plot   = y_low[-range_days:]

    # Overlay pickers
    oc1, oc2 = st.columns([2, 2])
    with oc1:
        chosen_overlays = st.multiselect(
            "Overlays",
            options=list(TECHNICALS_REGISTRY.keys()),
            default=get_settings()["overlays"]["defaults"],  # <-- use settings
            key="overlays",
        )
    with oc2:
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
                    dv = saved_params.get(pname, defaults.get(pname, lo))
                    if ptype == "int":
                        params[pname] = st.number_input(f"{key}.{pname}", value=int(dv), min_value=int(lo), max_value=int(hi), step=1)
                    elif ptype == "float":
                        params[pname] = st.number_input(f"{key}.{pname}", value=float(dv), min_value=float(lo), max_value=float(hi), step=0.1, format="%.2f")
                    else:
                        params[pname] = st.text_input(f"{key}.{pname}", value=str(dv))
                overlay_params[key] = params

    # Projections UI
    st.subheader("Projections")
    sproj = get_settings()["projections"]
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        show_projection = st.checkbox("Show projection simulations", value=bool(sproj["enabled"]))
    with pc2:
        proj_band = st.selectbox(
            "Projection band",
            ["10–90%", "5–95%", "20–80%", "25–75%", "Custom"],
            index=["10–90%","5–95%","20–80%","25–75%","Custom"].index(sproj["band"])
        )
    with pc3:
        proj_sims = st.number_input("Simulations", min_value=100, max_value=20000, value=int(sproj["sims"]), step=1000)
    with pc4:
        model_choice = st.selectbox("Projection model", ["EWMA+t", "GBM", "Bootstrap", "Jump"],
                                    index=["EWMA+t","GBM","Bootstrap","Jump"].index(sproj["model"]))
    with pc5:
        proj_months = st.number_input("Months to project", min_value=1, max_value=24, value=int(sproj["months"]), step=1)

    if proj_band == "Custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            pct_low = st.number_input("Lower percentile", min_value=0.0, max_value=49.9, value=10.0, step=0.5)
        with cc2:
            pct_high = st.number_input("Upper percentile", min_value=50.1, max_value=100.0, value=90.0, step=0.5)
    else:
        pct_low, pct_high = _parse_ci_label(proj_band)

    with st.expander("Advanced projection controls"):
        window = st.number_input(
            "Calibration window (trading days)",
            min_value=60,
            max_value=252*5,
            value=252,
            step=20,
            help="History length used to estimate drift/vol/jumps",
            key=K("review.proj", "window"),
        )
        lam = st.slider("EWMA lambda (vol persistence)", 0.80, 0.99, float(sproj["lam"]), 0.01,
                        key=K("review.proj", "lam"))
        df_t = st.slider("Student-t degrees of freedom", 3, 15, int(sproj["df_t"]), 1,
                        key=K("review.proj", "df_t"))
        antithetic = st.checkbox("Use antithetic variates (variance reduction)", value=bool(sproj["antithetic"]),
                                key=K("review.proj", "antithetic"))
        block = st.number_input("Bootstrap block size", 3, 30, int(sproj["block"]), 1,
                                key=K("review.proj", "block"))
        vol_mode = st.selectbox("Volatility estimator",
                                ["YangZhang","Parkinson","GK","RS","CloseEWMA","CloseRolling","MAD"],
                                index=["YangZhang","Parkinson","GK","RS","CloseEWMA","CloseRolling","MAD"].index(sproj["vol_mode"]),
                                key=K("review.proj", "vol_mode"))
        stochastic_vol = st.checkbox("Stochastic volatility (mean-reverting)", value=bool(sproj["stochastic_vol"]),
                                    key=K("review.proj", "stoch_vol"))
        use_fixed_seed = st.checkbox("Use fixed seed = 42", value=(sproj["seed_mode"] == "fixed"),
                                    key=K("review.proj", "fixed_seed"))
        seed = 42 if use_fixed_seed else int(st.number_input("Seed (for reproducibility)", 0, 2**32-1,
                                                            int(sproj["seed"]), 1,
                                                            key=K("review.proj", "seed")))
    
    sp = get_settings()["projections"]
    sp.update({
        "enabled": bool(show_projection),
        "band": proj_band,
        "sims": int(proj_sims),
        "model": model_choice,
        "months": int(proj_months),
        "window": int(window),
        "lam": float(lam),
        "df_t": int(df_t),
        "antithetic": bool(antithetic),
        "block": int(block),
        "vol_mode": vol_mode,
        "stochastic_vol": bool(stochastic_vol),
        "seed_mode": "fixed" if use_fixed_seed else "custom",
        "seed": int(seed),
    })

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_close_plot, mode="lines", name="Close", line=dict(color=th["close"])))
    if show_sma:
        fig.add_trace(go.Scatter(x=x_plot, y=y_sma_plot, mode="lines", name="SMA200", line=dict(color=th["sma200"])))

    shapes: List[dict] = []
    annotations: List[dict] = []
    if show_stop and np.isfinite(stop_for_band):
        shapes.append(dict(type="line", xref="x", yref="y", x0=x_plot.min(), x1=x_plot.max(),
                           y0=stop_for_band, y1=stop_for_band, line=dict(dash="dash", color=th["stop"])))
        annotations.append(dict(x=x_plot.max(), y=stop_for_band, xref="x", yref="y",
                                text=f"Stop {stop_for_band:.2f}", showarrow=False, xanchor="left",
                                font=dict(color=th["stop"])))
    if show_target and np.isfinite(tgt_for_band):
        shapes.append(dict(type="line", xref="x", yref="y", x0=x_plot.min(), x1=x_plot.max(),
                           y0=tgt_for_band, y1=tgt_for_band, line=dict(dash="dot", color=th["target"])))
        annotations.append(dict(x=x_plot.max(), y=tgt_for_band, xref="x", yref="y",
                                text=f"Target {tgt_for_band:.2f}", showarrow=False, xanchor="left",
                                font=dict(color=th["target"])))

    # Risk band fill
    if np.isfinite(stop_for_band) and np.isfinite(tgt_for_band):
        xs = pd.to_datetime([x_plot.min(), x_plot.max()])
        y_low_fill  = [min(stop_for_band, tgt_for_band)] * 2
        y_high_fill = [max(stop_for_band, tgt_for_band)] * 2
        fig.add_trace(go.Scatter(
            x=list(xs)+list(xs[::-1]),
            y=y_high_fill + y_low_fill[::-1],
            fill='toself', mode='lines', name='Risk Band',
            line=dict(width=0),
            fillcolor=th.get("risk_band", th["proj_band"]),
            opacity=0.25, hoverinfo="skip", showlegend=False,
        ))

    # Overlays
    for key in chosen_overlays:
        entry = TECHNICALS_REGISTRY.get(key)
        if not entry:
            continue
        params = overlay_params.get(key, entry.get("params", {}))
        res = entry["fn"](x_plot, pd.Series(y_close_plot), pd.Series(y_sma_plot), row, params)
        for tr in (res.traces or []):
            if not getattr(tr, "line", None) or not getattr(tr.line, "color", None):
                tr.line = dict(color=th["overlay"])
            fig.add_trace(tr)
        if res.shapes:
            shapes.extend(res.shapes)
        if res.annotations:
            annotations.extend(res.annotations)

    # Projection
    if show_projection:
        last_date = pd.to_datetime(x_plot.iloc[-1]) if hasattr(x_plot, "iloc") else pd.to_datetime(x_plot[-1])
        y_sma20_for_drift = (pd.Series(y_close_plot, dtype="float64").rolling(20, min_periods=10).mean().to_numpy())
        fut_dates, med, low, high = _project_next_month(
            y_close=y_close_plot,
            start_date=last_date,
            sims=int(proj_sims),
            pct_low=pct_low, pct_high=pct_high,
            model=model_choice,
            seed=seed,
            window=window, lam=lam, df_t=df_t,
            antithetic=antithetic, block=int(block),
            horizon_months=int(proj_months),
            vol_mode=vol_mode,
            y_open=y_open_plot, y_high=y_high_plot, y_low=y_low_plot,
            stochastic_vol=stochastic_vol,
            y_sma_short_for_drift=y_sma20_for_drift,
            y_sma_long_for_drift=y_sma_plot,
            sma_short_weight=0.9, sma_long_weight=0.1,
        )
        fig.add_trace(go.Scatter(
            x=list(fut_dates) + list(fut_dates[::-1]),
            y=list(high) + list(low[::-1]),
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=th["proj_band"], opacity=0.25,
            name=f"Projection {int(pct_low)}–{int(pct_high)}", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=fut_dates, y=med, mode="lines",
            line=dict(color=th["proj_mid"], dash="dash"),
            name="Projection median"
        ))

    aset = get_settings()["chart"]
    overlay_title = f" + {', '.join(chosen_overlays)}" if chosen_overlays else ""
    fig.update_layout(
        height=int(aset["plot_height"]),
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{row.get(cols['ticker'], 'Ticker')} — Close & SMA200{overlay_title}", 
        shapes=shapes,
        annotations=annotations,
        hovermode=aset["hovermode"],   # from settings
        yaxis=dict(title="Price"),
        template=aset.get("template", "plotly_white"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scores & Status")
    render_kpis(row, cols)

    st.divider()

    with st.expander("Raw row data"):
        st.json({c: (row[c] if pd.notna(row[c]) else None) for c in df_view.columns})

    st.download_button(
        "Download filtered CSV",
        df_view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_tickers.csv",
        mime="text/csv",
    )

# ────────────────────────────────────────────────────────────────────────────────
# 🧠 SIGNALS TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_signals:
    st.subheader("Buy/Sell Engines")

    # Basic inputs that depend on the Review tab’s last_val/reward_R
    # We recompute minimal context here to keep this tab self-contained.
    row = df_view.iloc[0] if df_view.empty is False else df.iloc[0]
    x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(
        row, cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]
    )

    profile = st.radio("Profile", ["Conservative", "Balanced", "Aggressive"], horizontal=True, index=1)

    preset = {
        "Conservative": dict(composite_threshold=0.70, w_rsi=0.20, w_trend=0.35, w_breakout=0.25, w_value=0.15, w_flow=0.05,
                             rsi_buy_max=40.0, vol_ratio_min=1.75, atr_mult=2.0, stop_pct=12.0, reward_R=2.0,
                             sell_threshold=0.65, rsi_overbought_min=72.0, ema_fast_span=21, sma_mid_window=50,
                             donch_lookback_sell=20, gap_down_min_pct=0.5),
        "Balanced": dict(
            composite_threshold=0.60,
            w_rsi=0.20, w_trend=0.20, w_value=0.10, w_flow=0.05,
            w_bbands=0.20, w_donchian=0.25, w_breakout=0.00,
            rsi_buy_max=45.0, vol_ratio_min=1.50, atr_mult=1.5, stop_pct=10.0, reward_R=2.0,
            donch_lookback=20, gap_min_pct=0.5,
            sell_threshold=0.60, rsi_overbought_min=70.0,
            ema_fast_span=21, sma_mid_window=50, donch_lookback_sell=20, gap_down_min_pct=0.5,
            w_rsi_sell=0.30, w_trend_down=0.30, w_breakdown=0.00, w_exhaustion=0.10, w_flow_out=0.05,
            w_bbands_sell=0.25, w_donchian_sell=0.25,
        ),
        "Aggressive": dict(composite_threshold=0.50, w_rsi=0.30, w_trend=0.20, w_breakout=0.30, w_value=0.10, w_flow=0.10,
                           rsi_buy_max=50.0, vol_ratio_min=1.25, atr_mult=1.2, stop_pct=9.0, reward_R=2.0,
                           sell_threshold=0.55, rsi_overbought_min=68.0, ema_fast_span=13, sma_mid_window=34,
                           donch_lookback_sell=15, gap_down_min_pct=0.3),
    }[profile]

with st.expander("Tune parameters", expanded=False):
    tab_buy, tab_sell, tab_risk = st.tabs(["BUY thresholds & weights", "SELL thresholds & weights", "Risk & sizing"])

    with tab_buy:
        c1, c2 = st.columns(2)
        with c1:
            composite_threshold = st.slider("BUY: Composite threshold", 0.30, 0.90, float(preset["composite_threshold"]), 0.01, help=h("composite_threshold"))
            rsi_buy_max        = st.slider("BUY: RSI buy max", 20, 70, int(preset["rsi_buy_max"]), 1, help=h("rsi_buy_max"))
            vol_ratio_min      = st.slider("Min vol / avg20 (both)", 1.0, 3.0, float(preset["vol_ratio_min"]), 0.05, help=h("vol_ratio_min"))
            value_center       = st.slider("BUY: Value center vs SMA200 (%)", -20.0, 10.0, -5.0, 0.5, help=h("value_center"))
            donch_lookback     = st.slider("BUY: Donchian lookback", 10, 60, int(preset.get("donch_lookback", 20)), 1, help=h("donch_lookback"))
            gap_min_pct        = st.slider("BUY: Gap-up min % (vs prev high)", 0.0, 3.0, float(preset.get("gap_min_pct", 0.5)), 0.1, help=h("gap_min_pct"))
        with c2:
            atr_mult           = st.slider("BUY: ATR stop ×", 0.8, 3.0, float(preset["atr_mult"]), 0.1, help=h("atr_mult"))
            stop_pct_ui        = st.slider("BUY: Fallback % stop", 3.0, 20.0, float(preset["stop_pct"]), 0.5, help=h("stop_pct_ui"))
            sma_window         = st.slider("Trend window (days, buy)", 100, 300, 200, 10, help=h("sma_window"))
            use_engine_stop    = st.checkbox("Use engine stop/target on chart", value=False, help=h("use_engine_stop"))
            bb_window          = st.slider("Bollinger window", 10, 60, int(preset.get("bb_window", 20)), 1, help=h("bb_window"))
            bb_k               = st.slider("Bollinger k (σ)", 1.0, 3.0, float(preset.get("bb_k", 2.0)), 0.1, help=h("bb_k"))
        st.markdown("**Weights**")
        c3, c4 = st.columns(2)
        with c3:
            w_rsi   = st.slider("Weight (BUY): RSI", 0.0, 1.0, float(preset["w_rsi"]), 0.05, help=h("w_rsi"))
            w_trend = st.slider("Weight (BUY): Trend", 0.0, 1.0, float(preset["w_trend"]), 0.05, help=h("w_trend"))
            w_value = st.slider("Weight (BUY): Value", 0.0, 1.0, float(preset["w_value"]), 0.05, help=h("w_value"))
        with c4:
            w_flow  = st.slider("Weight (BUY): Flow", 0.0, 1.0, float(preset["w_flow"]), 0.05, help=h("w_flow"))
            w_bbands = st.slider("Weight (BUY): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands", 0.20)), 0.05, help=h("w_bbands"))
            w_donch  = st.slider("Weight (BUY): Donchian",     0.0, 1.0, float(preset.get("w_donchian", 0.25)), 0.05, help=h("w_donch"))
            w_break  = st.slider("Weight (BUY): Legacy breakout", 0.0, 1.0, float(preset.get("w_breakout", 0.00)), 0.05, help=h("w_break"))

    with tab_sell:
        c1, c2 = st.columns(2)
        with c1:
            sell_threshold      = st.slider("SELL: Composite threshold", 0.30, 0.90, float(preset["sell_threshold"]), 0.01, help=h("sell_threshold"))
            rsi_overbought_min  = st.slider("SELL: RSI overbought min", 55, 85, int(preset["rsi_overbought_min"]), 1, help=h("rsi_overbought_min"))
            donch_lookback_sell = st.slider("SELL: Donchian lookback", 10, 60, int(preset["donch_lookback_sell"]), 1, help=h("donch_lookback_sell"))
        with c2:
            ema_fast_span   = st.slider("SELL: EMA fast span", 5, 55, int(preset["ema_fast_span"]), 1, help=h("ema_fast_span"))
            sma_mid_window  = st.slider("SELL: SMA mid window", 20, 100, int(preset["sma_mid_window"]), 1, help=h("sma_mid_window"))
            gap_down_min_pct = st.slider("SELL: Gap-down min % (vs prev low)", 0.0, 3.0, float(preset["gap_down_min_pct"]), 0.1, help=h("gap_down_min_pct"))
        st.markdown("**Weights**")
        c3, c4 = st.columns(2)
        with c3:
            w_rsi_sell   = st.slider("Weight (SELL): RSI", 0.0, 1.0, float(preset.get("w_rsi_sell", 0.30)), 0.05, help=h("w_rsi_sell"))
            w_trend_down = st.slider("Weight (SELL): Trend down", 0.0, 1.0, float(preset.get("w_trend_down", 0.30)), 0.05, help=h("w_trend_down"))
            w_breakdown  = st.slider("Weight (SELL): Breakdown", 0.0, 1.0, float(preset.get("w_breakdown", 0.25)), 0.05, help=h("w_breakdown"))
        with c4:
            w_exhaustion = st.slider("Weight (SELL): Exhaustion", 0.0, 1.0, float(preset.get("w_exhaustion", 0.10)), 0.05, help=h("w_exhaustion"))
            w_flow_out   = st.slider("Weight (SELL): Flow out", 0.0, 1.0, float(preset.get("w_flow_out", 0.05)), 0.05, help=h("w_flow_out"))
            w_bbands_sell = st.slider("Weight (SELL): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands_sell", 0.25)), 0.05, help=h("w_bbands_sell"))
            w_donch_sell  = st.slider("Weight (SELL): Donchian",     0.0, 1.0, float(preset.get("w_donchian_sell", 0.25)), 0.05, help=h("w_donch_sell"))

    with tab_risk:
        c1, c2 = st.columns(2)
        with c1:
            portfolio_value = st.number_input("Portfolio ($)", min_value=1000.0, value=20000.0, step=1000.0)
            risk_per_trade_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        with c2:
            min_adv_dollars = st.number_input("Min ADV$ (liquidity)", min_value=0.0, value=250000.0, step=25000.0)


    params = BuyParams(
        composite_threshold=float(composite_threshold),
        w_rsi=float(w_rsi), w_trend=float(w_trend), w_breakout=float(w_break), w_value=float(w_value), w_flow=float(w_flow),
        rsi_buy_max=float(rsi_buy_max), rsi_floor=20.0,
        sma200_window=int(sma_window),
        donch_lookback=int(donch_lookback), gap_min_pct=float(gap_min_pct),
        value_center_dev_pct=float(value_center), vol_ratio_min=float(vol_ratio_min),
        use_engine_stop=bool(use_engine_stop), atr_mult=float(atr_mult), stop_pct=float(stop_pct_ui),
        reward_R=float(preset["reward_R"]),
        portfolio_value=float(portfolio_value), risk_per_trade_pct=float(risk_per_trade_pct),
        min_price=1.0, min_adv_dollars=float(min_adv_dollars),
        w_bbands=float(w_bbands), w_donchian=float(w_donch),
        bb_window=int(bb_window), bb_k=float(bb_k),
        w_bbands_sell=float(w_bbands_sell), w_donchian_sell=float(w_donch_sell),
        sell_threshold=float(sell_threshold),
        w_rsi_sell=float(w_rsi_sell), w_trend_down=float(w_trend_down), w_breakdown=float(w_breakdown),
        w_exhaustion=float(w_exhaustion), w_flow_out=float(w_flow_out),
        rsi_overbought_min=float(rsi_overbought_min),
        ema_fast_span=int(ema_fast_span), sma_mid_window=int(sma_mid_window),
        donch_lookback_sell=int(donch_lookback_sell), gap_down_min_pct=float(gap_down_min_pct),
    )

    # Compute engines for current (full) series
    buy_res  = compute_buy_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    sell_res = compute_sell_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    action, action_margin = _decide_action(buy_res, sell_res, params.composite_threshold, params.sell_threshold)

    # Metrics summary
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

    with st.expander("Parameter sweep (grid)", expanded=False):
        c1, c2, c3 = st.columns(3)
        rsi_maxs = c1.multiselect("RSI buy max", [40,45,50], default=[45])
        donchs   = c2.multiselect("Donch lookback", [15,20,25], default=[20])
        thr      = c3.multiselect("BUY threshold", [0.55,0.60,0.65], default=[0.60])

        if st.button("Run grid"):
            rows = []
            grid = param_grid({"rsi_buy_max": rsi_maxs, "donch": donchs, "thr": thr})
            prog = st.progress(0.0)
            for j, g in enumerate(grid, start=1):
                params2 = params.__class__(**{**params.__dict__,
                                              "rsi_buy_max": g["rsi_buy_max"],
                                              "donch_lookback": g["donch"],
                                              "composite_threshold": g["thr"]})
                bi, si = compute_signal_series_for_row(row, x, y_close, y_sma, y_open, y_high, y_low, params2)
                eq, trades, ret = run_dca_backtest(x, y_close, bi, si)
                met = compute_metrics(eq, x)
                rows.append({**g, **met, "Return": ret})
                prog.progress(j/len(grid))
            res = pd.DataFrame(rows)
            st.dataframe(res, use_container_width=True)

st.session_state["last_used_params"] = params

# ────────────────────────────────────────────────────────────────────────────────
# 🧪 BACKTEST TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest & DCA simulator")

    # Use currently filtered first row for consistency
    row = df_view.iloc[0] if df_view.empty is False else df.iloc[0]
    x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(
        row, cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]
    )

    # Simple, safe default params (balanced) if none from Signals tab
    if "last_used_params" in st.session_state:
        params = st.session_state["last_used_params"]
    else:
        params = BuyParams(
            composite_threshold=0.60, w_rsi=0.20, w_trend=0.20, w_breakout=0.00, w_value=0.10, w_flow=0.05,
            rsi_buy_max=45.0, rsi_floor=20.0, sma200_window=200,
            donch_lookback=20, gap_min_pct=0.5, value_center_dev_pct=-5.0, vol_ratio_min=1.50,
            use_engine_stop=False, atr_mult=1.5, stop_pct=10.0, reward_R=2.0,
            portfolio_value=20000.0, risk_per_trade_pct=0.5, min_price=1.0, min_adv_dollars=250000.0,
            w_bbands=0.20, w_donchian=0.25, bb_window=20, bb_k=2.0,
            w_bbands_sell=0.25, w_donchian_sell=0.25,
            sell_threshold=0.60, w_rsi_sell=0.30, w_trend_down=0.30, w_breakdown=0.25, w_exhaustion=0.10, w_flow_out=0.05,
            rsi_overbought_min=70.0, ema_fast_span=21, sma_mid_window=50, donch_lookback_sell=20, gap_down_min_pct=0.5,
        )

    # Controls
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

    # Historical signal series using current params
    buy_idx, sell_idx = compute_signal_series_for_row(
        row=row, x=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params
    )

    equity, trades, total_ret = run_dca_backtest(
        dates=x, close=y_close, buy_idx=buy_idx, sell_idx=sell_idx,
        starting_cash=starting_cash,
        buy_pct_first=buy_pct_first, buy_pct_next=buy_pct_next,
        dca_trigger_drop_pct=dca_trigger_drop_pct, max_dca_legs=max_dca_legs,
        sell_pct_first=sell_pct_first, sell_pct_next=sell_pct_next,
    )

    # Plot price + markers + equity
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=x, y=y_close, mode="lines", name="Close"))
    # Respect "Show SMA" from Review
    if bool(st.session_state.get("show_sma", get_settings()["chart"]["show_sma"])):
        fig_bt.add_trace(go.Scatter(x=x, y=y_sma, mode="lines", name="SMA200", line=dict(color=th["sma200"])))

    # Reuse overlays (e.g., Bollinger) from Review
    chosen_overlays_bt = st.session_state.get("overlays", [])
    overlay_params_bt  = st.session_state.get("overlay_params_loaded", {})

    bt_shapes, bt_annotations = [], []
    for key in chosen_overlays_bt:
        entry = TECHNICALS_REGISTRY.get(key)
        if not entry:
            continue
        params = overlay_params_bt.get(key, entry.get("params", {}))
        res = entry["fn"](x, pd.Series(y_close), pd.Series(y_sma), row, params)
        for tr in (res.traces or []):
            # default a color if not set
            if not getattr(tr, "line", None) or not getattr(tr.line, "color", None):
                tr.line = dict(color=th["overlay"])
            fig_bt.add_trace(tr)
        if res.shapes:
            bt_shapes.extend(res.shapes)
        if res.annotations:
            bt_annotations.extend(res.annotations)
    
    if len(buy_idx):
        fig_bt.add_trace(go.Scatter(
            x=x[buy_idx], y=y_close[buy_idx], mode="markers",
            name="BUY (hist)", marker=dict(symbol="triangle-up", size=10, color=th["buy"])
    ))
    if len(sell_idx):
        fig_bt.add_trace(go.Scatter(
            x=x[sell_idx], y=y_close[sell_idx], mode="markers",
            name="SELL (hist)", marker=dict(symbol="triangle-down", size=10, color=th["sell"])
        ))

    aset = get_settings()["chart"]
    fig_bt.update_layout(
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Strategy equity", overlaying="y", side="right", showgrid=False),
        height=int(aset["plot_height"]),
        hovermode=aset["hovermode"],
        shapes=bt_shapes, annotations=bt_annotations,
        template=aset.get("template", "plotly_white"),
    )
    fig_bt.add_trace(go.Scatter(
        x=x, y=equity, mode="lines", name="Strategy equity",
        yaxis="y2", line=dict(dash="dash", color=th["equity"])
    ))
    st.plotly_chart(fig_bt, use_container_width=True)

    st.metric("DCA backtest return", f"{total_ret*100:.1f}%")
    with st.expander("Executed trades", expanded=False):
        if trades:
            tdf = pd.DataFrame(trades)
            st.dataframe(tdf[["date","side","price","qty","cash","shares"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No trades executed by this configuration.")

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️ SETTINGS TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_settings:
    st.subheader("App Settings")

    s = get_settings()
    tabA, tabB, tabC, tabD, tabE = st.tabs(
        ["🎨 Appearance", "📈 Chart defaults", "🧮 Projections defaults", "🧩 Overlays", "⚙️ Performance & Data"]
    )

    # ── Appearance (colors already persisted via load_theme/save_theme) ─────────
with tabA:
    st.caption("Pick theme colors. These are used across charts, overlays, and annotations.")

    prev = json.dumps(th, sort_keys=True)

    # ─ Price & Trend ─
    st.markdown("#### Price & Trend")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        th["close"] = st.color_picker(
            "Close", th["close"],
            key=K("appearance","close"),
            help=h("appearance.close"),
        )
    with c2:
        th["sma200"] = st.color_picker(
            "SMA200", th["sma200"],
            key=K("appearance","sma200"),
            help=h("appearance.sma200"),
        )
    with c3:
        th["overlay"] = st.color_picker(
            "Overlays (default)", th["overlay"],
            key=K("appearance","overlay"),
            help=h("appearance.overlay"),
        )

    st.divider()

    # ─ Trade Annotations ─
    st.markdown("#### Trade Annotations")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        th["stop"] = st.color_picker(
            "Stop line", th["stop"],
            key=K("appearance","stop"),
            help=h("appearance.stop"),
        )
    with c2:
        th["target"] = st.color_picker(
            "Target line", th["target"],
            key=K("appearance","target"),
            help=h("appearance.target"),
        )
    with c3:
        th["risk_band"] = st.color_picker(
            "Risk band fill", th.get("risk_band", "#8dd3c7"),
            key=K("appearance","risk_band"),
            help=h("appearance.risk_band"),
        )

    st.divider()

    # ─ Projections ─
    st.markdown("#### Projections")
    c1, c2 = st.columns([1,1])
    with c1:
        th["proj_mid"] = st.color_picker(
            "Projection median", th["proj_mid"],
            key=K("appearance","proj_mid"),
            help=h("appearance.proj_mid"),
        )
    with c2:
        th["proj_band"] = st.color_picker(
            "Projection band fill", th["proj_band"],
            key=K("appearance","proj_band"),
            help=h("appearance.proj_band"),
        )

    # Persist if anything changed
    new = json.dumps(th, sort_keys=True)
    if new != prev:
        save_theme(th)
        st.success("✅ Theme saved")

    # ── Chart defaults ─────────────────────────────────────────────────────────
    with tabB:
        st.caption("Defaults used when the Review tab widgets are first shown.")
        s["chart"]["template"] = st.selectbox(
            "Plotly template", ["plotly_white","plotly_dark"],
            index=["plotly_white","plotly_dark"].index(s["chart"].get("template","plotly_white")),
            key=K("settings.chart","template")
        )
        s["chart"]["show_sma"] = st.checkbox("Show SMA200 by default", value=s["chart"]["show_sma"],
                                            key=K("settings.chart", "show_sma"))
        s["chart"]["show_stop"] = st.checkbox("Show Stop line by default", value=s["chart"]["show_stop"],
                                            key=K("settings.chart", "show_stop"))
        s["chart"]["show_target"] = st.checkbox("Show Target line by default", value=s["chart"]["show_target"],
                                                key=K("settings.chart", "show_target"))
        s["chart"]["range_days"] = int(st.number_input("Default 'Last N days'", 10, 2000,
                                                    int(s["chart"]["range_days"]), 10,
                                                    key=K("settings.chart", "range_days")))
        s["chart"]["plot_height"] = int(st.number_input("Plot height (px)", 360, 1200,
                                                        int(s["chart"]["plot_height"]), 20,
                                                        key=K("settings.chart", "plot_height")))
        s["chart"]["hovermode"] = st.selectbox("Hover mode", ["x unified","closest","x"],
                                            index=["x unified","closest","x"].index(s["chart"]["hovermode"]),
                                            key=K("settings.chart", "hovermode"))

        if st.button("Apply these defaults to current page now"):
            # Push defaults into session for immediate effect
            for k in ("show_sma","show_stop","show_target","range_days"):
                st.session_state[k] = s["chart"][k]
            st.rerun()

    # ── Projections defaults ───────────────────────────────────────────────────
    with tabC:
        s["projections"]["enabled"] = st.checkbox("Enable projections by default",
                                                value=s["projections"]["enabled"],
                                                key=K("settings.proj", "enabled"))
        s["projections"]["band"] = st.selectbox("Default CI band", ["10–90%","5–95%","20–80%","25–75%","Custom"],
                                                index=["10–90%","5–95%","20–80%","25–75%","Custom"].index(s["projections"]["band"]),
                                                key=K("settings.proj", "band"))
        s["projections"]["sims"] = int(st.number_input("Default simulations", 100, 20000,
                                                    int(s["projections"]["sims"]), 1000,
                                                    key=K("settings.proj", "sims")))
        s["projections"]["model"] = st.selectbox("Default projection model", ["EWMA+t","GBM","Bootstrap","Jump"],
                                                index=["EWMA+t","GBM","Bootstrap","Jump"].index(s["projections"]["model"]),
                                                key=K("settings.proj", "model"))
        s["projections"]["months"] = int(st.number_input("Default months to project", 1, 24,
                                                        int(s["projections"]["months"]), 1,
                                                        key=K("settings.proj", "months")))
        with st.expander("Advanced defaults", expanded=False):
            s["projections"]["window"] = int(st.number_input("Calibration window (days)", 60, 252*5,
                                                            int(s["projections"]["window"]), 20,
                                                            key=K("settings.proj", "window")))
            s["projections"]["lam"] = float(st.slider("EWMA lambda (vol persistence)", 0.80, 0.99,
                                                    float(s["projections"]["lam"]), 0.01,
                                                    key=K("settings.proj", "lam")))
            s["projections"]["df_t"] = int(st.slider("Student-t df", 3, 15, int(s["projections"]["df_t"]), 1,
                                                    key=K("settings.proj", "df_t")))
            s["projections"]["antithetic"] = st.checkbox("Use antithetic variates",
                                                        value=bool(s["projections"]["antithetic"]),
                                                        key=K("settings.proj", "antithetic"))
            s["projections"]["block"] = int(st.number_input("Bootstrap block size", 3, 30,
                                                            int(s["projections"]["block"]), 1,
                                                            key=K("settings.proj", "block")))
            s["projections"]["vol_mode"] = st.selectbox("Volatility estimator",
                                                        ["YangZhang","Parkinson","GK","RS","CloseEWMA","CloseRolling","MAD"],
                                                        index=["YangZhang","Parkinson","GK","RS","CloseEWMA","CloseRolling","MAD"].index(s["projections"]["vol_mode"]),
                                                        key=K("settings.proj", "vol_mode"))
            s["projections"]["stochastic_vol"] = st.checkbox("Stochastic volatility (mean-reverting)",
                                                            value=bool(s["projections"]["stochastic_vol"]),
                                                            key=K("settings.proj", "stoch_vol"))
            seed_mode = st.radio("Seed mode", ["fixed","custom"],
                                index=["fixed","custom"].index(s["projections"]["seed_mode"]),
                                horizontal=True, key=K("settings.proj", "seed_mode"))
            s["projections"]["seed_mode"] = seed_mode
            if seed_mode == "custom":
                s["projections"]["seed"] = int(st.number_input("Default custom seed", 0, 2**32-1,
                                                            int(s["projections"]["seed"]), 1,
                                                            key=K("settings.proj", "seed")))
            else:
                s["projections"]["seed"] = int(st.number_input("Default custom seed", min_value=0, max_value=2**32-1, value=int(s["projections"]["seed"]), step=1))

        if st.button("Apply projection defaults to current page now"):
            # Nudge current widgets to new defaults
            st.session_state["range_days"] = s["chart"]["range_days"]
            st.rerun()

    # ── Overlays ───────────────────────────────────────────────────────────────
    with tabD:
        current = s["overlays"]["defaults"]
        all_opts = list(TECHNICALS_REGISTRY.keys())
        s["overlays"]["defaults"] = st.multiselect("Default overlays", options=all_opts, default=current)
        st.caption("These appear selected by default in Review → Overlays.")

    # ── Performance & Data ─────────────────────────────────────────────────────
    with tabE:
        st.caption("Plotly layout + data source stickies.")
        s["chart"]["plot_height"] = int(st.number_input("Plot height (px)", min_value=360, max_value=1200, value=int(s["chart"]["plot_height"]), step=20, key="perf_height"))
        s["chart"]["hovermode"] = st.selectbox("Hover mode", ["x unified","closest","x"], index=["x unified","closest","x"].index(s["chart"]["hovermode"]), key="perf_hover")
        st.divider()
        s["data"]["source_choice"] = st.selectbox("Default data source", ["Latest generated","Upload file","Path"],
                                                  index=["Latest generated","Upload file","Path"].index(s["data"]["source_choice"]))
        s["data"]["default_path"] = st.text_input("Default path", value=s["data"]["default_path"])

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reset ALL settings to defaults"):
                st.session_state.app_settings = deepcopy(DEFAULT_SETTINGS)
                st.success("Settings reset.")
                st.experimental_rerun()
        with col2:
            # Export settings
            st.download_button("Download settings JSON", data=json.dumps(s, indent=2).encode("utf-8"),
                               file_name="app_settings.json", mime="application/json")
        with col3:
            up = st.file_uploader("Import settings JSON", type=["json"])
            if up is not None:
                try:
                    loaded = json.load(up)
                    # naive merge: only update known top-level keys
                    for k in DEFAULT_SETTINGS.keys():
                        if k in loaded and isinstance(loaded[k], dict):
                            s[k].update(loaded[k])
                    st.success("Settings imported.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# 💾 PROFILES TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_profiles:
    st.subheader("Filter profiles")

    profile_name = st.text_input("Profile name", value="default")

    # Build the payload from current UI state (stored via session_state)
    def _current_profile_payload() -> dict:
        return {
            "min_sig":     int(st.session_state.get("min_sig", 0)),
            "min_comp":    float(st.session_state.get("min_comp", 0.0)),
            "rsi_min":     int(st.session_state.get("rsi_range", (0,100))[0]),
            "rsi_max":     int(st.session_state.get("rsi_range", (0,100))[1]),
            "owned_only":  bool(st.session_state.get("owned_only", False)),
            "search":      st.session_state.get("search", ""),

            "custom_rules": st.session_state.get("custom_rules", []),
            "logical":      st.session_state.get("cf_logical", "AND"),

            "sort_choice":  st.session_state.get("sort_choice"),
            "sort_asc":     bool(st.session_state.get("sort_asc", True)),
            "table_cols":   st.session_state.get("table_cols", []),

            "show_sma":     bool(st.session_state.get("show_sma", True)),
            "show_stop":    bool(st.session_state.get("show_stop", False)),
            "show_target":  bool(st.session_state.get("show_target", False)),
            "range_days":   int(st.session_state.get("range_days", 180)),

            "overlays":     st.session_state.get("overlays", []),
            "overlay_params": st.session_state.get("overlay_params_loaded", {}),

            "source_choice": st.session_state.get("source_choice", "Latest generated"),
            "default_path":  st.session_state.get("default_path", DEFAULT_OUTPUT.as_posix()),
        }

    colA, colB = st.columns(2)
    if colA.button("Save profile"):
        payload = _current_profile_payload()
        save_profile(profile_name, payload)
        st.success(f"Saved profile '{profile_name}'")

    existing = list_profiles()
    chosen_profile = st.selectbox("Load profile", options=["(select)"] + existing)

    def _apply_profile_to_session(prof: dict):
        st.session_state["custom_rules"] = prof.get("custom_rules", [])
        st.session_state["sort_choice"] = prof.get("sort_choice", st.session_state.get("sort_choice"))
        st.session_state["sort_asc"]    = prof.get("sort_asc", st.session_state.get("sort_asc", True))
        st.session_state["table_cols"]  = prof.get("table_cols", st.session_state.get("table_cols", []))
        st.session_state["show_sma"]    = prof.get("show_sma", st.session_state.get("show_sma", True))
        st.session_state["show_stop"]   = prof.get("show_stop", st.session_state.get("show_stop", False))
        st.session_state["show_target"] = prof.get("show_target", st.session_state.get("show_target", False))
        st.session_state["range_days"]  = prof.get("range_days", st.session_state.get("range_days", 180))
        st.session_state["overlays"]    = prof.get("overlays", st.session_state.get("overlays", []))
        st.session_state["overlay_params_loaded"] = prof.get("overlay_params", {})
        st.session_state["source_choice"] = prof.get("source_choice", st.session_state.get("source_choice", "Latest generated"))
        st.session_state["default_path"]  = prof.get("default_path", st.session_state.get("default_path", DEFAULT_OUTPUT.as_posix()))
        st.session_state["min_sig"]   = prof.get("min_sig", 0)
        st.session_state["min_comp"]  = prof.get("min_comp", 0.0)
        st.session_state["rsi_range"] = (prof.get("rsi_min", 0), prof.get("rsi_max", 100))
        st.session_state["owned_only"]= bool(prof.get("owned_only", 0))
        st.session_state["search"]    = prof.get("search", "")

    if st.button("Load", disabled=(chosen_profile == "(select)")):
        prof = load_profile(chosen_profile)
        if prof:
            _apply_profile_to_session(prof)
            st.rerun()
