from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import html as _html
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import datetime as dt

from core.paths import PROJECT_ROOT, DEFAULT_OUTPUT
from core.io import DEFAULT_THEME,RECO_COLORS, load_theme, save_theme, _run_main_and_reload, _read_path_cached, save_profile, load_profile, list_profiles
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

from market_trends.market_regimen import MarketRegimenConfig, build_market_regime_section

from sim.backtest import run_dca_backtest
from sim.projection import _project_next_month
from sim.metrics import compute_metrics

from ui.sections import render_projection_defaults_section, render_overlays_defaults, render_perf_and_data_defaults

from ui.css_elements import _inject_chip_css, _inject_tooltip_css, _inject_regimen_css, _fix_streamlit_clipping, _fix_streamlit_tooltip_overflow

from market_trends.market_regimen import (
    CATEGORY_WEIGHTS_DEFAULT,  # default category weights
    compute_category_scores,   # fallback if score_blocks not present
    combine_category_scores    # fallback combiner
)
from market_trends.components import CATEGORY_LABELS, CATEGORY_HELP, CATEGORY_ORDER
from market_trends.market_regimen import (
    MarketRegimenConfig,
    build_market_regime_section,
    HeadlineMode,
    RegimeLearnerConfig,
    RegimeModel,
    fit_regime_model_from_history,
)

# ────────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ticker Evaluator", layout="wide")


_inject_tooltip_css()
_inject_chip_css()
_inject_regimen_css()
_fix_streamlit_tooltip_overflow()
_fix_streamlit_clipping()

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
# Type helpers
# ────────────────────────────────────────────────────────────────────────────────

def _as_scalar(x):
    """Convert pandas/NumPy objects to plain Python scalars (use last non-NaN if Series)."""
    import numpy as np
    import pandas as pd

    if isinstance(x, pd.Series):
        if x.empty:
            return None
        # Prefer last non-NaN; else just last
        xx = x.dropna()
        v = xx.iloc[-1] if not xx.empty else x.iloc[-1]
        return _as_scalar(v)
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).to_pydatetime()
    if isinstance(x, np.generic):
        # numpy scalar → Python scalar
        try:
            return x.item()
        except Exception:
            return float(x) if np.issubdtype(x, np.number) else str(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return [ _as_scalar(v) for v in x ]
    return x

def _sanitize_dict(obj):
    """Recursively sanitize dicts/lists into Python types (no Series inside)."""
    if isinstance(obj, dict):
        return {k: _sanitize_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _sanitize_dict(v) for v in obj ]
    return _as_scalar(obj)

def _truthy(x) -> bool:
    """Deterministic boolean for possibly-Sequence inputs."""
    import numpy as np
    import pandas as pd
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, pd.Series):
        if x.empty:
            return False
        return bool(_as_scalar(x))
    if isinstance(x, np.ndarray):
        return bool(np.any(x))
    return bool(x)

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



def chip(label: str, help_text: str):
    st.markdown(f"""
      <span class="chip">
        {label}
        <span class="tip">{help_text}</span>
      </span>
    """, unsafe_allow_html=True)

def _reco_tip_html(buy_score: float, sell_score: float, params, reco_label: str) -> str:
    """Single-line tooltip: higher score first, e.g., 'Sell > Buy' with colored dots."""
    buy = ("Buy",  float(buy_score),  float(params.composite_threshold), RECO_COLORS.get("Buy",  "#90EE90"))
    sell= ("Sell", float(sell_score), float(params.sell_threshold),      RECO_COLORS.get("Sell", "#FFA500"))

    left, right = (sell, buy) if sell[1] > buy[1] else (buy, sell)
    sep = "=" if abs(left[1] - right[1]) < 1e-9 else ">"
    
    def pill(name, score, thr, color):
        return (
            "<span style='display:inline-flex;align-items:center;gap:6px;"
            "padding:4px 8px;border-radius:999px;border:1px solid rgba(0,0,0,.12);"
            "box-shadow:inset 0 1px 0 rgba(255,255,255,.18);white-space:nowrap;'>"
            f"<span style='width:8px;height:8px;border-radius:999px;background:{color};display:inline-block'></span>"
            f"<b>{name}</b>&nbsp;{score:.2f}"
            f"&nbsp;<span style='opacity:.8'>(vs the {thr:.2f})</span>"
            "</span>"
        )

    return (
        "<div class='section'>"
        "<h5>Recommendation</h5>"
        f"<div style='margin:0 0 6px 0'><b>{_html.escape(reco_label)}</b></div>"
        "<div style='display:flex;align-items:center;gap:8px;font-size:12px;'>"
        f"{pill(*left)}"
        f"<span style='opacity:.7;font-weight:700'>{sep}</span>"
        f"{pill(*right)}"
        "</div>"
        "</div>"
    )

def _quick_reco_from_norm(comp: float, sig: float) -> str:
    if not (np.isfinite(comp) and np.isfinite(sig)): return "⚪ Hold"
    if comp >= 0.80 and sig >= 0.70: return "🟢 Strong Buy"
    if comp >= 0.60 and sig >= 0.50: return "🟩 Buy"
    if comp <= 0.20 and sig <= 0.40: return "🔴 Strong Sell"
    if comp <= 0.40 and sig <= 0.50: return "🟧 Sell"
    return "⚪ Hold"

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

def _pretty_signal_name(k: str) -> str:
    if not isinstance(k, str):
        return str(k)
    return re.sub(r"\s+", " ", k.replace("_", " ").strip()).capitalize()

def _escape_attr(s: str) -> str:
    # Escape &, <, >, and " so HTML can safely live in an attribute.
    return _html.escape(s, quote=True)

def _extract_signal_names_from_row(row: pd.Series) -> list[str]:
    """Prefer precomputed signal lists from the row if available."""
    candidates = [
        "signals_list", "signals", "signals_fired", "signal_names",
        "signals_and_scores__signals", "signals_and_scores__signals_list",
        "signals_and_scores__signals_fired",
    ]
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            raw = row[c]
            if isinstance(raw, (list, tuple, np.ndarray)):
                return [str(v) for v in raw if str(v).strip()]
            if isinstance(raw, str) and raw.strip():
                s = raw.strip()
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        return [str(v) for v in obj if str(v).strip()]
                    if isinstance(obj, dict):
                        return [_pretty_signal_name(k) for k, v in obj.items() if bool(v)]
                except Exception:
                    pass
                parts = [p.strip() for p in re.split(r"[;,\|]", s) if p.strip()]
                if parts:
                    return parts
    return []

def classify_reco_label(buy_score: float, sell_score: float, params) -> str:
    mb = float(buy_score)  - float(params.composite_threshold)
    ms = float(sell_score) - float(params.sell_threshold)
    # Strong calls when one side clearly clears its threshold and the other is comfortably below
    if mb >= 0.12 and ms < -0.05: return "Strong Buy"
    if ms >= 0.12 and mb < -0.05: return "Strong Sell"
    if mb >= 0.00 and ms < 0.00:  return "Buy"
    if ms >= 0.00:                return "Sell"
    return "Hold"

def _normalize_to_unit(s: pd.Series, method: str = "percentile", robust_q=(0.05, 0.95)) -> pd.Series:
    """
    Return a 0–1 series using one of:
      - "percentile": percentile rank in [0,1]
      - "winsor_minmax": min–max between robust quantiles (clipped to [0,1])
      - "minmax": full-range min–max (may be sensitive to outliers)
    """
    x = pd.to_numeric(s, errors="coerce")

    if method == "percentile":
        return x.rank(pct=True, na_option="keep")

    if method == "winsor_minmax":
        lo, hi = np.nanquantile(x, robust_q[0]), np.nanquantile(x, robust_q[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        out = (x - lo) / (hi - lo)
        return out.clip(0.0, 1.0)

    if method == "minmax":
        mn, mx = np.nanmin(x.values), np.nanmax(x.values)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return pd.Series(np.nan, index=x.index, dtype="float64")
        return (x - mn) / (mx - mn)

    # fallback: percentile
    return x.rank(pct=True, na_option="keep")

def _balanced_fallback_params() -> BuyParams:
    return BuyParams(
        composite_threshold=0.60,
        w_rsi=0.20, w_trend=0.20, w_breakout=0.00, w_value=0.10, w_flow=0.05,
        rsi_buy_max=45.0, rsi_floor=20.0, sma200_window=200,
        donch_lookback=20, gap_min_pct=0.5, value_center_dev_pct=-5.0, vol_ratio_min=1.50,
        use_engine_stop=False, atr_mult=1.5, stop_pct=10.0, reward_R=2.0,
        portfolio_value=20000.0, risk_per_trade_pct=0.5, min_price=1.0, min_adv_dollars=250000.0,
        w_bbands=0.20, w_donchian=0.25, bb_window=20, bb_k=2.0,
        w_bbands_sell=0.25, w_donchian_sell=0.25,
        sell_threshold=0.60, w_rsi_sell=0.30, w_trend_down=0.30, w_breakdown=0.25, w_exhaustion=0.10, w_flow_out=0.05,
        rsi_overbought_min=70.0, ema_fast_span=21, sma_mid_window=50, donch_lookback_sell=20, gap_down_min_pct=0.5,
    )

def _signals_panel_md(row, x, y_close, y_sma, y_open, y_high, y_low) -> str:
    """Markdown for Signals breakdown."""
    # prefer row-provided signals if present
    names = _extract_signal_names_from_row(row)
    if names:
        names = sorted(set(_pretty_signal_name(s) for s in names))
        return "#### Signals (from table row)\n" + "\n".join(f"- {n}" for n in names)

    params, buy_res, sell_res = _compute_buy_sell(row, x, y_close, y_sma, y_open, y_high, y_low)
    buy_features  = [ _pretty_signal_name(k) for k, v in (buy_res.get("features") or {}).items() if bool(v) ]
    sell_features = [ _pretty_signal_name(k) for k, v in (sell_res.get("features") or {}).items() if bool(v) ]
    guards_bad    = buy_res.get("guard_reasons") or []
    sell_triggers = sell_res.get("reasons") or []

    md = []
    md.append(f"#### Signals")
    md.append(f"- **Buy score:** {buy_res.get('score', float('nan')):.2f}")
    md.append(f"- **Sell score:** {sell_res.get('score', float('nan')):.2f}")
    md.append("\n**BUY features**")
    md += [f"- {s}" for s in sorted(set(buy_features))] or ["- —"]
    md.append("\n**SELL features**")
    md += [f"- {s}" for s in sorted(set(sell_features))] or ["- —"]
    if guards_bad:
        md.append("\n**Guardrails triggered**")
        md += [f"- {g}" for g in guards_bad]
    if sell_triggers:
        md.append("\n**SELL triggers**")
        md += [f"- {g}" for g in sell_triggers]
    return "\n".join(md)

def _composite_breakdown_panel_md(row, x, y_close, y_sma, y_open, y_high, y_low) -> str:
    """
    Markdown for Composite breakdown.
    If your DF already has a composite decomposition column (e.g., JSON), you can parse it here.
    Otherwise we show the BUY composite-style breakdown from the current engine (components × weights).
    """
    params, buy_res, _ = _compute_buy_sell(row, x, y_close, y_sma, y_open, y_high, y_low)
    comps = buy_res.get("components") or {}
    # Map component -> weight attribute on params
    weight_map = {
        "rsi": "w_rsi",
        "trend": "w_trend",
        "value": "w_value",
        "flow": "w_flow",
        "bbands": "w_bbands",
        "donchian": "w_donchian",
        "breakout": "w_breakout",
        # add more if your engine exposes them
    }
    rows = []
    total = 0.0
    for k, v in comps.items():
        w_attr = weight_map.get(k)
        w = float(getattr(params, w_attr, 0.0)) if w_attr else 0.0
        contrib = float(v) * w
        total += contrib
        rows.append((k, float(v), w, contrib))
    rows.sort(key=lambda t: -abs(t[3]))

    md = []
    md.append("#### Composite score breakdown")
    md.append(f"- **Composite (engine)**: `{total:.2f}`  &nbsp;&nbsp; _(threshold: {params.composite_threshold:.2f})_")
    md.append("")
    md.append("| Component | Value (0–1) | Weight | Contribution |")
    md.append("|---|---:|---:|---:|")
    for k, val, w, c in rows:
        md.append(f"| {k} | {val:.2f} | {w:.2f} | {c:.2f} |")
    md.append(f"| **Total** |  |  | **{total:.2f}** |")
    md.append("\n<sub>Composite ≈ Σ(valueᵢ × weightᵢ). Guardrails/eligibility checks may cap or gate buys.</sub>")
    return "\n".join(md)

def _compute_buy_sell(row, x, y_close, y_sma, y_open, y_high, y_low):
    """Compute engines with last-used params or a balanced fallback."""
    params = st.session_state.get("last_used_params", _balanced_fallback_params())
    buy_res  = compute_buy_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    sell_res = compute_sell_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    return params, buy_res, sell_res

def _as_unit_score(v):
    try:
        v = float(v)
    except Exception:
        return np.nan
    if not np.isfinite(v):
        return np.nan
    # Heuristics: if it looks like a percent, normalize
    if 1.0 < v <= 100.0:
        return v / 100.0
    return v

def _normalize_score_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    med = float(np.nanmedian(s)) if s.notna().any() else np.nan
    if np.isfinite(med) and 1.0 < med <= 100.0:
        return s / 100.0
    return s

def _compute_signals_for_display(row, x, y_close, y_sma, y_open, y_high, y_low) -> dict:
    params = st.session_state.get("last_used_params", _balanced_fallback_params())
    buy_res  = compute_buy_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    sell_res = compute_sell_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    buy_features  = [ _pretty_signal_name(k) for k, v in (buy_res.get("features") or {}).items() if bool(v) ]
    sell_features = [ _pretty_signal_name(k) for k, v in (sell_res.get("features") or {}).items() if bool(v) ]
    return {
        "buy_features": sorted(set(buy_features)),
        "sell_features": sorted(set(sell_features)),
        "guards_bad": buy_res.get("guard_reasons") or [],
        "sell_triggers": sell_res.get("reasons") or [],
        "buy_score": float(buy_res.get("score", float("nan"))),
        "sell_score": float(sell_res.get("score", float("nan"))),
    }

def _build_signals_hover_html(row, x, y_close, y_sma, y_open, y_high, y_low) -> str:
    names = _extract_signal_names_from_row(row)
    if names:
        items = "".join(f"<li>{_pretty_signal_name(s)}</li>" for s in sorted(set(names)))
        return f'<div class="section"><h5>Signals (from table row)</h5><ul>{items or "<li>—</li>"}</ul></div>'

    params, buy_res, sell_res = _compute_buy_sell(row, x, y_close, y_sma, y_open, y_high, y_low)
    buy_features  = [ _pretty_signal_name(k) for k, v in (buy_res.get("features") or {}).items() if bool(v) ]
    sell_features = [ _pretty_signal_name(k) for k, v in (sell_res.get("features") or {}).items() if bool(v) ]
    guards_bad    = buy_res.get("guard_reasons") or []
    sell_triggers = sell_res.get("reasons") or []

    def _ul(lst): return "<ul>" + ("".join(f"<li>{_html.escape(s)}</li>" for s in lst) if lst else "<li>—</li>") + "</ul>"
    return (
        f'<div class="section"><h5>BUY features <span style="opacity:.7">(score: {buy_res.get("score", float("nan")):.2f})</span></h5>{_ul(sorted(set(buy_features)))}</div>'
        f'<div class="section"><h5>SELL features <span style="opacity:.7">(score: {sell_res.get("score", float("nan")):.2f})</span></h5>{_ul(sorted(set(sell_features)))}</div>'
        + (f'<div class="section"><h5>Guardrails</h5>{_ul(guards_bad)}</div>' if guards_bad else "")
        + (f'<div class="section"><h5>SELL triggers</h5>{_ul(sell_triggers)}</div>' if sell_triggers else "")
    )

def _build_composite_hover_html(row, x, y_close, y_sma, y_open, y_high, y_low) -> str:
    params, buy_res, _ = _compute_buy_sell(row, x, y_close, y_sma, y_open, y_high, y_low)
    comps = buy_res.get("components") or {}

    # Read the normalized column name from session (set once after df_view is built)
    comp_norm_col = st.session_state.get("_comp_norm_col")
    comp_norm_val = None
    if comp_norm_col and comp_norm_col in row.index and pd.notna(row[comp_norm_col]):
        comp_norm_val = float(row[comp_norm_col])

    # ⬇️ Your snippet goes right here
    header_line = ""
    if comp_norm_val is not None and np.isfinite(comp_norm_val):
        header_line = (
            f"<div style='font-size:12px;margin:2px 0 8px 0;'>"
            f"Dataset-normalized composite: <b>{comp_norm_val:.2f}</b> "
            f"(≈P{int(round(comp_norm_val*100)):d})</div>"
        )

    weight_map = {
        "rsi": "w_rsi", "trend": "w_trend", "value": "w_value", "flow": "w_flow",
        "bbands": "w_bbands", "donchian": "w_donchian", "breakout": "w_breakout",
    }

    rows, total = [], 0.0
    for k, v in comps.items():
        w = float(getattr(params, weight_map.get(k, ""), 0.0))
        val = float(v)
        contrib = val * w
        total += contrib
        rows.append((k, val, w, contrib))
    rows.sort(key=lambda t: -abs(t[3]))

    trs = "".join(
        f"<tr><td>{_html.escape(k)}</td>"
        f"<td style='text-align:right'>{val:.2f}</td>"
        f"<td style='text-align:right'>{w:.2f}</td>"
        f"<td style='text-align:right'>{contrib:.2f}</td></tr>"
        for k, val, w, contrib in rows
    )

    return f"""
      <div class="section">
        <h5>Composite breakdown</h5>
        {header_line}
        <div style="font-size:12px;opacity:.8;margin-bottom:6px;">
          Composite ≈ Σ(valueᵢ × weightᵢ) &nbsp; | &nbsp; threshold: {params.composite_threshold:.2f}
        </div>
        <table style="width:100%; border-collapse:collapse;">
          <thead><tr><th style="text-align:left">Component</th><th style="text-align:right">Value</th><th style="text-align:right">Weight</th><th style="text-align:right">Contrib</th></tr></thead>
          <tbody>{trs}</tbody>
          <tfoot><tr><td colspan="3" style="text-align:right;font-weight:600">Total</td><td style="text-align:right;font-weight:600">{total:.2f}</td></tr></tfoot>
        </table>
      </div>
    """

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
    chips: list[tuple],  # (label, value, color, icon[, hover_html_or_dict])
    *,
    scale: float = 1.2,
    force_white_on_dark: bool = True,
):
    _inject_chip_css()

    # Theme-aware tooltip colors
    dark = _is_dark_theme() if force_white_on_dark else False
    tip_bg     = "rgba(17,24,39,.98)" if dark else "rgba(255,255,255,.98)"
    tip_fg     = "#e5e7eb"            if dark else "#0f172a"
    tip_border = "rgba(255,255,255,.14)" if dark else "rgba(0,0,0,.08)"

    # scale paddings / font sizes
    pad_y = max(6, int(8 * scale))
    pad_x = max(10, int(12 * scale))
    gap   = max(8, int(10 * scale))
    fs    = max(12, int(13 * scale))
    val_pad_y = max(2, int(3 * scale))
    val_pad_x = max(6, int(8 * scale))
    val_fs    = max(11, int(12 * scale))

    html_parts = ['<div class="kpi-row">']

    for item in chips:
        label, value, color, icon = item[:4]
        hover = item[4] if len(item) >= 5 else None

        hover_html = None
        max_w = 520
        if isinstance(hover, dict):
            hover_html = hover.get("html")
            max_w = int(hover.get("max_width", max_w))
        elif isinstance(hover, str):
            hover_html = hover

        r, g, b = _hex_to_rgb(color or "#22c55e")
        top = f"rgba({r},{g},{b}, .30)"; bot = f"rgba({r},{g},{b}, .14)"; badge = f"rgba({r},{g},{b}, .18)"
        lum = (0.299*r + 0.587*g + 0.114*b)/255
        fg = "#ffffff" if dark else ("#0f172a" if lum > 0.7 else "white")

        chip_style = (
            f"background: linear-gradient(180deg, {top}, {bot});"
            f"border:1px solid rgba({r},{g},{b}, .45);"
            f"box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 2px 6px rgba(0,0,0,.25);"
            f"color:{fg}; border-radius:999px; display:inline-flex; align-items:center; gap:{gap}px;"
            f"padding:{pad_y}px {pad_x}px; font-size:{fs}px; line-height:1;"
            f"--chip-badge-bg:{badge}; --chip-fg:{fg};"
        )
        icon_html = f"<span style='font-size:{fs}px; line-height:1'>{icon}</span>" if icon else ""

        # Inline tooltip (if provided)
        tip_html = ""
        if hover_html:
            tip_html = (
                f"<div class='kpi-tip' "
                f"style='background:{tip_bg};color:{tip_fg};border:1px solid {tip_border};"
                f"max-width:{max_w}px;'>"
                f"{hover_html}"
                f"</div>"
            )

        html_parts.append(
            f"<div class='kpi-chip{' has-tip' if hover_html else ''}' style='{chip_style}'>"
            f"{icon_html}<span>{label}</span>"
            f"<span class='val' style='padding:{val_pad_y}px {val_pad_x}px; font-size:{val_fs}px;'>"
            f"{value}</span>"
            f"{tip_html}"
            f"</div>"
        )

    html_parts.append("</div>")
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)

def render_kpis(
    row: pd.Series,
    cols: dict,
    x, y_close, y_sma, y_open, y_high, y_low,
    df_all: Optional[pd.DataFrame] = None,
):
    comp_col = cols.get("comp_norm") or cols["comp"]
    sig_col  = cols.get("sig_norm")  or cols["sig"]

    # --- compute dataset-normalized composite (0..1) and percentile ---
    comp_norm_val, comp_pct = None, None
    comp_raw = float(row.get(cols['comp'], np.nan)) if cols.get('comp') else np.nan
    if (
        df_all is not None and cols.get('comp') and cols['comp'] in df_all.columns
        and np.isfinite(comp_raw)
    ):
        s_all = pd.to_numeric(df_all[cols['comp']], errors="coerce").dropna()
        if len(s_all) >= 2:
            lo, hi = float(s_all.min()), float(s_all.max())
            rng = hi - lo
            if rng > 1e-12:
                comp_norm_val = float(np.clip((comp_raw - lo) / rng, 0.0, 1.0))
            # Percentile rank (robust/intuitive)
            comp_pct = float((s_all <= comp_raw).mean())

    # --- tooltips ---
    comp_hover_core = _build_composite_hover_html(row, x, y_close, y_sma, y_open, y_high, y_low)
    header_line = ""
    if comp_norm_val is not None and np.isfinite(comp_norm_val):
        p_txt = f"P{int(round((comp_pct if comp_pct is not None else comp_norm_val)*100)):d}"
        header_line = (
            f"<div style='font-size:12px;opacity:.8;margin-bottom:6px;'>"
            f"Dataset-normalized composite: <b>{comp_norm_val:.2f}</b> (≈{p_txt})</div>"
        )
    comp_hover = header_line + comp_hover_core
    sigs_hover = _build_signals_hover_html(row, x, y_close, y_sma, y_open, y_high, y_low)

    # --- badges (prefer normalized for Composite; for Signals respect sig_col) ---
    comp_badge = comp_norm_val if comp_norm_val is not None else comp_raw
    sig_badge  = float(row.get(sig_col, 0.0))

    # Get scores for current row
    sig_info = _compute_signals_for_display(row, x, y_close, y_sma, y_open, y_high, y_low)
    params = st.session_state.get("last_used_params", _balanced_fallback_params())
    reco_label = classify_reco_label(sig_info["buy_score"], sig_info["sell_score"], params)
    reco_color = RECO_COLORS[reco_label]
    reco_tip = _reco_tip_html(
        sig_info["buy_score"],
        sig_info["sell_score"],
        params,
        reco_label,
    )

    # Prepend a 'Recommendation' chip
    chips = [
        ("Recommendation", reco_label, reco_color, "🏷️", {"html": reco_tip, "max_width": 420}),
        ("Composite", f"{comp_badge:,.2f}", "#22c55e", "✅", {"html": comp_hover, "max_width": 520}),
        ("Signals",   f"{sig_badge:,.2f}", "#3b82f6", "📊", {"html": sigs_hover, "max_width": 520}),
        ("RSI",       f"{row.get(cols['rsi'], float('nan')):,.2f}", "#ef4444", "🧭"),
        ("Last",      f"{row.get(cols['last'], float('nan')):,.2f}", "#a78bfa", "💵"),
        ("Owned",     "Yes" if bool(row.get(cols['owned'], False)) else "No", "#f59e0b", "📦"),
    ]
    kpi_row(chips, scale=1.35)

def _risk_hex(score: float | None) -> str:
    if score is None: return "#d1d5db"
    anchors = [
        (0,  (0x16,0xa3,0x4a)),  # green
        (30, (0x84,0xcc,0x16)),  # lime
        (60, (0xf5,0x9e,0x0b)),  # amber
        (80, (0xef,0x44,0x44)),  # red
        (100,(0x7f,0x1d,0x1d)),  # deep red
    ]
    s = max(0, min(100, int(round(float(score)))))
    for i in range(len(anchors)-1):
        s0,c0 = anchors[i]; s1,c1 = anchors[i+1]
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0 + 1e-9)
            mix = tuple(int(round(c0[k] + t*(c1[k]-c0[k]))) for k in range(3))
            return f"#{mix[0]:02x}{mix[1]:02x}{mix[2]:02x}"
    r,g,b = anchors[-1][1]; return f"#{r:02x}{g:02x}{b:02x}"

def _fmt_pct(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}%"

def _fmt_float(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"
def render_market_regime_backtest(fetch_fn):
    st.markdown("### 📈 Backtest & Calibration")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        start_bt = st.date_input("Backtest start", dt.date(2015,1,1))
    with col2:
        end_bt = st.date_input("Backtest end", dt.date.today())
    with col3:
        thr = st.slider("Risk threshold (enter market if risk < τ)", 0.0, 1.0, 0.25, 0.01)

    cfg = MarketRegimenConfig()
    model = _fit_regime_model_cached(cfg, fetch_fn, str(start_bt), str(end_bt))
    if model is None:
        st.warning("Model not available—fit failed or scikit-learn not installed.")
        return

    # Replay weekly, get p(default) and realized outcomes
    idx = pd.date_range(start_bt, end_bt, freq="W-FRI")
    rows = []
    closes = []
    for t in idx:
        res = build_market_regime_section(
            cfg, fetch_fn, start=pd.to_datetime(start_bt), end=pd.to_datetime(t),
            headline_mode=HeadlineMode("learned"), regime_model=model
        )
        p = res.get("learned_drawdown_prob_20d")
        rows.append({"date": t, "p": float(p) if p is not None else np.nan})
        closes.append(res.get("SPY_close"))
    df = pd.DataFrame(rows).set_index("date")
    spy = pd.Series(closes, index=df.index).ffill().dropna()
    # Realized next-20d MaxDD label (same as training definition)
    def _maxdd_next(px: pd.Series, h=20):
        fwd = px.shift(-1).rolling(h, min_periods=2)
        return (fwd.min() / fwd.max() - 1.0) * 100.0
    realized_dd = _maxdd_next(spy, 20)
    df = df.join(realized_dd.rename("dd")).dropna()
    df["event"] = (df["dd"] <= -5.0).astype(int)  # 1 if bad drawdown

    # Calibration (reliability)
    bins = pd.qcut(df["p"].clip(0,1), q=10, duplicates="drop")
    cal = df.groupby(bins).agg(p_hat=("p","mean"), event_rate=("event","mean"))
    cal_fig = go.Figure()
    cal_fig.add_scatter(x=cal["p_hat"], y=cal["event_rate"], mode="lines+markers", name="Observed")
    cal_fig.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash"))
    cal_fig.update_layout(title="Calibration (10 bins)", xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                          height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(cal_fig, use_container_width=True)

    # Decile lift (event rate / base rate)
    base = df["event"].mean()
    lift = (cal["event_rate"] / max(base, 1e-9)).rename("lift").reset_index(drop=True)
    lift_fig = go.Figure(go.Bar(x=list(range(1, len(lift)+1)), y=lift))
    lift_fig.update_layout(title=f"Decile Lift (base rate {base:.1%})", xaxis_title="Decile (low→high risk)",
                           yaxis_title="Lift vs base", height=340, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(lift_fig, use_container_width=True)

    # Simple strategy test (weekly signals)
    p = df["p"].clip(0,1)
    in_mkt = (p < thr).astype(int)
    ret = spy.pct_change().fillna(0)
    # hold for a week from signal; approx by multiplying by in_mkt shifted (enter next bar)
    strat = (in_mkt.shift(1).fillna(0) * ret).add(0.0)
    eq_curve = (1.0 + strat).cumprod()
    eq_spy   = (1.0 + ret).cumprod()

    perf_fig = go.Figure()
    perf_fig.add_scatter(x=eq_curve.index, y=eq_curve, name="Strategy")
    perf_fig.add_scatter(x=eq_spy.index, y=eq_spy,   name="Buy & Hold", line=dict(dash="dash"))
    perf_fig.update_layout(title=f"Strategy vs SPY (τ={thr:.2f})", yaxis_title="Growth of $1",
                           height=380, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(perf_fig, use_container_width=True)

    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        st.metric("CAGR (strategy)", f"{(eq_curve.iloc[-1]**(52/len(eq_curve))-1):.2%}")
    with colm2:
        st.metric("CAGR (SPY)", f"{(eq_spy.iloc[-1]**(52/len(eq_spy))-1):.2%}")
    with colm3:
        dd_strat = (eq_curve/eq_curve.cummax()-1).min()
        dd_spy   = (eq_spy/eq_spy.cummax()-1).min()
        st.metric("MaxDD (strategy)", f"{dd_strat:.2%}")
        st.caption(f"MaxDD (SPY): {dd_spy:.2%}")

def render_market_sentiment_dashboard(mr: dict, *, show_header: bool = True, show_legacy_headline: bool = False):
    """
    Beautiful per-category sentiment + interactive weights → recomputed headline.
    Expects mr to contain:
      - 'score_blocks': {category: score in [0,100]}
      - 'score_blocks_details': {category: {'parts': {...}}}
      - 'headline_score_0_100', 'headline_regime'   (optional; we recompute anyway)
    """
    if show_header:
        st.markdown("### 🌐 Market Sentiment Dashboard")
    # Fallback: if score_blocks missing, compute from raw features
    if not mr.get("score_blocks"):
        try:
            cats = compute_category_scores(mr)
            mr["score_blocks_details"] = cats
            mr["score_blocks"] = {k: (cats[k].get("score")) for k in cats.keys()}
        except Exception:
            mr["score_blocks"] = {}

    scores: dict[str, float | None] = mr.get("score_blocks", {}) or {}
    details: dict = mr.get("score_blocks_details", {}) or {}

    # Canonical order from the regimen module; fall back to the dict order
    default_order = list(CATEGORY_WEIGHTS_DEFAULT.keys())
    cat_order = [c for c in default_order if c in scores] + [c for c in scores.keys() if c not in default_order]

    # Session-scoped adjustable weights
    if "mr_weights" not in st.session_state or not st.session_state["mr_weights"]:
        st.session_state["mr_weights"] = {k: CATEGORY_WEIGHTS_DEFAULT.get(k, 0.1) for k in cat_order}

    palette = _category_palette()
    icons   = _category_icons()

    st.markdown("### 🌐 Market Sentiment Dashboard")

    # ——— Top: headline gauge (from adjustable weights) ———
    c_top1, c_top2 = st.columns([1.3, 1])
    with c_top1:
        # ── KPI callouts ────────────────────────────────────────────────
        try:
            theta = cat_order
            rvals = [float(scores.get(k) or 0.0) for k in theta]

            w_raw  = st.session_state.get("mr_weights", {k: CATEGORY_WEIGHTS_DEFAULT.get(k, 0.1) for k in theta})
            w_norm = _norm_weights(w_raw)                   # sums to 1.0
            wvals  = [w_norm.get(k, 0.0) * 100.0 for k in theta]  # ×100 only for ring

            pairs = [(k, float(scores.get(k) or float("nan"))) for k in theta]
            pairs = [p for p in pairs if np.isfinite(p[1])]
            if pairs:
                top_cat, top_val = max(pairs, key=lambda p: p[1])
                bot_cat, bot_val = min(pairs, key=lambda p: p[1])
                disp = float(np.nanstd([v for _, v in pairs]))

                palette = _category_palette()
                chips = [
                    ("Strongest", f"{top_cat.title()} {top_val:.1f}", palette.get(top_cat, "#16a34a"), "🏆",
                    {"html": _parts_tooltip_html(top_cat, (details.get(top_cat) or {}).get("parts")), "max_width": 460}),
                    ("Weakest",   f"{bot_cat.title()} {bot_val:.1f}", palette.get(bot_cat, "#ef4444"), "⚠️",
                    {"html": _parts_tooltip_html(bot_cat, (details.get(bot_cat) or {}).get("parts")), "max_width": 460}),
                    ("Dispersion σ", f"{disp:.1f}", "#64748b", "📐"),
                ]
                kpi_row(chips, scale=1.05)
        except Exception:
            pass

        # ── Radar (left) + Gauge (right) on the same row ───────────────────────────
        left, right = st.columns([1.35, 1])   # tweak ratio to taste
        with left:
            try:
                point_colors = [_category_palette().get(k, "#22c55e") for k in theta]
                tmpl  = get_settings()["chart"].get("template", "plotly_white")
                dark  = _is_dark_theme()
                txt   = "#e5e7eb" if dark else "#0f172a"
                grid  = "rgba(255,255,255,.18)" if dark else "rgba(0,0,0,.15)"
                accent = th.get("overlay", "#22c55e")
                r,g,b = _hex_to_rgb(accent)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=wvals + wvals[:1], theta=theta + theta[:1],
                    mode="lines",
                    line=dict(color="rgba(71,85,105,0.75)" if not dark else "rgba(148,163,184,0.85)", dash="dot", width=2),
                    hovertemplate="<b>%{theta}</b><br>Weight: %{r:.1f} (×100)<extra>Weights</extra>",
                    name="Weights (×100)"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=rvals + rvals[:1], theta=theta + theta[:1],
                    mode="lines", fill="toself",
                    line=dict(color=accent, width=3),
                    fillcolor=f"rgba({r},{g},{b},0.20)",
                    hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra>Score</extra>",
                    name="Category score"
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=rvals, theta=theta, mode="markers",
                    marker=dict(size=9, color=point_colors, line=dict(width=1, color="white")),
                    customdata=[w_norm.get(k, 0.0) * 100.0 for k in theta],
                    hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<br>Weight: %{customdata:.1f} (×100)<extra></extra>",
                    name="", showlegend=False
                ))
                fig_radar.update_layout(
                    template=tmpl,
                    height=300,
                    margin=dict(l=8, r=8, t=10, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=txt),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(range=[0,100], tickvals=[0,25,50,75,100], gridcolor=grid, gridwidth=1, tickfont=dict(size=11)),
                        angularaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(size=12), direction="clockwise", rotation=90),
                    ),
                    showlegend=True,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception:
                st.caption("Radar unavailable for current data.")

        with right:
            # Compute headline right here so the gauge lives next to the radar
            weights_norm = _norm_weights(st.session_state.get("mr_weights", w_raw))
            headline = float(mr.get("headline_score_0_100_final") or _weighted_headline_from(scores, weights_norm))
            lbl      = (mr.get("headline_regime") or _headline_label(headline))

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=headline,
                number=dict(suffix=""),
                title={"text": f"{lbl}"},
                gauge=dict(
                    axis=dict(range=[0,100]),
                    steps=[
                        {"range":[0,30],  "color":"#16a34a"},
                        {"range":[30,60], "color":"#84cc16"},
                        {"range":[60,80], "color":"#f59e0b"},
                        {"range":[80,100],"color":"#ef4444"},
                    ],
                    bar=dict(thickness=0.25),
                ),
            ))
            fig_g.update_layout(
                height=300,  # match radar height for visual alignment
                margin=dict(l=8, r=8, t=28, b=8),
                template=get_settings()["chart"].get("template","plotly_white"),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # ── Treemap full width below ────────────────────────────────────────────────
        try:
            labels  = theta
            parents = ["Market"] * len(labels)
            values  = [w_norm.get(k, 0.0) * 100.0 for k in labels]  # tile size = weight (%)
            colors  = [float(scores.get(k) or 0.0) for k in labels]  # tile color = score

            cs = [
                [0.00, "#16a34a"], [0.30, "#84cc16"], [0.60, "#f59e0b"], [0.80, "#ef4444"], [1.00, "#7f1d1d"],
            ]

            fig_tree = go.Figure(go.Treemap(
                labels=labels, parents=parents, values=values,
                marker=dict(colors=colors, colorscale=cs, cmin=0, cmax=100),
                branchvalues="total", pathbar=dict(visible=False),
                texttemplate="<b>%{label}</b><br>%{value:.1f}% • %{color:.1f}",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<br>Score: %{color:.1f}<extra></extra>",
            ))
            fig_tree.update_layout(
                height=240, margin=dict(l=4, r=4, t=4, b=4),
                template=get_settings()["chart"].get("template", "plotly_white"),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        except Exception:
            pass

    with c_top2:
        # Sliders to adjust weights
        with st.expander("Adjust category weights", expanded=True):
            new_w = {}
            for k in cat_order:
                label = CATEGORY_LABELS.get(k, k.title())
                # Prefer a specific category help if present, else a sensible fallback
                help_txt = (
                    globals().get("CATEGORY_HELP", {}).get(k)       # preferred per-category help
                    or globals().get("HELP", {}).get(k)             # legacy per-control help (if defined)
                    or (
                        f"Weight of {label}. 0 = ignore, 1 = strongest influence. "
                        "Final weights are normalized to sum to 1."
                    )
                )
                new_w[k] = st.slider(
                    label,
                    0.0, 1.0,
                    float(st.session_state['mr_weights'].get(k, CATEGORY_WEIGHTS_DEFAULT.get(k, 0.0))),
                    0.01,
                    help=help_txt,   # ← this adds the tooltip “?”
                )
            if st.button("Reset weights to defaults"):
                st.session_state["mr_weights"] = {k: CATEGORY_WEIGHTS_DEFAULT.get(k, 0.1) for k in cat_order}
            else:
                st.session_state["mr_weights"] = new_w

        weights_norm = _norm_weights(st.session_state["mr_weights"])
        headline = _weighted_headline_from(scores, weights_norm)
        lbl = _headline_label(headline)
        if show_legacy_headline:
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=headline,
                number=dict(suffix=""),
                title={"text": f"{lbl}"},
                gauge=dict(
                    axis=dict(range=[0,100]),
                    steps=[
                        {"range":[0,30], "color":"#16a34a"},
                        {"range":[30,60], "color":"#84cc16"},
                        {"range":[60,80], "color":"#f59e0b"},
                        {"range":[80,100],"color":"#ef4444"},
                    ],
                    bar=dict(thickness=0.25)
                )
            ))
            fig_g.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10),
                                template=get_settings()["chart"].get("template","plotly_white"))
            st.plotly_chart(fig_g, use_container_width=True)

    # ——— Middle: category chips with tooltips ———
    st.markdown("#### Category scores")
    chips = []
    for k in cat_order:
        val = scores.get(k)
        color = palette.get(k, "#22c55e")
        icon  = icons.get(k, "📊")
        hv_html = _parts_tooltip_html(k, (details.get(k) or {}).get("parts"))
        chips.append((k.title(), f"{'' if val is None else f'{float(val):.1f}'}", color, icon, {"html": hv_html, "max_width": 520}))
    kpi_row(chips, scale=1.15)

    # ——— Bottom: bar chart ———
    try:
        x = cat_order
        y = [float(scores.get(k) or 0.0) for k in x]
        bar_colors = [palette.get(k, "#22c55e") for k in x]
        fig_bar = go.Figure(go.Bar(x=x, y=y, marker_color=bar_colors, name="Score"))
        fig_bar.update_yaxes(range=[0,100], title="Score (0–100)")
        fig_bar.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10),
                              template=get_settings()["chart"].get("template","plotly_white"))
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        pass

    # Derived, for convenience elsewhere in the page
    st.session_state["mr_headline_custom"] = {"score": headline, "label": lbl, "weights": weights_norm}

# NEW: controller that computes/plots the learned/blend headline, then hands off to your renderer
def render_market_sentiment_page(fetch_fn):
    st.markdown("### 🌍 Market Sentiment Dashboard")

    # -------- Controls (left) --------
    with st.expander("Adjust category weights & headline mode", expanded=True):
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            mode_choice = st.radio(
                "Headline mode",
                options=["Blend (model + user)", "Learned (model only)", "User (sliders)"],
                index=0,
                horizontal=False,
                help="Blend respects the model while honoring your sliders.",
            )
        with colB:
            blend_alpha = st.slider("Blend: weight on model", 0.0, 1.0, 0.70, 0.05,
                                    help="0 = only sliders, 1 = only model")
        with colC:
            start_hist = st.date_input("Training start", dt.date(2015,1,1))
            end_hist   = st.date_input("Training end", dt.date.today())
            refit = st.button("Refit model", type="secondary")

        mode_map = {
            "Blend (model + user)": ("blend", blend_alpha),
            "Learned (model only)": ("learned", 1.0),
            "User (sliders)":       ("user", 0.0),
        }
        _mode, _alpha = mode_map[mode_choice]
        headline_mode = HeadlineMode(mode=_mode, blend_alpha=_alpha)

    # -------- Fit / get model --------
    model = _fit_regime_model_cached(
        cfg=MarketRegimenConfig(),
        fetch_fn=fetch_fn,
        start_date=str(start_hist),
        end_date=str(end_hist),
    )
    if refit:
        _fit_regime_model_cached.clear()
        model = _fit_regime_model_cached(
            cfg=MarketRegimenConfig(), fetch_fn=fetch_fn,
            start_date=str(start_hist), end_date=str(end_hist)
        )

    # -------- Compute today's regime --------
    cfg = MarketRegimenConfig()
    res = build_market_regime_section(
        cfg=cfg,
        fetch_fn=fetch_fn,
        start=pd.to_datetime(dt.date.today() - dt.timedelta(days=370)),
        end=pd.to_datetime(dt.date.today()),
        headline_mode=headline_mode,
        regime_model=model,
    )

    # ------------- TOP STRIP (your pasted UI) -------------
    top1, top2, top3 = st.columns([1.2, 1, 1])

    def _gauge(score: float, conf: float, title: str):
        score = float(score or 50.0)
        conf  = float(conf or 50.0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 36}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0,100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0,40], "color": "rgba(220,70,70,0.35)"},
                    {"range": [40,60], "color": "rgba(255,195,0,0.25)"},
                    {"range": [60,75], "color": "rgba(100,200,100,0.30)"},
                    {"range": [75,100], "color": "rgba(60,180,90,0.35)"},
                ],
                "threshold": {"line": {"width": 0}, "thickness": 0.95, "value": 100}
            }
        ))
        fig.add_annotation(
            x=0.5, y=0.15, xref="paper", yref="paper",
            text=f"Confidence: {conf:.0f}%",
            showarrow=False, font={"size": 12}
        )
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=0), height=220)
        return fig

    with top1:
        st.plotly_chart(_gauge(res.get("headline_score_0_100_final", 50.0),
                               res.get("headline_confidence_0_100", 50.0),
                               res.get("headline_regime","Neutral")), use_container_width=True)
        st.caption(res.get("headline_summary",""))

    with top2:
        st.metric("Agreement (categories in consensus)",
                  f"{res.get('headline_agreement_pct',0):.0f}%")
        st.metric("Learned 20-day drawdown risk",
                  f"{(res.get('learned_drawdown_prob_20d') or 0)*100:.1f}%",
                  help="Calibrated probability that SPY suffers a ≤−5% max drawdown in the next ~20 trading days.")
        _m = st.session_state.get("_mr_model_metrics") or (getattr(model, "metrics", None) if model else None)
        if _m:
            st.caption(f"Model (Brier {_m.get('brier',np.nan):.3f} • AUC {_m.get('auc',np.nan):.3f})")

    with top3:
        st.write("**Mode:**", mode_choice)
        st.progress(int(res.get("headline_confidence_0_100", 0)))
        st.caption("Higher is more internally consistent/less noisy signals.")

    # ------------- DRIVERS -------------
    st.markdown("#### Top drivers")
    contrib = res.get("contributions_from_neutral", {}) or {}
    if contrib:
        cdf = (pd.Series(contrib)
               .rename_axis("category")
               .reset_index(name="contribution"))
        cdf["side"] = np.where(cdf["contribution"]>=0, "Positive", "Negative")
        cdf = cdf.sort_values("contribution", ascending=True)

        fig = go.Figure()
        fig.add_bar(
            x=cdf["contribution"], y=cdf["category"],
            orientation="h", hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
        fig.update_layout(
            height=350, margin=dict(l=10,r=10,t=10,b=10),
            xaxis_title="Weighted contribution vs neutral (50)",
            yaxis_title=None
        )
        st.plotly_chart(fig, use_container_width=True)

    deltas = res.get("contributions_delta_since_yday")
    if isinstance(deltas, dict):
        st.caption("Δ since yesterday (what changed): " +
                   ", ".join(f"{k} {v:+.2f}" for k,v in sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]))

    # ------------- HAND OFF to your existing renderer -------------
    # It draws the radar/treemap/sliders and bars using mr['score_blocks'].
    # We pass the enriched dict so it can also read headline_summary etc if you choose.
    render_market_sentiment_dashboard(res, show_header=False, show_legacy_headline=False)


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
        signals_list=_resolve(cols, (
            "signals_list", "signals", "signals_fired", "signal_names",
            "signals_and_scores__signals", "signals_and_scores__signals_list",
            "signals_and_scores__signals_fired",
        )),
    )


def render_sidebar_filters(df: pd.DataFrame, cols: dict) -> tuple[pd.DataFrame, str]:
    """Render all sidebar filtering, custom rules, sort, and return filtered+sorted view and sort column."""
    st.sidebar.header("Filters")
    norm_method = st.sidebar.selectbox(
        "Score scaling",
        ["Percentile (0–1)", "Robust min–max (5–95%)", "Min–max (full)"],
        index=0,
        help="How composite/signals are mapped to a 0–1, dataset-relative scale."
    )
    _norm_key = {"Percentile (0–1)": "percentile",
                "Robust min–max (5–95%)": "winsor_minmax",
                "Min–max (full)": "minmax"}[norm_method]
    min_sig  = st.sidebar.number_input("Min signals (0–1, normalized)", 0.0, 1.0, 0.0, 0.05, key="min_sig")
    min_comp = st.sidebar.number_input("Min composite (0–1, normalized)", 0.0, 1.0, 0.0, 0.05, key="min_comp")
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

    if cols["comp"]:
        df[cols["comp"]] = _normalize_score_series(df[cols["comp"]])
    if cols["sig"]:
        df[cols["sig"]] = _normalize_score_series(df[cols["sig"]])

    if cols["comp"]:
        df["__comp_norm"] = _normalize_to_unit(df[cols["comp"]], method=_norm_key)
        cols["comp_norm"] = "__comp_norm"
    if cols["sig"]:
        df["__sig_norm"] = _normalize_to_unit(df[cols["sig"]], method=_norm_key)
        cols["sig_norm"] = "__sig_norm"

    mask = pd.Series(True, index=df.index)
    if cols.get("sig_norm"):
        mask &= df["__sig_norm"] >= float(min_sig)
    if cols.get("comp_norm"):
        mask &= df["__comp_norm"] >= float(min_comp)
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
    sort_candidates = [c for c in [cols.get("comp_norm"), cols.get("sig_norm"),cols["comp"], cols["sig"], cols["rsi"], cols["last"], cols["date"]] if c]
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

    # Row selection (read the canonical selection; do NOT create another selectbox)
    if cols["ticker"]:
        tickers = df_view[cols["ticker"]].astype(str).tolist()
        sel = st.session_state.get("sel_ticker")
        if not sel or sel not in tickers:
            sel = tickers[0] if tickers else None
            st.session_state["sel_ticker"] = sel
        row = df_view[df_view[cols["ticker"]].astype(str) == sel].iloc[0]
    else:
        row = df_view.iloc[0]

    # Series extraction
    x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(
        row, cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]
    )
    if x is None:
        st.warning("Selected row has empty or invalid series.")
        st.stop()

    # Table columns picker
    default_cols = [c for c in ["Reco", cols["ticker"], cols.get("comp_norm"), cols.get("sig_norm"),
                                cols["rsi"], cols["last"], cols["date"]]
                    if c and (c == "Reco" or c in df_view.columns)]
    chosen_cols = st.multiselect("Columns to show", options=list(df_view.columns),
                                default=default_cols, key="table_cols")

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
# Market Sentiment Dashboard helpers
# ────────────────────────────────────────────────────────────────────────────────
def _norm_weights(d: dict[str, float]) -> dict[str, float]:
    pairs = [(k, float(v) if v is not None else 0.0) for k, v in d.items()]
    total = sum(max(0.0, v) for _, v in pairs)
    if total <= 0:
        n = max(1, len(pairs))
        return {k: 1.0 / n for k, _ in pairs}
    return {k: max(0.0, v) / total for k, v in pairs}

def _headline_label(score: float) -> str:
    if score >= 75.0: return "Risk-On (Strong)"
    if score >= 60.0: return "Risk-On"
    if score >  40.0: return "Neutral"
    return "Risk-Off"

def _category_palette() -> dict[str, str]:
    # Pleasant, distinct accents for chips/plots
    return {
        "equities":    "#16a34a",
        "breadth":     "#10b981",
        "vol":         "#3b82f6",
        "rates_credit":"#0ea5e9",
        "fx":          "#a855f7",
        "intl":        "#f59e0b",
        "commodities": "#f97316",
        "reits":       "#8b5cf6",
        "crypto":      "#ef4444",
        "internals":   "#64748b",
    }

def _category_icons() -> dict[str, str]:
    return {
        "equities":"📈","breadth":"🧺","vol":"🌪️","rates_credit":"🏦",
        "fx":"💱","intl":"🌍","commodities":"⛏️","reits":"🏢","crypto":"🪙","internals":"🧭"
    }

def _parts_tooltip_html(cat: str, parts: dict | None) -> str:
    if not parts: return ""
    # Show top 6 contributors by absolute value, readable table
    try:
        items = sorted(parts.items(), key=lambda kv: (0 if kv[1] is None else -abs(float(kv[1]))))[:6]
    except Exception:
        items = list(parts.items())[:6]
    rows = "".join(
        f"<tr><td style='padding:2px 6px'>{_html.escape(str(k))}</td>"
        f"<td style='padding:2px 6px;text-align:right'>{'' if v is None else f'{float(v):.1f}'}</td></tr>"
        for k, v in items
    )
    return f"""
    <div class="section">
      <h5>{_html.escape(cat.title())} — components</h5>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr><th style='text-align:left'>Part</th><th style='text-align:right'>Score</th></tr></thead>
        <tbody>{rows or '<tr><td colspan="2">—</td></tr>'}</tbody>
      </table>
    </div>
    """

def _weighted_headline_from(cat_scores: dict[str, float | None], weights: dict[str, float]) -> float:
    ws = _norm_weights(weights)
    num, den = 0.0, 0.0
    for k, w in ws.items():
        v = cat_scores.get(k)
        if v is None or not np.isfinite(v): 
            continue
        num += float(v) * float(w)
        den += float(w)
    if den <= 0: 
        return 50.0
    return float(np.clip(num / den, 0.0, 100.0))

def render_weight_sliders(defaults: dict[str,float]) -> dict[str,float]:
    import streamlit as st
    with st.expander("Adjust category weights", expanded=False):
        st.caption("These control how the headline regime score is combined. Hover the (?) for details.")
        current = st.session_state.get("headline_weights", defaults).copy()

        new_weights = {}
        for key in CATEGORY_ORDER:
            label = CATEGORY_LABELS.get(key, key.title())
            help_txt = CATEGORY_HELP.get(key, "")
            new_weights[key] = st.slider(
                label=label,
                min_value=0.00, max_value=0.40, value=float(current.get(key, defaults[key])),
                step=0.01, help=help_txt, key=f"wt_{key}"
            )

        total = sum(new_weights.values())
        st.caption(f"Sum of sliders: **{total:.2f}**  (we normalize to 1.00 under the hood)")

        # Normalize to sum=1.00 so downstream math is stable
        if total <= 0:
            normed = defaults.copy()
        else:
            normed = {k: v/total for k, v in new_weights.items()}

        st.session_state["headline_weights"] = normed
        return normed

@st.cache_resource(show_spinner=False)
def _fit_regime_model_cached(cfg: MarketRegimenConfig, fetch_fn, start_date: str, end_date: str) -> RegimeModel | None:
    try:
        learner = RegimeLearnerConfig(
            enabled=True,
            horizon_days=20,
            dd_threshold=-0.05,  # next-20d MaxDD ≤ -5% event
            ridge_C=0.7,
            cv=5,
        )
        model = fit_regime_model_from_history(
            cfg=cfg,
            fetch_fn=fetch_fn,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            learner=learner,
        )
        return model
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Data source & load (shared across tabs)
# ────────────────────────────────────────────────────────────────────────────────
_inject_tooltip_css()

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
st.session_state["_comp_norm_col"] = cols.get("comp_norm")

# Add quick recommendation column (uses normalized comp/sig)
cn, sn = cols.get("comp_norm"), cols.get("sig_norm")
if cn and sn and cn in df_view.columns and sn in df_view.columns and "Reco" not in df_view.columns:
    df_view["Reco"] = [
        _quick_reco_from_norm(float(c), float(s))
        for c, s in zip(pd.to_numeric(df_view[cn], errors="coerce"),
                        pd.to_numeric(df_view[sn], errors="coerce"))
    ]

@st.cache_data(ttl=3600, show_spinner=False)
def _get_market_regime_cached() -> dict:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=400)
    cfg = MarketRegimenConfig()
    from core.utils import fetch_history
    raw = build_market_regime_section(cfg, fetch_history, start, end)
    return _sanitize_dict(raw)

# After df_view is built:
tickers = df_view[cols["ticker"]].astype(str).tolist() if cols["ticker"] else []
_default = tickers[0] if tickers else None

# canonical selection used by the app
shared = st.session_state.get("sel_ticker", _default)
if shared not in tickers:
    shared = _default
st.session_state["sel_ticker"] = shared

# seed per-tab widget state so both dropdowns show the same selection
st.session_state.setdefault("sel_ticker_review", shared)
st.session_state.setdefault("sel_ticker_backtest", shared)

# ────────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────────
tab_market, tab_review, tab_backtest, tab_settings, tab_profiles = st.tabs(
    ["🌐 Market Sentiment", "📊 Review", "🧠 Backtest", "⚙️ Settings", "💾 Profiles"]
)

# ────────────────────────────────────────────────────────────────────────
# 🌐 MARKET REGIME TAB
# ────────────────────────────────────────────────────────────────────────

with tab_market:
    st.subheader("Global Market Regime")
    try:
        market_regime = _get_market_regime_cached()
    except Exception:
        st.error("Market regime unavailable.")
    else:
        # ——— light styling for nicer-looking tabs ———
        st.markdown(
            """
            <style>
            .stTabs [data-baseweb="tab-list"] { gap: 8px; }
            .stTabs [data-baseweb="tab"] {
                padding: 10px 14px;
                border-radius: 12px;
                background: var(--secondary-background-color, rgba(255,255,255,0.04));
            }
            .stTabs [aria-selected="true"] {
                background: rgba(56, 189, 248, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.35);
            }
            .stTabs [data-baseweb="tab"]:hover {
                background: rgba(125,125,125,0.12);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Iconized tabs
        tab_dash, tab_bt = st.tabs(["📊 Dashboard", "🧪 Backtest"])

        with tab_dash:
            with st.spinner("Building dashboard…"):
                # Controller: uses market_regime as the fetch_fn
                render_market_sentiment_page(market_regime)

        with tab_bt:
            with st.spinner("Running backtest…"):
                render_market_regime_backtest(market_regime)

# ────────────────────────────────────────────────────────────────────────────────
# 📊 REVIEW TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_review:
    st.subheader("Results")

    if cols["ticker"]:
        tickers = df_view[cols["ticker"]].astype(str).tolist()

        def _sync_from_review():
            st.session_state["sel_ticker"] = st.session_state["sel_ticker_review"]
            # keep the other tab’s widget in sync next run
            st.session_state["sel_ticker_backtest"] = st.session_state["sel_ticker"]

        st.selectbox(
            "Select ticker",
            tickers,
            index=tickers.index(st.session_state["sel_ticker"]) if tickers else 0,
            key="sel_ticker_review",
            on_change=_sync_from_review,
        )

    # Row & series selection + table (table displayed here)
    row, chosen_cols, x, y_close, y_sma, y_open, y_high, y_low = pick_row_and_series(df_view, cols)

    column_config = {}
    comp_col = cols.get("comp_norm") or cols["comp"]
    sig_col  = cols.get("sig_norm")  or cols["sig"]

    if comp_col and comp_col in df_view.columns:
        column_config[comp_col] = st.column_config.ProgressColumn(
            "Composite (norm)", help="0–1, dataset-relative", min_value=0.0, max_value=1.0, format="%.2f"
        )
    if sig_col and sig_col in df_view.columns:
        column_config[sig_col] = st.column_config.ProgressColumn(
            "Signals (norm)", help="0–1, dataset-relative", min_value=0.0, max_value=1.0, format="%.2f"
        )
    if cols["rsi"] and cols["rsi"] in df_view.columns:
        column_config[cols["rsi"]] = st.column_config.ProgressColumn(
            "RSI", min_value=0, max_value=100, format="%d"
        )
    if cols["owned"] and cols["owned"] in df_view.columns:
        column_config[cols["owned"]] = st.column_config.CheckboxColumn("Owned")

    if "Reco" in df_view.columns:
        column_config["Reco"] = st.column_config.TextColumn(
            "Reco", help="Quick classification from normalized scores", width="small"
        )

    st.dataframe(
        df_view[chosen_cols],
        use_container_width=True, hide_index=True,
        column_config=column_config
    )

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
    render_kpis(row, cols, x, y_close, y_sma, y_open, y_high, y_low, df_view)

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
# 🧪 BACKTEST TAB
# ────────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest & DCA simulator")

    # ---- Shared ticker selection (syncs with Review) ----
    if cols["ticker"]:
        tickers = df_view[cols["ticker"]].astype(str).tolist()

        def _sync_from_backtest():
            st.session_state["sel_ticker"] = st.session_state["sel_ticker_backtest"]
            # keep Review tab’s widget in sync next run
            st.session_state["sel_ticker_review"] = st.session_state["sel_ticker"]

        st.selectbox(
            "Select ticker",
            tickers,
            index=tickers.index(st.session_state["sel_ticker"]) if tickers else 0,
            key="sel_ticker_backtest",
            on_change=_sync_from_backtest,
        )

    # use the shared selection to choose the row/series
    sel = st.session_state.get("sel_ticker")
    row = (df_view[df_view[cols["ticker"]].astype(str) == sel].iloc[0]
           if (sel and not df_view.empty) else (df_view.iloc[0] if not df_view.empty else df.iloc[0]))


    x, y_close, y_sma, y_open, y_high, y_low = _get_series_lists(
        row, cols["dates_series"], cols["close_series"], cols["sma200_series"], cols["open_series"], cols["high_series"], cols["low_series"]
    )

    # ---- Quick profile (same as before) ----
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

    # ---- Tune parameters (same as previous Signals tab, but inside this tab) ----
    with st.expander("Tune parameters", expanded=False):
        tab_buy, tab_sell, tab_risk = st.tabs(["BUY thresholds & weights", "SELL thresholds & weights", "Risk & sizing"])

        with tab_buy:
            c1, c2 = st.columns(2)
            with c1:
                composite_threshold = st.slider("BUY: Composite threshold", 0.30, 0.90, float(preset["composite_threshold"]), 0.01, help=h("composite_threshold"), key="sig_composite_threshold")
                rsi_buy_max        = st.slider("BUY: RSI buy max", 20, 70, int(preset["rsi_buy_max"]), 1, help=h("rsi_buy_max"), key="sig_rsi_buy_max")
                vol_ratio_min      = st.slider("Min vol / avg20 (both)", 1.0, 3.0, float(preset["vol_ratio_min"]), 0.05, help=h("vol_ratio_min"), key="sig_vol_ratio_min")
                value_center       = st.slider("BUY: Value center vs SMA200 (%)", -20.0, 10.0, -5.0, 0.5, help=h("value_center"), key="sig_value_center")
                donch_lookback     = st.slider("BUY: Donchian lookback", 10, 60, int(preset.get("donch_lookback", 20)), 1, help=h("donch_lookback"), key="sig_donch_lookback")
                gap_min_pct        = st.slider("BUY: Gap-up min % (vs prev high)", 0.0, 3.0, float(preset.get("gap_min_pct", 0.5)), 0.1, help=h("gap_min_pct"), key="sig_gap_min_pct")
            with c2:
                atr_mult           = st.slider("BUY: ATR stop ×", 0.8, 3.0, float(preset["atr_mult"]), 0.1, help=h("atr_mult"), key="sig_atr_mult")
                stop_pct_ui        = st.slider("BUY: Fallback % stop", 3.0, 20.0, float(preset["stop_pct"]), 0.5, help=h("stop_pct_ui"), key="sig_stop_pct_ui")
                sma_window         = st.slider("Trend window (days, buy)", 100, 300, 200, 10, help=h("sma_window"), key="sig_sma_window")
                use_engine_stop    = st.checkbox("Use engine stop/target on chart", value=False, help=h("use_engine_stop"), key="sig_use_engine_stop")
                bb_window          = st.slider("Bollinger window", 10, 60, int(preset.get("bb_window", 20)), 1, help=h("bb_window"), key="sig_bb_window")
                bb_k               = st.slider("Bollinger k (σ)", 1.0, 3.0, float(preset.get("bb_k", 2.0)), 0.1, help=h("bb_k"), key="sig_bb_k")
            st.markdown("**Weights**")
            c3, c4 = st.columns(2)
            with c3:
                w_rsi   = st.slider("Weight (BUY): RSI", 0.0, 1.0, float(preset["w_rsi"]), 0.05, help=h("w_rsi"), key="sig_w_rsi")
                w_trend = st.slider("Weight (BUY): Trend", 0.0, 1.0, float(preset["w_trend"]), 0.05, help=h("w_trend"), key="sig_w_trend")
                w_value = st.slider("Weight (BUY): Value", 0.0, 1.0, float(preset["w_value"]), 0.05, help=h("w_value"), key="sig_w_value")
            with c4:
                w_flow   = st.slider("Weight (BUY): Flow", 0.0, 1.0, float(preset["w_flow"]), 0.05, help=h("w_flow"), key="sig_w_flow")
                w_bbands = st.slider("Weight (BUY): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands", 0.20)), 0.05, help=h("w_bbands"), key="sig_w_bbands")
                w_donch  = st.slider("Weight (BUY): Donchian",     0.0, 1.0, float(preset.get("w_donchian", 0.25)), 0.05, help=h("w_donch"), key="sig_w_donch")
                w_break  = st.slider("Weight (BUY): Legacy breakout", 0.0, 1.0, float(preset.get("w_breakout", 0.00)), 0.05, help=h("w_break"), key="sig_w_break")

        with tab_sell:
            c1, c2 = st.columns(2)
            with c1:
                sell_threshold      = st.slider("SELL: Composite threshold", 0.30, 0.90, float(preset["sell_threshold"]), 0.01, help=h("sell_threshold"), key="sig_sell_threshold")
                rsi_overbought_min  = st.slider("SELL: RSI overbought min", 55, 85, int(preset["rsi_overbought_min"]), 1, help=h("rsi_overbought_min"), key="sig_rsi_overbought_min")
                donch_lookback_sell = st.slider("SELL: Donchian lookback", 10, 60, int(preset["donch_lookback_sell"]), 1, help=h("donch_lookback_sell"), key="sig_donch_lookback_sell")
            with c2:
                ema_fast_span   = st.slider("SELL: EMA fast span", 5, 55, int(preset["ema_fast_span"]), 1, help=h("ema_fast_span"), key="sig_ema_fast_span")
                sma_mid_window  = st.slider("SELL: SMA mid window", 20, 100, int(preset["sma_mid_window"]), 1, help=h("sma_mid_window"), key="sig_sma_mid_window")
                gap_down_min_pct = st.slider("SELL: Gap-down min % (vs prev low)", 0.0, 3.0, float(preset["gap_down_min_pct"]), 0.1, help=h("gap_down_min_pct"), key="sig_gap_down_min_pct")
            st.markdown("**Weights**")
            c3, c4 = st.columns(2)
            with c3:
                w_rsi_sell   = st.slider("Weight (SELL): RSI", 0.0, 1.0, float(preset.get("w_rsi_sell", 0.30)), 0.05, help=h("w_rsi_sell"), key="sig_w_rsi_sell")
                w_trend_down = st.slider("Weight (SELL): Trend down", 0.0, 1.0, float(preset.get("w_trend_down", 0.30)), 0.05, help=h("w_trend_down"), key="sig_w_trend_down")
                w_breakdown  = st.slider("Weight (SELL): Breakdown", 0.0, 1.0, float(preset.get("w_breakdown", 0.25)), 0.05, help=h("w_breakdown"), key="sig_w_breakdown")
            with c4:
                w_exhaustion = st.slider("Weight (SELL): Exhaustion", 0.0, 1.0, float(preset.get("w_exhaustion", 0.10)), 0.05, help=h("w_exhaustion"), key="sig_w_exhaustion")
                w_flow_out   = st.slider("Weight (SELL): Flow out", 0.0, 1.0, float(preset.get("w_flow_out", 0.05)), 0.05, help=h("w_flow_out"), key="sig_w_flow_out")
                w_bbands_sell = st.slider("Weight (SELL): Bollinger %B", 0.0, 1.0, float(preset.get("w_bbands_sell", 0.25)), 0.05, help=h("w_bbands_sell"), key="sig_w_bbands_sell")
                w_donch_sell  = st.slider("Weight (SELL): Donchian",     0.0, 1.0, float(preset.get("w_donchian_sell", 0.25)), 0.05, help=h("w_donch_sell"), key="sig_w_donch_sell")

        with tab_risk:
            c1, c2 = st.columns(2)
            with c1:
                portfolio_value = st.number_input("Portfolio ($)", min_value=1000.0, value=20000.0, step=1000.0, key="sig_portfolio_value")
                risk_per_trade_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="sig_risk_per_trade_pct")
            with c2:
                min_adv_dollars = st.number_input("Min ADV$ (liquidity)", min_value=0.0, value=250000.0, step=25000.0, key="sig_min_adv_dollars")

    # ---- Build params from current widget values ----
    params = BuyParams(
        composite_threshold=float(st.session_state.get("sig_composite_threshold", preset["composite_threshold"])),
        w_rsi=float(st.session_state.get("sig_w_rsi", preset["w_rsi"])),
        w_trend=float(st.session_state.get("sig_w_trend", preset["w_trend"])),
        w_breakout=float(st.session_state.get("sig_w_break", preset.get("w_breakout", 0.0))),
        w_value=float(st.session_state.get("sig_w_value", preset["w_value"])),
        w_flow=float(st.session_state.get("sig_w_flow", preset["w_flow"])),
        rsi_buy_max=float(st.session_state.get("sig_rsi_buy_max", preset["rsi_buy_max"])), rsi_floor=20.0,
        sma200_window=int(st.session_state.get("sig_sma_window", 200)),
        donch_lookback=int(st.session_state.get("sig_donch_lookback", preset.get("donch_lookback", 20))),
        gap_min_pct=float(st.session_state.get("sig_gap_min_pct", preset.get("gap_min_pct", 0.5))),
        value_center_dev_pct=float(st.session_state.get("sig_value_center", -5.0)),
        vol_ratio_min=float(st.session_state.get("sig_vol_ratio_min", preset["vol_ratio_min"])),
        use_engine_stop=bool(st.session_state.get("sig_use_engine_stop", False)),
        atr_mult=float(st.session_state.get("sig_atr_mult", preset["atr_mult"])),
        stop_pct=float(st.session_state.get("sig_stop_pct_ui", preset["stop_pct"])),
        reward_R=float(preset["reward_R"]),
        portfolio_value=float(st.session_state.get("sig_portfolio_value", 20000.0)),
        risk_per_trade_pct=float(st.session_state.get("sig_risk_per_trade_pct", 0.5)),
        min_price=1.0, min_adv_dollars=float(st.session_state.get("sig_min_adv_dollars", 250000.0)),
        w_bbands=float(st.session_state.get("sig_w_bbands", preset.get("w_bbands", 0.20))),
        w_donchian=float(st.session_state.get("sig_w_donch", preset.get("w_donchian", 0.25))),
        bb_window=int(st.session_state.get("sig_bb_window", preset.get("bb_window", 20))),
        bb_k=float(st.session_state.get("sig_bb_k", preset.get("bb_k", 2.0))),

        w_bbands_sell=float(st.session_state.get("sig_w_bbands_sell", preset.get("w_bbands_sell", 0.25))),
        w_donchian_sell=float(st.session_state.get("sig_w_donch_sell", preset.get("w_donchian_sell", 0.25))),
        sell_threshold=float(st.session_state.get("sig_sell_threshold", preset["sell_threshold"])),
        w_rsi_sell=float(st.session_state.get("sig_w_rsi_sell", preset.get("w_rsi_sell", 0.30))),
        w_trend_down=float(st.session_state.get("sig_w_trend_down", preset.get("w_trend_down", 0.30))),
        w_breakdown=float(st.session_state.get("sig_w_breakdown", preset.get("w_breakdown", 0.25))),
        w_exhaustion=float(st.session_state.get("sig_w_exhaustion", preset.get("w_exhaustion", 0.10))),
        w_flow_out=float(st.session_state.get("sig_w_flow_out", preset.get("w_flow_out", 0.05))),
        rsi_overbought_min=float(st.session_state.get("sig_rsi_overbought_min", preset["rsi_overbought_min"])),
        ema_fast_span=int(st.session_state.get("sig_ema_fast_span", preset["ema_fast_span"])),
        sma_mid_window=int(st.session_state.get("sig_sma_mid_window", preset["sma_mid_window"])),
        donch_lookback_sell=int(st.session_state.get("sig_donch_lookback_sell", preset["donch_lookback_sell"])),
        gap_down_min_pct=float(st.session_state.get("sig_gap_down_min_pct", preset["gap_down_min_pct"])),
    )

    # ---- Compute engines (Signals summary) ----
    buy_res  = compute_buy_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    sell_res = compute_sell_signal(row=row, dates=x, close=y_close, sma200=y_sma, open_=y_open, high=y_high, low=y_low, params=params)
    action, action_margin = _decide_action(buy_res, sell_res, params.composite_threshold, params.sell_threshold)

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

    # ---- Backtest (now in same tab) ----
    st.markdown("---")
    st.subheader("Backtest & DCA simulator")

    c1, c2, c3 = st.columns(3)
    with c1:
        starting_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0, key="bt_start_cash")
        buy_pct_first = st.slider("Buy % on first BUY", 0.0, 100.0, 25.0, 5.0, key="bt_buy_first")
        buy_pct_next  = st.slider("Buy % on next BUY",  0.0, 100.0, 25.0, 5.0, key="bt_buy_next")
    with c2:
        dca_trigger_drop_pct = st.slider("Extra BUY only if price ≤ last buy by (%)", 0.0, 50.0, 5.0, 1.0, key="bt_dca_trigger")
        max_dca_legs = st.slider("Max DCA legs per accumulation", 0, 10, 3, 1, key="bt_dca_legs")
    with c3:
        sell_pct_first = st.slider("Sell % on first SELL", 0.0, 100.0, 50.0, 5.0, key="bt_sell_first")
        sell_pct_next  = st.slider("Sell % on next SELL",  0.0, 100.0, 50.0, 5.0, key="bt_sell_next")

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

    aset = get_settings()["chart"]
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=x, y=y_close, mode="lines", name="Close"))
    if bool(st.session_state.get("show_sma", get_settings()["chart"]["show_sma"])):
        fig_bt.add_trace(go.Scatter(x=x, y=y_sma, mode="lines", name="SMA200", line=dict(color=th["sma200"])))

    # Reuse overlays (from Review)
    chosen_overlays_bt = st.session_state.get("overlays", [])
    overlay_params_bt  = st.session_state.get("overlay_params_loaded", {})
    bt_shapes, bt_annotations = [], []
    for key in chosen_overlays_bt:
        entry = TECHNICALS_REGISTRY.get(key)
        if not entry:
            continue
        p = overlay_params_bt.get(key, entry.get("params", {}))
        res = entry["fn"](x, pd.Series(y_close), pd.Series(y_sma), row, p)
        for tr in (res.traces or []):
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

    # Make params available to other places
    st.session_state["last_used_params"] = params

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
    with tabD:  # Overlays
        render_overlays_defaults(
            s,
            all_opts=list(TECHNICALS_REGISTRY.keys())
        )

    # ── Performance & Data ─────────────────────────────────────────────────────
    with tabE:  # Performance & Data
        render_perf_and_data_defaults(
            s,
            DEFAULT_SETTINGS  # pass the dict so the function can reset/import safely
        )

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
