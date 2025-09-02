"""
Market Regimen (macro guide, not per-stock dictator)

This module computes a market "regimen" snapshot using broad equity indexes,
volatility, rates/credit, risk-on/off pairs, international, commodities/real
assets, FX, crypto (optional), and market internals. Outputs are lightweight,
explainable features (trend, momentum, percentiles, ratios) plus composite
"risk level" and short/long-term "bias" labels to guide (not override) stock
decisions.

It reuses `sector_trend_score` from `etf_sector_analysis.py`.

---------------------------------------------------------------------------
ETF / INDEX MENU (What each target represents)
---------------------------------------------------------------------------
Core US Index & Breadth
- SPY  : S&P 500 market-cap weighted
- QQQ  : Nasdaq-100 (large-cap growth/tech tilt)
- DIA  : Dow Jones Industrial Average (price-weighted blue chips)
- IWM  : Russell 2000 (small caps)
- VTI  : Total US equity market
- RSP  : S&P 500 equal-weight (breadth proxy vs SPY)

Volatility (VIX Complex)
- ^VIX   : S&P 500 30-day implied volatility index level
- ^VIX9D : 9-day VIX (very short-term)
- ^VIX3M : 3-month VIX (aka ^VXV)
- VIXY   : Short-term VIX futures ETF (proxy if ^VIX unavailable)
- VIXM   : Mid-term VIX futures ETF (proxy for term structure)

Rates / Duration / Inflation
- BIL/SHV : T-bills (cash-like)
- SHY     : 1–3Y Treasuries
- IEF     : 7–10Y Treasuries
- TLT     : 20Y+ Treasuries
- ZROZ/EDV: STRIPS (ultra duration)
- TIP     : TIPS (inflation-protected)
(Useful ratios: TIP/IEF ≈ breakeven inflation; SHY/TLT ≈ curve/term-premium stress)

Credit (spread risk appetite)
- LQD : Investment-grade corporate bonds
- HYG : High-yield (junk) corporate bonds
- JNK : High-yield (alt)
- EMB : EM sovereign USD debt
(Useful ratio: HYG/IEF ↑ = credit risk-on)

Risk-on / Risk-off Pairs
- XLY/XLP : Consumer Discretionary vs Staples
- XLF/XLU : Financials vs Utilities
- XLI/XLU : Industrials vs Utilities
- RSP/SPY : Breadth vs mega-cap concentration
- IWM/SPY : Small vs large caps
- HYG/IEF : Credit vs Treasuries (again)

International Equities
- ACWI : All-country world (incl. US)
- ACWX : All-country ex-US
- EFA  : Developed ex-US
- VEA  : Developed ex-US (alt)
- EEM  : Emerging markets
- VWO  : Emerging markets (alt)
- VGK  : Europe
- EWJ  : Japan
- MCHI/FXI : China large-cap

Commodities / Real Assets / Equity Proxies
- DBC/BCI : Broad commodities
- USO/BNO : Crude oil (WTI/Brent)
- UNG     : Natural gas
- GLD/IAU : Gold
- SLV     : Silver
- DBB     : Base metals basket
- CPER    : Copper
- URA     : Uranium equities
- DBA     : Agriculture
- XLE     : Energy equities
- XLB     : Materials equities
- VNQ     : US REITs (real estate)
- IFRA/PAVE: Infrastructure

Currencies
- ^DXY : US Dollar Index
- UUP  : US Dollar ETF proxy
- FXE  : Euro
- FXY  : Yen
- CEW  : EM currencies basket

Crypto (optional risk sentiment)
- IBIT : Bitcoin (iShares spot ETF)
- FBTC : Bitcoin (Fidelity spot ETF)  (you may choose one)
- BITO : Bitcoin futures ETF
- EETH : Ether (iShares spot ETF)     (or any ETH spot ticker you track)

Market Internals & Sentiment (indexes, not ETFs)
- CPC   : Total put/call ratio (high = fear)
- CPCE  : Equity-only put/call ratio
- SKEW  : Tail-risk pricing (higher = more tail risk priced)
- ^MOVE : Bond volatility index (credit/liquidity stress)
- %>MA  : % of constituents above 50/200dma (approx via RSP/SPY if breadth breadth data not available)
- A-D   : Advance/Decline line (if you have access)
- Realized vol: rolling SPY stdev
- VIX term structure: ^VIX9D - ^VIX3M (backwardation > 0 = stress)

Macro (optional overlays; if your fetcher supports FRED or macro series)
- PMI/ISM, Initial Claims, Payrolls, CPI/PCE, UMich sentiment, NFIB
- Liquidity proxies: WALCL (Fed balance sheet), RRPONTSYD (ON RRP),
  Treasury General Account (TGA; series name varies), policy rate (FFR)

---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

import sys

try:
    # optional; we guard if not installed
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

from .etf_sector_analysis import sector_trend_score


# =========================== Machine Learning Tests =================================

@dataclass
class RegimeLearnerConfig:
    """Controls the learned-headline (probability) model."""
    enabled: bool = True                # turn on/off the learner
    horizon_days: int = 20              # predict risk over next N trading days
    dd_threshold: float = -0.05         # event = max drawdown <= -5%
    ridge_C: float = 0.7                # regularization (smaller -> more shrinkage)
    cv: int = 5                         # calibration folds
    use_features: Tuple[str, ...] = (   # default driver set (category-level + a few primitives)
        "equities","breadth","vol","rates_credit","fx","intl","commodities","reits","crypto","internals",
        "vix_pctile_1y","breadth_pctile_1y","credit_above_200dma","dxy_trend","uup_trend"
    )

@dataclass
class RegimeModel:
    """Fitted model + metadata kept in-memory (or pickle on disk in your app)."""
    model: Any
    feature_names: List[str]
    horizon_days: int
    dd_threshold: float
    metrics: Dict[str, float] | None = None  # brier, auc on validation (optional)

# Headline selector (user sliders vs learned vs blend)
@dataclass
class HeadlineMode:
    mode: str = "user"       # "user" | "learned" | "blend"
    blend_alpha: float = 0.7 # fraction of LEARNED in the blend, e.g., 0.7 * learned + 0.3 * user

# Overlap penalty between near-duplicate categories (to reduce double counting)
OVERLAP_PAIRS = [
    ("equities", "breadth"),
    ("rates_credit", "reits"),
]
OVERLAP_STRENGTH = 0.20  # max 20% shrink when two categories are basically identical at t

def _apply_overlap_penalty(weights: dict[str,float], cat_scores: dict[str,float|None]) -> dict[str,float]:
    """Shrink weights when two categories are moving in lockstep (same-side, similar magnitude)."""
    ws = {k: float(v) for k, v in weights.items()}
    for a, b in OVERLAP_PAIRS:
        sa = cat_scores.get(a); sb = cat_scores.get(b)
        if isinstance(sa, (int,float)) and isinstance(sb, (int,float)):
            # closeness 0..1 (identical -> 1, opposite -> 0)
            closeness = 1.0 - min(abs(sa - sb) / 100.0, 1.0)
            shrink = 1.0 - (OVERLAP_STRENGTH * closeness)
            ws[a] *= shrink
            ws[b] *= shrink
    # renormalize to sum ~1
    s = sum(max(0.0,v) for v in ws.values()) or 1.0
    return {k: max(0.0,v)/s for k,v in ws.items()}

def _contribs_from_scores(cat_scores: dict[str, float|None], weights: dict[str,float]) -> dict[str,float]:
    """Contribution relative to neutral 50: weight * (score - 50)."""
    out = {}
    for k, w in weights.items():
        v = cat_scores.get(k)
        out[k] = (w * ((float(v) if v is not None else 50.0) - 50.0)) if w>0 else 0.0
    return out

def _agreement_percent(cat_scores: dict[str, float|None], thr_hi=55.0, thr_lo=45.0) -> float:
    """Share of categories decisively on one side; mixed bins count as neutral."""
    pos = sum(1 for v in cat_scores.values() if isinstance(v,(int,float)) and v>=thr_hi)
    neg = sum(1 for v in cat_scores.values() if isinstance(v,(int,float)) and v<=thr_lo)
    tot = sum(1 for v in cat_scores.values() if isinstance(v,(int,float)))
    return 0.0 if tot==0 else 100.0 * max(pos,neg) / tot

def _confidence_from_dispersion(cat_scores: dict[str,float|None]) -> float:
    """High dispersion -> low confidence; low dispersion & strong agreement -> high."""
    xs = [float(v) for v in cat_scores.values() if isinstance(v,(int,float))]
    if not xs: return 50.0
    disp = np.std(xs)
    base = max(0.0, 100.0 - (disp * 1.1))  # 0..100, wider distribution -> smaller number
    # small bump if strong consensus
    return float(np.clip(base + 0.25 * _agreement_percent(cat_scores), 0, 100))

def _regime_with_hysteresis(score_now: float, prev: Optional[float], up=5.0, down=3.0) -> str:
    """Hysteresis around buckets to reduce flip-flops."""
    # Base label thresholds
    base = _regime_label(score_now)
    if prev is None: 
        return base
    # Nudged thresholds around previous bucket
    prev_lab = _regime_label(prev)
    if prev_lab.startswith("Risk-On") and score_now >= 60.0 - down: return "Risk-On"
    if prev_lab == "Neutral":
        if score_now >= 60.0 + up: return "Risk-On"
        if score_now <= 40.0 - down: return "Risk-Off"
        return "Neutral"
    if prev_lab == "Risk-Off" and score_now <= 40.0 + up: return "Risk-Off"
    return base
# ============================================================================ 

# --------------------------- Registries -------------------------------------

# Core index / breadth
CORE_INDEX: Dict[str, str] = {
    "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM",
    "VTI": "VTI", "RSP": "RSP",
}

# Volatility complex
VOL_SYMS: Dict[str, str] = {
    "VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M",  # ^VXV ~ VIX3M
    "VIXY": "VIXY", "VIXM": "VIXM",  # ETF proxies
}

# Rates / Duration / Inflation
RATES: Dict[str, str] = {
    "BIL": "BIL", "SHV": "SHV", "SHY": "SHY", "IEF": "IEF", "TLT": "TLT",
    "ZROZ": "ZROZ", "EDV": "EDV", "TIP": "TIP",
}

# Credit
CREDIT: Dict[str, str] = {"LQD": "LQD", "HYG": "HYG", "JNK": "JNK", "EMB": "EMB"}

# Risk-on/off pairs (compute on the fly as ratios)
RISK_PAIRS: List[Tuple[str, str]] = [
    ("XLY", "XLP"), ("XLF", "XLU"), ("XLI", "XLU"),
    ("RSP", "SPY"), ("IWM", "SPY"), ("HYG", "IEF"),
]


# Pairs whose quote strengthens when the denominator currency weakens -> invert to express local strength.
_PAIR_INVERT: Dict[str, bool] = {
    "USDCAD=X": True,   # invert so ↑ means stronger CAD
    "USDJPY=X": True,   # invert so ↑ means stronger JPY
    "USDCHF=X": True,   # invert so ↑ means stronger CHF
    # Already "local/USD" (↑ = stronger local), so no invert:
    "EURUSD=X": False,
    "GBPUSD=X": False,
    "AUDUSD=X": False,
}

# Commodities / Real assets
COMMS: Dict[str, str] = {
    "DBC": "DBC", "BCI": "BCI",
    "USO": "USO", "BNO": "BNO", "UNG": "UNG",
    "GLD": "GLD", "IAU": "IAU", "SLV": "SLV",
    "DBB": "DBB", "CPER": "CPER", "URA": "URA",
    "DBA": "DBA", "VNQ": "VNQ", "XLE": "XLE", "XLB": "XLB",
    "IFRA": "IFRA", "PAVE": "PAVE",
}

# --- Regional ETF Menus (Europe & Asia) -------------------------------------
EU_BROAD_ETFS: dict[str, str] = {
    "Europe_Dev": "IEUR",
    "Eurozone":   "EZU",
    "EuroStoxx50":"FEZ",
    "UK":         "EWU",
    "Switzerland":"EWL",
}

ASIA_BROAD_ETFS: dict[str, str] = {
    "Asia_Ex_Japan": "AAXJ",
    "Japan":         "EWJ",
    "Australia":     "EWA",
    "South_Korea":   "EWY",
    "Taiwan":        "EWT",
    "India":         "INDA",
    "Singapore":     "EWS",
    "Hong_Kong":     "EWH",
}

GLOBAL_REITS: dict[str, str] = {
    "US_VNQ":        "VNQ",
    "US_IYR":        "IYR",
    "Global_REITs":  "REET",
    "Global_exUS":   "VNQI",
    "Intl_Dev_RE":   "IFGL",
    "Europe_RE":     "IFEU",
    "AsiaPac_RE":    "IFAS",
    "Canada_RE_XRE": "XRE.TO",
}

# ---- Foreign Exchange (FX) proxies & spot pairs -----------------------------
# ETFs (direction = "local currency strength"): higher = stronger local FX.
FX: Dict[str, str] = {
    "DXY": "^DXY",   # US Dollar Index (broad USD strength)
    "UUP": "UUP",    # US Dollar ETF proxy (tracks DXY)
    "FXE": "FXE",    # Euro
    "FXB": "FXB",    # British Pound
    "FXY": "FXY",    # Japanese Yen
    "FXA": "FXA",    # Australian Dollar
    "FXC": "FXC",    # Canadian Dollar
    "CEW": "CEW",    # EM currencies basket
    # Optional: Yahoo spot pairs (used if available; some regions may be NA)
    "EURUSD=X": "EURUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "AUDUSD=X": "AUDUSD=X",
    "USDCAD=X": "USDCAD=X",
    "USDJPY=X": "USDJPY=X",
    "USDCHF=X": "USDCHF=X",
}

# ---- Expanded International equities (broad + country/region)
INTL: Dict[str, str] = {
    # keep yours
    "ACWI": "ACWI", "ACWX": "ACWX",
    "EFA": "EFA", "VEA": "VEA",
    "EEM": "EEM", "VWO": "VWO",
    "VGK": "VGK", "EWJ": "EWJ",
    "MCHI": "MCHI", "FXI": "FXI",
    # additions (Europe & Asia)
    "EZU": "EZU",          # Eurozone
    "IEUR": "IEUR",        # Dev. Europe broad
    "FEZ": "FEZ",          # Euro Stoxx 50
    "EWU": "EWU",          # UK
    "EWL": "EWL",          # Switzerland
    "EWA": "EWA",          # Australia
    "AAXJ": "AAXJ",        # Asia ex-Japan
    "EWY": "EWY",          # South Korea
    "EWT": "EWT",          # Taiwan
    "INDA": "INDA",        # India
    "EWS": "EWS",          # Singapore
    "EWH": "EWH",          # Hong Kong
}

# ---- Crypto (spot/futures ETFs) --------------------------------------------
CRYPTO: Dict[str, str] = {
    "IBIT": "IBIT",    # iShares Spot Bitcoin ETF (primary BTC proxy)
    "FBTC": "FBTC",    # Fidelity Spot Bitcoin ETF (alt BTC proxy)
    "BITO": "BITO",    # Bitcoin futures ETF (use if spot not available)
    "EETH": "EETH",    # iShares Spot Ether ETF (ETH proxy)
    # Optional: pull direct spot if your fetcher supports it (kept out by default)
    # "BTC-USD": "BTC-USD",
    # "ETH-USD": "ETH-USD",
}

# Internals / sentiment (indexes; support if fetcher can)
INTERNALS: Dict[str, str] = {"CPC": "CPC", "CPCE": "CPCE", "SKEW": "SKEW", "MOVE": "^MOVE"}

# ---- Regional REITs (Real Estate) ------------------------------------------
# US, global, ex-US, and regional REIT baskets to detect tilts into real assets.
# Note: Some tickers may be thin/region-limited on Yahoo; they will be skipped if not found.
REITS: Dict[str, str] = {
    "US_REITs_VNQ":     "VNQ",     # Vanguard US REITs
    "US_REITs_IYR":     "IYR",     # iShares US Real Estate
    "Global_REITs_REET":"REET",    # Global REITs (incl US)
    "Global_exUS_VNQI": "VNQI",    # Global ex-US real estate
    "Intl_Dev_RE_IFGL": "IFGL",    # Intl developed real estate (alt to VNQI)
    "Europe_REITs_IFEU":"IFEU",    # Europe dev real estate
    "AsiaPac_REITs_IFAS":"IFAS",   # Asia Pacific dev real estate (if available)
    "Canada_REITs_XRE": "XRE.TO",  # TSX Capped REIT (CAD listing)
}

# --------------------------- Config -----------------------------------------

@dataclass
class MarketRegimenConfig:
    """
    Configuration for market regimen computation.

    Attributes
    ----------
    include_groups : Dict[str, bool]
        Toggle which groups to fetch. Keys:
        ['core','vol','rates','credit','intl','comms','fx','crypto','internals']
    trend_lookback_wks : int
        Lookback window for weekly trend score.
    risk_weights : Dict[str, float]
        Weights for risk composite (0..1, ~sum to 1):
          - 'vix' : VIX percentile (default 0.50)
          - 'dd'  : SPY 52w drawdown severity (default 0.30)
          - 'mom' : Negative 13w momentum penalty (default 0.20)
    """
    # what to include (you can override per runtime)
    include_groups: Dict[str, bool] = field(default_factory=lambda: {
        "core": True, "vol": True, "rates": True, "credit": True,
        "intl": True, "comms": True, "fx": True, "crypto": False,  # crypto off by default
        "internals": True,
    })
    trend_lookback_wks: int = 26
    risk_weights: Dict[str, float] = field(default_factory=lambda: {"vix": 0.50, "dd": 0.30, "mom": 0.20})

    # custom symbol maps (override if you like)
    core: Dict[str, str] = field(default_factory=lambda: CORE_INDEX.copy())
    vol: Dict[str, str] = field(default_factory=lambda: VOL_SYMS.copy())
    rates: Dict[str, str] = field(default_factory=lambda: RATES.copy())
    credit: Dict[str, str] = field(default_factory=lambda: CREDIT.copy())
    intl: Dict[str, str] = field(default_factory=lambda: INTL.copy())
    comms: Dict[str, str] = field(default_factory=lambda: COMMS.copy())
    fx: Dict[str, str] = field(default_factory=lambda: FX.copy())
    crypto: Dict[str, str] = field(default_factory=lambda: CRYPTO.copy())
    internals: Dict[str, str] = field(default_factory=lambda: INTERNALS.copy())

    # optional knobs
    risk_pairs: List[Tuple[str, str]] = field(default_factory=lambda: RISK_PAIRS.copy())
    realized_vol_window_days: int = 20
    breadth_use_rsp_over_spy: bool = True  # when true, RSP/SPY serves as breadth proxy


# --------------------------- Utilities --------------------------------------

def _pct_change_safe(px: pd.Series | pd.DataFrame, periods: int = 1):
    return px / px.shift(periods) - 1.0

def _drawdown_from_high(px: pd.Series, lookback_days: int = 252) -> Optional[float]:
    if px is None or px.empty:
        return None
    w = px.tail(lookback_days).dropna()
    if w.empty:
        return None
    last = w.iloc[-1]
    if not np.isfinite(last):
        return None
    hi = float(np.nanmax(w.values))
    if not np.isfinite(hi) or hi <= 0:
        return None
    return float((last / hi - 1.0) * 100.0)  # negative % if below high
    

def ma(series: pd.Series, n: int = 200) -> pd.Series:
    return series.rolling(n, min_periods=max(5, n // 5)).mean()

def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    r = (a / b).replace([np.inf, -np.inf], np.nan)
    return r

def percentile_last(series: pd.Series, lookback: int = 252) -> Optional[float]:
    s = series.dropna().tail(lookback)
    if len(s) < max(40, lookback // 5):
        return None
    return float((s <= s.iloc[-1]).mean() * 100.0)

def zscore_last(series: pd.Series, lookback: int = 252) -> Optional[float]:
    s = series.dropna().tail(lookback)
    if len(s) < max(40, lookback // 5):
        return None
    mu, sd = s.mean(), s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return None
    return float((s.iloc[-1] - mu) / sd)

def _to_scalar(x):
    """Turn pandas/NumPy things (incl. len-1) into plain Python scalars; keep None."""
    if x is None:
        return None
    import numpy as _np
    import pandas as _pd
    if isinstance(x, _pd.Series):
        if len(x) == 0:
            return None
        x = x.iloc[-1]
    if isinstance(x, _pd.DataFrame):
        if x.shape[0] == 0 or x.shape[1] == 0:
            return None
        x = x.iloc[-1, 0]
    if isinstance(x, _np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            return x.reshape(()).item()
        return None
    if isinstance(x, _np.generic):
        return x.item()
    return x

def _as_float(x) -> Optional[float]:
    """Scalarize then float-cast; return None if not finite."""
    x = _to_scalar(x)
    try:
        f = float(x)
    except Exception:
        return None
    return f if np.isfinite(f) else None

def _ensure_dtindex(obj: pd.Series | pd.DataFrame, name: str = ""):
    """
    Return a copy of `obj` with a DatetimeIndex (tz-naive), sorted, de-duplicated.
    Tries 'Date'/'date' columns, else tries to parse the current index.
    Raises TypeError if it still can't get a DatetimeIndex.
    """
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        x = obj.copy()

        # If a date-like column exists, prefer it
        for col in ("Date", "date", "Datetime", "datetime", "Time", "time"):
            if isinstance(x, pd.DataFrame) and col in x.columns:
                idx = pd.to_datetime(x[col], errors="coerce")
                x = x.set_index(idx).drop(columns=[col])
                break
        else:
            # No date column: try to parse the existing index
            x.index = pd.to_datetime(x.index, errors="coerce")

        # Must now be datetime-like and not all-NaT
        if not isinstance(x.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            raise TypeError(f"{name or 'object'} has non-datetime index after coercion: {type(x.index).__name__}")
        if getattr(x.index, "tz", None) is not None:
            try:
                x.index = x.index.tz_convert(None)
            except Exception:
                x.index = x.index.tz_localize(None)

        # drop NaT, sort, dedupe
        if isinstance(x.index, pd.DatetimeIndex):
            mask = ~x.index.isna()
            x = x.loc[mask]
        x = x.sort_index()
        x = x[~x.index.duplicated(keep="last")]
        return x
    return obj

def _cmp_lt(a, b) -> bool:
    a = _as_float(a); b = _as_float(b)
    return (a is not None and b is not None and a < b)

def _cmp_ge(a, b) -> bool:
    a = _as_float(a); b = _as_float(b)
    return (a is not None and b is not None and a >= b)

def _mom(s: Optional[pd.Series], w: int) -> Optional[float]:
    if s is None:
        return None
    s = s.dropna()
    if len(s) < w + 2:
        return None
    r = _pct_change_safe(s, w).iloc[-1]
    try:
        r = float(r)
    except Exception:
        return None
    return r if np.isfinite(r) else None

def _maybe_invert(s: pd.Series, name: str) -> pd.Series:
    if name in _PAIR_INVERT and _PAIR_INVERT[name]:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / s
        return inv.replace([np.inf, -np.inf], np.nan)
    return s

def _last(x):
    import pandas as pd, numpy as np
    if isinstance(x, pd.Series):
        x = x.dropna()
        return x.iloc[-1] if not x.empty else np.nan
    return x

def _last_bool(x, fallback=False):
    """Safe truth test for Series/bools: use last value if Series."""
    if isinstance(x, pd.Series):
        if x.empty:
            return bool(fallback)
        return bool(x.iloc[-1])
    return bool(x)

def _scalarize(obj):
    import numpy as np, pandas as pd
    if isinstance(obj, dict):
        return {k: _scalarize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _scalarize(v) for v in obj ]
    if isinstance(obj, pd.Series):
        xx = obj.dropna()
        return xx.iloc[-1].item() if not xx.empty and hasattr(xx.iloc[-1], "item") else (xx.iloc[-1] if not xx.empty else None)
    if isinstance(obj, np.generic):
        try: return obj.item()
        except Exception: return float(obj)
    return obj

def _first_series(d: dict, *keys) -> Optional[pd.Series]:
    """Return the first non-empty pandas Series in d for the given keys."""
    for k in keys:
        s = d.get(k)
        if isinstance(s, pd.Series) and not s.empty:
            return s
    return None

# --------------------------- Scoring helpers ---------------------------------
def _nanmean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return (float(np.mean(xs)) if xs else None)

def _clip01(x: float | None) -> float | None:
    if x is None or not np.isfinite(x): return None
    return float(np.clip(x, 0.0, 1.0))

def _score_trend_tanh_to_100(tanh_val: float | None) -> float | None:
    """Map tanh trend in [-1,1] to [0,100]."""
    if tanh_val is None or not np.isfinite(tanh_val): return None
    return float(np.clip(50.0 * (tanh_val + 1.0), 0.0, 100.0))

def _score_bool(flag: bool | None, t: float = 65.0, f: float = 35.0, none: float = 50.0) -> float:
    if flag is None:
        return none
    try:
        return t if bool(flag) else f
    except Exception:
        return none

def _score_pctile_good_low(pct: float | None) -> float | None:
    """Percentile where 'lower is better' → map to [0,100] with 100 = best."""
    if pct is None or not np.isfinite(pct): return None
    return float(np.clip(100.0 - pct, 0.0, 100.0))

def _score_pctile_good_high(pct: float | None) -> float | None:
    """Percentile where 'higher is better' → map to [0,100] with 100 = best."""
    if pct is None or not np.isfinite(pct): return None
    return float(np.clip(pct, 0.0, 100.0))

def _score_momentum(m: float | None, k: float = 8.0) -> float | None:
    """
    Momentum (e.g., 4w/13w log/arith return) → squashed with tanh then [0,100].
    k controls sensitivity (higher = sharper).
    """
    if m is None or not np.isfinite(m): return None
    return float(np.clip(50.0 + 50.0 * np.tanh(k * m), 0.0, 100.0))

def _weighted_avg(pairs: list[tuple[float | None, float]]) -> float | None:
    num, den = 0.0, 0.0
    for v, w in pairs:
        if v is None or not np.isfinite(v) or w <= 0: continue
        num += v * w; den += w
    return float(num / den) if den > 0 else None

# --------------------------- Series Helpers ---------------------------------
def _nanmean(vals: list[float | None]) -> float | None:
    xs: list[float] = []
    for v in vals:
        fv = _as_float(v)  # <- robust scalarization
        if fv is not None and np.isfinite(fv):
            xs.append(fv)
    return float(np.mean(xs)) if xs else None

def _score_trend_tanh_to_100(tanh_val: float | None) -> float | None:
    tv = _as_float(tanh_val)
    if tv is None or not np.isfinite(tv):
        return None
    return float(np.clip(50.0 * (tv + 1.0), 0.0, 100.0))

def _score_pctile_good_low(pct: float | None) -> float | None:
    p = _as_float(pct)
    if p is None or not np.isfinite(p):
        return None
    return float(np.clip(100.0 - p, 0.0, 100.0))

def _score_pctile_good_high(pct: float | None) -> float | None:
    p = _as_float(pct)
    if p is None or not np.isfinite(p):
        return None
    return float(np.clip(p, 0.0, 100.0))

def _score_momentum(m: float | None, k: float = 8.0) -> float | None:
    mv = _as_float(m)
    if mv is None or not np.isfinite(mv):
        return None
    return float(np.clip(50.0 + 50.0 * np.tanh(k * mv), 0.0, 100.0))

def _weighted_avg(pairs: list[tuple[float | None, float]]) -> float | None:
    num, den = 0.0, 0.0
    for v, w in pairs:
        fv = _as_float(v)
        if fv is None or not np.isfinite(fv) or w <= 0:
            continue
        num += fv * w
        den += w
    return float(num / den) if den > 0 else None

# --------------------------- Fit Model from Sampling --------------------------------------
def _maxdd_next(px: pd.Series, horizon_days: int = 20) -> pd.Series:
    """Vectorized next-window Max Drawdown in %, aligned to current date (no look-ahead leak at t)."""
    # future rolling max and min from t+1..t+h
    fwd = px.shift(-1).rolling(horizon_days, min_periods=2)
    fmax = fwd.max()
    fmin = fwd.min()
    dd = (fmin / fmax - 1.0) * 100.0
    return dd

def _label_risk_event(px: pd.Series, horizon_days=20, dd_threshold=-5.0) -> pd.Series:
    """1 if next-20d MaxDD <= -5%, else 0."""
    dd = _maxdd_next(px, horizon_days=horizon_days)
    return (dd <= (dd_threshold*100 if abs(dd_threshold)>1 else dd_threshold)).astype(float)

def fit_regime_model_from_history(
    cfg: MarketRegimenConfig,
    fetch_fn: FetchFn,
    start: pd.Timestamp,
    end: pd.Timestamp,
    learner: RegimeLearnerConfig | None = None,
) -> Optional[RegimeModel]:
    """
    Replays your own pipeline weekly to build {features -> label} without look-ahead,
    then fits a calibrated ridge logistic regression. Returns a RegimeModel.
    """
    if not _SKLEARN_OK:
        return None
    learner = learner or RegimeLearnerConfig()

    # 1) Collect data (weekly) and compute category scores without look-ahead
    #    We sample on Fridays to keep runtime reasonable.
    idx = pd.date_range(start, end, freq="W-FRI")
    rows = []
    last_prices = {}
    last_extras = {}
    for dt in idx:
        try:
            # fetch up to dt (inclusive)
            res = build_market_regime_section(cfg, fetch_fn, start, dt, headline_mode=HeadlineMode("user"))
            # Just store what we need (category scores + a few primitives)
            cats = res.get("score_blocks") or {}
            row = {k: float(v) if v is not None else np.nan for k, v in cats.items()}
            # small set of primitives
            for k in ("vix_pctile_1y","breadth_pctile_1y","credit_above_200dma","dxy_trend","uup_trend"):
                val = res.get(k)
                row[k] = float(val) if isinstance(val,(int,float,np.floating)) else np.nan
            # spy close for labels
            row["_spy_close"] = res.get("SPY_close") if "SPY_close" in res else np.nan
            rows.append((dt, row))
        except Exception:
            continue
    if not rows:
        return None
    Xdf = pd.DataFrame([r for _, r in rows], index=[t for t, _ in rows]).sort_index()

    # 2) Get SPY closes for labels (use your fetcher to avoid data mismatch)
    #    If SPY_close wasn't carried, refetch here:
    if "_spy_close" not in Xdf.columns or Xdf["_spy_close"].isna().all():
        spyd = _cached_close_series("SPY", start, end, fetch_fn)
        if spyd is None: 
            return None
        Xdf["_spy_close"] = spyd.reindex(Xdf.index, method="ffill")
    spy = Xdf["_spy_close"].dropna()
    y = _label_risk_event(spy, horizon_days=learner.horizon_days, dd_threshold=learner.dd_threshold)
    y = y.reindex(Xdf.index).dropna()
    Xdf = Xdf.loc[y.index].drop(columns=["_spy_close"])

    # 3) Clean features: fill NaNs (neutral 50 for category scores; 0 for primitives)
    feats = []
    feat_names: List[str] = []
    for name in learner.use_features:
        if name in Xdf.columns:
            col = Xdf[name].copy()
        elif name in ("equities","breadth","vol","rates_credit","fx","intl","commodities","reits","crypto","internals"):
            # missing category -> neutral
            col = pd.Series(50.0, index=Xdf.index)
        else:
            col = pd.Series(0.0, index=Xdf.index)
        feats.append(col)
        feat_names.append(name)
    X = pd.concat(feats, axis=1)
    # NaN policy
    for c in X.columns:
        if c in ("equities","breadth","vol","rates_credit","fx","intl","commodities","reits","crypto","internals"):
            X[c] = X[c].fillna(50.0)
        else:
            X[c] = X[c].fillna(0.0)

    # 4) Fit ridge logistic + probability calibration
    base = LogisticRegression(penalty="l2", C=learner.ridge_C, max_iter=2000)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=learner.cv)
    cal.fit(X.values, y.values)

    # quick holdout-like metrics via CV predictions
    try:
        # Not perfect holdout, but useful: in-sample prob vs outcome
        p = cal.predict_proba(X.values)[:,1]
        metrics = {
            "brier": float(brier_score_loss(y.values, p)),
            "auc":   float(roc_auc_score(y.values, p)),
        }
    except Exception:
        metrics = None

    return RegimeModel(model=cal, feature_names=feat_names,
                       horizon_days=learner.horizon_days,
                       dd_threshold=learner.dd_threshold, metrics=metrics)

# --------------------------- Pure core --------------------------------------
def trend_scores_for(etf_map: dict[str, str], prices: pd.DataFrame, lookback_wks: int = 26) -> dict[str, float | None]:
    """
    Given a {name: ticker} map and a DataFrame of prices (columns=tickers),
    return {name: last_trend or None}.
    """
    out: dict[str, float | None] = {}
    cols_upper = {c.upper(): c for c in prices.columns}
    for name, sym in etf_map.items():
        key = sym.upper()
        if key in cols_upper:
            s = prices[cols_upper[key]]
            try:
                out[name] = float(sector_trend_score(s, lookback_wks).iloc[-1])
            except Exception:
                out[name] = None
        else:
            out[name] = None
    return out

def compute_currency_strength_block(
    prices: pd.DataFrame,
    lookback_wks: int = 26,
) -> Dict[str, Optional[float]]:
    """
    Returns end-of-period trend scores for a set of FX proxies (ETFs and/or spot pairs).
    Keys like 'usd_strength', 'eur_strength', 'cad_strength', etc., plus a broad 'dxy_trend'.
    """
    out: Dict[str, Optional[float]] = {}
    # DXY / UUP (direct)
    if "DXY" in prices.columns:
        try:
            out["dxy_trend"] = float(_to_scalar(sector_trend_score(prices["DXY"], lookback_wks).iloc[-1]))
        except Exception:
            out["dxy_trend"] = None
    if "UUP" in prices.columns:
        try:
            out["uup_trend"] = float(_to_scalar(sector_trend_score(prices["UUP"], lookback_wks).iloc[-1]))
        except Exception:
            out["uup_trend"] = None

    # Individual currencies via ETFs (direction already “local strength”)
    etf_map = {
        "eur_strength": "FXE",
        "gbp_strength": "FXB",
        "jpy_strength": "FXY",
        "aud_strength": "FXA",
        "cad_strength": "FXC",
        "emfx_strength": "CEW",
    }
    for k, sym in etf_map.items():
        if sym in prices.columns:
            try:
                out[k] = float(_to_scalar(sector_trend_score(prices[sym], lookback_wks).iloc[-1]))
            except Exception:
                out[k] = None

    # Spot pairs (invert where needed to transform to local strength)
    spot_map = {
        "eur_strength_spot": "EURUSD=X",
        "gbp_strength_spot": "GBPUSD=X",
        "aud_strength_spot": "AUDUSD=X",
        "cad_strength_spot": "USDCAD=X",  # invert
        "jpy_strength_spot": "USDJPY=X",  # invert
        "chf_strength_spot": "USDCHF=X",  # invert
    }
    for k, pair in spot_map.items():
        if pair in prices.columns:
            try:
                s = _maybe_invert(prices[pair], pair)
                out[k] = float(_to_scalar(sector_trend_score(s, lookback_wks).iloc[-1]))
            except Exception:
                out[k] = None

    # Optional composite (USD broadness vs rest): negative average of other strengths
    comps = [out.get("eur_strength"), out.get("gbp_strength"), out.get("jpy_strength"),
             out.get("aud_strength"), out.get("cad_strength"), out.get("emfx_strength")]
    comps = [c for c in comps if isinstance(c, (int, float))]
    out["usd_vs_g10_composite"] = float(-np.mean(comps)) if comps else None
    return out

def compute_reit_trends_block(prices: pd.DataFrame, lookback_wks: int = 26) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for name in [c for c in prices.columns if c.upper() in prices.columns]:  # no-op, keep style
        pass
    names = [
        "VNQ", "IYR", "REET", "VNQI", "IFGL", "IFEU", "IFAS", "XRE.TO"
    ]
    for sym in names:
        if sym in prices.columns:
            try:
                out[f"{sym.lower()}_trend"] = float(_to_scalar(sector_trend_score(prices[sym], lookback_wks).iloc[-1]))
            except Exception:
                out[f"{sym.lower()}_trend"] = None
    # A simple “RE over Equities” tilt signal: global REITs vs ACWI if both exist
    if "REET" in prices.columns and "ACWI" in prices.columns:
        r = ratio(prices["REET"], prices["ACWI"])
        if len(r) >= 205:
            try:
                out["reit_vs_equity_above_200dma"] = bool(_as_float(r.iloc[-1]) > _as_float(ma(r, 200).iloc[-1]))
            except Exception:
                out["reit_vs_equity_above_200dma"] = None
    return out

def compute_intl_expanded_trends_block(prices: pd.DataFrame, lookback_wks: int = 26) -> Dict[str, Optional[float]]:
    """
    Trend scores for the expanded international list (Europe & Asia adds).
    """
    out: Dict[str, Optional[float]] = {}
    for sym in ["EZU","IEUR","FEZ","EWU","EWL","EWA","AAXJ","EWY","EWT","INDA","EWS","EWH"]:
        if sym in prices.columns:
            try:
                out[f"{sym.lower()}_trend"] = float(_to_scalar(sector_trend_score(prices[sym], lookback_wks).iloc[-1]))
            except Exception:
                out[f"{sym.lower()}_trend"] = None
    return out

def compute_crypto_signals_block(
    prices: pd.DataFrame,
    lookback_wks: int = 26,
) -> Dict[str, Optional[object]]:
    """
    Uses BTC/ETH spot-ETF proxies to approximate trend/breadth/correlation signals.
    You can pass either IBIT/FBTC/BITO for BTC and EETH for ETH.
    """
    out: Dict[str, Optional[object]] = {}

    # Pick BTC and ETH proxies that exist in 'prices'
    btc = None
    for cand in ["IBIT", "FBTC", "BITO"]:
        if cand in prices.columns:
            btc = prices[cand]; out["btc_proxy"] = cand; break
    eth = prices["EETH"] if "EETH" in prices.columns else (prices["ETH-USD"] if "ETH-USD" in prices.columns else None)

    # 200dma status and weekly momentum
    def _ma_status(s: Optional[pd.Series], n=200):
        if s is None or len(s) < n+5: return None
        try:
            return bool(_as_float(s.iloc[-1]) > _as_float(ma(s, n).iloc[-1]))
        except Exception:
            return None

    if btc is not None:
        out["btc_above_200dma"] = _ma_status(btc, 200)
        out["btc_trend"] = float(_to_scalar(sector_trend_score(btc, lookback_wks).iloc[-1])) if len(btc) else None
    if eth is not None and len(eth):
        out["eth_above_200dma"] = _ma_status(eth, 200)
        out["eth_trend"] = float(_to_scalar(sector_trend_score(eth, lookback_wks).iloc[-1])) if len(eth) else None

    # Correlation with QQQ/SPY (rolling 90 trading days)
    def _roll_corr(a: pd.Series, b: pd.Series, win=90) -> Optional[float]:
        if a is None or b is None: return None
        a, b = a.align(b, join="inner")
        if len(a) < win+5: return None
        return float(a.pct_change().rolling(win).corr(b.pct_change()).iloc[-1])

    for idx in ["QQQ", "SPY"]:
        if btc is not None and idx in prices.columns:
            out[f"btc_corr_{idx.lower()}_90d"] = _roll_corr(btc, prices[idx], 90)
        if eth is not None and idx in prices.columns:
            out[f"eth_corr_{idx.lower()}_90d"] = _roll_corr(eth, prices[idx], 90)

    # Risk state (your tiered model – simplified, using what we have locally)
    vix = prices["^VIX"] if "^VIX" in prices.columns else (prices["VIXY"] if "VIXY" in prices.columns else None)
    vix_last = float(_to_scalar(vix.iloc[-1])) if isinstance(vix, pd.Series) and len(vix) else None
    breadth = None
    if "RSP" in prices.columns and "SPY" in prices.columns:
        br = ratio(prices["RSP"], prices["SPY"])
        if len(br) >= 205:
            try:
                breadth = bool(_as_float(br.iloc[-1]) > _as_float(ma(br, 200).iloc[-1]))
            except Exception:
                breadth = None

    def _state():
        btc_on = out.get("btc_above_200dma") is True
        vix_ok = (vix_last is not None and vix_last < 20.0)
        breadth_ok = (breadth is True)

        if btc_on and breadth_ok and vix_ok:
            return "Risk-On"
        if (out.get("btc_above_200dma") is False) and (vix_last is not None and vix_last > 25.0):
            return "Risk-Off"
        return "Neutral"

    out["crypto_risk_state"] = _state()
    out["crypto_action_hint"] = (
        "Increase equity exposure / growth tilt" if out["crypto_risk_state"] == "Risk-On" else
        "Trim equity beta, rotate defensive, raise cash/duration" if out["crypto_risk_state"] == "Risk-Off" else
        "Maintain, drip/reinvest, scale gradually"
    )

    # Placeholders for optional extras you may pass in via `extras` later:
    # - btc_dominance, fear_greed, funding_rates, etc.
    return out

def compute_market_regime_from_prices(
    prices: pd.DataFrame,
    vix: pd.Series | None,
    *,
    lookback_wks: int = 26,
    risk_weights: Optional[Dict[str, float]] = None,
    extras: Dict[str, pd.Series] | None = None,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    extras = extras or {}
    prices = prices.copy()
    prices = _ensure_dtindex(prices, name="prices")
    prices.columns = [str(c).upper() for c in prices.columns]

    if "SPY" in prices.columns:
        try:
            out["SPY_close"] = float(_to_scalar(prices["SPY"].iloc[-1]))
        except Exception:
            pass

    # Also coerce any extras/vix passed in
    if isinstance(vix, pd.Series) and not vix.empty:
        vix = _ensure_dtindex(vix, name="vix")
    for k, s in list(extras.items()):
        if isinstance(s, pd.Series) and not s.empty:
            extras[k] = _ensure_dtindex(s, name=f"extras[{k}]")

    # --- Trend scores (weekly tanh-scaled) for core US
    for c in ["SPY", "QQQ", "DIA", "IWM", "RSP", "VTI"]:
        out[f"{c.lower()}_trend"] = None
        if c in prices:
            try:
                tr = sector_trend_score(prices[c], lookback_wks)
                tr_last = _to_scalar(tr.iloc[-1] if hasattr(tr, "iloc") else tr)
                out[f"{c.lower()}_trend"] = float(tr_last) if tr_last is not None else None
            except Exception:
                out[f"{c.lower()}_trend"] = None

    # --- Momentum (weekly) for SPY
    wk = prices.resample("W-FRI").last()
    spy = wk["SPY"] if "SPY" in wk else None
    out["spy_mom_4w"]  = _mom(spy, 4)
    out["spy_mom_13w"] = _mom(spy, 13)
    out["spy_mom_26w"] = _mom(spy, 26)

    # --- Drawdown
    out["spy_dd_52w_pct"] = _drawdown_from_high(prices["SPY"], 252) if "SPY" in prices else None

    # --- VIX metrics
    if isinstance(vix, pd.Series) and not vix.empty:
        vix_last = _to_scalar(vix.iloc[-1])
        out["vix"] = float(vix_last) if vix_last is not None else None
        out["vix_pctile_1y"] = percentile_last(vix, 252)
    else:
        out["vix"] = None
        out["vix_pctile_1y"] = None
    vix_pct = out["vix_pctile_1y"]

    # --- VIX term structure (backwardation = stress)
    v9 = _first_series(extras, "VIX9D")
    v3 = _first_series(extras, "VIX3M", "VXV")
    if v9 is not None and v3 is not None:
        try:
            ts = (_as_float(v9.iloc[-1]) or 0.0) - (_as_float(v3.iloc[-1]) or 0.0)
            out["vix_term_spread"] = float(ts)
            vix_last_scalar = _as_float(out.get("vix"))
            out["vix_term_stress"] = bool(ts > 0.0)  # or include a VIX filter if you want
        except Exception:
            out["vix_term_spread"] = None
            out["vix_term_stress"] = None

    # --- Rates & Credit derived signals
    # Curve stress: SHY/TLT
    if "SHY" in prices and "TLT" in prices:
        cv = ratio(prices["SHY"], prices["TLT"])
        out["curve_ratio"] = float(_to_scalar(cv.iloc[-1])) if len(cv) else None
        out["curve_trend"] = None
        if len(cv) > 30:
            try:
                ctr = sector_trend_score(cv, lookback_wks)
                out["curve_trend"] = float(_to_scalar(ctr.iloc[-1] if hasattr(ctr, "iloc") else ctr))
            except Exception:
                pass

    # Breakeven inflation proxy: TIP/IEF
    if "TIP" in prices and "IEF" in prices:
        be = ratio(prices["TIP"], prices["IEF"])
        out["breakeven_ratio"] = float(_to_scalar(be.iloc[-1])) if len(be) else None
        out["breakeven_trend"] = None
        if len(be) > 30:
            try:
                btr = sector_trend_score(be, lookback_wks)
                out["breakeven_trend"] = float(_to_scalar(btr.iloc[-1] if hasattr(btr, "iloc") else btr))
            except Exception:
                pass

    # Credit appetite: HYG/IEF
    if "HYG" in prices and "IEF" in prices:
        cr = ratio(prices["HYG"], prices["IEF"])
        out["credit_ratio"] = float(_to_scalar(cr.iloc[-1])) if len(cr) else None
        if len(cr) >= 205:
            cr_ma = ma(cr, 200)
            try:
                out["credit_above_200dma"] = bool(float(_to_scalar(cr.iloc[-1])) > float(_to_scalar(cr_ma.iloc[-1])))
            except Exception:
                out["credit_above_200dma"] = None
        else:
            out["credit_above_200dma"] = None

    # --- Breadth proxy: RSP/SPY
    if "RSP" in prices and "SPY" in prices:
        br = ratio(prices["RSP"], prices["SPY"])
        out["breadth_ratio"] = float(_to_scalar(br.iloc[-1])) if len(br) else None
        if len(br) >= 205:
            br_ma = ma(br, 200)
            try:
                out["breadth_above_200dma"] = bool(float(_to_scalar(br.iloc[-1])) > float(_to_scalar(br_ma.iloc[-1])))
            except Exception:
                out["breadth_above_200dma"] = None
        else:
            out["breadth_above_200dma"] = None
        out["breadth_pctile_1y"] = percentile_last(br, 252)

    # --- Dollar trend (UUP or DXY)
    if "UUP" in prices:
        try:
            tr = sector_trend_score(prices["UUP"], lookback_wks)
            out["uup_trend"] = float(_to_scalar(tr.iloc[-1] if hasattr(tr, "iloc") else tr))
        except Exception:
            out["uup_trend"] = None
    if "DXY" in prices:
        try:
            tr = sector_trend_score(prices["DXY"], lookback_wks)
            out["dxy_trend"] = float(_to_scalar(tr.iloc[-1] if hasattr(tr, "iloc") else tr))
        except Exception:
            out["dxy_trend"] = None

    # --- Commodities & real assets trend snapshots
    for sym in ["XLE", "CPER", "DBC", "GLD", "SLV", "DBB", "VNQ"]:
        if sym in prices:
            try:
                tr = sector_trend_score(prices[sym], lookback_wks)
                out[f"{sym.lower()}_trend"] = float(_to_scalar(tr.iloc[-1] if hasattr(tr, "iloc") else tr))
            except Exception:
                out[f"{sym.lower()}_trend"] = None

    # --- Realized vol (SPY)
    if "SPY" in prices:
        spy_retn = prices["SPY"].pct_change().dropna()
        w = 20
        if len(spy_retn) >= w + 5:
            out["spy_realized_vol_20d"] = float(spy_retn.tail(w).std(ddof=0) * np.sqrt(252))

    # --- Internals
    for k in ["CPC", "CPCE", "SKEW", "MOVE"]:
        s = extras.get(k)
        if s is not None and len(s):
            out[k.lower()] = float(_to_scalar(s.iloc[-1]))
            out[f"{k.lower()}_pctile_1y"] = percentile_last(s, 252)

    # --- Macro liquidity proxies (optional)
    for k in ["WALCL", "RRPONTSYD", "TGA"]:
        s = extras.get(k)
        if s is not None and len(s):
            last = float(_to_scalar(s.iloc[-1]))
            out[k.lower()] = last
            out[f"{k.lower()}_z_1y"] = zscore_last(s, 252)

    # --- Composite risk score (0..100)
    wts = {"vix": 0.50, "dd": 0.30, "mom": 0.20}
    if risk_weights:
        wts.update(risk_weights)

    comp_vix = float(np.clip(vix_pct if vix_pct is not None else 0.0, 0.0, 100.0))
    dd = out.get("spy_dd_52w_pct")
    comp_dd = float(np.clip((-(dd if dd is not None else 0.0)) * (100.0 / 30.0), 0.0, 100.0))
    mom13 = out.get("spy_mom_13w")
    comp_mom = float(np.clip(-(mom13 if mom13 is not None else 0.0) * 500.0, 0.0, 100.0))

    risk_score = float(wts["vix"] * comp_vix + wts["dd"] * comp_dd + wts["mom"] * comp_mom)
    risk_level = ("Low" if risk_score < 30 else "Moderate" if risk_score < 60 else "High" if risk_score < 80 else "Extreme")
    out["risk_score_0_100"] = round(risk_score, 1)
    out["risk_level"] = risk_level

    # --- Biases (simple heuristics)
    spy_tr = _as_float(out.get("spy_trend"))
    vix_pct = out.get("vix_pctile_1y")  # number or None
    dd      = _as_float(out.get("spy_dd_52w_pct"))
    vix_last_scalar = _as_float(out.get("vix"))

    lt_bias = (
        "Risk-On" if (spy_tr is not None and spy_tr > 0.20 and (dd is None or dd > -10.0)) and (vix_pct is None or _cmp_lt(vix_pct, 40.0))
        else "Risk-Off" if (spy_tr is not None and spy_tr < -0.20) or (dd is not None and dd <= -15.0) or (vix_pct is not None and _cmp_ge(vix_pct, 70.0))
        else "Neutral"
    )
    m4 = _as_float(out.get("spy_mom_4w")) or 0.0
    st_bias = (
        "Risk-On" if (m4 > 0.0) and (vix_last_scalar is None or vix_last_scalar < 20.0)
        else "Risk-Off" if (m4 < 0.0) and (vix_last_scalar is not None and vix_last_scalar >= 20.0)
        else "Neutral"
    )
    out["long_term_bias"] = lt_bias
    out["short_term_bias"] = st_bias

    # --- Risk-on composite from pairs (scalar-safe)
    ro_components: List[float] = []
    def _pair_signal(a: str, b: str):
        if a in prices and b in prices:
            r = ratio(prices[a], prices[b])
            if len(r) >= 205:
                r_last = _as_float(r.iloc[-1])
                ma_last = _as_float(ma(r, 200).iloc[-1])
                if r_last is None or ma_last is None:
                    return None
                above = (r_last > ma_last)
                m13 = _as_float(_mom(r.resample("W-FRI").last(), 13)) or 0.0
                return 65.0 if above else 35.0 + (15.0 * float(np.sign(m13)))
        return None

    for (a, b) in [("XLY", "XLP"), ("RSP", "SPY"), ("IWM", "SPY"), ("HYG", "IEF")]:
        s = _pair_signal(a, b)
        if s is not None:
            ro_components.append(float(s))
    if ro_components:
        out["risk_on_score_0_100"] = float(np.clip(np.mean(ro_components), 0.0, 100.0))

    # --- Inflation/Growth mix (-100..+100)
    mix_parts: List[float] = []
    for sym, sign, key in [("XLE", +1.0, "xle_trend"),
                           ("CPER", +1.0, "cper_trend"),
                           ("TLT", -1.0, "tlt_trend"),
                           ("GLD", +1.0, "gld_trend")]:
        if sym in prices:
            try:
                tr = sector_trend_score(prices[sym], lookback_wks)
                val = _to_scalar(tr.iloc[-1] if hasattr(tr, "iloc") else tr)
                if val is not None:
                    if sym == "GLD":
                        be_tr = out.get("breakeven_trend")
                        weight = 1.0 if (be_tr is None or (isinstance(be_tr, (int, float)) and be_tr > 0)) else 0.3
                        mix_parts.append(float(val) * 100.0 * weight)
                    else:
                        mix_parts.append(float(val) * 100.0 * sign)
                out[key] = float(val) if val is not None else None
            except Exception:
                out[key] = None
    if mix_parts:
        out["inflation_growth_mix_-100_100"] = float(np.clip(np.mean(mix_parts), -100.0, 100.0))
    
    # --- FX strength block
    try:
        fx_block = compute_currency_strength_block(prices, lookback_wks)
        out.update({f"fx_{k}": v for k, v in fx_block.items()})
    except Exception:
        pass

    # --- Expanded INTL trends
    try:
        intl_block = compute_intl_expanded_trends_block(prices, lookback_wks)
        out.update(intl_block)
    except Exception:
        pass

    # --- REITs trends & tilt vs equities
    try:
        reit_block = compute_reit_trends_block(prices, lookback_wks)
        out.update(reit_block)
    except Exception:
        pass

    # --- Crypto playbook signals
    try:
        crypto_block = compute_crypto_signals_block(prices, lookback_wks)
        out.update({f"crypto_{k}": v for k, v in crypto_block.items()})
    except Exception:
        pass

    # --- Final normalization: coerce Series/arrays to scalars
    for k, v in list(out.items()):
        if isinstance(v, (pd.Series, np.ndarray, np.generic)):
            v = _to_scalar(v)
        if isinstance(v, (bool, type(None))):
            out[k] = v
            continue
        try:
            out[k] = float(v) if isinstance(v, (int, float)) else float(v)
        except Exception:
            out[k] = v

    # DEBUG: fail fast if anything non-scalar slipped through
    for k, v in out.items():
        if isinstance(v, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(f"Non-scalar in market regime output: {k} -> {type(v)}")

    # --- Category scores + Headline indicator
    try:
        # today
        _cats_today = compute_category_scores(out)
        # try to compute "yesterday" by truncating one row from all series (if possible)
        prev_out = None
        if isinstance(prices, pd.DataFrame) and len(prices) >= 2:
            # shallow truncs
            prices_prev = prices.iloc[:-1, :]
            vix_prev = vix.iloc[:-1] if isinstance(vix, pd.Series) and len(vix) >= 2 else vix
            extras_prev = {}
            for k, s in extras.items():
                if isinstance(s, pd.Series) and len(s) >= 2:
                    extras_prev[k] = s.iloc[:-1]
            prev_out = compute_market_regime_from_prices(
                prices=prices_prev, vix=vix_prev,
                lookback_wks=lookback_wks, risk_weights=risk_weights, extras=extras_prev
            )
            _cats_prev = compute_category_scores(prev_out)
            prev_headline_raw = _as_float(prev_out.get("headline_score_0_100") or prev_out.get("headline_score_0_100_final"))
        else:
            _cats_prev, prev_headline_raw = None, None

        # feed raw primitives too so the learner can use them
        raw_feats = {k: v for k, v in out.items() if isinstance(v, (int, float, np.floating))}

        # Select headline mode from caller context if present, else default user
        _mode = raw_feats.get("_headline_mode_obj")  # optional injection by wrapper
        if not isinstance(_mode, HeadlineMode):
            _mode = HeadlineMode("user", 0.7)

        _regime_model = raw_feats.get("_regime_model_obj")  # optional injection
        if not isinstance(_regime_model, RegimeModel):
            _regime_model = None

        _combo = combine_category_scores_plus(
            _cats_today,
            None,  # use internal defaults; UI can still pass explicit weights upstream if needed
            prev_cat_scores=_cats_prev,
            prev_headline=prev_headline_raw,
            mode=_mode,
            regime_model=_regime_model,
            raw_features=raw_feats,
        )

        # store compact & detailed
        out["score_blocks"] = {k: (v.get("score")) for k, v in _cats_today.items()}
        out["score_blocks_details"] = _cats_today
        out.update(_combo)

        # keep old key for backwards-compat with existing UI
        if "headline_score_0_100" not in out:
            out["headline_score_0_100"] = _combo["headline_score_0_100_final"]

    except Exception as ex:
        # Fail-safe: keep regimen usable even if scoring hits an edge case
        out["headline_score_0_100"] = out.get("headline_score_0_100", 50.0)
        out["headline_regime"] = out.get("headline_regime", "Neutral")

    return out
# --------------------------- Fetch & build wrapper ---------------------------

# Your fetch function signature: fetch_fn(symbol: str, start, end) -> DataFrame
FetchFn = Callable[[str, pd.Timestamp, pd.Timestamp], Optional[pd.DataFrame]]

@lru_cache(maxsize=256)
def _cached_close_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp, fetch_fn: FetchFn) -> Optional[pd.Series]:
    df = fetch_fn(symbol, start, end)
    if df is None or df.empty:
        return None

    # If Close is missing, bail
    if "Close" not in df.columns:
        return None

    # Normalize index to datetime (handles cases where Date is a column)
    df = _ensure_dtindex(df, name=f"{symbol}_df")

    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = symbol

    # Clip to [start, end] just in case
    s = s.loc[(s.index >= start) & (s.index <= end)]
    return s

def _collect_series(symbol_map: Dict[str, str], fetch_fn: FetchFn, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for name, sym in symbol_map.items():
        try:
            s = _cached_close_series(sym, pd.to_datetime(start), pd.to_datetime(end), fetch_fn)
            if s is not None and not s.empty:
                s = _ensure_dtindex(s, name=sym)  # <- enforce again (cheap, cached anyway)
                out[name.upper()] = s
        except Exception:
            continue
    return out

def build_market_regime_section(
    cfg: MarketRegimenConfig,
    fetch_fn: FetchFn,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    headline_mode: HeadlineMode | None = None,
    regime_model: RegimeModel | None = None,
) -> Dict[str, object]:
    """
    Fetch data per config, compute regimen, and return a dict suitable for your
    structured output under a separate `market_regime` section.

    Parameters
    ----------
    cfg : MarketRegimenConfig
    fetch_fn : callable
        Your data fetcher, e.g., `.utils.fetch_history`.
    start, end : pd.Timestamp

    Returns
    -------
    dict
        Regimen dictionary (flatten later via your existing flattener).
    """
    # 1) Collect prices (close) for enabled groups
    prices_maps: List[Dict[str, str]] = []
    if cfg.include_groups.get("core"):      prices_maps.append(cfg.core)
    if cfg.include_groups.get("rates"):     prices_maps.append(cfg.rates)
    if cfg.include_groups.get("credit"):    prices_maps.append(cfg.credit)
    if cfg.include_groups.get("intl"):      prices_maps.append(cfg.intl)
    if cfg.include_groups.get("comms"):     prices_maps.append(cfg.comms)
    if cfg.include_groups.get("fx"):        prices_maps.append(cfg.fx)
    if cfg.include_groups.get("crypto"):    prices_maps.append(cfg.crypto)

    price_series: Dict[str, pd.Series] = {}
    for smap in prices_maps:
        price_series.update(_collect_series(smap, fetch_fn, start, end))

    prices = pd.DataFrame(price_series).dropna(how="all")

    # 2) Volatility (we keep VIX as a separate series + extras for term structure)
    vix = None
    extras: Dict[str, pd.Series] = {}
    if cfg.include_groups.get("vol"):
        vol_series = _collect_series(cfg.vol, fetch_fn, start, end)
        # Priority: ^VIX; then VIXY; then VIXM
        if "VIX" in vol_series:
            vix = vol_series["VIX"]
        elif "VIXY" in vol_series:
            vix = vol_series["VIXY"]
        elif "VIXM" in vol_series:
            vix = vol_series["VIXM"]
        # Term structure extras
        for k in ["VIX9D", "VIX3M"]:
            if k in vol_series:
                extras[k] = vol_series[k]

    # 3) Internals (optional)
    if cfg.include_groups.get("internals"):
        internals_series = _collect_series(cfg.internals, fetch_fn, start, end)
        extras.update(internals_series)

    # 4) Compute regimen (pure)
    res = compute_market_regime_from_prices(
        prices=prices,
        vix=vix,
        lookback_wks=cfg.trend_lookback_wks,
        risk_weights=cfg.risk_weights,
        extras=extras,
    )
    return _scalarize(res)

# --------------------------- Category score composer -------------------------
def compute_category_scores(features: dict[str, object]) -> dict[str, dict[str, object]]:
    """
    Turn flat `features` (output from compute_market_regime_from_prices) into
    per-class scores in [0,100], with sub-part transparency.
    Returns: {category: {"score": float|None, "parts": {...}}}
    """
    f = features  # alias

    # --- Equities (trend & momentum)
    eq_parts = {
        "spy_trend": _score_trend_tanh_to_100(f.get("spy_trend")),
        "qqq_trend": _score_trend_tanh_to_100(f.get("qqq_trend")),
        "iwm_trend": _score_trend_tanh_to_100(f.get("iwm_trend")),
        "rsp_trend": _score_trend_tanh_to_100(f.get("rsp_trend")),
        "vti_trend": _score_trend_tanh_to_100(f.get("vti_trend")),
        "mom_4w":    _score_momentum(f.get("spy_mom_4w")),
        "mom_13w":   _score_momentum(f.get("spy_mom_13w")),
    }
    equities_score = _weighted_avg([
        (eq_parts["spy_trend"], 2.0),
        (eq_parts["qqq_trend"], 1.0),
        (eq_parts["iwm_trend"], 1.0),
        (eq_parts["rsp_trend"], 1.5),
        (eq_parts["vti_trend"], 1.0),
        (eq_parts["mom_4w"],   0.75),
        (eq_parts["mom_13w"],  1.25),
    ])

    # --- Breadth
    br_parts = {
        "breadth_above_200dma": _score_bool(f.get("breadth_above_200dma")),
        "breadth_pctile_1y":    _score_pctile_good_high(f.get("breadth_pctile_1y")),
    }
    breadth_score = _weighted_avg([
        (br_parts["breadth_above_200dma"], 1.0),
        (br_parts["breadth_pctile_1y"],    1.5),
    ])

    # --- Volatility (good when low; term backwardation is bad)
    vts = f.get("vix_term_stress")
    # coerce possible pandas objects to a plain bool
    if isinstance(vts, pd.Series):
        vts = bool(vts.iloc[-1])
    elif isinstance(vts, np.generic):
        vts = bool(vts)

    vol_parts = {
        "vix_low_pctile": _score_pctile_good_low(f.get("vix_pctile_1y")),
        # if stress True -> "not ok" (35), else "ok" (65). If None -> 50.
        "vix_term_ok":    _score_bool(False if vts is True else True, t=65, f=35, none=50),
    }
    vol_score = _weighted_avg([
        (vol_parts["vix_low_pctile"], 1.6),
        (vol_parts["vix_term_ok"],    0.9),
    ])

    # --- Rates & Credit
    # Interpret curve_trend (SHY/TLT) as: falling ratio (negative trend) ≈ easing LT yields → supportive.
    crv = f.get("curve_trend"); crv_as_equity_good = (-crv if isinstance(crv, (int, float)) else None)
    rates_credit_parts = {
        "credit_above_200dma": _score_bool(f.get("credit_above_200dma")),
        "breakeven_trend":     _score_trend_tanh_to_100(f.get("breakeven_trend")),
        "curve_as_eq_good":    _score_trend_tanh_to_100(crv_as_equity_good),
    }
    rates_credit_score = _weighted_avg([
        (rates_credit_parts["credit_above_200dma"], 1.2),
        (rates_credit_parts["breakeven_trend"],     0.9),
        (rates_credit_parts["curve_as_eq_good"],    0.9),
    ])

    # --- FX (USD strong = risk-off). Use DXY/UUP inverted + USD vs G10 composite if present.
    dxy = f.get("dxy_trend"); uup = f.get("uup_trend")
    fx_parts = {
        "usd_weak_via_dxy": _score_trend_tanh_to_100((-dxy) if isinstance(dxy, (int, float)) else None),
        "usd_weak_via_uup": _score_trend_tanh_to_100((-uup) if isinstance(uup, (int, float)) else None),
        "usd_vs_g10":       None,
    }
    # fx_usd_vs_g10_composite: positive = stronger USD, negative = weaker USD (by construction)
    uvsg10 = f.get("fx_usd_vs_g10_composite")
    if isinstance(uvsg10, (int, float)):
        fx_parts["usd_vs_g10"] = _score_trend_tanh_to_100(float(-np.tanh(2.0 * uvsg10)))  # invert & squash
    fx_score = _weighted_avg([
        (fx_parts["usd_weak_via_dxy"], 1.0),
        (fx_parts["usd_weak_via_uup"], 0.6),
        (fx_parts["usd_vs_g10"],       1.2),
    ])

    # --- International (Europe & Asia) – average of available trend keys
    intl_keys = ["ezu_trend","ieur_trend","fez_trend","vgk_trend","ewj_trend",
                 "aaxj_trend","ewa_trend","ewy_trend","ewt_trend","inda_trend","ews_trend","ewh_trend"]
    intl_parts = {k: _score_trend_tanh_to_100(f.get(k)) for k in intl_keys}
    intl_score  = _nanmean(list(intl_parts.values()))

    # --- Commodities & Real Assets (cyclical tilt)
    # +XLE +CPER +DBB +DBC,  -TLT (rates down = supportive; we invert sign below)
    xle = _score_trend_tanh_to_100(f.get("xle_trend"))
    cper= _score_trend_tanh_to_100(f.get("cper_trend"))
    dbb = _score_trend_tanh_to_100(f.get("dbb_trend"))
    dbc = _score_trend_tanh_to_100(f.get("dbc_trend"))
    tlt = f.get("tlt_trend"); tlt_eq_good = _score_trend_tanh_to_100((-tlt) if isinstance(tlt,(int,float)) else None)
    comm_parts = {
        "xle": xle, "cper": cper, "dbb": dbb, "dbc": dbc, "tlt_as_eq_good": tlt_eq_good,
    }
    commodities_score = _weighted_avg([
        (xle, 1.0), (cper, 1.0), (dbb, 0.8), (dbc, 0.8), (tlt_eq_good, 0.8),
    ])

    # --- REITs (plus relative tilt vs ACWI)
    re_keys = ["vnq_trend","iyr_trend","vnqi_trend","ifgl_trend","ifeu_trend","ifas_trend","xre.to_trend"]
    re_parts = {k: _score_trend_tanh_to_100(f.get(k)) for k in re_keys}
    re_parts["reit_vs_equity_ok"] = _score_bool(f.get("reit_vs_equity_above_200dma"))
    reits_score = _weighted_avg([(re_parts[k], 1.0) for k in re_keys] + [(re_parts["reit_vs_equity_ok"], 0.8)])

    # --- Crypto block (trend, 200dma status; correlations are informative but not scored here)
    btc_t = _score_trend_tanh_to_100(f.get("crypto_btc_trend"))
    eth_t = _score_trend_tanh_to_100(f.get("crypto_eth_trend"))
    btc_200 = _score_bool(f.get("crypto_btc_above_200dma"))
    eth_200 = _score_bool(f.get("crypto_eth_above_200dma"))
    crypto_state = f.get("crypto_crypto_risk_state")  # "Risk-On" | "Neutral" | "Risk-Off" | None
    state_boost = {"Risk-On": 65.0, "Neutral": 50.0, "Risk-Off": 35.0}.get(crypto_state, 50.0)
    crypto_parts = {
        "btc_trend": btc_t, "eth_trend": eth_t,
        "btc_above_200dma": btc_200, "eth_above_200dma": eth_200,
        "state_hint": state_boost,
    }
    crypto_score = _weighted_avg([
        (btc_t, 1.2), (eth_t, 0.8), (btc_200, 0.8), (eth_200, 0.4), (state_boost, 0.6),
    ])

    # --- Internals (low put/call + low MOVE is supportive)
    internals_parts = {
        "cpc_good":  _score_pctile_good_low(f.get("cpc_pctile_1y")),
        "cpce_good": _score_pctile_good_low(f.get("cpce_pctile_1y")),
        "move_good": _score_pctile_good_low(f.get("move_pctile_1y")),
        # SKEW is tricky; treat mid-high as neutral; score only if percentile very low (panic absent)
        "skew_hint": _score_pctile_good_high(f.get("skew_pctile_1y")),
    }
    internals_score = _weighted_avg([
        (internals_parts["cpc_good"],  0.9),
        (internals_parts["cpce_good"], 1.1),
        (internals_parts["move_good"], 1.1),
        (internals_parts["skew_hint"], 0.4),
    ])

    return {
        "equities":   {"score": equities_score,    "parts": eq_parts},
        "breadth":    {"score": breadth_score,     "parts": br_parts},
        "vol":        {"score": vol_score,         "parts": vol_parts},
        "rates_credit":{"score": rates_credit_score,"parts": rates_credit_parts},
        "fx":         {"score": fx_score,          "parts": fx_parts},
        "intl":       {"score": intl_score,        "parts": intl_parts},
        "commodities":{"score": commodities_score, "parts": comm_parts},
        "reits":      {"score": reits_score,       "parts": re_parts},
        "crypto":     {"score": crypto_score,      "parts": crypto_parts},
        "internals":  {"score": internals_score,   "parts": internals_parts},
    }

# --------------------------- Headline combiner --------------------------------
CATEGORY_WEIGHTS_DEFAULT: dict[str, float] = {
    # heavier on equities/breadth/vol/rates-credit; lighter on satellites
    "equities":     0.22,
    "breadth":      0.15,
    "vol":          0.18,
    "rates_credit": 0.15,
    "fx":           0.08,
    "intl":         0.07,
    "commodities":  0.05,
    "reits":        0.03,
    "crypto":       0.04,
    "internals":    0.03,
}

def _regime_label(score: float) -> str:
    # Tune to taste
    if score >= 75.0: return "Risk-On (Strong)"
    if score >= 60.0: return "Risk-On"
    if score >  40.0: return "Neutral"
    return "Risk-Off"

def combine_category_scores(
    cat_scores: dict[str, dict[str, object]],
    weights: dict[str, float] | None = None
) -> dict[str, object]:
    ws = (weights or CATEGORY_WEIGHTS_DEFAULT)
    pairs: list[tuple[float | None, float]] = []
    for k, w in ws.items():
        v = cat_scores.get(k, {}).get("score")
        pairs.append((v if isinstance(v, (int, float)) else None, float(w)))
    headline = _weighted_avg(pairs)
    headline = float(np.clip(headline if headline is not None else 50.0, 0.0, 100.0))
    return {
        "headline_score_0_100": headline,
        "headline_regime": _regime_label(headline),
        "headline_weights": ws,
    }

def combine_category_scores_plus(
    cat_scores: dict[str, dict[str, object]],
    user_weights: dict[str, float] | None,
    *,
    # optional yesterday for deltas/hysteresis
    prev_cat_scores: dict[str, dict[str, object]] | None = None,
    prev_headline: float | None = None,
    # learned-headline
    mode: HeadlineMode | None = None,
    regime_model: RegimeModel | None = None,
    raw_features: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Returns a rich dict: user_headline, learned_headline (if model), blended headline,
    contributions, deltas, agreement %, confidence, final regime with hysteresis.
    """
    mode = mode or HeadlineMode("user", 0.7)
    ws_user = (user_weights or CATEGORY_WEIGHTS_DEFAULT).copy()

    # Extract simple {cat: score} 0..100
    curr = {k: float(v.get("score")) for k, v in cat_scores.items() if isinstance(v.get("score"), (int, float))}
    prev = None
    if isinstance(prev_cat_scores, dict):
        prev = {k: float(v.get("score")) for k, v in prev_cat_scores.items() if isinstance(v.get("score"), (int, float))}

    # Apply overlap penalty, then compute user headline
    ws_eff = _apply_overlap_penalty(ws_user, curr)
    pairs = [(curr.get(k), w) for k, w in ws_eff.items()]
    user_headline = _weighted_avg(pairs)
    user_headline = float(np.clip(user_headline if user_headline is not None else 50.0, 0.0, 100.0))

    # Contributions & deltas (relative to neutral 50 baseline)
    contrib = _contribs_from_scores(curr, ws_eff)
    delta = None
    if prev is not None:
        delta = {k: contrib.get(k,0.0) - _contribs_from_scores(prev, ws_eff).get(k,0.0) for k in ws_eff.keys()}

    # Learned headline (probability of 20d drawdown); fallback if model missing
    learned_prob = None
    learned_headline = None
    if regime_model and _SKLEARN_OK:
        feats: list[float] = []
        names = regime_model.feature_names
        # priority: category scores first (same names)
        for name in names:
            if name in curr:
                feats.append(float(curr[name]))
            else:
                feats.append(float(_as_float(raw_features.get(name)) if raw_features else np.nan))
        X = np.array(feats, dtype=float).reshape(1, -1)
        # nan-safe: replace NaN with 50 (neutral) for category scores, or 0 for metrics
        for i, n in enumerate(names):
            if not np.isfinite(X[0, i]):
                X[0, i] = 50.0 if n in curr else 0.0
        try:
            p = regime_model.model.predict_proba(X)[:, 1][0]
            learned_prob = float(p)  # event = high drawdown risk
            learned_headline = float(np.clip(100.0 * (1.0 - learned_prob), 0.0, 100.0))
        except Exception:
            pass

    # Choose final headline
    if mode.mode == "learned" and learned_headline is not None:
        final_headline = learned_headline
    elif mode.mode == "blend" and learned_headline is not None:
        a = float(np.clip(mode.blend_alpha, 0.0, 1.0))
        final_headline = float(a * learned_headline + (1.0 - a) * user_headline)
    else:
        final_headline = user_headline

    # Agreement & confidence; regime with hysteresis
    agree = _agreement_percent(curr)
    conf = _confidence_from_dispersion(curr)
    regime = _regime_with_hysteresis(final_headline, prev_headline, up=5.0, down=3.0)

    # Top drivers / drags for narrative
    drivers_sorted = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    top_pos = [k for k, v in drivers_sorted if v > 0][:3]
    top_neg = [k for k, v in sorted(contrib.items(), key=lambda kv: kv[1]) if v < 0][:3]

    # Plain-English blurb
    msg = f"{regime} ({final_headline:.0f}). Pos: {', '.join(top_pos) or '—'}. Neg: {', '.join(top_neg) or '—'}."
    if learned_prob is not None:
        msg += f" 20d drawdown risk ≈ {100.0*learned_prob:.1f}%."

    return {
        # keep original outputs compatible
        "headline_score_0_100_user": user_headline,
        "headline_score_0_100_learned": learned_headline,
        "headline_score_0_100_final": final_headline,
        "headline_regime": regime,
        "headline_confidence_0_100": conf,
        "headline_agreement_pct": agree,
        "headline_weights_effective": ws_eff,

        "contributions_from_neutral": contrib,  # weight*(score-50)
        "contributions_delta_since_yday": delta,

        "headline_summary": msg,
        "learned_drawdown_prob_20d": learned_prob,
    }

if __name__ == "__main__":
    import argparse, json, traceback, datetime as _dt, logging
    from pathlib import Path

    # If you sometimes run this file directly (not via -m), help Python find siblings.
    try:
        # Try relative import fallback so this module can run as a script too.
        from . import etf_sector_analysis as _eta  # type: ignore
    except Exception:
        # Running as a plain script: allow `from etf_sector_analysis import ...`
        here = Path(__file__).resolve().parent
        parent = here.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))

    parser = argparse.ArgumentParser(description="Standalone runner for market_regimen.")
    today = _dt.date.today()
    parser.add_argument("--start", default=(today.replace(year=today.year - 1)).isoformat(),
                        help="ISO date (YYYY-MM-DD) for start (default: ~1y ago)")
    parser.add_argument("--end", default=today.isoformat(),
                        help="ISO date (YYYY-MM-DD) for end (default: today)")
    parser.add_argument("--with-crypto", action="store_true", help="Include crypto block")
    parser.add_argument("--source", choices=["utils", "yfinance", "dummy"], default="utils",
                        help="Data source for fetcher")
    parser.add_argument("--log", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("market_regimen_main")

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    # --- build a fetcher -----------------------------------------------------
    def _fetch_via_utils(symbol: str, s: pd.Timestamp, e: pd.Timestamp):
        try:
            # Try your package-relative utils first, then a top-level fallback.
            try:
                from .utils import fetch_history as _fh  # type: ignore
            except Exception:
                from utils import fetch_history as _fh    # type: ignore
            df = _fh(symbol, s, e)
            return df if df is not None and not df.empty else None
        except Exception as ex:
            log.debug("utils fetch failed for %s: %s", symbol, ex)
            return None

    def _fetch_via_yf(symbol: str, s: pd.Timestamp, e: pd.Timestamp):
        try:
            import yfinance as yf  # optional
            df = yf.download(symbol, start=s, end=e, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return None
            if "Close" not in df.columns:
                return None
            return df[["Close"]]
        except Exception as ex:
            log.debug("yfinance fetch failed for %s: %s", symbol, ex)
            return None

    def _fetch_dummy(symbol: str, s: pd.Timestamp, e: pd.Timestamp):
        idx = pd.date_range(s, e, freq="B")
        if len(idx) == 0:
            return None
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        # Simple geometric random walk
        steps = rng.normal(0.0006, 0.012, len(idx)).cumsum()
        series = pd.Series(100.0 * np.exp(steps), index=idx, name="Close")
        return pd.DataFrame(series)

    if args.source == "utils":
        fetch_fn = _fetch_via_utils
    elif args.source == "yfinance":
        fetch_fn = _fetch_via_yf
    else:
        fetch_fn = _fetch_dummy

    # --- config --------------------------------------------------------------
    cfg = MarketRegimenConfig()
    if args.with_crypto:
        cfg.include_groups["crypto"] = True

    # --- run and report ------------------------------------------------------
    try:
        res = build_market_regime_section(cfg, fetch_fn, start, end)
        # Flag any non-scalars that slipped through (helps catch array/Series leaks)
        non_scalars = {
            k: type(v).__name__
            for k, v in res.items()
            if isinstance(v, (pd.Series, pd.DataFrame, np.ndarray))
        }
        if non_scalars:
            log.warning("Non-scalar outputs detected: %s", non_scalars)

        # Print a compact, sorted JSON of the scalar outputs (easy to eyeball)
        printable = {}
        for k, v in res.items():
            if isinstance(v, (str, bool)) or _as_float(v) is not None:
                printable[k] = v
        print(json.dumps(dict(sorted(printable.items())), indent=2, default=str))

    except Exception as e:
        print("❌ Exception while computing market regime:", type(e).__name__, str(e))
        traceback.print_exc()
        # Exit non-zero to make CI or shell scripts fail loudly
        raise