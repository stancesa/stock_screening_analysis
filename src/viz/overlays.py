from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import pandas as pd
import plotly.graph_objects as go

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