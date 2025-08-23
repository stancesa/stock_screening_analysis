from __future__ import annotations
import ast, numpy as np, pandas as pd, re
from typing import Any, Iterable, Optional, Tuple

def _to_list(x: Any) -> Optional[list]:
    if x is None: return None
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, pd.Series):     return x.tolist()
    if isinstance(x, np.ndarray):    return x.ravel().tolist()
    if isinstance(x, str):
        s = x.strip()
        if not s: return None
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)): return list(v)
        except Exception:
            if "," in s: return [p.strip() for p in s.split(",")]
        return None
    try: return list(x)
    except Exception: return None

def _resolve(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cs = set(cols)
    for c in candidates:
        if c in cs: return c
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
        return (None,)*6
    n = min(len(dates), len(close), len(sma), len(open_), len(high), len(low))
    if n == 0: return (None,)*6
    x = pd.to_datetime(dates[:n], errors="coerce"); mask = x.notna()
    return (x[mask],
            np.asarray(close[:n], dtype=float)[mask],
            np.asarray(sma[:n],   dtype=float)[mask],
            np.asarray(open_[:n], dtype=float)[mask],
            np.asarray(high[:n],  dtype=float)[mask],
            np.asarray(low[:n],   dtype=float)[mask])

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _coerce_boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s): return s
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})

def _safe_cast_number(x: str):
    import numpy as np
    try: return float(x)
    except Exception: return np.nan
    
def _parse_ci_label(lbl: str) -> tuple[float,float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%?\s*$", lbl)
    if not m: return (10.0, 90.0)
    a,b = float(m.group(1)), float(m.group(2))
    a,b = min(a,b), max(a,b)
    a = max(0.0, min(49.999, a)); b = min(100.0, max(50.001, b))
    return a,b
