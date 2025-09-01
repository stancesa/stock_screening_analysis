from __future__ import annotations
import ast, numpy as np, pandas as pd, re
from typing import Any, Iterable, Optional, Tuple

import os, datetime as dt
from typing import List
import yfinance as yf

from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd

from .log import get_log

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

def _to_scalar(x):
    """Convert pandas/NumPy scalars/1-elts to plain Python scalars; leave None."""
    if x is None:
        return None
    # Pandas Series/DataFrame: take the last value if it's a 1-D series; otherwise leave None
    import numpy as _np
    import pandas as _pd
    if isinstance(x, _pd.Series):
        if len(x) == 0:
            return None
        x = x.iloc[-1]
    if isinstance(x, _pd.DataFrame):
        # not expected here, but guard anyway
        if x.shape[0] == 0 or x.shape[1] == 0:
            return None
        x = x.iloc[-1, 0]
    # NumPy scalar/array(1), coerce to Python scalar
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, _np.ndarray):
        if x.size == 0:
            return None
        if x.size == 1:
            return x.reshape(()).item()
        # multi-item arrays aren't expected; return None to avoid ambiguity
        return None
    return x

def _as_float(x) -> Optional[float]:
    x = _to_scalar(x)
    try:
        return float(x) if x is not None and np.isfinite(x) else None
    except Exception:
        return None

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

## ======== Original Utils for Scanner ==============

def read_list(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def fetch_history(ticker: str, start: dt.date, end: dt.date):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df["AdjClose"] = df["Adj Close"]
    return df

def market_risk_ok(threshold: float, enabled: bool) -> bool:
    if not enabled:
        return True

    vix = yf.download("^VIX", period="6mo", interval="1d", progress=False, auto_adjust=False)
    if vix is None or vix.empty or "Close" not in vix:
        # If we can't fetch VIX, fail-open (don't block entries)
        return True

    # Get the last close as a clean Python float (no FutureWarning)
    last_arr = vix["Close"].to_numpy()
    if last_arr.size == 0 or pd.isna(last_arr[-1]):
        return True

    latest = float(last_arr[-1].item() if hasattr(last_arr[-1], "item") else last_arr[-1])
    return latest <= float(threshold)

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

def read_merged_list(files: Union[str, Path, Iterable[Union[str, Path]]]) -> List[str]:
    """
    Read and coalesce one or many files into a single ordered list.
    Accepts a single path or an iterable of paths.
    """
    if isinstance(files, (str, Path)):
        files = [files]
    return _parse_list_lines(_iter_file_lines(files))