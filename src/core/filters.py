from __future__ import annotations
import numpy as np, pandas as pd
from typing import Any, Optional

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
    "is true":  lambda s, v: s.astype("boolean") == True,   # noqa: E712
    "is false": lambda s, v: s.astype("boolean") == False,  # noqa: E712
}

TRUE_TOKENS  = {"true", "1", "yes", "y", "t"}
FALSE_TOKENS = {"false", "0", "no", "n", "f"}

def _looks_bool_token(val: str) -> Optional[bool]:
    s = str(val).strip().lower()
    if s in TRUE_TOKENS:  return True
    if s in FALSE_TOKENS: return False
    return None

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

def _coerce_series_for_rule(series: pd.Series, op: str, raw_value: str) -> tuple[pd.Series, Any]:
    """
    Decide how to coerce the series/value based on the operator + provided value.
    Returns (coerced_series, coerced_value).
    """
    # text ops keep strings
    if op in {"contains","not contains"}: return series.astype("string"), raw_value
    if op in {"isna","notna"}: return series, None
    if op in {"is true","is false"}: return series.astype("boolean"), None

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
    from .utils import _safe_cast_number
    return pd.to_numeric(series, errors="coerce"), _safe_cast_number(raw_value)
