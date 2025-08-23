import numpy as np, pandas as pd
from .buy_engine import compute_buy_signal
from .sell_engine import compute_sell_signal
from core.types import BuyParams

def _row_without_point_indicators(row: pd.Series) -> pd.Series:
    """Return a copy of row with point-in-time indicators nulled, so engines recompute series-based values."""
    r = row.copy()
    for c in ("momentum__rsi", "rsi", "vol", "volume__vol", "vol_avg20", "volume__vol_avg20"):
        if c in r:
            r[c] = np.nan
    return r

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

