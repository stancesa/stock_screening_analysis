from typing import Tuple
import pandas as pd

def _scalar(x) -> float:
    # Safely convert pandas/NumPy scalars to Python float (no FutureWarnings)
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

# ========== Core indicators ==========

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bullish_macd_cross(macd_line: pd.Series, signal_line: pd.Series, lookback: int = 5) -> bool:
    diff = (macd_line - signal_line).dropna()
    if diff.empty:
        return False
    recent = [ _scalar(v) for v in diff.tail(lookback + 1).to_numpy() ]
    for i in range(1, len(recent)):
        if recent[i-1] < 0 and recent[i] > 0:
            return True
    return False

# ========== Extras useful for entries & stops ==========

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["AdjClose"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def donchian(df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    hh = df["High"].rolling(lookback).max()
    ll = df["Low"].rolling(lookback).min()
    return hh, ll

def swing_low(df: pd.DataFrame, left: int = 2, right: int = 2) -> float:
    """
    Most recent pivot low: a low with higher lows to the left and right.
    Returns NaN if not found.
    """
    lows = df["Low"]
    n = len(lows)
    if n < left + right + 1:
        return float("nan")

    # walk backward looking for the most recent pivot
    for i in range(n - 1 - right, left - 1, -1):
        window = lows.iloc[i - left : i + right + 1]
        # center of the window is at position 'left'
        center = window.iloc[left]
        center_val = _scalar(center)

        first_val = _scalar(window.iloc[0])
        last_val  = _scalar(window.iloc[-1])

        # min might return a pandas scalar; normalize
        min_val = window.min()
        min_val = _scalar(min_val)

        if center_val == min_val and first_val > center_val and last_val > center_val:
            return center_val

    return float("nan")

def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma
    return ma, upper, lower, pct_b, bandwidth
