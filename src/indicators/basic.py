import numpy as np, pandas as pd

def _sigmoid(x: float, k: float = 1.0) -> float:
    try:
        return float(1.0 / (1.0 + np.exp(-x / max(k, 1e-9))))
    except Exception:
        return 0.5
    
def _bbands_series(close: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ma, upper, lower) Bollinger Bands arrays for the given close series."""
    s = pd.Series(close, dtype="float64")
    ma = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    up = (ma + k * sd).to_numpy()
    lo = (ma - k * sd).to_numpy()
    return ma.to_numpy(), up, lo

def _bb_percent_b_and_bw(close: np.ndarray, window: int = 20, k: float = 2.0) -> tuple[float, float]:
    """
    Return Bollinger %B and BandWidth at the last bar.
    %B = (Close - Lower)/(Upper - Lower), clipped to [0,1] if bands valid.
    BandWidth = (Upper - Lower)/MA. Returns (np.nan, np.nan) if not enough history.
    """
    ma, up, lo = _bbands_series(close, window, k)
    if len(close) == 0 or not np.isfinite(up[-1]) or not np.isfinite(lo[-1]):
        return np.nan, np.nan
    denom = max(up[-1] - lo[-1], 1e-12)
    percent_b = (float(close[-1]) - lo[-1]) / denom
    if np.isfinite(ma[-1]) and ma[-1] > 0:
        bw = (up[-1] - lo[-1]) / ma[-1]
    else:
        bw = np.nan
    return float(np.clip(percent_b, 0.0, 1.0)), float(bw)

def _donch_breakout_strength(last: float, prev_hi: float, prev_lo: float, atr: float) -> tuple[float, float]:
    """
    Return Donchian breakout strengths for BUY and SELL:
      buy_strength  = max((last - prev_hi)/ATR, 0)  in [0, +)
      sell_strength = max((prev_lo - last)/ATR, 0)  in [0, +)
    These are later squashed to [0,1] with a sigmoid (so ~0.5·ATR above/below starts to count).
    If ATR is not finite, returns (1.0/0.0 based on plain breakout booleans) as a fallback.
    """
    if not np.isfinite(atr) or atr <= 0:
        return (1.0 if (np.isfinite(prev_hi) and last >= prev_hi) else 0.0,
                1.0 if (np.isfinite(prev_lo) and last <= prev_lo) else 0.0)
    return (max((last - prev_hi) / atr, 0.0) if np.isfinite(prev_hi) else 0.0,
            max((prev_lo - last) / atr, 0.0) if np.isfinite(prev_lo) else 0.0)

def _wilder_rma(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA used by RSI/ATR. Returns same-length array with NaNs for warmup."""
    a = pd.Series(arr, dtype="float64")
    rma = a.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return rma.to_numpy()

def _rsi_wilder(close: np.ndarray, period: int = 14) -> float:
    x = pd.Series(close, dtype="float64").dropna().to_numpy()
    if len(x) < period + 1:
        return float("nan")
    diff = np.diff(x)
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    avg_gain = _wilder_rma(up, period)
    avg_loss = _wilder_rma(dn, period)
    rs = np.divide(avg_gain, np.where(avg_loss == 0, np.nan, avg_loss))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi[-1])

def _rsi_series_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    x = pd.Series(close, dtype="float64").dropna().to_numpy()
    if len(x) < period + 1:
        return np.array([], dtype="float64")
    diff = np.diff(x)
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    avg_gain = pd.Series(up).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(dn).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # align back to price length by padding a nan at start
    return np.concatenate([[np.nan], rsi.to_numpy()])

def _atr_from_ohlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    h = pd.Series(high, dtype="float64").to_numpy()
    l = pd.Series(low,  dtype="float64").to_numpy()
    c = pd.Series(close, dtype="float64").to_numpy()
    n = min(len(h), len(l), len(c))
    if n < period + 1:
        return float("nan")
    h, l, c = h[-n:], l[-n:], c[-n:]
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    atr = _wilder_rma(tr, period)
    return float(atr[-1])

def _boll_lower(close: np.ndarray, window: int = 20, k: float = 2.0) -> float:
    s = pd.Series(close, dtype="float64").dropna()
    if len(s) < max(window, 5):
        return float("nan")
    ma = s.rolling(window, min_periods=window).mean().iloc[-1]
    sd = s.rolling(window, min_periods=window).std(ddof=0).iloc[-1]
    return float(ma - k * sd)

def _donchian_prev_high(high: np.ndarray, lookback: int = 20) -> float:
    h = pd.Series(high, dtype="float64").dropna().to_numpy()
    if len(h) < lookback + 1:
        return float("nan")
    # use yesterday's channel
    return float(np.max(h[-(lookback+1):-1]))

def _donchian_prev_low(low: np.ndarray, lookback: int = 20) -> float:
    l = pd.Series(low, dtype="float64").dropna().to_numpy()
    if len(l) < lookback + 1:
        return float("nan")
    return float(np.min(l[-(lookback+1):-1]))

def _slope_pct_per_day(series: np.ndarray, window: int) -> float:
    s = pd.Series(series, dtype="float64").dropna()
    if len(s) < window:
        return 0.0
    y = s.iloc[-window:].to_numpy()
    x = np.arange(len(y), dtype="float64")
    try:
        m = np.polyfit(x, y, 1)[0]
        if y[-1] <= 0 or not np.isfinite(y[-1]):
            return 0.0
        return float(m / y[-1])
    except Exception:
        return 0.0