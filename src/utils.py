
import os, datetime as dt
from typing import List
import yfinance as yf

import pandas as pd

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
