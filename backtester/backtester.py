
import argparse, json, math, datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from src.indicators import rsi, macd

@dataclass
class BTParams:
    universe: List[str]
    start: str
    end: str
    rsi_thresh: float = 35
    macd_lookback: int = 5
    sma_long: int = 200
    sma_dev_req: float = 0.10
    vol_avg_period: int = 20
    vol_spike_mult: float = 1.5
    stop_loss_pct: float = 0.11
    target_gain_pct: float = 0.10
    trailing_trigger_gain: float = 0.05
    trailing_pct: float = 0.07
    max_hold_days: int = 60
    initial_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    margin_ltv_cap: float = 0.30
    max_positions: int = 2
    annual_margin_rate: float = 0.055
    evaluate_weekly: bool = True

def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df["AdjClose"] = df["Adj Close"]
    return df

def compute_signals(df: pd.DataFrame, p: BTParams) -> pd.DataFrame:
    close = df["AdjClose"]
    df["RSI"] = rsi(close, 14)
    macd_line, signal_line, _ = macd(close, 12, 26, 9)
    diff = macd_line - signal_line
    df["MACD_DIFF"] = diff
    df["CROSS_UP"] = (diff.shift(1) < 0) & (diff > 0)
    df["SMA200"] = close.rolling(p.sma_long).mean()
    df["VOL_AVG"] = df["Volume"].rolling(p.vol_avg_period).mean()
    return df

def entry_ok(df: pd.DataFrame, idx: int, p: BTParams) -> bool:
    if idx < max(p.sma_long, p.vol_avg_period) or pd.isna(df.iloc[idx]["SMA200"]):
        return False
    rsi_ok = df.iloc[idx]["RSI"] <= p.rsi_thresh
    macd_ok = df["CROSS_UP"].iloc[idx-p.macd_lookback+1:idx+1].any()
    price = df.iloc[idx]["AdjClose"]
    sma200 = df.iloc[idx]["SMA200"]
    sma_dev_ok = (price - sma200) / sma200 <= -p.sma_dev_req
    up_day = df.iloc[idx]["Close"] > df.iloc[idx-1]["Close"]
    vol_spike_ok = up_day and df.iloc[idx]["Volume"] >= p.vol_spike_mult * df.iloc[idx]["VOL_AVG"]
    return bool(rsi_ok and macd_ok and sma_dev_ok and vol_spike_ok)

def backtest(params: BTParams):
    price_data: Dict[str, pd.DataFrame] = {}
    for t in params.universe:
        df = fetch_history(t, params.start, params.end)
        if df.empty:
            continue
        df = compute_signals(df, params)
        price_data[t] = df

    if not price_data:
        raise RuntimeError("No data for the selected tickers/timeframe.")

    dates = sorted(set().union(*[df.index for df in price_data.values()]))
    equity = params.initial_equity
    cash = equity
    borrow = 0.0
    margin_cap = equity * params.margin_ltv_cap
    positions: Dict[str, Dict[str, Any]] = {}
    trades = []
    curve = []  # daily equity

    for i, d in enumerate(dates):
        # accrue interest on borrow
        daily_rate = params.annual_margin_rate / 252.0
        if borrow > 0:
            cash -= borrow * daily_rate

        # exits
        to_close = []
        for t, pos in list(positions.items()):
            df = price_data[t]
            if d not in df.index: 
                continue
            px = float(df.loc[d, "Close"])
            held_days = (d - pos["entry_date"]).days

            if (not pos["trailing_on"]) and px >= pos["entry_price"] * (1 + params.trailing_trigger_gain):
                pos["trailing_on"] = True
                pos["peak_price"] = px
            if pos["trailing_on"]:
                pos["peak_price"] = max(pos["peak_price"], px)
                trail_stop = pos["peak_price"] * (1 - params.trailing_pct)
                if px <= trail_stop:
                    to_close.append((t, px, d, "trailing")); continue
            if px <= pos["stop_price"]:
                to_close.append((t, px, d, "stop")); continue
            if px >= pos["target_price"]:
                to_close.append((t, px, d, "target")); continue
            if held_days >= params.max_hold_days:
                to_close.append((t, px, d, "time")); continue

        for t, px, d_close, reason in to_close:
            pos = positions.pop(t)
            proceeds = px * pos["shares"]
            repay = min(borrow, proceeds)
            borrow -= repay
            cash += proceeds - repay
            trades.append({"ticker": t, "entry_date": pos["entry_date"], "exit_date": d_close,
                           "entry": pos["entry_price"], "exit": px, "shares": pos["shares"], "reason": reason})

        # entries weekly (Mondays) or daily if evaluate_weekly=False
        weekday = d.weekday()
        if (not params.evaluate_weekly) or weekday == 0:
            capacity = params.max_positions - len(positions)
            if capacity > 0:
                # rank candidates
                ranked = []
                for t, df in price_data.items():
                    if t in positions or d not in df.index: 
                        continue
                    idx = df.index.get_loc(d)
                    if isinstance(idx, slice): 
                        continue
                    if idx < 2: 
                        continue
                    if entry_ok(df, idx, params):
                        rsi_val = float(df.loc[d, "RSI"])
                        sma_dev = (float(df.loc[d, "AdjClose"]) - float(df.loc[d, "SMA200"])) / float(df.loc[d, "SMA200"])
                        score = (35 - rsi_val) + abs(sma_dev) * 100
                        ranked.append((score, t))
                ranked.sort(reverse=True)
                for _, t in ranked[:capacity]:
                    df = price_data[t]
                    px = float(df.loc[d, "Close"])
                    stop_price = px * (1 - params.stop_loss_pct)
                    target_price = px * (1 + params.target_gain_pct)
                    stop_dist = px - stop_price
                    per_trade_risk = params.initial_equity * params.risk_per_trade_pct
                    shares = int(per_trade_risk / stop_dist) if stop_dist > 0 else 0
                    # fit within margin cap
                    open_exposure = sum(float(price_data[tt].loc[d, "Close"]) * positions[tt]["shares"]
                                        for tt in positions if d in price_data[tt].index)
                    new_exposure = px * shares
                    total_exposure = open_exposure + new_exposure
                    borrow_needed = max(0.0, total_exposure - cash)
                    if borrow + borrow_needed > margin_cap:
                        allowed_extra_borrow = margin_cap - borrow
                        max_new_exposure = cash + allowed_extra_borrow - open_exposure
                        shares = int(max_new_exposure / px) if px > 0 else 0
                    if shares <= 0: 
                        continue
                    cost = px * shares
                    if cost <= cash:
                        cash -= cost
                    else:
                        borrow_inc = cost - cash
                        cash = 0.0
                        borrow += borrow_inc
                    positions[t] = {"entry_date": d, "entry_price": px, "shares": shares,
                                    "stop_price": stop_price, "target_price": target_price,
                                    "trailing_on": False, "peak_price": px}
                    trades.append({"ticker": t, "entry_date": d, "entry": px, "shares": shares, "reason": "entry"})

        # mark-to-market equity
        mtm = 0.0
        for t, pos in positions.items():
            df = price_data[t]
            if d in df.index:
                mtm += float(df.loc[d, "Close"]) * pos["shares"]
        equity = cash + mtm - borrow
        curve.append({"date": d, "equity": equity, "cash": cash, "borrow": borrow, "positions": len(positions)})

    curve_df = pd.DataFrame(curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return curve_df, trades_df

def performance_stats(curve: pd.DataFrame) -> Dict[str, float]:
    eq = curve["equity"].astype(float)
    rets = eq.pct_change().dropna()
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252/len(rets)) - 1 if len(rets) > 0 else 0.0
    vol = rets.std() * (252 ** 0.5) if len(rets) > 0 else 0.0
    sharpe = (rets.mean()*252) / vol if vol > 1e-9 else 0.0
    # max drawdown
    roll_max = eq.cummax()
    dd = (eq/roll_max - 1.0)
    max_dd = dd.min()
    return {"CAGR": round(cagr,4), "Vol": round(vol,4), "Sharpe": round(sharpe,2), "MaxDD": round(float(max_dd),4)}

def plot_equity(curve: pd.DataFrame, out_path: str = "data/bt_equity.png"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(curve.index, curve["equity"], label="Equity")
    plt.xticks(rotation=45)
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSX Margin Strategy Backtester")
    parser.add_argument("--config", type=str, default="", help="Path to JSON config, else defaults used")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        p = BTParams(**cfg)
    else:
        p = BTParams(
            universe=["RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","ENB.TO","BCE.TO","CNQ.TO","SU.TO","CP.TO","CNR.TO"],
            start="2018-01-01",
            end="2025-01-01",
            initial_equity=100_000.0,
            risk_per_trade_pct=0.01,
            margin_ltv_cap=0.30,
            max_positions=2,
            annual_margin_rate=0.055,
        )

    curve, trades = backtest(p)
    stats = performance_stats(curve)
    print("Performance:", stats)
    curve.to_csv("data/bt_equity_curve.csv")
    trades.to_csv("data/bt_trades.csv", index=False)
    plot_equity(curve, out_path="data/bt_equity.png")
    print("Saved: data/bt_equity_curve.csv, data/bt_trades.csv, data/bt_equity.png")
