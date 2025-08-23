import numpy as np
import pandas as pd
from .projection import apply_slippage_px, clip_to_participation
from core.types import ExecParams

def run_dca_backtest(
    dates: pd.Series,
    close: np.ndarray,
    buy_idx: np.ndarray,
    sell_idx: np.ndarray,
    *,
    starting_cash: float = 10_000.0,
    buy_pct_first: float = 25.0,
    buy_pct_next: float = 25.0,
    dca_trigger_drop_pct: float = 5.0,
    max_dca_legs: int = 3,
    sell_pct_first: float = 50.0,
    sell_pct_next: float = 50.0,
) -> tuple[np.ndarray, list[dict], float]:
    """
    DCA/backtest on top of signal series.
    Returns (equity_curve, trades, total_return_fraction).
    """
    buy_set  = set(map(int, buy_idx))
    sell_set = set(map(int, sell_idx))

    cash = float(starting_cash)
    shares = 0.0
    last_buy_px = np.nan
    dca_legs = 0
    sold_once_since_last_buy = False

    equity = np.zeros(len(close), dtype=float)
    trades: list[dict] = []

    for i, px in enumerate(close):
        # -- SELL first (priority to risk management)
        if i in sell_set and shares > 0:
            pct = sell_pct_first if not sold_once_since_last_buy else sell_pct_next
            qty = shares * max(min(pct, 100.0), 0.0) / 100.0
            if qty > 0:
                proceeds = qty * px
                cash += proceeds
                shares -= qty
                trades.append({"side":"SELL","i":i,"date":dates[i],"price":float(px),"qty":float(qty),"cash":float(cash),"shares":float(shares)})
                sold_once_since_last_buy = True
                if shares <= 1e-9:
                    shares = 0.0
                    dca_legs = 0
                    last_buy_px = np.nan
                    sold_once_since_last_buy = False

        # -- BUY (DCA)
        if i in buy_set:
            gate = (not np.isfinite(last_buy_px)) or (px <= last_buy_px * (1.0 - dca_trigger_drop_pct / 100.0))
            if dca_legs < max_dca_legs and gate and cash > 0:
                pct = buy_pct_first if not np.isfinite(last_buy_px) else buy_pct_next
                invest = cash * max(min(pct, 100.0), 0.0) / 100.0
                if invest > 0:
                    qty = invest / px
                    cash -= invest
                    shares += qty
                    last_buy_px = float(px)
                    dca_legs += 1
                    sold_once_since_last_buy = False
                    trades.append({"side":"BUY","i":i,"date":dates[i],"price":float(px),"qty":float(qty),"cash":float(cash),"shares":float(shares)})

        equity[i] = cash + shares * px

    total_return = (equity[-1] / starting_cash) - 1.0 if len(equity) else 0.0
    return equity, trades, float(total_return)
    

def run_backtest_v2(
    dates: pd.Series,
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
    buy_idx: np.ndarray, sell_idx: np.ndarray,
    *,
    starting_cash: float = 10_000.0,
    risk_per_trade: float = 0.005,          # 0.5%
    stop_prices: dict[int, float] | None = None,   # optional precomputed {i: stop}
    target_prices: dict[int, float] | None = None, # optional {i: target}
    execp: ExecParams = ExecParams()
) -> tuple[np.ndarray, list[dict], float]:
    """
    Event-driven loop, entries at next open with slippage, exits via signals or OCO.
    """
    n = len(close)
    cash = float(starting_cash)
    shares = 0.0
    entry_i = None
    entry_px = np.nan

    buy_set  = set(map(int, buy_idx))
    sell_set = set(map(int, sell_idx))
    stop_prices = stop_prices or {}
    target_prices = target_prices or {}

    eq = np.zeros(n, float)
    trades: list[dict] = []

    for i in range(n - 1):  # execute at i+1 open
        # Check active position OCO on current bar close (intraday hit)
        if shares != 0:
            stp = stop_prices.get(i, np.nan)
            tgt = target_prices.get(i, np.nan)
            hit_stop = np.isfinite(stp) and low[i+1] <= stp
            hit_tgt  = np.isfinite(tgt) and high[i+1] >= tgt

            if hit_stop or hit_tgt or (i in sell_set and shares > 0):
                side = "SELL" if shares > 0 else "BUY"
                px = open_[i+1]
                # gap-through logic: if gap beyond level, execute at open (worse fill)
                px = apply_slippage_px(px, side, execp.slippage_bps)
                qty = abs(shares)
                # commission
                fee = qty * execp.commission_per_share
                cash += qty * px - fee if side == "SELL" else -qty * px - fee
                trades.append({
                    "side": side, "i": i+1, "date": dates.iloc[i+1],
                    "price": float(px), "qty": float(qty), "cash": float(cash), "shares": 0.0,
                    "exit_reason": "STOP" if hit_stop else "TARGET" if hit_tgt else "SIGNAL"
                })
                shares = 0.0
                entry_i = None
                entry_px = np.nan

        # New entry
        if (i in buy_set) and shares == 0:
            # risk-based sizing (use stop if available)
            stp = stop_prices.get(i, np.nan)
            px_next = apply_slippage_px(open_[i+1], "BUY", execp.slippage_bps)
            risk_dollars = cash * risk_per_trade
            per_share_risk = max(px_next - stp, 1e-9) if np.isfinite(stp) and stp < px_next else px_next * 0.10
            intended_qty = int(risk_dollars / per_share_risk)

            # participation cap
            qty = clip_to_participation(intended_qty, volume[i+1], execp.max_participation)
            if qty > 0:
                fee = qty * execp.commission_per_share
                if cash >= qty * px_next + fee:
                    cash -= qty * px_next + fee
                    shares += qty
                    entry_i = i+1
                    entry_px = px_next
                    trades.append({
                        "side": "BUY", "i": i+1, "date": dates.iloc[i+1],
                        "price": float(px_next), "qty": float(qty), "cash": float(cash), "shares": float(shares)
                    })

        eq[i] = cash + shares * close[i]

    eq[-1] = cash + shares * close[-1]
    total_return = (eq[-1] / starting_cash) - 1.0
    return eq, trades, float(total_return)

