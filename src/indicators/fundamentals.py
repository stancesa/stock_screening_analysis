import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf

def robust_dividend_yield_pct(tkr: yf.Ticker, fallback_info: dict | None = None) -> float | None:
    """
    Compute trailing-12-month dividend yield from actual dividend cashflows.
    Falls back to forward 'dividendRate' if TTM data is missing.
    Also tries to ignore an obvious one-off 'special' dividend.
    """
    try:
        # 1) Sum last 12 months of dividends
        divs = tkr.dividends
        if divs is None or divs.empty:
            raise ValueError("No dividend history")

        cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=365))
        ttm = divs[divs.index >= cutoff]
        if ttm.empty:
            raise ValueError("No TTM dividends")

        # Optional: drop a likely special dividend (very crude heuristic)
        if len(ttm) >= 3:
            med = ttm.median()
            # If any payment > 2.5x median, treat as special and drop it
            ttm = ttm[ttm <= 2.5 * med]

        ttm_sum = float(ttm.sum())

        # 2) Get a recent price
        try:
            price = float(tkr.fast_info.get("last_price"))
            if not np.isfinite(price) or price <= 0:
                raise Exception
        except Exception:
            hist = tkr.history(period="5d", interval="1d")
            price = float(hist["Close"].dropna().iloc[-1])

        if price <= 0:
            raise ValueError("Invalid price")

        yld = 100.0 * (ttm_sum / price)
        # sanity check: if absurd (>20%), try a forward fallback
        if yld > 20.0 and fallback_info:
            fwd_rate = fallback_info.get("dividendRate")
            if fwd_rate:
                yld = 100.0 * (float(fwd_rate) / price)
        return round(yld, 2)
    except Exception:
        # Forward fallback if available
        if fallback_info:
            try:
                # dividendRate is the annual forward amount, same currency as price
                hist_price = tkr.fast_info.get("last_price")
                if not hist_price:
                    hist = tkr.history(period="5d", interval="1d")
                    hist_price = float(hist["Close"].dropna().iloc[-1])
                rate = float(fallback_info.get("dividendRate"))
                return round(100.0 * rate / float(hist_price), 2)
            except Exception:
                return None
        return None

def get_fundamentals(ticker: str) -> dict:
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        qf = tkr.quarterly_financials
        qb = tkr.quarterly_balance_sheet
        qc = tkr.quarterly_cashflow

        fundamentals = {}

        # Basic PE / Forward PE
        fundamentals["pe_ttm"] = info.get("trailingPE")
        fundamentals["pe_fwd"] = info.get("forwardPE")
        fundamentals["div_yield_pct"] = robust_dividend_yield_pct(tkr, fallback_info=info)

        # Debt-to-Equity
        total_debt = qb.loc["Total Debt"].iloc[0] if "Total Debt" in qb.index else np.nan
        total_equity = qb.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in qb.index else np.nan
        fundamentals["debt_to_equity"] = (total_debt / total_equity) if (total_equity and total_equity != 0) else np.nan

        # Revenue & EPS growth (latest vs previous quarter)
        if "Total Revenue" in qf.index:
            rev_now = qf.loc["Total Revenue"].iloc[0]
            rev_prev = qf.loc["Total Revenue"].iloc[1] if qf.shape[1] > 1 else np.nan
            fundamentals["revenue_growth_qoq"] = ((rev_now - rev_prev) / rev_prev) * 100 if rev_prev else np.nan

        if "Basic EPS" in qf.index:
            eps_now = qf.loc["Basic EPS"].iloc[0]
            eps_prev = qf.loc["Basic EPS"].iloc[1] if qf.shape[1] > 1 else np.nan
            fundamentals["eps_growth_qoq"] = ((eps_now - eps_prev) / eps_prev) * 100 if eps_prev else np.nan

        # Profit margin
        if "Net Income" in qf.index and "Total Revenue" in qf.index:
            ni = qf.loc["Net Income"].iloc[0]
            tr = qf.loc["Total Revenue"].iloc[0]
            fundamentals["net_profit_margin"] = (ni / tr) * 100 if tr else np.nan

        # Free Cash Flow margin
        if "Total Revenue" in qf.index and "Total Cash From Operating Activities" in qc.index:
            fcf = qc.loc["Total Cash From Operating Activities"].iloc[0]
            tr = qf.loc["Total Revenue"].iloc[0]
            fundamentals["fcf_margin"] = (fcf / tr) * 100 if tr else np.nan

        # PEG ratio
        eps_growth_yr = info.get("earningsQuarterlyGrowth")
        if fundamentals["pe_ttm"] and eps_growth_yr:
            fundamentals["peg_ratio"] = fundamentals["pe_ttm"] / (eps_growth_yr * 100 if eps_growth_yr < 1 else eps_growth_yr)
        else:
            fundamentals["peg_ratio"] = np.nan

        return fundamentals

    except Exception as e:
        print(f"[WARN] Fundamentals fetch failed for {ticker}: {e}")
        return {}
