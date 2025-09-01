
import math, datetime as dt
from typing import Dict, List
import numpy as np
import yfinance as yf
import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = True
except Exception:
    _VADER = False

CATALYST_KEYWORDS = [
    "earnings","guidance","dividend","buyback","repurchase","merger","acquisition","m&a","deal","takeover",
    "upgrade","downgrade","initiated","price target","contract","award","order","tender","rfp",
    "approval","clearance","authorization","permit","partnership","collaboration","licensing",
    "lawsuit","investigation","probe","settlement",
]
POSITIVE_HINTS = {"beats","beat","raises","hike","increase","surge","record","wins","approves"}
NEGATIVE_HINTS = {"miss","misses","cuts","slump","delay","recall","halts","downgrade","investigation"}

def get_news_and_sentiment(ticker: str, lookback_days: int = 14) -> Dict[str, object]:
    t = yf.Ticker(ticker)
    news_items = getattr(t, "news", []) or []
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=lookback_days)
    headlines: List[str] = []
    for it in news_items:
        try:
            ts = dt.datetime.utcfromtimestamp(int(it.get("providerPublishTime", 0)))
            if ts >= cutoff:
                headlines.append(it.get("title", ""))
        except Exception:
            continue
    vader_score = None
    if _VADER and headlines:
        analyzer = SentimentIntensityAnalyzer()
        vals = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        if vals:
            vader_score = float(np.mean(vals))
    if vader_score is None and headlines:
        score = 0
        for h in headlines:
            hlow = h.lower()
            for p in POSITIVE_HINTS:
                if p in hlow: score += 1
            for n in NEGATIVE_HINTS:
                if n in hlow: score -= 1
        vader_score = score / max(1, len(headlines))
    catalysts = []
    for h in headlines:
        hlow = h.lower()
        for kw in CATALYST_KEYWORDS:
            if kw in hlow:
                catalysts.append(kw)
    catalysts = sorted(list(set(catalysts)))
    return {
        "headlines": headlines[:10],
        "sentiment": vader_score if vader_score is not None else float("nan"),
        "catalysts": ",".join(catalysts) if catalysts else "",
    }

def get_fundamentals(ticker: str) -> Dict[str, float]:
    t = yf.Ticker(ticker)

    # -------- helpers --------
    def _first_col(df: pd.DataFrame):
        if df is None or df.empty:
            return None
        return df.columns[0]

    def _get_row(df: pd.DataFrame, names) -> float:
        """Try multiple possible row labels and return latest quarter value (first column)."""
        if df is None or df.empty:
            return float("nan")
        col = _first_col(df)
        if col is None:
            return float("nan")
        for n in names:
            if n in df.index:
                try:
                    return float(df.loc[n, col])
                except Exception:
                    continue
        return float("nan")

    def _two_quarters(df: pd.DataFrame, names):
        """Return (q0, q1) for a row across the first two columns if available."""
        if df is None or df.empty or df.shape[1] < 1:
            return (float("nan"), float("nan"))
        # first two most recent quarters (yfinance has most-recent as first column)
        cols = list(df.columns)[:2]
        for n in names:
            if n in df.index:
                try:
                    q0 = float(df.loc[n, cols[0]])
                except Exception:
                    q0 = float("nan")
                try:
                    q1 = float(df.loc[n, cols[1]]) if len(cols) > 1 else float("nan")
                except Exception:
                    q1 = float("nan")
                return (q0, q1)
        return (float("nan"), float("nan"))

    # -------- fetch raw blocks --------
    try:
        info = t.info or {}
    except Exception:
        info = {}

    try:
        qf = t.quarterly_financials  # IS
    except Exception:
        qf = None
    try:
        qb = t.quarterly_balance_sheet  # BS (quarterly)
    except Exception:
        qb = None
    try:
        qc = t.quarterly_cashflow  # CF (quarterly)
    except Exception:
        qc = None
    try:
        bs_annual = t.balance_sheet  # keep your original annual D/E fallback
    except Exception:
        bs_annual = None

    # -------- basics already in your block --------
    pe  = info.get("trailingPE")
    fpe = info.get("forwardPE")
    div_yield = info.get("dividendYield")
    payout = info.get("payoutRatio")

    # normalize dividend to percent if needed
    if isinstance(div_yield, (int, float)) and div_yield is not None and div_yield < 1:
        div_yield = div_yield * 100.0

    # -------- Debt / Equity (prefer quarterly, fallback to annual) --------
    def _de_from_bs(df):
        if df is None or df.empty:
            return float("nan")
        col = _first_col(df)
        if col is None:
            return float("nan")
        # Try multiple label variants that appear across tickers
        total_debt = _get_row(df, ["Total Debt", "Total Liab", "Total Liabilities Net Minority Interest"])
        total_equity = _get_row(df, [
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
            "Total stockholders' equity",
        ])
        if not np.isfinite(total_debt) or not np.isfinite(total_equity) or total_equity == 0:
            return float("nan")
        return float(total_debt / total_equity)

    de_q = _de_from_bs(qb)
    if not np.isfinite(de_q):
        # your original fallback (annual)
        if bs_annual is not None and not bs_annual.empty:
            col = _first_col(bs_annual)
            try:
                tl = float(bs_annual.loc["Total Liab", col]) if "Total Liab" in bs_annual.index else float("nan")
            except Exception:
                tl = float("nan")
            try:
                te = float(bs_annual.loc["Total Stockholder Equity", col]) if "Total Stockholder Equity" in bs_annual.index else float("nan")
            except Exception:
                te = float("nan")
            de_q = (tl / te) if (np.isfinite(tl) and np.isfinite(te) and te != 0) else float("nan")

    # -------- Quarterly revenue & EPS growth --------
    rev0, rev1 = _two_quarters(qf, ["Total Revenue", "Revenue", "Revenues"])
    # EPS row names vary; try both basic/diluted; fallback to quarterly_earnings "Earnings"
    eps0, eps1 = _two_quarters(qf, ["Basic EPS", "Diluted EPS", "EPS (Basic)", "EPS (Diluted)"])

    # If EPS rows missing, try quarterly_earnings (Earnings) to proxy EPS growth (imperfect but useful)
    if not np.isfinite(eps0) or not np.isfinite(eps1):
        try:
            qis = getattr(t, "quarterly_income_stmt", None)
            if qis is not None and not qis.empty:
                # Try "Net Income" as a proxy
                net_inc = _two_quarters(qis, ["Net Income", "Net Income Common Stockholders"])
                eps0, eps1 = net_inc
        except Exception:
            pass

    def _pct_change(new, old):
        if not np.isfinite(new) or not np.isfinite(old) or old == 0:
            return float("nan")
        return (new - old) / old * 100.0

    revenue_growth_qoq = _pct_change(rev0, rev1)
    eps_growth_qoq = _pct_change(eps0, eps1)

    # -------- Margins --------
    net_income = _get_row(qf, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
    total_revenue = rev0 if np.isfinite(rev0) else _get_row(qf, ["Total Revenue", "Revenue", "Revenues"])
    net_profit_margin = float("nan")
    if np.isfinite(net_income) and np.isfinite(total_revenue) and total_revenue != 0:
        net_profit_margin = (net_income / total_revenue) * 100.0

    # Free Cash Flow = CFO - CapEx (quarterly)
    cfo = _get_row(qc, ["Total Cash From Operating Activities", "Operating Cash Flow"])
    capex = _get_row(qc, ["Capital Expenditures", "Investments In Property, Plant, and Equipment"])
    fcf = float("nan")
    if np.isfinite(cfo) and np.isfinite(capex):
        fcf = cfo - capex
    fcf_margin = float("nan")
    if np.isfinite(fcf) and np.isfinite(total_revenue) and total_revenue != 0:
        fcf_margin = (fcf / total_revenue) * 100.0

    # -------- PEG ratio --------
    # Prefer explicit growth; fallback to annualizing QoQ EPS growth if needed.
    eqg = info.get("earningsQuarterlyGrowth")  # often ~ YoY EPS growth as decimal (e.g., 0.25 = 25%)
    growth_pct = float("nan")
    if isinstance(eqg, (int, float)) and np.isfinite(eqg):
        growth_pct = eqg * 100.0
    elif np.isfinite(eps_growth_qoq):
        try:
            annualized = (1.0 + eps_growth_qoq / 100.0) ** 4 - 1.0
            growth_pct = annualized * 100.0
        except Exception:
            growth_pct = float("nan")

    peg_ratio = float("nan")
    if isinstance(pe, (int, float)) and np.isfinite(pe) and np.isfinite(growth_pct) and growth_pct != 0:
        # PEG = PE / growth(%)
        peg_ratio = pe / growth_pct

    return {
        "pe_ttm": round(pe, 2) if isinstance(pe, (int, float)) else None,
        "pe_fwd": round(fpe, 2) if isinstance(fpe, (int, float)) else None,
        "div_yield_pct": round(div_yield, 2) if isinstance(div_yield, (int, float)) else None,
        "payout_ratio": round(payout, 2) if isinstance(payout, (int, float)) else None,

        "debt_to_equity": round(de_q, 2) if isinstance(de_q, (int, float)) and np.isfinite(de_q) else None,

        "revenue_growth_qoq": round(revenue_growth_qoq, 2) if np.isfinite(revenue_growth_qoq) else None,
        "eps_growth_qoq": round(eps_growth_qoq, 2) if np.isfinite(eps_growth_qoq) else None,
        "net_profit_margin": round(net_profit_margin, 2) if np.isfinite(net_profit_margin) else None,
        "fcf_margin": round(fcf_margin, 2) if np.isfinite(fcf_margin) else None,
        "peg_ratio": round(peg_ratio, 2) if np.isfinite(peg_ratio) else None,
    }
