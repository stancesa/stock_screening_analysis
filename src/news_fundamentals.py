
import math, datetime as dt
from typing import Dict, List
import numpy as np
import yfinance as yf

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

def get_fundamentals(ticker: str):
    t = yf.Ticker(ticker)
    pe = fpe = div_yield = payout = de = None
    try:
        info = t.info or {}
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
    except Exception:
        pass
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            col = bs.columns[0]
            tl = float(bs.loc["Total Liab", col]) if "Total Liab" in bs.index else None
            te = float(bs.loc["Total Stockholder Equity", col]) if "Total Stockholder Equity" in bs.index else None
            if tl is not None and te and te != 0:
                de = tl / te
    except Exception:
        pass
    if div_yield is not None and div_yield < 1: div_yield *= 100.0
    return {
        "pe_ttm": round(pe, 2) if isinstance(pe, (int, float)) else None,
        "pe_fwd": round(fpe, 2) if isinstance(fpe, (int, float)) else None,
        "div_yield_pct": round(div_yield, 2) if isinstance(div_yield, (int, float)) else None,
        "payout_ratio": round(payout, 2) if isinstance(payout, (int, float)) else None,
        "debt_to_equity": round(de, 2) if isinstance(de, (int, float)) else None,
    }
