# ---------- Pretty labels + hover help for headline weights ----------
CATEGORY_LABELS = {
    "equities":     "US Equities — trend + momentum",
    "breadth":      "Breadth — participation vs mega-caps",
    "vol":          "Volatility — VIX level & term structure",
    "rates_credit": "Rates & Credit — curve, breakevens, credit",
    "fx":           "Forex — USD vs G10 / DXY",
    "intl":         "International Equities — ex-US trends",
    "commodities":  "Commodities & Real Assets",
    "reits":        "REITs — real estate tilt",
    "crypto":       "Crypto Proxy — BTC/ETH trend",
    "internals":    "Market Internals — put/call, MOVE, SKEW",
}

CATEGORY_HELP = {
    "equities": (
        "US Equities — trend + momentum.\n\n"
        "Core pulse of the market (SPY/QQQ/IWM/RSP). When this weight is high, the composite "
        "leans heavily on equity price action itself. Strong uptrends lift the headline more, "
        "and selloffs drag it harder. Lower weight = equities have less sway; higher = the market "
        "regime is defined by equity benchmarks."
    ),

    "breadth": (
        "Breadth — participation vs mega-caps.\n\n"
        "Measures whether rallies are broad-based or dominated by a few giants. High breadth "
        "weight rewards environments where many stocks participate (healthier, sustainable rallies). "
        "Low breadth or weak internals drag the headline. Raising this weight emphasizes the quality "
        "of participation over index-level gains."
    ),

    "vol": (
        "Volatility — VIX level & term structure.\n\n"
        "Captures stress or calm in option markets. Low vol → risk-on, high vol → risk-off. "
        "High weight = spikes in VIX punish the composite more, calm conditions boost it. "
        "Lower weight = volatility plays a minor role in sentiment."
    ),

    "rates_credit": (
        "Rates & Credit — curve, breakevens, credit spreads.\n\n"
        "Represents funding/economic conditions. Steep curves, tight spreads, and benign breakevens "
        "signal easier conditions (risk-on). Inversions, widening spreads, or stressed credit weigh "
        "on the composite if this weight is high. Reducing the weight minimizes macro/rates impact."
    ),

    "fx": (
        "Forex — USD vs G10 / DXY.\n\n"
        "A strong dollar can pressure risk assets; a weak USD often supports them. "
        "When this weight is high, swings in FX meaningfully shift the composite. "
        "Lower weight = FX is less influential, focusing more on domestic equity/credit signals."
    ),

    "intl": (
        "International Equities — ex-US trends.\n\n"
        "Signals global risk appetite. Strong EM/DM ex-US rallies reinforce bullish tone if weighted. "
        "Weakness abroad drags when weight is high. Setting low weight makes the headline US-centric; "
        "raising it ties sentiment more to global flows."
    ),

    "commodities": (
        "Commodities & Real Assets.\n\n"
        "Reflects cyclical and reflationary forces (energy, metals). Rising commodities can imply "
        "growth optimism—or inflation risk. With higher weight, commodity moves swing the composite "
        "more (pro-growth or inflation-driven). With lower weight, commodities play a minor supporting role."
    ),

    "reits": (
        "REITs — real estate tilt.\n\n"
        "Highly sensitive to interest rates and credit conditions. Strong REITs = supportive backdrop "
        "for risk. Weak REITs = stress in rate-sensitive sectors. Raising this weight makes rate/risk "
        "conditions more visible in the headline. Lower weight = less emphasis on real estate sensitivity."
    ),

    "crypto": (
        "Crypto Proxy — BTC/ETH trend.\n\n"
        "Acts as a speculative risk barometer. Strong crypto = high speculative risk-on tone; "
        "weak crypto = risk-off. With high weight, crypto swings can move the composite strongly. "
        "With lower weight, crypto is just a fringe signal."
    ),

    "internals": (
        "Market Internals — put/call, MOVE, SKEW, etc.\n\n"
        "Captures hidden cross-asset signals (option skew, bond vol, hedging flows). "
        "Raising this weight makes subtle cross-market stress/optimism show up in the headline. "
        "Lower weight = composite is driven more by surface-level price action."
    ),
}
# Keep a stable order in the UI
CATEGORY_ORDER = [
    "equities","breadth","vol","rates_credit","fx",
    "intl","commodities","reits","crypto","internals"
]