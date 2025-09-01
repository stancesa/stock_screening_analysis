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
        "Core U.S. equity trend and momentum signals, combining weekly tanh trends for SPY, QQQ, IWM, RSP, "
        "and VTI plus SPY’s 4w/13w momentum. Higher values = stronger equity tape. Raising this weight makes "
        "the overall headline more equity-driven."
    ),
    "breadth": (
        "Participation breadth indicators such as the RSP/SPY ratio vs. 200dma and 1-year percentile. "
        "Higher values = healthier, broader participation vs. narrow mega-cap leadership."
    ),
    "vol": (
        "Volatility regime using VIX levels/percentiles and term structure (VIX9D vs. VIX3M). "
        "Lower volatility percentiles = calmer, risk-on conditions. Backwardation (short-term VIX > long-term) "
        "counts as stress. Higher weight penalizes spikes in volatility."
    ),
    "rates_credit": (
        "Interest rates and credit proxies: credit risk-on (HYG/IEF above 200dma), breakevens (TIP/IEF), "
        "and curve slope (SHY/TLT; falling = easing). Higher values = looser/easier financial conditions."
    ),
    "fx": (
        "Foreign Exchange (U.S. Dollar trend). Based on DXY/UUP and USD vs. G10 composite. "
        "USD weakness is generally risk-on for global assets. More weight increases headline sensitivity to USD swings."
    ),
    "international": (
        "Developed and Emerging Market equity tone, using trends across regional ETFs (Europe, Asia, EM). "
        "Higher values = stronger ex-U.S. equity performance."
    ),
    "commodities": (
        "Energy, metals, and broad commodity baskets. Rising commodities often reflect reflationary/cyclical strength. "
        "Includes TLT inversion as a supportive rates component. Higher values = stronger commodity trends."
    ),
    "reits": (
        "Real Estate Investment Trusts (U.S./Global/Regional) and REITs vs. ACWI relative strength. "
        "Sensitive to real rates and liquidity conditions. Higher values = stronger REIT performance."
    ),
    "crypto": (
        "Crypto assets (BTC/ETH) weekly trend and 200dma plus a simple crypto ‘risk state’. "
        "Serves as a speculative risk/liquidity proxy. Higher values = stronger speculative risk appetite."
    ),
    "internals": (
        "Market internals and positioning stress measures: Put/Call ratios (CPC/CPCE), MOVE index percentiles, "
        "and SKEW hints. Lower values = healthier/less stressed market tape."
    ),
}
# Keep a stable order in the UI
CATEGORY_ORDER = [
    "equities","breadth","vol","rates_credit","fx",
    "intl","commodities","reits","crypto","internals"
]