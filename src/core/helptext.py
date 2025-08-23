# Specific per-control help (multi-line tooltips). Hover the "?" next to each widget.

def h(k: str) -> str:  # tiny helper to keep widget lines tidy
    return HELP[k]

HELP = {
    # ===== BUY: thresholds & knobs =====
    "composite_threshold":
        "BUY fires only if the weighted composite ≥ this value.\n"
        "Composite = Σ(weight × component) + gap_bonus.\n"
        "Higher ↗ = pickier (fewer buys). Lower ↘ = more frequent buys.",

    "rsi_buy_max":
        "RSI(14, Wilder). BUY uses s_rsi = sigmoid((rsi_buy_max − RSI)/5).\n"
        "Lower RSI ↘ → higher BUY score (more oversold). Values above rsi_buy_max add little.",

    "vol_ratio_min":
        "Volume spike gate using vol / 20-day avg volume.\n"
        "Higher ↗ = require stronger accumulation days. Lower ↘ = allow subtle flows.",

    "donch_lookback":
        "N-bar Donchian channel. BUY uses prior N-day high; SELL uses prior N-day low (yesterday’s channel).\n"
        "Smaller N ↘ = more, earlier breakouts. Larger N ↗ = stricter, fewer signals.",

    "gap_min_pct":
        "Extra BUY bonus if today’s open > yesterday’s high by ≥ this % (gap-up momentum).\n"
        "Higher ↗ = only big gaps earn a bonus. Set 0 to disable.",

    "atr_mult":
        "Engine stop = last − ATR(14) × multiple (when ATR available).\n"
        "Higher ↗ = wider stop (smaller position, more room). Lower ↘ = tighter stop.",

    "stop_pct_ui":
        "Fallback % stop if ATR isn’t available: stop = last × (1 − %/100).",

    "value_center":
        "Value compares last vs SMA200 and centers preference at this %.\n"
        "More negative ↘ = favors buying discounts below SMA200. Less negative/positive ↗ = less emphasis on discount.",

    "sma_window":
        "Window (days) for price & SMA200 slopes used in Trend.\n"
        "Larger ↗ = smoother/slower. Smaller ↘ = faster/more reactive.",

    "use_engine_stop":
        "If ON, chart shows engine’s stop/target (ATR/swing/band/% fallback) instead of the manual risk band.",

    "bb_window":
        "Bollinger MA window for %B/bandwidth (typical: 20). Larger ↗ = smoother, slower bands.",

    "bb_k":
        "Bollinger width (k·σ). Larger ↗ = wider bands (slower %B). Smaller ↘ = tighter bands.",

    # ===== BUY: weights (0–1) – what the indicator means =====
    "w_rsi":
        "Weight of RSI in BUY. Lower RSI ↘ (more oversold) → higher component → more BUY when weight is higher.",

    "w_trend":
        "Weight of Trend (above/below SMA200 + positive slopes).\n"
        "Stronger uptrend ↗ → higher component when weight is higher.",

    "w_value":
        "Weight of Value vs SMA200. Bigger discount ↘ from SMA200 (near your value_center) → higher component.",

    "w_flow":
        "Weight of Accumulation Flow (vol / avg20).\n"
        "Bigger spikes ↗ → higher component when weight is higher.",

    "w_bbands":
        "Weight of Bollinger %B. %B≈0 near lower band, %B≈1 near upper band.\n"
        "Lower %B ↘ (near/below lower band) → higher BUY component.",

    "w_donch":
        "Weight of Donchian breakout strength.\n"
        "Strength ↗ as (last − prev_N_high) / ATR grows → more BUY bias.",

    "w_break":
        "Weight of legacy breakout flag (last ≥ prev_N_high).\n"
        "Use small/0 when using Donchian to avoid double-counting.",

    # ===== SELL: thresholds & knobs =====
    "sell_threshold":
        "SELL fires if SELL composite ≥ this value.\n"
        "Higher ↗ = pickier (fewer sells). Lower ↘ = trims/exits earlier/more often.",

    "rsi_overbought_min":
        "SELL RSI uses s_rsi ≈ sigmoid((RSI − threshold)/5) and boosts on roll-over.\n"
        "Lower threshold ↘ = reacts sooner to high RSI; higher ↗ = requires more extreme RSI.",

    "donch_lookback_sell":
        "N-bar Donchian for SELL prior low.\n"
        "Smaller N ↘ = earlier breakdowns; larger N ↗ = stricter.",

    "ema_fast_span":
        "Fast EMA span for SELL trend-down checks (cross-under + slope).\n"
        "Smaller ↘ = quicker to flip down; larger ↗ = slower.",

    "sma_mid_window":
        "Mid SMA window for SELL trend-down checks. Larger ↗ = smoother/laggier.",

    "gap_down_min_pct":
        "Extra SELL bonus if today’s open < yesterday’s low by ≥ this % (gap-down). Higher ↗ = only big gaps count.",

    # ===== SELL: weights (0–1) – what the indicator means =====
    "w_rsi_sell":
        "Weight of RSI-overbought/rollover. Higher RSI ↗ or rollover → higher SELL component.",

    "w_trend_down":
        "Weight of down-trend (below EMA/SMA + negative slopes). Stronger downtrend ↗ → higher SELL component.",

    "w_breakdown":
        "Weight of simple breakdown (last ≤ prior N-day low). Deep/decisive breaks ↗ → higher SELL component.",

    "w_exhaustion":
        "Weight of upper-band exhaustion (last > upper band). Farther above upper band ↗ → stronger SELL signal.",

    "w_flow_out":
        "Weight of distribution (down candle + volume spike). Bigger vol/avg20 ↗ → higher SELL component.",

    "w_bbands_sell":
        "Weight of %B for SELL. Higher %B ↗ (near upper band) → higher SELL component.",

    "w_donch_sell":
        "Weight of Donchian SELL strength.\n"
        "Strength ↗ as (prev_N_low − last) / ATR grows (deeper breakdown).",

    # ===== Appearance (theme colors) =====
    "appearance.close":
        "Primary price line (daily close).",
    "appearance.sma200":
        "Long-term trend (200-day SMA).",
    "appearance.overlay":
        "Default color for technical overlays (SMA/EMA/Donchian/etc.) unless an overlay specifies its own.",
    "appearance.stop":
        "Horizontal guideline for stop level.",
    "appearance.target":
        "Horizontal guideline for profit target.",
    "appearance.risk_band":
        "Filled zone between Stop and Target to visualize risk/reward.",
    "appearance.proj_mid":
        "Median path of the projection fan (dashed).",
    "appearance.proj_band":
        "Filled projection band between lower/upper percentiles.",

    # ===== Chart defaults =====
    "chart.show_sma":
        "Show SMA200 by default on Review charts.",
    "chart.show_stop":
        "Show Stop guideline by default (when available).",
    "chart.show_target":
        "Show Target guideline by default (when available).",
    "chart.range_days":
        "Default lookback (days) for Review chart. Larger ↗ = more history.",
    "chart.plot_height":
        "Default chart height in pixels. Increase for many overlays or a 2nd axis.",
    "chart.hovermode":
        "Tooltip mode: ‘x unified’ (one tooltip across traces), ‘closest’, or ‘x’.",

    # ===== Projections (defaults & advanced) =====
    "proj.enabled":
        "Show projection simulations by default.",
    "proj.band":
        "Credible interval for the fan (e.g., 10–90%). Narrower ↘ = tighter band; wider ↗ = more uncertainty shown.",
    "proj.sims":
        "Monte-Carlo paths. More ↗ = smoother estimates (slower).",
    "proj.model":
        "Engine: EWMA+t (fat-tailed), GBM, Bootstrap resampling, or Jump diffusion.",
    "proj.months":
        "Projection horizon in calendar months.",
    "proj.window":
        "History window used to calibrate drift/vol/jumps. Longer ↗ = smoother, more stable; shorter ↘ = more reactive.",
    "proj.lam":
        "EWMA volatility persistence (0.80–0.99). Higher ↗ = smoother vol; lower ↘ = more reactive.",
    "proj.df_t":
        "Student-t degrees of freedom. Lower ↘ = fatter tails (more extreme moves).",
    "proj.antithetic":
        "Use antithetic variates to reduce simulation noise.",
    "proj.block":
        "Bootstrap block size. Larger ↗ preserves longer serial dependence.",
    "proj.vol_mode":
        "Volatility estimator for σ: Yang-Zhang, Parkinson, GK, RS, CloseEWMA, etc.",
    "proj.stochastic_vol":
        "Enable mean-reverting stochastic volatility in simulations.",
    "proj.seed_mode":
        "‘fixed’ = reproducible (seed=42). ‘custom’ = enter your own seed.",
    "proj.seed":
        "Custom RNG seed (used when seed_mode is ‘custom’).",

    # ===== Overlays defaults =====
    "overlays.defaults":
        "Overlay names pre-selected by default in Review → Overlays.",

    # ===== Data / Performance stickies =====
    "data.source_choice":
        "Default source shown at the top: Latest generated, Upload, or Path.",
    "data.default_path":
        "Sticky path when using ‘Path’, also used as the hint in the input.",

    # ===== Review: Risk/Reward quick controls =====
    "review.risk_pct":
        "Synthetic stop distance: % below current price when a row stop isn’t used.\n"
        "Higher ↗ = wider stop (smaller size).",
    "review.reward_R":
        "Target multiple of risk (R). Target = Last + R × (Last − Stop).\n"
        "Higher ↗ = farther target (bigger take-profit distance).",
}