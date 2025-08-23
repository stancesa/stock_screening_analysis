from dataclasses import dataclass

@dataclass
class ExecParams:
    use_next_open: bool = True
    slippage_bps: float = 10.0          # 10 bps each side
    commission_per_share: float = 0.0   # $/share
    max_participation: float = 0.10     # ≤10% of next-day volume
    allow_shorts: bool = False
    borrow_bps_annual: float = 0.0

@dataclass
class BuyParams:
    # BUY weights (all ∈ [0,1]; final score is clipped to [0,1])
    w_rsi: float = 0.20
    w_trend: float = 0.20
    w_value: float = 0.10
    w_flow: float = 0.05
    w_bbands: float = 0.20          # NEW: Bollinger %B (oversold near lower band)
    w_donchian: float = 0.25        # NEW: Donchian breakout (ATR-scaled)
    w_breakout: float = 0.00        # (kept for backward compat, default 0 to avoid double-counting)

    composite_threshold: float = 0.60

    # BUY feature centers / knobs
    rsi_buy_max: float = 45.0
    rsi_floor: float = 20.0
    sma200_window: int = 200
    donch_lookback: int = 20
    gap_min_pct: float = 0.5
    value_center_dev_pct: float = -5.0
    vol_ratio_min: float = 1.50

    # Bollinger settings (shared)
    bb_window: int = 20             
    bb_k: float = 2.0                

    # Stops / sizing
    use_engine_stop: bool = True
    atr_mult: float = 1.5
    stop_pct: float = 10.0
    reward_R: float = 2.0

    portfolio_value: float = 20000.0
    risk_per_trade_pct: float = 0.5
    min_price: float = 1.0
    min_adv_dollars: float = 250_000.0

    # SELL weights
    w_rsi_sell: float = 0.30
    w_trend_down: float = 0.30
    w_breakdown: float = 0.00        
    w_exhaustion: float = 0.10
    w_flow_out: float = 0.05
    w_bbands_sell: float = 0.25      
    w_donchian_sell: float = 0.25    

    sell_threshold: float = 0.60

    # SELL features
    rsi_overbought_min: float = 70.0
    ema_fast_span: int = 21
    sma_mid_window: int = 50
    donch_lookback_sell: int = 20
    gap_down_min_pct: float = 0.5