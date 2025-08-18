
from dataclasses import dataclass

@dataclass
class ScannerConfig:
    portfolio_value: float = 100_000.0
    risk_per_trade_pct: float = 0.01
    margin_ltv_cap: float = 0.30
    max_open_margin_positions: int = 2
    vix_filter_enabled: bool = True
    vix_threshold: float = 25.0
    lookback_days: int = 420
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_long: int = 200
    sma_mid: int = 50
    vol_avg_period: int = 20
    macd_cross_lookback: int = 5
    sma_deviation_req: float = 0.10
    volume_spike_mult: float = 1.5
    target_gain_pct: float = 0.10
    stop_loss_pct: float = 0.11
    trailing_trigger_gain: float = 0.05
    trailing_pct: float = 0.07
    max_hold_days: int = 60
    holdings_boost_score: int = 1
    top_k_charts: int = 212
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""
    make_charts: bool = True
    attach_charts_in_email: bool = True
    atr_period: int = 14
    atr_multiple: float = 1.5
    ema_fast_span: int = 21
    ema_slow_span: int = 50
    boll_period: int = 20
    boll_std: float = 2.0
    donchian_lookback: int = 20
    rsi_oversold: int = 35
