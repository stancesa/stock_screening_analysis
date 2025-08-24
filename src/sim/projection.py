from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Optional, Any, Dict, Tuple
from collections.abc import Iterable

from scipy.stats import kurtosis

def apply_slippage_px(px: float, side: str, bps: float) -> float:
    m = 1.0 + (bps/1e4) * (+1 if side == "BUY" else -1)
    return float(px * m)

def clip_to_participation(intended_qty: float, next_vol: float, cap: float) -> float:
    max_qty = max(next_vol * cap, 0.0)
    return float(np.sign(intended_qty) * min(abs(intended_qty), max_qty))

def stationary_bootstrap(returns: np.ndarray, T: int, sims: int, p: float, rng: np.random.Generator):
    """
    p is the probability to start a new block each step (expected block length = 1/p).
    Returns matrix of log-returns [T, sims].
    """
    r = np.asarray(returns)
    n = len(r)
    if n == 0:
        return np.zeros((T, sims))
    idx = rng.integers(0, n, size=sims)  # starting indices
    out = np.empty((T, sims))
    for j in range(sims):
        t = 0
        i = idx[j]
        while t < T:
            out[t, j] = r[i]
            t += 1
            if rng.random() < p:  # start a new block
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
    return out

def make_event_mask(future_dates, earnings_dates: list[pd.Timestamp], window=1):
    earn = set(pd.to_datetime(earnings_dates).date)
    mask = np.array([(d.date() in earn) or any((d.date()-pd.Timedelta(days=k)).isoformat()==e.isoformat() for k in range(window+1) for e in []) for d in future_dates])
    return mask

def apply_event_overrides(steps: np.ndarray, event_mask: np.ndarray, *, lam_jump=6.0, mu_jump=-0.03, sig_jump=0.10, rng: np.random.Generator=None):
    """
    steps: base log-return increments [T,N]; modify rows where event_mask=True
    """
    if rng is None:
        rng = np.random.default_rng()
    T, N = steps.shape
    evt_idx = np.where(event_mask)[0]
    for t in evt_idx:
        Nj = rng.poisson(lam=lam_jump / 252.0, size=N)
        J = rng.normal(mu_jump, sig_jump, size=N) * Nj
        steps[t, :] += J
    return steps

def shrink_drift(mu_ann: float, vol_ann: float, n_obs: int, k: float = 10.0) -> float:
    """
    James–Stein style: shrink MU toward 0 by factor depending on vol and sample size.
    Larger vol / fewer obs -> stronger shrink.
    """
    strength = k * (vol_ann / np.sqrt(max(n_obs, 1)))
    w = 1.0 / (1.0 + strength)
    return float(w * mu_ann)

def df_from_kurtosis(r: np.ndarray, df_min=3.0, df_max=30.0):
    if len(r) < 60:
        return 5.0
    ek = float(max(kurtosis(r, fisher=True, bias=False), 0.0))
    # simple mapping: more kurtosis -> smaller df
    df = df_max - (df_max - df_min) * (ek / (ek + 3.0))
    return float(np.clip(df, df_min, df_max))

def compute_beta_and_residuals(stock_close: np.ndarray, mkt_close: np.ndarray, lookback=252):
    s = pd.Series(stock_close, dtype="float64").dropna()
    m = pd.Series(mkt_close, dtype="float64").dropna()
    n = min(len(s), len(m), lookback)
    if n < 60:  # not enough data
        return 1.0, np.log(s).diff().dropna().values  # fallback: beta=1

    rs = np.log(s.iloc[-n:]).diff().dropna().values
    rm = np.log(m.iloc[-n:]).diff().dropna().values
    X = np.c_[np.ones_like(rm), rm]                 # [alpha, beta]
    beta = np.linalg.lstsq(X, rs, rcond=None)[0][1]
    resid = rs - beta * rm
    return float(beta), resid

def residual_vol(resid: np.ndarray) -> float:
    """Annualized vol of residuals."""
    if len(resid) < 2:
        return 0.0
    return float(np.std(resid, ddof=1) * np.sqrt(252.0))

def conformal_quantile_shift(misses: list[int], total: list[int], target=0.80):
    """
    misses[k], total[k] from previous windows for your (low, high) band.
    Returns delta_q to add/subtract to percentile to hit 'target' coverage.
    """
    if not total or sum(total) == 0:
        return 0.0
    realized = 1.0 - (sum(misses) / sum(total))
    # simple proportional correction
    err = target - realized
    return float(np.clip(err * 10.0, -5.0, 5.0))  # shift by up to ±5 percentile points

def isotonic_like_calibration(preds: np.ndarray, outcomes: np.ndarray, bins=10):
    """
    preds in [0,1], outcomes in {0,1}. Returns (bin_edges, calibrated_vals).
    """
    edges = np.linspace(0,1,bins+1)
    cal = np.zeros(bins)
    for i in range(bins):
        m = (preds>=edges[i]) & (preds<edges[i+1])
        cal[i] = outcomes[m].mean() if m.any() else (edges[i]+edges[i+1])/2
    return edges, cal

def apply_calibration(p: float, edges, cal_vals):
    idx = np.searchsorted(edges, p, side="right") - 1
    idx = int(np.clip(idx, 0, len(cal_vals)-1))
    return float(np.clip(cal_vals[idx], 0.0, 1.0))

def winsorize_returns(r: np.ndarray, lo=1.0, hi=99.0):
    a, b = np.percentile(r, [lo, hi])
    return np.clip(r, a, b)

def crps_from_quantiles(y_true: float, qs: np.ndarray, qvals: np.ndarray):
    """
    qs: sorted quantile levels in [0,1], qvals: corresponding values
    Approximate CRPS (lower is better).
    """
    # piecewise linear CDF
    diffs = np.diff(qvals)
    probs = np.diff(qs)
    # integral of (F - 1{y<=x})^2 dx; here a simple Riemann approximation:
    below = qvals <= y_true
    F = np.interp(y_true, qvals, qs, left=0.0, right=1.0)
    return float((F*(1-F)))  # cheap proxy; use full CRPS formula if desired


def combine_market_and_residual_paths(last_px: float, market_paths: np.ndarray, resid_paths: np.ndarray, beta: float) -> np.ndarray:
    """
    market_paths, resid_paths: arrays of log-returns per step [T,N] (not price levels!)
    Returns price paths [T,N].
    """
    steps = beta * market_paths + resid_paths
    return last_px * np.exp(np.cumsum(steps, axis=0))

def adaptive_half_life_from_vol(vol_ann: float, vol_hist: np.ndarray, tau_min=6.0, tau_max=20.0) -> float:
    """Map current annualized vol to a half-life in [tau_min, tau_max]; higher vol -> shorter half-life."""
    if len(vol_hist) < 40:
        return 10.0
    p = (vol_ann <= np.percentile(vol_hist, [0,100])).nonzero()  # placeholder to avoid errors
    prc = float((vol_ann - np.min(vol_hist)) / (np.max(vol_hist) - np.min(vol_hist) + 1e-12))
    prc = np.clip(prc, 0.0, 1.0)
    return float(tau_max - (tau_max - tau_min) * prc)

def _annual_drift_from_sma(
    sma_series: np.ndarray,
    lookback: int = 200,
    min_points: int = 60,
) -> float:
    """
    Estimate annualized drift from the slope of log(SMA) via OLS.
    Returns mu_annual (e.g., 0.08 = +8%/yr). Falls back to 0.0 if not enough data.
    """
    s = pd.Series(sma_series, dtype="float64").dropna()
    if len(s) < max(min_points, 5):
        return 0.0
    s = s.iloc[-min(lookback, len(s)):]
    y = np.log(s.values)
    x = np.arange(len(y), dtype="float64")
    # robust to constant series / nans
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=np.nanmedian(y))
    if not np.isfinite(x).all():
        return 0.0
    # slope per trading day
    try:
        slope_per_day = np.polyfit(x, y, 1)[0]
    except Exception:
        return 0.0
    # annualize (252 trading days)
    mu_annual = float(slope_per_day * 252.0)
    return mu_annual

def _ann_sigma_from_estimator(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    open_: Optional[np.ndarray] = None,
    mode: str = "EWMA",
    lam: float = 0.94,
    window: int = 252,
) -> float:
    """
    Return annualized volatility (sigma) using close-only or OHLC estimators.

    Modes (case-insensitive):
      Close-only: "EWMA", "ROLLING", "MAD"
      OHLC-based: "PARKINSON", "GK" (Garman–Klass), "RS" (Rogers–Satchell), "YANGZHANG" (a.k.a. "YZ")

    Notes:
      - Applies 'window' to the most recent data.
      - Annualizes with sqrt(252).
      - Falls back safely to close-only EWMA if OHLC not available/valid.
    """
    # --- prepare close & close-to-close returns
    c_full = pd.Series(close, dtype="float64").dropna()
    if len(c_full) < 2:
        return 0.0

    if window and len(c_full) > window:
        c_full = c_full[-window:]

    r_cc = np.log(c_full).diff().dropna().values
    if len(r_cc) == 0:
        return 0.0

    m = (mode or "EWMA").strip().lower()

    def ann_from_daily_var(var_daily: float) -> float:
        var_daily = float(max(var_daily, 0.0))
        return float(np.sqrt(var_daily) * np.sqrt(252.0))

    # ---- Close-only estimators ----
    if m == "ewma":
        var = pd.Series(r_cc).ewm(alpha=(1 - lam), adjust=False, min_periods=10).var().iloc[-1]
        if not np.isfinite(var) or var <= 0:
            var = np.var(r_cc, ddof=1)
        return ann_from_daily_var(var)

    if m == "rolling":
        rr = pd.Series(r_cc)
        var = rr.rolling(20, min_periods=10).var().iloc[-1]
        if not np.isfinite(var) or var <= 0:
            var = np.var(r_cc, ddof=1)
        return ann_from_daily_var(var)

    if m == "mad":
        mabs = float(np.mean(np.abs(r_cc)))
        var = (np.pi / 2.0) * (mabs ** 2)
        return ann_from_daily_var(var)

    # ---- OHLC-based estimators (need O/H/L) ----
    need_ohlc = m in {"parkinson", "gk", "garman-klass", "garmanklass", "rs", "rogers-satchell", "yangzhang", "yang-zhang", "yz"}
    have_ohlc = (open_ is not None) and (high is not None) and (low is not None)

    if need_ohlc and have_ohlc:
        # Align arrays to a common recent length and clean
        o = pd.Series(np.asarray(open_, dtype="float64")).dropna()
        h = pd.Series(np.asarray(high, dtype="float64")).dropna()
        l = pd.Series(np.asarray(low,  dtype="float64")).dropna()
        c = c_full.copy()

        n = min(len(o), len(h), len(l), len(c))
        if n >= 2:
            o = o.iloc[-n:].to_numpy()
            h = h.iloc[-n:].to_numpy()
            l = l.iloc[-n:].to_numpy()
            c = c.iloc[-n:].to_numpy()

            mask = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (o > 0) & (h > 0) & (l > 0) & (c > 0)
            o, h, l, c = o[mask], h[mask], l[mask], c[mask]
            n = len(c)

            if n >= 2:
                hl = np.log(h / l)             # intraday range
                co = np.log(c / o)             # open->close
                c_prev = np.roll(c, 1)
                overnight = np.log(o / c_prev) # close_prev->open (overnight gap)

                # drop first element to align deltas
                hl = hl[1:]; co = co[1:]; overnight = overnight[1:]
                o1 = o[1:]; h1 = h[1:]; l1 = l[1:]; c1 = c[1:]

                if m == "parkinson":
                    var = float(np.mean(hl**2) / (4.0 * np.log(2.0)))
                    return ann_from_daily_var(var)

                if m in {"gk", "garman-klass", "garmanklass"}:
                    var = float(np.mean(0.5 * (hl**2) - (2.0 * np.log(2.0) - 1.0) * (co**2)))
                    return ann_from_daily_var(var)

                if m in {"rs", "rogers-satchell"}:
                    ho = np.log(h1 / o1); hc = np.log(h1 / c1)
                    lo = np.log(l1 / o1); lc = np.log(l1 / c1)
                    rs = ho * hc + lo * lc
                    var = float(np.mean(rs))
                    return ann_from_daily_var(var)

                if m in {"yangzhang", "yang-zhang", "yz"}:
                    # Yang–Zhang decomposition
                    n_yz = len(c1)
                    if n_yz >= 2:
                        # sample variances for components
                        sigma_o2 = float(np.var(overnight, ddof=1)) if len(overnight) > 1 else float(np.var(overnight))
                        sigma_c2 = float(np.var(co, ddof=1))        if len(co) > 1        else float(np.var(co))
                        ho = np.log(h1 / o1); hc = np.log(h1 / c1)
                        lo = np.log(l1 / o1); lc = np.log(l1 / c1)
                        rs = ho * hc + lo * lc
                        sigma_rs2 = float(np.mean(rs))
                        # k weight
                        k = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1)) if n_yz > 1 else 0.34
                        var = sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2
                        return ann_from_daily_var(var)
                # If we fell through (unknown OHLC mode), drop to fallback below.

    # ---- Fallbacks ----
    # If an OHLC mode was requested but OHLC not available/invalid -> EWMA on closes
    rr = pd.Series(r_cc)
    var = rr.ewm(alpha=(1 - lam), adjust=False, min_periods=10).var().iloc[-1]
    if not np.isfinite(var) or var <= 0:
        var = np.var(r_cc, ddof=1)
    return ann_from_daily_var(var)

def _simulate_paths_for_projection(
    *,
    y_close: np.ndarray,
    start_date: pd.Timestamp,
    model: str = "EWMA+t",
    sims: int = 2000,
    horizon_days: int = 22,
    horizon_months: Optional[int] = None,
    # volatility estimator
    vol_mode: str = "EWMA",
    window: int = 252,
    lam: float = 0.94,
    df_t: int = 5,
    antithetic: bool = False,
    stochastic_vol: bool = False,
    block: int = 5, 
    # OHLC
    y_open: Optional[np.ndarray] = None,
    y_high: Optional[np.ndarray] = None,
    y_low:  Optional[np.ndarray] = None,
    # SMA drift inputs
    y_sma_short_for_drift: Optional[np.ndarray] = None,  # e.g., SMA20
    y_sma_long_for_drift:  Optional[np.ndarray] = None,  # e.g., SMA200
    use_sma_drift: bool = True,
    sma_short_weight: float = 0.4,   # used only if both SMA series missing
    sma_long_weight:  float = 0.6,
    risk_free_ann: float = 0.015,
    drift_cap_ann: float = 0.40,
    # schedule (kept internal to avoid breaking your signature)
    _sma_decay_half_life_days: float = 10.0,
    _short_floor: float = 0.10,
    _long_ceiling: float = 0.90,
    # OHLC initial kick
    _ohlc_k_gap: float = 1.0,
    _ohlc_k_intraday: float = 1.0,
    _ohlc_k_range: float = 0.10,
    # NEW optional conditioning / events
    market_close: Optional[np.ndarray] = None,
    earnings_dates: Optional[Iterable] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      future_dates, paths [T,N], w_short[T], w_long[T], extras{sigma_ann, mu_ann_t}
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    close = pd.Series(y_close, dtype="float64").dropna()

    # ---- build future business days
    if horizon_months and horizon_months > 0:
        end_dt = start_date + pd.offsets.BMonthEnd(horizon_months)
        future = pd.bdate_range(start=start_date + pd.offsets.BDay(1), end=end_dt)
    else:
        T_days = int(horizon_days if horizon_days and horizon_days > 0 else 22)
        future = pd.bdate_range(start=start_date, periods=T_days + 1)[1:]

    T = len(future)
    if len(close) < 20 or T == 0:
        last = float(close.iloc[-1]) if len(close) else 0.0
        paths = np.full((T, sims), last, dtype="float64")
        w_short = np.ones(T, dtype="float64")
        w_long  = np.zeros(T, dtype="float64")
        extras = {"sigma_ann": 0.0, "mu_ann_t": np.zeros(T, dtype="float64")}
        return future, paths, w_short, w_long, extras

    # ---- returns window (winsorized for stability)
    r_full = np.log(close).diff().dropna().values
    r = r_full[-window:] if len(r_full) > window else r_full
    r_w = winsorize_returns(r, 1.0, 99.0) if len(r) else r
    last = float(close.iloc[-1])
    N = int(sims)

    # ---- vol series for adaptive half-life
    vol_hist_ann = (pd.Series(r_full).rolling(20, min_periods=10).std() * np.sqrt(252.0)).tail(252).dropna().values
    vol_ann_now = float(vol_hist_ann[-1]) if vol_hist_ann.size else 0.0
    tau = adaptive_half_life_from_vol(vol_ann_now, vol_hist_ann, tau_min=6.0, tau_max=20.0) if vol_hist_ann.size else max(1.0, float(_sma_decay_half_life_days))

    # ---- helpers
    def _annual_drift_from_sma_local(sma_series: np.ndarray, lookback: int = 200, min_points: int = 60) -> float:
        s = pd.Series(sma_series, dtype="float64").dropna()
        if len(s) < max(min_points, 5): return 0.0
        s = s.iloc[-min(lookback, len(s)):]
        y = np.log(s.values); x = np.arange(len(y), dtype="float64")
        try:
            slope_per_day = np.polyfit(x, y, 1)[0]
        except Exception:
            return 0.0
        return float(slope_per_day * 252.0)

    def _sma_slope(series: np.ndarray, window_: int) -> float:
        if series is None or len(series) < window_: return 0.0
        y = pd.Series(series[-window_:], dtype="float64")
        if not np.isfinite(y.iloc[-1]) or y.iloc[-1] <= 0: return 0.0
        x = np.arange(len(y)); slope, _ = np.polyfit(x, y, 1)
        return float(slope / y.iloc[-1])  # % per day

    # ---- DRIFT signals (with shrinkage)
    sma20_slope = _sma_slope(y_close, window_=20)      # %/day
    sma200_slope = _sma_slope(y_close, window_=200)    # %/day
    mu_slope_short_ann = shrink_drift(sma20_slope * 252.0,  vol_ann_now, len(r_w))
    mu_slope_long_ann  = shrink_drift(sma200_slope * 252.0, vol_ann_now, len(r_w))

    mu_level_short_ann = 0.0
    mu_level_long_ann  = 0.0
    if use_sma_drift and y_sma_short_for_drift is not None:
        mu_level_short_ann = shrink_drift(_annual_drift_from_sma_local(y_sma_short_for_drift, lookback=min(90, window),  min_points=30),
                                          vol_ann_now, len(r_w))
    if use_sma_drift and y_sma_long_for_drift is not None:
        mu_level_long_ann  = shrink_drift(_annual_drift_from_sma_local(y_sma_long_for_drift,  lookback=min(252, window), min_points=60),
                                          vol_ann_now, len(r_w))
    mu_hist_ann = shrink_drift(float(np.mean(r_w) * 252.0) if len(r_w) > 5 else 0.0, vol_ann_now, len(r_w))

    # ---- TIME-VARYING WEIGHTS: SMA20 -> SMA200 (adaptive half-life)
    t_idx = np.arange(T, dtype="float64")
    w_short = _short_floor + (1.0 - _short_floor) * np.exp(-(t_idx / tau))
    w_long  = np.minimum(1.0 - w_short, _long_ceiling)

    # normalize if SMAs missing
    if (y_sma_short_for_drift is None) and (y_sma_long_for_drift is not None):
        w_short[:] = 0.0; w_long[:] = np.minimum(1.0, _long_ceiling)
    elif (y_sma_long_for_drift is None) and (y_sma_short_for_drift is not None):
        w_short[:] = 1.0; w_long[:] = 0.0
    elif (y_sma_short_for_drift is None) and (y_sma_long_for_drift is None):
        w_short[:] = float(np.clip(sma_short_weight, 0.0, 1.0))
        w_long[:]  = float(np.clip(sma_long_weight,  0.0, 1.0))

    mu_ann_t = (
        0.50 * (w_short * mu_slope_short_ann + w_long * mu_slope_long_ann) +
        0.30 * (w_short * mu_level_short_ann + w_long * mu_level_long_ann) +
        0.10 * mu_hist_ann +
        0.10 * risk_free_ann
    )
    mu_ann_t = np.clip(mu_ann_t, -drift_cap_ann, drift_cap_ann)
    mu_d_t = (mu_ann_t * dt).reshape(-1, 1)  # (T,1)

    # ---- volatility level (uses your estimator)
    sigma_ann = _ann_sigma_from_estimator(
        close=y_close, high=y_high, low=y_low, open_=y_open,
        mode=vol_mode, lam=lam, window=window
    )
    sigma_d0 = float(sigma_ann * np.sqrt(dt))

    # ---- heavy-tail df (auto) if using EWMA+t
    m_lower = (model or "EWMA+t").strip().lower()
    df_t_local = int(round(df_from_kurtosis(r_w))) if m_lower == "ewma+t" else df_t

    # ---- OHLC initial kick (deterministic bias to step 0)
    kick0 = 0.0
    try:
        if y_open is not None and y_high is not None and y_low is not None and len(y_close) >= 2:
            o_t = float(pd.Series(y_open, dtype="float64").dropna().iloc[-1])
            h_t = float(pd.Series(y_high, dtype="float64").dropna().iloc[-1])
            l_t = float(pd.Series(y_low,  dtype="float64").dropna().iloc[-1])
            c_t = float(close.iloc[-1]); c_tm1 = float(close.iloc[-2])
            if min(o_t, h_t, l_t, c_t, c_tm1) > 0:
                gap = np.log(o_t / c_tm1)
                intraday = np.log(c_t / o_t)
                rng_log = np.log(h_t / l_t)
                direction = np.sign(intraday) if intraday != 0.0 else 0.0
                kick0 = (_ohlc_k_gap * gap) + (_ohlc_k_intraday * intraday) + (_ohlc_k_range * direction * rng_log)
                kick0 = float(np.clip(kick0, -3.0 * sigma_d0, 3.0 * sigma_d0))
    except Exception:
        kick0 = 0.0

    # ---- antithetic draws
    def _antithetic_norm(T_, N_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_normal((T_, half))
        return np.concatenate([Z_half, -Z_half], axis=1)[:, :N_]

    def _antithetic_t(T_, N_, df_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_t(df_, size=(T_, half)) / np.sqrt(df_ / (df_ - 2))
        return np.concatenate([Z_half, -Z_half], axis=1)[:, :N_]

    # ==========================
    # Simulate steps / paths
    # ==========================
    use_market_cond = (market_close is not None) and (len(pd.Series(market_close).dropna()) >= 60) and (m_lower in {"gbm","ewma+t"})
    if use_market_cond:
        beta, resid = compute_beta_and_residuals(y_close, market_close, lookback=window)
        mkt = pd.Series(market_close, dtype="float64").dropna()

        # Market drift schedule (price-based slopes on market, shrunk)
        def _sma_slope_m(series: pd.Series, window_: int) -> float:
            if len(series) < window_: return 0.0
            y = series.iloc[-window_:].values
            if y[-1] <= 0: return 0.0
            x = np.arange(window_); slope, _ = np.polyfit(x, y, 1)
            return float(slope / y[-1])

        mkt_r = np.log(mkt).diff().dropna().values
        mkt_r_w = winsorize_returns(mkt_r, 1.0, 99.0) if len(mkt_r) else mkt_r
        mkt_vol_hist_ann = (pd.Series(mkt_r).rolling(20, min_periods=10).std() * np.sqrt(252.0)).tail(252).dropna().values
        mkt_vol_ann_now = float(mkt_vol_hist_ann[-1]) if mkt_vol_hist_ann.size else 0.0
        tau_m = adaptive_half_life_from_vol(mkt_vol_ann_now, mkt_vol_hist_ann, tau_min=6.0, tau_max=20.0) if mkt_vol_hist_ann.size else tau

        m20  = shrink_drift(_sma_slope_m(mkt, 20)  * 252.0, mkt_vol_ann_now, len(mkt_r_w))
        m200 = shrink_drift(_sma_slope_m(mkt, 200) * 252.0, mkt_vol_ann_now, len(mkt_r_w))
        m_hist_ann = shrink_drift(float(np.mean(mkt_r_w) * 252.0) if len(mkt_r_w) > 5 else 0.0, mkt_vol_ann_now, len(mkt_r_w))

        t_idx = np.arange(T, dtype=float)
        w_s_m = _short_floor + (1.0 - _short_floor) * np.exp(-(t_idx / tau_m))
        w_l_m = np.minimum(1.0 - w_s_m, _long_ceiling)
        mu_ann_t_mkt = (0.50*(w_s_m*m20 + w_l_m*m200) + 0.30*(w_s_m*m20 + w_l_m*m200) + 0.10*m_hist_ann + 0.10*risk_free_ann)
        mu_ann_t_mkt = np.clip(mu_ann_t_mkt, -drift_cap_ann, drift_cap_ann)
        mu_d_t_mkt = (mu_ann_t_mkt * dt).reshape(-1, 1)

        # Vol terms
        sigma_ann_mkt = _ann_sigma_from_estimator(close=mkt.values, mode=vol_mode, lam=lam, window=window)
        sigma_d_mkt = float(sigma_ann_mkt * np.sqrt(dt))
        sigma_ann_resid = residual_vol(resid)
        sigma_d_resid = float(sigma_ann_resid * np.sqrt(dt))

        # Residual drift so total drift matches base
        mu_d_t_resid = mu_d_t - beta * mu_d_t_mkt

        # Draws
        if m_lower == "gbm":
            Zm = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            Zr = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            steps_mkt   = (mu_d_t_mkt - 0.5*(sigma_d_mkt**2)) + sigma_d_mkt * Zm
            steps_resid = (mu_d_t_resid - 0.5*(sigma_d_resid**2)) + sigma_d_resid * Zr
        else:  # "EWMA+t"
            Zm = (_antithetic_t(T, N, df_t_local) if antithetic
                  else (rng.standard_t(df_t_local, size=(T, N)) / np.sqrt(df_t_local / (df_t_local - 2))))
            Zr = (_antithetic_t(T, N, df_t_local) if antithetic
                  else (rng.standard_t(df_t_local, size=(T, N)) / np.sqrt(df_t_local / (df_t_local - 2))))
            steps_mkt   = (mu_d_t_mkt - 0.5*(sigma_d_mkt**2)) + sigma_d_mkt * Zm
            steps_resid = (mu_d_t_resid - 0.5*(sigma_d_resid**2)) + sigma_d_resid * Zr

        steps = steps_mkt + steps_resid

    else:
        # ----- Single-leg simulation -----
        if m_lower == "gbm":
            Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            if stochastic_vol:
                rr2 = r_w**2
                rho = 0.8
                if len(rr2) >= 30:
                    c = np.corrcoef(rr2[:-1], rr2[1:])[0, 1]
                    rho = 0.8 if not np.isfinite(c) else float(np.clip(c, 0.0, 0.98))
                v0 = sigma_d0**2
                sig = np.empty((T, N)); sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_d_t - 0.5 * (sig**2)) + sig * Z
            else:
                steps = (mu_d_t - 0.5 * (sigma_d0**2)) + sigma_d0 * Z

        elif m_lower == "ewma+t":
            Z = (_antithetic_t(T, N, df_t_local) if antithetic
                 else (rng.standard_t(df_t_local, size=(T, N)) / np.sqrt(df_t_local / (df_t_local - 2))))
            if stochastic_vol:
                rho = 0.85
                v0 = sigma_d0**2
                sig = np.empty((T, N)); sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_d_t - 0.5 * (sig**2)) + sig * Z
            else:
                steps = (mu_d_t - 0.5 * (sigma_d0**2)) + sigma_d0 * Z

        elif m_lower == "bootstrap":
            # Stationary bootstrap (expected block length = 'block')
            p = 1.0 / max(int(block), 2)
            boot = stationary_bootstrap(r_full, T, N, p=p, rng=rng)
            steps = boot + mu_d_t  # tilt by time-varying drift

        elif m_lower == "jump":
            Z = rng.standard_normal((T, N))
            sigma_d = float(np.std(r_w, ddof=1) * np.sqrt(dt)) if len(r_w) > 1 else sigma_d0
            mu_j, sig_j, lam_j = -0.02, 0.06, 3.0
            Njump = rng.poisson(lam=(lam_j * dt), size=(T, N))
            J = rng.normal(mu_j, sig_j, size=(T, N)) * Njump
            steps = (mu_d_t - 0.5 * (sigma_d**2)) + sigma_d * Z + J

        else:
            Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            steps = (mu_d_t - 0.5 * (sigma_d0**2)) + sigma_d0 * Z

    # Event-aware jumps (if provided)
    if earnings_dates is not None:
        mask = make_event_mask(future, earnings_dates)
        steps = apply_event_overrides(steps, mask, rng=rng)

    # OHLC kick on first step
    steps[0, :] += kick0

    paths = last * np.exp(np.cumsum(steps, axis=0))
    extras = {"sigma_ann": sigma_ann, "mu_ann_t": mu_ann_t}
    return future, paths, w_short, w_long, extras

def _direction_from_paths(paths: np.ndarray, last: float) -> Dict[str, float | str]:
    """
    Compute UP/DOWN/SIDEWAYS, confidence, and diagnostics from simulated paths [T,N].
    """
    T, N = paths.shape
    med_path = np.median(paths, axis=1)
    p_up_close = float((paths[-1, :] > last).mean())

    x = np.arange(T, dtype=float)
    slope_per_day = np.polyfit(x, np.log(np.clip(med_path, 1e-12, None)), 1)[0]
    slope_med_ann = float(slope_per_day * 252.0)

    step_up = np.diff(med_path) > 0
    consistency = float(step_up.mean()) if len(step_up) else 0.5

    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    score = (0.55 * p_up_close) + (0.25 * consistency) + (0.20 * _sigmoid(slope_med_ann / 0.25))

    if score >= 0.65:
        label, conf = "UP", ("High" if score >= 0.75 else "Medium")
    elif score <= 0.35:
        label, conf = "DOWN", ("High" if score <= 0.25 else "Medium")
    else:
        label, conf = "SIDEWAYS", "Low"

    return {
        "p_up_close": p_up_close,
        "slope_med_ann": slope_med_ann,
        "consistency": consistency,
        "score": float(np.clip(score, 0.0, 1.0)),
        "label": label,
        "confidence": conf,
    }

def run_projection_ensemble(
    *,
    y_close: np.ndarray,
    start_date: pd.Timestamp,
    sims: int = 2000,
    horizon_days: int = 22,
    y_open: Optional[np.ndarray] = None,
    y_high: Optional[np.ndarray] = None,
    y_low:  Optional[np.ndarray] = None,
    y_sma20: Optional[np.ndarray] = None,
    y_sma200: Optional[np.ndarray] = None,
    market_close: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    models = ["EWMA+t", "GBM", "Bootstrap"]
    outs = []
    for i, m in enumerate(models):
        future, paths, w_s, w_l, ex = _simulate_paths_for_projection(
            y_close=y_close, start_date=start_date, model=m, sims=sims, horizon_days=horizon_days,
            y_open=y_open, y_high=y_high, y_low=y_low,
            y_sma_short_for_drift=y_sma20, y_sma_long_for_drift=y_sma200,
            market_close=market_close,
            seed=(None if seed is None else seed + i)
        )
        outs.append((m, future, paths, w_s, w_l, ex))

    # choose EWMA+t as primary display set
    _, future, paths_primary, w_s, w_l, _ = next(o for o in outs if o[0] == "EWMA+t")
    low  = np.nanpercentile(paths_primary, 10.0, axis=1)
    high = np.nanpercentile(paths_primary, 90.0, axis=1)
    med  = np.nanpercentile(paths_primary, 50.0, axis=1)

    last = float(pd.Series(y_close, dtype="float64").dropna().iloc[-1])
    model_diags = {m: _direction_from_paths(p, last) for (m, _, p, _, _, _) in outs}

    # ensemble averaging
    p_up = float(np.mean([d["p_up_close"] for d in model_diags.values()]))
    slope = float(np.mean([d["slope_med_ann"] for d in model_diags.values()]))
    cons  = float(np.mean([d["consistency"] for d in model_diags.values()]))
    score = float(np.mean([d["score"] for d in model_diags.values()]))

    flags = compute_regime_flags(y_close, market_close=market_close)
    score_adj = regime_adjust(score, flags)
    label = ("UP" if score_adj>=0.65 else "DOWN" if score_adj<=0.35 else "SIDEWAYS")
    conf  = ("High" if score_adj>=0.75 or score_adj<=0.25 else
             "Medium" if (score_adj>=0.60 or score_adj<=0.40) else "Low")
    ensemble = {"p_up_close": p_up, "slope_med_ann": slope, "consistency": cons, "score": score_adj, "label": label, "confidence": conf}

    return {
        "future": future,
        "median": med,
        "low": low,
        "high": high,
        "paths_primary": paths_primary,
        "diag_primary": _direction_from_paths(paths_primary, last),
        "ensemble": ensemble,
        "model_diags": model_diags,
        "weights": {"short": w_s, "long": w_l},
        "flags": flags,
    }

def compute_regime_flags(
    y_close: np.ndarray,
    market_close: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Simple regime flags for confidence adjustment (no plotting).
    """
    s = pd.Series(y_close, dtype="float64").dropna()
    r = np.log(s).diff().dropna()
    sig_series = r.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    vol_ann = float(sig_series.iloc[-1]) if len(sig_series) else 0.0
    vol_thresh_hi = float(sig_series.quantile(0.80)) if len(sig_series) else vol_ann

    sma200 = s.rolling(200, min_periods=50).mean()
    below_sma200 = bool(len(s) >= 50 and s.iloc[-1] < (sma200.iloc[-1] if pd.notna(sma200.iloc[-1]) else s.iloc[-1]))

    market_down = False
    if market_close is not None and len(market_close) >= 200:
        m = pd.Series(market_close, dtype="float64").dropna()
        m_sma200 = m.rolling(200, min_periods=50).mean()
        x = np.arange(min(len(m), 60))
        m_tail = m.iloc[-len(x):]
        try:
            slope_pd = np.polyfit(x, np.log(np.clip(m_tail, 1e-12, None)), 1)[0]
            m_slope_ann = float(slope_pd * 252.0)
        except Exception:
            m_slope_ann = 0.0
        market_down = bool((m_slope_ann < 0) and (m.iloc[-1] < (m_sma200.iloc[-1] if pd.notna(m_sma200.iloc[-1]) else m.iloc[-1])))

    return {"vol_ann": vol_ann, "vol_thresh_hi": vol_thresh_hi, "below_sma200": below_sma200, "market_down": market_down}

def regime_adjust(score: float, flags: Dict[str, Any]) -> float:
    """
    Shrink score toward 0.5 in hostile regimes (high vol / below SMA200 + weak market).
    """
    vol_ann = flags.get("vol_ann", 0.0)
    vol_hi  = flags.get("vol_thresh_hi", vol_ann)
    below_sma200 = bool(flags.get("below_sma200", False))
    market_down  = bool(flags.get("market_down", False))

    if vol_ann > vol_hi:
        score = 0.5 + 0.6*(score - 0.5)
    if below_sma200 and market_down:
        score = 0.5 + 0.7*(score - 0.5)
    return float(np.clip(score, 0.0, 1.0))


def _project_next_month(
    y_close: np.ndarray,
    start_date: pd.Timestamp,
    horizon_days: int = 22,
    sims: int = 2000,
    pct_low: float = 10.0,
    pct_high: float = 90.0,
    model: str = "EWMA+t",
    seed: Optional[int] = None,
    # tuning knobs
    window: int = 252,
    lam: float = 0.94,
    df_t: int = 5,
    antithetic: bool = False,
    block: int = 5,
    horizon_months: Optional[int] = None,
    vol_mode: str = "EWMA",
    y_high: Optional[np.ndarray] = None,
    y_low: Optional[np.ndarray] = None,
    y_open: Optional[np.ndarray] = None,
    stochastic_vol: bool = False,
    # drift from SMA (short/long)
    y_sma_short_for_drift: Optional[np.ndarray] = None,
    y_sma_long_for_drift: Optional[np.ndarray]  = None,
    use_sma_drift: bool = True,
    sma_short_weight: float = 0.4,
    sma_long_weight: float  = 0.6,
    risk_free_ann: float = 0.015,
    drift_cap_ann: float = 0.40,
    # NEW optional extras (all default to None; safe no-ops)
    market_close: Optional[np.ndarray] = None,
    earnings_dates: Optional[Iterable] = None,
    conformal_stats: Optional[Dict[str, Any]] = None,   # {'misses': int, 'total': int, 'target': 0.80}
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project price paths and return (future_dates, median, low, high).
    """
    future, paths, _w_s, _w_l, _ex = _simulate_paths_for_projection(
        y_close=y_close,
        start_date=start_date,
        model=model,
        sims=sims,
        horizon_days=horizon_days,
        horizon_months=horizon_months,
        vol_mode=vol_mode,
        window=window,
        lam=lam,
        df_t=df_t,
        antithetic=antithetic,
        stochastic_vol=stochastic_vol,
        y_open=y_open,
        y_high=y_high,
        y_low=y_low,
        y_sma_short_for_drift=y_sma_short_for_drift,
        y_sma_long_for_drift=y_sma_long_for_drift,
        use_sma_drift=use_sma_drift,
        sma_short_weight=sma_short_weight,
        sma_long_weight=sma_long_weight,
        risk_free_ann=risk_free_ann,
        drift_cap_ann=drift_cap_ann,
        market_close=market_close,
        earnings_dates=earnings_dates,
        seed=seed,
    )

    # Conformal tweak to hit target coverage on average (optional)
    q_low, q_high = float(pct_low), float(pct_high)
    if conformal_stats and isinstance(conformal_stats, dict):
        target = float(conformal_stats.get("target", 0.80))
        misses = [int(conformal_stats.get("misses", 0))]
        total  = [int(conformal_stats.get("total", 0))]
        if total[0] > 0:
            delta = conformal_quantile_shift(misses, total, target=target)
            q_low  = float(np.clip(q_low  - abs(delta), 0.0, 50.0))
            q_high = float(np.clip(q_high + abs(delta), 50.0, 100.0))

    low  = np.nanpercentile(paths, q_low,  axis=1)
    high = np.nanpercentile(paths, q_high, axis=1)
    med  = np.nanpercentile(paths, 50.0,   axis=1)
    return future, med, low, high