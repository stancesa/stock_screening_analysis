from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Optional

def apply_slippage_px(px: float, side: str, bps: float) -> float:
    m = 1.0 + (bps/1e4) * (+1 if side == "BUY" else -1)
    return float(px * m)

def clip_to_participation(intended_qty: float, next_vol: float, cap: float) -> float:
    max_qty = max(next_vol * cap, 0.0)
    return float(np.sign(intended_qty) * min(abs(intended_qty), max_qty))

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
    vol_mode: str = "EWMA",                 # "EWMA","ROLLING","MAD","PARKINSON","GK","RS","YANGZHANG"
    y_high: Optional[np.ndarray] = None,
    y_low: Optional[np.ndarray] = None,
    y_open: Optional[np.ndarray] = None,
    stochastic_vol: bool = False,
    # drift from SMA (short/long)
    y_sma_short_for_drift: Optional[np.ndarray] = None,  # e.g., SMA20
    y_sma_long_for_drift: Optional[np.ndarray]  = None,  # e.g., SMA200
    use_sma_drift: bool = True,
    sma_short_weight: float = 0.4,   # weight on SMA20 slope/level
    sma_long_weight: float  = 0.6,   # weight on SMA200 slope/level
    risk_free_ann: float = 0.015,
    drift_cap_ann: float = 0.40,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project price paths and return (future_dates, median, low, high).
    `vol_mode` supports OHLC estimators; pass y_open/y_high/y_low to activate.
    Drift uses a blend of SMA-slope (from price), SMA-level slopes (from SMA20/200),
    plus small historical + risk-free components.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    close = pd.Series(y_close, dtype="float64").clip(lower=1e-9)

    # --- future business dates
    if horizon_months and horizon_months > 0:
        end_dt = start_date + pd.offsets.BMonthEnd(horizon_months)
        future = pd.bdate_range(start=start_date + pd.offsets.BDay(1), end=end_dt)
    else:
        T_days = int(horizon_days if horizon_days and horizon_days > 0 else 22)
        future = pd.bdate_range(start=start_date, periods=T_days + 1)[1:]

    T = len(future)
    if len(close) < 20 or T == 0:
        last = float(close.iloc[-1])
        med = np.full(T, last)
        return future, med, med, med

    # recent log returns (close-to-close)
    r = np.log(close).diff().dropna().values
    r = r[-window:] if len(r) > window else r
    last = float(close.iloc[-1])
    N = int(sims)

    # --- helpers
    def _annual_drift_from_sma_local(sma_series: np.ndarray, lookback: int = 200, min_points: int = 60) -> float:
        """Annualized drift from slope of log(SMA) via OLS."""
        s = pd.Series(sma_series, dtype="float64").dropna()
        if len(s) < max(min_points, 5):
            return 0.0
        s = s.iloc[-min(lookback, len(s)):]
        y = np.log(s.values)
        x = np.arange(len(y), dtype="float64")
        try:
            slope_per_day = np.polyfit(x, y, 1)[0]
        except Exception:
            return 0.0
        return float(slope_per_day * 252.0)

    def _sma_slope(series: np.ndarray, window_: int) -> float:
        """Normalized slope (per-day %) of last `window_` bars of price."""
        if series is None or len(series) < window_:
            return 0.0
        y = pd.Series(series[-window_:], dtype="float64")
        if not np.isfinite(y.iloc[-1]) or y.iloc[-1] <= 0:
            return 0.0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope / y.iloc[-1])  # % per day

    # ---- DRIFT: combine SMA20/200 slope signals + SMA level slopes + tiny hist + risk-free
    # 1) price-based SMA slopes (directional)
    sma20_slope = _sma_slope(y_close, window_=20)
    sma200_slope = _sma_slope(y_close, window_=200)
    daily_drift_from_slope = (sma_short_weight * sma20_slope) + (sma_long_weight * sma200_slope)
    ann_drift_from_slope = daily_drift_from_slope * 252.0

    # 2) SMA-level-based log-slope drift (uses provided SMA20/SMA200 series if available)
    mu_short_ann = 0.0
    mu_long_ann  = 0.0
    if use_sma_drift and y_sma_short_for_drift is not None:
        mu_short_ann = _annual_drift_from_sma_local(y_sma_short_for_drift, lookback=min(90, window), min_points=30)
    if use_sma_drift and y_sma_long_for_drift is not None:
        mu_long_ann = _annual_drift_from_sma_local(y_sma_long_for_drift, lookback=min(252, window), min_points=60)

    # normalize weights if one SMA missing
    w_s = float(np.clip(sma_short_weight, 0.0, 1.0))
    w_l = float(np.clip(sma_long_weight,  0.0, 1.0))
    if y_sma_short_for_drift is None and y_sma_long_for_drift is not None:
        w_s, w_l = 0.0, 1.0
    elif y_sma_long_for_drift is None and y_sma_short_for_drift is not None:
        w_s, w_l = 1.0, 0.0
    elif (y_sma_short_for_drift is None) and (y_sma_long_for_drift is None):
        w_s, w_l = 0.0, 0.0

    mu_sma_combo = (w_s * mu_short_ann) + (w_l * mu_long_ann)

    # 3) tiny historical + risk-free baselines
    mu_hist_ann = float(np.mean(r) * 252.0) if len(r) > 5 else 0.0

    # Final blend (weights are heuristics; tune to taste)
    mu_blend_ann = (
        0.50 * float(ann_drift_from_slope) +   # momentum direction from SMA slopes on price
        0.30 * float(mu_sma_combo) +           # regime anchor from SMA level slopes
        0.10 * float(mu_hist_ann) +            # tiny historical
        0.10 * float(risk_free_ann)            # small baseline
    )
    mu_blend_ann = float(np.clip(mu_blend_ann, -drift_cap_ann, drift_cap_ann))
    mu_d = mu_blend_ann * dt

    # --- antithetic helpers
    def _antithetic_norm(T_, N_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_normal((T_, half))
        Z_full = np.concatenate([Z_half, -Z_half], axis=1)
        return Z_full[:, :N_]

    def _antithetic_t(T_, N_, df_):
        half = (N_ + 1) // 2
        Z_half = rng.standard_t(df_, size=(T_, half)) / np.sqrt(df_ / (df_ - 2))
        Z_full = np.concatenate([Z_half, -Z_half], axis=1)
        return Z_full[:, :N_]

    # --- volatility level (uses OHLC if provided)
    sigma_ann = _ann_sigma_from_estimator(
        close=y_close,
        high=y_high, low=y_low, open_=y_open,
        mode=vol_mode, lam=lam, window=window
    )
    sigma_d0 = sigma_ann * np.sqrt(dt)

    # --- simulate
    if model == "GBM" or model == "EWMA+t":
        if model == "GBM":
            Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
            if stochastic_vol:
                # mean-reverting random variance around sigma_d0
                rr2 = r**2
                if len(rr2) >= 30:
                    xcv = rr2[:-1]; ycv = rr2[1:]
                    rho = np.corrcoef(xcv, ycv)[0, 1]
                    rho = 0.8 if not np.isfinite(rho) else float(np.clip(rho, 0.0, 0.98))
                else:
                    rho = 0.8
                v0 = sigma_d0**2
                sig = np.empty((T, N))
                sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_d - 0.5 * sig**2) + sig * Z
            else:
                steps = (mu_d - 0.5 * sigma_d0**2) + sigma_d0 * Z
            paths = last * np.exp(np.cumsum(steps, axis=0))

        else:  # "EWMA+t"
            # keep median neutral by default (driftless-t). Use mu_t = mu_d if you want trend here too.
            mu_t = 0.0
            Z = _antithetic_t(T, N, df_t) if antithetic else (rng.standard_t(df_t, size=(T, N)) / np.sqrt(df_t / (df_t - 2)))
            if stochastic_vol:
                rho = 0.85
                v0 = sigma_d0**2
                sig = np.empty((T, N))
                sig[0, :] = np.sqrt(v0)
                for t in range(1, T):
                    v0 = (1 - rho) * (sigma_d0**2) + rho * (sig[t-1, :]**2) * np.exp(rng.normal(0.0, 0.10, size=N))
                    sig[t, :] = np.sqrt(np.maximum(v0, 1e-12))
                steps = (mu_t - 0.5 * sig**2) + sig * Z
            else:
                steps = (mu_t - 0.5 * sigma_d0**2) + sigma_d0 * Z
            paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Bootstrap":
        def block_bootstrap(returns, T_, sims_, block_=5):
            returns = np.asarray(returns)
            if len(returns) == 0:
                return np.zeros((T_, sims_))
            if block_ <= 1 or len(returns) < block_:
                idx = rng.integers(0, len(returns), size=(T_, sims_))
                return returns[idx]
            R = np.empty((T_, sims_))
            max_start = max(0, len(returns) - block_)
            for j in range(sims_):
                path = []
                while len(path) < T_:
                    start = rng.integers(0, max_start + 1)
                    path.extend(returns[start:start+block_].tolist())
                R[:, j] = path[:T_]
            return R

        steps = block_bootstrap(r, T, N, block=int(block))
        paths = last * np.exp(np.cumsum(steps, axis=0))

    elif model == "Jump":
        Z = _antithetic_norm(T, N) if antithetic else rng.standard_normal((T, N))
        sigma_d = np.std(r, ddof=1) * np.sqrt(dt) if len(r) > 1 else sigma_d0
        mu_j, sig_j, lam_j = -0.02, 0.06, 3.0
        Njump = rng.poisson(lam=(lam_j * dt), size=(T, N))
        J = rng.normal(mu_j, sig_j, size=(T, N)) * Njump
        steps = (0.0 - 0.5 * sigma_d**2) + sigma_d * Z + J
        paths = last * np.exp(np.cumsum(steps, axis=0))

    else:
        # fallback to GBM (propagate all OHLC + drift knobs, incl. SMA20/200 combo)
        return _project_next_month(
            y_close, start_date,
            horizon_days=horizon_days, sims=sims,
            pct_low=pct_low, pct_high=pct_high,
            model="GBM", seed=seed,
            window=window, lam=lam, df_t=df_t, antithetic=antithetic, block=block,
            horizon_months=horizon_months,
            vol_mode=vol_mode, y_high=y_high, y_low=y_low, y_open=y_open,
            stochastic_vol=stochastic_vol,
            y_sma_short_for_drift=y_sma_short_for_drift,
            y_sma_long_for_drift=y_sma_long_for_drift,
            use_sma_drift=use_sma_drift,
            sma_short_weight=sma_short_weight,
            sma_long_weight=sma_long_weight,
            risk_free_ann=risk_free_ann,
            drift_cap_ann=drift_cap_ann,
        )

    # summarize
    low  = np.nanpercentile(paths, pct_low, axis=1)
    high = np.nanpercentile(paths, pct_high, axis=1)
    med  = np.nanpercentile(paths, 50, axis=1)
    return future, med, low, high