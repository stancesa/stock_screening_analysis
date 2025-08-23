import numpy as np
import pandas as pd

def compute_metrics(equity: np.ndarray, dates: pd.Series) -> dict:
    r = pd.Series(equity, index=dates).pct_change().dropna()
    if r.empty:
        return {}
    ann = 252
    cagr = (equity[-1] / equity[0]) ** (ann / max(len(r), 1)) - 1
    vol  = r.std(ddof=1) * np.sqrt(ann)
    dr   = (r[r<0]).std(ddof=1) * np.sqrt(ann)
    sharpe  = r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ann)
    sortino = r.mean() / (dr + 1e-12) * np.sqrt(ann)
    # drawdown
    curve = pd.Series(equity, index=dates)
    peak  = curve.cummax()
    dd    = (curve/peak - 1.0)
    mdd   = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=mdd, Calmar=calmar)

def underwater(equity: np.ndarray) -> np.ndarray:
    eq = pd.Series(equity)
    return (eq/eq.cummax() - 1.0).to_numpy()