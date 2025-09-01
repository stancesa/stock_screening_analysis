# etf_sector_analysis.py
import pandas as pd
import numpy as np
from typing import Iterable, Union, Sequence, Optional, Dict
from pathlib import Path

from core.utils import read_merged_list as _read_merged_list

PathLike = Union[str, Path]
PathList = Union[PathLike, Iterable[PathLike]]

# ----  column sniffers ---------------------------------------------------
_TICKER_COLS = ["ticker", "Ticker", "Symbol", "symbol"]
_SECTOR_COLS = ["sector", "Sector", "GICS Sector", "Category", "category"]
_WEIGHT_COLS = ["weight", "Weight", "% Weight", "weight_pct", "Weight (%)"]

def _pick_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ----  derive sector mix from ETF holdings -------------------------------

def _as_paths(items: PathList) -> list[Path]:
    """
    Normalize a single path or an iterable of paths into a list[Path].

    Parameters
    ----------
    items : str | Path | Iterable[str | Path]
        A single path or a collection of paths.

    Returns
    -------
    list[Path]
        Normalized list of Path objects.
    """
    if items is None:
        return []
    if isinstance(items, (str, Path)):
        return [Path(items)]
    return [Path(p) for p in items]


def extract_all_etfs(files: PathList) -> pd.DataFrame:
    """
    Load and merge ETF data from one or more files using `_read_merged_list`.

    Parameters
    ----------
    files : str | Path | Iterable[str | Path]
        One or many file paths (CSV/Parquet/etc. as supported by your reader).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing all ETF rows from the provided files.
        (Exact merge semantics are delegated to `_read_merged_list`.)
    """
    paths = _as_paths(files)
    if not paths:
        return pd.DataFrame()
    return _read_merged_list(paths)

def sector_trend_score(prices: pd.Series, lookback_wks: int = 26) -> pd.Series:
    """
    Compute a weekly trend score for a single price series.

    Method
    ------
    - Resample to weekly (Friday close): 'W-FRI'
    - Take log prices; for each rolling window of length `lookback_wks`,
      fit a simple linear regression slope vs. time (x = 0..L-1).
    - Normalize the slope by volatility (std of first-differenced logs),
      annualized by sqrt(52).
    - Squash to [-1, 1] via tanh(5 * slope/vol).

    Parameters
    ----------
    prices : pd.Series
        Price series with a DatetimeIndex (daily or higher frequency).
    lookback_wks : int, default 26
        Rolling lookback window length in weeks.

    Returns
    -------
    pd.Series
        Weekly-indexed series of trend scores in [-1, 1]. Values before
        the first complete window remain NaN. If fewer than `lookback_wks + 5`
        weekly points are available, returns a NaN series indexed like the
        weekly resample.
    """
    wk = prices.resample("W-FRI").last().dropna()
    L = lookback_wks
    if len(wk) < L + 5:
        return pd.Series(index=wk.index, dtype=float)

    lp = np.log(wk)
    out = pd.Series(index=wk.index, dtype=float)

    for i in range(L, len(lp)):
        y = lp.iloc[i - L : i].values
        x = np.arange(L)
        slope = np.polyfit(x, y, 1)[0]
        vol = np.std(np.diff(y)) * np.sqrt(52)
        out.iloc[i] = np.tanh(5 * (slope / (vol + 1e-8)))

    return out

def sector_mix_for_ticker(etfs: pd.DataFrame, ticker: str) -> Dict[str, float]:
    """
    Return a normalized sector weight mix for `ticker` from merged ETF holdings.

    If no explicit weights are present, counts are used as equal weights.
    """
    if etfs is None or etfs.empty:
        return {}

    tcol = _pick_col(etfs, _TICKER_COLS)
    scol = _pick_col(etfs, _SECTOR_COLS)
    wcol = _pick_col(etfs, _WEIGHT_COLS)

    if not tcol or not scol:
        return {}

    df = etfs.loc[etfs[tcol].astype(str).str.upper() == ticker.upper(), [scol] + ([wcol] if wcol else [])].copy()
    if df.empty:
        return {}

    if wcol and np.issubdtype(df[wcol].dtype, np.number):
        g = df.groupby(scol, dropna=True)[wcol].sum()
    else:
        g = df.groupby(scol, dropna=True).size().astype(float)

    if g.sum() == 0:
        return {}

    mix = (g / g.sum()).sort_values(ascending=False)
    return mix.to_dict()

def mix_concentration_stats(mix: Dict[str, float]) -> Dict[str, float]:
    """
    Given a normalized sector mix (sums to ~1), compute concentration stats:
      - top_share: share of the largest category (0..1)
      - hhi: Herfindahl-Hirschman index (0..1)
    """
    if not mix:
        return {"top_share": np.nan, "hhi": np.nan}
    weights = np.array(list(mix.values()), dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    s = weights.sum()
    if s <= 0:
        return {"top_share": np.nan, "hhi": np.nan}
    w = weights / s
    top_share = float(w.max())
    hhi = float(np.sum(w ** 2))
    return {"top_share": top_share, "hhi": hhi}

# ---- correlation-based sector affinity from prices ---------------------
def sector_corr_affinity(
    px: pd.Series,                 # ticker prices
    sector_prices: pd.DataFrame,   # columns = sector names, values = prices
    lookback_wks: int = 26,
    method: str = "pearson",
) -> Dict[str, object]:
    """
    Compute weekly-return correlations between the ticker and each sector ETF.
    Returns:
      {
        "corrs": {sector: corr, ...},
        "best_sector": <name or None>,
        "best_corr": <float or np.nan>
      }
    """
    if px is None or px.empty or sector_prices is None or sector_prices.empty:
        return {"corrs": {}, "best_sector": None, "best_corr": np.nan}

    wk_t = px.resample("W-FRI").last().dropna()
    wk_s = sector_prices.resample("W-FRI").last().dropna(how="all")

    # align
    idx = wk_t.index.intersection(wk_s.index)
    if len(idx) < lookback_wks + 5:
        return {"corrs": {}, "best_sector": None, "best_corr": np.nan}

    wk_t = wk_t.loc[idx]
    wk_s = wk_s.loc[idx]

    # use a rolling window (last L weeks)
    L = min(lookback_wks, len(idx) - 1)
    rt = wk_t.pct_change().iloc[-L:]
    rs = wk_s.pct_change().iloc[-L:]

    # compute correlations
    corrs = {}
    for col in rs.columns:
        s = rs[col]
        if s.notna().sum() >= max(8, L // 2) and rt.notna().sum() >= max(8, L // 2):
            corr = float(rt.corr(s, method=method))
            if np.isfinite(corr):
                corrs[col] = corr

    if not corrs:
        return {"corrs": {}, "best_sector": None, "best_corr": np.nan}

    best_sector = max(corrs, key=lambda k: abs(corrs[k]))
    best_corr = float(corrs[best_sector])
    return {"corrs": corrs, "best_sector": best_sector, "best_corr": best_corr}
