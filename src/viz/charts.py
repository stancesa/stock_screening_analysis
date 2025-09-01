
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

# ---------- helpers ----------

def safe_print(msg: str, ok_prefix="✅"):
    try:
        print(f"{ok_prefix} {msg}")
    except UnicodeEncodeError:
        # Console can't encode the emoji (common on Windows CP1252) -> ASCII fallback
        print(f"[OK] {msg}")

def _to_list(x):
    # already a list/tuple/ndarray
    if isinstance(x, (list, tuple)):
        return list(x)
    # pandas Series (unlikely here, but just in case)
    if isinstance(x, pd.Series):
        return x.tolist()
    # stringified list from CSV -> parse
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return list(v) if isinstance(v, (list, tuple)) else None
        except Exception:
            return None
    return None

# ---------- main ----------

def save_charts(df: pd.DataFrame, top_k: int, out_dir: str = "data/charts"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sub = df.head(top_k)

    # figure out which column names exist (legacy vs flat)
    dates_col = next((c for c in ("dates_series", "series__dates_series") if c in sub.columns), None)
    close_col = next((c for c in ("close_series", "series__close_series") if c in sub.columns), None)
    sma200_col = next((c for c in ("sma200_series", "series__sma200_series") if c in sub.columns), None)
    ticker_col = next((c for c in ("ticker", "meta__ticker") if c in sub.columns), None)

    if not all([dates_col, close_col, sma200_col, ticker_col]):
        missing = [("dates", dates_col), ("close", close_col), ("sma200", sma200_col), ("ticker", ticker_col)]
        raise KeyError("Missing required series columns: " + ", ".join(k for k,v in missing if v is None))

    for _, row in sub.iterrows():
        dates = pd.to_datetime(_to_list(row[dates_col]))
        close = _to_list(row[close_col])
        sma200 = _to_list(row[sma200_col])
        ticker = row[ticker_col]

        # sanity checks
        if dates.empty or close is None or sma200 is None:
            print(f"[WARN] {ticker}: missing/empty series, skipping.")
            continue
        n = min(len(dates), len(close), len(sma200))
        if n == 0:
            print(f"[WARN] {ticker}: zero-length series, skipping.")
            continue

        dates, close, sma200 = dates[:n], close[:n], sma200[:n]

        plt.figure()
        plt.plot(dates, close, label="Close")
        plt.plot(dates, sma200, label="SMA200")
        plt.xticks(rotation=45)
        plt.title(f"{ticker} — Close & SMA200")
        plt.legend()
        plt.tight_layout()
        out = Path(out_dir) / f"{str(ticker).replace('.', '_')}.png"
        plt.savefig(out)
        plt.close()

def export_analysis_to_excel(df: pd.DataFrame, top_k: int, out_dir: str = "data/charts"):
    # ---- require xlsxwriter so formatting is preserved ----
    try:
        import xlsxwriter  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "xlsxwriter is required for color coding/formatting. "
            "Install with: pip install xlsxwriter"
        ) from e

    import numpy as np
    import pandas as pd
    from pathlib import Path

    # ---- helpers ----
    def excel_col(n: int) -> str:
        s = ""; n += 1
        while n: n, r = divmod(n-1, 26); s = chr(65+r)+s
        return s

    def _autowidths(ws, dfv, extra=2, max_w=60):
        for j, col in enumerate(dfv.columns):
            maxlen = max([len(str(col))] + [len(str(v)) for v in dfv[col].fillna("").astype(str)])
            ws.set_column(j, j, min(maxlen + extra, max_w))

    def _coerce_percent(col: pd.Series) -> pd.Series:
        return pd.to_numeric(
            col.astype(str).str.strip().str.rstrip("%").replace({"": np.nan}),
            errors="coerce"
        ) / 100.0

    def _first_present(cols: list[str], pool: set[str]) -> str | None:
        for c in cols:
            if c in pool:
                return c
        return None

    def _present(cols_available: set[str], *names) -> list[str]:
        return [n for n in names if n in cols_available]

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "analysis.xlsx"

    # limit to top_k
    df_top = df.head(int(top_k)).copy()

    # save charts next to workbook (uses your existing function if present)
    try:
        save_charts(df_top, top_k=top_k, out_dir=str(out_dir))
    except NameError:
        pass

    # hide heavy series on the Summary sheet (support legacy & flat)
    series_prefix = "series__"
    series_cols = {c for c in df_top.columns if c.startswith(series_prefix)}
    series_cols |= {"close_series", "sma200_series", "dates_series"}
    keep_cols = [c for c in df_top.columns if c not in series_cols]
    visible = df_top[keep_cols].copy()

    # ---- normalize types so conditional formats actually apply ----
    cols_available = set(visible.columns)

    # flexible name resolution
    rsi_col   = _first_present(["rsi","momentum__rsi"], cols_available)
    macd_col  = _first_present(["macd","momentum__macd","momentum__macd_line"], cols_available)
    macd_sig_col = _first_present(["macd_signal","momentum__macd_signal"], cols_available)

    sma200_col = _first_present(["sma200","trend__sma200"], cols_available)
    sma50_col  = _first_present(["sma50","trend__sma50"], cols_available)
    sma20_col  = _first_present(["sma20","trend__sma20"], cols_available)

    sma_dev_col = _first_present(["sma_dev_pct","price_context__sma_dev_pct"], cols_available)
    pct_hi_col  = _first_present(["pct_from_52w_high","price_context__pct_from_52w_high"], cols_available)

    bbpb_col    = _first_present(["bb_percent_b","bands_breakouts__bb_percent_b"], cols_available)
    above_boll_mid_col = _first_present(["above_boll_mid","trend__above_boll_mid"], cols_available)

    vol_col     = _first_present(["vol","volume__vol"], cols_available)
    vol20_col   = _first_present(["vol_avg20","volume__vol_avg20"], cols_available)

    ema_fast_cross_col = _first_present(["ema_fast_cross_up","trend__ema_fast_cross_up"], cols_available)
    ema50_reclaim_col  = _first_present(["ema50_reclaim","trend__ema50_reclaim"], cols_available)

    donchian_col = _first_present(["donchian_breakout","bands_breakouts__donchian_breakout"], cols_available)
    vol_spike_col = _first_present(["vol_spike_ok","volume__vol_spike_ok"], cols_available)

    sigscore_col = _first_present(["signals_score","signals_and_scores__signals_score"], cols_available)
    comp_col     = _first_present(["composite_score","signals_and_scores__composite_score"], cols_available)
    sent_col     = _first_present(["headline_sentiment","fundamentals_and_sentiment__headline_sentiment"], cols_available)

    last_col_name = _first_present(["last","meta__last"], cols_available)
    stop_col_name = _first_present(["stop_price","stops_and_risk__planned_stop_price"], cols_available)
    tgt_col_name  = _first_present(["target_price","stops_and_risk__planned_target_price"], cols_available)

    atr_col  = _first_present(["atr","stops_and_risk__atr"], cols_available)
    trail_pct_col = _first_present(["trailing_pct","stops_and_risk__trailing_pct"], cols_available)

    alloc_cap_col = _first_present(["alloc_cap_dollars","position_sizing__alloc_cap_dollars"], cols_available)
    risk_dollars_col = _first_present(["risk_dollars","position_sizing__risk_dollars"], cols_available)
    sugg_shares_col  = _first_present(["suggested_shares","position_sizing__suggested_shares"], cols_available)

    owned_col = _first_present(["owned","meta__owned","position__owned"], cols_available)

    # typing
    money_cols = _present(cols_available, "last","stop_price","target_price","risk_dollars","alloc_cap_dollars",
                          "stops_and_risk__planned_stop_price","stops_and_risk__planned_target_price",
                          "position_sizing__risk_dollars","position_sizing__alloc_cap_dollars",
                          "meta__last")
    int_cols   = _present(cols_available, "signals_score","suggested_shares","max_hold_days",
                          "signals_and_scores__signals_score","position_sizing__suggested_shares",
                          "stops_and_risk__max_hold_days")
    pct_cols   = [c for c in [sma_dev_col, sent_col, trail_pct_col] if c] + \
                 _present(cols_available, "div_yield_pct","fundamentals_and_sentiment__div_yield_pct")

    # coerce numerics
    for c in money_cols + int_cols:
        if c in visible.columns:
            visible[c] = pd.to_numeric(visible[c], errors="coerce")

    for c in pct_cols:
        if c in visible.columns:
            ser = visible[c]
            if ser.dropna().astype(str).str.contains("%").any():
                visible[c] = _coerce_percent(ser)
            else:
                visible[c] = pd.to_numeric(ser, errors="coerce")

    # normalize booleans
    bool_gate_candidates = [
        "rsi_ok","macd_cross_ok","sma_dev_ok","vol_spike_ok","ema_fast_cross_up","ema50_reclaim",
        "above_boll_mid","donchian_breakout","rsi_hook_up","gap_up",
        "trend__ema_fast_cross_up","trend__ema50_reclaim","trend__above_boll_mid",
        "bands_breakouts__donchian_breakout","momentum__rsi_ok","momentum__rsi_hook_up","momentum__gap_up",
        "volume__vol_spike_ok","price_context__sma_dev_ok","momentum__macd_cross_ok",
        "owned","meta__owned","position__owned"
    ]
    bool_cols = [c for c in bool_gate_candidates if c in cols_available]
    bool_cols += [c for c in visible.columns if c.endswith("_ok")]  # catch-alls

    for c in bool_cols:
        visible[c] = (
            visible[c]
            .map(lambda x: np.nan if pd.isna(x) else bool(x) if isinstance(x, (int, float, np.bool_)) else str(x).strip().lower() in {"true","1","yes"})
            .astype("boolean")
        )

    # add chart hyperlink
    tkr_col = _first_present(["ticker","meta__ticker"], cols_available)
    def chart_link(t):
        p = out_dir / f"{str(t).replace('.','_')}.png"
        return f'=HYPERLINK("{p.as_posix()}", "Open")' if p.exists() else ""
    if tkr_col:
        visible.insert(1, "chart", visible[tkr_col].map(chart_link))
    else:
        visible.insert(1, "chart", "")

    # text columns for word-based highlights
    candidate_text_cols = {"notes","catalysts","status","thesis","comment","summary"}
    text_cols = [c for c in visible.columns if c in candidate_text_cols or visible[c].dtype == "object"]

    # ---- write with XlsxWriter ----
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
        visible.to_excel(xl, sheet_name="Summary", index=False)
        wb, ws = xl.book, xl.sheets["Summary"]

        # formats
        f_head = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "bg_color": "#F2F2F2", "border": 1})
        f_money = wb.add_format({"num_format": "$#,##0.00"})
        f_pct   = wb.add_format({"num_format": "0.00%"})
        f_int   = wb.add_format({"num_format": "0"})
        f_text  = wb.add_format({})
        f_link  = wb.add_format({"font_color": "blue", "underline": 1})
        f_date  = wb.add_format({"num_format": "yyyy-mm-dd"})
        f_alt   = wb.add_format({"bg_color": "#FAFAFA"})
        # booleans
        f_bool_t = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        f_bool_f = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        # text highlights
        f_warn  = wb.add_format({"bg_color": "#FFE699"})
        f_bad   = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        f_good  = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        # owned row
        f_owned = wb.add_format({"bg_color": "#E7F3FF"})

        # header & base widths
        ws.freeze_panes(1, 0)
        for j, name in enumerate(visible.columns):
            ws.write(0, j, name, f_head)

        col_idx = {c: i for i, c in enumerate(visible.columns)}
        last_row = len(visible)
        last_col_idx = len(visible.columns) - 1

        # column formats
        for c in money_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 12, f_money)
        for c in pct_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 10, f_pct)
        for c in int_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 8, f_int)
        if "date" in col_idx:   ws.set_column(col_idx["date"], col_idx["date"], 12, f_date)
        if tkr_col and tkr_col in col_idx: ws.set_column(col_idx[tkr_col], col_idx[tkr_col], 12, f_text)
        if "catalysts" in col_idx: ws.set_column(col_idx["catalysts"], col_idx["catalysts"], 36, f_text)
        if "chart" in col_idx:  ws.set_column(col_idx["chart"], col_idx["chart"], 10, f_link)

        # ---- Heat/Scales/Data bars ----
        # Signals score heat
        if sigscore_col and sigscore_col in col_idx:
            j = col_idx[sigscore_col]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#F8696B","mid_color":"#FFEB84","max_color":"#63BE7B"})
        # Composite score heat + rank
        if comp_col and comp_col in col_idx:
            j = col_idx[comp_col]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#F8696B","mid_color":"#FFEB84","max_color":"#63BE7B"})
            # rank_composite helper
            rank_col_idx = last_col_idx + 1
            ws.write(0, rank_col_idx, "rank_composite", f_head)
            c_comp = excel_col(j)
            ws.set_column(rank_col_idx, rank_col_idx, 10, f_int)
            ws.write_formula(1, rank_col_idx,
                f"=IFERROR(RANK.EQ({c_comp}2,{c_comp}$2:{c_comp}${last_row+1},0),\"\")")
            for r in range(2, last_row+1):
                ws.write_formula(r-1, rank_col_idx,
                    f"=IFERROR(RANK.EQ({c_comp}{r},{c_comp}$2:{c_comp}${last_row+1},0),\"\")")
            last_col_idx = rank_col_idx

        # Sentiment
        if sent_col and sent_col in col_idx:
            j = col_idx[sent_col]
            ws.conditional_format(1, j, last_row, j, {"type":"2_color_scale",
                "min_color":"#F8696B","max_color":"#63BE7B"})

        # SMA deviation (near/below 0 = greener)
        if sma_dev_col and sma_dev_col in col_idx:
            j = col_idx[sma_dev_col]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#63BE7B","mid_color":"#FFFFFF","max_color":"#F8696B"})

        # Volume ratio data bar + spike thresholds
        volratio_col = None
        if vol_col and vol20_col and vol_col in col_idx and vol20_col in col_idx:
            volratio_col = last_col_idx + 1
            ws.write(0, volratio_col, "vol_ratio", f_head)
            c_v  = excel_col(col_idx[vol_col])
            c_v20= excel_col(col_idx[vol20_col])
            for r in range(1, last_row+1):
                ws.write_formula(r, volratio_col, f"=IFERROR({c_v}{r+1}/{c_v20}{r+1},\"\")")
            ws.set_column(volratio_col, volratio_col, 10)
            # keep bar for quick sense
            ws.conditional_format(1, volratio_col, last_row, volratio_col, {"type":"data_bar","bar_border_color":"#A6A6A6"})
            # thresholds: ≥2 green, 1.5–1.99 yellow, <1 red
            ws.conditional_format(1, volratio_col, last_row, volratio_col, {"type":"cell","criteria":">=","value":2, "format": f_good})
            ws.conditional_format(1, volratio_col, last_row, volratio_col, {"type":"cell","criteria":"between","minimum":1.5,"maximum":1.99, "format": f_warn})
            ws.conditional_format(1, volratio_col, last_row, volratio_col, {"type":"cell","criteria":"<","value":1, "format": f_bad})
            last_col_idx = volratio_col

        # Booleans: traffic lights
        for c in [b for b in bool_cols if b in col_idx]:
            j = col_idx[c]
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": True,  "format": f_bool_t})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": '"TRUE"',  "format": f_bool_t})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": False, "format": f_bool_f})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": '"FALSE"', "format": f_bool_f})

        # ===== RSI: <30 green, 30–70 yellow, >70 red (applies to ANY numeric 'rsi' column) =====
        rsi_numeric_cols = [c for c in visible.columns
                            if ("rsi" in c.lower()) and (c in col_idx)
                            and not (c in bool_cols)]
        for c in rsi_numeric_cols:
            j = col_idx[c]
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"<","value":30, "format": f_good})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"between","minimum":30,"maximum":70, "format": f_warn})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":">","value":70, "format": f_bad})

        # ===== MACD: histogram heat + MACD > Signal boolean =====
        if macd_col and macd_sig_col and macd_col in col_idx and macd_sig_col in col_idx:
            helper = last_col_idx + 1
            ws.write(0, helper, "macd_hist", f_head)
            c_macd = excel_col(col_idx[macd_col])
            c_sig  = excel_col(col_idx[macd_sig_col])
            for r in range(1, last_row+1):
                ws.write_formula(r, helper, f"=IFERROR({c_macd}{r+1}-{c_sig}{r+1},\"\")")
            ws.set_column(helper, helper, 10)
            ws.conditional_format(1, helper, last_row, helper, {
                "type":"2_color_scale","min_color":"#F8696B","max_color":"#63BE7B"})
            last_col_idx = helper

            # boolean: macd_line_gt_signal
            macd_gt_col = last_col_idx + 1
            ws.write(0, macd_gt_col, "macd_line_gt_signal", f_head)
            for r in range(1, last_row+1):
                ws.write_formula(r, macd_gt_col, f"=IFERROR({c_macd}{r+1}>{c_sig}{r+1},\"\")")
            ws.set_column(macd_gt_col, macd_gt_col, 14)
            ws.conditional_format(1, macd_gt_col, last_row, macd_gt_col, {"type":"cell","criteria":"==","value": True,  "format": f_bool_t})
            ws.conditional_format(1, macd_gt_col, last_row, macd_gt_col, {"type":"cell","criteria":"==","value": False, "format": f_bool_f})
            last_col_idx = macd_gt_col

        # ===== Bollinger %B: heatmap + edge flags =====
        if bbpb_col and bbpb_col in col_idx:
            j = col_idx[bbpb_col]
            # heatmap across 0..1
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#63BE7B","mid_color":"#FFEB84","max_color":"#F8696B"})
            # edge flags remain
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"<=","value":0.05, "format": f_good})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":">=","value":0.95, "format": f_bad})

        # 52-week posture
        if pct_hi_col and pct_hi_col in col_idx:
            j = col_idx[pct_hi_col]
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":">","value":-2, "format": f_warn})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"<","value":-30, "format": f_good})

        # ===== Donchian: breakout booleans already handled; heatmap ALL numeric 'donchian' readouts =====
        import re
        don_numeric_cols = []
        pat = re.compile("donchian", re.IGNORECASE)
        for c in visible.columns:
            if (c in col_idx) and pat.search(c) and (c != donchian_col) and (c not in bool_cols):
                # try to treat column as numeric in Excel (just formatting, not coercion)
                don_numeric_cols.append(c)
        for c in don_numeric_cols:
            j = col_idx[c]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#63BE7B","mid_color":"#FFEB84","max_color":"#F8696B"})

        # ---- Checklist helpers (your daily flow) ----
        def _add_bool_helper(name: str, formula_builder, width=12):
            nonlocal last_col_idx
            last_col_idx += 1
            j = last_col_idx
            ws.write(0, j, name, f_head)
            for r in range(1, last_row+1):
                ws.write_formula(r, j, formula_builder(r+1))
            ws.set_column(j, j, width)
            # traffic lights for helper
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": True,  "format": f_bool_t})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": False, "format": f_bool_f})
            return j

        def _add_value_helper(name: str, formula_builder, fmt=None, width=12):
            nonlocal last_col_idx
            last_col_idx += 1
            j = last_col_idx
            ws.write(0, j, name, f_head)
            for r in range(1, last_row+1):
                ws.write_formula(r, j, formula_builder(r+1))
            ws.set_column(j, j, width, fmt)
            return j

        # Column letters
        c_last  = excel_col(col_idx[last_col_name]) if last_col_name and last_col_name in col_idx else None
        c_sma200= excel_col(col_idx[sma200_col]) if sma200_col and sma200_col in col_idx else None
        c_ema_fast = excel_col(col_idx[ema_fast_cross_col]) if ema_fast_cross_col and ema_fast_cross_col in col_idx else None
        c_ema50rec = excel_col(col_idx[ema50_reclaim_col]) if ema50_reclaim_col and ema50_reclaim_col in col_idx else None
        c_rsi   = excel_col(col_idx[rsi_col]) if rsi_col and rsi_col in col_idx else None
        c_macd_ok = excel_col(col_idx.get("momentum__macd_cross_ok") or col_idx.get("macd_cross_ok")) if ("momentum__macd_cross_ok" in col_idx or "macd_cross_ok" in col_idx) else None
        c_pct_hi= excel_col(col_idx[pct_hi_col]) if pct_hi_col and pct_hi_col in col_idx else None
        c_don   = excel_col(col_idx[donchian_col]) if donchian_col and donchian_col in col_idx else None
        c_vspike= excel_col(col_idx[vol_spike_col]) if vol_spike_col and vol_spike_col in col_idx else None
        c_stop  = excel_col(col_idx[stop_col_name]) if stop_col_name and stop_col_name in col_idx else None
        c_tgt   = excel_col(col_idx[tgt_col_name]) if tgt_col_name and tgt_col_name in col_idx else None
        c_alloc = excel_col(col_idx[alloc_cap_col]) if alloc_cap_col and alloc_cap_col in col_idx else None
        c_riskd = excel_col(col_idx[risk_dollars_col]) if risk_dollars_col and risk_dollars_col in col_idx else None
        c_trail = excel_col(col_idx[trail_pct_col]) if trail_pct_col and trail_pct_col in col_idx else None

        # trend_bias_ok
        if c_last and c_sma200 and (c_ema_fast or c_ema50rec):
            def f_trend(r):
                left = f"AND(ISNUMBER({c_last}{r}),ISNUMBER({c_sma200}{r}),{c_last}{r}>={c_sma200}{r})"
                right = []
                if c_ema_fast:  right.append(f"{c_ema_fast}{r}=TRUE")
                if c_ema50rec:  right.append(f"{c_ema50rec}{r}=TRUE")
                return f"=IF({left}, OR({','.join(right)}), FALSE)"
            trend_bias_col = _add_bool_helper("trend_bias_ok", f_trend)

        # timing_window_ok  (RSI 45–60, MACD cross OK, within 20% of 52W high)
        if c_rsi and (c_macd_ok or macd_col) and c_pct_hi:
            def f_timing(r):
                macd_ok_expr = f"{c_macd_ok}{r}=TRUE" if c_macd_ok else "TRUE"
                return f"=AND(ISNUMBER({c_rsi}{r}), {c_rsi}{r}>=45, {c_rsi}{r}<=60, {macd_ok_expr}, {c_pct_hi}{r}>=-20)"
            timing_col = _add_bool_helper("timing_window_ok", f_timing)

        # confirm_ok  (Donchian breakout OR volume spike)
        if c_don or c_vspike:
            def f_confirm(r):
                parts = []
                if c_don:    parts.append(f"{c_don}{r}=TRUE")
                if c_vspike: parts.append(f"{c_vspike}{r}=TRUE")
                return f"=OR({','.join(parts)})" if parts else "=FALSE"
            confirm_col = _add_bool_helper("confirm_ok", f_confirm)

        # risk_reward & rr_ok  (>= 2)
        rr_col = None
        if c_last and c_stop and c_tgt:
            rr_col = last_col_idx + 1
            ws.write(0, rr_col, "risk_reward", f_head)
            for r in range(1, last_row+1):
                ws.write_formula(r, rr_col, f"=IFERROR( ({c_tgt}{r+1}-{c_last}{r+1}) / MAX({c_last}{r+1}-{c_stop}{r+1},1E-9) ,\"\")")
            ws.set_column(rr_col, rr_col, 10)
            ws.conditional_format(1, rr_col, last_row, rr_col, {"type": "icon_set", "icon_style": "3_arrows"})
            last_col_idx = rr_col

            def f_rrok(r):
                return f"=IFERROR({excel_col(rr_col)}{r}>=2, FALSE)"
            rr_ok_col = _add_bool_helper("rr_ok", f_rrok)

        # size_shares_calc = MIN( alloc_cap/price , risk_dollars/(price−stop) ), floored
        if c_last and c_stop and (c_alloc or c_riskd):
            def f_size(r):
                part1 = f"IF({c_alloc}{r}>0, {c_alloc}{r}/{c_last}{r}, 1E99)" if c_alloc else "1E99"
                part2 = f"IF({c_riskd}{r}>0, {c_riskd}{r}/MAX({c_last}{r}-{c_stop}{r},1E-9), 1E99)" if c_riskd else "1E99"
                return f"=IFERROR(ROUNDDOWN(MIN({part1},{part2}),0),\"\")"
            size_col = _add_value_helper("size_shares_calc", f_size, fmt=f_int, width=14)

        # trail_stop_price = price * (1 - trailing_pct)
        if c_last and c_trail:
            def f_trailstop(r):
                return f"=IF(AND(ISNUMBER({c_last}{r}),ISNUMBER({c_trail}{r})), {c_last}{r}*(1-{c_trail}{r}), \"\")"
            trail_col = _add_value_helper("trail_stop_price", f_trailstop, fmt=f_money, width=14)

        # setup_ok = AND(trend_bias_ok, timing_window_ok, confirm_ok, rr_ok)
        letters = {}
        if 'trend_bias_col' in locals(): letters['trend'] = excel_col(trend_bias_col)
        if 'timing_col' in locals():     letters['timing'] = excel_col(timing_col)
        if 'confirm_col' in locals():    letters['confirm'] = excel_col(confirm_col)
        if 'rr_ok_col' in locals():      letters['rr'] = excel_col(rr_ok_col)

        if len(letters) >= 3:  # need at least trend/timing/confirm; rr optional but preferred
            def f_setup(r):
                parts = [f"{letters[k]}{r}=TRUE" for k in letters]
                return f"=AND({','.join(parts)})"
            setup_col = _add_bool_helper("setup_ok", f_setup, width=12)

        # ---- Text-based highlights for catalysts/sentiment hooks ----
        CAUTION = {"caution","heads up","watch","volatility","extended","overbought","overextended"}
        NEG     = {"warning","risk","downgrade","miss","lawsuit","investigation","delay","halt","recall","guidance cut","cuts"}
        POS     = {"good","strong","beat","beats","upgrade","contract","award","approval","clearance","partnership","buyback","dividend hike"}
        def _add_contains_rule(col_name: str, phrases: set, fmt):
            if col_name not in col_idx: return
            j = col_idx[col_name]
            for p in phrases:
                ws.conditional_format(1, j, last_row, j, {"type": "text", "criteria": "containing", "value": p, "format": fmt})
        for c in text_cols:
            _add_contains_rule(c, CAUTION, f_warn)
            _add_contains_rule(c, NEG, f_bad)
            _add_contains_rule(c, POS, f_good)

        # ---- Row highlight for owned positions ----
        if owned_col and owned_col in col_idx:
            j = col_idx[owned_col]
            owned_letter = excel_col(j)
            ws.conditional_format(1, 0, last_row, last_col_idx, {
                "type": "formula",
                "criteria": f"=${owned_letter}2=TRUE",
                "format": f_owned
            })

        # zebra stripes (after all columns are added)
        ws.conditional_format(1, 0, last_row, last_col_idx, {
            "type": "formula",
            "criteria": "=MOD(ROW(),2)=0",
            "format": f_alt,
            "stop_if_true": False,
        })

        # update filter to new rightmost column
        ws.autofilter(0, 0, last_row, last_col_idx)

        # widths last for the original dataframe block
        _autowidths(ws, visible)

        # simple charts sheet (only for the top_k subset)
        if tkr_col and tkr_col in df_top.columns:
            imgs = []
            for t in df_top[tkr_col]:
                p = out_dir / f"{str(t).replace('.','_')}.png"
                if p.exists(): imgs.append((t, p))
            if imgs:
                ws2 = wb.add_worksheet("Charts")
                ws2.set_column(0, 0, 18)
                row = 0
                for ticker, p in imgs:
                    ws2.write(row, 0, ticker)
                    ws2.insert_image(row, 1, str(p), {"x_scale": 0.6, "y_scale": 0.6})
                    row += 20

    safe_print(f"Excel with signals/checklist saved to: {xlsx_path}")
