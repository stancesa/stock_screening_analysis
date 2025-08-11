
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_charts(df: pd.DataFrame, top_k: int, out_dir: str = "data/charts"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sub = df.head(top_k)
    for _, row in sub.iterrows():
        dates = row["dates_series"]
        close = row["close_series"]
        sma200 = row["sma200_series"]
        plt.figure()
        plt.plot(dates, close, label="Close")
        plt.plot(dates, sma200, label="SMA200")
        plt.xticks(rotation=45)
        plt.title(f"{row['ticker']} — Close & SMA200")
        plt.legend()
        plt.tight_layout()
        out = Path(out_dir) / f"{row['ticker'].replace('.', '_')}.png"
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

    # ---- helpers ----
    def excel_col(n: int) -> str:
        s = ""; n += 1
        while n: n, r = divmod(n-1, 26); s = chr(65+r)+s
        return s

    def _autowidths(ws, dfv, extra=2, max_w=60):
        for j, col in enumerate(dfv.columns):
            maxlen = max([len(str(col))] + [len(str(v)) for v in dfv[col].fillna("").astype(str)])
            ws.set_column(j, j, min(maxlen + extra, max_w))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "analysis.xlsx"

    # limit to top_k
    df_top = df.head(int(top_k)).copy()

    # save charts next to workbook (uses your existing function if present)
    try:
        save_charts(df_top, top_k=top_k, out_dir=str(out_dir))
    except NameError:
        pass

    # hide heavy series on the Summary sheet
    series_cols = {"close_series", "sma200_series", "dates_series"}
    keep_cols = [c for c in df_top.columns if c not in series_cols]
    visible = df_top[keep_cols].copy()

    # dtypes to ensure rules fire (numbers must be numbers; booleans must be boolean)
    money_cols = ["last", "stop_price", "target_price", "risk_dollars", "alloc_cap_dollars"]
    pct_cols   = ["sma_dev_pct", "div_yield_pct", "headline_sentiment"]
    int_cols   = ["signals_score", "suggested_shares", "max_hold_days"]
    bool_cols  = ["rsi_ok","macd_cross_ok","sma_dev_ok","vol_spike_ok"]

    for c in money_cols + pct_cols + int_cols:
        if c in visible.columns:
            visible[c] = pd.to_numeric(visible[c], errors="coerce")
    for c in bool_cols:
        if c in visible.columns:
            # accept 1/0, "true"/"false", etc.
            visible[c] = (
                visible[c]
                .map(lambda x: np.nan if pd.isna(x) else bool(x) if isinstance(x, (int, float, np.bool_)) else str(x).strip().lower() in {"true","1","yes"})
                .astype("boolean")
            )

    # add chart hyperlink
    def chart_link(t):
        p = out_dir / f"{str(t).replace('.','_')}.png"
        return f'=HYPERLINK("{p.as_posix()}", "Open")' if p.exists() else ""
    if "ticker" in visible.columns:
        visible.insert(1, "chart", visible["ticker"].map(chart_link))
    else:
        visible.insert(1, "chart", "")

    # ---- write with XlsxWriter (enforced) ----
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
        f_bool_t = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        f_bool_f = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

        # header & filter
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(visible), len(visible.columns)-1)
        for j, name in enumerate(visible.columns):
            ws.write(0, j, name, f_head)

        # column formats
        col_idx = {c: i for i, c in enumerate(visible.columns)}
        for c in money_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 12, f_money)
        for c in pct_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 10, f_pct)
        for c in int_cols:
            if c in col_idx: ws.set_column(col_idx[c], col_idx[c], 8, f_int)
        if "date" in col_idx:   ws.set_column(col_idx["date"], col_idx["date"], 12, f_date)
        if "ticker" in col_idx: ws.set_column(col_idx["ticker"], col_idx["ticker"], 12, f_text)
        if "catalysts" in col_idx: ws.set_column(col_idx["catalysts"], col_idx["catalysts"], 30, f_text)
        if "chart" in col_idx:  ws.set_column(col_idx["chart"], col_idx["chart"], 10, f_link)

        # zebra stripes
        last_row = len(visible)
        last_col = len(visible.columns) - 1
        ws.conditional_format(1, 0, last_row, last_col, {
            "type": "formula", "criteria": "=MOD(ROW(),2)=0", "format": f_alt
        })

        # color scales / data bars
        if "signals_score" in col_idx:
            j = col_idx["signals_score"]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#F8696B","mid_color":"#FFEB84","max_color":"#63BE7B"})
        if "headline_sentiment" in col_idx:
            j = col_idx["headline_sentiment"]
            ws.conditional_format(1, j, last_row, j, {"type":"2_color_scale",
                "min_color":"#F8696B","max_color":"#63BE7B"})
        if "sma_dev_pct" in col_idx:
            j = col_idx["sma_dev_pct"]
            ws.conditional_format(1, j, last_row, j, {"type":"3_color_scale",
                "min_color":"#63BE7B","mid_color":"#FFFFFF","max_color":"#F8696B"})
        if "risk_dollars" in col_idx:
            j = col_idx["risk_dollars"]
            ws.conditional_format(1, j, last_row, j, {"type":"data_bar","bar_border_color":"#A6A6A6"})

        # boolean traffic lights (handle both TRUE and "TRUE" just in case)
        for c in [b for b in bool_cols if b in col_idx]:
            j = col_idx[c]
            # TRUE
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": True, "format": f_bool_t})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": '"TRUE"', "format": f_bool_t})
            # FALSE
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": False, "format": f_bool_f})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"==","value": '"FALSE"', "format": f_bool_f})

        # RSI bands
        if "rsi" in col_idx:
            j = col_idx["rsi"]
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":">","value":70,
                                                      "format": wb.add_format({"bg_color":"#FFE699"})})
            ws.conditional_format(1, j, last_row, j, {"type":"cell","criteria":"<","value":30,
                                                      "format": wb.add_format({"bg_color":"#C6EFCE"})})

        # MACD histogram helper column (macd - macd_signal) for color scale
        if "macd" in col_idx and "macd_signal" in col_idx:
            helper = last_col + 1
            ws.write(0, helper, "macd_hist", f_head)
            c_macd = excel_col(col_idx["macd"])
            c_sig  = excel_col(col_idx["macd_signal"])
            for r in range(1, last_row+1):
                ws.write_formula(r, helper, f"=IFERROR({c_macd}{r+1}-{c_sig}{r+1},\"\")")
            ws.set_column(helper, helper, 10)
            ws.conditional_format(1, helper, last_row, helper, {"type":"2_color_scale",
                                                                "min_color":"#F8696B","max_color":"#63BE7B"})

        # widths last
        _autowidths(ws, visible)

        # simple charts sheet (only for the top_k subset)
        if "ticker" in df_top.columns:
            imgs = []
            for t in df_top["ticker"]:
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

    print(f"✅ Excel with color coding saved to: {xlsx_path}")