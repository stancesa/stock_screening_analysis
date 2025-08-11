
from src.config import ScannerConfig
from src.scanner import run_scan
from src.charts import export_analysis_to_excel

if __name__ == "__main__":
    cfg = ScannerConfig()
    df = run_scan("canadian_tickers.txt", "my_holdings.txt", cfg, out_csv="data/weekly_margin_report.csv")
    if df is not None and cfg.make_charts and cfg.top_k_charts > 0:
        export_analysis_to_excel(df, cfg.top_k_charts, out_dir="data/charts")
        print("Saved charts to data/charts")
    print("Done.")
