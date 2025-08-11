# TSX Margin Scanner + Backtester

VS Code–ready repo for scanning TSX names with technical + sentiment + fundamentals and backtesting strategies with margin rules.

## VS Code Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python main.py      # run scanner test
python backtester/backtester.py   # run backtest
```

## GitHub Automation (included)
- `.github/workflows/weekly.yml` — runs every Monday 9:00 AM Toronto (13:00 UTC), uploads artifact, and commits the report into `reports/YYYY-MM-DD/`.
- `.github/workflows/manual.yml` — run-on-demand from the Actions tab.

### Email (optional)
Add repo **Actions Secrets** if you want emailed CSV+charts:
`WS_EMAIL_ENABLED, WS_SMTP_SERVER, WS_SMTP_PORT, WS_SMTP_USER, WS_SMTP_PASSWORD, WS_EMAIL_FROM, WS_EMAIL_TO`.

## Backtester quick examples
- Edit `backtester/params.sample.json` and run:
```bash
python backtester/backtester.py --config backtester/params.sample.json
```
Outputs: `data/bt_equity_curve.csv`, `data/bt_trades.csv`, `data/bt_equity.png` and a console performance summary.
