from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project-root/src/core/ -> project-root
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "weekly_margin_report.csv"