from __future__ import annotations
import json, subprocess
from pathlib import Path
import streamlit as st
from .paths import PROJECT_ROOT, DEFAULT_OUTPUT
import pandas as pd

PROFILES_DIR = PROJECT_ROOT / ".app_state"
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# Profiles (save/load filters)
# =========================

PROFILES_DIR = PROJECT_ROOT / ".app_state"
PROFILES_DIR.mkdir(exist_ok=True, parents=True)

def save_profile(name: str, data: dict):
    (PROFILES_DIR / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_profile(name: str) -> dict | None:
    fp = PROFILES_DIR / f"{name}.json"
    try:
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
    except Exception:
        return None

def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))

# ===== Plot Theme =====
THEME_FILE = PROFILES_DIR / "plot_theme.json"
DEFAULT_THEME = {
    # price/overlays
    "close": "#1f77b4",
    "sma200": "#ff7f0e",
    "overlay": "#6a5acd",
    # annotations
    "stop": "#e74c3c",
    "target": "#2ecc71",
    "risk_band": "#8dd3c7",
    # projections
    "proj_mid": "#7f7f7f",
    "proj_band": "#9ecae1",
    # new: trading/bt colors
    "buy": "#16a34a",
    "sell": "#dc2626",
    "equity": "#6366f1",
    # new: kpi/table accents
    "kpi_good": "#ecfdf5",
    "kpi_bad": "#fef2f2",
    "table_pos": "#dcfce7",
    "table_neg": "#fee2e2",
}

RECO_COLORS = {
    "Strong Buy": "#008000",
    "Buy": "#90EE90",
    "Hold": "#FFFF00",
    "Sell": "#FFA500",
    "Strong Sell": "#FF0000",
}

def load_theme() -> dict:
    if THEME_FILE.exists():
        try:
            return json.load(open(THEME_FILE, "r", encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_THEME.copy()

def save_theme(theme: dict):
    try:
        json.dump(theme, open(THEME_FILE, "w", encoding="utf-8"), indent=2)
    except Exception:
        pass

@st.cache_data(show_spinner=False)
def _read_path_cached(p: Path) -> dict:
    import pandas as pd
    ext = p.suffix.lower()
    if ext == ".csv":
        return {"df": pd.read_csv(p), "sheets": None, "picked": None}
    if ext in {".xlsx", ".xls"}:
        xls = pd.read_excel(p, sheet_name=None)
        names = list(xls.keys())
        return {"df": xls[names[0]], "sheets": names, "picked": names[0]}
    raise ValueError(f"Unsupported file type: {ext}")

def run_main_and_reload() -> tuple[bool, str]:
    cmd = ["python", "main.py"]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        ok = (proc.returncode == 0)
        logs = (proc.stdout or "") + ("\n" + (proc.stderr or ""))
        return ok, logs
    except Exception as e:
        return False, f"Failed to run {' '.join(cmd)}: {e}"
    
@st.cache_data(show_spinner=False)

def _read_path_cached(p: Path) -> dict:
    """Return {'df': DataFrame, 'sheets': [names], 'picked': name or None}."""
    ext = p.suffix.lower()
    if ext == ".csv":
        return {"df": pd.read_csv(p), "sheets": None, "picked": None}
    if ext in {".xlsx", ".xls"}:
        xls = pd.read_excel(p, sheet_name=None)
        names = list(xls.keys())
        # default to first sheet; UI can re-pick
        return {"df": xls[names[0]], "sheets": names, "picked": names[0]}
    raise ValueError(f"Unsupported file type: {ext}")

def _run_main_and_reload() -> tuple[bool, str]:
    """Runs `python main.py` from project root and returns (success, logs_text)."""
    cmd = ["python", "main.py"]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        success = (proc.returncode == 0)
        logs = (proc.stdout or "") + ("\n" + (proc.stderr or ""))
        return success, logs
    except Exception as e:
        return False, f"Failed to run {' '.join(cmd)}: {e}"