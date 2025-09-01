from __future__ import annotations
import os, json, logging, inspect
from logging import LoggerAdapter
from typing import Optional
import numpy as np
import pandas as pd

def setup_logging(level: Optional[str] = None) -> None:
    """Idempotent root-logger setup. Safe to call in every entrypoint."""
    if getattr(setup_logging, "_configured", False):
        return
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_obj = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(level_obj)
    setup_logging._configured = True

class TickerLogger(LoggerAdapter):
    def process(self, msg, kwargs):
        t = (self.extra or {}).get("ticker")
        prefix = f"[{t}] " if t else ""
        return prefix + str(msg), kwargs

def get_log(name: Optional[str] = None, ticker: Optional[str] = None) -> TickerLogger:
    """Get a module-named logger, optionally decorated with a ticker prefix."""
    if name is None:
        # Best effort: use the caller's module name
        frame = inspect.stack()[1].frame
        name = frame.f_globals.get("__name__", "app")
    base = logging.getLogger(name)
    return TickerLogger(base, {"ticker": ticker})

def log_value(log: TickerLogger, name: str, val, level=logging.DEBUG, sample_n: int = 3) -> None:
    """Log type/shape/preview for debugging mysterious values."""
    info = {"name": name, "type": type(val).__name__}
    try:
        if isinstance(val, pd.DataFrame):
            info["shape"] = list(val.shape)
            info["columns"] = list(map(str, val.columns[:10]))
        elif isinstance(val, pd.Series):
            info["shape"] = [len(val)]
            info["head"] = val.head(sample_n).tolist()
        elif isinstance(val, np.ndarray):
            info["shape"] = list(val.shape)
            info["sample"] = val.ravel()[:sample_n].tolist()
        else:
            info["value"] = val if isinstance(val, (int, float, str, bool, type(None))) else repr(val)
    except Exception as e:
        info["error"] = f"failed_introspect: {e}"
    log.log(level, "VALUE %s", json.dumps(info, default=str))

# --- auto-configure on import ---
setup_logging()

__all__ = ["setup_logging", "get_log", "TickerLogger", "log_value"]