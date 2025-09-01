from __future__ import annotations
from typing import Sequence, Any, Callable
import streamlit as st

def pick_index(choices: Sequence[Any], value: Any, fallback: int = 0) -> int:
    try: return list(choices).index(value)
    except Exception: return fallback

def bind_selectbox(label: str, choices: list, state_dict: dict, key_path: tuple[str,...], *, key=None, help=None):
    cur = _get(state_dict, key_path)
    idx = pick_index(choices, cur, 0)
    sel = st.selectbox(label, choices, index=idx, key=key, help=help)
    _set(state_dict, key_path, sel)
    return sel

def bind_number(label: str, state_dict: dict, key_path: tuple[str,...],
                *, min_value=None, max_value=None, value=None, step=1, key=None, help=None, int_cast=False):
    cur = _get(state_dict, key_path, default=value)
    num = st.number_input(label, min_value=min_value, max_value=max_value,
                          value=cur, step=step, key=key, help=help)
    _set(state_dict, key_path, int(num) if int_cast else float(num))
    return num

def bind_checkbox(label: str, state_dict: dict, key_path: tuple[str,...], *, key=None, help=None):
    cur = bool(_get(state_dict, key_path, default=False))
    val = st.checkbox(label, value=cur, key=key, help=help)
    _set(state_dict, key_path, bool(val))
    return val

def bind_slider(label: str, state_dict: dict, key_path: tuple[str,...],
                *, min_value, max_value, step=None, key=None, help=None):
    cur = _get(state_dict, key_path)
    val = st.slider(label, min_value, max_value, cur, step, key=key, help=help)
    _set(state_dict, key_path, val)
    return val

def _get(d: dict, path: tuple[str,...], default=None):
    ref = d
    for p in path: ref = ref.get(p, {})
    return ref if ref != {} else default

def _set(d: dict, path: tuple[str,...], v):
    ref = d
    for p in path[:-1]:
        if p not in ref or not isinstance(ref[p], dict): ref[p] = {}
        ref = ref[p]
    ref[path[-1]] = v