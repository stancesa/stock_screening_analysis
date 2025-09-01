from core.constants import BAND_CHOICES, MODEL_CHOICES, VOL_CHOICES, SEED_CHOICES
from ui.bind import bind_selectbox, bind_number, bind_checkbox, bind_slider
import streamlit as st
import json

def render_projection_defaults_section(s: dict, K):
    st.subheader("Projection defaults")

    # Top-line settings
    bind_selectbox("Percentile band", BAND_CHOICES, s, ("projections","band"),
                   key=K("settings.proj","band"))
    bind_number("Default simulations", s, ("projections","sims"),
                min_value=100, max_value=20000, step=1000,
                key=K("settings.proj","sims"), int_cast=True)
    bind_selectbox("Default projection model", MODEL_CHOICES, s, ("projections","model"),
                   key=K("settings.proj","model"))
    bind_number("Default months to project", s, ("projections","months"),
                min_value=1, max_value=24, step=1,
                key=K("settings.proj","months"), int_cast=True)

    with st.expander("Advanced defaults", expanded=False):
        bind_number("Calibration window (days)", s, ("projections","window"),
                    min_value=60, max_value=252*5, step=20,
                    key=K("settings.proj","window"), int_cast=True)
        bind_slider("EWMA lambda (vol persistence)", s, ("projections","lam"),
                    min_value=0.80, max_value=0.99, step=0.01,
                    key=K("settings.proj","lam"))
        bind_slider("Student-t df", s, ("projections","df_t"),
                    min_value=3, max_value=15, step=1,
                    key=K("settings.proj","df_t"))
        bind_checkbox("Use antithetic variates", s, ("projections","antithetic"),
                      key=K("settings.proj","antithetic"))
        bind_number("Bootstrap block size", s, ("projections","block"),
                    min_value=3, max_value=30, step=1,
                    key=K("settings.proj","block"), int_cast=True)
        bind_selectbox("Volatility estimator", VOL_CHOICES, s, ("projections","vol_mode"),
                       key=K("settings.proj","vol_mode"))
        bind_checkbox("Stochastic volatility (mean-reverting)", s, ("projections","stochastic_vol"),
                      key=K("settings.proj","stoch_vol"))

        seed_mode = bind_selectbox("Seed mode", SEED_CHOICES, s, ("projections","seed_mode"),
                                   key=K("settings.proj","seed_mode"))
        if seed_mode == "custom":
            bind_number("Default custom seed", s, ("projections","seed"),
                        min_value=0, max_value=2**32-1, step=1,
                        key=K("settings.proj","seed"), int_cast=True)
        else:
            # keep the seed visible but inactive to prevent drift
            st.number_input("Default custom seed", min_value=0, max_value=2**32-1,
                            value=int(s["projections"]["seed"]), step=1, disabled=True)

    if st.button("Apply projection defaults to current page now"):
        st.session_state["range_days"] = s["chart"]["range_days"]
        st.rerun()

def collect_profile_payload(DEFAULT_OUTPUT) -> dict:
    ss = st.session_state
    return {
        "min_sig":     float(ss.get("min_sig", 0.0)),
        "min_comp":    float(ss.get("min_comp", 0.0)),
        "rsi_min":     int(ss.get("rsi_range", (0,100))[0]),
        "rsi_max":     int(ss.get("rsi_range", (0,100))[1]),
        "owned_only":  bool(ss.get("owned_only", False)),
        "search":      ss.get("search", ""),
        "custom_rules": ss.get("custom_rules", []),
        "logical":      ss.get("cf_logical", "AND"),
        "sort_choice":  ss.get("sort_choice"),
        "sort_asc":     bool(ss.get("sort_asc", True)),
        "table_cols":   ss.get("table_cols", []),
        "show_sma":     bool(ss.get("show_sma", True)),
        "show_stop":    bool(ss.get("show_stop", False)),
        "show_target":  bool(ss.get("show_target", False)),
        "range_days":   int(ss.get("range_days", 180)),
        "overlays":     ss.get("overlays", []),
        "overlay_params": ss.get("overlay_params_loaded", {}),
        "source_choice": ss.get("source_choice", "Latest generated"),
        "default_path":  ss.get("default_path", DEFAULT_OUTPUT.as_posix()),
    }

def apply_profile_to_session(prof: dict, DEFAULT_OUTPUT):
    ss = st.session_state
    ss["custom_rules"] = prof.get("custom_rules", [])
    ss["cf_logical"]   = prof.get("logical", ss.get("cf_logical", "AND"))
    ss["sort_choice"]  = prof.get("sort_choice", ss.get("sort_choice"))
    ss["sort_asc"]     = prof.get("sort_asc", ss.get("sort_asc", True))
    ss["table_cols"]   = prof.get("table_cols", ss.get("table_cols", []))
    for k in ("show_sma","show_stop","show_target"): ss[k] = prof.get(k, ss.get(k, False))
    ss["range_days"]   = prof.get("range_days", ss.get("range_days", 180))
    ss["overlays"]     = prof.get("overlays", ss.get("overlays", []))
    ss["overlay_params_loaded"] = prof.get("overlay_params", {})
    ss["source_choice"] = prof.get("source_choice", ss.get("source_choice","Latest generated"))
    ss["default_path"]  = prof.get("default_path", ss.get("default_path", DEFAULT_OUTPUT.as_posix()))
    ss["min_sig"]   = float(prof.get("min_sig", 0.0))
    ss["min_comp"]  = float(prof.get("min_comp", 0.0))
    ss["rsi_range"] = (int(prof.get("rsi_min", 0)), int(prof.get("rsi_max", 100)))
    ss["owned_only"]= bool(prof.get("owned_only", 0))
    ss["search"]    = prof.get("search", "")

def render_profiles_tab(DEFAULT_OUTPUT, save_profile, list_profiles, load_profile):
    st.subheader("Filter profiles")
    name = st.text_input("Profile name", value="default")

    cA, cB = st.columns(2)
    if cA.button("Save profile"):
        payload = collect_profile_payload(DEFAULT_OUTPUT)
        save_profile(name, payload)
        st.success(f"Saved profile '{name}'")

    existing = list_profiles()
    chosen = st.selectbox("Load profile", options=["(select)"] + existing)
    if cB.button("Load", disabled=(chosen == "(select)")):
        prof = load_profile(chosen)
        if prof:
            apply_profile_to_session(prof, DEFAULT_OUTPUT)
            st.rerun()

def render_overlays_defaults(s: dict, all_opts: list[str]):
    st.subheader("Overlays")
    s["overlays"]["defaults"] = st.multiselect(
        "Default overlays",
        options=all_opts,
        default=s["overlays"].get("defaults", [])
    )
    st.caption("These appear selected by default in Review → Overlays.")

def render_perf_and_data_defaults(s: dict, DEFAULT_SETTINGS: dict):
    st.subheader("Plotly & Data")
    s["chart"]["plot_height"] = int(bind_number(
        "Plot height (px)", s, ("chart","plot_height"),
        min_value=360, max_value=1200, step=20, key="perf_height", int_cast=True
    ))
    s["chart"]["hovermode"] = bind_selectbox(
        "Hover mode", ["x unified","closest","x"], s, ("chart","hovermode"), key="perf_hover"
    )
    st.divider()
    s["data"]["source_choice"] = bind_selectbox(
        "Default data source", ["Latest generated","Upload file","Path"], s, ("data","source_choice")
    )
    s["data"]["default_path"] = st.text_input("Default path", value=s["data"]["default_path"])

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reset ALL settings to defaults"):
            from copy import deepcopy
            st.session_state.app_settings = deepcopy(DEFAULT_SETTINGS)
            st.success("Settings reset.")
            st.rerun()
    with c2:
        st.download_button(
            "Download settings JSON",
            data=json.dumps(s, indent=2).encode("utf-8"),
            file_name="app_settings.json", mime="application/json"
        )
    with c3:
        up = st.file_uploader("Import settings JSON", type=["json"])
        if up is not None:
            try:
                loaded = json.load(up)
                for k in DEFAULT_SETTINGS.keys():
                    if k in loaded and isinstance(loaded[k], dict):
                        s[k].update(loaded[k])
                st.success("Settings imported.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

                