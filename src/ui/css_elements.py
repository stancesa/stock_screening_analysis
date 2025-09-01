import streamlit as st

def _fix_streamlit_clipping():
    st.markdown("""
    <style>
      /* Let tooltips escape Streamlit layout wrappers */
      .block-container { overflow: visible !important; }
      [data-testid="stVerticalBlock"] { overflow: visible !important; }
      [data-testid="stHorizontalBlock"] { overflow: visible !important; }
      [data-testid="column"] { overflow: visible !important; }
      [data-testid="stMarkdown"] { overflow: visible !important; }
      /* Make sure the chip row itself never clips children */
      .kpi-row, .kpi-chip { overflow: visible !important; position: relative; }
      /* Belt-and-suspenders: be on top of Plotly & DataFrame canvases */
      .kpi-tip { z-index: 2147483000 !important; }
    </style>
    """, unsafe_allow_html=True)

def _fix_streamlit_tooltip_overflow():
    st.markdown("""
    <style>
      /* Let tooltips escape layout wrappers */
      .block-container { overflow: visible !important; }
      [data-testid="stVerticalBlock"],
      [data-testid="stHorizontalBlock"],
      [data-testid="column"],
      [data-testid="stExpander"],
      [data-testid="stExpander"] > details { overflow: visible !important; }

      /* Put BaseWeb tooltips above Plotly/DataFrame canvases */
      [data-baseweb="tooltip"] { z-index: 2147483646 !important; }
    </style>
    """, unsafe_allow_html=True)

def _inject_chip_css():
    css = """
    <style>
      /* ... your existing .reco-badge, .kpi-row, .kpi-chip, .kpi-tip CSS exactly as-is ... */
    </style>
    """
    # persist a slot, but rewrite every rerun
    if "_chip_css_slot" not in st.session_state:
        st.session_state["_chip_css_slot"] = st.empty()
    st.session_state["_chip_css_slot"].markdown(css, unsafe_allow_html=True)

    st.markdown("""
    <style>
      /* Recommendation badge */
      .reco-badge {
        display:inline-block; padding:6px 10px; border-radius:999px;
        font-weight:700; font-size:12px; line-height:1;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 2px 6px rgba(0,0,0,.25);
        border:1px solid rgba(0,0,0,.12);
        color:#fff;
      }
      .reco-strongbuy  { background:#008000; }  /* Solid Green   */
      .reco-buy        { background:#90EE90; }  /* Light Green   */
      .reco-hold       { background:#FFFF00; }  /* Yellow        */
      .reco-sell       { background:#FFA500; }  /* Orange        */
      .reco-strongsell { background:#FF0000; }  /* Vivid Red     */

      .kpi-row { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-start; margin:6px 0; }
      .kpi-chip {
        position: relative;
        display:inline-flex; align-items:center; gap:10px;
        padding:8px 12px; border-radius:999px; font-weight:600; font-size:13px; line-height:1;
        backdrop-filter: blur(6px);
        border:1px solid rgba(255,255,255,.14);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 2px 6px rgba(0,0,0,.25);
        color: var(--chip-fg, #0f172a);
        overflow: visible; /* ensure tooltip can render out of bounds */
        z-index: 2147483646;
      }
      .kpi-chip .val {
        font-variant-numeric: tabular-nums;
        padding:3px 8px; border-radius:999px;
        background: var(--chip-badge-bg, rgba(255,255,255,.25));
        border:1px solid rgba(255,255,255,.22);
        color: inherit;
      }
      .kpi-chip:hover { transform: translateY(-1px); transition: transform .15s ease; z-index: 2147483647; }

      /* Tooltip */
      .kpi-tip {
        position:absolute;
        top:50%; left:100%;
        transform: translate(10px, -50%);
        max-width:520px; min-width:320px;
        background:#0f172a; color:#e5e7eb;          /* visible tip */
        border:1px solid #334155; border-radius:12px;
        padding:12px 14px;
        box-shadow:0 14px 40px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.08);
        z-index:2147483000;

        /* hidden by default; fade in on hover */
        visibility:hidden; opacity:0;
        transition: opacity .15s ease, visibility .15s ease;
        pointer-events:none; /* avoid flicker when not shown */
        white-space:normal; word-break:normal; overflow-wrap:anywhere;
      }
      .kpi-chip:hover > .kpi-tip {
        visibility:visible; opacity:1;
        pointer-events:auto; /* allow interacting with the tip */
      }

      /* Optional variants: align to the right or above when near edges */
      .kpi-tip.right { right:100%; left:auto; transform: translate(-10px, -50%); }
      .kpi-tip.top   { top:auto; bottom:100%; left:50%; transform: translate(-50%, -10px); }

      /* Tooltip inner typography */
      .kpi-tip h5 { margin:0 0 6px 0; font-size:13px; color:#fff; }
      .kpi-tip .section { margin-top:8px; font-size:12px; line-height:1.35; }
      .kpi-tip table { width:100%; border-collapse:collapse; font-size:12px; }
      .kpi-tip th, .kpi-tip td { padding:2px 0; }
      .kpi-tip thead th { text-align:left; border-bottom:1px solid #475569; color:#cbd5e1; }
      .kpi-tip tfoot td { border-top:1px solid #475569; font-weight:600; }
      .kpi-tip td[style*="text-align:right"] { text-align:right; }
    </style>
    """, unsafe_allow_html=True)

def _inject_tooltip_css():
    css = """
    <style>
      /* ... your .chip and .chip .tip CSS ... */
    </style>
    """
    if "_tip_css_slot" not in st.session_state:
        st.session_state["_tip_css_slot"] = st.empty()
    st.session_state["_tip_css_slot"].markdown(css, unsafe_allow_html=True)

    st.markdown("""
    <style>
      .chip { position: relative; display:inline-flex; align-items:center; gap:.4rem; 
              padding:.25rem .6rem; border-radius:999px; border:1px solid rgba(0,0,0,.1); }
      .chip .tip { 
        position:absolute; left:50%; transform:translateX(-50%); bottom:calc(100% + 8px);
        white-space:nowrap; padding:.4rem .6rem; border-radius:.5rem; 
        background:rgba(0,0,0,.85); color:#fff; font-size:.85rem; 
        pointer-events:none; opacity:0; visibility:hidden; transition:opacity .12s ease;
      }
      .chip:hover .tip { opacity:1; visibility:visible; }
    </style>
    """, unsafe_allow_html=True)

def _inject_regimen_css():
    css = """
    <style>
      /* ... your .mr-* CSS ... */
    </style>
    """
    if "_regimen_css_slot" not in st.session_state:
        st.session_state["_regimen_css_slot"] = st.empty()
    st.session_state["_regimen_css_slot"].markdown(css, unsafe_allow_html=True)

    st.markdown("""
    <style>
      .mr-wrap { margin: 6px 0 18px 0; }
      .mr-drop {
        border: 1px solid rgba(0,0,0,.12);
        border-radius: 12px;
        background: #0b1220; color: #e5e7eb;
        box-shadow: 0 6px 28px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.06);
        overflow: visible;
      }
      .mr-drop > summary {
        list-style: none; cursor: pointer;
        padding: 12px 14px;
        display: flex; align-items: center; gap: 12px; justify-content: space-between;
        background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,0));
      }
      .mr-drop > summary::-webkit-details-marker { display:none; }

      .mr-title { font-weight: 700; font-size: 14px; letter-spacing: .3px; opacity: .95; }
      .mr-badges { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }

      .mr-pill {
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 10px; border-radius:9999px; font-weight:700; font-size:12px;
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.12);
        color:#0b1220; background:#d1d5db;
      }
      .mr-pill.light { color:#0b1220; background:#d1d5db; }

      .mr-meter {
        margin: 0 14px 14px 14px;
        padding: 12px 12px 14px 12px;
        border-top: 1px dashed rgba(255,255,255,.10);
      }
      .mr-meter .scale {
        position: relative; width: 100%; height: 14px; border-radius: 9999px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.25), 0 2px 10px rgba(0,0,0,.35);
        background: linear-gradient(90deg,
          #16a34a 0%,   /* green   */
          #84cc16 30%,  /* lime    */
          #f59e0b 60%,  /* amber   */
          #ef4444 80%,  /* red     */
          #7f1d1d 100%  /* deep red*/
        );
      }
      .mr-meter .needle {
        position: absolute; top: -4px;
        width: 2px; height: 22px; background: #fff; box-shadow: 0 0 0 2px rgba(0,0,0,.3);
        transform: translateX(-50%); border-radius: 1px;
      }
      .mr-meter .labels {
        display:flex; justify-content: space-between; font-size: 11px; opacity: .85;
        margin-top: 6px;
      }

      .mr-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr));
        gap: 12px; padding: 0 14px 14px 14px; }
      @media (max-width: 720px) { .mr-grid { grid-template-columns: 1fr; } }

      .mr-card {
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 10px; padding: 10px 12px; background: rgba(255,255,255,.02);
      }
      .mr-card h5 { margin: 0 0 6px 0; font-size: 12px; letter-spacing:.2px; opacity:.85; }
      .mr-kv { display:flex; justify-content: space-between; font-size: 12px; margin: 2px 0; }
      .mr-kv .k { opacity: .7; }
      .mr-kv .v { font-variant-numeric: tabular-nums; }

      .mr-disclaimer {
        margin: 10px 14px 14px 14px;
        font-size: 12px; line-height: 1.45; opacity: .9;
        border-top: 1px dashed rgba(255,255,255,.10);
        padding-top: 10px;
      }
    </style>
    """, unsafe_allow_html=True)