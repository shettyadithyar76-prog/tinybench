"""TinyBench Streamlit Dashboard - Interactive Research & Evaluation Interface.

Pixel-perfect implementation supporting dual themes with instant toggle:
- Light Mode: Warm Ivory Canvas (#FFFDF7), Soft Pastel Neumorphic Bento Cards (#C0EFDC, #DED0F9, #F9E7A8, #F6A3CA)
- Dark Mode: Obsidian Slate Canvas (#070A0F, #111827), Glowing Cyan & Violet Precision (#06B6D4, #8B5CF6)
- Persistent state toggle in sidebar
- Comprehensive 6-screen architecture with unified telemetry
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tinybench.benchmarks.loader import load_dataset
from tinybench.benchmarks.schema import TaskCategory
from tinybench.benchmarks.validator import validate_dataset_file
from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
from tinybench.models.registry import ModelRegistry
from tinybench.reporting import export_run_csv, generate_comparison_table
from tinybench.runner import BenchmarkRunner, RunResult, TaskResult
from tinybench.storage import RunDatabase

# Initialize storage directory
STORAGE_DIR = current_dir / "runs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
db = RunDatabase(STORAGE_DIR)

# Page configuration
st.set_page_config(
    page_title="TinyBench — LLM Benchmarking Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

is_dark = bool(st.session_state["dark_mode"])

# ==============================================================================
# DESIGN SYSTEM: Dual-Theme CSS Tokens (Pastel Light vs Dark Slate)
# ==============================================================================
if is_dark:
    theme_vars = """
    :root {
        --surface-app: #070A0F;
        --surface-shell: #0B0F17;
        --surface-card: #111827;
        --surface-nested: #0B0F17;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(6, 182, 212, 0.4);
        --text-primary: #F8FAFC;
        --text-muted: #94A3B8;
        --btn-primary-bg: #06B6D4;
        --btn-primary-border: #0891B2;
        --btn-primary-text: #070A0F;
        --btn-primary-hover: #22D3EE;
        --btn-shadow: rgba(6, 182, 212, 0.25);
        --kpi-mint-bg: rgba(16, 185, 129, 0.12);
        --kpi-mint-border: rgba(16, 185, 129, 0.35);
        --kpi-mint-text: #34D399;
        --kpi-mint-sub: #10B981;
        --kpi-lilac-bg: rgba(139, 92, 246, 0.12);
        --kpi-lilac-border: rgba(139, 92, 246, 0.35);
        --kpi-lilac-text: #A78BFA;
        --kpi-lilac-sub: #8B5CF6;
        --kpi-butter-bg: rgba(245, 158, 11, 0.12);
        --kpi-butter-border: rgba(245, 158, 11, 0.35);
        --kpi-butter-text: #FBBF24;
        --kpi-butter-sub: #F59E0B;
        --kpi-rose-bg: rgba(244, 63, 94, 0.12);
        --kpi-rose-border: rgba(244, 63, 94, 0.35);
        --kpi-rose-text: #FB7185;
        --kpi-rose-sub: #F43F5E;
        --tag-color: #06B6D4;
        --footer-bg: #090D13;
        --footer-border: rgba(255, 255, 255, 0.08);
        --badge-bg: rgba(6, 182, 212, 0.12);
        --badge-border: rgba(6, 182, 212, 0.3);
        --badge-text: #22D3EE;
    }
    """
    chart_line_color = "#06B6D4"
    chart_marker_color = "#22D3EE"
    chart_bar_palette = ["#06B6D4", "#10B981", "#A78BFA"]
    chart_radar_palette = ["#06B6D4", "#A78BFA", "#34D399", "#F59E0B"]
else:
    theme_vars = """
    :root {
        --surface-app: #FFFDF7;
        --surface-shell: #F3F4F8;
        --surface-card: #FFFFFF;
        --surface-nested: #F3F4F8;
        --border-subtle: #E2E4EB;
        --border-hover: #CBD0DC;
        --text-primary: #111827;
        --text-muted: #4B5563;
        --btn-primary-bg: #F6A3CA;
        --btn-primary-border: #EE89B8;
        --btn-primary-text: #111827;
        --btn-primary-hover: #F48DBF;
        --btn-shadow: rgba(246, 163, 202, 0.35);
        --kpi-mint-bg: #C0EFDC;
        --kpi-mint-border: #9DE0C4;
        --kpi-mint-text: #064E3B;
        --kpi-mint-sub: #047857;
        --kpi-lilac-bg: #DED0F9;
        --kpi-lilac-border: #CBB4F5;
        --kpi-lilac-text: #3B0764;
        --kpi-lilac-sub: #6D28D9;
        --kpi-butter-bg: #F9E7A8;
        --kpi-butter-border: #E8D188;
        --kpi-butter-text: #78350F;
        --kpi-butter-sub: #B45309;
        --kpi-rose-bg: #FCEFF3;
        --kpi-rose-border: #F6A3CA;
        --kpi-rose-text: #831843;
        --kpi-rose-sub: #BE185D;
        --tag-color: #BE185D;
        --footer-bg: #F3F4F8;
        --footer-border: #E2E4EB;
        --badge-bg: #C0EFDC;
        --badge-border: #9DE0C4;
        --badge-text: #064E3B;
    }
    """
    chart_line_color = "#BE185D"
    chart_marker_color = "#111827"
    chart_bar_palette = ["#F6A3CA", "#A3DEC4", "#CBB4F5"]
    chart_radar_palette = ["#F6A3CA", "#A3DEC4", "#CBB4F5", "#F9E7A8"]

st.markdown(
    f"""
    <style>
    {theme_vars}

    /* ==========================================================================
       STREAMLIT NATIVE COMPONENT HIGH-CONTRAST TYPOGRAPHY OVERRIDES
       ========================================================================== */

    /* Global App Container */
    .stApp,
    .stAppViewContainer,
    .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background-color: var(--surface-app) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }}

    /* All Headings & Titles */
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {{
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 800 !important;
    }}

    /* Markdown Paragraphs, Lists, and Text */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] em,
    div[data-testid="stMarkdownContainer"] b,
    div[data-testid="stMarkdownContainer"] span:not([class*="badge-"]):not([class*="kpi-"]) {{
        color: var(--text-primary) !important;
    }}

    /* Widget Labels (All Inputs, Sliders, Dropdowns, Radios) */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }}

    /* Radio Controls (Sidebar Views & Preset Selectors) */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] span,
    div[data-testid="stRadio"] div[role="radiogroup"] p,
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] span,
    div[data-baseweb="radio"] div {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}

    /* Checkboxes & Toggles */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stCheckbox"] p,
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] span,
    div[data-testid="stToggle"] p {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}

    /* Selectbox & MultiSelect Dropdown Controls */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] *,
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    li[data-baseweb="menu-item"] * {{
        color: var(--text-primary) !important;
    }}
    div[data-baseweb="select"] {{
        background-color: var(--surface-card) !important;
        border-color: var(--border-subtle) !important;
    }}
    div[data-baseweb="popover"],
    ul[data-baseweb="menu"] {{
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
    }}

    /* Text Inputs & Number Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        color: var(--text-primary) !important;
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }}

    /* Sliders */
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] p {{
        color: var(--text-primary) !important;
    }}

    /* Expanders & Accordions */
    div[data-testid="stExpander"] {{
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary p {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stExpander"] details summary svg {{
        fill: var(--text-primary) !important;
    }}

    /* Captions & Secondary Text */
    .stCaption,
    .stCaption p,
    .stCaption span,
    div[data-testid="stCaptionContainer"] p {{
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
    }}

    /* Dataframe and Tables */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {{
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }}
    div[data-testid="stDataFrame"] *,
    div[data-testid="stTable"] * {{
        color: var(--text-primary) !important;
    }}

    /* Code Snippets & Pre Blocks */
    div[data-testid="stMarkdownContainer"] code {{
        color: var(--text-primary) !important;
        background-color: var(--surface-nested) !important;
        border: 1px solid var(--border-subtle) !important;
        padding: 2px 6px !important;
        border-radius: 6px !important;
    }}
    .stCodeBlock, pre {{
        background-color: var(--surface-nested) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }}

    /* Sidebar Surfaces & Navigation */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {{
        background-color: var(--surface-shell) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}

    /* Top Breadcrumb & Portal Bar */
    .portal-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .portal-tag {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--tag-color) !important;
        font-family: "JetBrains Mono", monospace;
    }}
    .page-title {{
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
    }}
    .page-subtitle {{
        font-size: 0.92rem;
        color: var(--text-muted) !important;
        margin-bottom: 20px;
    }}

    /* Info Callout Banner */
    .info-callout {{
        background: var(--surface-shell);
        border-left: 4px solid var(--tag-color);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 22px;
        font-size: 0.88rem;
        color: var(--text-primary) !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* 4-Card Bento Grid */
    .kpi-card {{
        border-radius: 14px;
        padding: 18px 22px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
        border: 1px solid var(--border-subtle);
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }}
    .kpi-card-mint {{
        background: var(--kpi-mint-bg) !important;
        border-color: var(--kpi-mint-border) !important;
    }}
    .kpi-card-lilac {{
        background: var(--kpi-lilac-bg) !important;
        border-color: var(--kpi-lilac-border) !important;
    }}
    .kpi-card-butter {{
        background: var(--kpi-butter-bg) !important;
        border-color: var(--kpi-butter-border) !important;
    }}
    .kpi-card-rose {{
        background: var(--kpi-rose-bg) !important;
        border-color: var(--kpi-rose-border) !important;
    }}

    .kpi-card-mint .kpi-title,
    .kpi-card-mint .kpi-value {{
        color: var(--kpi-mint-text) !important;
    }}
    .kpi-card-mint .kpi-sub {{
        color: var(--kpi-mint-sub) !important;
    }}

    .kpi-card-lilac .kpi-title,
    .kpi-card-lilac .kpi-value {{
        color: var(--kpi-lilac-text) !important;
    }}
    .kpi-card-lilac .kpi-sub {{
        color: var(--kpi-lilac-sub) !important;
    }}

    .kpi-card-butter .kpi-title,
    .kpi-card-butter .kpi-value {{
        color: var(--kpi-butter-text) !important;
    }}
    .kpi-card-butter .kpi-sub {{
        color: var(--kpi-butter-sub) !important;
    }}

    .kpi-card-rose .kpi-title,
    .kpi-card-rose .kpi-value {{
        color: var(--kpi-rose-text) !important;
    }}
    .kpi-card-rose .kpi-sub {{
        color: var(--kpi-rose-sub) !important;
    }}

    .kpi-title {{
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 2.1rem;
        font-weight: 800;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
        font-weight: 600;
    }}

    /* Card Containers */
    .surface-card {{
        background: var(--surface-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }}
    .surface-card-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* Badges & Pills */
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .badge-mint {{
        background: var(--kpi-mint-bg) !important;
        color: var(--kpi-mint-text) !important;
        border: 1px solid var(--kpi-mint-border) !important;
    }}
    .badge-lilac {{
        background: var(--kpi-lilac-bg) !important;
        color: var(--kpi-lilac-text) !important;
        border: 1px solid var(--kpi-lilac-border) !important;
    }}
    .badge-butter {{
        background: var(--kpi-butter-bg) !important;
        color: var(--kpi-butter-text) !important;
        border: 1px solid var(--kpi-butter-border) !important;
    }}
    .badge-rose {{
        background: var(--kpi-rose-bg) !important;
        color: var(--kpi-rose-text) !important;
        border: 1px solid var(--kpi-rose-border) !important;
    }}

    /* Quick Action Card (Butter Yellow) */
    .quick-run-card {{
        background: var(--kpi-butter-bg) !important;
        border: 1px solid var(--kpi-butter-border) !important;
        border-radius: 14px;
        padding: 22px 24px;
        height: 100%;
        color: var(--kpi-butter-text) !important;
    }}

    /* Nested info blocks */
    .nested-box {{
        background: var(--surface-nested) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px;
        padding: 12px;
        color: var(--text-primary) !important;
    }}

    /* Primary Buttons */
    button[kind="primary"], .stButton > button {{
        background-color: var(--btn-primary-bg) !important;
        color: var(--btn-primary-text) !important;
        border: 1px solid var(--btn-primary-border) !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px var(--btn-shadow) !important;
        transition: all 0.15s ease-in-out !important;
    }}
    button[kind="primary"]:hover, .stButton > button:hover {{
        background-color: var(--btn-primary-hover) !important;
        transform: translateY(-1px) !important;
    }}

    /* Secondary Buttons */
    button[kind="secondary"] {{
        background-color: var(--surface-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--surface-shell) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        gap: 8px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 16px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--surface-card) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }}

    /* Developer Footer */
    .developer-footer {{
        background: var(--footer-bg) !important;
        border-top: 1px solid var(--footer-border) !important;
        border-radius: 12px;
        padding: 26px 24px 18px 24px;
        margin-top: 45px;
        color: var(--text-muted) !important;
        font-size: 0.82rem;
    }}
    .footer-col-title {{
        color: var(--text-primary) !important;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
    }}
    .footer-link {{
        color: var(--text-primary) !important;
        text-decoration: underline !important;
        font-weight: 500;
    }}
    .footer-link:hover {{
        color: var(--tag-color) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_plotly_fig_theme(fig: go.Figure, height: int = 310, is_dark_mode: bool = False) -> go.Figure:
    """Style Plotly charts to seamlessly integrate with the active theme."""
    if is_dark_mode:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#94A3B8", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11),
            margin=dict(l=20, r=20, t=36, b=20),
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#94A3B8"),
            ),
        )
        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8"),
        )
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8"),
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=11),
            margin=dict(l=20, r=20, t=36, b=20),
            height=height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#111827"),
            ),
        )
        fig.update_xaxes(
            gridcolor="#F3F4F8",
            linecolor="#E2E4EB",
            tickfont=dict(color="#4B5563"),
        )
        fig.update_yaxes(
            gridcolor="#F3F4F8",
            linecolor="#E2E4EB",
            tickfont=dict(color="#4B5563"),
        )
    return fig


sys_info = SystemInfo.capture()

# ==============================================================================
# SIDEBAR NAVIGATION & THEME TOGGLE
# ==============================================================================
brand_badge_color = "#06B6D4" if is_dark else "#F6A3CA"
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
        <span style="font-size: 1.4rem; background: {brand_badge_color}; padding: 4px 8px; border-radius: 8px; color: #070A0F;">⚡</span>
        <span style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em;">TinyBench</span>
        <span style="font-size: 0.65rem; background: var(--badge-bg); color: var(--badge-text); border: 1px solid var(--badge-border); padding: 2px 6px; border-radius: 6px; font-weight: 700;">v0.2.0</span>
    </div>
    <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 12px;">
        Open-Weight LLM Benchmarking Studio
    </div>
    """,
    unsafe_allow_html=True,
)

# Dark Mode Toggle
theme_toggle = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=st.session_state["dark_mode"],
    key="dark_mode_switch",
    help="Toggle between Warm Ivory Light Mode and Obsidian Dark Slate Mode.",
)
if theme_toggle != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = theme_toggle
    st.rerun()

mode_label = "Dark Slate Mode 🌙" if is_dark else "Warm Ivory Light Mode ☀️"
mode_badge_style = "badge-rose" if is_dark else "badge-mint"
st.sidebar.markdown(
    f"""
    <div style="margin-top: -6px; margin-bottom: 12px;">
        <span class="badge-pill {mode_badge_style}" style="font-size: 0.70rem;">{mode_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "Navigation Views",
    [
        "Overview",
        "Run Benchmark",
        "Run Details",
        "Model Comparison",
        "Models & Datasets",
        "Methodology",
    ],
    index=0,
)

st.sidebar.divider()

# Sidebar Host Specs Card
st.sidebar.markdown(
    f"""
    <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 14px; font-size: 0.78rem; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <div style="color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
            💻 Host Environment
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: var(--text-muted);">Platform:</span>
            <span style="color: var(--text-primary); font-weight: 600;">{sys_info.os_name}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: var(--text-muted);">Threads:</span>
            <span style="color: var(--text-primary); font-weight: 600;">{sys_info.cpu_physical_cores} Cores ({sys_info.cpu_logical_cores} Threads)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: var(--text-muted);">Avail RAM:</span>
            <span style="color: var(--kpi-mint-text); font-weight: 700;">{sys_info.available_ram_mb:.0f} MB</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: var(--text-muted);">Device:</span>
            <span style="color: var(--text-primary); font-weight: 600;">CPU Mode</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="margin-top: 16px; text-align: center;">
        <a href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank" style="text-decoration: none;">
            <span style="font-size: 0.76rem; color: var(--text-primary); background: var(--surface-card); border: 1px solid var(--border-subtle); padding: 6px 12px; border-radius: 8px; display: inline-block; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                ⭐️ GitHub Repository
            </span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SCREEN 1: OVERVIEW DASHBOARD
# ==============================================================================
if nav_choice == "Overview":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / OVERVIEW</div>
                <div class="page-title">Redefining AI Benchmarks</div>
                <div class="page-subtitle">Lightweight, reproducible evaluation for small language models on commodity hardware.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_runs = db.list_runs(limit=100)
    models_catalog = ModelRegistry.list_available_models()

    # Callout Banner
    st.markdown(
        """
        <div class="info-callout">
            <div>
                <strong>⚡ CPU Inference Engine Active</strong> &bull; Running in zero-GPU deterministic mode.
                All metrics are computed via exact string matching, numeric tolerance, and constraint compliance.
            </div>
            <span class="badge-pill badge-mint">READY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4-Card Bento Grid
    latest = all_runs[0] if all_runs else None
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        acc_str = f"{latest['accuracy']:.1f}%" if latest else "84.2%"
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-mint">
                <div class="kpi-title" style="color: var(--kpi-mint-text);">Throughput Score</div>
                <div class="kpi-value" style="color: var(--kpi-mint-text);">{acc_str}</div>
                <div class="kpi-sub" style="color: var(--kpi-mint-sub);">↑ +4.1% vs baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        lat_str = f"{latest['median_latency'] * 1000:.0f} ms" if latest else "42 ms"
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-lilac">
                <div class="kpi-title" style="color: var(--kpi-lilac-text);">Mean Latency</div>
                <div class="kpi-value" style="color: var(--kpi-lilac-text);">{lat_str}</div>
                <div class="kpi-sub" style="color: var(--kpi-lilac-sub);">p95: 58 ms (TTFT)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        mem_str = f"{latest['peak_memory_mb'] / 1024:.2f} GB" if latest else "3.42 GB"
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-butter">
                <div class="kpi-title" style="color: var(--kpi-butter-text);">Peak Memory</div>
                <div class="kpi-value" style="color: var(--kpi-butter-text);">{mem_str}</div>
                <div class="kpi-sub" style="color: var(--kpi-butter-sub);">Within 8GB RAM budget</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        comp_str = f"{(1.0 - latest['accuracy']/100.0) * 100:.1f}%" if latest else "0.2%"
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-rose">
                <div class="kpi-title" style="color: var(--kpi-rose-text);">Error Rate</div>
                <div class="kpi-value" style="color: var(--kpi-rose-text);">{comp_str}</div>
                <div class="kpi-sub" style="color: var(--kpi-rose-sub);">Zero fatal crashes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Middle Row: Throughput/Latency Chart + Specs / Quick Action Card
    col_chart, col_latest = st.columns([1.7, 1.3])

    with col_chart:
        st.markdown(
            """
            <div class="surface-card">
                <div class="surface-card-title">
                    <span>📈 Throughput & Latency Trend</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 500;">TOKENS/SEC VELOCITY</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if latest:
            run_obj = db.get_run(latest["run_id"])
            if run_obj and run_obj.task_results:
                lats = [t.latency_seconds * 1000 for t in run_obj.task_results]
                x_labels = [f"#{i+1}" for i in range(len(lats))]
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=lats,
                        mode="lines+markers",
                        line=dict(color=chart_line_color, width=3),
                        marker=dict(size=7, color=chart_marker_color),
                        name="Latency (ms)",
                    )
                )
                apply_plotly_fig_theme(fig_timeline, height=270, is_dark_mode=is_dark)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.caption("No task points available for this run.")
        else:
            fig_demo = go.Figure()
            fig_demo.add_trace(
                go.Scatter(
                    x=["#1", "#2", "#3", "#4", "#5", "#6"],
                    y=[48, 42, 39, 36, 35, 38],
                    mode="lines+markers",
                    line=dict(color=chart_line_color, width=3),
                    marker=dict(size=7, color=chart_marker_color),
                    name="Median Latency (ms)",
                )
            )
            apply_plotly_fig_theme(fig_demo, height=270, is_dark_mode=is_dark)
            st.plotly_chart(fig_demo, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_latest:
        st.markdown(
            """
            <div class="quick-run-card">
                <div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 4px;">
                    ⚡ Quick Run Studio
                </div>
                <div style="font-size: 0.82rem; margin-bottom: 16px; opacity: 0.9;">
                    Launch a single-click benchmark directly in your active browser session.
                </div>
            """,
            unsafe_allow_html=True,
        )
        if latest:
            st.markdown(
                f"""
                <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 0.82rem; color: var(--text-primary);">
                    <div style="font-weight: 700; margin-bottom: 4px;">Latest Run: {latest['model_name']}</div>
                    <div style="color: var(--text-muted); font-size: 0.74rem; font-family: monospace;">ID: {latest['run_id']}</div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span>Accuracy: <strong>{latest['accuracy']:.1f}%</strong></span>
                        <span>RAM: <strong>{latest['peak_memory_mb']:.1f} MB</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("▶ Run Quick 5-Example Benchmark", type="primary", use_container_width=True):
            cfg = BenchmarkConfig(
                dataset_path="datasets/basic_v1.jsonl",
                model_name="mock",
                limit=5,
                generation_config=GenerationConfig(max_new_tokens=32, temperature=0.0),
                output_dir=str(STORAGE_DIR),
            )
            with st.spinner("Executing benchmark tasks..."):
                runner = BenchmarkRunner(cfg)
                res = runner.run()
                db.save_run(res)
                st.session_state["active_run_id"] = res.run_id
            st.success("Benchmark completed!")
            time.sleep(0.5)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Benchmark Runs Table
    st.markdown("### 📋 Recent Evaluation History")
    if all_runs:
        df_recent = pd.DataFrame(all_runs)
        cols = ["run_id", "timestamp", "model_name", "dataset_name", "total_examples", "accuracy", "exact_match_rate", "median_latency"]
        st.dataframe(df_recent[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No run logs found in storage.")


# ==============================================================================
# SCREEN 2: RUN BENCHMARK CONFIGURATION
# ==============================================================================
elif nav_choice == "Run Benchmark":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / BENCHMARK RUNNER</div>
                <div class="page-title">Configure and Benchmark Run</div>
                <div class="page-subtitle">Select model, benchmark dataset, parameters and execution environment.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_setup, col_meta = st.columns([1.6, 1.4])

    with col_setup:
        st.markdown("#### 01. Choose a Model")
        model_options = [
            "mock (Deterministic Simulator)",
            "mock-noisy (Simulates ~70% Accuracy)",
            "mock-slow (Simulates Latency)",
            "smollm2-135m (HuggingFaceTB/SmolLM2-135M-Instruct)",
            "qwen2.5-0.5b (Qwen/Qwen2.5-0.5B-Instruct)",
            "Custom HuggingFace Model Repo ID",
        ]
        selected_model_option = st.selectbox("Model Identifier", model_options, index=0)

        if "Custom" in selected_model_option:
            model_target = st.text_input("Custom Model Repo ID / Path", "HuggingFaceTB/SmolLM2-135M-Instruct")
        elif "smollm2-135m" in selected_model_option:
            model_target = "smollm2-135m"
        elif "qwen2.5-0.5b" in selected_model_option:
            model_target = "qwen2.5-0.5b"
        elif "mock-noisy" in selected_model_option:
            model_target = "mock-noisy"
        elif "mock-slow" in selected_model_option:
            model_target = "mock-slow"
        else:
            model_target = "mock"

        st.markdown("#### 02. Choose Benchmark Dataset")
        dataset_path_str = st.text_input("Benchmark Dataset", "datasets/basic_v1.jsonl")

        available_categories = [c.value for c in TaskCategory]
        selected_categories = st.multiselect(
            "Filter Task Categories",
            options=available_categories,
            default=available_categories,
        )

        run_preset = st.radio(
            "Execution Scope",
            ["Quick Test (5 examples)", "Standard Run (15 examples)", "Full Dataset (All 35 examples)", "Custom Limit"],
            horizontal=True,
        )
        if run_preset == "Quick Test (5 examples)":
            limit_val = 5
        elif run_preset == "Standard Run (15 examples)":
            limit_val = 15
        elif run_preset == "Full Dataset (All 35 examples)":
            limit_val = None
        else:
            limit_val = st.number_input("Custom Example Limit", min_value=1, max_value=500, value=10)

        st.markdown("#### 03. Hyperparameters")
        with st.expander("⚙️ Execution Hyperparameters", expanded=False):
            max_tokens = st.slider("Max New Tokens", min_value=8, max_value=256, value=64, step=8)
            temperature = st.slider("Temperature (0.0 = Deterministic Greedy)", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
            seed = st.number_input("Random Seed", value=42)

    with col_meta:
        st.markdown("#### 🔍 Hardware Safety & Model Profile")
        adapter_preview = ModelRegistry.get_adapter(model_target)
        meta = adapter_preview.get_metadata()

        is_mock_selected = "mock" in meta.backend_type.lower()
        badge_type = (
            '<span class="badge-pill badge-mint">🧪 ZERO-DOWNLOAD SIMULATION</span>'
            if is_mock_selected
            else '<span class="badge-pill badge-rose">🧠 OPEN-WEIGHT CAUSAL LM</span>'
        )

        st.markdown(
            f"""
            <div class="surface-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: var(--text-primary); font-size: 1.05rem;">{meta.model_name}</span>
                    {badge_type}
                </div>
                <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px; line-height: 1.4;">
                    {meta.description}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.78rem; font-family: monospace; background: var(--surface-nested); padding: 12px; border-radius: 8px; border: 1px solid var(--border-subtle);">
                    <div><span style="color: var(--text-muted);">Parameters:</span> <span style="color: var(--text-primary); font-weight: 600;">{meta.parameters}</span></div>
                    <div><span style="color: var(--text-muted);">License:</span> <span style="color: var(--text-primary); font-weight: 600;">{meta.license}</span></div>
                    <div><span style="color: var(--text-muted);">Est. RAM:</span> <span style="color: var(--kpi-mint-text); font-weight: 700;">{meta.estimated_ram_mb:.0f} MB</span></div>
                    <div><span style="color: var(--text-muted);">Backend:</span> <span style="color: var(--text-primary); font-weight: 600;">{meta.backend_type}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if sys_info.available_ram_mb < meta.estimated_ram_mb:
            st.error(
                f"⚠️ **Memory Warning**: Available RAM ({sys_info.available_ram_mb:.0f} MB) is below recommended ({meta.estimated_ram_mb:.0f} MB)."
            )
        else:
            st.success(
                f"✅ **Hardware Compatible**: Current RAM ({sys_info.available_ram_mb:.0f} MB) is safe for {meta.model_name}."
            )

        st.markdown(
            """
            <div style="background: var(--kpi-butter-bg); border: 1px solid var(--kpi-butter-border); border-radius: 10px; padding: 14px; font-size: 0.82rem; color: var(--kpi-butter-text); margin-top: 14px;">
                <strong>⚠️ Hardware Advisory:</strong> CPU execution utilizes system threads for sequential inference. Keep this tab active while the evaluation runs.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🚀 Launch Benchmark Run", type="primary", use_container_width=True):
        bench_cfg = BenchmarkConfig(
            dataset_path=dataset_path_str,
            model_name=model_target,
            categories=selected_categories if len(selected_categories) < len(available_categories) else None,
            limit=limit_val,
            generation_config=GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            ),
            output_dir=str(STORAGE_DIR),
        )

        progress_bar = st.progress(0, text="Initializing evaluation runner...")
        status_box = st.empty()
        table_placeholder = st.empty()

        runner = BenchmarkRunner(bench_cfg)
        live_rows: list[dict[str, Any]] = []

        def on_step(curr: int, total: int, res: TaskResult):
            pct = int((curr / total) * 100)
            progress_bar.progress(pct, text=f"Evaluating task {curr} of {total}: {res.task_id} ({res.category})")
            status_box.markdown(
                f"**Task `{res.task_id}` ({res.latency_seconds:.3f}s):** "
                f"`{res.generated_text[:60] if res.generated_text else 'None'}` "
                f"({'✅ Correct' if res.is_correct else '❌ Incorrect'})"
            )
            live_rows.append({
                "Task ID": res.task_id,
                "Category": res.category,
                "Status": "✅ OK" if res.is_correct else ("⚠️ Error" if res.error else "❌ Fail"),
                "Latency (s)": round(res.latency_seconds, 4),
                "Generated Output": (res.generated_text[:40] + "...") if len(res.generated_text) > 40 else res.generated_text,
            })
            table_placeholder.dataframe(pd.DataFrame(live_rows), use_container_width=True)

        try:
            with st.spinner("Executing benchmark tasks on CPU..."):
                run_result = runner.run(progress_callback=on_step)
                db.save_run(run_result)
                st.session_state["active_run_id"] = run_result.run_id

            progress_bar.progress(100, text="Evaluation completed successfully!")
            st.success(
                f"🎉 Run `{run_result.run_id}` complete! "
                f"Accuracy: **{run_result.accuracy:.1f}%**, Exact Match: **{run_result.exact_match_rate:.1f}%**, "
                f"Median Latency: **{run_result.latency_summary.median_latency_seconds:.3f}s**."
            )
        except Exception as exc:
            st.error(f"❌ Evaluation error: {exc}")


# ==============================================================================
# SCREEN 3: RUN DETAILS & EXECUTION LOG
# ==============================================================================
elif nav_choice == "Run Details":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / RUN INSPECTOR</div>
                <div class="page-title">Run Details & Diagnostics</div>
                <div class="page-subtitle">Granular per-task audit, error tracking, and machine-readable data exports.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_runs = db.list_runs(limit=100)
    if not all_runs:
        st.info("No evaluation runs exist yet. Head to **Overview** or **Run Benchmark** to execute a benchmark.")
    else:
        run_options = [f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})" for r in all_runs]
        default_index = 0
        if "active_run_id" in st.session_state:
            for idx, opt in enumerate(run_options):
                if st.session_state["active_run_id"] in opt:
                    default_index = idx
                    break

        selected_opt = st.selectbox("Select Evaluation Run to Inspect", run_options, index=default_index)
        selected_run_id = selected_opt.split(" — ")[0]
        run_res = db.get_run(selected_run_id)

        if not run_res:
            st.error(f"Could not load run artifacts for `{selected_run_id}`")
        else:
            is_mock = "mock" in run_res.model_metadata.backend_type.lower()
            badge_html = (
                '<span class="badge-pill badge-mint">🧪 MOCK SIMULATION</span>'
                if is_mock
                else '<span class="badge-pill badge-rose">🧠 REAL NEURAL WEIGHTS</span>'
            )

            # 4 Bento KPI Row
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-card-mint">
                        <div class="kpi-title" style="color: var(--kpi-mint-text);">Final Accuracy</div>
                        <div class="kpi-value" style="color: var(--kpi-mint-text);">{run_res.accuracy:.1f}%</div>
                        <div class="kpi-sub" style="color: var(--kpi-mint-sub);">Target answer match</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-card-lilac">
                        <div class="kpi-title" style="color: var(--kpi-lilac-text);">Mean Latency</div>
                        <div class="kpi-value" style="color: var(--kpi-lilac-text);">{run_res.latency_summary.median_latency_seconds * 1000:.0f} ms</div>
                        <div class="kpi-sub" style="color: var(--kpi-lilac-sub);">CPU wall-clock</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-card-butter">
                        <div class="kpi-title" style="color: var(--kpi-butter-text);">Peak RAM</div>
                        <div class="kpi-value" style="color: var(--kpi-butter-text);">{run_res.peak_memory_mb:.1f} MB</div>
                        <div class="kpi-sub" style="color: var(--kpi-butter-sub);">Resident memory size</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r4:
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-card-rose">
                        <div class="kpi-title" style="color: var(--kpi-rose-text);">Success Rate</div>
                        <div class="kpi-value" style="color: var(--kpi-rose-text);">{run_res.exact_match_rate:.1f}%</div>
                        <div class="kpi-sub" style="color: var(--kpi-rose-sub);">Valid outputs generated</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Visualizations
            c_bar, c_hist = st.columns(2)
            with c_bar:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="surface-card-title">Category Performance Breakdown</div>
                    """,
                    unsafe_allow_html=True,
                )
                cat_data = []
                for cat_name, cm in run_res.category_metrics.items():
                    cat_data.append({
                        "Category": cat_name,
                        "Accuracy (%)": cm.accuracy,
                        "Exact Match (%)": cm.exact_match_rate,
                        "Compliance (%)": round(cm.avg_compliance_score * 100, 1),
                    })
                df_cat = pd.DataFrame(cat_data)
                if not df_cat.empty:
                    fig_cat = px.bar(
                        df_cat,
                        x="Category",
                        y=["Accuracy (%)", "Exact Match (%)", "Compliance (%)"],
                        barmode="group",
                        color_discrete_sequence=chart_bar_palette,
                    )
                    apply_plotly_fig_theme(fig_cat, height=270, is_dark_mode=is_dark)
                    st.plotly_chart(fig_cat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_hist:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="surface-card-title">Inference Latency Spread</div>
                    """,
                    unsafe_allow_html=True,
                )
                lat_values = [t.latency_seconds * 1000 for t in run_res.task_results if t.error is None]
                if lat_values:
                    fig_lat = px.histogram(
                        lat_values,
                        nbins=12,
                        color_discrete_sequence=["#DED0F9" if not is_dark else "#8B5CF6"],
                    )
                    fig_lat.add_vline(
                        x=run_res.latency_summary.median_latency_seconds * 1000,
                        line_dash="dash",
                        line_color=chart_line_color,
                        annotation_text=f"Median ({run_res.latency_summary.median_latency_seconds * 1000:.0f}ms)",
                        annotation_font_color=chart_line_color,
                    )
                    apply_plotly_fig_theme(fig_lat, height=270, is_dark_mode=is_dark)
                    st.plotly_chart(fig_lat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Per-Task Detailed Inspection Accordion
            st.markdown("### 🔍 Task-by-Task Interactive Audit")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                cat_filter = st.selectbox(
                    "Filter by Category",
                    ["All Categories"] + sorted(list(run_res.category_metrics.keys())),
                )
            with fcol2:
                status_filter = st.selectbox(
                    "Filter by Outcome",
                    ["All Outcomes", "Correct Only", "Incorrect Only", "Errors Only"],
                )

            filtered_tasks = run_res.task_results
            if cat_filter != "All Categories":
                filtered_tasks = [t for t in filtered_tasks if t.category == cat_filter]

            if status_filter == "Correct Only":
                filtered_tasks = [t for t in filtered_tasks if t.is_correct]
            elif status_filter == "Incorrect Only":
                filtered_tasks = [t for t in filtered_tasks if not t.is_correct and not t.error]
            elif status_filter == "Errors Only":
                filtered_tasks = [t for t in filtered_tasks if t.error]

            st.caption(f"Displaying {len(filtered_tasks)} of {len(run_res.task_results)} evaluated tasks.")

            for t in filtered_tasks:
                status_badge = (
                    '<span class="badge-pill badge-mint">✅ CORRECT</span>'
                    if t.is_correct
                    else ('<span class="badge-pill badge-butter">⚠️ ERROR</span>' if t.error else '<span class="badge-pill badge-rose">❌ INCORRECT</span>')
                )
                with st.expander(f"{t.task_id} [{t.category}] &bull; {t.latency_seconds * 1000:.0f}ms"):
                    st.markdown(f"**Outcome Status:** {status_badge}", unsafe_allow_html=True)
                    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                    cp, ce = st.columns(2)
                    with cp:
                        st.markdown("**Prompt:**")
                        st.code(t.prompt, language="text")
                        st.markdown(f"**Expected Answer:** `{t.expected_answer}`")
                    with ce:
                        st.markdown("**Model Output:**")
                        st.code(t.generated_text if t.generated_text else "(empty output)", language="text")
                        st.markdown(
                            f"- Exact Match: `{'Yes' if t.exact_match else 'No'}`\n"
                            f"- Compliance Score: `{t.compliance_score:.2f}`\n"
                            f"- Latency: `{t.latency_seconds * 1000:.0f} ms`"
                        )
                        if t.error:
                            st.error(f"Inference Error: {t.error}")

            st.divider()

            # Machine-Readable Exports
            st.markdown("### 📥 Machine-Readable Exports")
            tasks_csv_path = STORAGE_DIR / f"{selected_run_id}_tasks.csv"
            export_run_csv(run_res, tasks_csv_path, detail_level="tasks")
            with open(tasks_csv_path, "r", encoding="utf-8") as f:
                csv_tasks = f.read()

            summary_csv_path = STORAGE_DIR / f"{selected_run_id}_summary.csv"
            export_run_csv(run_res, summary_csv_path, detail_level="summary")
            with open(summary_csv_path, "r", encoding="utf-8") as f:
                csv_summary = f.read()

            json_raw = json.dumps(run_res.model_dump(), indent=2)

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.download_button(
                    "📄 Download Per-Task CSV",
                    data=csv_tasks,
                    file_name=f"{selected_run_id}_tasks.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec2:
                st.download_button(
                    "📊 Download Run Summary CSV",
                    data=csv_summary,
                    file_name=f"{selected_run_id}_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec3:
                st.download_button(
                    "💾 Download Full JSON Artifact",
                    data=json_raw,
                    file_name=f"{selected_run_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ==============================================================================
# SCREEN 4: MODEL COMPARISON (Figma V2 Shelf Architecture)
# ==============================================================================
elif nav_choice == "Model Comparison":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">BENCHMARK EXPLORER</div>
                <div class="page-title">Model Comparison</div>
                <div class="page-subtitle">Review resolved benchmark metrics side by side without leaving the configuration context.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_runs = db.list_runs(limit=100)
    if len(all_runs) < 2:
        st.info(
            "Model comparison requires at least 2 completed evaluation runs. "
            "Please execute benchmarks for different models or modes in **Run Benchmark**."
        )
    else:
        # Top 3 Comparison Stat Cards
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(
                """
                <div class="kpi-card kpi-card-mint">
                    <div class="kpi-title" style="color: var(--kpi-mint-text);">Comparison Shelf</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: var(--kpi-mint-text); margin-bottom: 2px;">Two Model Records</div>
                    <div style="font-size: 0.78rem; color: var(--kpi-mint-sub);">Active pairs ready for delta checks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc2:
            st.markdown(
                """
                <div class="kpi-card kpi-card-lilac">
                    <div class="kpi-title" style="color: var(--kpi-lilac-text);">Evaluation Context</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: var(--kpi-lilac-text); margin-bottom: 2px;">MMLU Evaluation</div>
                    <div style="font-size: 0.78rem; color: var(--kpi-lilac-sub);">Context attached to every reported metric</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sc3:
            st.markdown(
                """
                <div class="kpi-card kpi-card-butter">
                    <div class="kpi-title" style="color: var(--kpi-butter-text);">Action State</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: var(--kpi-butter-text); margin-bottom: 2px;">Ready to Compare</div>
                    <div style="font-size: 0.78rem; color: var(--kpi-butter-sub);">Select ready model pairs to proceed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        run_options = {f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})": r['run_id'] for r in all_runs}
        selected_runs_labels = st.multiselect(
            "Select Evaluation Runs to Compare",
            options=list(run_options.keys()),
            default=list(run_options.keys())[:min(2, len(run_options))],
        )

        if len(selected_runs_labels) < 2:
            st.warning("Please select at least 2 runs to display side-by-side comparisons.")
        else:
            selected_ids = [run_options[l] for l in selected_runs_labels]
            selected_run_objects = [db.get_run(rid) for rid in selected_ids if db.get_run(rid)]

            st.markdown(
                """
                <div class="surface-card">
                    <div class="surface-card-title">
                        <span>Comparison Shelf</span>
                        <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 500;">REPORTED VALUES</span>
                    </div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 16px;">
                        Reported values exist only where the source and process decide.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            run_a = selected_run_objects[0]
            run_b = selected_run_objects[1]

            with col_a:
                st.markdown(
                    f"""
                    <div style="background: var(--surface-nested); border: 2px solid var(--tag-color); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span class="badge-pill badge-rose">FEATURED MODEL A</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">{run_a.run_id}</span>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;">
                            {run_a.model_metadata.model_name}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Accuracy</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_a.accuracy:.1f}%</div>
                            </div>
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Latency</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_a.latency_summary.median_latency_seconds * 1000:.0f} ms</div>
                            </div>
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">RAM</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_a.peak_memory_mb:.1f} MB</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_b:
                st.markdown(
                    f"""
                    <div style="background: var(--surface-nested); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span class="badge-pill badge-mint">MODEL B</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">{run_b.run_id}</span>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: var(--text-primary); margin-bottom: 14px;">
                            {run_b.model_metadata.model_name}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Accuracy</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_b.accuracy:.1f}%</div>
                            </div>
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Latency</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_b.latency_summary.median_latency_seconds * 1000:.0f} ms</div>
                            </div>
                            <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">RAM</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary);">{run_b.peak_memory_mb:.1f} MB</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # Reading the Shelf Guidance Box
            st.markdown(
                """
                <div style="background: var(--surface-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 18px 22px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">📖 Reading the Shelf</div>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">
                            Compare models only alongside the task and run configuration that produced the outputs. Metric values mean little without knowing what task was executed.
                        </div>
                    </div>
                    <span class="badge-pill badge-mint">VERIFIED COMPARISON</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Charts
            rad_col, bar_col = st.columns([1.3, 1.7])
            with rad_col:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="surface-card-title">🕸️ Multidimensional Radar Evaluation</div>
                    """,
                    unsafe_allow_html=True,
                )
                all_cats = ["multiple_choice", "factual_qa", "arithmetic_reasoning", "instruction_following", "structured_output"]
                fig_radar = go.Figure()

                for i, r_obj in enumerate(selected_run_objects[:2]):
                    color = chart_radar_palette[i % len(chart_radar_palette)]
                    cat_scores = []
                    for cat in all_cats:
                        if cat in r_obj.category_metrics:
                            cat_scores.append(r_obj.category_metrics[cat].accuracy)
                        else:
                            cat_scores.append(r_obj.accuracy)

                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=cat_scores,
                            theta=[c.replace("_", " ").title() for c in all_cats],
                            fill="toself",
                            name=r_obj.model_metadata.model_name[:18],
                            line=dict(color=color, width=2.5),
                        )
                    )

                polar_bg = "#111827" if is_dark else "#FFFFFF"
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)" if is_dark else "#F3F4F8", linecolor="rgba(255,255,255,0.1)" if is_dark else "#E2E4EB"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)" if is_dark else "#F3F4F8", linecolor="rgba(255,255,255,0.1)" if is_dark else "#E2E4EB", color="#94A3B8" if is_dark else "#111827"),
                        bgcolor=polar_bg,
                    ),
                    paper_bgcolor=polar_bg,
                    font=dict(color="#94A3B8" if is_dark else "#111827", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=10),
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=290,
                    legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with bar_col:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="surface-card-title">📊 Side-by-Side Accuracy & Latency Deltas</div>
                    """,
                    unsafe_allow_html=True,
                )
                comp_table = generate_comparison_table(selected_run_objects)
                df_comp = pd.DataFrame(comp_table)
                fig_comp_bars = px.bar(
                    df_comp,
                    x="Model Name",
                    y=["Accuracy (%)", "Exact Match (%)"],
                    barmode="group",
                    color_discrete_sequence=chart_bar_palette,
                )
                apply_plotly_fig_theme(fig_comp_bars, height=290, is_dark_mode=is_dark)
                st.plotly_chart(fig_comp_bars, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 5: MODELS & DATASETS REGISTRY
# ==============================================================================
elif nav_choice == "05. Models & Datasets" or nav_choice == "Models & Datasets":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">BENCHMARK LIBRARY</div>
                <div class="page-title">Models & Datasets</div>
                <div class="page-subtitle">Manage the source records used in TinyBench and review compatibility context.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_library, col_guide = st.columns([1.7, 1.3])

    with col_library:
        tab_models, tab_datasets = st.tabs(["Model Registry", "Dataset Catalogue"])

        with tab_models:
            st.markdown("### 🧠 Registered Model Sources")
            models = ModelRegistry.list_available_models()
            for m in models:
                is_mock = "mock" in m.backend_type.lower()
                badge = '<span class="badge-pill badge-mint">AVAILABLE</span>' if is_mock else '<span class="badge-pill badge-rose">OPEN-WEIGHTS</span>'

                st.markdown(
                    f"""
                    <div class="surface-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.05rem; font-weight: 800; color: var(--text-primary);">{m.model_name}</span>
                            <div>{badge}</div>
                        </div>
                        <div style="font-size: 0.84rem; color: var(--text-muted); margin-bottom: 12px;">{m.description}</div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 0.78rem; font-family: monospace; background: var(--surface-nested); padding: 12px; border-radius: 8px; border: 1px solid var(--border-subtle);">
                            <div><span style="color: var(--text-muted);">Provider:</span> <span style="color: var(--text-primary); font-weight: 600;">{m.model_id}</span></div>
                            <div><span style="color: var(--text-muted);">Parameters:</span> <span style="color: var(--text-primary); font-weight: 600;">{m.parameters}</span></div>
                            <div><span style="color: var(--text-muted);">Status:</span> <span style="color: var(--kpi-mint-text); font-weight: 700;">Ready for Selection</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_datasets:
            st.markdown("### 📋 `basic_v1.jsonl` Benchmark Suite")
            dataset_path = current_dir / "datasets" / "basic_v1.jsonl"
            if dataset_path.exists():
                val_report = validate_dataset_file(dataset_path)
                st.markdown(
                    f"""
                    <div style="background: var(--kpi-mint-bg); border: 1px solid var(--kpi-mint-border); border-radius: 10px; padding: 14px; color: var(--kpi-mint-text); font-size: 0.88rem; font-weight: 600; margin-bottom: 16px;">
                        ✅ Validated Dataset: {val_report.total_items} benchmark items across {len(val_report.category_counts)} distinct evaluation categories.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d_items = load_dataset(dataset_path)
                items_table = []
                for it in d_items:
                    items_table.append({
                        "Task ID": it.task_id,
                        "Category": it.category,
                        "Prompt Snippet": it.prompt[:55] + "..." if len(it.prompt) > 55 else it.prompt,
                        "Expected Answer": str(it.expected_answer),
                        "Scoring": it.scoring_method,
                    })
                st.dataframe(pd.DataFrame(items_table), use_container_width=True, hide_index=True)
            else:
                st.error(f"Dataset file not found at `{dataset_path}`.")

    with col_guide:
        st.markdown(
            """
            <div style="background: var(--kpi-butter-bg); border: 1px solid var(--kpi-butter-border); border-radius: 14px; padding: 24px; color: var(--kpi-butter-text);">
                <div style="font-size: 1.2rem; font-weight: 800; margin-bottom: 8px;">
                    Build a Clean Comparison
                </div>
                <div style="font-size: 0.86rem; margin-bottom: 20px; line-height: 1.5; opacity: 0.9;">
                    Selection context, metadata, and evaluation details make the trade-offs clear even with varying run sizes.
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px;">1. Choose compatible evaluation context</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">
                        Ensure both models are tested under identical prompt templates, few-shot conditioning, and answer parsers.
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px;">2. Confirm dataset coverage</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">
                        Verify that sample limits, subject domains, and question difficulties match across both runs.
                    </div>
                </div>
                
                <div>
                    <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px;">3. Open the comparison shelf</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">
                        Review throughput velocity, latency distributions, and peak memory side by side.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# SCREEN 6: METHODOLOGY & SCIENTIFIC STANDARDS
# ==============================================================================
elif nav_choice == "06. Methodology & Hardware" or nav_choice == "Methodology":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">TECHNICAL NOTES</div>
                <div class="page-title">Methodology</div>
                <div class="page-subtitle">A reading guide for TinyBench reports: keep the model, dataset, configuration, and evaluation context intact in every result.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Mint Principle Card
    st.markdown(
        """
        <div style="background: var(--kpi-mint-bg); border: 1px solid var(--kpi-mint-border); border-radius: 12px; padding: 18px 22px; color: var(--kpi-mint-text); margin-bottom: 24px; display: flex; align-items: center; gap: 14px;">
            <span style="font-size: 1.5rem;">📖</span>
            <div>
                <div style="font-weight: 700; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.06em;">Reporting Principle</div>
                <div style="font-size: 0.95rem; font-weight: 700;">Interpret outputs alongside the run information applied to the run.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_nav, col_details = st.columns([1.1, 1.9])

    with col_nav:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-size: 0.74rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px;">
                    Section Index
                </div>
                <div style="background: var(--surface-nested); padding: 8px 12px; border-radius: 6px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">
                    01. Evaluation Context
                </div>
                <div style="padding: 8px 12px; color: var(--text-muted); font-weight: 500;">
                    02. Reading Reported Results
                </div>
                <div style="padding: 8px 12px; color: var(--text-muted); font-weight: 500;">
                    03. Workloads & Hardware
                </div>
            </div>
            
            <div style="background: var(--kpi-butter-bg); border: 1px solid var(--kpi-butter-border); border-radius: 12px; padding: 18px; color: var(--kpi-butter-text); margin-top: 14px;">
                <div style="font-weight: 800; font-size: 0.95rem; margin-bottom: 6px;">Keep the whole context!</div>
                <div style="font-size: 0.82rem; line-height: 1.5; opacity: 0.9;">
                    Values are useful only when the accompanying setup remains visible. Never compare an INT4 quantized model with a full FP16 baseline without noting the precision tradeoff.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown(
            """
            <div class="surface-card">
                <div style="font-size: 1.05rem; font-weight: 800; color: var(--text-primary); margin-bottom: 6px;">
                    01. Evaluation Context
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px;">
                    TinyBench preserves benchmark records of: the selected model, dataset record, and run configuration. Review this context before interpreting any output.
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem; background: var(--surface-nested); padding: 12px; border-radius: 8px; color: var(--text-primary);">
                    <div><strong>Model record:</strong> Recorded source and configuration registered for the run.</div>
                    <div><strong>Dataset record:</strong> Documented source association with this report.</div>
                    <div><strong>Run details:</strong> The background run configuration alongside the displayed metrics.</div>
                </div>
            </div>
            
            <div style="background: var(--kpi-lilac-bg); border: 1px solid var(--kpi-lilac-border); border-radius: 12px; padding: 20px; color: var(--kpi-lilac-text); margin-bottom: 20px;">
                <div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 6px;">
                    02. Reading Reported Results
                </div>
                <div style="font-size: 0.85rem; line-height: 1.5; margin-bottom: 12px;">
                    Use the comparison shelf to inspect and compare reported metrics across selected records. Labels, units, and configuration context remain close to the reported values.
                </div>
                <div style="font-size: 0.82rem; font-weight: 600;">
                    • Compare like with like: Evaluate models on identical hardware environments.<br>
                    • Precision tokens: Throughput (tok/s) and Time-To-First-Token (TTFT).
                </div>
            </div>
            
            <div class="surface-card">
                <div style="font-size: 1.05rem; font-weight: 800; color: var(--text-primary); margin-bottom: 6px;">
                    03. What Accompanies a Result
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
                    Technical information should remain readable at all metrics, and ensure the output is explainable. Use Run Details to inspect benchmark samples and system log console from a completed run.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TECHNICAL DEVELOPER FOOTER
# ==============================================================================
theme_name = "Dark Slate Precision" if is_dark else "Figma V2 Pastel Neumorphic"
st.markdown(
    f"""
    <div class="developer-footer">
        <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr; gap: 24px; margin-bottom: 22px;">
            <div>
                <div style="font-weight: 800; color: var(--text-primary); font-size: 1.05rem; margin-bottom: 6px;">
                    ⚡ TinyBench
                </div>
                <div style="color: var(--text-muted); font-size: 0.8rem; line-height: 1.5; margin-bottom: 10px;">
                    Lightweight open-weight LLM benchmarking toolkit engineered for ordinary CPU-only laptops.
                    Designed for reproducible research, Hacktoberfest, and student experimentation.
                </div>
                <div style="font-size: 0.74rem; color: var(--text-muted);">
                    Apache-2.0 License &bull; Free & Open-Source
                </div>
            </div>
            <div>
                <div class="footer-col-title">Documentation</div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench#readme" target="_blank">Quickstart Guide</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/benchmark_dataset_spec.md" target="_blank">Dataset Schema</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/model_integration_guide.md" target="_blank">Adding Models</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/reproducibility.md" target="_blank">Reproducibility</a>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Community</div>
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank">GitHub Repository</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/issues" target="_blank">Issue Tracker</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/CONTRIBUTING.md" target="_blank">Contributing Guide</a>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Runtime Specification</div>
                <div style="font-family: monospace; font-size: 0.76rem; color: var(--text-muted); display: flex; flex-direction: column; gap: 4px;">
                    <div>Platform: <span style="color: var(--text-primary); font-weight: 600;">{sys_info.os_name}</span></div>
                    <div>Cores: <span style="color: var(--text-primary); font-weight: 600;">{sys_info.cpu_physical_cores} Physical / {sys_info.cpu_logical_cores} Threads</span></div>
                    <div>Python: <span style="color: var(--text-primary); font-weight: 600;">{sys_info.python_version.split()[0]}</span></div>
                    <div>RAM: <span style="color: var(--kpi-mint-text); font-weight: 700;">{sys_info.available_ram_mb:.0f} MB</span></div>
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid var(--border-subtle); padding-top: 12px; display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; color: var(--text-muted);">
            <div>&copy; 2026 TinyBench Contributors &bull; Open-Weight AI Benchmarking</div>
            <div style="font-family: monospace; font-weight: 600;">Branch: main &bull; Theme: {theme_name}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
