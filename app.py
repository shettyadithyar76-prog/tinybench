"""TinyBench — Local Open-Weight LLM Benchmarking Platform.

Production-quality Streamlit dashboard for evaluating small language models
on commodity CPU hardware. Dark-first design with optional light mode.

Architecture: 6-screen single-page application with sidebar navigation,
real-time execution telemetry, and comprehensive result analytics.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure src/ and repo root are on sys.path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st

# ---------------------------------------------------------------------------
# Favicon: inline SVG data URI — benchmark gauge on dark chip
# ---------------------------------------------------------------------------
_FAVICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='6' fill='%23090D13'/>"
    "<path d='M16 8a8 8 0 0 1 7.75 10' stroke='%2300D2FF' stroke-width='2.5' "
    "stroke-linecap='round' fill='none'/>"
    "<path d='M16 8a8 8 0 0 0-7.75 10' stroke='%23334155' stroke-width='2.5' "
    "stroke-linecap='round' fill='none'/>"
    "<line x1='16' y1='18' x2='20' y2='11' stroke='%23F8FAFC' stroke-width='2' "
    "stroke-linecap='round'/>"
    "<circle cx='16' cy='18' r='2' fill='%2300D2FF'/></svg>"
)

st.set_page_config(
    page_title="TinyBench | AI Model Benchmarking Studio",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from tinybench.benchmarks.loader import load_dataset
    from tinybench.benchmarks.schema import TaskCategory
    from tinybench.benchmarks.validator import validate_dataset_file
    from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
    from tinybench.models.downloader import download_model, is_model_installed
    from tinybench.models.ollama_backend import check_ollama_daemon, get_ollama_base_url
    from tinybench.models.registry import APPROVED_MODEL_IDS, ModelRegistry
    from tinybench.reporting import export_run_csv, generate_comparison_table
    from tinybench.runner import BenchmarkRunner, RunResult, TaskResult
    from tinybench.storage import RunDatabase

    STORAGE_DIR = current_dir / "runs"
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    db = RunDatabase(STORAGE_DIR)
except Exception as startup_err:
    st.error("### TinyBench failed to start")
    st.markdown(
        f"**Import error:** `{type(startup_err).__name__}: {startup_err}`"
    )
    st.code(traceback.format_exc(), language="python")
    st.json({
        "python_version": sys.version,
        "platform": sys.platform,
        "current_dir": str(current_dir),
        "src_dir": str(src_dir),
        "sys_path": sys.path[:6],
    })
    st.stop()


# ===========================================================================
# SVG ICON SYSTEM (Lucide-inspired inline vectors)
# ===========================================================================
def tb_icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Return an inline SVG icon string. Uses Lucide-style 24×24 viewBox."""
    _paths = {
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
        "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
        "gauge": '<path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
        "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
        "server": '<rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>',
        "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
        "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "x-circle": '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
        "bar-chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
        "git-branch": '<line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
        "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
        "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
        "book-open": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
        "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
        "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
        "external-link": '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>',
        "monitor": '<rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>',
        "hard-drive": '<line x1="22" y1="12" x2="2" y2="12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" y1="16" x2="6.01" y2="16"/><line x1="10" y1="16" x2="10.01" y2="16"/>',
        "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
        "arrow-right": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
        "box": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
        "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
        "sun": '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
        "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    }
    svg_inner = _paths.get(name, _paths["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;">{svg_inner}</svg>'
    )


# ===========================================================================
# THEME STATE
# ===========================================================================
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True  # Dark-first default

is_dark = bool(st.session_state["dark_mode"])


# ===========================================================================
# DESIGN TOKENS & GLOBAL CSS
# ===========================================================================
if is_dark:
    theme_vars = """
    :root {
        --tb-canvas: #090D13;
        --tb-surface: #0E141F;
        --tb-elevated: #151D2A;
        --tb-inset: #0A0F17;
        --tb-border: rgba(255,255,255,0.07);
        --tb-border-md: rgba(255,255,255,0.12);
        --tb-border-hover: rgba(0,210,255,0.35);
        --tb-text: #F8FAFC;
        --tb-text-2: #94A3B8;
        --tb-text-3: #64748B;
        --tb-accent: #00D2FF;
        --tb-accent-hover: #38BDF8;
        --tb-accent-muted: rgba(0,210,255,0.10);
        --tb-accent-border: rgba(0,210,255,0.25);
        --tb-accent-glow: 0 0 20px rgba(0,210,255,0.15);
        --tb-success: #10B981;
        --tb-success-bg: rgba(16,185,129,0.10);
        --tb-success-border: rgba(16,185,129,0.25);
        --tb-warn: #F59E0B;
        --tb-warn-bg: rgba(245,158,11,0.10);
        --tb-warn-border: rgba(245,158,11,0.25);
        --tb-danger: #EF4444;
        --tb-danger-bg: rgba(239,68,68,0.10);
        --tb-danger-border: rgba(239,68,68,0.25);
    }
    """
    chart_template = "plotly_dark"
    chart_bg = "#0E141F"
    chart_grid = "rgba(255,255,255,0.05)"
    chart_line_clr = "rgba(255,255,255,0.08)"
    chart_font_clr = "#94A3B8"
    chart_accent = "#00D2FF"
    chart_accent2 = "#10B981"
    chart_accent3 = "#A78BFA"
    chart_palette = ["#00D2FF", "#10B981", "#A78BFA", "#F59E0B"]
else:
    theme_vars = """
    :root {
        --tb-canvas: #F8FAFC;
        --tb-surface: #FFFFFF;
        --tb-elevated: #F1F5F9;
        --tb-inset: #F8FAFC;
        --tb-border: #E2E8F0;
        --tb-border-md: #CBD5E1;
        --tb-border-hover: rgba(2,132,199,0.40);
        --tb-text: #0F172A;
        --tb-text-2: #475569;
        --tb-text-3: #64748B;
        --tb-accent: #0284C7;
        --tb-accent-hover: #0369A1;
        --tb-accent-muted: rgba(2,132,199,0.08);
        --tb-accent-border: rgba(2,132,199,0.25);
        --tb-accent-glow: 0 0 20px rgba(2,132,199,0.10);
        --tb-success: #059669;
        --tb-success-bg: rgba(5,150,105,0.08);
        --tb-success-border: rgba(5,150,105,0.25);
        --tb-warn: #D97706;
        --tb-warn-bg: rgba(217,119,6,0.08);
        --tb-warn-border: rgba(217,119,6,0.25);
        --tb-danger: #DC2626;
        --tb-danger-bg: rgba(220,38,38,0.08);
        --tb-danger-border: rgba(220,38,38,0.25);
    }
    """
    chart_template = "plotly_white"
    chart_bg = "#FFFFFF"
    chart_grid = "#F1F5F9"
    chart_line_clr = "#E2E8F0"
    chart_font_clr = "#475569"
    chart_accent = "#0284C7"
    chart_accent2 = "#059669"
    chart_accent3 = "#7C3AED"
    chart_palette = ["#0284C7", "#059669", "#7C3AED", "#D97706"]


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    {theme_vars}

    /* ===== GLOBAL RESET & CANVAS ===== */
    .stApp, .stAppViewContainer, .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background-color: var(--tb-canvas) !important;
        color: var(--tb-text) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* ===== HEADER & TOOLBAR ===== */
    header[data-testid="stHeader"], .stAppHeader,
    [data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stHeaderActionElements"],
    div[data-testid="collapsedControl"] {{
        background-color: var(--tb-canvas) !important;
        background: var(--tb-canvas) !important;
        color: var(--tb-text) !important;
        border-bottom: 1px solid var(--tb-border) !important;
    }}
    div[data-testid="stDecoration"] {{
        background-image: none !important;
        background-color: var(--tb-accent) !important;
        height: 2px !important;
    }}
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] span,
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] svg,
    [data-testid="stStatusWidget"] *,
    div[data-testid="stSidebarCollapseButton"] button,
    div[data-testid="stSidebarCollapseButton"] svg {{
        color: var(--tb-text-2) !important;
        fill: var(--tb-text-2) !important;
    }}
    header[data-testid="stHeader"] button:hover,
    [data-testid="stToolbar"] button:hover,
    div[data-testid="stSidebarCollapseButton"] button:hover {{
        background-color: var(--tb-elevated) !important;
        color: var(--tb-accent) !important;
    }}

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {{
        color: var(--tb-text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] em,
    div[data-testid="stMarkdownContainer"] b,
    div[data-testid="stMarkdownContainer"] span:not([class]) {{
        color: var(--tb-text) !important;
    }}

    /* ===== WIDGET LABELS ===== */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span {{
        color: var(--tb-text) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}

    /* ===== RADIO / CHECKBOX / TOGGLE ===== */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] span,
    div[data-baseweb="radio"] label,
    div[data-baseweb="radio"] span,
    div[data-baseweb="radio"] div,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] span {{
        color: var(--tb-text) !important;
        font-weight: 500 !important;
    }}

    /* ===== SELECT / MULTISELECT ===== */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] *,
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] *,
    li[data-baseweb="menu-item"] * {{
        color: var(--tb-text) !important;
    }}
    div[data-baseweb="select"] {{
        background-color: var(--tb-elevated) !important;
        border-color: var(--tb-border) !important;
    }}
    div[data-baseweb="popover"],
    ul[data-baseweb="menu"] {{
        background-color: var(--tb-elevated) !important;
        border: 1px solid var(--tb-border) !important;
    }}

    /* ===== TEXT / NUMBER / TEXTAREA INPUTS ===== */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        color: var(--tb-text) !important;
        background-color: var(--tb-elevated) !important;
        border: 1px solid var(--tb-border) !important;
        border-radius: 6px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {{
        border-color: var(--tb-accent) !important;
        box-shadow: 0 0 0 2px var(--tb-accent-muted) !important;
    }}

    /* ===== SLIDERS ===== */
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] span,
    div[data-testid="stSlider"] p {{
        color: var(--tb-text) !important;
    }}

    /* ===== EXPANDERS ===== */
    div[data-testid="stExpander"] {{
        background-color: var(--tb-surface) !important;
        border: 1px solid var(--tb-border) !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary p {{
        color: var(--tb-text) !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stExpander"] details summary svg {{
        fill: var(--tb-text-2) !important;
    }}

    /* ===== CAPTIONS ===== */
    .stCaption, .stCaption p, .stCaption span,
    div[data-testid="stCaptionContainer"] p {{
        color: var(--tb-text-3) !important;
        font-size: 0.8rem !important;
    }}

    /* ===== DATAFRAMES & TABLES ===== */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {{
        background-color: var(--tb-surface) !important;
        border: 1px solid var(--tb-border) !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stDataFrame"] *,
    div[data-testid="stTable"] * {{
        color: var(--tb-text) !important;
    }}

    /* ===== CODE BLOCKS ===== */
    div[data-testid="stMarkdownContainer"] code {{
        color: var(--tb-accent) !important;
        background-color: var(--tb-accent-muted) !important;
        border: 1px solid var(--tb-accent-border) !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }}
    .stCodeBlock, pre {{
        background-color: var(--tb-inset) !important;
        border: 1px solid var(--tb-border) !important;
        border-radius: 6px !important;
    }}

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {{
        background-color: var(--tb-surface) !important;
        border-right: 1px solid var(--tb-border) !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: var(--tb-text) !important;
    }}

    /* ===== PAGE HEADER ===== */
    .tb-eyebrow {{
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--tb-accent) !important;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }}
    .tb-page-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--tb-text) !important;
        letter-spacing: -0.03em;
        font-family: 'Plus Jakarta Sans', sans-serif;
        margin-bottom: 2px;
        line-height: 1.2;
    }}
    .tb-page-desc {{
        font-size: 0.88rem;
        color: var(--tb-text-2) !important;
        margin-bottom: 24px;
        line-height: 1.5;
    }}

    /* ===== KPI METRIC CARDS ===== */
    .tb-kpi {{
        background: var(--tb-surface);
        border: 1px solid var(--tb-border);
        border-radius: 8px;
        padding: 16px 20px;
        height: 100%;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .tb-kpi:hover {{
        border-color: var(--tb-border-md);
        box-shadow: var(--tb-accent-glow);
    }}
    .tb-kpi-label {{
        font-size: 0.68rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--tb-text-3) !important;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 6px;
    }}
    .tb-kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--tb-text) !important;
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }}
    .tb-kpi-sub {{
        font-size: 0.75rem;
        color: var(--tb-text-3) !important;
        margin-top: 4px;
        font-weight: 500;
    }}
    .tb-kpi-accent .tb-kpi-value {{
        color: var(--tb-accent) !important;
    }}

    /* ===== SURFACE CARD ===== */
    .tb-card {{
        background: var(--tb-surface);
        border: 1px solid var(--tb-border);
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }}
    .tb-card-title {{
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--tb-text) !important;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    /* ===== CALLOUT / BANNER ===== */
    .tb-callout {{
        background: var(--tb-surface);
        border-left: 3px solid var(--tb-accent);
        border-top: 1px solid var(--tb-border);
        border-right: 1px solid var(--tb-border);
        border-bottom: 1px solid var(--tb-border);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.84rem;
        color: var(--tb-text) !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }}

    /* ===== BADGES & PILLS ===== */
    .tb-badge {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
    }}
    .tb-badge-accent {{
        background: var(--tb-accent-muted) !important;
        color: var(--tb-accent) !important;
        border: 1px solid var(--tb-accent-border) !important;
    }}
    .tb-badge-success {{
        background: var(--tb-success-bg) !important;
        color: var(--tb-success) !important;
        border: 1px solid var(--tb-success-border) !important;
    }}
    .tb-badge-warn {{
        background: var(--tb-warn-bg) !important;
        color: var(--tb-warn) !important;
        border: 1px solid var(--tb-warn-border) !important;
    }}
    .tb-badge-danger {{
        background: var(--tb-danger-bg) !important;
        color: var(--tb-danger) !important;
        border: 1px solid var(--tb-danger-border) !important;
    }}

    /* ===== PRIMARY BUTTONS ===== */
    button[kind="primary"], .stButton > button {{
        background-color: var(--tb-accent) !important;
        color: #090D13 !important;
        border: 1px solid var(--tb-accent) !important;
        font-weight: 600 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        transition: all 0.15s ease !important;
    }}
    button[kind="primary"]:hover, .stButton > button:hover {{
        background-color: var(--tb-accent-hover) !important;
        border-color: var(--tb-accent-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--tb-accent-glow) !important;
    }}

    /* ===== SECONDARY / DOWNLOAD BUTTONS ===== */
    button[kind="secondary"] {{
        background-color: var(--tb-surface) !important;
        color: var(--tb-text) !important;
        border: 1px solid var(--tb-border) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }}

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--tb-inset) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid var(--tb-border) !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px !important;
        color: var(--tb-text-2) !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 6px 14px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--tb-surface) !important;
        color: var(--tb-text) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }}

    /* ===== FOOTER ===== */
    .tb-footer {{
        background: var(--tb-surface) !important;
        border-top: 1px solid var(--tb-border) !important;
        border-radius: 8px;
        padding: 24px 24px 16px;
        margin-top: 48px;
        color: var(--tb-text-3) !important;
        font-size: 0.78rem;
    }}
    .tb-footer a {{
        color: var(--tb-text-2) !important;
        text-decoration: none !important;
        transition: color 0.15s ease !important;
    }}
    .tb-footer a:hover {{
        color: var(--tb-accent) !important;
    }}

    /* ===== EMPTY STATE ===== */
    .tb-empty {{
        text-align: center;
        padding: 48px 24px;
        color: var(--tb-text-3);
    }}
    .tb-empty-icon {{
        margin-bottom: 16px;
        opacity: 0.4;
    }}
    .tb-empty-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--tb-text-2) !important;
        margin-bottom: 6px;
    }}
    .tb-empty-desc {{
        font-size: 0.84rem;
        color: var(--tb-text-3) !important;
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.5;
    }}

    /* ===== INLINE MONO STAT ===== */
    .tb-mono {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }}

    /* ===== UTILITY ===== */
    .tb-spacer-sm {{ height: 8px; }}
    .tb-spacer {{ height: 16px; }}
    .tb-spacer-lg {{ height: 24px; }}
    .tb-flex-between {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .tb-grid-2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        font-size: 0.78rem;
    }}
    .tb-grid-4 {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        font-size: 0.78rem;
    }}
    .tb-spec-row {{
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        font-size: 0.78rem;
    }}
    .tb-spec-label {{ color: var(--tb-text-3); }}
    .tb-spec-value {{
        color: var(--tb-text);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PLOTLY THEME HELPER
# ===========================================================================
def apply_plotly_fig_theme(fig: go.Figure, height: int = 300, is_dark_mode: bool = False) -> go.Figure:
    """Style a Plotly figure to match the active TinyBench theme."""
    fig.update_layout(
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=chart_bg,
        font=dict(
            color=chart_font_clr,
            family="Plus Jakarta Sans, -apple-system, sans-serif",
            size=11,
        ),
        margin=dict(l=16, r=16, t=32, b=16),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=chart_font_clr),
        ),
    )
    fig.update_xaxes(
        gridcolor=chart_grid,
        linecolor=chart_line_clr,
        tickfont=dict(color=chart_font_clr, family="JetBrains Mono, monospace", size=10),
    )
    fig.update_yaxes(
        gridcolor=chart_grid,
        linecolor=chart_line_clr,
        tickfont=dict(color=chart_font_clr, family="JetBrains Mono, monospace", size=10),
    )
    return fig


sys_info = SystemInfo.capture()


# ===========================================================================
# SIDEBAR — BRAND, THEME TOGGLE & NAVIGATION
# ===========================================================================
st.sidebar.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
        <div style="width:32px;height:32px;background:var(--tb-accent);border-radius:6px;display:flex;align-items:center;justify-content:center;">
            {tb_icon("zap", 18, "#090D13")}
        </div>
        <div>
            <span style="font-size:1.15rem;font-weight:700;color:var(--tb-text);letter-spacing:-0.02em;font-family:'Plus Jakarta Sans',sans-serif;">TinyBench</span>
            <span style="font-size:0.6rem;background:var(--tb-accent-muted);color:var(--tb-accent);border:1px solid var(--tb-accent-border);padding:1px 5px;border-radius:3px;font-weight:600;margin-left:6px;font-family:'JetBrains Mono',monospace;">v0.2.0</span>
        </div>
    </div>
    <div style="font-size:0.74rem;color:var(--tb-text-3);margin-bottom:12px;font-family:'Plus Jakarta Sans',sans-serif;">
        Local LLM Benchmarking Platform
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme toggle
_icon_name = "sun" if is_dark else "moon"
_mode_label = "Light Mode" if is_dark else "Dark Mode"
theme_toggle = st.sidebar.toggle(
    f"{_mode_label}",
    value=st.session_state["dark_mode"],
    key="dark_mode_switch",
    help="Switch between dark and light interface themes.",
)
if theme_toggle != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = theme_toggle
    st.rerun()

st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Run Benchmark",
        "Run Details",
        "Model Comparison",
        "Models & Datasets",
        "Methodology",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.divider()

# Sidebar: Host environment specs
st.sidebar.markdown(
    f"""
    <div style="background:var(--tb-elevated);border:1px solid var(--tb-border);border-radius:8px;padding:12px;font-size:0.76rem;">
        <div style="color:var(--tb-text-3);font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;font-family:'JetBrains Mono',monospace;font-size:0.65rem;display:flex;align-items:center;gap:6px;">
            {tb_icon("monitor", 13, "var(--tb-text-3)")} Host Environment
        </div>
        <div class="tb-spec-row">
            <span class="tb-spec-label">Platform</span>
            <span class="tb-spec-value">{sys_info.os_name}</span>
        </div>
        <div class="tb-spec-row">
            <span class="tb-spec-label">CPU</span>
            <span class="tb-spec-value">{sys_info.cpu_physical_cores}C / {sys_info.cpu_logical_cores}T</span>
        </div>
        <div class="tb-spec-row">
            <span class="tb-spec-label">RAM</span>
            <span class="tb-spec-value" style="color:var(--tb-success);">{sys_info.available_ram_mb:.0f} MB</span>
        </div>
        <div class="tb-spec-row">
            <span class="tb-spec-label">Runtime</span>
            <span class="tb-spec-value">CPU</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="margin-top:16px;text-align:center;">
        <a href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank"
           style="text-decoration:none;font-size:0.74rem;color:var(--tb-text-2);background:var(--tb-elevated);border:1px solid var(--tb-border);padding:5px 12px;border-radius:6px;display:inline-flex;align-items:center;gap:5px;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif;transition:border-color 0.15s ease;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            View on GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# HELPER: Page header block
# ===========================================================================
def _page_header(eyebrow: str, title: str, description: str):
    st.markdown(
        f"""
        <div style="margin-bottom:4px;">
            <div class="tb-eyebrow">{eyebrow}</div>
            <div class="tb-page-title">{title}</div>
            <div class="tb-page-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _empty_state(icon_name: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="tb-empty">
            <div class="tb-empty-icon">{tb_icon(icon_name, 48, "var(--tb-text-3)")}</div>
            <div class="tb-empty-title">{title}</div>
            <div class="tb-empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, sub: str, accent: bool = False):
    cls = "tb-kpi tb-kpi-accent" if accent else "tb-kpi"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="tb-kpi-label">{label}</div>
            <div class="tb-kpi-value">{value}</div>
            <div class="tb-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# SCREEN 1: OVERVIEW DASHBOARD
# ===========================================================================
if nav_choice == "Overview":
    _page_header(
        "System Overview",
        "Benchmarking Dashboard",
        "Live metrics from your most recent evaluation run, system status, and quick actions.",
    )

    all_runs = db.list_runs(limit=100)
    models_catalog = ModelRegistry.list_available_models()

    # Status callout
    st.markdown(
        f"""
        <div class="tb-callout">
            <div style="display:flex;align-items:center;gap:10px;">
                {tb_icon("cpu", 18, "var(--tb-accent)")}
                <span><strong>CPU inference active</strong> &mdash; deterministic mode, zero-GPU. Metrics computed via exact string matching, numeric tolerance, and constraint compliance.</span>
            </div>
            <span class="tb-badge tb-badge-success">Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest = all_runs[0] if all_runs else None

    if latest and latest.get("is_demo"):
        st.markdown(
            f"""
            <div class="tb-callout" style="border-left-color:var(--tb-warn);">
                <div style="display:flex;align-items:center;gap:10px;">
                    {tb_icon("alert-triangle", 16, "var(--tb-warn)")}
                    <span>Showing latest run: <strong>simulated data</strong>. Run a benchmark on an installed model for verified measurements.</span>
                </div>
                <span class="tb-badge tb-badge-warn">Demo</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4 KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        acc_str = f"{latest['accuracy']:.1f}%" if latest else "—"
        acc_sub = f"Dataset: {latest['dataset_name']}" if latest else "No completed runs"
        _kpi_card("Accuracy", acc_str, acc_sub, accent=True)
    with k2:
        lat_str = f"{latest['median_latency'] * 1000:.0f}ms" if latest else "—"
        lat_sub = f"Mean: {latest['mean_latency'] * 1000:.0f}ms" if latest else "Awaiting benchmark"
        _kpi_card("Median Latency", lat_str, lat_sub)
    with k3:
        mem_str = f"{latest['peak_memory_mb']:.1f} MB" if latest else "—"
        mem_sub = "Peak RSS" if latest else "No active process"
        _kpi_card("Peak Memory", mem_str, mem_sub)
    with k4:
        err_str = f"{(1.0 - latest['accuracy']/100.0) * 100:.1f}%" if latest else "—"
        err_sub = f"Exact match: {latest['exact_match_rate']:.1f}%" if latest else "Awaiting execution"
        _kpi_card("Error Rate", err_str, err_sub)

    st.markdown('<div class="tb-spacer"></div>', unsafe_allow_html=True)

    # Chart + Quick Run
    col_chart, col_latest = st.columns([1.7, 1.3])

    with col_chart:
        st.markdown(
            f"""
            <div class="tb-card">
                <div class="tb-card-title">
                    <span style="display:flex;align-items:center;gap:6px;">{tb_icon("activity", 15, "var(--tb-accent)")} Latency Trend</span>
                    <span style="font-size:0.68rem;color:var(--tb-text-3);font-family:'JetBrains Mono',monospace;">PER-TASK ms</span>
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
                        line=dict(color=chart_accent, width=2),
                        marker=dict(size=5, color=chart_accent),
                        name="Latency (ms)",
                        fill="tozeroy",
                        fillcolor=f"rgba({int(chart_accent[1:3],16)},{int(chart_accent[3:5],16)},{int(chart_accent[5:7],16)},0.05)",
                    )
                )
                apply_plotly_fig_theme(fig_timeline, height=260, is_dark_mode=is_dark)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.caption("No task-level data available for this run.")
        else:
            _empty_state("activity", "No latency data yet", "Run an evaluation to record and visualize CPU inference latency.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_latest:
        st.markdown(
            f"""
            <div class="tb-card" style="height:100%;">
                <div class="tb-card-title">
                    <span style="display:flex;align-items:center;gap:6px;">{tb_icon("play", 15, "var(--tb-accent)")} Quick Run</span>
                </div>
                <div style="font-size:0.84rem;color:var(--tb-text-2);margin-bottom:16px;line-height:1.5;">
                    Launch a single-click benchmark with the default model and dataset.
                </div>
            """,
            unsafe_allow_html=True,
        )
        if latest:
            st.markdown(
                f"""
                <div style="background:var(--tb-elevated);border:1px solid var(--tb-border);border-radius:6px;padding:10px 12px;margin-bottom:14px;font-size:0.8rem;">
                    <div style="font-weight:600;color:var(--tb-text);margin-bottom:3px;">{latest['model_name']}</div>
                    <div style="color:var(--tb-text-3);font-family:'JetBrains Mono',monospace;font-size:0.7rem;margin-bottom:6px;">{latest['run_id']}</div>
                    <div style="display:flex;justify-content:space-between;color:var(--tb-text-2);">
                        <span>Accuracy: <strong style="color:var(--tb-text);">{latest['accuracy']:.1f}%</strong></span>
                        <span>RAM: <strong style="color:var(--tb-text);">{latest['peak_memory_mb']:.1f} MB</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        installed_models = [mid for mid in APPROVED_MODEL_IDS if is_model_installed(mid)]
        quick_target = installed_models[0] if installed_models else "mock"
        btn_label = f"Run Benchmark ({quick_target})" if installed_models else "Run Demo Simulation"

        if st.button(btn_label, type="primary", use_container_width=True):
            cfg = BenchmarkConfig(
                dataset_path="datasets/basic_v1.jsonl",
                model_name=quick_target,
                limit=5,
                generation_config=GenerationConfig(max_new_tokens=384, temperature=0.0),
                output_dir=str(STORAGE_DIR),
            )
            with st.spinner("Executing benchmark tasks..."):
                runner = BenchmarkRunner(cfg)
                res = runner.run()
                db.save_run(res)
                st.session_state["active_run_id"] = res.run_id
            st.success("Benchmark completed.")
            time.sleep(0.5)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent runs table
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;margin-top:8px;">
            {tb_icon("clock", 16, "var(--tb-accent)")}
            <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">Recent Evaluations</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if all_runs:
        df_recent = pd.DataFrame(all_runs)
        cols = ["run_id", "timestamp", "model_name", "dataset_name", "total_examples", "accuracy", "exact_match_rate", "median_latency"]
        st.dataframe(df_recent[cols], use_container_width=True, hide_index=True)
    else:
        _empty_state("database", "No runs recorded", "Execute a benchmark from the Run Benchmark tab to populate evaluation history.")



# ===========================================================================
# SCREEN 2: RUN BENCHMARK CONFIGURATION
# ===========================================================================
elif nav_choice == "Run Benchmark":
    _page_header(
        "Benchmark Runner",
        "Configure & Execute",
        "Select model, dataset, parameters, and launch a deterministic CPU evaluation.",
    )

    col_setup, col_meta = st.columns([1.6, 1.4])

    with col_setup:
        st.markdown(
            f"""<div style="font-size:0.88rem;font-weight:600;color:var(--tb-text);margin-bottom:8px;display:flex;align-items:center;gap:6px;">
                {tb_icon("box", 15, "var(--tb-accent)")} 1. Select Model
            </div>""",
            unsafe_allow_html=True,
        )
        model_options = [
            "Gemma 4 E2B (gemma4:e2b - Ollama)",
            "DeepSeek R1 Distill 1.5B (deepseek-r1:1.5b - Ollama)",
            "SmolLM2 1.7B Instruct (HuggingFaceTB/SmolLM2-1.7B-Instruct)",
            "Llama 3.2 1B Instruct (meta-llama/Llama-3.2-1B-Instruct)",
            "Offline Demo Simulator (Simulated Data / CI Only)",
        ]
        selected_model_option = st.selectbox("Model Identifier", model_options, index=0)

        if "Gemma" in selected_model_option:
            model_target = "gemma4:e2b"
        elif "DeepSeek" in selected_model_option:
            model_target = "deepseek-r1:1.5b"
        elif "SmolLM2" in selected_model_option:
            model_target = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        elif "Llama" in selected_model_option:
            model_target = "meta-llama/Llama-3.2-1B-Instruct"
        else:
            model_target = "mock"

        hf_token_val = None
        if model_target == "meta-llama/Llama-3.2-1B-Instruct":
            hf_token_val = st.text_input(
                "Hugging Face Access Token",
                value=os.environ.get("HF_TOKEN", ""),
                type="password",
                help="Required to download Llama 3.2 weights after accepting terms at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
            )
            if hf_token_val and hf_token_val.strip():
                clean_tok = hf_token_val.strip()
                os.environ["HF_TOKEN"] = clean_tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = clean_tok

        if model_target in ("gemma4:e2b", "deepseek-r1:1.5b"):
            current_ollama_url = get_ollama_base_url()
            is_ollama_active, daemon_status = check_ollama_daemon(current_ollama_url)

            with st.expander("Ollama Connection Settings", expanded=not is_ollama_active):
                ollama_url_input = st.text_input(
                    "Ollama Endpoint URL",
                    value=current_ollama_url,
                    help="Default: http://localhost:11434. For cloud deployments, use a public tunnel (e.g., ngrok).",
                )
                if ollama_url_input and ollama_url_input.strip() != current_ollama_url:
                    clean_input = ollama_url_input.strip()
                    os.environ["OLLAMA_HOST"] = clean_input
                    os.environ["OLLAMA_BASE_URL"] = clean_input
                    st.rerun()

                if is_ollama_active:
                    st.success(f"Connected: {daemon_status} at `{current_ollama_url}`")
                else:
                    st.warning(
                        f"**Ollama unreachable at `{current_ollama_url}`**\n\n"
                        "**Why this happens:**\n"
                        "- If you're using TinyBench on **Streamlit Community Cloud**, the cloud container cannot access `localhost` on your machine.\n\n"
                        "**Solutions:**\n"
                        "1. **Run locally** (recommended): `streamlit run app.py` on your machine connects to Ollama automatically.\n"
                        "2. **Public tunnel**: Run `ngrok http 11434` and paste the forwarding URL above.\n"
                        "3. **Verify Ollama**: Ensure `ollama serve` is running on your machine."
                    )

        st.markdown(
            f"""<div style="font-size:0.88rem;font-weight:600;color:var(--tb-text);margin-bottom:8px;margin-top:16px;display:flex;align-items:center;gap:6px;">
                {tb_icon("database", 15, "var(--tb-accent)")} 2. Select Dataset
            </div>""",
            unsafe_allow_html=True,
        )
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

        st.markdown(
            f"""<div style="font-size:0.88rem;font-weight:600;color:var(--tb-text);margin-bottom:8px;margin-top:16px;display:flex;align-items:center;gap:6px;">
                {tb_icon("sliders", 15, "var(--tb-accent)")} 3. Hyperparameters
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Generation Settings", expanded=False):
            max_tokens = st.slider("Max New Tokens", min_value=16, max_value=1024, value=384, step=16)
            temperature = st.slider("Temperature (0.0 = Deterministic Greedy)", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
            seed = st.number_input("Random Seed", value=42)

    with col_meta:
        st.markdown(
            f"""<div style="font-size:0.88rem;font-weight:600;color:var(--tb-text);margin-bottom:8px;display:flex;align-items:center;gap:6px;">
                {tb_icon("shield", 15, "var(--tb-accent)")} Model Profile & Safety
            </div>""",
            unsafe_allow_html=True,
        )
        adapter_preview = ModelRegistry.get_adapter(model_target)
        meta = adapter_preview.get_metadata()
        meta_min_ram = int(getattr(meta, 'minimum_ram_mb', 2000))
        meta_rec_ram = int(getattr(meta, 'recommended_ram_mb', 4000))

        is_mock_selected = "mock" in meta.backend_type.lower()
        is_installed = is_model_installed(meta.model_id) if not is_mock_selected else True

        if is_mock_selected:
            status_badge = '<span class="tb-badge tb-badge-warn">Simulator</span>'
        elif is_installed:
            status_badge = '<span class="tb-badge tb-badge-success">Installed</span>'
        elif meta.status == "requires_access_approval":
            status_badge = '<span class="tb-badge tb-badge-warn">Approval Required</span>'
        else:
            status_badge = '<span class="tb-badge tb-badge-danger">Not Downloaded</span>'

        st.markdown(
            f"""
            <div class="tb-card">
                <div class="tb-flex-between" style="margin-bottom:8px;">
                    <span style="font-weight:700;color:var(--tb-text);font-size:1rem;">{meta.model_name}</span>
                    {status_badge}
                </div>
                <div style="font-size:0.82rem;color:var(--tb-text-2);margin-bottom:14px;line-height:1.45;">
                    {meta.description}
                </div>
                <div class="tb-grid-2" style="background:var(--tb-elevated);padding:12px;border-radius:6px;border:1px solid var(--tb-border);font-family:'JetBrains Mono',monospace;">
                    <div><span style="color:var(--tb-text-3);">Developer</span><br><span style="color:var(--tb-text);font-weight:600;">{getattr(meta, 'developer', 'Unknown')}</span></div>
                    <div><span style="color:var(--tb-text-3);">Parameters</span><br><span style="color:var(--tb-text);font-weight:600;">{meta.parameters}</span></div>
                    <div><span style="color:var(--tb-text-3);">Download</span><br><span style="color:var(--tb-text);font-weight:600;">{getattr(meta, 'download_size', 'N/A')}</span></div>
                    <div><span style="color:var(--tb-text-3);">Quantization</span><br><span style="color:var(--tb-text);font-weight:600;">{getattr(meta, 'quantization', 'N/A')}</span></div>
                    <div><span style="color:var(--tb-text-3);">License</span><br><span style="color:var(--tb-text);font-weight:600;">{meta.license}</span></div>
                    <div><span style="color:var(--tb-text-3);">Runtime</span><br><span style="color:var(--tb-text);font-weight:600;">{meta.supported_runtime}</span></div>
                    <div><span style="color:var(--tb-text-3);">Min RAM</span><br><span style="color:var(--tb-success);font-weight:600;">{meta_min_ram} MB</span></div>
                    <div><span style="color:var(--tb-text-3);">Rec. RAM</span><br><span style="color:var(--tb-success);font-weight:600;">{meta_rec_ram} MB</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_mock_selected:
            if not is_installed:
                st.warning(f"**Model not cached locally**: {meta.status_reason or 'Weights must be downloaded before benchmarking.'}")
                if st.button(f"Download {meta.model_name}", use_container_width=True):
                    with st.spinner(f"Downloading {meta.model_name}..."):
                        prog = st.progress(0, text="Initializing download...")
                        def dl_cb(msg, pct):
                            prog.progress(int((pct or 0.5) * 100), text=msg)
                        try:
                            token_arg = hf_token_val if hf_token_val else None
                            dest_path = download_model(meta.model_id, token=token_arg, progress_callback=dl_cb)
                            st.success(f"Model downloaded and verified: {dest_path}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as dl_err:
                            st.error(f"Download failed: {dl_err}")
            else:
                st.success("Model verified in local cache and ready for benchmarking.")

        if sys_info.available_ram_mb < meta.estimated_ram_mb:
            st.error(
                f"**Memory warning**: Available RAM ({sys_info.available_ram_mb:.0f} MB) is below the recommended minimum ({meta.estimated_ram_mb:.0f} MB)."
            )
        else:
            st.info(
                f"System RAM: {sys_info.available_ram_mb:.0f} MB available."
            )

    st.divider()

    if st.button("Launch Benchmark", type="primary", use_container_width=True):
        if not is_mock_selected and not is_installed:
            st.error(f"Cannot launch: '{meta.model_name}' is not installed. Download it first.")
            st.stop()

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
            progress_bar.progress(pct, text=f"Task {curr}/{total}: {res.task_id} ({res.category})")
            status_icon = tb_icon("check-circle", 14, "var(--tb-success)") if res.is_correct else tb_icon("x-circle", 14, "var(--tb-danger)")
            status_box.markdown(
                f"**`{res.task_id}`** ({res.latency_seconds:.3f}s): "
                f"`{res.generated_text[:60] if res.generated_text else 'None'}` "
                f"{'Correct' if res.is_correct else 'Incorrect'}"
            )
            live_rows.append({
                "Task ID": res.task_id,
                "Category": res.category,
                "Status": "Pass" if res.is_correct else ("Error" if res.error else "Fail"),
                "Latency (s)": round(res.latency_seconds, 4),
                "Output": (res.generated_text[:40] + "...") if len(res.generated_text) > 40 else res.generated_text,
            })
            table_placeholder.dataframe(pd.DataFrame(live_rows), use_container_width=True)

        try:
            with st.spinner("Executing benchmark tasks on CPU..."):
                run_result = runner.run(progress_callback=on_step)
                db.save_run(run_result)
                st.session_state["active_run_id"] = run_result.run_id

            progress_bar.progress(100, text="Evaluation complete.")
            st.success(
                f"Run `{run_result.run_id}` complete. "
                f"Accuracy: **{run_result.accuracy:.1f}%**, Exact Match: **{run_result.exact_match_rate:.1f}%**, "
                f"Median Latency: **{run_result.latency_summary.median_latency_seconds:.3f}s**."
            )
        except Exception as exc:
            st.error(f"Evaluation error: {exc}")



# ===========================================================================
# SCREEN 3: RUN DETAILS & EXECUTION LOG
# ===========================================================================
elif nav_choice == "Run Details":
    _page_header(
        "Run Inspector",
        "Run Details & Diagnostics",
        "Per-task audit trail, error tracking, and data exports for any completed evaluation.",
    )

    all_runs = db.list_runs(limit=100)
    if not all_runs:
        _empty_state("database", "No evaluation runs found", "Execute a benchmark from the Overview or Run Benchmark tab to populate run details.")
    else:
        run_options = [f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})" for r in all_runs]
        default_index = 0
        if "active_run_id" in st.session_state:
            for idx, opt in enumerate(run_options):
                if st.session_state["active_run_id"] in opt:
                    default_index = idx
                    break

        selected_opt = st.selectbox("Select Evaluation Run", run_options, index=default_index)
        selected_run_id = selected_opt.split(" — ")[0]
        run_res = db.get_run(selected_run_id)

        if not run_res:
            st.markdown(
                f"""
                <div class="tb-card" style="border-left:3px solid var(--tb-danger);">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                        {tb_icon("alert-triangle", 18, "var(--tb-danger)")}
                        <span style="font-weight:600;color:var(--tb-text);">Could not load run artifacts for <code>{selected_run_id}</code></span>
                    </div>
                    <div style="font-size:0.84rem;color:var(--tb-text-2);">
                        The JSON file may be missing or corrupted. Select a different run or re-execute the benchmark.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            is_mock = "mock" in run_res.model_metadata.backend_type.lower() or getattr(run_res, "is_demo", False)
            badge_html = (
                '<span class="tb-badge tb-badge-warn">Simulated</span>'
                if is_mock
                else '<span class="tb-badge tb-badge-success">Verified</span>'
            )

            # 4 KPI cards
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                _kpi_card("Accuracy", f"{run_res.accuracy:.1f}%", "Target answer match", accent=True)
            with r2:
                _kpi_card("Median Latency", f"{run_res.latency_summary.median_latency_seconds * 1000:.0f}ms", "CPU wall-clock")
            with r3:
                _kpi_card("Peak RAM", f"{run_res.memory_measurement.peak_rss_mb:.1f} MB", "Resident memory")
            with r4:
                _kpi_card("Exact Match", f"{run_res.exact_match_rate:.1f}%", "Valid outputs generated")

            st.markdown('<div class="tb-spacer"></div>', unsafe_allow_html=True)

            # Charts
            c_bar, c_hist = st.columns(2)
            with c_bar:
                st.markdown(
                    f"""
                    <div class="tb-card">
                        <div class="tb-card-title">
                            <span style="display:flex;align-items:center;gap:6px;">{tb_icon("bar-chart", 15, "var(--tb-accent)")} Category Breakdown</span>
                        </div>
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
                        color_discrete_sequence=[chart_accent, chart_accent2, chart_accent3],
                    )
                    apply_plotly_fig_theme(fig_cat, height=260, is_dark_mode=is_dark)
                    st.plotly_chart(fig_cat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_hist:
                st.markdown(
                    f"""
                    <div class="tb-card">
                        <div class="tb-card-title">
                            <span style="display:flex;align-items:center;gap:6px;">{tb_icon("clock", 15, "var(--tb-accent)")} Latency Distribution</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                lat_values = [t.latency_seconds * 1000 for t in run_res.task_results if t.error is None]
                if lat_values:
                    fig_lat = px.histogram(
                        lat_values,
                        nbins=12,
                        color_discrete_sequence=[chart_accent],
                    )
                    fig_lat.add_vline(
                        x=run_res.latency_summary.median_latency_seconds * 1000,
                        line_dash="dash",
                        line_color=chart_accent2,
                        annotation_text=f"Median ({run_res.latency_summary.median_latency_seconds * 1000:.0f}ms)",
                        annotation_font_color=chart_accent2,
                    )
                    apply_plotly_fig_theme(fig_lat, height=260, is_dark_mode=is_dark)
                    st.plotly_chart(fig_lat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Task-by-task inspection
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;margin-top:8px;">
                    {tb_icon("search", 16, "var(--tb-accent)")}
                    <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">Task-by-Task Audit</span>
                </div>""",
                unsafe_allow_html=True,
            )
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

            st.caption(f"Showing {len(filtered_tasks)} of {len(run_res.task_results)} tasks.")

            if not filtered_tasks:
                _empty_state("search", "No tasks match filters", "Adjust category or outcome filters to view task details.")

            for t in filtered_tasks:
                if t.is_correct:
                    status_badge = '<span class="tb-badge tb-badge-success">Correct</span>'
                elif t.error:
                    status_badge = '<span class="tb-badge tb-badge-warn">Error</span>'
                else:
                    status_badge = '<span class="tb-badge tb-badge-danger">Incorrect</span>'

                with st.expander(f"{t.task_id} [{t.category}] — {t.latency_seconds * 1000:.0f}ms"):
                    st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
                    st.markdown('<div class="tb-spacer-sm"></div>', unsafe_allow_html=True)
                    cp, ce = st.columns(2)
                    with cp:
                        st.markdown("**Prompt:**")
                        st.code(t.prompt, language="text")
                        st.markdown(f"**Expected Answer:** `{t.expected_answer}`")
                    with ce:
                        st.markdown("**Model Output:**")
                        st.code(t.generated_text if t.generated_text else "(empty output)", language="text")
                        if getattr(t, "thinking_text", None):
                            with st.expander("Chain-of-Thought Reasoning"):
                                st.markdown(
                                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;background:var(--tb-inset);padding:10px;border-radius:6px;white-space:pre-wrap;max-height:250px;overflow-y:auto;color:var(--tb-text-2);'>{t.thinking_text}</div>",
                                    unsafe_allow_html=True,
                                )
                        st.markdown(
                            f"- Exact Match: `{'Yes' if t.exact_match else 'No'}`\n"
                            f"- Compliance: `{t.compliance_score:.2f}`\n"
                            f"- Latency: `{t.latency_seconds * 1000:.0f}ms`"
                        )
                        if t.error:
                            st.error(f"Inference Error: {t.error}")

            st.divider()

            # Exports
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    {tb_icon("download", 16, "var(--tb-accent)")}
                    <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">Data Exports</span>
                </div>""",
                unsafe_allow_html=True,
            )
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
                    "Per-Task CSV",
                    data=csv_tasks,
                    file_name=f"{selected_run_id}_tasks.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec2:
                st.download_button(
                    "Summary CSV",
                    data=csv_summary,
                    file_name=f"{selected_run_id}_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec3:
                st.download_button(
                    "Full JSON Artifact",
                    data=json_raw,
                    file_name=f"{selected_run_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )



# ===========================================================================
# SCREEN 4: MODEL COMPARISON
# ===========================================================================
elif nav_choice == "Model Comparison":
    _page_header(
        "Comparative Analytics",
        "Model Comparison",
        "Side-by-side evaluation metrics across different models and run configurations.",
    )

    all_runs = db.list_runs(limit=100)
    if len(all_runs) < 2:
        _empty_state(
            "layers",
            "Insufficient data for comparison",
            "At least 2 completed evaluation runs are required. Execute benchmarks for different models in the Run Benchmark tab.",
        )
    else:
        run_options = {f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})": r['run_id'] for r in all_runs}
        selected_runs_labels = st.multiselect(
            "Select Runs to Compare",
            options=list(run_options.keys()),
            default=list(run_options.keys())[:min(2, len(run_options))],
        )

        if len(selected_runs_labels) < 2:
            st.info("Select at least 2 runs to display comparisons.")
        else:
            selected_ids = [run_options[l] for l in selected_runs_labels]
            loaded_runs = [db.get_run(rid) for rid in selected_ids]
            selected_run_objects = [r for r in loaded_runs if r is not None]

            if len(selected_run_objects) < 2:
                st.error("Could not load artifacts for the selected runs. Please select different runs.")
                st.stop()

            run_a = selected_run_objects[0]
            run_b = selected_run_objects[1]

            # Side-by-side model cards
            st.markdown(
                f"""
                <div class="tb-card">
                    <div class="tb-card-title">
                        <span style="display:flex;align-items:center;gap:6px;">{tb_icon("layers", 15, "var(--tb-accent)")} Selected Models</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(
                    f"""
                    <div style="background:var(--tb-elevated);border:2px solid var(--tb-accent);border-radius:8px;padding:16px;">
                        <div class="tb-flex-between" style="margin-bottom:10px;">
                            <span class="tb-badge tb-badge-accent">Model A</span>
                            <span style="font-size:0.7rem;color:var(--tb-text-3);font-family:'JetBrains Mono',monospace;">{run_a.run_id}</span>
                        </div>
                        <div style="font-size:1.05rem;font-weight:700;color:var(--tb-text);margin-bottom:12px;">
                            {run_a.model_metadata.model_name}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Accuracy</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_a.accuracy:.1f}%</div>
                            </div>
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Latency</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_a.latency_summary.median_latency_seconds * 1000:.0f}ms</div>
                            </div>
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">RAM</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_a.memory_measurement.peak_rss_mb:.1f}MB</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_b:
                st.markdown(
                    f"""
                    <div style="background:var(--tb-elevated);border:1px solid var(--tb-border);border-radius:8px;padding:16px;">
                        <div class="tb-flex-between" style="margin-bottom:10px;">
                            <span class="tb-badge tb-badge-success">Model B</span>
                            <span style="font-size:0.7rem;color:var(--tb-text-3);font-family:'JetBrains Mono',monospace;">{run_b.run_id}</span>
                        </div>
                        <div style="font-size:1.05rem;font-weight:700;color:var(--tb-text);margin-bottom:12px;">
                            {run_b.model_metadata.model_name}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Accuracy</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_b.accuracy:.1f}%</div>
                            </div>
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">Latency</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_b.latency_summary.median_latency_seconds * 1000:.0f}ms</div>
                            </div>
                            <div style="background:var(--tb-surface);border:1px solid var(--tb-border);border-radius:6px;padding:8px;text-align:center;">
                                <div style="font-size:0.65rem;color:var(--tb-text-3);text-transform:uppercase;font-family:'JetBrains Mono',monospace;">RAM</div>
                                <div style="font-size:1.15rem;font-weight:700;color:var(--tb-text);font-family:'JetBrains Mono',monospace;">{run_b.memory_measurement.peak_rss_mb:.1f}MB</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Guidance callout
            st.markdown(
                f"""
                <div class="tb-callout">
                    <div style="display:flex;align-items:center;gap:10px;">
                        {tb_icon("info", 16, "var(--tb-accent)")}
                        <span style="font-size:0.84rem;color:var(--tb-text-2);">Compare models only alongside the task configuration and hardware environment that produced these results. Metrics across different datasets or run scopes are not directly comparable.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Charts
            rad_col, bar_col = st.columns([1.3, 1.7])
            with rad_col:
                st.markdown(
                    f"""
                    <div class="tb-card">
                        <div class="tb-card-title">
                            <span style="display:flex;align-items:center;gap:6px;">{tb_icon("activity", 15, "var(--tb-accent)")} Category Radar</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                all_cats = ["multiple_choice", "factual_qa", "arithmetic_reasoning", "instruction_following", "structured_output"]
                fig_radar = go.Figure()

                for i, r_obj in enumerate(selected_run_objects[:2]):
                    color = chart_palette[i % len(chart_palette)]
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
                            name=r_obj.model_metadata.model_name[:20],
                            line=dict(color=color, width=2),
                        )
                    )

                polar_bg = chart_bg
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor=chart_grid, linecolor=chart_line_clr),
                        angularaxis=dict(gridcolor=chart_grid, linecolor=chart_line_clr, color=chart_font_clr),
                        bgcolor=polar_bg,
                    ),
                    paper_bgcolor=polar_bg,
                    font=dict(color=chart_font_clr, family="Plus Jakarta Sans, sans-serif", size=10),
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=280,
                    legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with bar_col:
                st.markdown(
                    f"""
                    <div class="tb-card">
                        <div class="tb-card-title">
                            <span style="display:flex;align-items:center;gap:6px;">{tb_icon("bar-chart", 15, "var(--tb-accent)")} Accuracy & Match Deltas</span>
                        </div>
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
                    color_discrete_sequence=[chart_accent, chart_accent2],
                )
                apply_plotly_fig_theme(fig_comp_bars, height=280, is_dark_mode=is_dark)
                st.plotly_chart(fig_comp_bars, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="tb-spacer"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    {tb_icon("file-text", 16, "var(--tb-accent)")}
                    <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">Comparison Table</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.download_button(
                "Export Comparison CSV",
                data=df_comp.to_csv(index=False),
                file_name="tinybench_model_comparison.csv",
                mime="text/csv",
                use_container_width=False,
            )



# ===========================================================================
# SCREEN 5: MODELS & DATASETS REGISTRY
# ===========================================================================
elif nav_choice == "05. Models & Datasets" or nav_choice == "Models & Datasets":
    _page_header(
        "Registry",
        "Models & Datasets",
        "Approved open-weight models and benchmark dataset catalog with compatibility details.",
    )

    col_library, col_guide = st.columns([1.7, 1.3])

    with col_library:
        tab_models, tab_datasets = st.tabs(["Model Registry", "Dataset Catalog"])

        with tab_models:
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                    {tb_icon("server", 16, "var(--tb-accent)")}
                    <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">Approved Benchmark Models</span>
                </div>""",
                unsafe_allow_html=True,
            )
            models = ModelRegistry.list_available_models()
            for m in models:
                if m.status == "available":
                    badge_cls = "tb-badge-success"
                    status_label = "Available"
                elif m.status == "requires_access_approval":
                    badge_cls = "tb-badge-warn"
                    status_label = "Approval Required"
                else:
                    badge_cls = "tb-badge-danger"
                    status_label = "Unavailable"

                m_min_ram = f"{getattr(m, 'minimum_ram_mb', 2000.0):.0f}"
                m_rec_ram = f"{getattr(m, 'recommended_ram_mb', 4000.0):.0f}"
                m_verified = getattr(m, 'verification_timestamp', 'Pre-configured')

                st.markdown(
                    f"""
                    <div class="tb-card">
                        <div class="tb-flex-between" style="margin-bottom:6px;">
                            <div>
                                <span style="font-size:1rem;font-weight:700;color:var(--tb-text);">{m.model_name}</span>
                                <span style="font-size:0.75rem;color:var(--tb-text-3);margin-left:8px;">by {getattr(m, 'developer', 'Unknown')}</span>
                            </div>
                            <span class="tb-badge {badge_cls}">{status_label}</span>
                        </div>
                        <div style="font-size:0.82rem;color:var(--tb-text-2);margin-bottom:12px;line-height:1.45;">{m.description}</div>
                        <div class="tb-grid-4" style="font-family:'JetBrains Mono',monospace;background:var(--tb-elevated);padding:12px;border-radius:6px;border:1px solid var(--tb-border);">
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">ID</span><br><span style="color:var(--tb-text);font-weight:600;font-size:0.72rem;">{m.model_id}</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">Params</span><br><span style="color:var(--tb-text);font-weight:600;font-size:0.72rem;">{m.parameters}</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">Download</span><br><span style="color:var(--tb-text);font-weight:600;font-size:0.72rem;">{getattr(m, 'download_size', 'N/A')}</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">Quantization</span><br><span style="color:var(--tb-text);font-weight:600;font-size:0.72rem;">{getattr(m, 'quantization', 'N/A')}</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">License</span><br><a href="{getattr(m, 'license_url', '#')}" target="_blank" style="color:var(--tb-accent);font-weight:600;font-size:0.72rem;">{m.license}</a></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">Runtime</span><br><span style="color:var(--tb-text);font-weight:600;font-size:0.72rem;">{getattr(m, 'supported_runtime', m.backend_type)}</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">RAM (Min/Rec)</span><br><span style="color:var(--tb-success);font-weight:600;font-size:0.72rem;">{m_min_ram} / {m_rec_ram} MB</span></div>
                            <div><span style="color:var(--tb-text-3);font-size:0.68rem;">Verified</span><br><span style="color:var(--tb-text-3);font-size:0.72rem;">{m_verified}</span></div>
                        </div>
                        {f'<div style="font-size:0.76rem;color:var(--tb-text-2);font-weight:500;padding:6px 0 0;margin-top:8px;">{m.status_reason}</div>' if getattr(m, 'status_reason', None) else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_datasets:
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
                    {tb_icon("database", 16, "var(--tb-accent)")}
                    <span style="font-size:0.95rem;font-weight:600;color:var(--tb-text);">basic_v1.jsonl Benchmark Suite</span>
                </div>""",
                unsafe_allow_html=True,
            )
            dataset_path = current_dir / "datasets" / "basic_v1.jsonl"
            if dataset_path.exists():
                val_report = validate_dataset_file(dataset_path)
                st.markdown(
                    f"""
                    <div class="tb-callout" style="border-left-color:var(--tb-success);">
                        <div style="display:flex;align-items:center;gap:8px;">
                            {tb_icon("check-circle", 16, "var(--tb-success)")}
                            <span>Validated: <strong>{val_report.total_items} items</strong> across {len(val_report.category_counts)} categories.</span>
                        </div>
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
                        "Prompt": it.prompt[:55] + "..." if len(it.prompt) > 55 else it.prompt,
                        "Expected": str(it.expected_answer),
                        "Scoring": it.scoring_method,
                    })
                st.dataframe(pd.DataFrame(items_table), use_container_width=True, hide_index=True)
            else:
                st.error(f"Dataset file not found at `{dataset_path}`.")

    with col_guide:
        st.markdown(
            f"""
            <div class="tb-card" style="border-left:3px solid var(--tb-accent);">
                <div style="font-size:1.05rem;font-weight:700;color:var(--tb-text);margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                    {tb_icon("book-open", 18, "var(--tb-accent)")} Building a Fair Comparison
                </div>
                <div style="font-size:0.84rem;color:var(--tb-text-2);margin-bottom:16px;line-height:1.5;">
                    Proper model comparison requires controlling for configuration, hardware, and dataset variables.
                </div>
                <div style="margin-bottom:14px;">
                    <div style="font-weight:600;font-size:0.86rem;color:var(--tb-text);margin-bottom:3px;">1. Match evaluation context</div>
                    <div style="font-size:0.8rem;color:var(--tb-text-2);line-height:1.5;">
                        Both models should be tested with identical prompt templates, few-shot conditioning, and answer parsers.
                    </div>
                </div>
                <div style="margin-bottom:14px;">
                    <div style="font-weight:600;font-size:0.86rem;color:var(--tb-text);margin-bottom:3px;">2. Verify dataset coverage</div>
                    <div style="font-size:0.8rem;color:var(--tb-text-2);line-height:1.5;">
                        Confirm that sample limits, categories, and difficulty levels are consistent across runs.
                    </div>
                </div>
                <div>
                    <div style="font-weight:600;font-size:0.86rem;color:var(--tb-text);margin-bottom:3px;">3. Review side by side</div>
                    <div style="font-size:0.8rem;color:var(--tb-text-2);line-height:1.5;">
                        Use the Model Comparison tab to inspect accuracy, latency, and memory deltas on the same chart.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# SCREEN 6: METHODOLOGY & SCIENTIFIC STANDARDS
# ===========================================================================
elif nav_choice == "06. Methodology & Hardware" or nav_choice == "Methodology":
    _page_header(
        "Technical Notes",
        "Methodology",
        "How to read TinyBench reports: evaluation context, scoring methods, and reproducibility guidelines.",
    )

    # Principle banner
    st.markdown(
        f"""
        <div class="tb-callout" style="border-left-color:var(--tb-success);">
            <div style="display:flex;align-items:center;gap:10px;">
                {tb_icon("shield", 18, "var(--tb-success)")}
                <div>
                    <div style="font-weight:600;text-transform:uppercase;font-size:0.68rem;letter-spacing:0.06em;color:var(--tb-text-3);font-family:'JetBrains Mono',monospace;">Reporting Principle</div>
                    <div style="font-size:0.9rem;font-weight:600;color:var(--tb-text);">Always interpret results alongside the model, dataset, and configuration that produced them.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_nav, col_details = st.columns([1.1, 1.9])

    with col_nav:
        st.markdown(
            f"""
            <div class="tb-card">
                <div style="font-size:0.68rem;font-weight:600;color:var(--tb-text-3);text-transform:uppercase;margin-bottom:12px;font-family:'JetBrains Mono',monospace;">
                    Section Index
                </div>
                <div style="background:var(--tb-elevated);padding:8px 12px;border-radius:6px;font-weight:600;color:var(--tb-text);margin-bottom:6px;font-size:0.84rem;border-left:2px solid var(--tb-accent);">
                    01. Evaluation Context
                </div>
                <div style="padding:8px 12px;color:var(--tb-text-2);font-weight:500;font-size:0.84rem;">
                    02. Reading Results
                </div>
                <div style="padding:8px 12px;color:var(--tb-text-2);font-weight:500;font-size:0.84rem;">
                    03. Hardware & Workloads
                </div>
            </div>

            <div class="tb-card" style="border-left:3px solid var(--tb-warn);">
                <div style="font-weight:700;font-size:0.9rem;color:var(--tb-text);margin-bottom:6px;">Preserve Context</div>
                <div style="font-size:0.82rem;color:var(--tb-text-2);line-height:1.5;">
                    Values are meaningful only when the accompanying setup remains visible. Do not compare an INT4 quantized model with a full FP16 baseline without noting the precision tradeoff.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown(
            f"""
            <div class="tb-card">
                <div style="font-size:1rem;font-weight:700;color:var(--tb-text);margin-bottom:6px;">
                    01. Evaluation Context
                </div>
                <div style="font-size:0.84rem;color:var(--tb-text-2);line-height:1.5;margin-bottom:14px;">
                    TinyBench preserves complete records of the selected model, dataset, and run configuration. Review this context before interpreting any metrics.
                </div>
                <div style="display:flex;flex-direction:column;gap:8px;font-size:0.82rem;background:var(--tb-elevated);padding:12px;border-radius:6px;color:var(--tb-text);">
                    <div><strong>Model record:</strong> Source, weights, quantization, and runtime registered for the run.</div>
                    <div><strong>Dataset record:</strong> Question set, categories, scoring method, and version.</div>
                    <div><strong>Run config:</strong> Token limits, temperature, seed, and hardware snapshot.</div>
                </div>
            </div>

            <div class="tb-card" style="border-left:3px solid var(--tb-accent);">
                <div style="font-size:1rem;font-weight:700;color:var(--tb-text);margin-bottom:6px;">
                    02. Reading Results
                </div>
                <div style="font-size:0.84rem;color:var(--tb-text-2);line-height:1.5;margin-bottom:12px;">
                    Use the comparison and inspection views to review metrics alongside their source configuration. Labels, units, and hardware context stay close to reported values.
                </div>
                <div style="font-size:0.82rem;color:var(--tb-text);">
                    &bull; Compare like with like: evaluate models on identical hardware and datasets.<br>
                    &bull; Key metrics: Throughput (tokens/s), Time-To-First-Token, Median Latency, Peak RSS.
                </div>
            </div>

            <div class="tb-card">
                <div style="font-size:1rem;font-weight:700;color:var(--tb-text);margin-bottom:6px;">
                    03. What Accompanies a Result
                </div>
                <div style="font-size:0.84rem;color:var(--tb-text-2);line-height:1.5;">
                    Every exported run artifact includes the full system information, generation parameters, per-task outputs with prompt/expected/generated text, and timing data. Use the Run Details tab to inspect individual samples and the raw data exports for external analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# FOOTER
# ===========================================================================
theme_name = "Dark" if is_dark else "Light"
st.markdown(
    f"""
    <div class="tb-footer">
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr;gap:24px;margin-bottom:18px;">
            <div>
                <div style="font-weight:700;color:var(--tb-text);font-size:0.95rem;margin-bottom:6px;display:flex;align-items:center;gap:6px;">
                    {tb_icon("zap", 16, "var(--tb-accent)")} TinyBench
                </div>
                <div style="color:var(--tb-text-3);font-size:0.78rem;line-height:1.5;margin-bottom:8px;">
                    Lightweight open-weight LLM benchmarking for commodity CPUs.<br>
                    Designed for reproducible research and local model evaluation.
                </div>
                <div style="font-size:0.72rem;color:var(--tb-text-3);">
                    Apache-2.0 License &bull; Open-Source
                </div>
            </div>
            <div>
                <div style="color:var(--tb-text-2);font-weight:600;margin-bottom:6px;text-transform:uppercase;font-size:0.68rem;letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;">Docs</div>
                <div style="display:flex;flex-direction:column;gap:5px;">
                    <a href="https://github.com/shettyadithyar76-prog/tinybench#readme" target="_blank">Quickstart</a>
                    <a href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/benchmark_dataset_spec.md" target="_blank">Dataset Schema</a>
                    <a href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/model_integration_guide.md" target="_blank">Adding Models</a>
                    <a href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/reproducibility.md" target="_blank">Reproducibility</a>
                </div>
            </div>
            <div>
                <div style="color:var(--tb-text-2);font-weight:600;margin-bottom:6px;text-transform:uppercase;font-size:0.68rem;letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;">Community</div>
                <div style="display:flex;flex-direction:column;gap:5px;">
                    <a href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank">GitHub</a>
                    <a href="https://github.com/shettyadithyar76-prog/tinybench/issues" target="_blank">Issues</a>
                    <a href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/CONTRIBUTING.md" target="_blank">Contributing</a>
                </div>
            </div>
            <div>
                <div style="color:var(--tb-text-2);font-weight:600;margin-bottom:6px;text-transform:uppercase;font-size:0.68rem;letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;">Runtime</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--tb-text-3);display:flex;flex-direction:column;gap:3px;">
                    <div>Platform: <span style="color:var(--tb-text);font-weight:600;">{sys_info.os_name}</span></div>
                    <div>CPU: <span style="color:var(--tb-text);font-weight:600;">{sys_info.cpu_physical_cores}C / {sys_info.cpu_logical_cores}T</span></div>
                    <div>Python: <span style="color:var(--tb-text);font-weight:600;">{sys_info.python_version.split()[0]}</span></div>
                    <div>RAM: <span style="color:var(--tb-success);font-weight:600;">{sys_info.available_ram_mb:.0f} MB</span></div>
                </div>
            </div>
        </div>
        <div style="border-top:1px solid var(--tb-border);padding-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:0.72rem;color:var(--tb-text-3);">
            <div>&copy; 2026 TinyBench Contributors</div>
            <div style="font-family:'JetBrains Mono',monospace;font-weight:500;">main &bull; {theme_name} &bull; v0.2.0</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
