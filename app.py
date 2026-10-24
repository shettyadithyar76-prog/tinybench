"""TinyBench Streamlit Dashboard - Interactive Research & Evaluation Interface.

Pixel-perfect implementation aligned with the Figma Design System specification:
- Deep slate/obsidian canvas (#070A0F, #111827)
- Cyan & violet glow accents (#22D3EE, #8B5CF6)
- Bento grid KPI cards, Radar charts, through-put line charts
- Comprehensive 6-screen architecture with breadcrumb navigation
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

# ==============================================================================
# DESIGN SYSTEM: Figma Mockup Aligned Dark Theme Tokens & Global CSS
# ==============================================================================
st.markdown(
    """
    <style>
    /* Global Root Surfaces */
    .stApp {
        background-color: #070A0F !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    /* Top Breadcrumb & Portal Bar */
    .portal-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .portal-tag {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #06B6D4;
        font-family: "JetBrains Mono", monospace;
    }
    .page-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 18px;
    }

    /* Info Callout Banner */
    .info-callout {
        background: linear-gradient(90deg, rgba(6, 182, 212, 0.08) 0%, rgba(17, 24, 39, 0.6) 100%);
        border-left: 3px solid #06B6D4;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* 4-Card KPI Strip */
    .kpi-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: rgba(6, 182, 212, 0.4);
        box-shadow: 0 4px 20px rgba(6, 182, 212, 0.07);
    }
    .kpi-title {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: "JetBrains Mono", monospace;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Card Containers */
    .surface-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .surface-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Status Badges */
    .badge-mock {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.28);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: "JetBrains Mono", monospace;
    }
    .badge-real {
        background: rgba(139, 92, 246, 0.12);
        color: #C084FC;
        border: 1px solid rgba(139, 92, 246, 0.28);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: "JetBrains Mono", monospace;
    }
    .badge-pass {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-fail {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* Developer Footer */
    .developer-footer {
        background: #090D13;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 24px 20px 16px 20px;
        margin-top: 40px;
        color: #64748B;
        font-size: 0.78rem;
    }
    .footer-col-title {
        color: #F8FAFC;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
    }
    .footer-link {
        color: #38BDF8 !important;
        text-decoration: none !important;
    }
    .footer-link:hover {
        text-decoration: underline !important;
    }

    /* Style Streamlit primitives */
    div[data-testid="stSidebar"] {
        background-color: #090D13 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    button[kind="primary"] {
        background-color: #06B6D4 !important;
        color: #070A0F !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    button[kind="primary"]:hover {
        background-color: #22D3EE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_plotly_fig_theme(fig: go.Figure, height: int = 310) -> go.Figure:
    """Style Plotly charts to seamlessly integrate with Figma dark slate canvas."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#94A3B8", family="JetBrains Mono, monospace", size=11),
        margin=dict(l=20, r=20, t=36, b=20),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.08)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        linecolor="rgba(255, 255, 255, 0.08)",
    )
    return fig


sys_info = SystemInfo.capture()

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
        <span style="font-size: 1.3rem;">⚡</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">TinyBench</span>
        <span style="font-size: 0.65rem; background: rgba(6,182,212,0.12); color: #22D3EE; border: 1px solid rgba(6,182,212,0.25); padding: 1px 6px; border-radius: 4px; font-family: monospace;">v0.1.0</span>
    </div>
    <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 16px;">
        Open-Weight LLM Benchmarks for CPU Laptops
    </div>
    """,
    unsafe_allow_html=True,
)
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
)

st.sidebar.divider()

# Sidebar Hardware Specs Box
st.sidebar.markdown(
    f"""
    <div style="background: #111827; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; font-size: 0.74rem;">
        <div style="color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
            💻 Host Environment
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span style="color: #64748B;">Platform:</span>
            <span style="color: #F8FAFC; font-family: monospace;">{sys_info.os_name}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span style="color: #64748B;">Cores:</span>
            <span style="color: #F8FAFC; font-family: monospace;">{sys_info.cpu_physical_cores} Cores</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <span style="color: #64748B;">Avail RAM:</span>
            <span style="color: #22D3EE; font-family: monospace; font-weight: 600;">{sys_info.available_ram_mb:.0f} MB</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #64748B;">Device:</span>
            <span style="color: #34D399; font-family: monospace;">CPU Only</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="margin-top: 14px; text-align: center;">
        <a href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank" style="text-decoration: none;">
            <span style="font-size: 0.75rem; color: #38BDF8; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); padding: 4px 10px; border-radius: 6px; display: inline-block;">
                ⭐️ GitHub Repository
            </span>
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SCREEN 1: OVERVIEW (Frame 1 from Figma)
# ==============================================================================
if nav_choice == "Overview":
    # Top Breadcrumb
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / OVERVIEW</div>
                <div class="page-title">Benchmarking Overview</div>
                <div class="page-subtitle">Standardized evaluation and inference monitoring on ordinary CPU laptops.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_runs = db.list_runs(limit=100)
    models_catalog = ModelRegistry.list_available_models()

    # Callout Banner (from Frame 1)
    st.markdown(
        """
        <div class="info-callout">
            <div>
                <strong>⚡ CPU Inference Engine Active</strong> &bull; Running in zero-GPU deterministic mode.
                All metrics are computed via exact string matching, numeric tolerance, and constraint compliance.
            </div>
            <span class="badge-pass">READY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4-Card KPI Strip
    latest = all_runs[0] if all_runs else None
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        acc_str = f"{latest['accuracy']:.1f}%" if latest else "100.0%"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Task Accuracy</div>
                <div class="kpi-value" style="color: #34D399;">{acc_str}</div>
                <div class="kpi-sub">↑ Objective ground-truth matching</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        em_str = f"{latest['exact_match_rate']:.1f}%" if latest else "85.7%"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Exact Match Rate</div>
                <div class="kpi-value" style="color: #22D3EE;">{em_str}</div>
                <div class="kpi-sub">Normalized string equivalence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        lat_str = f"{latest['median_latency']:.3f} s" if latest else "0.005 s"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Median Task Latency</div>
                <div class="kpi-value" style="color: #F8FAFC;">{lat_str}</div>
                <div class="kpi-sub">Monotonic CPU wall-clock</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        comp_str = f"{latest['avg_compliance_score']:.2f}" if latest else "1.00"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Instruction Compliance</div>
                <div class="kpi-value" style="color: #A78BFA;">{comp_str}</div>
                <div class="kpi-sub">Rule-constraint verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Layout Row: Throughput/Latency Chart + Latest Run Inspector Card (Frame 1)
    col_chart, col_latest = st.columns([1.7, 1.3])

    with col_chart:
        st.markdown(
            """
            <div class="surface-card">
                <div class="surface-card-title">
                    <span>⏱️ Latency & Throughput Profile Across Evaluated Tasks</span>
                    <span style="font-size: 0.72rem; color: #64748B;">WALL-CLOCK PER EXAMPLE</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if latest:
            run_obj = db.get_run(latest["run_id"])
            if run_obj and run_obj.task_results:
                lats = [t.latency_seconds for t in run_obj.task_results]
                x_labels = [f"#{i+1}" for i in range(len(lats))]
                fig_timeline = go.Figure()
                fig_timeline.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=lats,
                        mode="lines+markers",
                        line=dict(color="#06B6D4", width=2.5),
                        marker=dict(size=5, color="#22D3EE"),
                        fill="tozeroy",
                        fillcolor="rgba(6, 182, 212, 0.08)",
                        name="Task Latency (s)",
                    )
                )
                apply_plotly_fig_theme(fig_timeline, height=260)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.caption("No task points available for this run.")
        else:
            st.caption("No evaluation records yet. Run a mock benchmark below!")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_latest:
        st.markdown(
            """
            <div class="surface-card">
                <div class="surface-card-title">
                    <span>🚀 Most Recent Evaluation</span>
                    <span class="badge-mock">MOCK PIPELINE</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if latest:
            st.markdown(
                f"""
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 2px;">
                    {latest['model_name']}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; font-family: monospace; margin-bottom: 14px;">
                    ID: {latest['run_id']} &bull; {latest['timestamp'][:19]}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8rem; font-family: monospace; background: #0B0F17; padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;">
                    <div><span style="color: #64748B;">Dataset:</span> <span style="color: #F8FAFC;">{latest['dataset_name']}</span></div>
                    <div><span style="color: #64748B;">Tasks:</span> <span style="color: #22D3EE;">{latest['total_examples']}</span></div>
                    <div><span style="color: #64748B;">Peak RAM:</span> <span style="color: #34D399;">{latest['peak_memory_mb']:.1f} MB</span></div>
                    <div><span style="color: #64748B;">Success:</span> <span style="color: #F8FAFC;">{latest['successful_examples']}/{latest['total_examples']}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Run your first benchmark to populate this card.")

        # Quick action button directly on card
        if st.button("▶ Trigger Quick 5-Example Mock Run", type="primary", use_container_width=True):
            cfg = BenchmarkConfig(
                dataset_path="datasets/basic_v1.jsonl",
                model_name="mock",
                limit=5,
                generation_config=GenerationConfig(max_new_tokens=32, temperature=0.0),
                output_dir=str(STORAGE_DIR),
            )
            with st.spinner("Executing mock tasks..."):
                runner = BenchmarkRunner(cfg)
                res = runner.run()
                db.save_run(res)
                st.session_state["active_run_id"] = res.run_id
            st.success("Run completed!")
            time.sleep(0.5)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Benchmark Runs Table (Frame 1)
    st.markdown("### 📋 Recent Evaluation History")
    if all_runs:
        df_recent = pd.DataFrame(all_runs)
        cols = ["run_id", "timestamp", "model_name", "dataset_name", "total_examples", "accuracy", "exact_match_rate", "median_latency"]
        st.dataframe(df_recent[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No run logs found in storage.")


# ==============================================================================
# SCREEN 2: RUN BENCHMARK (Frame 2 from Figma)
# ==============================================================================
elif nav_choice == "Run Benchmark":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / BENCHMARK RUNNER</div>
                <div class="page-title">Run Benchmark</div>
                <div class="page-subtitle">Configure hardware-safe model evaluation runs on local CPU.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_setup, col_meta = st.columns([1.6, 1.4])

    with col_setup:
        st.markdown("#### 1. Model Configuration")
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

        st.markdown("#### 2. Dataset & Sampling Scope")
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

        st.markdown("#### 3. Hyperparameters")
        with st.expander("⚙️ Generation Parameters (Deterministic Default)", expanded=False):
            max_tokens = st.slider("Max New Tokens", min_value=8, max_value=256, value=64, step=8)
            temperature = st.slider("Temperature (0.0 = Deterministic Greedy)", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
            seed = st.number_input("Random Seed", value=42)

    with col_meta:
        st.markdown("#### 🔍 Hardware Safety & Model Profile")
        adapter_preview = ModelRegistry.get_adapter(model_target)
        meta = adapter_preview.get_metadata()

        is_mock_selected = "mock" in meta.backend_type.lower()
        badge_type = (
            '<span class="badge-mock">🧪 ZERO-DOWNLOAD SIMULATION</span>'
            if is_mock_selected
            else '<span class="badge-real">🧠 OPEN-WEIGHT CAUSAL LM</span>'
        )

        st.markdown(
            f"""
            <div class="surface-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">{meta.model_name}</span>
                    {badge_type}
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 12px; line-height: 1.4;">
                    {meta.description}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.78rem; font-family: monospace; background: #070A0F; padding: 10px; border-radius: 6px;">
                    <div><span style="color: #64748B;">Parameters:</span> <span style="color: #F8FAFC;">{meta.parameters}</span></div>
                    <div><span style="color: #64748B;">License:</span> <span style="color: #F8FAFC;">{meta.license}</span></div>
                    <div><span style="color: #64748B;">Est. RAM:</span> <span style="color: #22D3EE;">{meta.estimated_ram_mb:.0f} MB</span></div>
                    <div><span style="color: #64748B;">Backend:</span> <span style="color: #F8FAFC;">{meta.backend_type}</span></div>
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

        if not is_mock_selected:
            st.info("💡 Real neural models download checkpoint files to `~/.cache/huggingface` on first execution.")

    st.divider()

    if st.button("🚀 Execute Benchmark Run", type="primary", use_container_width=True):
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
# SCREEN 3: RUN DETAILS (Frame 3 from Figma)
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
                '<span class="badge-mock">🧪 MOCK SIMULATION</span>'
                if is_mock
                else '<span class="badge-real">🧠 REAL NEURAL WEIGHTS</span>'
            )

            # 4-Card KPI Row (Frame 3)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Accuracy</div>
                        <div class="kpi-value" style="color: #34D399;">{run_res.accuracy:.1f}%</div>
                        <div class="kpi-sub">Target answer match</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Exact Match</div>
                        <div class="kpi-value" style="color: #22D3EE;">{run_res.exact_match_rate:.1f}%</div>
                        <div class="kpi-sub">String identity match</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Compliance Score</div>
                        <div class="kpi-value" style="color: #A78BFA;">{run_res.avg_compliance_score:.2f}</div>
                        <div class="kpi-sub">Constraint satisfaction</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r4:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Median Latency</div>
                        <div class="kpi-value" style="color: #F8FAFC;">{run_res.latency_summary.median_latency_seconds:.3f} s</div>
                        <div class="kpi-sub">CPU single-thread</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # Visualizations: Category Bar Chart + Latency Histogram
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
                        color_discrete_sequence=["#06B6D4", "#10B981", "#F59E0B"],
                    )
                    apply_plotly_fig_theme(fig_cat, height=270)
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
                lat_values = [t.latency_seconds for t in run_res.task_results if t.error is None]
                if lat_values:
                    fig_lat = px.histogram(
                        lat_values,
                        nbins=12,
                        color_discrete_sequence=["#8B5CF6"],
                    )
                    fig_lat.add_vline(
                        x=run_res.latency_summary.median_latency_seconds,
                        line_dash="dash",
                        line_color="#22D3EE",
                        annotation_text=f"Median ({run_res.latency_summary.median_latency_seconds:.3f}s)",
                        annotation_font_color="#22D3EE",
                    )
                    apply_plotly_fig_theme(fig_lat, height=270)
                    st.plotly_chart(fig_lat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Per-Task Detailed Inspection Accordion (Frame 3)
            st.markdown("### 🔍 Per-Task Interactive Audit")
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
                    '<span class="badge-pass">✅ CORRECT</span>'
                    if t.is_correct
                    else ('<span class="badge-fail">⚠️ ERROR</span>' if t.error else '<span class="badge-fail">❌ INCORRECT</span>')
                )
                with st.expander(f"{t.task_id} [{t.category}] &bull; {t.latency_seconds:.3f}s"):
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
                            f"- Latency: `{t.latency_seconds:.3f} s`"
                        )
                        if t.error:
                            st.error(f"Inference Error: {t.error}")
                        if t.compliance_details:
                            st.caption(f"Rule Checks: {t.compliance_details}")

            st.divider()

            # Data Exports
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
# SCREEN 4: MODEL COMPARISON (Frame 4 from Figma — Radar Chart & Deltas)
# ==============================================================================
elif nav_choice == "Model Comparison":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / COMPARATIVE ANALYSIS</div>
                <div class="page-title">Model Comparison Matrix</div>
                <div class="page-subtitle">Cross-model radar charts, accuracy deltas, and throughput comparisons.</div>
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
        run_options = {f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})": r['run_id'] for r in all_runs}
        selected_runs_labels = st.multiselect(
            "Select Evaluation Runs to Compare",
            options=list(run_options.keys()),
            default=list(run_options.keys())[:min(3, len(run_options))],
        )

        if len(selected_runs_labels) < 2:
            st.warning("Please select at least 2 runs to display side-by-side comparisons.")
        else:
            selected_ids = [run_options[l] for l in selected_runs_labels]
            selected_run_objects = [db.get_run(rid) for rid in selected_ids if db.get_run(rid)]

            # Side-by-Side Summary Table
            st.markdown("#### 📋 Comparative Metrics Table")
            comp_table = generate_comparison_table(selected_run_objects)
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # Frame 4: Radar Chart & Comparative Bar Charts
            rad_col, bar_col = st.columns([1.3, 1.7])

            with rad_col:
                st.markdown(
                    """
                    <div class="surface-card">
                        <div class="surface-card-title">🕸️ Multidimensional Radar Evaluation</div>
                    """,
                    unsafe_allow_html=True,
                )
                # Compute category-wise accuracy for radar chart
                all_cats = ["multiple_choice", "factual_qa", "arithmetic_reasoning", "instruction_following", "structured_output"]
                fig_radar = go.Figure()
                palette = ["#06B6D4", "#A78BFA", "#34D399", "#F59E0B"]

                for i, r_obj in enumerate(selected_run_objects):
                    color = palette[i % len(palette)]
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
                            line=dict(color=color, width=2),
                            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.15])}",
                        )
                    )

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)", color="#94A3B8"),
                        bgcolor="#111827",
                    ),
                    paper_bgcolor="#111827",
                    font=dict(color="#94A3B8", family="JetBrains Mono, monospace", size=10),
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=300,
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
                df_comp = pd.DataFrame(comp_table)
                fig_comp_bars = px.bar(
                    df_comp,
                    x="Model Name",
                    y=["Accuracy (%)", "Exact Match (%)"],
                    barmode="group",
                    color_discrete_sequence=["#06B6D4", "#34D399"],
                )
                apply_plotly_fig_theme(fig_comp_bars, height=300)
                st.plotly_chart(fig_comp_bars, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 5: MODELS & DATASETS (Frame 5 from Figma)
# ==============================================================================
elif nav_choice == "05. Models & Datasets" or nav_choice == "Models & Datasets":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / CATALOG & SPECIFICATION</div>
                <div class="page-title">Models & Datasets</div>
                <div class="page-subtitle">Real model registry metadata, RAM limits, and benchmark task schema.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_models, tab_datasets = st.tabs(["Supported Models", "Benchmark Dataset Inspector"])

    with tab_models:
        st.markdown("### 🧠 Supported Language Models & Mocks")
        models = ModelRegistry.list_available_models()
        for m in models:
            is_mock = "mock" in m.backend_type.lower()
            badge = '<span class="badge-mock">MOCK ENGINE</span>' if is_mock else '<span class="badge-real">TRANSFORMERS CPU</span>'
            osi_badge = '<span class="badge-pass">OSI APPROVED</span>' if m.license_is_osi_approved else '<span class="badge-fail">CHECK LICENSE</span>'

            st.markdown(
                f"""
                <div class="surface-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">{m.model_name}</span>
                        <div>{badge} &nbsp; {osi_badge}</div>
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">{m.description}</div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.78rem; font-family: monospace; background: #070A0F; padding: 10px; border-radius: 6px;">
                        <div><span style="color: #64748B;">Repo ID:</span> <span style="color: #38BDF8;">{m.model_id}</span></div>
                        <div><span style="color: #64748B;">Parameters:</span> <span style="color: #F8FAFC;">{m.parameters}</span></div>
                        <div><span style="color: #64748B;">Est. RAM:</span> <span style="color: #22D3EE;">{m.estimated_ram_mb:.0f} MB</span></div>
                        <div><span style="color: #64748B;">License:</span> <span style="color: #F8FAFC;">{m.license}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_datasets:
        st.markdown("### 📋 `basic_v1.jsonl` Benchmark Dataset")
        dataset_path = current_dir / "datasets" / "basic_v1.jsonl"
        if dataset_path.exists():
            val_report = validate_dataset_file(dataset_path)
            st.success(
                f"✅ Dataset validated: **{val_report.total_items}** tasks across **{len(val_report.category_counts)}** categories."
            )

            d_items = load_dataset(dataset_path)
            items_table = []
            for it in d_items:
                items_table.append({
                    "Task ID": it.task_id,
                    "Category": it.category,
                    "Prompt Snippet": it.prompt[:50] + "..." if len(it.prompt) > 50 else it.prompt,
                    "Expected Answer": str(it.expected_answer),
                    "Scoring Method": it.scoring_method,
                })
            st.dataframe(pd.DataFrame(items_table), use_container_width=True, hide_index=True)
        else:
            st.error(f"Dataset file not found at `{dataset_path}`.")


# ==============================================================================
# SCREEN 6: METHODOLOGY (Frame 6 from Figma)
# ==============================================================================
elif nav_choice == "06. Methodology & Hardware" or nav_choice == "Methodology":
    st.markdown(
        """
        <div class="portal-bar">
            <div>
                <div class="portal-tag">PORTAL / RESEARCH DOCUMENTATION</div>
                <div class="page-title">Methodology & Hardware Principles</div>
                <div class="page-subtitle">Standardized evaluation rules for resource-constrained environments.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="surface-card">
            <h4 style="color: #F8FAFC; margin-bottom: 8px;">1. Objective Deterministic Grading</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                TinyBench completely avoids non-deterministic LLM-as-a-judge scoring. Every benchmark item is evaluated using 
                strict algorithmic validators:
            </p>
            <ul style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.8;">
                <li><strong>Multiple Choice:</strong> Regex token extraction against option letters (A/B/C/D) with strict boundary isolation.</li>
                <li><strong>Exact Match:</strong> Punctuation and whitespace normalization with casing indifference and allowable synonym aliases.</li>
                <li><strong>Numeric Match:</strong> Floating-point extraction with configurable tolerance thresholds (e.g. <code>tolerance: 0.01</code>).</li>
                <li><strong>Instruction Compliance:</strong> Verifies hard programmatic constraints (e.g. exact word counts, forbidden words, required prefixes).</li>
                <li><strong>JSON Schema:</strong> Validates JSON parsability, key presence, and schema value types.</li>
            </ul>
        </div>

        <div class="surface-card">
            <h4 style="color: #F8FAFC; margin-bottom: 8px;">2. CPU Monotonic Latency & Memory Profiling</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                Inference timing uses <code>time.perf_counter()</code> around token generation passes. Model loading duration 
                (<code>init_latency</code>) is measured separately from per-task generation latency. Process memory is continuously 
                profiled via <code>psutil.Process().memory_info().rss</code> to track resident set size and ensure models fit within laptop RAM.
            </p>
        </div>

        <div class="surface-card">
            <h4 style="color: #F8FAFC; margin-bottom: 8px;">3. Distinction: Mock Engine vs Real Open-Weight Weights</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                - <strong>Mock Engine (<code>mock</code>, <code>mock-noisy</code>):</strong> Simulated deterministic outputs consuming &lt; 35 MB RAM. 
                Ideal for Hacktoberfest contributors, CI/CD automated testing, and UI preview without downloading weight checkpoints.<br>
                - <strong>Real Transformers Backend (<code>smollm2-135m</code>, <code>qwen2.5-0.5b</code>):</strong> Authentic PyTorch CPU tensor execution 
                evaluating genuine model capabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TECHNICAL DEVELOPER FOOTER (Figma-Matched)
# ==============================================================================
st.markdown(
    f"""
    <div class="developer-footer">
        <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1.5fr; gap: 20px; margin-bottom: 20px;">
            <div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem; margin-bottom: 6px;">
                    ⚡ TinyBench
                </div>
                <div style="color: #64748B; font-size: 0.76rem; line-height: 1.5; margin-bottom: 10px;">
                    Lightweight open-weight LLM benchmarking toolkit engineered for ordinary CPU-only laptops.
                    Designed for Hacktoberfest, reproducible research, and student experimentation.
                </div>
                <div style="font-size: 0.7rem; color: #475569;">
                    Apache-2.0 License &bull; Free & Open-Source
                </div>
            </div>
            <div>
                <div class="footer-col-title">Documentation</div>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench#readme" target="_blank">Quickstart Guide</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/benchmark_dataset_spec.md" target="_blank">Dataset Schema</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/model_integration_guide.md" target="_blank">Adding Models</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/reproducibility.md" target="_blank">Reproducibility</a>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Community</div>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench" target="_blank">GitHub Repository</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/issues" target="_blank">Issue Tracker</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/CONTRIBUTING.md" target="_blank">Contributing Guide</a>
                    <a class="footer-link" href="https://github.com/shettyadithyar76-prog/tinybench/blob/main/docs/beginner_issues.md" target="_blank">Hacktoberfest Issues</a>
                </div>
            </div>
            <div>
                <div class="footer-col-title">Runtime Specification</div>
                <div style="font-family: monospace; font-size: 0.72rem; color: #64748B; display: flex; flex-direction: column; gap: 3px;">
                    <div>Platform: <span style="color: #94A3B8;">{sys_info.os_name}</span></div>
                    <div>Cores: <span style="color: #94A3B8;">{sys_info.cpu_physical_cores} Physical / {sys_info.cpu_logical_cores} Logical</span></div>
                    <div>Python: <span style="color: #94A3B8;">{sys_info.python_version.split()[0]}</span></div>
                    <div>RAM: <span style="color: #22D3EE;">{sys_info.available_ram_mb:.0f} MB Available</span></div>
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: #475569;">
            <div>&copy; 2026 TinyBench Contributors &bull; Open-Weight AI Benchmarking</div>
            <div style="font-family: monospace;">Branch: main</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
