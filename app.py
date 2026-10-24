"""TinyBench Streamlit Dashboard - Interactive Research & Evaluation Interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add src to sys.path so tinybench modules are importable
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tinybench.benchmarks.schema import TaskCategory
from tinybench.benchmarks.validator import validate_dataset_file
from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
from tinybench.models.registry import ModelRegistry
from tinybench.reporting import export_run_csv, generate_comparison_table
from tinybench.runner import BenchmarkRunner, RunResult, TaskResult
from tinybench.storage import RunDatabase

# Initialize storage
STORAGE_DIR = current_dir / "runs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
db = RunDatabase(STORAGE_DIR)

st.set_page_config(
    page_title="TinyBench — LLM Benchmarking Toolkit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for research aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .status-badge-ok {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.markdown("### ⚡ TinyBench Toolkit")
st.sidebar.markdown("*Lightweight LLM evaluation on CPU laptops*")
st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "Navigation",
    ["Overview", "Run Benchmark", "Evaluation Results", "Model Comparison", "Run History & Export"],
    index=0,
)

st.sidebar.divider()
sys_info = SystemInfo.capture()
st.sidebar.caption(
    f"**System Specs:**\n"
    f"- OS: {sys_info.os_name} ({sys_info.architecture})\n"
    f"- CPU: {sys_info.processor[:24]}...\n"
    f"- Physical Cores: {sys_info.cpu_physical_cores}\n"
    f"- Available RAM: {sys_info.available_ram_mb:.0f} MB / {sys_info.total_ram_mb:.0f} MB\n"
    f"- Python: {sys_info.python_version}\n"
    f"- PyTorch: {sys_info.torch_version or 'Not installed'}"
)


# ==========================================
# 1. OVERVIEW PAGE
# ==========================================
if nav_choice == "Overview":
    st.markdown('<div class="main-header">⚡ TinyBench Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A modular, reproducible open-weight language model benchmarking toolkit '
        'engineered for ordinary CPU hardware.</div>',
        unsafe_allow_html=True,
    )

    all_runs = db.list_runs(limit=100)
    models_catalog = ModelRegistry.list_available_models()

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Completed Runs</div>'
            f'<div class="metric-value">{len(all_runs)}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Registered Models</div>'
            f'<div class="metric-value">{len(models_catalog)}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Standard Tasks</div>'
            f'<div class="metric-value">35 Tasks</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">RAM Available</div>'
            f'<div class="metric-value">{sys_info.available_ram_mb:.0f} MB</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🎯 Key Design Principles")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.info(
            "**CPU-Accessible**\n\n"
            "Runs entirely on modest laptops without requiring dedicated GPUs, CUDA drivers, "
            "or paid cloud APIs. Perfect for students and independent researchers."
        )
    with p2:
        st.success(
            "**Deterministic Grading**\n\n"
            "Evaluates accuracy, exact string matches, numerical solutions, and strict rule-based "
            "instruction compliance without subjective LLM-as-a-judge bias."
        )
    with p3:
        st.warning(
            "**Zero-Download Testing**\n\n"
            "Features built-in mock simulation backends so developers and Hacktoberfest contributors "
            "can verify pipelines and tests instantly without downloading checkpoints."
        )

    st.markdown("### 🚀 Latest Evaluation Summary")
    if all_runs:
        latest = all_runs[0]
        st.markdown(f"**Most Recent Run:** `{latest['run_id']}` ({latest['timestamp'][:19]})")
        lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
        lcol1.metric("Model", latest["model_name"])
        lcol2.metric("Accuracy", f"{latest['accuracy']:.1f}%")
        lcol3.metric("Exact Match", f"{latest['exact_match_rate']:.1f}%")
        lcol4.metric("Compliance", f"{latest['avg_compliance_score']:.2f}")
        lcol5.metric("Median Latency", f"{latest['median_latency']:.3f} s")
    else:
        st.markdown(
            """
            > **No evaluations have been run yet.**
            >
            > Head over to the **Run Benchmark** tab on the left to start your first evaluation with a mock model
            > or a local open-weight model!
            """
        )

    st.markdown("### 📋 Supported Model Catalog")
    table_data = []
    for m in models_catalog:
        table_data.append({
            "Model Identifier": m.model_id,
            "Name": m.model_name,
            "Parameters": m.parameters,
            "Est. RAM (MB)": f"{m.estimated_ram_mb:.0f} MB",
            "License": m.license,
            "OSI Approved": "✅ Yes" if m.license_is_osi_approved else "⚠️ Check License",
            "Backend": m.backend_type,
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ==========================================
# 2. RUN BENCHMARK PAGE
# ==========================================
elif nav_choice == "Run Benchmark":
    st.markdown('<div class="main-header">⚡ Benchmark Runner Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure and execute standardized evaluations.</div>', unsafe_allow_html=True)

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
        selected_model_option = st.selectbox("Select Model Backend", model_options, index=0)

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

        st.markdown("#### 2. Dataset & Sampling")
        dataset_path_str = st.text_input("Dataset Path", "datasets/basic_v1.jsonl")

        # Category selection
        available_categories = [c.value for c in TaskCategory]
        selected_categories = st.multiselect(
            "Filter Categories (optional)",
            options=available_categories,
            default=available_categories,
        )

        run_preset = st.radio("Evaluation Scope", ["Quick Evaluation (5 examples)", "Standard Evaluation (15 examples)", "Full Dataset (All examples)", "Custom Limit"], horizontal=True)
        if run_preset == "Quick Evaluation (5 examples)":
            limit_val = 5
        elif run_preset == "Standard Evaluation (15 examples)":
            limit_val = 15
        elif run_preset == "Full Dataset (All examples)":
            limit_val = None
        else:
            limit_val = st.number_input("Custom Example Limit", min_value=1, max_value=500, value=10)

        st.markdown("#### 3. Generation Parameters")
        with st.expander("Tuning Parameters", expanded=False):
            max_tokens = st.slider("Max New Tokens", min_value=8, max_value=256, value=64, step=8)
            temperature = st.slider("Temperature (0.0 = Deterministic Greedy)", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
            seed = st.number_input("Random Seed", value=42)

    # Model metadata sidebar info
    with col_meta:
        st.markdown("#### 🔍 Selected Model Profile")
        adapter_preview = ModelRegistry.get_adapter(model_target)
        meta = adapter_preview.get_metadata()

        st.markdown(
            f"""
            - **Model Name**: {meta.model_name}
            - **Architecture Family**: `{meta.family}`
            - **Parameters**: `{meta.parameters}`
            - **Backend Implementation**: `{meta.backend_type}`
            - **License**: `{meta.license}` (OSI: {'Yes' if meta.license_is_osi_approved else 'No / Custom'})
            - **Estimated RAM Required**: **{meta.estimated_ram_mb:.0f} MB**
            """
        )

        # Hardware safety assessment
        if sys_info.available_ram_mb < meta.estimated_ram_mb:
            st.error(
                f"⚠️ **Memory Warning**: This system currently has **{sys_info.available_ram_mb:.0f} MB** available RAM, "
                f"which is below the recommended **{meta.estimated_ram_mb:.0f} MB** for this model."
            )
        else:
            st.success(
                f"✅ **Hardware Compatible**: Current available RAM ({sys_info.available_ram_mb:.0f} MB) "
                f"is sufficient for estimated needs ({meta.estimated_ram_mb:.0f} MB)."
            )

        if "mock" not in model_target:
            st.info(
                "💡 **Open-Weight Execution Note**\n\n"
                "Local inference will load neural weights onto CPU via PyTorch and Transformers. "
                "If this is your first time loading this checkpoint, it will be cached locally in `~/.cache/huggingface`."
            )

    st.divider()

    # Execution Action
    if st.button("🚀 Start Benchmark Evaluation", type="primary", use_container_width=True):
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
                f"**Latest Output for `{res.task_id}` ({res.latency_seconds:.3f}s):** "
                f"`{res.generated_text[:60] if res.generated_text else 'None'}` "
                f"({'✅ Correct' if res.is_correct else '❌ Incorrect'})"
            )
            live_rows.append({
                "Task ID": res.task_id,
                "Category": res.category,
                "Status": "✅ OK" if res.is_correct else ("❌ Error" if res.error else "❌ Fail"),
                "Latency (s)": res.latency_seconds,
                "Generated Output": (res.generated_text[:40] + "...") if len(res.generated_text) > 40 else res.generated_text,
            })
            table_placeholder.dataframe(pd.DataFrame(live_rows), use_container_width=True)

        try:
            with st.spinner("Executing benchmark tasks..."):
                run_result = runner.run(progress_callback=on_step)
                db.save_run(run_result)
                st.session_state["active_run_id"] = run_result.run_id

            progress_bar.progress(100, text="Evaluation completed successfully!")
            st.success(
                f"🎉 Evaluation `{run_result.run_id}` completed! "
                f"Accuracy: **{run_result.accuracy:.1f}%**, Exact Match: **{run_result.exact_match_rate:.1f}%**, "
                f"Median Latency: **{run_result.latency_summary.median_latency_seconds:.3f}s**."
            )
            st.balloons()
        except Exception as exc:
            st.error(f"❌ Evaluation encountered an error: {exc}")


# ==========================================
# 3. EVALUATION RESULTS PAGE
# ==========================================
elif nav_choice == "Evaluation Results":
    st.markdown('<div class="main-header">📊 Evaluation Results & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed performance metrics, error audits, and category diagnostics.</div>', unsafe_allow_html=True)

    all_runs = db.list_runs(limit=100)
    if not all_runs:
        st.info("No evaluation runs exist yet. Go to **Run Benchmark** to perform an evaluation.")
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
            st.error(f"Could not load details for run {selected_run_id}")
        else:
            # High-level Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Task Accuracy", f"{run_res.accuracy:.1f}%", help="Percentage of objectively correct responses.")
            m2.metric("Exact Match", f"{run_res.exact_match_rate:.1f}%", help="Normalized string match rate.")
            m3.metric("Instruction Compliance", f"{run_res.avg_compliance_score:.2f}", help="Score based on deterministic constraints.")
            m4.metric("Median Latency", f"{run_res.latency_summary.median_latency_seconds:.3f} s", help="Median CPU inference time.")
            m5.metric("Peak Process RAM", f"{run_res.memory_measurement.peak_rss_mb:.1f} MB", help="Peak Resident Set Size during run.")

            st.divider()

            # Visualizations Row
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("##### Performance by Benchmark Category")
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
                        title="Category Metric Breakdown",
                        color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B"],
                    )
                    fig_cat.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_cat, use_container_width=True)

            with col_chart2:
                st.markdown("##### Task Generation Latency Distribution")
                lat_values = [t.latency_seconds for t in run_res.task_results if t.error is None]
                if lat_values:
                    fig_lat = px.histogram(
                        lat_values,
                        nbins=12,
                        labels={"value": "Latency (seconds)"},
                        title="Inference Latency Spread (s)",
                        color_discrete_sequence=["#6366F1"],
                    )
                    fig_lat.add_vline(
                        x=run_res.latency_summary.median_latency_seconds,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Median ({run_res.latency_summary.median_latency_seconds:.3f}s)",
                    )
                    fig_lat.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_lat, use_container_width=True)

            st.divider()

            # Task Results Inspection
            st.markdown("### 🔍 Per-Task Detailed Inspection")
            fcol1, fcol2 = st.columns([1, 1])
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

            # Filter tasks
            filtered_tasks = run_res.task_results
            if cat_filter != "All Categories":
                filtered_tasks = [t for t in filtered_tasks if t.category == cat_filter]

            if status_filter == "Correct Only":
                filtered_tasks = [t for t in filtered_tasks if t.is_correct]
            elif status_filter == "Incorrect Only":
                filtered_tasks = [t for t in filtered_tasks if not t.is_correct and not t.error]
            elif status_filter == "Errors Only":
                filtered_tasks = [t for t in filtered_tasks if t.error]

            st.caption(f"Displaying {len(filtered_tasks)} of {len(run_res.task_results)} tasks.")

            for t in filtered_tasks:
                status_icon = "✅ Correct" if t.is_correct else ("⚠️ Error" if t.error else "❌ Failed")
                with st.expander(f"{status_icon} — {t.task_id} [{t.category}] ({t.latency_seconds:.3f}s)"):
                    c_p, c_e = st.columns(2)
                    with c_p:
                        st.markdown("**Prompt:**")
                        st.code(t.prompt, language="text")
                        st.markdown(f"**Expected Answer:** `{t.expected_answer}`")
                    with c_e:
                        st.markdown("**Model Output:**")
                        st.code(t.generated_text if t.generated_text else "(empty output)", language="text")
                        st.markdown(
                            f"- Exact Match: {'Yes' if t.exact_match else 'No'}\n"
                            f"- Compliance Score: {t.compliance_score:.2f}\n"
                            f"- Latency: {t.latency_seconds:.3f} s"
                        )
                        if t.error:
                            st.error(f"Execution Error: {t.error}")
                        if t.compliance_details:
                            st.caption(f"Rule Checks: {t.compliance_details}")


# ==========================================
# 4. MODEL COMPARISON PAGE
# ==========================================
elif nav_choice == "Model Comparison":
    st.markdown('<div class="main-header">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare multiple models side-by-side across quality, latency, and resource metrics.</div>', unsafe_allow_html=True)

    all_runs = db.list_runs(limit=100)
    if len(all_runs) < 2:
        st.info(
            "Model comparison requires at least 2 completed evaluation runs. "
            "Please run a benchmark for two different models or settings in the **Run Benchmark** tab."
        )
    else:
        run_options = {f"{r['run_id']} — {r['model_name']} ({r['timestamp'][:19]})": r['run_id'] for r in all_runs}
        selected_runs_labels = st.multiselect(
            "Select Runs to Compare",
            options=list(run_options.keys()),
            default=list(run_options.keys())[:min(3, len(run_options))],
        )

        if len(selected_runs_labels) < 2:
            st.warning("Please select at least 2 runs to display comparisons.")
        else:
            selected_ids = [run_options[l] for l in selected_runs_labels]
            selected_run_objects = [db.get_run(rid) for rid in selected_ids if db.get_run(rid)]

            st.markdown("#### 📋 Side-by-Side Summary Table")
            comp_table = generate_comparison_table(selected_run_objects)
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("##### Task Quality Metrics Comparison")
                df_comp = pd.DataFrame(comp_table)
                fig_q = px.bar(
                    df_comp,
                    x="Model Name",
                    y=["Accuracy (%)", "Exact Match (%)"],
                    barmode="group",
                    title="Accuracy vs Exact Match (%)",
                    color_discrete_sequence=["#3B82F6", "#10B981"],
                )
                fig_q.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_q, use_container_width=True)

            with c2:
                st.markdown("##### Latency & Memory Profile Comparison")
                fig_l = px.bar(
                    df_comp,
                    x="Model Name",
                    y=["Median Latency (s)", "Mean Latency (s)"],
                    barmode="group",
                    title="Inference Latency (seconds)",
                    color_discrete_sequence=["#8B5CF6", "#EC4899"],
                )
                fig_l.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_l, use_container_width=True)


# ==========================================
# 5. RUN HISTORY & EXPORT PAGE
# ==========================================
elif nav_choice == "Run History & Export":
    st.markdown('<div class="main-header">💾 Run History & Data Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect historical benchmark logs and export machine-readable CSV/JSON artifacts.</div>', unsafe_allow_html=True)

    all_runs = db.list_runs(limit=100)
    if not all_runs:
        st.info("No runs stored in the local SQLite database.")
    else:
        st.markdown(f"**Found {len(all_runs)} runs in `{STORAGE_DIR}`:**")
        df_runs = pd.DataFrame(all_runs)
        cols_to_show = ["run_id", "timestamp", "model_name", "dataset_name", "total_examples", "accuracy", "exact_match_rate", "median_latency", "peak_memory_mb"]
        st.dataframe(df_runs[cols_to_show], use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("#### 📥 Export Benchmark Run Artifacts")
        target_run_id = st.selectbox(
            "Select Run ID to Export",
            [r["run_id"] for r in all_runs],
        )

        run_to_export = db.get_run(target_run_id)
        if run_to_export:
            # Prepare CSV in memory
            tasks_csv_path = STORAGE_DIR / f"{target_run_id}_tasks.csv"
            export_run_csv(run_to_export, tasks_csv_path, detail_level="tasks")
            with open(tasks_csv_path, "r", encoding="utf-8") as f:
                csv_tasks_content = f.read()

            summary_csv_path = STORAGE_DIR / f"{target_run_id}_summary.csv"
            export_run_csv(run_to_export, summary_csv_path, detail_level="summary")
            with open(summary_csv_path, "r", encoding="utf-8") as f:
                csv_summary_content = f.read()

            json_content = json.dumps(run_to_export.model_dump(), indent=2)

            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.download_button(
                    label="📄 Download Per-Task CSV",
                    data=csv_tasks_content,
                    file_name=f"{target_run_id}_tasks.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dcol2:
                st.download_button(
                    label="📊 Download Run Summary CSV",
                    data=csv_summary_content,
                    file_name=f"{target_run_id}_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dcol3:
                st.download_button(
                    label="💾 Download Full JSON Artifact",
                    data=json_content,
                    file_name=f"{target_run_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()
            with st.expander("🛠️ Reproducibility Metadata Inspector"):
                st.json(run_to_export.system_info.model_dump())
