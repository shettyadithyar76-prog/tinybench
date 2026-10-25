"""Command-line interface for the TinyBench evaluation toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tinybench.benchmarks.validator import validate_dataset_file
from tinybench.config import BenchmarkConfig, GenerationConfig
from tinybench.models.downloader import download_model, is_model_installed
from tinybench.models.registry import APPROVED_MODEL_IDS, ModelRegistry
from tinybench.reporting import (
    export_run_csv,
    export_run_json,
    format_markdown_summary,
    generate_comparison_table,
)
from tinybench.runner import BenchmarkRunner, RunResult
from tinybench.storage import RunDatabase


def create_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="tinybench",
        description="TinyBench: Lightweight Open-Weight LLM Benchmarking Toolkit for CPU Laptops",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. list-models
    subparsers.add_parser("list-models", help="List the three approved open-weight benchmark models.")

    # 2. download
    dl_parser = subparsers.add_parser("download", help="Download and verify model weights into local storage.")
    dl_parser.add_argument(
        "--model",
        required=True,
        help="Approved model name or alias (e.g. smollm2-1.7b, llama-3.2-1b, gemma4-e2b).",
    )
    dl_parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (required for gated models like Llama 3.2 1B).",
    )

    # 3. validate-dataset
    val_parser = subparsers.add_parser("validate-dataset", help="Validate benchmark dataset schema and balance.")
    val_parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/basic_v1.jsonl",
        help="Path to .jsonl dataset file (default: datasets/basic_v1.jsonl).",
    )

    # 4. evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Execute an evaluation run against a model.")
    eval_parser.add_argument(
        "--model",
        default="smollm2-1.7b",
        help="Model identifier or alias (e.g. smollm2-1.7b, llama-3.2-1b, gemma4-e2b) (default: smollm2-1.7b).",
    )
    eval_parser.add_argument(
        "--dataset",
        default="datasets/basic_v1.jsonl",
        help="Path to .jsonl dataset (default: datasets/basic_v1.jsonl).",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to evaluate.")
    eval_parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated category filters (e.g. multiple_choice,factual_qa).",
    )
    eval_parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum new tokens to generate (default: 64).")
    eval_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0).")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for generation (default: 42).")
    eval_parser.add_argument("--output-dir", default="runs", help="Output directory for runs (default: runs).")
    eval_parser.add_argument("--quiet", action="store_true", help="Suppress live progress messages.")

    # 5. compare
    comp_parser = subparsers.add_parser("compare", help="Compare multiple models or evaluation runs.")
    comp_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model aliases to run and compare.",
    )
    comp_parser.add_argument(
        "--runs",
        default=None,
        help="Comma-separated list of existing run IDs to compare.",
    )
    comp_parser.add_argument(
        "--dataset",
        default="datasets/basic_v1.jsonl",
        help="Dataset to use when evaluating --models (default: datasets/basic_v1.jsonl).",
    )
    comp_parser.add_argument("--limit", type=int, default=10, help="Example limit when evaluating --models (default: 10).")
    comp_parser.add_argument("--output-dir", default="runs", help="Directory where runs are stored (default: runs).")

    # 6. export
    export_parser = subparsers.add_parser("export", help="Export an evaluation run to CSV or JSON.")
    export_parser.add_argument("--run-id", required=True, help="Run ID to export.")
    export_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv).",
    )
    export_parser.add_argument(
        "--detail-level",
        choices=["tasks", "summary"],
        default="tasks",
        help="CSV detail level: 'tasks' (per-example) or 'summary' (one-row) (default: tasks).",
    )
    export_parser.add_argument("--output", default=None, help="Target file path (defaults to runs/<run_id>.<format>).")
    export_parser.add_argument("--output-dir", default="runs", help="Directory where runs are stored (default: runs).")

    # 7. history
    hist_parser = subparsers.add_parser("history", help="List previous evaluation runs.")
    hist_parser.add_argument("--limit", type=int, default=20, help="Max runs to display (default: 20).")
    hist_parser.add_argument("--output-dir", default="runs", help="Directory where runs are stored (default: runs).")

    return parser


def handle_list_models() -> int:
    """Print verified single source of truth for the 3 approved models."""
    models = ModelRegistry.list_available_models()
    print("\n" + "=" * 115)
    print("TinyBench Approved Open-Weight Benchmark Models (Official Catalogue)")
    print("=" * 115)
    fmt = "{:<24} {:<16} {:<12} {:<10} {:<16} {:<14} {:<18}"
    print(fmt.format("Model Name", "Developer", "Params", "Download", "License", "Runtime", "Status"))
    print("-" * 115)
    for m in models:
        print(fmt.format(
            m.model_name[:23],
            m.developer[:15],
            m.parameters[:11],
            m.download_size[:9],
            m.license[:15],
            m.supported_runtime[:13],
            m.status[:17],
        ))
    print("=" * 115)
    print("Exact approved upstream identifiers:")
    for m in models:
        print(f"  • {m.model_name}: {m.model_id} ({m.license_url})")
    print("=" * 115 + "\n")
    return 0


def handle_download(args: argparse.Namespace) -> int:
    """Download model weights with explicit user initiation and integrity checks."""
    model_input = args.model.strip()
    if not ModelRegistry.is_approved_model(model_input):
        print(
            f"\n[ERROR] '{model_input}' is not an approved model. "
            f"Approved models: {', '.join(APPROVED_MODEL_IDS)}\n",
            file=sys.stderr,
        )
        return 1

    adapter = ModelRegistry.get_adapter(model_input)
    meta = adapter.get_metadata()

    print(f"\nInitiating verified download for {meta.model_name}...")
    print(f"Upstream Repository: {meta.model_id}")
    print(f"Developer: {meta.developer} | License: {meta.license}")
    print(f"Download Size: {meta.download_size} | Runtime: {meta.supported_runtime}")
    print("-" * 70)

    def on_progress(status_text: str, pct: float | None) -> None:
        pct_str = f" [{int(pct * 100)}%]" if pct is not None else ""
        print(f"  --> {status_text}{pct_str}")

    try:
        dest = download_model(meta.model_id, token=args.token, progress_callback=on_progress)
        print(f"\n[SUCCESS] Model successfully verified and ready at: {dest}\n")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}\n", file=sys.stderr)
        return 1


def handle_validate_dataset(dataset_path: str) -> int:
    """Validate dataset and report stats."""
    path = Path(dataset_path)
    print(f"\nValidating dataset: {path} ...")
    report = validate_dataset_file(path)

    if not report.is_valid:
        print("\n[FAILED] Dataset validation failed with errors:")
        for err in report.errors:
            print(f"  - {err}")
        return 1

    print("\n[SUCCESS] Dataset is valid!")
    print(f"Total Valid Examples: {report.total_items}")
    print("\nCategory Distribution:")
    for cat, count in sorted(report.category_counts.items()):
        print(f"  - {cat:<24}: {count:>3} examples")

    print("\nScoring Method Distribution:")
    for sm, count in sorted(report.scoring_method_counts.items()):
        print(f"  - {sm:<24}: {count:>3} examples")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  * {w}")
    print()
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """Execute evaluation run."""
    model_name = args.model.strip()

    # Distinguish real models vs offline demo simulator
    is_demo_mode = model_name.startswith("mock")
    if is_demo_mode:
        print("\n" + "!" * 70)
        print("⚠️ [NOTICE] Executing in DEMO / SIMULATED DATA mode.")
        print("Results will be flagged as simulated data and excluded from real comparisons.")
        print("!" * 70 + "\n")
    elif not ModelRegistry.is_approved_model(model_name):
        print(
            f"\n[FATAL ERROR] Model '{model_name}' is not in the approved TinyBench catalogue.\n"
            f"Approved models:\n"
            f"  1. SmolLM2 1.7B Instruct (smollm2-1.7b)\n"
            f"  2. Llama 3.2 1B Instruct (llama-3.2-1b)\n"
            f"  3. Gemma 4 E2B (gemma4-e2b)\n",
            file=sys.stderr,
        )
        return 1
    else:
        # Check whether model is installed
        adapter = ModelRegistry.get_adapter(model_name)
        canonical_id = adapter.model_id
        if not is_model_installed(canonical_id):
            print(
                f"\n[ERROR] Model '{canonical_id}' is not installed locally.\n"
                f"Please download it first using:\n"
                f"  tinybench download --model {canonical_id}\n",
                file=sys.stderr,
            )
            return 1

    cat_list = [c.strip() for c in args.categories.split(",")] if args.categories else None

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    bench_cfg = BenchmarkConfig(
        dataset_path=args.dataset,
        model_name=model_name,
        categories=cat_list,
        limit=args.limit,
        generation_config=gen_cfg,
        output_dir=args.output_dir,
    )

    print(f"\nStarting TinyBench Evaluation Run")
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    if cat_list:
        print(f"Categories: {cat_list}")
    print("-" * 60)

    runner = BenchmarkRunner(bench_cfg)

    def on_progress(curr: int, total: int, res: Any) -> None:
        if not args.quiet:
            status_sym = "[OK]" if res.is_correct else "[FAIL]"
            if res.error:
                status_sym = "[ERR]"
            print(f"[{curr:>2}/{total:>2}] {status_sym} {res.task_id:<12} ({res.category}): {res.latency_seconds:.3f}s")

    try:
        run_res = runner.run(progress_callback=on_progress)
    except Exception as exc:
        print(f"\n[FATAL ERROR] Run aborted: {exc}", file=sys.stderr)
        return 1

    # Save to storage (JSON + SQLite)
    db = RunDatabase(args.output_dir)
    json_path = db.save_run(run_res)

    print("\n" + format_markdown_summary(run_res))
    print(f"\nSaved detailed run artifact to: {json_path}")
    print(f"Saved run index to: {db.db_path}\n")
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Compare multiple runs or execute side-by-side benchmark."""
    db = RunDatabase(args.output_dir)
    results: list[RunResult] = []

    if args.runs:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        for rid in run_ids:
            run = db.get_run(rid)
            if not run:
                print(f"[ERROR] Run ID '{rid}' not found in {args.output_dir}.", file=sys.stderr)
                return 1
            results.append(run)

    elif args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"\nEvaluating models for comparison: {model_names}")
        for m in model_names:
            print(f"\nRunning benchmark for: {m} ...")
            cfg = BenchmarkConfig(
                dataset_path=args.dataset,
                model_name=m,
                limit=args.limit,
                output_dir=args.output_dir,
            )
            runner = BenchmarkRunner(cfg)
            res = runner.run()
            db.save_run(res)
            results.append(res)
    else:
        print("[ERROR] Please provide either --runs or --models to compare.", file=sys.stderr)
        return 1

    table = generate_comparison_table(results)
    print("\n" + "=" * 125)
    print("Model Comparison Summary")
    print("=" * 125)
    header = "{:<24} {:<18} {:<8} {:<12} {:<12} {:<12} {:<14} {:<12}"
    print(header.format("Model", "Data Mode", "Total", "Accuracy", "Exact Match", "Compliance", "Median Latency", "Peak RAM"))
    print("-" * 125)
    for r in table:
        print(header.format(
            r["Model Name"][:23],
            r["Data Mode"][:17],
            str(r["Total"]),
            f"{r['Accuracy (%)']:.1f}%",
            f"{r['Exact Match (%)']:.1f}%",
            f"{r['Compliance']:.2f}",
            f"{r['Median Latency (s)']:.3f} s",
            f"{r['Peak RAM (MB)']:.1f} MB",
        ))
    print("=" * 125 + "\n")
    return 0


def handle_export(args: argparse.Namespace) -> int:
    """Export run to CSV or JSON."""
    db = RunDatabase(args.output_dir)
    run_res = db.get_run(args.run_id)
    if not run_res:
        print(f"[ERROR] Run '{args.run_id}' not found.", file=sys.stderr)
        return 1

    out_file = args.output or f"{args.output_dir}/{args.run_id}.{args.format}"
    if args.format == "csv":
        path = export_run_csv(run_res, out_file, detail_level=args.detail_level)
    else:
        path = export_run_json(run_res, out_file)

    print(f"\nExported run '{args.run_id}' to: {path}\n")
    return 0


def handle_history(args: argparse.Namespace) -> int:
    """List historical evaluation runs."""
    db = RunDatabase(args.output_dir)
    runs = db.list_runs(limit=args.limit)
    if not runs:
        print(f"\nNo previous evaluation runs found in '{args.output_dir}'.")
        print("Run an evaluation using: tinybench evaluate --model smollm2-1.7b\n")
        return 0

    print("\n" + "=" * 120)
    print(f"Evaluation Run History (Last {len(runs)} runs in {args.output_dir})")
    print("=" * 120)
    fmt = "{:<28} {:<24} {:<16} {:<16} {:<10} {:<10} {:<12}"
    print(fmt.format("Run ID", "Model", "Data Mode", "Dataset", "Examples", "Accuracy", "Median Lat"))
    print("-" * 120)
    for r in runs:
        mode_str = "🧪 Demo" if r.get("is_demo") else "🧠 Verified"
        print(fmt.format(
            r["run_id"][:27],
            r["model_name"][:23],
            mode_str,
            r["dataset_name"][:15],
            f"{r['total_examples']}",
            f"{r['accuracy']:.1f}%",
            f"{r['median_latency']:.3f} s",
        ))
    print("=" * 120 + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "list-models":
        return handle_list_models()
    elif args.subcommand == "download":
        return handle_download(args)
    elif args.subcommand == "validate-dataset":
        return handle_validate_dataset(args.dataset_path)
    elif args.subcommand == "evaluate":
        return handle_evaluate(args)
    elif args.subcommand == "compare":
        return handle_compare(args)
    elif args.subcommand == "export":
        return handle_export(args)
    elif args.subcommand == "history":
        return handle_history(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
