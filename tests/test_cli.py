"""Unit tests for the TinyBench command-line interface."""

from pathlib import Path
import pytest

from tinybench.cli import create_parser, main


def test_cli_parser_defaults():
    parser = create_parser()

    # Test list-models
    args = parser.parse_args(["list-models"])
    assert args.subcommand == "list-models"

    # Test validate-dataset
    args = parser.parse_args(["validate-dataset", "datasets/basic_v1.jsonl"])
    assert args.subcommand == "validate-dataset"
    assert args.dataset_path == "datasets/basic_v1.jsonl"

    # Test evaluate
    args = parser.parse_args(["evaluate", "--model", "mock", "--limit", "5"])
    assert args.subcommand == "evaluate"
    assert args.model == "mock"
    assert args.limit == 5

    # Test compare
    args = parser.parse_args(["compare", "--models", "mock,mock-noisy"])
    assert args.subcommand == "compare"
    assert args.models == "mock,mock-noisy"

    # Test export
    args = parser.parse_args(["export", "--run-id", "test_id", "--format", "csv"])
    assert args.subcommand == "export"
    assert args.run_id == "test_id"
    assert args.format == "csv"


def test_cli_execution_list_models(capsys):
    ret = main(["list-models"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "TinyBench Available Models" in captured.out
    assert "mock" in captured.out


def test_cli_execution_validate(capsys):
    ret = main(["validate-dataset", "datasets/basic_v1.jsonl"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "[SUCCESS] Dataset is valid!" in captured.out


def test_cli_execution_evaluate(tmp_path: Path, sample_jsonl_file: Path, capsys):
    ret = main([
        "evaluate",
        "--model", "mock",
        "--dataset", str(sample_jsonl_file),
        "--limit", "2",
        "--output-dir", str(tmp_path / "runs"),
        "--quiet",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "TinyBench Evaluation Summary" in captured.out


def test_cli_execution_history(tmp_path: Path, capsys):
    # Empty history
    ret = main(["history", "--output-dir", str(tmp_path / "runs")])
    assert ret == 0
    captured = capsys.readouterr()
    assert "No previous evaluation runs" in captured.out
