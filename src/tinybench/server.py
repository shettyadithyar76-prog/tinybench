"""TinyBench High-Performance Local Backend Server.

Provides a lightweight, asynchronous Starlette REST and Server-Sent Events (SSE) API
for the Vercel frontend (tinybench.vercel.app) to directly control the local laptop engine,
execute benchmarks on CPU, stream live hardware & inference metrics, and query run history.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ensure src/ and repo root are on sys.path
CURRENT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import psutil
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tinybench.benchmarks.loader import load_dataset
from tinybench.benchmarks.schema import TaskCategory
from tinybench.config import BenchmarkConfig, GenerationConfig, SystemInfo
from tinybench.models.downloader import is_model_installed
from tinybench.models.ollama_backend import check_ollama_daemon, get_ollama_base_url
from tinybench.models.registry import APPROVED_MODEL_IDS, ModelRegistry
from tinybench.runner import BenchmarkRunner, RunResult, TaskResult
from tinybench.storage import RunDatabase

logger = logging.getLogger("tinybench.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RUNS_DIR = CURRENT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
db = RunDatabase(RUNS_DIR)

# Global active execution state
execution_lock = threading.Lock()
active_execution: dict[str, Any] = {
    "status": "idle",  # idle, running, completed, error, cancelled
    "run_id": None,
    "model_id": None,
    "progress": 0,
    "total": 0,
    "current_task": None,
    "latest_result": None,
    "recent_results": [],
    "error": None,
    "started_at": None,
    "completed_result": None,
    "should_cancel": False,
}

event_subscribers: list[asyncio.Queue] = []
main_loop: asyncio.AbstractEventLoop | None = None


def get_system_hardware() -> dict[str, Any]:
    """Retrieve detailed laptop hardware telemetry."""
    mem = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    cpu_name = platform.processor() or "CPU"
    try:
        uname_proc = platform.uname().processor
        if uname_proc and ("Intel" in uname_proc or "AMD" in uname_proc):
            cpu_name = uname_proc
    except Exception:
        pass

    return {
        "platform": sys.platform,
        "os_version": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "cpu_name": cpu_name,
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_freq_mhz": round(cpu_freq.current, 1) if (cpu_freq and cpu_freq.current) else None,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "used_ram_gb": round(mem.used / (1024 ** 3), 2),
        "ram_percent": mem.percent,
        "python_version": sys.version.split()[0],
    }


def broadcast_event(event_type: str, data: dict[str, Any]) -> None:
    """Send SSE payload to all connected frontend clients thread-safely."""
    global main_loop
    payload = json.dumps({"event": event_type, "timestamp": time.time(), "data": data})
    for queue in list(event_subscribers):
        try:
            if main_loop and main_loop.is_running():
                main_loop.call_soon_threadsafe(queue.put_nowait, payload)
            else:
                queue.put_nowait(payload)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")


async def api_health(request: Request) -> JSONResponse:
    """Healthcheck and hardware status."""
    global main_loop
    main_loop = asyncio.get_running_loop()

    try:
        ollama_ok, ollama_version = check_ollama_daemon()
    except Exception as e:
        ollama_ok, ollama_version = False, str(e)

    try:
        hw = get_system_hardware()
    except Exception as e:
        hw = {"error": str(e)}

    return JSONResponse({
        "status": "online",
        "service": "tinybench-laptop-backend",
        "version": "0.3.0",
        "system": hw,
        "ollama": {
            "available": ollama_ok,
            "version": ollama_version,
            "base_url": get_ollama_base_url(),
        },
        "active_execution_status": active_execution["status"],
    })


async def api_models(request: Request) -> JSONResponse:
    """List approved models and their local availability on this laptop."""
    models_out = []
    try:
        ollama_ok, _ = check_ollama_daemon()
    except Exception:
        ollama_ok = False

    try:
        for m in ModelRegistry.list_available_models():
            try:
                is_inst = is_model_installed(m.model_id)
            except Exception:
                is_inst = False

            is_ready = False
            if m.supported_runtime == "ollama":
                is_ready = ollama_ok and is_inst
            elif m.supported_runtime == "transformers_cpu":
                is_ready = is_inst
            elif "mock" in m.model_id:
                is_ready = True

            models_out.append({
                "id": m.model_id,
                "name": m.model_name,
                "developer": m.developer,
                "family": m.family,
                "parameters": m.parameters,
                "quantization": m.quantization,
                "download_size": m.download_size,
                "runtime": m.supported_runtime,
                "description": m.description,
                "is_installed": is_inst,
                "is_ready": is_ready,
                "minimum_ram_mb": m.minimum_ram_mb,
                "recommended_ram_mb": m.recommended_ram_mb,
                "status": m.status,
                "status_reason": m.status_reason,
            })
    except Exception as e:
        logger.error(f"Error fetching models: {e}", exc_info=True)

    # Built-in mock simulator
    models_out.append({
        "id": "mock",
        "name": "TinyBench Mock Simulator",
        "developer": "TinyBench Core",
        "family": "Simulator",
        "parameters": "0M (Simulated)",
        "quantization": "Deterministic",
        "download_size": "0 MB",
        "runtime": "Deterministic CPU",
        "description": "Instantaneous zero-GPU mock evaluation backend for UI benchmarking and testing.",
        "is_installed": True,
        "is_ready": True,
        "minimum_ram_mb": 50,
        "recommended_ram_mb": 100,
        "status": "available",
        "status_reason": "Built-in deterministic test simulator ready.",
    })

    return JSONResponse({"models": models_out, "count": len(models_out)})


async def api_runs(request: Request) -> JSONResponse:
    """Retrieve persisted evaluation runs from runs.db."""
    try:
        limit = int(request.query_params.get("limit", 50))
        all_runs = db.list_runs(limit=limit)
        return JSONResponse({"runs": all_runs, "count": len(all_runs)})
    except Exception as e:
        logger.error(f"Error fetching runs: {e}", exc_info=True)
        return JSONResponse({"error": str(e), "runs": []}, status_code=500)


async def api_run_detail(request: Request) -> JSONResponse:
    """Retrieve full metrics and tasks for a specific run."""
    run_id = request.path_params["run_id"]
    try:
        run_res = db.get_run(run_id)
        if not run_res:
            return JSONResponse({"error": f"Run '{run_id}' not found."}, status_code=404)
        return JSONResponse(run_res.model_dump())
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def _execute_benchmark_thread(config: BenchmarkConfig, selected_categories: list[str] | None):
    """Worker thread running TinyBench benchmark runner."""
    global active_execution
    try:
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            dataset_path = CURRENT_DIR / "data" / "golden_benchmark.jsonl"
            if not dataset_path.exists():
                dataset_path = CURRENT_DIR / "tests" / "fixtures" / "sample_benchmark.jsonl"
            config.dataset_path = str(dataset_path)

        categories = None
        if selected_categories:
            categories = [TaskCategory(c) for c in selected_categories if c in [tc.value for tc in TaskCategory]]

        dataset = load_dataset(dataset_path, categories=categories, limit=config.limit)
        total_items = len(dataset.items)

        active_execution["total"] = total_items
        broadcast_event("started", {
            "run_id": active_execution["run_id"],
            "model_id": config.model_name,
            "total_tasks": total_items,
        })

        def progress_cb(current: int, total: int, task_res: TaskResult):
            with execution_lock:
                if active_execution.get("should_cancel"):
                    raise KeyboardInterrupt("Benchmark cancelled by user.")
                active_execution["progress"] = current
                active_execution["latest_result"] = task_res.model_dump()
                active_execution["recent_results"].append(task_res.model_dump())
                active_execution["current_task"] = {
                    "task_id": task_res.task_id,
                    "category": task_res.category,
                    "is_correct": task_res.is_correct,
                    "exact_match": task_res.exact_match,
                    "latency_seconds": task_res.latency_seconds,
                    "generated_tokens": task_res.generated_tokens,
                }

            broadcast_event("progress", {
                "current": current,
                "total": total,
                "task": task_res.model_dump(),
                "system": get_system_hardware(),
            })

        runner = BenchmarkRunner(config)
        run_result = runner.run(progress_callback=progress_cb)

        # Save to database
        db.save_run(run_result)

        with execution_lock:
            active_execution["status"] = "completed"
            active_execution["completed_result"] = run_result.model_dump()

        broadcast_event("completed", {
            "run_id": run_result.run_id,
            "accuracy": run_result.accuracy,
            "exact_match_rate": run_result.exact_match_rate,
            "mean_latency_seconds": run_result.latency_summary.mean_latency_seconds,
            "result": run_result.model_dump(),
        })

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Benchmark execution cancelled.")
        with execution_lock:
            active_execution["status"] = "cancelled"
            active_execution["error"] = "Execution cancelled."
        broadcast_event("cancelled", {"run_id": active_execution.get("run_id")})
    except Exception as exc:
        logger.error(f"Benchmark execution failed: {exc}", exc_info=True)
        with execution_lock:
            active_execution["status"] = "error"
            active_execution["error"] = str(exc)
        broadcast_event("error", {"error": str(exc), "run_id": active_execution.get("run_id")})


async def api_benchmark_start(request: Request) -> JSONResponse:
    """Trigger a new benchmark execution on this laptop."""
    global active_execution, main_loop
    main_loop = asyncio.get_running_loop()

    try:
        body = await request.json()
    except Exception:
        body = {}

    model_id = body.get("model_id", "mock")
    limit = int(body.get("limit", 5))
    temperature = float(body.get("temperature", 0.0))
    max_tokens = int(body.get("max_tokens", 384))
    seed = int(body.get("seed", 42))
    categories = body.get("categories", None)

    with execution_lock:
        if active_execution["status"] == "running":
            return JSONResponse({"error": "A benchmark is already actively running on this laptop."}, status_code=409)

        run_id = f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{model_id.replace(':', '_').replace('/', '_')}"
        active_execution = {
            "status": "running",
            "run_id": run_id,
            "model_id": model_id,
            "progress": 0,
            "total": limit,
            "current_task": None,
            "latest_result": None,
            "recent_results": [],
            "error": None,
            "started_at": time.time(),
            "completed_result": None,
            "should_cancel": False,
        }

    dataset_path = CURRENT_DIR / "data" / "golden_benchmark.jsonl"
    if not dataset_path.exists():
        dataset_path = CURRENT_DIR / "tests" / "fixtures" / "sample_benchmark.jsonl"

    config = BenchmarkConfig(
        model_name=model_id,
        dataset_path=str(dataset_path),
        categories=categories,
        limit=limit,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens,
            seed=seed,
        ),
    )

    thread = threading.Thread(
        target=_execute_benchmark_thread,
        args=(config, categories),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "status": "started",
        "run_id": run_id,
        "model_id": model_id,
        "limit": limit,
    })


async def api_benchmark_status(request: Request) -> JSONResponse:
    """Get active benchmark execution state."""
    with execution_lock:
        data = dict(active_execution)
        if data.get("completed_result"):
            data["summary"] = {
                "accuracy": data["completed_result"].get("accuracy"),
                "exact_match_rate": data["completed_result"].get("exact_match_rate"),
                "total_examples": data["completed_result"].get("total_examples"),
            }
    data["system"] = get_system_hardware()
    return JSONResponse(data)


async def api_benchmark_cancel(request: Request) -> JSONResponse:
    """Abort currently running benchmark."""
    global active_execution
    with execution_lock:
        if active_execution["status"] == "running":
            active_execution["should_cancel"] = True
            active_execution["status"] = "cancelling"
            return JSONResponse({"status": "cancelling"})
    return JSONResponse({"status": "not_running"})


async def api_benchmark_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events (SSE) stream for real-time benchmark telemetry."""
    global main_loop
    main_loop = asyncio.get_running_loop()

    client_queue: asyncio.Queue = asyncio.Queue()
    event_subscribers.append(client_queue)

    async def event_generator():
        # Send initial status
        try:
            init_payload = json.dumps({
                "event": "connected",
                "timestamp": time.time(),
                "data": {
                    "active_execution": active_execution["status"],
                    "system": get_system_hardware(),
                }
            })
            yield f"data: {init_payload}\n\n"
        except Exception:
            return

        try:
            while True:
                # If disconnected by client, break cleanly
                try:
                    if await request.is_disconnected():
                        break
                except Exception:
                    break

                try:
                    payload = await asyncio.wait_for(client_queue.get(), timeout=2.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat / keepalive
                    try:
                        heartbeat = json.dumps({
                            "event": "ping",
                            "timestamp": time.time(),
                            "system": {
                                "cpu_percent": psutil.cpu_percent(interval=None),
                                "ram_percent": psutil.virtual_memory().percent,
                            }
                        })
                        yield f"data: {heartbeat}\n\n"
                    except Exception:
                        break
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if client_queue in event_subscribers:
                event_subscribers.remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Assemble routes with explicit OPTIONS support for cross-origin preflight
routes = [
    Route("/api/health", api_health, methods=["GET", "OPTIONS"]),
    Route("/api/system", api_health, methods=["GET", "OPTIONS"]),
    Route("/api/models", api_models, methods=["GET", "OPTIONS"]),
    Route("/api/runs", api_runs, methods=["GET", "OPTIONS"]),
    Route("/api/runs/{run_id}", api_run_detail, methods=["GET", "OPTIONS"]),
    Route("/api/benchmark/start", api_benchmark_start, methods=["POST", "OPTIONS"]),
    Route("/api/benchmark/status", api_benchmark_status, methods=["GET", "OPTIONS"]),
    Route("/api/benchmark/cancel", api_benchmark_cancel, methods=["POST", "OPTIONS"]),
    Route("/api/benchmark/stream", api_benchmark_stream, methods=["GET", "OPTIONS"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
]

app = Starlette(debug=True, routes=routes, middleware=middleware)


def main():
    import uvicorn
    port = int(os.environ.get("TINYBENCH_PORT", 8000))
    host = os.environ.get("TINYBENCH_HOST", "127.0.0.1")
    logger.info(f"Starting TinyBench local API server on http://{host}:{port}")
    uvicorn.run("tinybench.server:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
