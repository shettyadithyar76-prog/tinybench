"""Launch script for TinyBench Local Server with automatic Cloudflare Tunnel.

Usage:
    python scripts/serve.py
    python scripts/serve.py --no-tunnel
    python scripts/serve.py --port 8000
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add src/ to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def find_cloudflared() -> str | None:
    """Check if cloudflared binary is accessible."""
    # Check PATH
    try:
        res = subprocess.run(["cloudflared", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return "cloudflared"
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Start TinyBench backend server with optional Cloudflare Tunnel")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-tunnel", action="store_true", help="Do not start cloudflared tunnel")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("  🚀 TinyBench Local Inference & Hardware API Server", flush=True)
    print(f"  📍 Host: http://{args.host}:{args.port}", flush=True)
    print("=" * 70, flush=True)

    tunnel_proc = None
    tunnel_url = None

    if not args.no_tunnel:
        cf_bin = find_cloudflared()
        if cf_bin:
            print("\n  🌐 Launching Cloudflare Tunnel (cloudflared)...", flush=True)
            tunnel_cmd = [cf_bin, "tunnel", "--url", f"http://{args.host}:{args.port}"]
            try:
                tunnel_proc = subprocess.Popen(
                    tunnel_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                # Read output to capture trycloudflare.com URL
                start_time = time.time()
                while time.time() - start_time < 20.0:
                    line = tunnel_proc.stderr.readline()
                    if line:
                        match = re.search(r"https://([a-zA-Z0-9-]+)\.trycloudflare\.com", line)
                        if match and match.group(1).lower() != "api":
                            tunnel_url = match.group(0)
                            break
                    time.sleep(0.05)

                if tunnel_url:
                    print("\n  " + "—" * 66, flush=True)
                    print(f"  ⚡ CLOUDFLARE TUNNEL LIVE: {tunnel_url}", flush=True)
                    print("  —" * 66, flush=True)
                    print("  👉 Enter this URL into your Vercel frontend (tinybench.vercel.app)", flush=True)
                    print("     under 'Connect Laptop Backend' to run real benchmarks!", flush=True)
                    print("  —" * 66 + "\n", flush=True)
                else:
                    print("  ⚠️  Tunnel started, but URL regex match timed out. Check cloudflared output.", flush=True)
            except Exception as e:
                print(f"  ⚠️  Could not start cloudflared: {e}", flush=True)
        else:
            print("  ℹ️  'cloudflared' not found. Running in local mode only.", flush=True)
            print(f"     Direct local URL: http://{args.host}:{args.port}", flush=True)

    # Launch uvicorn
    import uvicorn
    from tinybench.server import app

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if tunnel_proc:
            print("\n  🛑 Shutting down Cloudflare Tunnel...")
            tunnel_proc.terminate()
            tunnel_proc.wait()


if __name__ == "__main__":
    main()
