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

    print("=" * 70)
    print("  🚀 TinyBench Local Inference & Hardware API Server")
    print(f"  📍 Host: http://{args.host}:{args.port}")
    print("=" * 70)

    tunnel_proc = None
    tunnel_url = None

    if not args.no_tunnel:
        cf_bin = find_cloudflared()
        if cf_bin:
            print("\n  🌐 Launching Cloudflare Tunnel (cloudflared)...")
            tunnel_cmd = [cf_bin, "tunnel", "--url", f"http://{args.host}:{args.port}"]
            try:
                # cloudflared prints the tunnel URL to stderr
                tunnel_proc = subprocess.Popen(
                    tunnel_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                # Read output to capture trycloudflare.com URL
                start_time = time.time()
                while time.time() - start_time < 12.0:
                    line = tunnel_proc.stderr.readline()
                    if line:
                        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                        if match:
                            tunnel_url = match.group(0)
                            break
                    time.sleep(0.1)

                if tunnel_url:
                    print("\n  " + "—" * 66)
                    print(f"  ⚡ CLOUDFLARE TUNNEL LIVE: \033[92m\033[1m{tunnel_url}\033[0m")
                    print("  —" * 66)
                    print("  👉 Enter this URL into your Vercel frontend (tinybench.vercel.app)")
                    print("     under 'Connect Laptop Backend' to run real benchmarks!")
                    print("  —" * 66 + "\n")
                else:
                    print("  ⚠️  Tunnel started, but URL regex match timed out. Check cloudflared output.")
            except Exception as e:
                print(f"  ⚠️  Could not start cloudflared: {e}")
        else:
            print("  ℹ️  'cloudflared' not found. Running in local mode only.")
            print(f"     Direct local URL: http://{args.host}:{args.port}")

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
