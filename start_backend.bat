@echo off
set PYTHONPATH=src;%PYTHONPATH%
echo ======================================================================
echo   Starting TinyBench Laptop Backend with Cloudflare Tunnel
echo ======================================================================
python scripts/serve.py %*
pause
