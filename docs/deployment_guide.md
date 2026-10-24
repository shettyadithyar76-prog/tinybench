# TinyBench — Deployment & Hosting Guide

This guide details how to deploy TinyBench across various environments, including **Vercel**, **Streamlit Community Cloud**, and **Docker**.

---

## 1. Architectural Overview & Hosting Suitability

TinyBench consists of two primary operational components:

1. **Local Evaluation Engine & Interactive Studio (`app.py`)**:
   - Executes real PyTorch CPU model inference and mock simulations.
   - Reads hardware sensors (RAM, CPU physical cores) via `psutil`.
   - Writes runs to local SQLite (`runs/runs.db`) and exports JSON/CSV artifacts.
   - **Hosting Requirements**: Persistent Python daemon process with WebSocket connectivity.
   - **Recommended Platforms**: Local workstation (primary), Streamlit Community Cloud, Hugging Face Spaces, or Docker on Fly.io/Render.

2. **Web Benchmark Viewer & Static Leaderboard (`web/`)**:
   - High-performance, zero-server-overhead web interface styled with Tailwind and Aceternity design tokens.
   - Renders benchmark summaries, model catalog specifications, and links to reproducible runs.
   - **Hosting Requirements**: Pure static JAMstack hosting.
   - **Recommended Platform**: **Vercel**, GitHub Pages, or Netlify.

---

## 2. Deploying to Vercel

The `web/` directory is pre-configured for instant zero-configuration deployment to [Vercel](https://vercel.com/).

### Deployment Steps:
1. **Push your repository to GitHub** (already configured at `https://github.com/shettyadithyar76-prog/tinybench`).
2. **Log into Vercel** using your GitHub account at [https://vercel.com](https://vercel.com).
3. Click **Add New... &rarr; Project**.
4. Import the **`shettyadithyar76-prog/tinybench`** repository.
5. Vercel automatically detects [`vercel.json`](file:///C:/Users/Adithya/.gemini/antigravity/scratch/tinybench/vercel.json):
   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave default)
   - **Build Command**: None (pure static asset delivery)
   - **Output Directory**: `web`
6. Click **Deploy**. Your live dashboard will be instantly accessible at `https://tinybench-<your-username>.vercel.app`.

### Security & Privacy on Vercel:
- **No Secrets Required**: The static frontend contains zero API keys, secrets, or access tokens.
- **Private Repositories**: Vercel natively supports deploying from private GitHub repositories.
- **Custom Domains**: You can attach a custom domain with automatic SSL via the Vercel dashboard.

---

## 3. Deploying the Full Interactive Studio to Streamlit Community Cloud

To make the full interactive Streamlit dashboard (`app.py`) accessible to collaborators without needing a local terminal:

1. Visit [https://share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **New app**.
3. Select:
   - **Repository**: `shettyadithyar76-prog/tinybench`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**.
5. Streamlit Community Cloud will automatically install dependencies from `pyproject.toml` and provide a public URL.

*Note: In cloud-hosted Streamlit environments, use the built-in mock backends (`mock`, `mock-noisy`) for zero-download instant testing, or select `smollm2-135m` within the free tier's RAM limit (1 GB).*

---

## 4. Docker Container Deployment (Self-Hosted / Private Cloud)

For isolated on-premise execution or deployment to Render/Fly.io/AWS EC2:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY datasets/ datasets/
COPY app.py ./

RUN pip install --no-cache-dir -e ".[dashboard,models]"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t tinybench:latest .
docker run -p 8501:8501 tinybench:latest
```
