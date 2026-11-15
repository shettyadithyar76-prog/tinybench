FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/api/health || exit 1

ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000", "--no-tunnel"]
