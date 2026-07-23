# syntax=docker/dockerfile:1

# Multi-architecture Python image (supports linux/amd64 and linux/arm64).
# Build for your platform:  docker compose build
# Cross-build both arches:  docker buildx build --platform linux/amd64,linux/arm64 -t smile-classifier .

FROM python:3.10-slim

ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libs for Pillow and psycopg2-binary wheels; build-essential only when needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# All Python deps (numpy, pillow, scikit-learn, psycopg2-binary) ship pre-built
# wheels for both x86_64 and aarch64 — no PyTorch or native compile step required.
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p models_store app/static/uploads
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
