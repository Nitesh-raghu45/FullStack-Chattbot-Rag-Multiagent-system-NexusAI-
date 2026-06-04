# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — dependency builder
# Install packages in an isolated layer so they get Docker cache hits
# when only source code changes (not requirements.txt)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed to compile some Python packages
# (faiss-cpu, sentence-transformers need gcc + libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first — Docker caches this layer
# and won't re-run pip install unless requirements.txt changes
COPY requirements.txt .

# ── IMPORTANT: install CPU-only torch FIRST ────────────────────────────────
# sentence-transformers depends on torch. Without this step, pip resolves
# torch from PyPI and pulls the full CUDA build (~1.5 GB of GPU libraries
# that are completely useless on Render/CPU-only servers).
# Installing the CPU wheel first satisfies the torch dependency cheaply.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Pre-download the HuggingFace embedding model into the image ─────────────
# Baking the model into the image means:
#   - No HF network download at container startup (faster cold starts)
#   - No "unauthenticated requests" rate-limit warnings
#   - No OOM from downloading + loading simultaneously at runtime
ENV PYTHONPATH=/install/lib/python3.10/site-packages
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — final runtime image (slim, no build tools)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS runtime

# Non-root user for security — never run prod containers as root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy pre-downloaded HuggingFace model cache from builder
# (avoids downloading at runtime — no HF network calls at startup)
COPY --from=builder /app/.cache /app/.cache

# Copy application source code
COPY --chown=appuser:appuser . .

# Create directories the app writes to at runtime and fix ownership of /app
# WORKDIR creates /app as root — appuser needs write access to create chatbot.db
RUN mkdir -p logs vectorstore data/raw data/processed && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ── Environment defaults (overridden by docker-compose / .env) ─────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Expose the port uvicorn listens on
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────
# Docker will restart the container if this fails 3 times in a row
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" \
    || exit 1

# ── Start command ──────────────────────────────────────────────────────────
# Render injects $PORT dynamically — we must listen on that port.
# Shell form used (not JSON array) so $PORT is expanded at runtime.
CMD exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info