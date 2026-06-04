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


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — final runtime image (slim, no build tools)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.10-slim AS runtime

# Non-root user for security — never run prod containers as root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY --chown=appuser:appuser . .

# Create directories the app writes to at runtime
# (logs/ and vectorstore/ and data/ must exist inside container)
RUN mkdir -p logs vectorstore data/raw data/processed && \
    chown -R appuser:appuser logs vectorstore data

# Switch to non-root user
USER appuser

# ── Environment defaults (overridden by docker-compose / .env) ─────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

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