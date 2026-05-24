# ── AeroLex Dockerfile ────────────────────────────────────────────────────
#
# WHAT:
#   Containerizes the AeroLex FastAPI application.
#   Multi-stage build for smaller final image size.
#
# WHY Multi-stage?
#   Stage 1 (builder): Install all dependencies — includes build tools
#   Stage 2 (runtime): Copy only what's needed — no build tools
#   Result: ~40% smaller image, faster deployment
#
# BEST PRACTICES USED:
#   - Non-root user (security)
#   - .dockerignore (smaller build context)
#   - Layer caching (requirements before code)
#   - Health check (Docker knows if app is healthy)
#
# Official Docs:
#   https://docs.docker.com/build/building/multi-stage/
# ─────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer cache optimization
# If requirements.txt unchanged, this layer is cached
COPY requirements.txt .

# Install Python dependencies into /install
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL maintainer="Preeti <United Airlines>"
LABEL project="AeroLex"
LABEL version="1.0.0"
LABEL description="Aviation Regulatory Compliance Assistant"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    PORT=8000

# Create non-root user — security best practice
# Never run containers as root in production
RUN groupadd -r aerolex && \
    useradd -r -g aerolex -d /app -s /sbin/nologin aerolex

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=aerolex:aerolex . .

# Switch to non-root user
USER aerolex

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker monitors this
# Checks every 30s, 3 failures = unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" \
    || exit 1

# Start FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]