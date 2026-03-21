# ── Serving container only ────────────────────────────────────────
# Packages the FastAPI inference service, NOT training code.
# Build: docker build -t house-price-api:latest .
# Run:   docker run -p 8000:8000 --env-file .env house-price-api:latest
# ─────────────────────────────────────────────────────────────────

FROM continuumio/miniconda3:latest

WORKDIR /app

# Forces Python to stream logs instantly to Render dashboard
# so we can debug live rather than holding them in a buffer
ENV PYTHONUNBUFFERED=1

# Prevents writing .pyc files — reduces image size
ENV PYTHONDONTWRITEBYTECODE=1

# Ensures absolute imports like src.api resolve correctly
ENV PYTHONPATH=/app

# Copy the pre-calculated lock file first to leverage Docker layer caching.
# Generated locally with: conda-lock -p linux-64 -f environment.yml
COPY conda-lock.yml .

# 1. Install conda-lock into the base environment
# 2. Install exact frozen dependencies from the lock file
# 3. Install curl for the HEALTHCHECK below
# 4. Clean all caches to keep the image lean
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops_project conda-lock.yml && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

# Use the mlops conda environment as the default Python
ENV PATH=/opt/conda/envs/mlops_project/bin:$PATH

# Copy application code — .dockerignore excludes noise
COPY . .

EXPOSE 8000

# Render checks this before routing real traffic to the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Use sh -c so ${PORT} is expanded at runtime (Render sets PORT dynamically)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]