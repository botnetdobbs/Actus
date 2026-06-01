FROM python:3.13-slim

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies before copying app code. Better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Download the Presidio spaCy model at build time so startup is fast
# Without this, the first request pays the 2-3s model load cost
RUN uv run python -m spacy download en_core_web_lg

# Copy application code and agent config
COPY app/ app/
COPY config/ config/

# Non-root user with explicit UID for host volume permission matching
RUN useradd -m -u 1000 actus \
    && mkdir -p /app/data \
    && chown -R actus:actus /app
USER actus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# Single worker required: APScheduler runs in the lifespan of each worker process.
# Multiple workers would start duplicate schedulers and fire every job N times.
# To scale horizontally, move the scheduler to a dedicated container first.
CMD ["uv", "run", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
