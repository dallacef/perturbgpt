# CPU-only inference image for the PerturbGPT API.
#
# The image contains code + Python dependencies only.  Model checkpoints,
# embeddings and FAISS indices are large (hundreds of MB) and are mounted at
# runtime instead — see docker-compose.yml.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is only needed for the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only PyTorch (the default wheel bundles CUDA and is ~2 GB larger).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[api,retrieval]"

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/runs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "perturbgpt.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
