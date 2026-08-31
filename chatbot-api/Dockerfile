# ============================================================
# Stage 1: Builder — install dependencies in isolated venv
# ============================================================
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=2.0,<3.0"

WORKDIR /build

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-interaction --no-ansi

# ============================================================
# Stage 2: Runtime — lean image (dev/prod same entrypoint)
# ============================================================
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv

COPY main.py run_chatbot.py ./
COPY app/ ./app/
COPY config/ ./config/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# HOST / PORT / RELOAD come from root .env via compose env_file.
# No listen port is hardcoded here — run_chatbot.py reads PORT from env.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/docs')"

CMD ["python", "run_chatbot.py"]
