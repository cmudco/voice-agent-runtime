# Pipecat's native audio dependencies currently need a stable wheel set.
# Keep the runtime on Python 3.12 until the Pipecat stack supports newer versions.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-pipecat.txt ./
RUN pip install -r requirements.txt \
    && pip install --no-deps -r requirements-pipecat.txt


FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# curl: the health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kokoro downloads its model to ~/.cache/kokoro-onnx on first use; mount a volume there to keep it.
RUN groupadd --gid 1001 voiceagent \
    && useradd --uid 1001 --gid voiceagent --create-home --home-dir /home/voiceagent --shell /usr/sbin/nologin voiceagent \
    && mkdir -p /home/voiceagent/.cache/kokoro-onnx /app \
    && chown -R voiceagent:voiceagent /home/voiceagent /app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=voiceagent:voiceagent runtime ./runtime

USER voiceagent

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8010/health || exit 1

CMD ["uvicorn", "runtime.pipecat_app:app", "--host", "0.0.0.0", "--port", "8010"]
