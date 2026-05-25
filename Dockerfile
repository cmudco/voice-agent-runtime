# Pipecat's native audio dependencies currently need a stable wheel set.
# Keep the runtime on Python 3.12 until the Pipecat stack supports newer versions.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-pipecat.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements-pipecat.txt

COPY runtime ./runtime

EXPOSE 8010

CMD ["uvicorn", "runtime.pipecat_app:app", "--host", "0.0.0.0", "--port", "8010"]
