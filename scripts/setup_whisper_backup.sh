#!/bin/bash
# Generates the shared token, writes it to .env, and prints the worker command.
# Safe to re-run: an existing token is reused, never regenerated.
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=".env"
PORT="${WHISPER_PORT:-8011}"

[ -f "$ENV_FILE" ] || { echo "No .env found. Copy .env.example first." >&2; exit 1; }

TOKEN="$(grep -E '^VOICE_RUNTIME_WHISPER_TOKEN=' "$ENV_FILE" | cut -d= -f2- || true)"
if [ -z "$TOKEN" ]; then
  TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  cp "$ENV_FILE" "$ENV_FILE.bak-$(date +%Y%m%d%H%M%S)"
  printf '\n# Self-hosted Whisper STT backup.\nVOICE_RUNTIME_WHISPER_URL=http://127.0.0.1:%s\nVOICE_RUNTIME_WHISPER_TOKEN=%s\n' "$PORT" "$TOKEN" >> "$ENV_FILE"
  echo "Wrote VOICE_RUNTIME_WHISPER_URL and VOICE_RUNTIME_WHISPER_TOKEN to $ENV_FILE"
else
  echo "Reusing the existing token in $ENV_FILE"
fi

cat <<EOF

Start the worker in its own environment (token must be exported, not inlined,
because the worker reads it at import):

  set -a && . ./.env && set +a
  export WHISPER_MODEL=tiny.en WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 PYTHONPATH=.
  .venv-whisper/bin/python -m uvicorn runtime.whisper_worker:app \\
    --host 127.0.0.1 --port $PORT --no-access-log

Then verify:

  .venv/bin/python -m runtime.preflight
EOF
