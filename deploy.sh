#!/bin/bash
set -e

REPO_DIR="/home/deployer/voice-agent-service/voice-agent-runtime"  # repo is a subdirectory of where this script lives
VENV="$REPO_DIR/.venv"
SERVICE="voice-agent-runtime"

echo "==> Pulling latest code"
cd "$REPO_DIR"
git pull

echo "==> Installing requirements"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements-pipecat.txt

echo "==> Restarting service"
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl status "$SERVICE" --no-pager

echo "==> Reloading nginx"
sudo nginx -t
sudo systemctl reload nginx

echo "==> Health check"
curl -sf http://127.0.0.1:8010/health && echo " OK" || echo " FAILED"

echo "==> Done"
