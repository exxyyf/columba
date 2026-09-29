#!/usr/bin/env bash
# Серверный скрипт деплоя. Запускается GitHub Actions после пуша в main
# (см. .github/workflows/deploy.yml) или вручную: bash /opt/columba/deploy/deploy.sh
# Идемпотентен: повторный запуск безопасен.
set -euo pipefail

REPO_DIR=/opt/columba
cd "$REPO_DIR"

git fetch origin main
git reset --hard origin/main

# venv хоста — его использует bridge-скрипт viewer'а (interface/bridge/describe.py)
uv sync --frozen --no-dev

# Веса: download докачивает только отсутствующие файлы, verify сверяет sha256
# с weights_registry.json и падает при расхождении.
uv run python -m columba.weights download
uv run python -m columba.weights verify

# API-сервис инференса в Docker (CPU-образ), порт 8000 только на loopback —
# наружу смотрит Caddy с basic auth.
docker compose -f deploy/docker-compose.yml up -d --build

# Viewer: сборка TypeScript и systemd-юнит на порту 8425 (тоже за Caddy).
cd interface
npm ci
npm run build
cd "$REPO_DIR"
install -m 644 deploy/columba-viewer.service /etc/systemd/system/columba-viewer.service
systemctl daemon-reload
systemctl enable columba-viewer >/dev/null 2>&1
systemctl restart columba-viewer

echo "Deploy OK: $(git rev-parse --short HEAD)"
