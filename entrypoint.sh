#!/bin/bash
# Minimon API entrypoint
# Usage: docker compose up | docker run --rm minimonapi

set -e

# Ensure .env file exists (generated on first run)
if [ ! -f /app/env ]; then
    echo "[minimon] Generating .env with defaults — edit /app/env to customize"
fi

export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"

echo "[minimon] Starting on port 22222 ..."
cd /app
exec uv run python gen-server.py
