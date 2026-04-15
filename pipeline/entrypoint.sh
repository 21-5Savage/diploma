#!/bin/bash
set -e

echo "=== Stock Prediction Pipeline Container Starting ==="
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Run an initial pipeline pass on startup (useful for testing the container)
if [ "${RUN_ON_START:-true}" = "true" ]; then
    echo "Running initial pipeline pass..."
    cd /app
    python -m pipeline.run_pipeline --lookback "${LOOKBACK:-90}" || true
fi

echo "Starting cron daemon..."
service cron start

echo "Container ready. Cron schedule active."
# Keep container alive
tail -f /app/logs/pipeline.log 2>/dev/null || tail -f /dev/null
