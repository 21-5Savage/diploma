FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cron curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY pipeline/requirements.txt /app/pipeline/requirements.txt
RUN pip install --no-cache-dir -r pipeline/requirements.txt

# Copy source code
COPY pipeline/  /app/pipeline/
COPY src/       /app/src/
COPY dataset/tickers.csv /app/dataset/tickers.csv

# Environment defaults (can be overridden via docker-compose / -e flags)
ENV PIPELINE_DB=/app/pipeline/db/pipeline.db
ENV MODELS_DIR=/app/pipeline/models
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create writable directories
RUN mkdir -p /app/pipeline/db /app/pipeline/models /app/pipeline/results /app/logs

# Copy cron schedule
COPY pipeline/cron/pipeline.cron /etc/cron.d/pipeline
RUN chmod 0644 /etc/cron.d/pipeline && crontab /etc/cron.d/pipeline

# Entrypoint: run pipeline immediately (useful for testing) then keep cron running
COPY pipeline/entrypoint.sh /app/pipeline/entrypoint.sh
RUN chmod +x /app/pipeline/entrypoint.sh

ENTRYPOINT ["/app/pipeline/entrypoint.sh"]
