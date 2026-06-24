#!/bin/bash
set -e

if [ ! -f /app/data/.db_initialized ]; then
  uv run python -m scripts.setup_db --overwrite
  touch /app/data/.db_initialized
fi

if [ ! -f /app/data/.db_seeded ]; then
  uv run python -m scripts.seed_db
  touch /app/data/.db_seeded
fi

exec uv run python main.py
