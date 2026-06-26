FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY .python-version pyproject.toml uv.lock ./

RUN uv sync --locked

COPY src/ ./src/

WORKDIR /app/src

RUN chmod +x entrypoint.sh

CMD ["uv", "run", "main.py"]
