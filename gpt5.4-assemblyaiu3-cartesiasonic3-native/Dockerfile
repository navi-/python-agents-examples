FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy application code
COPY utils.py ./
COPY inbound/ ./inbound/
COPY outbound/ ./outbound/

EXPOSE 8000

# Default: run inbound server
CMD ["uv", "run", "python", "-m", "inbound.server"]
