FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY utils.py utils.py
COPY inbound/ inbound/
COPY outbound/ outbound/

# Run the inbound agent worker by default
# Override with "python -m outbound.agent start" for outbound
CMD ["uv", "run", "python", "-m", "inbound.agent", "start"]
