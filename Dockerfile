FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY api/pyproject.toml api/pyproject.toml
COPY app/pyproject.toml app/pyproject.toml
COPY common/pyproject.toml common/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace

COPY src ./src
COPY api ./api
COPY app ./app
COPY common ./common

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --frozen


# entry point necessary, otherwise uv will be used as entrypoint -> janky error messages
ENTRYPOINT []


FROM base AS train

COPY . .


FROM base AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --frozen --group dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install rust-just
