FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY api/pyproject.toml api/pyproject.toml
COPY app/pyproject.toml app/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace

COPY src ./src
COPY api ./api
COPY app ./app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen


# entry point necessary, otherwise uv will be used as entrypoint -> janky error messages
ENTRYPOINT []


FROM base AS notebook

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group notebook


FROM notebook AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group notebook --group dev
    
RUN --mount=type=cache,target=/root/.cache/uv \
    uv tool install rust-just
