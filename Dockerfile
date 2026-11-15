FROM node:22-slim AS ui-builder
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends clang && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY csrc ./csrc
COPY src ./src
COPY benchmarks ./benchmarks
RUN uv sync --frozen --no-dev
COPY --from=ui-builder /ui/dist ./ui/dist

EXPOSE 8040
CMD ["uv", "run", "--no-dev", "veritas", "serve", "--host", "0.0.0.0", "--port", "8040"]
