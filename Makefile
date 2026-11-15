.PHONY: sync lint typecheck test ui demo serve

sync:
	uv sync --python 3.13 --all-extras

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest --cov=veritas_rl --cov-report=term-missing --cov-fail-under=95

ui:
	cd ui && npm ci && npm run build

demo:
	uv run veritas demo

serve:
	uv run veritas serve --host 127.0.0.1 --port 8040
