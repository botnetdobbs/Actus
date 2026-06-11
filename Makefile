.PHONY: test test-cov lint typecheck migrate backfill-rag migrations docker-build docker-up docker-up-d ollama-pull docker-down docker-logs docker-restart docker-rebuild dev dev-d dev-down dev-rebuild

test:
	uv run python -m pytest tests/ -v

test-cov:
	uv run python -m pytest tests/ --cov=app --cov-report=term-missing

lint:
	uv run ruff check .

typecheck:
	uv run mypy app

migrate:
	uv run alembic upgrade head

backfill-rag:
	uv run python scripts/backfill_rag.py

migrations:
	uv run alembic revision --autogenerate -m "$(msg)"

# ── Docker (production) ───────────────────────────────────────────────────────

DC = docker compose

docker-build:
	$(DC) build

docker-up:
	mkdir -p config/agents
	$(DC) up

docker-up-d:
	mkdir -p config/agents
	$(DC) up -d

ollama-pull:
	$(DC) -f docker-compose.yml -f docker-compose.dev.yml exec ollama ollama pull mistral

docker-down:
	$(DC) down

docker-logs:
	$(DC) logs -f

docker-restart:
	$(DC) restart actus

docker-rebuild:
	$(DC) down && $(DC) up --build -d

# ── Docker (development) ──────────────────────────────────────────────────────

DEV = docker compose -f docker-compose.yml -f docker-compose.dev.yml

dev:
	mkdir -p config/agents
	$(DEV) up

dev-d:
	mkdir -p config/agents
	$(DEV) up -d

dev-down:
	$(DEV) down

dev-rebuild:
	$(DEV) down && $(DEV) up --build
