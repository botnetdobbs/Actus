# Makefile guard
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
	echo "ERROR: venv not activated. Run: source .venv/bin/activate"; \
	exit 1; \
	fi

run: check-venv
	uvicorn app.main:app --reload

test: check-venv
	python -m pytest tests/ -v

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-up:
	mkdir -p data config/agents
	docker compose up

docker-up-d:
	mkdir -p data config/agents
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart actus