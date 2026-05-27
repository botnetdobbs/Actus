# Makefile guard
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
	echo "ERROR: venv not activated. Run: source .venv/bin/activate"; \
	exit 1; \
	fi

run: check-venv
	uvicorn app.main:app --reload