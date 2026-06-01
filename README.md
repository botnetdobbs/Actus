# Actus

**FastAPI-based multi-agent platform with LLM routing, RAG-ready context engineering, and operational automation.**

Actus is infrastructure for AI agents that do real work, not a chatbot wrapper. Define agents in YAML, connect them to your data, give them tools, and run them on a schedule or on demand. You own the code. You control what data the agents see and what they are allowed to do.

---

## Stack

```
Python 3.13      FastAPI + Uvicorn       SQLModel + SQLite/PostgreSQL
LiteLLM          Ollama                  Presidio (PII)
APScheduler      structlog               bcrypt + python-jose
pytest + httpx
```

---

## Quick Start

**Prerequisites (both paths):** Ollama running on your machine (`ollama serve && ollama pull mistral`).

### Virtual environment

**Prerequisites:** Python 3.13, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/you/actus && cd actus

uv sync
source .venv/bin/activate

cat > .env << 'EOF'
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///./actus.db
EOF

mkdir -p config/agents

make run      # uvicorn with hot reload
make test     # run the test suite
```

### Docker

**Prerequisites:** Docker, Docker Compose.

```bash
git clone https://github.com/you/actus && cd actus

cat > .env << 'EOF'
SECRET_KEY=your-secret-key-here
DEBUG=false
DATABASE_URL=sqlite:///./data/actus.db
OLLAMA_BASE_URL=http://host.docker.internal:11434
EOF

make docker-up
```

The first build takes a few minutes (downloads the spaCy NLP model). Subsequent starts are fast.

**Docker commands:**

| Command | What it does |
|---|---|
| `make docker-up` | Build (if needed) and start in foreground |
| `make docker-up-d` | Start in background |
| `make docker-logs` | Tail container logs |
| `make docker-restart` | Restart without rebuilding |
| `make docker-down` | Stop and remove the container |


---

**Verify either setup:**

```bash
curl http://localhost:8000/healthz
# {"status":"ok","checks":{"database":"ok","ollama":"ok"}}
```

API docs: `http://localhost:8000/docs`

---

## Documentation

## License

MIT.
