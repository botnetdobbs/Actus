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

**Prerequisites:** Docker and Docker Compose.

```bash
git clone https://github.com/you/actus && cd actus

cat > .env << 'EOF'
SECRET_KEY=your-secret-key-here
POSTGRES_PASSWORD=your-postgres-password-here
GRAFANA_PASSWORD=your-grafana-password-here
DEBUG=false
EOF

make docker-up-d       # start all services in background
make ollama-pull       # pull the model into the Ollama container (first run only)
```

The first build takes a few minutes. The spaCy NLP model is downloaded into the image. Subsequent starts are fast. Actus starts immediately; Ollama initialises in the background (typically 1-3 minutes on first run).

To check when Ollama is ready:

```bash
curl http://localhost:8000/healthz
# Not ready yet:  {"status":"degraded","checks":{"database":"ok","ollama":"unreachable"}}
# Ready:          {"status":"ok","checks":{"database":"ok","ollama":"ok"}}
```

**Services:**

| Service | URL | Notes |
|---|---|---|
| Actus API | `http://localhost:8000` | API docs at `/docs` |
| Prometheus | `http://localhost:9090` | Metrics storage |
| Grafana | `http://localhost:3000` | Dashboards — login: `admin` / `GRAFANA_PASSWORD` |

**Commands:**

| Command | What it does |
|---|---|
| `make docker-up` | Build (if needed) and start in foreground |
| `make docker-up-d` | Start in background |
| `make docker-logs` | Tail all service logs |
| `make docker-restart` | Restart Actus without rebuilding |
| `make docker-rebuild` | Rebuild image and restart all services |
| `make docker-down` | Stop and remove all containers |
| `make ollama-pull` | Pull a model into the Ollama container |

Agent YAML files in `config/agents/` are volume-mounted — add or edit agents and `make docker-restart`, no rebuild needed. Database data persists in a Docker volume across restarts.

---

## Documentation

## License

MIT.
