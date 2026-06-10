# Actus

**FastAPI-based multi-agent platform with LLM routing, RAG-powered context retrieval, and operational automation.**

Actus is an internal automation platform. Deploy it, register agents for your operations, and let them work autonomously on your data. A self-hosted infrastructure layer your team owns and controls.

Define agents in YAML for whatever you need to automate: analyse customer data, monitor servers, digest logs, process documents, generate reports. Each agent runs on a schedule or on demand, calls your internal tools, and reads your domain knowledge through RAG. Multiple agents run independently.

Actus provides the platform, you bring the tools. A tool is a Python function decorated with `@tool` that connects an agent to your systems: your database, your APIs, your filesystem. The platform handles the agent loop, retries, timeouts, PII scrubbing, and observability. You write the functions that do the actual work.

---

## Stack

```
Python 3.13      FastAPI + Uvicorn       SQLModel + SQLite/PostgreSQL
LiteLLM          Ollama                  Presidio (PII)
APScheduler      structlog               bcrypt + PyJWT
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
# Not ready yet:  {"status":"degraded","core":{"database":"ok"},"info":{"ollama":"unreachable","redis":"ok"}}
# Ready:          {"status":"ok","core":{"database":"ok"},"info":{"ollama":"ok","redis":"ok"}}
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

## Demo Agents

Pre-built agents in `config/agents/` that demonstrate the platform's capabilities end-to-end.

### Document Q&A

Answers questions about uploaded PDF and DOCX files. Demonstrates: file upload → parse → chunk → vector index → semantic retrieval → grounded answer → cleanup.

**Step 1 — Upload the document**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/login \
  -d "username=admin&password=pass" | jq -r .access_token)

FILE_PATH=$(curl -s -X POST http://localhost:8000/v1/doc-qa/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/report.pdf" | jq -r .file_path)
```

**Step 2 — Trigger the agent with the file path and your question**

```bash
WF_ID=$(curl -s -X POST http://localhost:8000/v1/automation/trigger/doc_qa \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"extra_context\": {\"file_path\": \"$FILE_PATH\", \"question\": \"What are the main findings?\"}}" \
  | jq -r .workflow_id)
```

**Step 3 — Stream the result (or poll)**

```bash
# Stream live events — closes automatically when the agent finishes
curl -N "http://localhost:8000/v1/automation/workflows/$WF_ID/stream" \
  -H "Authorization: Bearer $TOKEN"

# Or watch the poll endpoint (refreshes every 2 s)
watch -n 2 curl -s "http://localhost:8000/v1/automation/workflows/$WF_ID" \
  -H "Authorization: Bearer $TOKEN"
```

**Supported formats:** `.pdf`, `.docx` — up to 20 MB per file.

**How it works:** The agent calls `chunk_and_index_document` (parse + embed chunks into pgvector), then `search_document` (semantic retrieval), then composes an answer grounded in the retrieved passages, then calls `cleanup_document` to remove the session's vector index rows before signalling done.

**SQLite dev mode:** Indexing and retrieval are no-ops on SQLite. The agent will respond that no document information was found — expected behaviour. Run with PostgreSQL for full functionality.

---

## API

All application routes are versioned under `/v1` (e.g. `/v1/auth/login`, `/v1/automation/...`, `/v1/llm/...`). `/healthz` and `/docs` remain unprefixed.

### Auth

| Endpoint | Notes |
|---|---|
| `POST /v1/auth/login` | OAuth2 password grant. Returns `access_token` + `refresh_token` |
| `POST /v1/auth/refresh` | Exchange a refresh token for a new token pair |
| `POST /v1/auth/logout` | Revokes the presented access token |

Admin actions that change a user's credentials or privileges (password reset, role change, account deletion) bump that user's `token_version`, which immediately invalidates all of their previously issued access and refresh tokens.

### Model allow-list

By default, `/v1/llm/*` accepts any LiteLLM-routable model string. Set `ALLOWED_MODELS` (JSON list) in your environment to restrict which models callers may request. Requests for any other model return `403`. See `.env.example`.

---

## Production: TLS & Reverse Proxy

Actus does not terminate TLS itself. Run it behind a reverse proxy (Caddy, nginx, Traefik) that:

- Terminates TLS and forwards plain HTTP to the `actus` container.
- Forwards the real client IP via `X-Forwarded-For`, the `actus` container's uvicorn is started with `--proxy-headers --forwarded-allow-ips=...` (see `docker-compose.yml` / `FORWARDED_ALLOW_IPS` in `.env.example`) so `request.client.host` (used in audit logs and rate-limit keys) reflects the real client, not the proxy. The default trusts the Docker bridge network range (`172.16.0.0/12`); narrow this to your proxy's actual address in production, and never set it to `*`. Anyone able to reach the `actus` container directly (e.g. the published port) could otherwise spoof their source IP.

Minimal Caddy example:

```
app.example.com {
    reverse_proxy localhost:8000
}
```

Set `CORS_ORIGINS` to your public HTTPS origin(s), e.g. `CORS_ORIGINS=["https://app.example.com"]`.

---

## Documentation

## License

MIT.
