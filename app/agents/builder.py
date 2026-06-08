import yaml
from pathlib import Path
from pydantic import BaseModel, Field
import structlog

log = structlog.get_logger()


class AgentSchedule(BaseModel):
    cron: str | None = None


class WebhookConfig(BaseModel):
    secret: str


class AgentConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    model: str = "ollama/mistral"
    max_iterations: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str = ""
    tools: list[str] = []
    context: dict = {}
    token_budget: int = Field(default=10_000, ge=1)
    max_response_tokens: int = Field(default=1024, ge=1, le=8192)
    api_base: str = ""
    rag_query_template: str = ""
    rag_top_k: int = Field(default=5, ge=1)
    schedule: AgentSchedule | None = None
    webhook: WebhookConfig | None = None


_agents: dict[str, AgentConfig] = {}


def load_agents(config_dir: str = "config/agents") -> None:
    path = Path(config_dir)
    if not path.exists():
        raise RuntimeError(f"Agent config directory not found: '{path.resolve()}'")
    loaded = 0
    failed = 0
    for yaml_file in sorted(path.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if not data:
                log.warning("agent_yaml_empty", file=str(yaml_file))
                continue
            config = AgentConfig(**data)
            _agents[config.id] = config
            log.info("agent_loaded", agent_id=config.id, name=config.name,
                     tools=config.tools, has_schedule=config.schedule is not None)
            loaded += 1
        except Exception as e:
            log.error("agent_load_failed", file=str(yaml_file), error=str(e))
            failed += 1
    log.info("agents_loaded", count=loaded, total_files=len(list(path.glob("*.yaml"))))
    if failed:
        raise RuntimeError(f"{failed} agent file(s) failed to load — fix the errors above before starting.")


def get_agent(agent_id: str) -> AgentConfig:
    if agent_id not in _agents:
        raise KeyError(f"Agent not found: '{agent_id}'. Loaded agents: {list(_agents.keys())}")
    return _agents[agent_id]


def list_agents() -> list[AgentConfig]:
    return list(_agents.values())


def reload_agents(config_dir: str = "config/agents") -> None:
    _agents.clear()
    load_agents(config_dir)
