from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

llm_calls_total = Counter(
    "actus_llm_calls_total",
    "Total LLM API calls",
    ["model", "status"],
)
llm_latency = Histogram(
    "actus_llm_latency_seconds",
    "LLM call latency in seconds",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
agent_runs_total = Counter(
    "actus_agent_runs_total",
    "Total agent runs",
    ["agent_id", "status"],
)
active_agent_runs = Gauge(
    "actus_active_agent_runs",
    "Currently running agents",
)


def instrument_app(app: FastAPI) -> None:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
    ).instrument(app).expose(app, endpoint="/metrics")
