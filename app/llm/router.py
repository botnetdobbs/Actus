import logging
import time
import litellm
from litellm.exceptions import APIConnectionError, Timeout, ServiceUnavailableError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import get_settings, Settings
from app.llm.models import CompletionRequest, CompletionResponse
from app.llm.pii import scrub_pii
from app.observability.metrics import llm_calls_total, llm_latency
import structlog

log = structlog.get_logger()
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_settings = get_settings()


def llm_retry(fn):
    return retry(
        stop=stop_after_attempt(_settings.llm_max_retries),
        wait=wait_exponential(multiplier=_settings.llm_retry_base_delay, min=1, max=30),
        retry=retry_if_exception_type((
            APIConnectionError,
            Timeout,
            ServiceUnavailableError,
            RateLimitError,
        )),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )(fn)


@llm_retry
async def call_llm_with_retry(model, messages, **kwargs):
    return await litellm.acompletion(model=model, messages=messages, **kwargs)

@router.post("/complete", response_model=CompletionResponse)
@limiter.limit("20/minute")
async def complete(
    request: Request,
    req: CompletionRequest,
    settings: Settings = Depends(get_settings),
):
    model = req.model or settings.default_model
    pii_found = False
    clean_messages = []

    for msg in req.messages:
        if msg.role == "user":
            clean_content, detected = scrub_pii(msg.content)
            pii_found = pii_found or detected
            clean_messages.append({"role": msg.role, "content": clean_content})
        else:
            clean_messages.append(msg.model_dump())

    if settings.debug:
        log.debug("llm_prompt", messages=clean_messages)
    else:
        log.info(
            "llm_call",
            model=model,
            message_count=len(clean_messages),
            total_chars=sum(len(m["content"]) for m in clean_messages),
            pii_scrubbed=pii_found,
        )

    start = time.monotonic()
    try:
        response = await call_llm_with_retry(
            model=model,
            messages=clean_messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            api_base=settings.ollama_base_url if "ollama" in model else None,
        )
    except Exception:
        llm_calls_total.labels(model=model, status="error").inc()
        raise
    latency_s = time.monotonic() - start
    llm_calls_total.labels(model=model, status="ok").inc()
    llm_latency.labels(model=model).observe(latency_s)
    latency_ms = latency_s * 1000
    cost = litellm.completion_cost(completion_response=response)
    log.info(
        "llm_call_complete",
        model=model,
        latency_ms=round(latency_ms, 1),
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        cost_usd=round(cost, 6),
        pii_detected=pii_found,
    )
    return CompletionResponse(
        content=response.choices[0].message.content,
        model=response.model,
        usage=dict(response.usage),
        pii_detected=pii_found,
        request_id=getattr(request.state, "request_id", None),
    )
