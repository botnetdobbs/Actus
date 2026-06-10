import asyncio
import logging
import time
import litellm
from litellm.exceptions import APIConnectionError, Timeout, ServiceUnavailableError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.config import get_settings
from app.limiter import limiter
from app.llm.models import CompletionRequest, CompletionResponse
from app.llm.pii import scrub_pii_async
from app.observability.metrics import llm_calls_total, llm_latency
import structlog

log = structlog.get_logger()
router = APIRouter()

_settings = get_settings()


def llm_retry(fn):
    return retry(
        stop=stop_after_attempt(_settings.llm_max_retries),
        wait=wait_exponential(multiplier=_settings.llm_retry_base_delay, min=1, max=30),
        retry=retry_if_exception_type((
            APIConnectionError,
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
async def complete(request: Request, req: CompletionRequest):
    model = req.model or _settings.default_model
    if _settings.allowed_models and model not in _settings.allowed_models:
        raise HTTPException(status_code=403, detail=f"Model '{model}' is not in the allow-list")
    pii_found = False
    clean_messages = []

    for msg in req.messages:
        if msg.role == "user":
            clean_content, detected = await scrub_pii_async(msg.content)
            pii_found = pii_found or detected
            clean_messages.append({"role": msg.role, "content": clean_content})
        else:
            clean_messages.append(msg.model_dump())

    if _settings.debug:
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
            api_base=_settings.ollama_base_url if "ollama" in model else None,
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
        prompt_tokens=response.usage.prompt_tokens,  # pyright: ignore[reportAttributeAccessIssue]
        completion_tokens=response.usage.completion_tokens,  # pyright: ignore[reportAttributeAccessIssue]
        total_tokens=response.usage.total_tokens,  # pyright: ignore[reportAttributeAccessIssue]
        cost_usd=round(cost, 6),
        pii_detected=pii_found,
    )
    return CompletionResponse(
        content=response.choices[0].message.content,  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        model=response.model,  # pyright: ignore[reportArgumentType]
        usage=dict(response.usage),  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        pii_detected=pii_found,
        request_id=getattr(request.state, "request_id", None),
    )


@router.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, req: CompletionRequest):
    model = req.model or _settings.default_model
    if _settings.allowed_models and model not in _settings.allowed_models:
        raise HTTPException(status_code=403, detail=f"Model '{model}' is not in the allow-list")
    clean_messages = []
    for msg in req.messages:
        if msg.role == "user":
            clean_content, _ = await scrub_pii_async(msg.content)
            clean_messages.append({"role": msg.role, "content": clean_content})
        else:
            clean_messages.append(msg.model_dump())

    async def token_generator():
        try:
            response = await litellm.acompletion(
                model=model,
                messages=clean_messages,
                stream=True,
                api_base=_settings.ollama_base_url if "ollama" in model else None,
            )
            async for chunk in response:  # pyright: ignore[reportGeneralTypeIssues]
                delta = chunk.choices[0].delta.content  # pyright: ignore[reportAttributeAccessIssue]
                if delta:
                    yield delta
        except asyncio.TimeoutError:
            log.error("stream_timeout", model=model)
            yield "\n\n[Error: LLM response timed out]"
        except APIConnectionError:
            log.error("stream_connection_error", model=model)
            yield "\n\n[Error: Could not connect to LLM]"
        except Exception as e:
            log.error("stream_error", model=model, error=str(e))
            yield "\n\n[Error: internal error]"

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )
