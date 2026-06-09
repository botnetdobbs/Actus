import asyncio
import json
import urllib.parse
from collections.abc import AsyncGenerator
import structlog

log = structlog.get_logger()

_redis = None  # module-level singleton; set by init_redis()


async def init_redis(url: str) -> None:
    global _redis
    if not url:
        log.info("redis_not_configured", note="SSE will fall back to DB polling")
        return
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(url, decode_responses=True)
        await client.ping()
        _redis = client
        parsed = urllib.parse.urlparse(url)
        log.info("redis_connected", host=parsed.hostname, port=parsed.port)
    except Exception as e:
        log.warning("redis_unavailable", error=str(e),
                    note="SSE will fall back to DB polling")


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None


async def publish_event(channel: str, event: dict) -> None:
    if _redis is None:
        return
    try:
        await _redis.publish(channel, json.dumps(event))
    except Exception as e:
        log.warning("redis_publish_failed", channel=channel, error=str(e))


async def subscribe_workflow(channel: str, timeout: float = 660.0) -> AsyncGenerator[dict, None]:
    """Yield events from a Redis pub/sub channel until sse_end, timeout, or error.

    The timeout (default: AGENT_TOTAL_TIMEOUT + 60s) handles the race where a
    fast-completing agent publishes sse_end before the subscriber attaches,
    the generator exits and the SSE caller falls through to the authoritative DB read.
    """
    if _redis is None:
        return
    pubsub_conn = _redis.pubsub()
    try:
        await pubsub_conn.subscribe(channel)
        deadline = asyncio.get_event_loop().time() + timeout
        async for message in pubsub_conn.listen():
            if asyncio.get_event_loop().time() > deadline:
                log.warning("redis_subscribe_timeout", channel=channel)
                break
            if message["type"] != "message":
                continue
            try:
                event = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict) and event.get("type") == "sse_end":
                break
            yield event
    except Exception as e:
        log.warning("redis_subscribe_error", channel=channel, error=str(e))
    finally:
        try:
            await pubsub_conn.unsubscribe(channel)
            await pubsub_conn.aclose()
        except Exception:
            pass


def is_available() -> bool:
    return _redis is not None
