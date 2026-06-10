import asyncio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import structlog

log = structlog.get_logger()

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# DATE_TIME causes false positives on UUID hex segments and file paths.
_EXCLUDE_ENTITIES = {"DATE_TIME"}


def scrub_pii(text: str, language: str = "en") -> tuple[str, bool]:
    results = _analyzer.analyze(text=text, language=language)
    results = [r for r in results if r.entity_type not in _EXCLUDE_ENTITIES]
    if not results:
        return text, False
    log.warning(
        "pii_detected",
        entity_types=[r.entity_type for r in results],
        count=len(results),
    )
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)  # pyright: ignore[reportArgumentType]
    return anonymized.text, True


async def scrub_pii_async(text: str, language: str = "en") -> tuple[str, bool]:
    return await asyncio.get_running_loop().run_in_executor(None, scrub_pii, text, language)
