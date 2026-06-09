import asyncio
from app.agents.builder import get_agent
from app.agents.orchestrator import run_agent_with_timeout
from app.evals.assertions import check_assertion
from app.evals.models import AssertionResult, EvalCase, EvalResult, EvalSuite


async def _run_case(suite: EvalSuite, case: EvalCase, dry_run: bool) -> EvalResult:
    if dry_run:
        return EvalResult(
            case_id=case.id,
            agent_id=suite.agent_id,
            passed=True,
            error=None,
            assertion_results=[
                AssertionResult(assertion_type=a.type, passed=True, message="(dry-run)")
                for a in case.assertions
            ],
        )

    try:
        config = get_agent(suite.agent_id)
    except KeyError as e:
        return EvalResult(
            case_id=case.id,
            agent_id=suite.agent_id,
            passed=False,
            error=str(e),
        )

    extra_context = case.input.get("extra_context") if case.input else None

    try:
        result = await asyncio.wait_for(
            run_agent_with_timeout(config, extra_context=extra_context),
            timeout=case.timeout_seconds,
        )
    except asyncio.TimeoutError:
        return EvalResult(
            case_id=case.id,
            agent_id=suite.agent_id,
            passed=False,
            error=f"Case timed out after {case.timeout_seconds}s",
        )
    except Exception as e:
        return EvalResult(
            case_id=case.id,
            agent_id=suite.agent_id,
            passed=False,
            error=str(e),
        )

    assertion_results = [check_assertion(result, a) for a in case.assertions]
    passed = all(ar.passed for ar in assertion_results)
    return EvalResult(
        case_id=case.id,
        agent_id=suite.agent_id,
        passed=passed,
        assertion_results=assertion_results,
        run_result=result,
    )


async def run_suite(
    suite: EvalSuite,
    timeout_override: int | None = None,
    concurrency: int = 3,
    dry_run: bool = False,
) -> list[EvalResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(case: EvalCase) -> EvalResult:
        async with semaphore:
            if timeout_override is not None:
                case = case.model_copy(update={"timeout_seconds": timeout_override})
            return await _run_case(suite, case, dry_run)

    return list(await asyncio.gather(*[_bounded(c) for c in suite.cases]))
