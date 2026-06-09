import json
from app.evals.models import AssertionResult, EvalAssertion


def check_assertion(result: dict, assertion: EvalAssertion) -> AssertionResult:
    atype = assertion.type

    if atype == "status":
        actual = result.get("status", "")
        passed = actual == assertion.value
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else f"expected status={assertion.value!r}, got {actual!r}",
        )

    if atype == "result_contains":
        text = str(result.get("result") or "")
        needle = str(assertion.value)
        passed = needle in text
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else f"result does not contain {needle!r}",
        )

    if atype == "result_not_contains":
        text = str(result.get("result") or "")
        needle = str(assertion.value)
        passed = needle not in text
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else f"result unexpectedly contains {needle!r}",
        )

    if atype == "schema_valid":
        schema_valid = result.get("output_schema_valid")
        if schema_valid is None:
            return AssertionResult(
                assertion_type=atype,
                passed=False,
                message="Agent has no output_schema. Schema_valid assertion skipped",
            )
        passed = bool(schema_valid)
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else "output_schema_valid is False",
        )

    if atype == "max_tokens":
        actual = result.get("total_tokens", 0)
        limit = int(assertion.value)  # type: ignore[arg-type]
        passed = actual <= limit
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else f"total_tokens={actual} exceeds max={limit}",
        )

    if atype == "min_confidence":
        confidence = result.get("confidence")
        if confidence is None:
            return AssertionResult(
                assertion_type=atype,
                passed=False,
                message="confidence is None (native tool calling mode does not emit confidence)",
            )
        passed = float(confidence) >= float(assertion.value)  # type: ignore[arg-type]
        return AssertionResult(
            assertion_type=atype,
            passed=passed,
            message="" if passed else f"confidence={confidence} < min={assertion.value}",
        )

    return AssertionResult(
        assertion_type=atype,
        passed=False,
        message=f"Unknown assertion type: {atype!r}",
    )
