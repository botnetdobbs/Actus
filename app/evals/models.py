from typing import Literal
from pydantic import BaseModel


class EvalAssertion(BaseModel):
    type: Literal["status", "result_contains", "result_not_contains",
                  "schema_valid", "max_tokens", "min_confidence"]
    value: str | int | float | None = None


class EvalCase(BaseModel):
    id: str
    description: str = ""
    input: dict = {}
    assertions: list[EvalAssertion] = []
    timeout_seconds: int = 60


class EvalSuite(BaseModel):
    agent_id: str
    description: str = ""
    cases: list[EvalCase]


class AssertionResult(BaseModel):
    assertion_type: str
    passed: bool
    message: str = ""


class EvalResult(BaseModel):
    case_id: str
    agent_id: str
    passed: bool
    error: str | None = None
    assertion_results: list[AssertionResult] = []
    run_result: dict | None = None
