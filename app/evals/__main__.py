"""CLI entry point: python -m app.evals [agent_id|--all] [--timeout N] [--dry-run] [-v]"""
import argparse
import asyncio
import sys
from app.agents.builder import load_agents
from app.agents.discovery import discover_tools
from app.evals.loader import load_all_suites, load_suite
from app.evals.models import EvalResult
from app.evals.runner import run_suite


def _print_result(result: EvalResult, verbose: bool) -> None:
    status = "PASS" if result.passed else "FAIL"
    prefix = f"  [{status}] {result.case_id}"
    if result.error:
        print(f"{prefix}; ERROR: {result.error}")
        return
    print(prefix)
    if verbose or not result.passed:
        for ar in result.assertion_results:
            mark = "✓" if ar.passed else "✗"
            line = f"    {mark} {ar.assertion_type}"
            if ar.message:
                line += f": {ar.message}"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Actus evaluation harness")
    parser.add_argument("agent_id", nargs="?", help="Agent ID to evaluate (omit with --all)")
    parser.add_argument("--all", action="store_true", help="Run all eval suites in evals/")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout per case (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; all assertions pass")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all assertion results")
    parser.add_argument("--evals-dir", default="evals", help="Directory containing .yaml fixtures")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent cases")
    args = parser.parse_args()

    if not args.agent_id and not args.all:
        parser.error("Provide an agent_id or --all")

    discover_tools()
    try:
        load_agents()
    except Exception as e:
        print(f"ERROR: Could not load agents: {e}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        suites = load_all_suites(args.evals_dir)
        if not suites:
            print(f"No .yaml fixtures found in '{args.evals_dir}/'")
            sys.exit(0)
    else:
        try:
            suites = [load_suite(args.agent_id, args.evals_dir)]
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    total_cases = 0
    total_pass = 0

    async def _run_all():
        out = []
        for suite in suites:
            results = await run_suite(
                suite, timeout_override=args.timeout,
                concurrency=args.concurrency, dry_run=args.dry_run,
            )
            out.append((suite, results))
        return out

    all_data = asyncio.run(_run_all())

    for suite, results in all_data:
        print(f"\n=== {suite.agent_id} — {suite.description} ===")
        for result in results:
            _print_result(result, args.verbose)
            total_cases += 1
            if result.passed:
                total_pass += 1

    total_fail = total_cases - total_pass
    print(f"\n{'─' * 40}")
    print(f"Results: {total_pass}/{total_cases} passed, {total_fail} failed")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
