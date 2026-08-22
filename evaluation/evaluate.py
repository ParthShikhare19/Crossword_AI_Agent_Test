"""
Deterministic evaluation runner for the Aster & Row support agent.

Run from the repository root:

    python evaluation/evaluate.py

Optional:

    python evaluation/evaluate.py --visible-only
    python evaluation/evaluate.py --original-only

The evaluator checks:

    - required answer claims
    - forbidden answer claims
    - required sources
    - forbidden sources
    - intent / routing
    - human handoff
    - order lookup usage
    - order lookup arguments
    - multi-turn behavior

For failed cases, the evaluator prints the complete interaction so that
we can distinguish:

    1. A real agent problem
    2. An incorrect evaluation expectation
    3. An evaluation-runner problem

Results are also saved to:

    evaluation/results/final-results.json
    evaluation/results/final-results.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch


# ============================================================================
# PROJECT PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

BACKEND = ROOT / "backend"

VISIBLE_CASES = ROOT / "evaluation" / "visible-cases.json"

ORIGINAL_CASES = ROOT / "evaluation" / "original-cases.json"

RESULTS_DIR = ROOT / "evaluation" / "results"


# ============================================================================
# BACKEND IMPORT PATH
# ============================================================================

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ============================================================================
# APPLICATION IMPORTS
# ============================================================================

from app.agent import agent as agent_module
from app.agent.agent import SupportAgent, create_llm
from app.agent.memory import conversation_memory
from app.rag.retriever import create_retriever


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

CATEGORY_MAP = {
    "retrieval": "Retrieval",
    "multi-source-grounding": "Groundedness",
    "multi_source_grounding": "Groundedness",
    "groundedness": "Groundedness",

    "tool-use": "Tool Use",
    "tool_use": "Tool Use",
    "tool": "Tool Use",

    "privacy": "Privacy / Safety",
    "safety": "Privacy / Safety",
    "privacy-safety": "Privacy / Safety",

    "multi-turn": "Multi-turn",
    "multi_turn": "Multi-turn",
    "multiturn": "Multi-turn",
}


# ============================================================================
# LOADING
# ============================================================================

def load_cases(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load evaluation cases from a JSON file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "cases" in data:
            cases = data["cases"]

            if isinstance(cases, list):
                return cases

    raise ValueError(
        f"Unsupported evaluation format: {path}"
    )


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def normalize(
    value: Any,
) -> str:
    """
    Normalize text for case-insensitive matching.
    """

    return " ".join(
        str(value)
        .lower()
        .split()
    )


def answer_contains(
    answer: str,
    phrase: str,
) -> bool:
    """
    Check whether an answer contains a phrase.

    Matching is:

        - case insensitive
        - whitespace normalized
    """

    return (
        normalize(phrase)
        in normalize(answer)
    )


# ============================================================================
# SOURCE UTILITIES
# ============================================================================

def get_source_filenames(
    result: Any,
) -> list[str]:
    """
    Extract filenames from AgentResult.sources.
    """

    sources = result.sources or []

    return [
        str(
            source.get(
                "filename",
                "",
            )
        )
        for source in sources
    ]


def get_source_headings(
    result: Any,
) -> list[str]:
    """
    Extract headings from AgentResult.sources.
    """

    sources = result.sources or []

    return [
        str(
            source.get(
                "heading",
                "",
            )
        )
        for source in sources
    ]


# ============================================================================
# CATEGORY DETECTION
# ============================================================================

def classify_category(
    case: dict[str, Any],
) -> str:
    """
    Determine the README evaluation category for a case.

    Explicit category metadata always takes priority.

    If no category is supplied, infer it from the expectation fields.
    """

    raw_category = str(
        case.get(
            "category",
            "",
        )
    ).strip().lower()

    if raw_category in CATEGORY_MAP:
        return CATEGORY_MAP[
            raw_category
        ]

    expect = case.get(
        "expect",
        {},
    )

    # Multi-turn cases should be identified first.
    if any(
        key.startswith("last_")
        for key in expect
    ):
        return "Multi-turn"

    if (
        "tool" in expect
        or "tool_args" in expect
        or "intent" in expect
    ):
        return "Tool Use"

    if (
        "required_sources" in expect
        or "forbidden_sources_as_authority"
        in expect
    ):
        return "Retrieval"

    if (
        "must_include" in expect
        or "must_not_include" in expect
    ):
        return "Groundedness"

    return "Groundedness"


# ============================================================================
# CLAIM CHECKS
# ============================================================================

def check_claims(
    answer: str,
    must_include: list[str],
    must_not_include: list[str],
) -> list[dict[str, Any]]:
    """
    Check required and forbidden answer phrases.
    """

    checks = []

    for phrase in must_include:

        checks.append(
            {
                "type": "required_claim",
                "value": phrase,
                "passed": answer_contains(
                    answer,
                    phrase,
                ),
            }
        )

    for phrase in must_not_include:

        checks.append(
            {
                "type": "forbidden_claim",
                "value": phrase,
                "passed": not answer_contains(
                    answer,
                    phrase,
                ),
            }
        )

    return checks


# ============================================================================
# SOURCE CHECKS
# ============================================================================

def check_sources(
    result: Any,
    required_sources: list[str],
    forbidden_sources: list[str],
) -> list[dict[str, Any]]:
    """
    Check required and forbidden source filenames.
    """

    actual_sources = (
        get_source_filenames(
            result
        )
    )

    checks = []

    for filename in required_sources:

        checks.append(
            {
                "type": "required_source",
                "value": filename,
                "passed": (
                    filename
                    in actual_sources
                ),
                "actual_sources": actual_sources,
            }
        )

    for filename in forbidden_sources:

        checks.append(
            {
                "type": "forbidden_source",
                "value": filename,
                "passed": (
                    filename
                    not in actual_sources
                ),
                "actual_sources": actual_sources,
            }
        )

    return checks


# ============================================================================
# TOOL CHECKS
# ============================================================================

def check_tool_behavior(
    expected_tool: str | None,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Check whether the expected tool behavior occurred.
    """

    if not expected_tool:
        return []

    if expected_tool == "not_called":

        passed = (
            len(tool_calls) == 0
        )

    elif expected_tool in {
        "order_lookup",
        "lookup_order",
    }:

        passed = (
            len(tool_calls) >= 1
        )

    else:

        # Unknown tool names are not automatically treated as failures.
        # They are reported so the evaluator can be extended later.
        passed = True

    return [
        {
            "type": "tool_behavior",
            "value": expected_tool,
            "passed": passed,
            "actual_calls": tool_calls,
        }
    ]


def check_tool_arguments(
    expected_arguments: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compare expected tool arguments with the first actual tool call.
    """

    if expected_arguments is None:
        return []

    actual_arguments = (
        tool_calls[0]["arguments"]
        if tool_calls
        else None
    )

    return [
        {
            "type": "tool_arguments",
            "value": expected_arguments,
            "passed": (
                actual_arguments
                == expected_arguments
            ),
            "actual": actual_arguments,
        }
    ]


# ============================================================================
# INTENT / HANDOFF CHECKS
# ============================================================================

def check_intent(
    expected_intent: str | None,
    result: Any,
) -> list[dict[str, Any]]:
    """
    Check final application intent.
    """

    if expected_intent is None:
        return []

    return [
        {
            "type": "intent",
            "value": expected_intent,
            "passed": (
                result.intent
                == expected_intent
            ),
            "actual": result.intent,
        }
    ]


def check_handoff(
    expected_handoff: bool | None,
    result: Any,
) -> list[dict[str, Any]]:
    """
    Check human handoff behavior.
    """

    if expected_handoff is None:
        return []

    return [
        {
            "type": "handoff",
            "value": expected_handoff,
            "passed": (
                result.handoff
                == expected_handoff
            ),
            "actual": result.handoff,
        }
    ]


# ============================================================================
# RESULT EVALUATION
# ============================================================================

def evaluate_result(
    result: Any,
    expect: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    prefix: str = "",
) -> list[dict[str, Any]]:
    """
    Evaluate a single AgentResult.

    For multi-turn cases, prefix="last_" is used so that only the
    final turn is evaluated against last_* expectations.
    """

    checks = []

    answer = result.answer or ""

    # ------------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------------

    checks.extend(
        check_claims(
            answer=answer,
            must_include=expect.get(
                f"{prefix}must_include",
                [],
            ),
            must_not_include=expect.get(
                f"{prefix}must_not_include",
                [],
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------------

    checks.extend(
        check_sources(
            result=result,
            required_sources=expect.get(
                f"{prefix}required_sources",
                [],
            ),
            forbidden_sources=expect.get(
                f"{prefix}forbidden_sources_as_authority",
                [],
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------------

    checks.extend(
        check_tool_behavior(
            expected_tool=expect.get(
                f"{prefix}tool"
            ),
            tool_calls=tool_calls,
        )
    )

    # ------------------------------------------------------------------------
    # Tool arguments
    # ------------------------------------------------------------------------

    checks.extend(
        check_tool_arguments(
            expected_arguments=expect.get(
                f"{prefix}tool_args"
            ),
            tool_calls=tool_calls,
        )
    )

    # ------------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------------

    checks.extend(
        check_intent(
            expected_intent=expect.get(
                f"{prefix}intent"
            ),
            result=result,
        )
    )

    # ------------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------------

    checks.extend(
        check_handoff(
            expected_handoff=expect.get(
                f"{prefix}handoff"
            ),
            result=result,
        )
    )

    return checks


# ============================================================================
# CASE EXECUTION
# ============================================================================

def run_case(
    agent: SupportAgent,
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one evaluation case.

    Every case receives an isolated session.

    Multi-turn cases use the same session across all turns.
    """

    case_id = str(
        case["id"]
    )

    category = classify_category(
        case
    )

    messages = case.get(
        "messages",
        [],
    )

    expect = case.get(
        "expect",
        {},
    )

    session_id = (
        f"evaluation-{case_id}"
    )

    conversation_memory.clear_session(
        session_id
    )

    tool_calls: list[
        dict[str, Any]
    ] = []

    turns: list[
        dict[str, Any]
    ] = []

    real_lookup = (
        agent_module.lookup_order
    )

    # ------------------------------------------------------------------------
    # Wrap order lookup so we can observe tool usage without changing it.
    # ------------------------------------------------------------------------

    def traced_lookup(
        order_id: str,
    ):
        tool_calls.append(
            {
                "tool": "order_lookup",
                "arguments": {
                    "order_id": order_id
                },
            }
        )

        return real_lookup(
            order_id
        )

    start = time.perf_counter()

    execution_error = None

    try:

        with patch.object(
            agent_module,
            "lookup_order",
            side_effect=traced_lookup,
        ):

            for index, message in enumerate(
                messages
            ):

                if message.get(
                    "role"
                ) != "user":
                    continue

                result = agent.handle_message(
                    session_id=session_id,
                    user_message=message[
                        "content"
                    ],
                )

                turns.append(
                    {
                        "turn": index + 1,
                        "user": message[
                            "content"
                        ],
                        "answer": result.answer,
                        "sources": result.sources,
                        "handoff": result.handoff,
                        "intent": result.intent,
                    }
                )

        if not turns:

            raise ValueError(
                "Evaluation case contains no user messages."
            )

        # --------------------------------------------------------------------
        # Single turn
        # --------------------------------------------------------------------

        if len(turns) == 1:

            checks = evaluate_result(
                result=result,
                expect=expect,
                tool_calls=tool_calls,
            )

        # --------------------------------------------------------------------
        # Multi-turn
        # --------------------------------------------------------------------

        else:

            checks = evaluate_result(
                result=result,
                expect=expect,
                tool_calls=tool_calls,
                prefix="last_",
            )

    except Exception as exc:

        execution_error = repr(
            exc
        )

        checks = [
            {
                "type": "execution",
                "value": (
                    "case completes without exception"
                ),
                "passed": False,
                "error": execution_error,
            }
        ]

    elapsed_ms = round(
        (
            time.perf_counter()
            - start
        )
        * 1000,
        2,
    )

    passed = (
        execution_error is None
        and bool(checks)
        and all(
            check["passed"]
            for check in checks
        )
    )

    return {
        "id": case_id,
        "category": category,
        "passed": passed,
        "duration_ms": elapsed_ms,
        "turns": turns,
        "tool_calls": tool_calls,
        "checks": checks,
        "execution_error": execution_error,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build category and overall statistics.
    """

    by_category = defaultdict(list)

    for result in results:

        by_category[
            result["category"]
        ].append(
            result
        )

    categories = {}

    for category, cases in sorted(
        by_category.items()
    ):

        total = len(cases)

        passed = sum(
            1
            for case in cases
            if case["passed"]
        )

        score = (
            passed / total * 100
            if total
            else 0.0
        )

        categories[category] = {
            "passed": passed,
            "total": total,
            "failed": total - passed,
            "score_percent": round(
                score,
                2,
            ),
        }

    total_cases = len(
        results
    )

    passed_cases = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_cases = (
        total_cases
        - passed_cases
    )

    overall_score = (
        passed_cases
        / total_cases
        * 100
        if total_cases
        else 0.0
    )

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "overall_score_percent": round(
            overall_score,
            2,
        ),
        "categories": categories,
    }


# ============================================================================
# CONSOLE REPORT
# ============================================================================

def print_failed_case_details(
    result: dict[str, Any],
) -> None:
    """
    Print complete details for a failed case.

    This is intentionally verbose so we can distinguish an agent failure
    from an evaluator expectation failure.
    """

    print()

    print(
        "       "
        "--------------------------------------------------"
    )

    # ------------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------------

    for turn in result["turns"]:

        print(
            "       QUESTION:"
        )

        print(
            "       "
            + str(
                turn["user"]
            )
        )

        print(
            "       ANSWER:"
        )

        print(
            "       "
            + str(
                turn["answer"]
            )
        )

        print(
            "       INTENT:"
        )

        print(
            "       "
            + str(
                turn["intent"]
            )
        )

        print(
            "       HANDOFF:"
        )

        print(
            "       "
            + str(
                turn["handoff"]
            )
        )

        print(
            "       SOURCES:"
        )

        if turn["sources"]:

            for source in turn[
                "sources"
            ]:

                print(
                    "       - "
                    f"{source.get('filename')} | "
                    f"{source.get('heading')}"
                )

        else:

            print(
                "       - None"
            )

    # ------------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------------

    print(
        "       TOOL CALLS:"
    )

    if result["tool_calls"]:

        for tool_call in result[
            "tool_calls"
        ]:

            print(
                "       - "
                f"{tool_call['tool']} "
                f"{tool_call['arguments']}"
            )

    else:

        print(
            "       - None"
        )

    # ------------------------------------------------------------------------
    # Failed checks
    # ------------------------------------------------------------------------

    print(
        "       FAILED CHECKS:"
    )

    failed_checks = [
        check
        for check in result[
            "checks"
        ]
        if not check[
            "passed"
        ]
    ]

    if failed_checks:

        for check in failed_checks:

            print(
                "       - "
                f"{check['type']}: "
                f"{check.get('value')}"
            )

            if "actual" in check:

                print(
                    "         actual: "
                    f"{check['actual']}"
                )

            if "actual_sources" in check:

                print(
                    "         actual sources: "
                    f"{check['actual_sources']}"
                )

            if "actual_calls" in check:

                print(
                    "         actual calls: "
                    f"{check['actual_calls']}"
                )

    else:

        print(
            "       - None"
        )

    if result[
        "execution_error"
    ]:

        print(
            "       EXECUTION ERROR:"
        )

        print(
            "       "
            + result[
                "execution_error"
            ]
        )

    print(
        "       "
        "--------------------------------------------------"
    )


def print_report(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """
    Print the complete evaluation report.
    """

    print()
    print("=" * 70)
    print(
        "ASTER & ROW SUPPORT AGENT"
    )
    print(
        "DETERMINISTIC EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        "INDIVIDUAL CASE RESULTS"
    )
    print("-" * 70)

    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{result['id']} "
            f"({result['category']})"
        )

        if not result["passed"]:

            print_failed_case_details(
                result
            )

    print()
    print(
        "CATEGORY RESULTS"
    )
    print("-" * 70)

    for category, data in (
        summary[
            "categories"
        ].items()
    ):

        print(
            f"{category:<20}"
            f"{data['passed']:>3}/"
            f"{data['total']:<3}"
            f"{data['score_percent']:>8.2f}%"
        )

    print()
    print(
        "OVERALL"
    )
    print("-" * 70)

    print(
        f"Passed: "
        f"{summary['passed_cases']}/"
        f"{summary['total_cases']}"
    )

    print(
        f"Failed: "
        f"{summary['failed_cases']}"
    )

    print(
        f"Score : "
        f"{summary['overall_score_percent']:.2f}%"
    )

    print("=" * 70)
    print()


# ============================================================================
# TEXT RESULT
# ============================================================================

def build_text_result(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    """
    Build a readable text report.
    """

    lines = []

    lines.append(
        "ASTER & ROW SUPPORT AGENT"
    )

    lines.append(
        "DETERMINISTIC EVALUATION"
    )

    lines.append("")

    lines.append(
        f"Passed: "
        f"{summary['passed_cases']}/"
        f"{summary['total_cases']}"
    )

    lines.append(
        f"Failed: "
        f"{summary['failed_cases']}"
    )

    lines.append(
        f"Overall: "
        f"{summary['overall_score_percent']:.2f}%"
    )

    lines.append("")

    lines.append(
        "CATEGORY RESULTS"
    )

    lines.append("")

    for category, data in (
        summary[
            "categories"
        ].items()
    ):

        lines.append(
            f"{category}: "
            f"{data['passed']}/"
            f"{data['total']} "
            f"({data['score_percent']:.2f}%)"
        )

    lines.append("")

    lines.append(
        "FAILED CASES"
    )

    lines.append("")

    for result in results:

        if result["passed"]:
            continue

        lines.append(
            f"[FAIL] "
            f"{result['id']} "
            f"({result['category']})"
        )

        for turn in result[
            "turns"
        ]:

            lines.append(
                f"Question: {turn['user']}"
            )

            lines.append(
                f"Answer: {turn['answer']}"
            )

            lines.append(
                f"Intent: {turn['intent']}"
            )

            lines.append(
                f"Handoff: {turn['handoff']}"
            )

            source_names = [
                source.get(
                    "filename"
                )
                for source in (
                    turn["sources"]
                    or []
                )
            ]

            lines.append(
                "Sources: "
                + (
                    ", ".join(
                        source_names
                    )
                    if source_names
                    else "None"
                )
            )

        lines.append(
            "Tool calls: "
            + (
                json.dumps(
                    result[
                        "tool_calls"
                    ]
                )
                if result[
                    "tool_calls"
                ]
                else "None"
            )
        )

        failed_checks = [
            check
            for check in result[
                "checks"
            ]
            if not check[
                "passed"
            ]
        ]

        for check in failed_checks:

            lines.append(
                "Failed check: "
                f"{check['type']} = "
                f"{check.get('value')}"
            )

        lines.append("")

    return "\n".join(
        lines
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """
    Main evaluation entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Aster & Row behavioral evaluation."
        )
    )

    parser.add_argument(
        "--visible-only",
        action="store_true",
        help=(
            "Run only the supplied visible cases."
        ),
    )

    parser.add_argument(
        "--original-only",
        action="store_true",
        help=(
            "Run only the original cases."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------------

    if (
        args.visible_only
        and args.original_only
    ):

        print(
            "ERROR: "
            "Do not use --visible-only and "
            "--original-only together."
        )

        return 1

    # ------------------------------------------------------------------------
    # Load visible cases
    # ------------------------------------------------------------------------

    try:

        visible_cases = load_cases(
            VISIBLE_CASES
        )

    except Exception as exc:

        print(
            f"ERROR loading visible cases: {exc}"
        )

        return 1

    # ------------------------------------------------------------------------
    # Load original cases
    # ------------------------------------------------------------------------

    original_cases = []

    if ORIGINAL_CASES.exists():

        try:

            original_cases = load_cases(
                ORIGINAL_CASES
            )

        except Exception as exc:

            print(
                f"ERROR loading original cases: {exc}"
            )

            return 1

    # ------------------------------------------------------------------------
    # Select evaluation set
    # ------------------------------------------------------------------------

    if args.visible_only:

        cases = visible_cases

    elif args.original_only:

        cases = original_cases

    else:

        cases = (
            visible_cases
            + original_cases
        )

    print()
    print(
        f"Visible cases  : "
        f"{len(visible_cases)}"
    )

    print(
        f"Original cases : "
        f"{len(original_cases)}"
    )

    print(
        f"Total cases    : "
        f"{len(cases)}"
    )

    print()

    if not cases:

        print(
            "ERROR: No evaluation cases found."
        )

        return 1

    # ------------------------------------------------------------------------
    # Initialize application
    # ------------------------------------------------------------------------

    print(
        "Initializing retriever..."
    )

    try:

        retriever = create_retriever()

    except Exception as exc:

        print(
            "ERROR initializing retriever:"
        )

        print(
            repr(exc)
        )

        return 1

    print(
        "Initializing LLM..."
    )

    try:

        llm = create_llm()

    except Exception as exc:

        print(
            "ERROR initializing LLM:"
        )

        print(
            repr(exc)
        )

        return 1

    agent = SupportAgent(
        retriever=retriever,
        llm=llm,
    )

    # ------------------------------------------------------------------------
    # Run cases
    # ------------------------------------------------------------------------

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            f"Running "
            f"{index}/{len(cases)}: "
            f"{case['id']}"
        )

        result = run_case(
            agent=agent,
            case=case,
        )

        results.append(
            result
        )

    # ------------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------------

    summary = build_summary(
        results
    )

    # ------------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------------

    print_report(
        summary=summary,
        results=results,
    )

    # ------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation_type": (
            "deterministic_behavioral"
        ),
        "visible_case_count": (
            len(visible_cases)
        ),
        "original_case_count": (
            len(original_cases)
        ),
        "summary": summary,
        "cases": results,
    }

    json_path = (
        RESULTS_DIR
        / "final-results.json"
    )

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_result = build_text_result(
        summary=summary,
        results=results,
    )

    txt_path = (
        RESULTS_DIR
        / "final-results.txt"
    )

    txt_path.write_text(
        text_result,
        encoding="utf-8",
    )

    print(
        "Results saved to:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {txt_path}"
    )

    # ------------------------------------------------------------------------
    # Exit code
    # ------------------------------------------------------------------------

    if summary[
        "failed_cases"
    ] == 0:

        return 0

    return 2


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )