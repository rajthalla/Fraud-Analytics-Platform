import sys
from pathlib import Path

from dotenv import load_dotenv

from llm import generate
from answer import answer as rag_answer
from text_to_sql import query as sql_query

load_dotenv(Path(__file__).parent.parent / ".env")

_CLASSIFY_PROMPT = """\
You are routing a question for a financial analytics assistant that has two tools:

  sql — executes a SELECT against a DuckDB warehouse; use for counts, totals, averages,
        lists of rows, numeric aggregations, or any question whose answer is a number or table.

  rag — searches compliance policy and data dictionary documents; use for questions about
        what something means, a policy rule, an SLA, a procedure, a column definition,
        or a threshold explanation.

Reply with exactly one word: sql  or  rag.
If genuinely uncertain, reply: rag

Question: {question}"""


def _classify(question: str) -> str:
    response = generate(_CLASSIFY_PROMPT.format(question=question)).strip().lower()
    # Accept any response that contains "sql" as a word; default to rag otherwise
    if "sql" in response.split():
        return "sql"
    return "rag"


def route(question: str) -> dict:
    """Classify the question and dispatch to the correct tool.

    Returns a dict that always contains:
        route      — "sql" or "rag"
    Plus all keys from the underlying tool's return value.
    """
    classification = _classify(question)

    if classification == "sql":
        result = sql_query(question)
    else:
        result = rag_answer(question)

    return {"route": classification, **result}


def _print_result(question: str) -> None:
    print(f"\nQ: {question}")
    r = route(question)
    print(f"→ routed to: {r['route'].upper()}")
    print("-" * 60)

    if r["route"] == "sql":
        print(f"SQL: {r['sql']}")
        if r.get("used_retry"):
            print("(used retry)")
        if r.get("error"):
            print(f"Error: {r['error']}")
        elif r.get("result") is not None:
            print(r["result"].to_string(index=False))

    else:  # rag
        if r.get("is_grounded"):
            print(r["answer"])
            print("\nSources:")
            for s in r.get("sources", []):
                print(f"  · {s}")
        else:
            print(r["answer"])
            print(f"  [out of scope — top similarity {r.get('top_similarity', '?'):.3f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print_result(" ".join(sys.argv[1:]))
    else:
        tests = [
            "what SLA applies to confirmed fraud?",
            "what is the total transaction volume by type?",
            "which transaction types are considered high-risk?",
        ]
        for q in tests:
            _print_result(q)
