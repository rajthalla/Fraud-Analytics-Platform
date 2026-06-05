"""
Eval harness for the FinTxn RAG + Text-to-SQL pipeline.

Metrics:
  recall@k        — fraction of RAG questions where expected chunk is in top-k
  MRR             — mean reciprocal rank of expected chunk across RAG questions
  RAG accuracy    — LLM-as-judge correctness on RAG answers
  SQL exact-match — numeric match on SQL result vs expected value
  routing         — fraction routed to the correct tool (excludes "either" questions)
  guardrail       — fraction of out-of-scope questions correctly refused

Run modes (from project root):
    .venv/bin/python rag/eval/run_eval.py            # full eval (~52 LLM calls)
    .venv/bin/python rag/eval/run_eval.py --minimal  # ~18 LLM calls: skips judge + routing
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make rag/ importable when run from project root
RAG_DIR      = Path(__file__).parent.parent
PROJECT_ROOT = RAG_DIR.parent
sys.path.insert(0, str(RAG_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from retriever import retrieve, DEFAULT_K
from answer import answer
from text_to_sql import query as sql_query
from router import _classify
from llm import generate

EVAL_PATH   = Path(__file__).parent / "eval_set.jsonl"
CALL_DELAY  = 6.0   # seconds between generate() calls — stays under 10 RPM free-tier limit


def load_eval():
    items = []
    with open(EVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def chunk_matches(chunk, expected: str) -> bool:
    full = f"{chunk.source} › {chunk.section}"
    return expected.strip() in full or full in expected.strip()


def llm_judge(question: str, expected: str, actual: str) -> tuple[bool, str]:
    """Binary correctness check via LLM. Returns (correct, one-line reason)."""
    prompt = (
        f"Question: {question}\n"
        f"Expected answer (key facts): {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Does the actual answer contain the key facts from the expected answer, "
        "even if worded differently?\n"
        "Reply with CORRECT or INCORRECT on the first line, then one sentence why."
    )
    resp = generate(prompt).strip()
    time.sleep(CALL_DELAY)
    correct = resp.upper().startswith("CORRECT")
    reason  = resp.splitlines()[0] if resp else ""
    return correct, reason


def scalar_match(actual, expected_val: float) -> bool:
    """Numeric match with tolerance: 0.5 for integers, 0.1% for floats."""
    try:
        a, e = float(actual), float(expected_val)
        tol = 0.5 if abs(e) > 10 else max(abs(e) * 0.001, 0.01)
        return abs(a - e) <= tol
    except (TypeError, ValueError):
        return False


def _sleep():
    time.sleep(CALL_DELAY)


def run_rag(item: dict, skip_judge: bool = False, skip_routing: bool = False) -> dict:
    # Retrieval (no LLM call)
    ret  = retrieve(item["question"], k=DEFAULT_K)
    rank = None
    for i, c in enumerate(ret.chunks, 1):
        if item.get("expected_chunk") and chunk_matches(c, item["expected_chunk"]):
            rank = i
            break

    # Answer (1 LLM call)
    ans = answer(item["question"])
    _sleep()

    # Judge (1 LLM call, skippable)
    answer_correct = None
    judge_reason   = ""
    if not skip_judge and item.get("expected_answer") and ans["is_grounded"]:
        answer_correct, judge_reason = llm_judge(
            item["question"], item["expected_answer"], ans["answer"]
        )

    # Routing (1 LLM call, skippable)
    route = "—"
    route_correct = None
    if not skip_routing:
        route = _classify(item["question"])
        route_correct = route == "rag"
        _sleep()

    return {
        "rank":           rank,
        "rr":             1.0 / rank if rank else 0.0,
        "answer_correct": answer_correct,
        "judge_reason":   judge_reason,
        "route":          route,
        "route_correct":  route_correct,
        "actual_answer":  ans["answer"][:120],
    }


def run_sql(item: dict, skip_routing: bool = False) -> dict:
    r = sql_query(item["question"])  # 1 LLM call
    _sleep()

    sql_correct = None
    actual_val  = None
    if r.get("result") is not None and not r["result"].empty and item.get("expected_value") is not None:
        actual_val  = r["result"].iloc[0, 0]
        sql_correct = scalar_match(actual_val, item["expected_value"])
    elif r.get("error"):
        sql_correct = False

    route = "—"
    route_correct = None
    if not skip_routing:
        route = _classify(item["question"])  # 1 LLM call
        route_correct = route == "sql"
        _sleep()

    return {
        "sql":           (r.get("sql") or "")[:80],
        "actual_value":  actual_val,
        "sql_correct":   sql_correct,
        "error":         r.get("error"),
        "used_retry":    r.get("used_retry", False),
        "route":         route,
        "route_correct": route_correct,
    }


def run_guardrail(item: dict, skip_routing: bool = False) -> dict:
    ans   = answer(item["question"])   # retrieve only — no LLM call when not grounded
    fired = not ans["is_grounded"]

    route = "—"
    route_correct = None
    if not skip_routing:
        route = _classify(item["question"])   # 1 LLM call
        route_correct = route == "rag"
        _sleep()

    return {
        "guardrail_fired": fired,
        "top_similarity":  ans.get("top_similarity"),
        "route":           route,
        "route_correct":   route_correct,
    }


def run_either(item: dict) -> dict:
    route = _classify(item["question"])
    _sleep()
    return {"route": route, "route_correct": True}


def run_eval(skip_judge: bool = False, skip_routing: bool = False, rag_only: bool = False):
    items = load_eval()
    if rag_only:
        items = [i for i in items if i["tool"] == "rag"]
        mode = "rag-judge-only"
    elif skip_judge:
        mode = "minimal (no judge, no routing)"
    else:
        mode = "full"
    print(f"\nEval set: {len(items)} questions  |  k={DEFAULT_K}  |  mode={mode}\n")

    # Metric accumulators
    recall_hits, rrs          = [], []
    rag_correct               = []
    sql_correct_list          = []
    route_correct_list        = []
    guardrail_caught          = []

    # Per-question log rows
    rows = []

    for item in items:
        q_id   = item["id"]
        tool   = item["tool"]
        diff   = item.get("difficulty", "")
        print(f"  {q_id} [{tool:10}] {item['question'][:55]}")

        if tool == "rag":
            try:
                r = run_rag(item, skip_judge=skip_judge, skip_routing=skip_routing)
            except Exception as e:
                print(f"    ERROR: {str(e)[:80]} — skipping")
                rows.append({"id": q_id, "tool": tool, "diff": diff,
                             "rank": "err", "rr": "—", "ans_ok": "err",
                             "route": "—", "rt_ok": "—", "note": str(e)[:55]})
                continue

            if item.get("expected_chunk"):
                recall_hits.append(r["rank"] is not None)
                rrs.append(r["rr"])

            if r["answer_correct"] is not None:
                rag_correct.append(r["answer_correct"])

            if not skip_routing:
                route_correct_list.append(r["route_correct"])

            rows.append({
                "id": q_id, "tool": tool, "diff": diff,
                "rank":   str(r["rank"]) if r["rank"] else "miss",
                "rr":     f"{r['rr']:.2f}",
                "ans_ok": "✓" if r["answer_correct"] else ("✗" if r["answer_correct"] is not None else "—"),
                "route":  r["route"],
                "rt_ok":  "✓" if r["route_correct"] else "✗",
                "note":   r["judge_reason"][:55] if r["judge_reason"] else item.get("notes","")[:55],
            })

        elif tool == "sql":
            r = run_sql(item, skip_routing=skip_routing)

            if r["sql_correct"] is not None:
                sql_correct_list.append(r["sql_correct"])

            if not skip_routing:
                route_correct_list.append(r["route_correct"])

            note = (
                f"got {r['actual_value']}, exp {item.get('expected_value')}"
                if r["actual_value"] is not None
                else (r.get("error") or "")[:55]
            )
            rows.append({
                "id": q_id, "tool": tool, "diff": diff,
                "rank": "—", "rr": "—",
                "ans_ok": "✓" if r["sql_correct"] else ("✗" if r["sql_correct"] is not None else "—"),
                "route":  r["route"],
                "rt_ok":  "✓" if r["route_correct"] else "✗",
                "note":   note[:55],
            })

        elif tool == "guardrail":
            r = run_guardrail(item, skip_routing=skip_routing)
            guardrail_caught.append(r["guardrail_fired"])
            if not skip_routing:
                route_correct_list.append(r["route_correct"])

            rows.append({
                "id": q_id, "tool": tool, "diff": diff,
                "rank": "—", "rr": "—",
                "ans_ok": "✓" if r["guardrail_fired"] else "✗",
                "route":  r["route"],
                "rt_ok":  "✓" if r["route_correct"] else "✗",
                "note":   f"sim={r['top_similarity']:.3f}" if r.get("top_similarity") else "",
            })

        elif tool == "either":
            r = run_either(item) if not skip_routing else {"route": "—", "route_correct": True}
            rows.append({
                "id": q_id, "tool": tool, "diff": diff,
                "rank": "—", "rr": "—",
                "ans_ok": "—",
                "route":  r["route"],
                "rt_ok":  "either",
                "note":   item.get("notes","")[:55],
            })

    sep = "=" * 68

    print(f"\n\n{sep}")
    print("  EVAL RESULTS — FinTxn RAG + Text-to-SQL Pipeline")
    print(sep)

    if recall_hits:
        r5  = sum(recall_hits) / len(recall_hits)
        mrr = sum(rrs) / len(rrs)
        print(f"\nRETRIEVAL  (k={DEFAULT_K}, {len(recall_hits)} labeled RAG questions)")
        print(f"  recall@{DEFAULT_K}:  {r5:.3f}   ({sum(recall_hits)}/{len(recall_hits)} chunks in top-{DEFAULT_K})")
        print(f"  MRR:       {mrr:.3f}")

    if rag_correct:
        acc = sum(rag_correct) / len(rag_correct)
        print(f"\nRAG ANSWER ACCURACY  (LLM-as-judge, {len(rag_correct)} questions)")
        print(f"  accuracy:  {acc:.3f}   ({sum(rag_correct)}/{len(rag_correct)} correct)")
        print(f"  * Caveat: LLM-as-judge; manually verify any INCORRECT results")

    if sql_correct_list:
        acc = sum(sql_correct_list) / len(sql_correct_list)
        print(f"\nSQL EXACT-MATCH  ({len(sql_correct_list)} questions)")
        print(f"  accuracy:  {acc:.3f}   ({sum(sql_correct_list)}/{len(sql_correct_list)} correct)")

    if route_correct_list:
        acc = sum(route_correct_list) / len(route_correct_list)
        n_either = sum(1 for i in items if i["tool"] == "either")
        print(f"\nROUTING ACCURACY  ({len(route_correct_list)} questions, {n_either} 'either' excluded)")
        print(f"  accuracy:  {acc:.3f}   ({sum(route_correct_list)}/{len(route_correct_list)} correct)")

    if guardrail_caught:
        rate = sum(guardrail_caught) / len(guardrail_caught)
        print(f"\nGUARDRAIL CATCH RATE  ({len(guardrail_caught)} out-of-scope questions)")
        print(f"  rate:      {rate:.3f}   ({sum(guardrail_caught)}/{len(guardrail_caught)} correctly refused)")

    # Per-question detail
    print(f"\n{'-' * 68}")
    print(f"{'ID':<6} {'Tool':<10} {'Diff':<7} {'Rank':<5} {'RR':<5} {'Ans':<4} {'Rte':<5} {'RT':<4}  Note")
    print(f"{'-' * 68}")
    for row in rows:
        print(
            f"{row['id']:<6} {row['tool']:<10} {row['diff']:<7} "
            f"{row['rank']:<5} {row['rr']:<5} {row['ans_ok']:<4} "
            f"{row['route']:<5} {str(row['rt_ok']):<4}  {row['note']}"
        )
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Skip LLM judge and routing classification (~18 calls vs ~52; fits free-tier daily cap)",
    )
    parser.add_argument(
        "--rag-judge-only",
        action="store_true",
        help="Run only RAG questions with judge; skip SQL generation and routing (~20 calls)",
    )
    args = parser.parse_args()
    if args.rag_judge_only:
        run_eval(skip_judge=False, skip_routing=True, rag_only=True)
    else:
        run_eval(skip_judge=args.minimal, skip_routing=args.minimal)
