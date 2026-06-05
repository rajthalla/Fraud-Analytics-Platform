import sys
from pathlib import Path

from dotenv import load_dotenv

from retriever import retrieve, RetrievalResult
from llm import generate

load_dotenv(Path(__file__).parent.parent / ".env")

OUT_OF_SCOPE_MSG = (
    "I don't have information on that in the FinTxn corpus. "
    "My knowledge is limited to the data dictionary and compliance policy documents."
)

SYSTEM_PROMPT = """\
You are a financial compliance and data assistant for the FinTxn platform.
Answer questions using ONLY the numbered context sections provided.
Do not draw on outside knowledge.
At the end of your answer, cite the context sections you used as [1], [2], etc.
If the context does not contain enough information to answer, say so explicitly.\
"""


def _format_context(chunks) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.source} › {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, k: int = 7) -> dict:
    """Return a dict with keys: answer, is_grounded, sources, top_similarity."""
    result: RetrievalResult = retrieve(question, k=k)

    if not result.is_grounded:
        return {
            "answer":         OUT_OF_SCOPE_MSG,
            "is_grounded":    False,
            "top_similarity": result.top_similarity,
            "sources":        [],
        }

    context  = _format_context(result.chunks)
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"

    answer_text = generate(user_msg, system_prompt=SYSTEM_PROMPT)

    sources = [
        f"{c.source} › {c.section}  (similarity {c.similarity:.3f})"
        for c in result.chunks
    ]

    return {
        "answer":         answer_text,
        "is_grounded":    True,
        "top_similarity": result.top_similarity,
        "sources":        sources,
    }


def _print_answer(question: str) -> None:
    print(f"\nQ: {question}")
    print("-" * 60)
    result = answer(question)
    print(result["answer"])
    if result["is_grounded"]:
        print("\nSources retrieved:")
        for s in result["sources"]:
            print(f"  · {s}")
    else:
        print(f"  [out of scope — top similarity {result['top_similarity']:.3f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _print_answer(" ".join(sys.argv[1:]))
    else:
        # Default test questions
        questions = [
            "What is the anomaly score threshold for flagging transactions, and why is it at that value?",
            "What happens when a CASH_OUT transaction fully drains the sender's account?",
            "What does the fraud_risk_score column mean and where should I look for the active risk signal?",
            "What is the interest rate on a savings account?",
        ]
        for q in questions:
            _print_answer(q)
