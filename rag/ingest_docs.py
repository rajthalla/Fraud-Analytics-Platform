import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DOCS_DIR   = Path(__file__).parent.parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

DOCS = [
    "data_dictionary.md",
    "compliance_policy.md",
]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown doc into one chunk per ## section.

    Each chunk carries the section header as metadata so answers can cite it.
    """
    chunks = []
    # Split on ## headings (keep the heading in the chunk)
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Extract the heading for metadata
        first_line = section.splitlines()[0]
        heading = first_line.lstrip("#").strip()
        chunks.append({
            "text":    section,
            "source":  source,
            "section": heading,
        })
    return chunks


def main():
    ef     = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe and recreate so re-runs don't accumulate duplicates
    try:
        client.delete_collection("fintxn_docs")
    except Exception:
        pass
    collection = client.create_collection("fintxn_docs", embedding_function=ef)

    all_chunks = []
    for filename in DOCS:
        path = DOCS_DIR / filename
        text = path.read_text()
        chunks = chunk_markdown(text, source=filename)
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    documents = [c["text"]                                   for c in all_chunks]
    metadatas = [{"source": c["source"], "section": c["section"]} for c in all_chunks]
    ids       = [f"doc_{i}"                                  for i in range(len(all_chunks))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"\nIndexed {len(all_chunks)} chunks → {CHROMA_DIR}")

    print("\n" + "="*60)
    print("VERIFICATION — raw similarity queries")
    print("="*60)

    queries = [
        "What is the anomaly score threshold for flagging transactions?",
        "What happens to a fully-drained CASH_OUT account?",
        "What does the fraud_risk_score column mean?",
        "What is the SLA for confirmed fraud escalation?",
    ]

    for q in queries:
        results = collection.query(query_texts=[q], n_results=2)
        print(f"\nQ: {q}")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta['source']} › {meta['section']}]")
            print(f"  {doc[:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
