from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CHROMA_DIR          = Path(__file__).parent / "chroma_db"
COLLECTION_NAME     = "fintxn_docs"
DEFAULT_K           = 7
GROUNDING_THRESHOLD = -0.30   # top similarity below this → out of scope
# Calibrated against eval set: worst in-scope = -0.206, best out-of-scope = -0.382
# Gap of 0.18 between the two populations; -0.30 sits in the middle.


@dataclass
class Chunk:
    text:       str
    source:     str
    section:    str
    similarity: float


@dataclass
class RetrievalResult:
    is_grounded:    bool
    top_similarity: float
    chunks:         list[Chunk] = field(default_factory=list)


def _get_collection():
    ef     = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME, embedding_function=ef)


def retrieve(question: str, k: int = DEFAULT_K) -> RetrievalResult:
    collection = _get_collection()
    results    = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = [
        Chunk(
            text       = doc,
            source     = meta["source"],
            section    = meta["section"],
            similarity = 1 - dist,
        )
        for doc, meta, dist in zip(docs, metas, distances)
    ]

    top_similarity = chunks[0].similarity if chunks else -999.0
    is_grounded    = top_similarity >= GROUNDING_THRESHOLD

    return RetrievalResult(
        is_grounded    = is_grounded,
        top_similarity = top_similarity,
        chunks         = chunks,
    )
