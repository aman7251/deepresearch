"""Per-run ephemeral RAG index.

Each research run builds its own in-memory Chroma collection over the pages the
agent fetched. The verifier later queries it claim-by-claim to check support.
"""
from __future__ import annotations

import re
import uuid
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings


@lru_cache
def _embedding_fn():
    settings = get_settings()
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)


def _chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


class EphemeralIndex:
    """A throwaway vector index scoped to a single research run."""

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"run_{uuid.uuid4().hex[:8]}",
            embedding_function=_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
        self._count = 0

    def add_document(self, text: str, source_n: int, url: str) -> int:
        chunks = _chunk(text)
        if not chunks:
            return 0
        ids = [f"s{source_n}_c{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source_n": source_n, "url": url} for _ in chunks],
        )
        self._count += len(chunks)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if self._count == 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=min(k, self._count))
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "source_n": meta.get("source_n"),
                    "url": meta.get("url"),
                    "score": round(1.0 - float(dist), 4),  # cosine distance -> similarity
                }
            )
        return out

    @property
    def size(self) -> int:
        return self._count
