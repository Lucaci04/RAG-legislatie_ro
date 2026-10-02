"""Index lexical BM25 peste chunk-uri.

Se construiește în memorie la pornire: pentru ~1.000 de chunk-uri durează sub o secundă,
deci nu merită persistat.
"""

from rank_bm25 import BM25Okapi

from legislatie_rag.index.text import tokenize
from legislatie_rag.ingest.chunk import Chunk


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(
        self, query: str, k: int = 30, laws: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        allowed = [
            i for i in range(len(self.chunks)) if not laws or self.chunks[i].law_slug in laws
        ]
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
