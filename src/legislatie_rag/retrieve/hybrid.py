"""Retrieval hibrid: referințe explicite + BM25 + vectori → RRF → reranker → articole.

Utilizare:
    uv run python -m legislatie_rag.retrieve.hybrid "Câte zile de concediu de odihnă am?"
"""

import sys
import time
from dataclasses import dataclass, field

from legislatie_rag.index.bm25 import BM25Index
from legislatie_rag.index.vector import VectorIndex
from legislatie_rag.ingest.chunk import Chunk, load_chunks
from legislatie_rag.retrieve.refs import detect_article_refs
from legislatie_rag.retrieve.rerank import rerank_scores

RRF_K = 60  # constanta standard din articolul original despre Reciprocal Rank Fusion


@dataclass
class SearchConfig:
    """Fiecare pas se poate opri, pentru a-i măsura contribuția în evaluare."""

    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    use_explicit_refs: bool = True
    candidates: int = 30
    """Câți candidați aduce fiecare metodă de căutare."""
    rerank_top: int = 20
    """Câți candidați fuzionați trec prin reranker."""
    bm25_weight: float = 1.0
    """Greutatea BM25 în fuziune (vectorii au 1.0)."""


@dataclass
class ArticleHit:
    """Un articol găsit, cu chunk-urile relevante din el (în ordinea din articol)."""

    law_slug: str
    article: str
    chunks: list[Chunk]
    score: float
    explicit: bool = False
    sources: set[str] = field(default_factory=set)

    @property
    def citation(self) -> str:
        return self.chunks[0].citation

    @property
    def key(self) -> tuple[str, str]:
        return (self.law_slug, self.article)


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], weights: list[float] | None = None, k: int = RRF_K
) -> dict[str, float]:
    """Scor = Σ w / (k + rang). Combină clasamente cu scoruri incomparabile (BM25 vs. cosinus)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (k + rank)
    return scores


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self._by_id = {c.id: c for c in self.chunks}
        self._by_article: dict[tuple[str, str], list[Chunk]] = {}
        for c in self.chunks:
            self._by_article.setdefault((c.law_slug, c.article), []).append(c)
        self.bm25 = BM25Index(self.chunks)
        self.vector = VectorIndex(self.chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        laws: list[str] | None = None,
        config: SearchConfig | None = None,
    ) -> list[ArticleHit]:
        cfg = config or SearchConfig()

        # 1. Referințe explicite: articolul cerut intră direct, primul.
        pinned: list[ArticleHit] = []
        if cfg.use_explicit_refs:
            for ref in detect_article_refs(query):
                if chunks := self._by_article.get((ref.law_slug, ref.article)):
                    pinned.append(
                        ArticleHit(ref.law_slug, ref.article, chunks, float("inf"), True, {"ref"})
                    )

        # 2. Căutare lexicală și semantică.
        rankings: dict[str, list[Chunk]] = {}
        if cfg.use_bm25:
            rankings["bm25"] = [c for c, _ in self.bm25.search(query, cfg.candidates, laws)]
        if cfg.use_vector:
            rankings["vector"] = [c for c, _ in self.vector.search(query, cfg.candidates, laws)]

        # 3. Fuziune.
        weights = [cfg.bm25_weight if name == "bm25" else 1.0 for name in rankings]
        fused = reciprocal_rank_fusion(list(rankings.values()), weights)
        candidates = sorted(fused, key=fused.get, reverse=True)

        # 4. Reranking: scorul cross-encoder-ului înlocuiește scorul RRF.
        if cfg.use_rerank and candidates:
            candidates = candidates[: cfg.rerank_top]
            scores = rerank_scores(query, [self._by_id[cid].text for cid in candidates])
            final = dict(zip(candidates, scores, strict=True))
        else:
            final = fused

        # 5. Grupare pe articol: mai multe părți din același articol devin un singur rezultat.
        found_by = {name: {c.id for c in ranking} for name, ranking in rankings.items()}
        hits: dict[tuple[str, str], ArticleHit] = {h.key: h for h in pinned}
        for cid in sorted(final, key=final.get, reverse=True):
            chunk = self._by_id[cid]
            key = (chunk.law_slug, chunk.article)
            sources = {name for name, ids in found_by.items() if cid in ids}
            if key not in hits:
                hits[key] = ArticleHit(*key, [chunk], final[cid], sources=sources)
            elif not hits[key].explicit:
                hits[key].chunks.append(chunk)
                hits[key].sources |= sources

        for hit in hits.values():
            hit.chunks.sort(key=lambda c: c.part)
        return list(hits.values())[:k]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Câte zile de concediu de odihnă am minim pe an?"
    start = time.perf_counter()
    retriever = Retriever()
    print(f"Încărcare: {time.perf_counter() - start:.1f}s\n")

    retriever.search(query)  # încălzește modelele (prima rulare pe MPS e lentă)
    start = time.perf_counter()
    hits = retriever.search(query)
    print(f"Întrebare: {query}   ({(time.perf_counter() - start) * 1000:.0f} ms)\n")
    for i, hit in enumerate(hits, start=1):
        tag = "explicit" if hit.explicit else f"{hit.score:.2f}"
        print(f"{i}. {hit.citation:<28} [{tag}] via {', '.join(sorted(hit.sources))}")
        print(f"   {hit.chunks[0].body[:160].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
