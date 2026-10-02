"""Reranking cu un cross-encoder: citește împreună întrebarea și chunk-ul.

Mai lent decât căutarea vectorială (un pas de model per pereche), dar mult mai precis,
așa că îl aplicăm doar pe primii ~20 de candidați.
"""

from functools import cache

from sentence_transformers import CrossEncoder

from legislatie_rag.config import RERANKER_MODEL
from legislatie_rag.index.embed import device


@cache
def get_reranker(name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(name, device=device(), max_length=1024)


def rerank_scores(query: str, texts: list[str], batch_size: int = 8) -> list[float]:
    if not texts:
        return []
    scores = get_reranker().predict(
        [(query, t) for t in texts], batch_size=batch_size, activation_fn=None
    )
    return [float(s) for s in scores]
