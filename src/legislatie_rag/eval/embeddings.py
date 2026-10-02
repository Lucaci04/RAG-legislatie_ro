"""Compară modele de embeddings pe căutarea vectorială (fără BM25 și reranker).

Fiecare model indexează în memorie aceleași chunk-uri; măsurăm calitatea căutării,
timpul de indexare și dimensiunea modelului.

Utilizare:
    uv run python -m legislatie_rag.eval.embeddings
"""

import gc
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from legislatie_rag.eval.dataset import RESULTS_DIR, load_questions
from legislatie_rag.eval.retrieval import K_VALUES, first_gold_rank, summarize
from legislatie_rag.index.embed import device
from legislatie_rag.ingest.chunk import load_chunks


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    """Modelele E5 au fost antrenate cu prefixe „query: ” / „passage: ” și le cer la inferență."""


MODELS = [
    EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    EmbeddingModel("intfloat/multilingual-e5-small", "query: ", "passage: "),
    EmbeddingModel("intfloat/multilingual-e5-base", "query: ", "passage: "),
    EmbeddingModel("BAAI/bge-m3"),
]


@dataclass
class EmbeddingResult:
    model: str
    params_m: float
    dim: int
    recall: dict[int, float]
    mrr: float
    index_seconds: float


def evaluate(spec: EmbeddingModel, chunks, questions) -> EmbeddingResult:
    model = SentenceTransformer(spec.name, device=device())

    start = time.perf_counter()
    corpus = model.encode(
        [spec.passage_prefix + c.text for c in chunks],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    index_seconds = time.perf_counter() - start

    queries = model.encode(
        [spec.query_prefix + q.question for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    similarities = queries @ corpus.T

    ranks = []
    for q, sims in zip(questions, similarities, strict=True):
        keys: list[str] = []
        for i in np.argsort(-sims):
            key = f"{chunks[i].law_slug}:{chunks[i].article}"
            if key not in keys:
                keys.append(key)
            if len(keys) == max(K_VALUES):
                break
        ranks.append(first_gold_rank(keys, q.gold))

    recall, mrr = summarize(ranks)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    result = EmbeddingResult(
        spec.name, round(params), corpus.shape[1], recall, mrr, round(index_seconds, 1)
    )

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def main() -> None:
    chunks = load_chunks()
    questions = [q for q in load_questions() if q.gold]

    results = []
    print(f"{'Model':<58} {'param':>6} {'dim':>5} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'index':>7}")
    for spec in MODELS:
        r = evaluate(spec, chunks, questions)
        results.append(r)
        print(
            f"{r.model:<58} {r.params_m:>5.0f}M {r.dim:>5} {r.recall[1]:>6.1%} "
            f"{r.recall[5]:>6.1%} {r.mrr:>6.3f} {r.index_seconds:>6.0f}s"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "embeddings.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
