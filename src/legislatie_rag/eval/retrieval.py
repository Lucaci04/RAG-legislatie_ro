"""Evaluarea căutării: găsește sistemul articolul corect, și pe ce poziție?

Metrici (doar pe întrebările care au un articol corect):
- Recall@k: procentul de întrebări la care un articol corect apare în primele k rezultate
- MRR: media lui 1/poziția primului articol corect (1 = mereu primul)

Utilizare:
    uv run python -m legislatie_rag.eval.retrieval
"""

import json
import time
from dataclasses import asdict, dataclass

from legislatie_rag.eval.dataset import RESULTS_DIR, Question, load_questions
from legislatie_rag.retrieve.hybrid import Retriever, SearchConfig

K_VALUES = (1, 3, 5, 10)

VARIANTS: dict[str, SearchConfig] = {
    "BM25": SearchConfig(use_vector=False, use_rerank=False, use_explicit_refs=False),
    "Vectori (bge-m3)": SearchConfig(use_bm25=False, use_rerank=False, use_explicit_refs=False),
    "Hibrid (RRF)": SearchConfig(use_rerank=False, use_explicit_refs=False),
    "Hibrid ponderat (BM25 × 0.3)": SearchConfig(
        use_rerank=False, use_explicit_refs=False, bm25_weight=0.3
    ),
    "Vectori + reranker": SearchConfig(use_bm25=False, use_explicit_refs=False),
    "Hibrid + reranker": SearchConfig(use_explicit_refs=False),
    "Hibrid + reranker + referințe": SearchConfig(),
}


@dataclass
class RetrievalMetrics:
    variant: str
    n: int
    recall: dict[int, float]
    mrr: float
    latency_ms: float
    by_type: dict[str, dict[str, float]]


def first_gold_rank(ranked_keys: list[str], gold: list[str]) -> int | None:
    for rank, key in enumerate(ranked_keys, start=1):
        if key in gold:
            return rank
    return None


def summarize(ranks: list[int | None]) -> tuple[dict[int, float], float]:
    n = len(ranks)
    recall = {k: sum(r is not None and r <= k for r in ranks) / n for k in K_VALUES}
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return recall, mrr


def evaluate_variant(
    retriever: Retriever, questions: list[Question], name: str, config: SearchConfig
) -> tuple[RetrievalMetrics, list[dict]]:
    ranks: list[int | None] = []
    rows: list[dict] = []
    latencies: list[float] = []
    for q in questions:
        start = time.perf_counter()
        hits = retriever.search(q.question, k=max(K_VALUES), config=config)
        latencies.append((time.perf_counter() - start) * 1000)
        keys = [f"{h.law_slug}:{h.article}" for h in hits]
        rank = first_gold_rank(keys, q.gold)
        ranks.append(rank)
        rows.append({"variant": name, "id": q.id, "type": q.type, "rank": rank, "top5": keys[:5]})

    recall, mrr = summarize(ranks)
    by_type = {}
    for type_ in sorted({q.type for q in questions}):
        type_ranks = [r for r, q in zip(ranks, questions, strict=True) if q.type == type_]
        t_recall, t_mrr = summarize(type_ranks)
        by_type[type_] = {"n": len(type_ranks), "recall@5": t_recall[5], "mrr": t_mrr}

    latencies.sort()
    metrics = RetrievalMetrics(
        name, len(questions), recall, mrr, latencies[len(latencies) // 2], by_type
    )
    return metrics, rows


def main() -> None:
    questions = [q for q in load_questions() if q.gold]
    retriever = Retriever()
    retriever.search("încălzire modele")  # prima rulare pe MPS e lentă; nu o măsurăm

    results, all_rows = [], []
    print(f"{'Variantă':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'ms':>6}")
    for name, config in VARIANTS.items():
        metrics, rows = evaluate_variant(retriever, questions, name, config)
        results.append(metrics)
        all_rows.extend(rows)
        r = metrics.recall
        print(
            f"{name:<32} {r[1]:>6.1%} {r[3]:>6.1%} {r[5]:>6.1%} {r[10]:>6.1%} "
            f"{metrics.mrr:>6.3f} {metrics.latency_ms:>6.0f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(
        json.dumps([asdict(m) for m in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (RESULTS_DIR / "retrieval_details.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    misses = [r for r in all_rows if r["variant"] == list(VARIANTS)[-1] and r["rank"] != 1]
    if misses:
        print("\nVarianta completă — articolul corect nu e pe locul 1:")
        for row in misses:
            print(f"  {row['id']:<24} poziția {row['rank']}  top3: {row['top5'][:3]}")


if __name__ == "__main__":
    main()
