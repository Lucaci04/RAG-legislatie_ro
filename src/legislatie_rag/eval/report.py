"""Generează raportul de evaluare: docs/results.md + grafice PNG pentru README.

Utilizare (după rularea evaluărilor):
    uv run python -m legislatie_rag.eval.report
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from legislatie_rag.config import ROOT_DIR  # noqa: E402
from legislatie_rag.eval.dataset import RESULTS_DIR, load_questions  # noqa: E402

DOCS_DIR = ROOT_DIR / "docs"
IMG_DIR = DOCS_DIR / "img"

# Paleta de referință (validată: CVD ΔE 24.7, contrast ≥ 3:1 pe suprafața deschisă).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834"]


def _load(name: str):
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def grouped_hbar(path, title: str, labels: list[str], series: dict[str, list[float]]) -> None:
    """Bare orizontale grupate (≤ 2 serii), valori ca procente, etichete directe la capăt."""
    n_series = len(series)
    height = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(labels) + 1.4), facecolor=SURFACE)
    for s, (name, values) in enumerate(series.items()):
        y = [i + (s - (n_series - 1) / 2) * height for i in range(len(labels))]
        ax.barh(
            y,
            values,
            height=height,
            color=SERIES[s],
            label=name,
            edgecolor=SURFACE,
            linewidth=2,  # spațiu de 2px între bare
        )
        for yi, v in zip(y, values, strict=True):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax, "", "")
    # Titlul și legenda stau pe figură, aliniate la stânga, ca să nu se suprapună.
    fig.text(0.01, 0.97, title, fontsize=12, fontweight="bold", color=TEXT_PRIMARY, va="top")
    fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 0.925),
        ncol=n_series,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def llm_small_multiples(path, rows: list[dict]) -> None:
    """Calitate și latență în două grafice alăturate (nu o axă dublă)."""
    models = [r["model"].removeprefix("gemini-") for r in rows]
    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(10, 0.6 * len(rows) + 1.6), facecolor=SURFACE)

    quality = [r["fully_correct"] for r in rows]
    ax_q.barh(models, quality, color=SERIES[0], height=0.5, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(quality):
        ax_q.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax_q.set_xlim(0, 1.12)
    ax_q.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax_q, "Răspunsuri complet corecte", "")

    latency = [r["latency_p50_s"] for r in rows]
    ax_t.barh(models, latency, color=SERIES[0], height=0.5, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(latency):
        ax_t.text(
            v + max(latency) * 0.02, i, f"{v:.1f}s", va="center", fontsize=8, color=TEXT_SECONDARY
        )
    ax_t.set_xlim(0, max(latency) * 1.2)
    ax_t.set_yticklabels([])
    _style(ax_t, "Latență mediană", "secunde")

    for ax in (ax_q, ax_t):
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    n_answerable = sum(1 for q in questions if q.gold)
    md = [
        "# Rezultatele evaluării",
        "",
        f"Set de evaluare: **{len(questions)} de întrebări** "
        f"([data/eval/questions.jsonl](../data/eval/questions.jsonl)), fiecare cu articolul corect "
        f"verificat manual în textul legii: {n_answerable} cu răspuns în legile indexate și "
        f"{len(questions) - n_answerable} fără (pentru a testa refuzul).",
        "",
        f"> La {n_answerable} de întrebări, o întrebare valorează ~{1 / n_answerable:.0%}. "
        "Diferențele de câteva puncte procentuale trebuie citite cu prudență.",
        "",
    ]

    if retrieval := _load("retrieval.json"):
        grouped_hbar(
            IMG_DIR / "retrieval.png",
            "Căutare: articolul corect în rezultate",
            [r["variant"] for r in retrieval],
            {
                "pe locul 1": [r["recall"]["1"] for r in retrieval],
                "în primele 5": [r["recall"]["5"] for r in retrieval],
            },
        )
        md += [
            "## 1. Variante de căutare",
            "",
            "![Căutare](img/retrieval.png)",
            "",
            "| Variantă | Recall@1 | Recall@3 | Recall@5 | MRR | Latență |",
            "|---|---|---|---|---|---|",
            *(
                f"| {r['variant']} | {pct(r['recall']['1'])} | {pct(r['recall']['3'])} | "
                f"{pct(r['recall']['5'])} | {r['mrr']:.3f} | {r['latency_ms']:.0f} ms |"
                for r in retrieval
            ),
            "",
            "**Concluzii**",
            "",
            "- Căutarea vectorială e mult mai bună decât BM25 la întrebări în limbaj natural.",
            "- Fuziunea simplă BM25 + vectori (RRF) e *mai slabă* decât vectorii singuri: "
            "BM25 primește aceeași greutate deși e mult mai slab.",
            "- Cu reranker, BM25 devine util: aduce candidați pe care vectorii nu-i găsesc, "
            "iar reranker-ul îi alege pe cei buni. Varianta completă găsește articolul corect "
            "în primele 5 rezultate la toate întrebările.",
            "- Reranker-ul costă ~1 s pe întrebare (pe Apple M4 Pro), un compromis acceptabil.",
            "",
        ]

    if embeddings := _load("embeddings.json"):
        grouped_hbar(
            IMG_DIR / "embeddings.png",
            "Modele de embeddings (doar căutare vectorială)",
            [f"{r['model'].split('/')[-1]} ({r['params_m']}M)" for r in embeddings],
            {
                "pe locul 1": [r["recall"]["1"] for r in embeddings],
                "în primele 5": [r["recall"]["5"] for r in embeddings],
            },
        )
        md += [
            "## 2. Modele de embeddings",
            "",
            "![Embeddings](img/embeddings.png)",
            "",
            "| Model | Parametri | Dimensiune | Recall@1 | Recall@5 | MRR | Indexare |",
            "|---|---|---|---|---|---|---|",
            *(
                f"| {r['model']} | {r['params_m']}M | {r['dim']} | {pct(r['recall']['1'])} | "
                f"{pct(r['recall']['5'])} | {r['mrr']:.3f} | {r['index_seconds']:.0f} s |"
                for r in embeddings
            ),
            "",
            "**Concluzii:** bge-m3 câștigă clar la Recall@1. Modelul E5-base, deși de 2,4× mai "
            "mare, nu bate E5-small. MiniLM, antrenat pe parafraze scurte, e nepotrivit pentru "
            "articole de lege.",
            "",
        ]

    if generation := _load("generation.json"):
        llm_small_multiples(IMG_DIR / "llm.png", generation)
        md += [
            "## 3. Modele LLM (generarea răspunsului)",
            "",
            "Toate modelele primesc **aceleași surse** (căutarea rulează o dată). Un judecător "
            "LLM (gemini-3.8-flash, temperatură 0) notează răspunsurile **anonimizate și în "
            "ordine aleatorie**, comparându-le cu răspunsul de referință și cu sursele.",
            "",
            "![LLM](img/llm.png)",
            "",
            "| Model | Scor | Complet corecte | Fidele surselor | Citează articolul corect | "
            "Refuz corect | Refuz greșit | Capcane | p50 | p90 | Tokeni |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *(
                f"| {r['model']} | {pct(r['score'])} | {pct(r['fully_correct'])} | "
                f"{pct(r['faithful'])} | {pct(r['cites_gold'])} | {pct(r['correct_refusals'])} | "
                f"{pct(r['false_refusals'])} | {pct(r['traps'])} | {r['latency_p50_s']:.1f} s | "
                f"{r['latency_p90_s']:.1f} s | {r['output_tokens']:.0f} |"
                for r in generation
            ),
            "",
            "- **Scor**: media notelor (0 = greșit, 1 = parțial, 2 = corect), ca procent.",
            "- **Refuz corect**: la întrebările fără răspuns în legi, a spus că nu știe.",
            "- **Refuz greșit**: a spus că nu știe, deși răspunsul era în surse.",
            "- **Capcane**: întrebări unde articolul găsit pare relevant, dar nu se aplică.",
            "- Latența măsoară doar apelul reușit (fără reîncercări la erori 429/503).",
            "",
            "> Limitare: judecătorul e tot un model Gemini, deci poate favoriza stilul propriei "
            "familii. Notele au fost verificate manual pe un eșantion.",
            "",
        ]

    (DOCS_DIR / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✓ {DOCS_DIR / 'results.md'}")
    for img in sorted(IMG_DIR.glob("*.png")):
        print(f"✓ {img}")


if __name__ == "__main__":
    main()
