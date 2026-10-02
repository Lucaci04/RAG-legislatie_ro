"""Pipeline-ul complet: întrebare → retrieval → LLM → răspuns cu citări verificate.

Utilizare:
    uv run python -m legislatie_rag.generate.answer "Câte zile de concediu de odihnă am?"
"""

import re
import sys
import time
from dataclasses import dataclass, field

from legislatie_rag.generate.llm import LLM, LLMUnavailableError, get_llm
from legislatie_rag.generate.prompt import NO_SOURCES_ANSWER, SYSTEM_PROMPT, build_prompt
from legislatie_rag.retrieve.hybrid import ArticleHit, Retriever

_SOURCE_MARKER = re.compile(r"\[(\d+)\]")

DISCLAIMER = "Informație orientativă pe baza textului legii, nu consultanță juridică."
LLM_UNAVAILABLE_ANSWER = (
    "Modelul de limbaj nu este disponibil momentan (limită de utilizare sau suprasolicitare). "
    "Mai jos sunt articolele de lege găsite pentru întrebarea ta."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[ArticleHit]
    cited: list[int] = field(default_factory=list)
    """Numerele surselor citate în răspuns, în ordinea primei apariții."""
    invalid_citations: list[int] = field(default_factory=list)
    """Marcaje [n] fără sursă corespunzătoare: semn că modelul a inventat o citare."""
    model: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    llm_error: bool = False
    """Generarea a eșuat; `sources` conține totuși articolele găsite."""

    @property
    def cited_sources(self) -> list[ArticleHit]:
        return [self.sources[n - 1] for n in self.cited]


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    markers = [int(m) for m in _SOURCE_MARKER.findall(text)]
    unique = list(dict.fromkeys(markers))
    valid = [n for n in unique if 1 <= n <= n_sources]
    invalid = [n for n in unique if not 1 <= n <= n_sources]
    return valid, invalid


class RAG:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None, k: int = 5):
        self.retriever = retriever or Retriever()
        self.llm = llm or get_llm()
        self.k = k

    def ask(self, question: str, laws: list[str] | None = None) -> Answer:
        start = time.perf_counter()
        hits = self.retriever.search(question, k=self.k, laws=laws)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not hits:
            return Answer(question, NO_SOURCES_ANSWER, [], retrieval_ms=retrieval_ms)

        start = time.perf_counter()
        try:
            completion = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        except LLMUnavailableError:
            return Answer(
                question, LLM_UNAVAILABLE_ANSWER, hits, retrieval_ms=retrieval_ms, llm_error=True
            )
        generation_ms = (time.perf_counter() - start) * 1000

        cited, invalid = check_citations(completion.text, len(hits))
        return Answer(
            question=question,
            text=completion.text,
            sources=hits,
            cited=cited,
            invalid_citations=invalid,
            model=completion.model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )


def print_answer(answer: Answer) -> None:
    print(f"\n❓ {answer.question}\n")
    print(answer.text)
    print("\n📚 Surse citate:")
    for n in answer.cited:
        hit = answer.sources[n - 1]
        date = hit.chunks[0].consolidation_date or "forma republicată"
        print(f"  [{n}] {hit.citation}  (versiune: {date})  {hit.chunks[0].url}")
    if answer.invalid_citations:
        print(f"⚠️  Citări fără sursă: {answer.invalid_citations}")
    print(
        f"\n⏱  căutare {answer.retrieval_ms:.0f} ms · generare {answer.generation_ms:.0f} ms"
        f" · {answer.model}\nℹ️  {DISCLAIMER}"
    )


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Câte zile de concediu de odihnă am minim pe an?"
    print_answer(RAG().ask(question))


if __name__ == "__main__":
    main()
