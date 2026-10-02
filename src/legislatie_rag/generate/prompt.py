"""Promptul de sistem și formatarea surselor pentru LLM."""

from legislatie_rag.config import LAWS
from legislatie_rag.retrieve.hybrid import ArticleHit

_INDEXED_LAWS = "; ".join(f"{law.name} ({law.short_ref})" for law in LAWS)

SYSTEM_PROMPT = f"""Ești un asistent care explică legislația din România pe baza textului oficial.

Primești o întrebare și o listă de surse numerotate [1], [2], ... cu articole de lege.

Reguli:
1. Răspunde EXCLUSIV pe baza surselor primite. Nu folosi alte cunoștințe despre lege, chiar \
dacă le ai: legile se modifică, iar sursele sunt forma în vigoare.
2. După fiecare afirmație pune marcajul sursei în paranteze drepte, ex. [1]. Menționează și \
referința precisă în text, ex. „potrivit art. 145 alin. (1) din Codul muncii”.
3. Dacă sursele nu conțin răspunsul, spune clar că nu ai găsit informația în legile indexate \
și nu ghici. Legile indexate sunt: {_INDEXED_LAWS}.
4. Verifică domeniul de aplicare: dacă o prevedere se referă la altă situație decât cea din \
întrebare (alt tip de societate, altă categorie de persoane), spune explicit asta în loc să o \
aplici greșit.
5. Răspunde în română, concis și pe înțelesul oricui. Începe cu răspunsul direct, apoi detaliile.
6. Nu da sfaturi juridice personalizate. Pentru situații concrete complexe poți recomanda, \
pe scurt, consultarea unui avocat."""

NO_SOURCES_ANSWER = (
    "Nu am găsit în legile indexate prevederi relevante pentru această întrebare. "
    f"Momentan pot răspunde doar pe baza următoarelor acte: {_INDEXED_LAWS}."
)


def format_source(n: int, hit: ArticleHit) -> str:
    first = hit.chunks[0]
    body = "\n".join(c.body for c in hit.chunks)
    parts = "" if first.n_parts == len(hit.chunks) else " (extras)"
    return f"[{n}] {first.header}{parts}\n{body}"


def build_prompt(question: str, hits: list[ArticleHit]) -> str:
    sources = "\n\n".join(format_source(i, hit) for i, hit in enumerate(hits, start=1))
    return f"SURSE:\n\n{sources}\n\n---\nÎNTREBARE: {question}"
