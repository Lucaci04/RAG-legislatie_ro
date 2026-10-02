"""Detectează referințe explicite la articole în întrebare („art. 145 din Codul muncii”)."""

import re
from dataclasses import dataclass

from legislatie_rag.config import LAWS
from legislatie_rag.index.text import strip_diacritics

# „art. 145”, „art 145”, „articolul 152^2”, „art. 152 ind. 2”, „art. 152 indice 2”
_ARTICLE = re.compile(
    r"\bart(?:icolul|icolului|icol|\.)?\s*(\d+)(?:\s*(?:\^|ind(?:ice)?\.?)\s*(\d+))?",
)


@dataclass(frozen=True)
class ArticleRef:
    law_slug: str
    article: str


def detect_laws(query: str) -> list[str]:
    """Legile menționate explicit în întrebare, în ordinea din config."""
    text = strip_diacritics(query.lower())
    return [
        law.slug
        for law in LAWS
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in law.aliases)
    ]


def detect_article_refs(query: str) -> list[ArticleRef]:
    """Referințe la articole, doar când legea e clară.

    „art. 1” fără lege e ambiguu (există în toate legile), așa că îl lăsăm căutării normale.
    Dacă e menționată o singură lege, toate articolele din întrebare îi aparțin.
    """
    laws = detect_laws(query)
    if len(laws) != 1:
        return []
    refs = []
    for number, superscript in _ARTICLE.findall(strip_diacritics(query.lower())):
        article = f"{number}^{superscript}" if superscript else number
        refs.append(ArticleRef(laws[0], article))
    return list(dict.fromkeys(refs))
