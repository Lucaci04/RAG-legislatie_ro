"""Tokenizare pentru căutarea lexicală (BM25) în limba română."""

import re
import unicodedata

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("romanian")
_TOKEN = re.compile(r"\w+(?:\^\w+)?")  # păstrează „152^2” ca un singur token

# Cuvinte funcționale frecvente; nu ajută la diferențierea articolelor.
STOPWORDS = frozenset(
    """
    a acea aceasta aceea acel acela acest acesta aceste acestea acestei acestor acestui
    ai al ale alt alte altor am ar are as asa asupra au avea ca care cat ce cea cei cel
    cele celor ci cu cum da daca de decat deci din dintre e ei el ele este eu fara fi
    fie fiecare fost i ia ii il in inca incat insa intr intre isi iar la le li lor lui
    ma mai mi ne nici nu o ori pe pentru prin sa sau se si sunt spre sub sus ta te tu
    un una unei unor unui va vor
    """.split()  # noqa: SIM905 — mai lizibil decât o listă literală
)


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Litere mici → fără diacritice → fără stopwords → stem.

    Diacriticele se elimină înainte de stemming ca textul legii și întrebările scrise
    fără diacritice („concediu de odihna”) să producă aceleași tokenuri.
    """
    words = _TOKEN.findall(strip_diacritics(text.lower()))
    return _STEMMER.stemWords([w for w in words if w not in STOPWORDS])
