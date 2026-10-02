from legislatie_rag.index.bm25 import BM25Index
from legislatie_rag.index.text import strip_diacritics, tokenize
from legislatie_rag.ingest.chunk import Chunk


def make_chunk(id_: str, body: str) -> Chunk:
    return Chunk(
        id=id_,
        law_slug="t",
        law_ref="Legea 1/2000",
        article=id_,
        part=1,
        n_parts=1,
        header="Legea de test",
        body=body,
        url="",
        consolidation_date=None,
    )


def test_strip_diacritics():
    assert strip_diacritics("Ăă Ââ Îî Șș Țț") == "Aa Aa Ii Ss Tt"


def test_tokenize_matches_with_and_without_diacritics():
    assert tokenize("concediul de odihnă") == tokenize("Concediul de ODIHNA")


def test_tokenize_stems_inflections_and_drops_stopwords():
    assert tokenize("concediului") == tokenize("concedii")
    assert "de" not in tokenize("concediu de odihnă")


def test_tokenize_keeps_superscript_article_numbers():
    assert "152^2" in tokenize("art. 152^2")


def test_bm25_ranks_relevant_chunk_first():
    index = BM25Index(
        [
            make_chunk("1", "Durata minimă a concediului de odihnă anual este de 20 de zile."),
            make_chunk("2", "Salariul de bază minim brut pe țară garantat în plată."),
            make_chunk("3", "Contractul individual de muncă se încheie în formă scrisă."),
        ]
    )

    results = index.search("cate zile de concediu de odihna am")

    assert results[0][0].id == "1"


def test_bm25_query_with_only_stopwords_returns_nothing():
    index = BM25Index([make_chunk("1", "text oarecare")])
    assert index.search("de la pe") == []
