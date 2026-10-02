import pytest

from legislatie_rag.retrieve.hybrid import reciprocal_rank_fusion
from legislatie_rag.retrieve.refs import ArticleRef, detect_article_refs, detect_laws
from tests.test_text_bm25 import make_chunk


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Ce prevede art. 145 din Codul muncii?", [ArticleRef("codul_muncii", "145")]),
        ("articolul 152^2 din codul muncii", [ArticleRef("codul_muncii", "152^2")]),
        ("art 152 ind. 2 codul muncii", [ArticleRef("codul_muncii", "152^2")]),
        (
            "art. 10 si art. 11 din legea 31/1990",
            [
                ArticleRef("legea_societatilor", "10"),
                ArticleRef("legea_societatilor", "11"),
            ],
        ),
        ("Ce spune articolul 37 din Constituție?", [ArticleRef("constitutia", "37")]),
    ],
)
def test_detect_article_refs(query, expected):
    assert detect_article_refs(query) == expected


def test_article_without_law_is_ambiguous():
    assert detect_article_refs("ce spune art. 1?") == []


def test_article_with_two_laws_is_ambiguous():
    assert detect_article_refs("art. 1 din constitutie si codul muncii") == []


def test_detect_laws_without_diacritics():
    assert detect_laws("Ce spune Constituția și codul rutier?") == ["constitutia", "codul_rutier"]


def test_law_alias_needs_word_boundary():
    assert detect_laws("legea 310/2004") == []


def test_rrf_rewards_chunks_found_by_both_methods():
    a, b, c = make_chunk("a", ""), make_chunk("b", ""), make_chunk("c", "")
    scores = reciprocal_rank_fusion([[a, b], [c, b]])

    assert max(scores, key=scores.get) == "b"
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 62)
