from legislatie_rag.ingest.chunk import chunk_article, group_units, split_long_unit

ARTICLE = {
    "law_slug": "codul_muncii",
    "law_name": "Codul muncii",
    "law_ref": "Legea 53/2003",
    "number": "145",
    "title": "",
    "hierarchy": ["Titlul III - Timpul de muncă", "Capitolul III - Concediile"],
    "units": ["(1) Durata minimă este de 20 de zile.", "(2) Durata efectivă se stabilește."],
    "url": "https://legislatie.just.ro/Public/DetaliiDocument/309240",
    "consolidation_date": "2026-04-27",
    "abrogated": False,
}


def test_short_article_is_one_chunk_with_context_header():
    [chunk] = chunk_article(ARTICLE)

    assert chunk.id == "codul_muncii:art145:1"
    assert chunk.header == (
        "Codul muncii (Legea 53/2003) > Titlul III - Timpul de muncă > Capitolul III - Concediile\n"
        "Art. 145"
    )
    assert chunk.text.endswith("(2) Durata efectivă se stabilește.")
    assert chunk.citation == "Legea 53/2003, art. 145"


def test_long_article_split_on_unit_boundaries():
    chunks = chunk_article(ARTICLE, max_chars=45)

    assert [c.body for c in chunks] == ARTICLE["units"]
    assert [c.part for c in chunks] == [1, 2]
    assert all(c.n_parts == 2 for c in chunks)


def test_abrogated_article_has_no_chunks():
    assert chunk_article({**ARTICLE, "abrogated": True}) == []


def test_group_units_never_splits_a_unit():
    assert group_units(["a" * 10, "b" * 10, "c" * 10], max_chars=15) == [
        ["a" * 10],
        ["b" * 10],
        ["c" * 10],
    ]


def test_split_long_unit_repeats_intro_line():
    unit = "(3) Constituie contravenție:\na) fapta A;\nb) fapta B;\nc) fapta C."
    parts = split_long_unit(unit, max_chars=45)

    assert len(parts) > 1
    assert all(p.startswith("(3) Constituie contravenție:\n") for p in parts)
    assert "\n".join(p.split("\n", 1)[1] for p in parts) == "a) fapta A;\nb) fapta B;\nc) fapta C."
