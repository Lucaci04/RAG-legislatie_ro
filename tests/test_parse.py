import json

import pytest

from legislatie_rag.config import LAWS, RAW_DIR
from legislatie_rag.ingest.parse import normalize, parse_law

META = {
    "slug": "test",
    "name": "Legea de test",
    "short_ref": "Legea 1/2000",
    "url": "https://example.test",
    "consolidation_date": "2026-01-01",
}


def aln(number: int, text: str) -> str:
    return (
        f'<span class="S_ALN"><span class="S_ALN_TTL">({number})<!-- --></span> '
        f'<span class="S_ALN_BDY">{text}</span>'
        f'<span class="S_ALN_SHORT">...</span></span>'
    )


def article(number: str, body: str, den: str = "") -> str:
    return (
        f'<span class="S_ART"><span class="TAG_COLLAPSED"> + </span>'
        f'<span class="S_ART_TTL">Articolul {number}</span>'
        f'<span class="S_ART_DEN">{den}</span>'
        f'<span class="S_ART_BDY">{body}</span></span>'
    )


HTML = f"""
<span class="S_TTL"><span class="S_TTL_TTL">Titlul I</span><span class="S_TTL_DEN">Dispoziţii</span>
<span class="S_TTL_BDY">
<span class="S_CAP"><span class="S_CAP_TTL">Capitolul II</span>
<span class="S_CAP_DEN">Concedii</span>
<span class="S_CAP_BDY">
{article("10", aln(1, "Concediul este de 20 de zile.")
    + '<span class="S_PAR">(la 25-01-2015, Alin. (2) a fost modificat de LEGEA nr. 12)</span>'
    + aln(2, 'Vezi <span class="S_LGI">art. 152^2</span>.'))}
{article("11", '<span class="S_PAR">Abrogat.</span>'
    + '<span class="S_PAR">(la 01-02-2014, Art. 11 a fost abrogat de LEGEA nr. 2)</span>')}
{article("12", '<span class="S_PAR">Suveranitatea</span>' + aln(1, "Puterea aparţine poporului."))}
</span></span></span></span>
<span class="S_NTA"><span class="S_NTA_PAR">Notă: Articolul IV din altă lege...</span></span>
"""


@pytest.fixture(scope="module")
def articles():
    return {a.number: a for a in parse_law(HTML, META)}


def test_removes_amendment_notes_and_collapsed_markers(articles):
    text = articles["10"].text
    assert text == "(1) Concediul este de 20 de zile.\n(2) Vezi art. 152^2."
    assert "(la 25-01-2015" not in text
    assert "..." not in text


def test_hierarchy_and_diacritics(articles):
    assert articles["10"].hierarchy == ["Titlul I - Dispoziții", "Capitolul II - Concedii"]


def test_references(articles):
    assert articles["10"].references == ["art. 152^2"]


def test_abrogated_article(articles):
    assert articles["11"].abrogated
    assert not articles["10"].abrogated


def test_marginal_title_extracted_from_first_paragraph(articles):
    assert articles["12"].title == "Suveranitatea"
    assert articles["12"].units == ["(1) Puterea aparține poporului."]


def test_notes_outside_articles_are_ignored(articles):
    assert set(articles) == {"10", "11", "12"}


def test_normalize_cedilla_diacritics():
    assert normalize("Secţiunea şi ŞŢ") == "Secțiunea și ȘȚ"


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.slug)
def test_real_laws_parse_every_article(law):
    html_path = RAW_DIR / f"{law.slug}.html"
    if not html_path.exists():
        pytest.skip("rulează întâi downloader-ul")
    meta = json.loads((RAW_DIR / f"{law.slug}.meta.json").read_text(encoding="utf-8"))
    parsed = parse_law(html_path.read_text(encoding="utf-8"), meta)

    assert len(parsed) == meta["n_articles"]
    assert all(a.units for a in parsed if not a.abrogated)
    assert not any("(la " in u[:4] for a in parsed for u in a.units)
