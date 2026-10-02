from datetime import date

from legislatie_rag.ingest.download import count_articles, parse_consolidations

HISTORY_HTML = """
<div id="istoric_fa">
  <a title='Consolidarea din 27.04.2026' id='fa_selectata'>27.04.2026</a><br/>
  <a title='Consolidarea din 18.12.2025' href='/Public/DetaliiDocument/304539'>18.12.2025</a>
  <a title='Consolidarea din 03.06.2024' href='/Public/DetaliiDocument/282881'>03.06.2024</a>
</div>
"""


def test_parse_consolidations_newest_first_and_current_page_id():
    result = parse_consolidations(HISTORY_HTML, current_doc_id=309240)

    assert [c.doc_id for c in result] == [309240, 304539, 282881]
    assert result[0].date == date(2026, 4, 27)


def test_parse_consolidations_without_history():
    assert parse_consolidations("<html><body></body></html>", current_doc_id=1) == []


def test_count_articles():
    html = '<p class="S_ART">a</p><p class="S_ART_TTL">x</p><p class="S_ART">b</p>'
    assert count_articles(html) == 2
