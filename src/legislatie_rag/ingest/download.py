"""Descarcă forma consolidată cea mai recentă a fiecărei legi de pe legislatie.just.ro.

Utilizare:
    uv run python -m legislatie_rag.ingest.download
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

import requests
from bs4 import BeautifulSoup

from legislatie_rag.config import LAWS, PORTAL_URL, RAW_DIR, Law

USER_AGENT = "legislatie-rag/0.1 (educational portfolio project)"
REQUEST_DELAY_S = 1.5

_CONSOLIDATION_TITLE = re.compile(r"Consolidarea din (\d{2})\.(\d{2})\.(\d{4})")
_DOC_ID_IN_HREF = re.compile(r"DetaliiDocument(?:Afis)?/(\d+)")


@dataclass
class Consolidation:
    doc_id: int
    date: date


def fetch(doc_id: int, session: requests.Session) -> str:
    response = session.get(PORTAL_URL.format(doc_id=doc_id), timeout=60)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_S)
    return response.text


def parse_consolidations(html: str, current_doc_id: int) -> list[Consolidation]:
    """Extrage istoricul consolidărilor, de la cea mai nouă la cea mai veche.

    Versiunea afișată pe pagină apare fără href (id='fa_selectata'),
    deci primește ID-ul paginii curente.
    """
    soup = BeautifulSoup(html, "lxml")
    history = soup.find(id="istoric_fa")
    if history is None:
        return []

    consolidations = []
    for link in history.find_all("a", title=_CONSOLIDATION_TITLE):
        day, month, year = _CONSOLIDATION_TITLE.search(link["title"]).groups()
        href_match = _DOC_ID_IN_HREF.search(link.get("href", ""))
        doc_id = int(href_match.group(1)) if href_match else current_doc_id
        consolidations.append(Consolidation(doc_id, date(int(year), int(month), int(day))))

    return sorted(consolidations, key=lambda c: c.date, reverse=True)


def count_articles(html: str) -> int:
    return html.count('class="S_ART"')


def download_law(law: Law, session: requests.Session) -> dict:
    html = fetch(law.seed_id, session)
    consolidations = parse_consolidations(html, law.seed_id)

    if consolidations:
        latest = consolidations[0]
        if latest.doc_id != law.seed_id:
            html = fetch(latest.doc_id, session)
        doc_id, consolidation_date = latest.doc_id, latest.date.isoformat()
    else:
        # Acte fără consolidări (ex. Constituția republicată): pagina seed e forma în vigoare.
        doc_id, consolidation_date = law.seed_id, None

    n_articles = count_articles(html)
    if n_articles == 0:
        raise ValueError(f"{law.slug}: niciun articol găsit în documentul {doc_id}")

    (RAW_DIR / f"{law.slug}.html").write_text(html, encoding="utf-8")
    metadata = {
        "slug": law.slug,
        "name": law.name,
        "short_ref": law.short_ref,
        "doc_id": doc_id,
        "url": PORTAL_URL.format(doc_id=doc_id),
        "consolidation_date": consolidation_date,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_articles": n_articles,
    }
    (RAW_DIR / f"{law.slug}.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        for law in LAWS:
            meta = download_law(law, session)
            print(
                f"✓ {meta['short_ref']:<16} {meta['n_articles']:>4} articole  "
                f"consolidare: {meta['consolidation_date'] or '-'}  (doc {meta['doc_id']})"
            )


if __name__ == "__main__":
    main()
