"""Transformă HTML-ul de pe legislatie.just.ro în articole structurate.

Utilizare:
    uv run python -m legislatie_rag.ingest.parse
"""

import json
import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup, Comment, Tag

from legislatie_rag.config import LAWS, PROCESSED_DIR, RAW_DIR

# Unități de structură care conțin articole, de la cea mai generală la cea mai specifică.
HIERARCHY_CLASSES = ["S_ANX", "S_TTL", "S_CAP", "S_SEC", "S_PRG"]
# Subdiviziunile unui articol: alineat, literă, punct, liniuță, paragraf liber.
BLOCK_CLASSES = ["S_ALN", "S_LIT", "S_PCT", "S_LIN", "S_PAR"]

_AMENDMENT_NOTE = re.compile(r"^\(la (data de )?\d{2}-\d{2}-\d{4}")
_ARTICLE_TITLE = re.compile(r"^Articolul\s+(.+)$")
_ABROGATED = re.compile(r"^(\(\d+(\^\d+)?\)\s*|[a-z](\^\d+)?\)\s*|\d+\.\s*)?Abrogat[ăe]?\.?$")
_WHITESPACE = re.compile(r"[ \t\r\f\v\xa0]+")
# Diacritice cu sedilă (ş, ţ), folosite inconsecvent pe portal → forma corectă cu virgulă.
_DIACRITICS = str.maketrans("şţŞŢ", "șțȘȚ")


def normalize(text: str) -> str:
    return text.translate(_DIACRITICS)


def _looks_like_marginal_title(text: str) -> bool:
    """Denumiri marginale precum „Statul român” (Constituția) apar ca primul paragraf al
    articolului: scurte și fără punctuație la final."""
    return len(text) < 120 and "\n" not in text and not text.endswith((".", ":", ";", ","))


@dataclass
class Article:
    law_slug: str
    law_name: str
    law_ref: str
    number: str
    title: str
    hierarchy: list[str]
    units: list[str]
    """Subdiviziunile de prim nivel ale articolului (alineate sau paragrafe), ca text curat."""
    url: str
    consolidation_date: str | None
    abrogated: bool = False
    references: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.units)


def _classes(tag: Tag) -> list[str]:
    return tag.get("class") or []


def _has_class(tag: Tag, names: list[str]) -> bool:
    return any(c in names for c in _classes(tag))


def remove_noise(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.select('[class$="_SHORT"], .S_NTA, .TAG_COLLAPSED'):
        tag.decompose()
    for par in soup.select(".S_PAR"):
        if _AMENDMENT_NOTE.match(par.get_text(strip=True)):
            par.decompose()


def render(tag: Tag) -> str:
    """Text curat pentru un element: fiecare subdiviziune pe rândul ei, ex. „(1) ...”, „a) ...”."""
    parts: list[str] = []

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Tag):
                if _has_class(child, BLOCK_CLASSES):
                    parts.append("\n")
                walk(child)
                if any(c.endswith("_TTL") for c in _classes(child)):
                    parts.append(" ")
            else:
                parts.append(str(child))

    walk(tag)
    lines = (_WHITESPACE.sub(" ", line).strip() for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _heading(container: Tag, cls: str) -> str:
    ttl = container.find(class_=f"{cls}_TTL", recursive=False)
    den = container.find(class_=f"{cls}_DEN", recursive=False)
    ttl_text = ttl.get_text(" ", strip=True) if ttl else ""
    den_text = den.get_text(" ", strip=True) if den else ""
    return " - ".join(p for p in (ttl_text, den_text) if p)


def hierarchy_of(article: Tag) -> list[str]:
    path = []
    for parent in article.find_parents():
        for cls in HIERARCHY_CLASSES:
            if cls in _classes(parent):
                path.append(_heading(parent, cls))
    return [p for p in reversed(path) if p]


def parse_article(tag: Tag, meta: dict) -> Article:
    title_text = tag.find(class_="S_ART_TTL").get_text(" ", strip=True)
    match = _ARTICLE_TITLE.match(title_text)
    number = match.group(1).strip() if match else title_text
    denomination = tag.find(class_="S_ART_DEN")
    body = tag.find(class_="S_ART_BDY")

    blocks = [c for c in body.children if isinstance(c, Tag) and _has_class(c, BLOCK_CLASSES)]
    if blocks:
        units = [render(b) for b in blocks]
        # Text liber direct în corpul articolului (fără alineat), dacă există.
        loose = "".join(str(c) for c in body.children if not isinstance(c, Tag)).strip()
        if loose:
            units.insert(0, _WHITESPACE.sub(" ", loose))
    else:
        units = [render(body)]
    units = [normalize(u) for u in units if u]

    title = normalize(denomination.get_text(" ", strip=True)) if denomination else ""
    if not title and len(units) > 1 and _looks_like_marginal_title(units[0]):
        title = units.pop(0)

    abrogated = not units or all(_ABROGATED.match(u) for u in units)
    if not abrogated:
        units = [u for u in units if not _ABROGATED.match(u)]

    references = sorted({lgi.get_text(" ", strip=True) for lgi in body.select(".S_LGI")})

    return Article(
        law_slug=meta["slug"],
        law_name=meta["name"],
        law_ref=meta["short_ref"],
        number=number,
        title=title,
        hierarchy=[normalize(h) for h in hierarchy_of(tag)],
        units=units,
        url=meta["url"],
        consolidation_date=meta["consolidation_date"],
        abrogated=abrogated,
        references=references,
    )


def parse_law(html: str, meta: dict) -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    remove_noise(soup)
    return [parse_article(tag, meta) for tag in soup.select(".S_ART")]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for law in LAWS:
        meta = json.loads((RAW_DIR / f"{law.slug}.meta.json").read_text(encoding="utf-8"))
        html = (RAW_DIR / f"{law.slug}.html").read_text(encoding="utf-8")
        articles = parse_law(html, meta)

        out = PROCESSED_DIR / f"{law.slug}.articles.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

        active = sum(not a.abrogated for a in articles)
        print(f"✓ {law.short_ref:<16} {len(articles):>4} articole ({active} în vigoare)")


if __name__ == "__main__":
    main()
