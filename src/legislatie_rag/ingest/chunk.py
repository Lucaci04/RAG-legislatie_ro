"""Împarte articolele în chunk-uri pentru indexare.

Regula: un chunk = un articol. Articolele lungi se împart pe alineate, grupate
până la MAX_CHARS. Fiecare chunk primește un antet cu legea, ierarhia și articolul,
ca embedding-ul să conțină contextul juridic.

Utilizare:
    uv run python -m legislatie_rag.ingest.chunk
"""

import json
from dataclasses import asdict, dataclass

from legislatie_rag.config import CHUNKS_PATH, LAWS, PROCESSED_DIR

# ~800 de tokeni pentru texte în română (≈ 4 caractere / token).
MAX_CHARS = 3200


@dataclass
class Chunk:
    id: str
    law_slug: str
    law_ref: str
    article: str
    part: int
    n_parts: int
    header: str
    body: str
    url: str
    consolidation_date: str | None

    @property
    def text(self) -> str:
        """Textul care se indexează: antet + conținut."""
        return f"{self.header}\n{self.body}"

    @property
    def citation(self) -> str:
        return f"{self.law_ref}, art. {self.article}"


def load_chunks(path=CHUNKS_PATH) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def build_header(article: dict) -> str:
    label = f"Art. {article['number']}"
    if article["title"]:
        label += f" - {article['title']}"
    path = " > ".join([f"{article['law_name']} ({article['law_ref']})", *article["hierarchy"]])
    return f"{path}\n{label}"


def split_long_unit(unit: str, max_chars: int) -> list[str]:
    """Împarte un alineat prea lung pe subdiviziuni (litere, puncte), câte una pe rând.

    Fiecare bucată păstrează fraza introductivă (ex. „(3) Constituie contravenție...:”),
    ca să rămână de sine stătătoare. O singură linie nu se taie niciodată.
    """
    if len(unit) <= max_chars:
        return [unit]
    intro, *items = unit.split("\n")
    if not items:
        return [unit]
    parts: list[str] = []
    for group in group_units(items, max_chars - len(intro) - 1):
        parts.append("\n".join([intro, *group]))
    return parts


def group_units(units: list[str], max_chars: int) -> list[list[str]]:
    """Grupează alineatele consecutive fără să depășească max_chars."""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(unit)
        size += len(unit) + 1
    if current:
        groups.append(current)
    return groups


def chunk_article(article: dict, max_chars: int = MAX_CHARS) -> list[Chunk]:
    if article["abrogated"]:
        return []
    header = build_header(article)
    units = [part for unit in article["units"] for part in split_long_unit(unit, max_chars)]
    groups = group_units(units, max_chars)
    return [
        Chunk(
            id=f"{article['law_slug']}:art{article['number']}:{i}",
            law_slug=article["law_slug"],
            law_ref=article["law_ref"],
            article=article["number"],
            part=i,
            n_parts=len(groups),
            header=header,
            body="\n".join(group),
            url=article["url"],
            consolidation_date=article["consolidation_date"],
        )
        for i, group in enumerate(groups, start=1)
    ]


def main() -> None:
    all_chunks: list[Chunk] = []
    for law in LAWS:
        path = PROCESSED_DIR / f"{law.slug}.articles.jsonl"
        articles = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        chunks = [c for a in articles for c in chunk_article(a)]
        all_chunks.extend(chunks)
        print(f"✓ {law.short_ref:<16} {len(chunks):>4} chunk-uri")

    out = CHUNKS_PATH
    with out.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    sizes = sorted(len(c.text) for c in all_chunks)
    print(
        f"Total: {len(all_chunks)} chunk-uri | caractere: "
        f"median {sizes[len(sizes) // 2]}, max {sizes[-1]}"
    )


if __name__ == "__main__":
    main()
