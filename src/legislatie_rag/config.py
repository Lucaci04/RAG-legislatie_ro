"""Configurație centrală: căi și lista legilor indexate."""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
# Sufixul .nosync exclude indexul din iCloud: sincronizarea unui SQLite în uz îl poate corupe.
INDEX_DIR = DATA_DIR / "index.nosync"

EMBEDDING_MODEL = "BAAI/bge-m3"
CHROMA_COLLECTION = "legislatie"

PORTAL_URL = "https://legislatie.just.ro/Public/DetaliiDocument/{doc_id}"


@dataclass(frozen=True)
class Law:
    slug: str
    name: str
    short_ref: str
    seed_id: int
    """ID-ul unei pagini a legii pe legislatie.just.ro.

    Nu trebuie să fie ultima versiune: downloader-ul urmează istoricul
    consolidărilor și descarcă automat forma cea mai recentă.
    """
    aliases: tuple[str, ...] = ()
    """Cum se referă oamenii la lege într-o întrebare (litere mici, fără diacritice)."""


LAWS: list[Law] = [
    Law(
        "constitutia",
        "Constituția României",
        "Constituția",
        47355,
        aliases=("constitutia", "constitutie", "constitutiei"),
    ),
    Law(
        "codul_muncii",
        "Codul muncii",
        "Legea 53/2003",
        309240,
        aliases=("codul muncii", "codului muncii", "codul de munca", "53/2003", "legea 53"),
    ),
    Law(
        "codul_rutier",
        "Circulația pe drumurile publice",
        "OUG 195/2002",
        84237,
        aliases=("codul rutier", "codului rutier", "195/2002", "oug 195"),
    ),
    Law(
        "legea_societatilor",
        "Legea societăților",
        "Legea 31/1990",
        169688,
        aliases=("legea societatilor", "legii societatilor", "31/1990", "legea 31"),
    ),
]

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
