"""Index vectorial persistent în Chroma.

Utilizare (construire / actualizare index):
    uv run python -m legislatie_rag.index.vector
"""

import hashlib
import time

import chromadb

from legislatie_rag.config import CHROMA_COLLECTION, EMBEDDING_MODEL, INDEX_DIR
from legislatie_rag.index.embed import embed
from legislatie_rag.ingest.chunk import Chunk, load_chunks


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    return client.get_or_create_collection(
        CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


class VectorIndex:
    def __init__(self, chunks: list[Chunk]):
        self._by_id = {c.id: c for c in chunks}
        self._collection = get_collection()

    def search(
        self, query: str, k: int = 30, laws: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        result = self._collection.query(
            query_embeddings=embed([query]).tolist(),
            n_results=k,
            where={"law_slug": {"$in": laws}} if laws else None,
            include=["distances"],
        )
        return [
            (self._by_id[chunk_id], 1.0 - distance)
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
            if chunk_id in self._by_id
        ]


def sync_index(chunks: list[Chunk], batch_size: int = 64) -> None:
    """Aduce colecția Chroma la zi: re-embedează doar chunk-urile noi sau modificate."""
    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    existing_hash = {
        id_: meta.get("hash")
        for id_, meta in zip(existing["ids"], existing["metadatas"], strict=True)
    }

    wanted = {c.id: c for c in chunks}
    stale = [id_ for id_ in existing_hash if id_ not in wanted]
    todo = [c for c in chunks if existing_hash.get(c.id) != content_hash(c)]

    if stale:
        collection.delete(ids=stale)
    print(
        f"Index: {len(existing_hash)} existente, {len(todo)} de (re)calculat, {len(stale)} șterse"
    )

    start = time.perf_counter()
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        collection.upsert(
            ids=[c.id for c in batch],
            embeddings=embed([c.text for c in batch]).tolist(),
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "law_slug": c.law_slug,
                    "law_ref": c.law_ref,
                    "article": c.article,
                    "part": c.part,
                    "hash": content_hash(c),
                }
                for c in batch
            ],
        )
        print(f"  {min(i + batch_size, len(todo))}/{len(todo)}", end="\r", flush=True)

    if todo:
        print(f"\n✓ {len(todo)} embeddings în {time.perf_counter() - start:.0f}s")
    print(f"Total în index: {collection.count()}")


if __name__ == "__main__":
    sync_index(load_chunks())
