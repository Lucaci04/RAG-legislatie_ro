# Legislație RO: asistent RAG

Asistent care răspunde la întrebări despre legislația românească pe baza textului oficial de pe [legislatie.just.ro](https://legislatie.just.ro). Fiecare răspuns citează legea, articolul și alineatul.

> 🚧 În dezvoltare. Vezi [PLAN.md](PLAN.md).

## Legi indexate

- Constituția României
- Codul muncii (Legea 53/2003)
- Circulația pe drumurile publice (OUG 195/2002)
- Legea societăților (Legea 31/1990)

## Rulare locală

```bash
uv sync
uv run python -m legislatie_rag.ingest.download   # descarcă forma consolidată a legilor
uv run python -m legislatie_rag.ingest.parse      # HTML → articole structurate
uv run python -m legislatie_rag.ingest.chunk      # articole → chunk-uri pentru indexare
uv run pytest
```

## Notă

Textele actelor normative nu sunt protejate de dreptul de autor (Legea 8/1996, art. 9). Aplicația oferă informații orientative, nu consultanță juridică.
