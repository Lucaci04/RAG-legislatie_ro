# Legislație RO: asistent RAG

Asistent care răspunde la întrebări despre legislația românească pe baza textului oficial de pe [legislatie.just.ro](https://legislatie.just.ro). Fiecare răspuns citează legea, articolul și alineatul.

> 🚧 În dezvoltare. Vezi [PLAN.md](PLAN.md).

## Legi indexate

- Constituția României
- Codul muncii (Legea 53/2003)
- Circulația pe drumurile publice (OUG 195/2002)
- Legea societăților (Legea 31/1990)
- Codul penal (Legea 286/2009)
- Regimul juridic al contravențiilor (OG 2/2001)

## Rulare locală

```bash
cp .env.example .env    # apoi completează GEMINI_API_KEY (gratuit pe aistudio.google.com)
uv sync
uv run python -m legislatie_rag.ingest.download   # descarcă forma consolidată a legilor
uv run python -m legislatie_rag.ingest.parse      # HTML → articole structurate
uv run python -m legislatie_rag.ingest.chunk      # articole → chunk-uri pentru indexare
uv run python -m legislatie_rag.index.vector      # embeddings bge-m3 → Chroma (incremental)
uv run python -m legislatie_rag.retrieve.hybrid "Câte zile de concediu am?"   # doar căutare
uv run python -m legislatie_rag.generate.answer "Câte zile de concediu am?"   # răspuns complet
uv run streamlit run src/legislatie_rag/app/streamlit_app.py                  # interfața web
uv run pytest
```

## Notă

Textele actelor normative nu sunt protejate de dreptul de autor (Legea 8/1996, art. 9). Aplicația oferă informații orientative, nu consultanță juridică.
