# Rezultatele evaluării

Set de evaluare: **52 de întrebări** ([data/eval/questions.jsonl](../data/eval/questions.jsonl)), fiecare cu articolul corect verificat manual în textul legii: 47 cu răspuns în legile indexate și 5 fără (pentru a testa refuzul).

> La 47 de întrebări, o întrebare valorează ~2%. Diferențele de câteva puncte procentuale trebuie citite cu prudență.

## 1. Variante de căutare

![Căutare](img/retrieval.png)

| Variantă | Recall@1 | Recall@3 | Recall@5 | MRR | Latență |
|---|---|---|---|---|---|
| BM25 | 49% | 68% | 74% | 0.602 | 0 ms |
| Vectori (bge-m3) | 81% | 91% | 94% | 0.863 | 19 ms |
| Hibrid (RRF) | 64% | 77% | 85% | 0.732 | 20 ms |
| Hibrid ponderat (BM25 × 0.3) | 70% | 85% | 89% | 0.788 | 20 ms |
| Vectori + reranker | 83% | 96% | 98% | 0.894 | 1068 ms |
| Hibrid + reranker | 85% | 96% | 100% | 0.914 | 1091 ms |
| Hibrid + reranker + referințe | 87% | 96% | 100% | 0.924 | 1081 ms |

**Concluzii**

- Căutarea vectorială e mult mai bună decât BM25 la întrebări în limbaj natural.
- Fuziunea simplă BM25 + vectori (RRF) e *mai slabă* decât vectorii singuri: BM25 primește aceeași greutate deși e mult mai slab.
- Cu reranker, BM25 devine util: aduce candidați pe care vectorii nu-i găsesc, iar reranker-ul îi alege pe cei buni. Varianta completă găsește articolul corect în primele 5 rezultate la toate întrebările.
- Reranker-ul costă ~1 s pe întrebare (pe Apple M4 Pro), un compromis acceptabil.

## 2. Modele de embeddings

![Embeddings](img/embeddings.png)

| Model | Parametri | Dimensiune | Recall@1 | Recall@5 | MRR | Indexare |
|---|---|---|---|---|---|---|
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 118M | 384 | 32% | 62% | 0.449 | 2 s |
| intfloat/multilingual-e5-small | 118M | 384 | 66% | 89% | 0.758 | 4 s |
| intfloat/multilingual-e5-base | 278M | 768 | 66% | 85% | 0.733 | 10 s |
| BAAI/bge-m3 | 568M | 1024 | 81% | 94% | 0.863 | 38 s |

**Concluzii:** bge-m3 câștigă clar la Recall@1. Modelul E5-base, deși de 2,4× mai mare, nu bate E5-small. MiniLM, antrenat pe parafraze scurte, e nepotrivit pentru articole de lege.
