# Plan de dezvoltare: RAG pe legislația românească

## Obiectiv

Un asistent care răspunde la întrebări despre legile din România **pe baza textului oficial**. Fiecare răspuns citează sursa exactă (lege, articol, alineat) și spune clar „nu știu” când informația lipsește din documente.

Exemplu:
> **Î:** Câte zile de concediu de odihnă am minim pe an?
> **R:** Durata minimă a concediului de odihnă anual este de 20 de zile lucrătoare. *(Legea 53/2003, Codul Muncii, art. 145 alin. 1)*

---

## Scop: proiect de portofoliu

Proiectul trebuie să arate unui recrutor sau tech lead că știu să construiesc un sistem RAG **serios**, nu un tutorial copiat. Ce contează:

1. **Demo live**: link public pe Hugging Face Spaces (gratuit), ca să fie testat în 10 secunde
2. **README excelent**: GIF demo, diagramă de arhitectură, cum rulezi local, decizii tehnice
3. **Rezultate măsurate**: tabel cu Recall@5, MRR și acuratețe pentru fiecare variantă (doar vectori → hibrid → hibrid + reranker). Asta diferențiază proiectul de 90% dintre RAG-urile din portofolii
4. **Cod curat**: structură clară, type hints, teste, `ruff`, un CI simplu pe GitHub Actions
5. **Istoric git curat**: commit-uri mici, cu mesaje clare
6. **Cost zero pentru vizitatori**: LLM configurabil (Claude sau un model gratuit), embeddings precalculate

Fără autentificare, bază de date de producție sau scalare: nu aduc valoare pentru portofoliu.

---

## Decizii tehnice (implicite, se pot schimba)

| Componentă | Alegere | De ce |
|---|---|---|
| Limbaj | Python 3.11+ | Ecosistemul standard pentru RAG |
| Sursă date | legislatie.just.ro (forma consolidată) | Oficială, gratuită, publică |
| Chunking | Pe articol, împărțit pe alineate dacă e lung | Legile au deja o structură naturală |
| Embeddings | `BAAI/bge-m3` (local) | Multilingv, funcționează bine pe română, gratuit |
| Vector DB | Chroma (local) | Simplu, fără server |
| Căutare cuvinte cheie | BM25 (`rank_bm25`) | Esențial pentru „art. 145”, termeni juridici exacți |
| Reranker | `BAAI/bge-reranker-v2-m3` | Crește mult precizia, rulează local |
| LLM | Claude (API Anthropic) | Calitate bună în română, respectă instrucțiunile de citare |
| Interfață | CLI, apoi Streamlit | Iterare rapidă, apoi demo vizual |

---

## Structura proiectului

```
proiect_1/
├── data/
│   ├── raw/            # HTML descărcat, neatins
│   ├── processed/      # JSON cu articole + metadate
│   └── eval/           # set de întrebări de test
├── src/legislatie_rag/
│   ├── config.py       # căi + lista legilor
│   ├── ingest/         # descărcare + parsare
│   ├── index/          # chunking, embeddings, Chroma, BM25
│   ├── retrieve/       # căutare hibridă + reranking
│   ├── generate/       # prompt + apel LLM
│   └── app/            # CLI și Streamlit
├── eval/               # scripturi de evaluare
├── tests/
├── .env                # ANTHROPIC_API_KEY (nu se pune în git)
├── pyproject.toml
└── PLAN.md
```

---

## Faze

### Faza 0: Setup *(~0,5 zile)*
- [x] `git init`, `.gitignore` (inclusiv `.env`, `data/raw`, modele)
- [x] Mediu virtual (`uv` sau `venv`) și dependențe
- [ ] Cheie API Anthropic în `.env`

### Faza 1: Colectarea datelor *(~1 zi)*
Pornim cu legi mici și foarte folosite:
1. **Constituția României**
2. **Legea 53/2003**: Codul Muncii
3. **OUG 195/2002**: Codul Rutier
4. **Legea 31/1990**: Legea societăților

Mai târziu: Codul Civil și Codul Fiscal (mari, cu structură complexă).

- [x] Script care descarcă forma consolidată și salvează HTML-ul brut, plus metadate (URL, data consolidării, data descărcării)
- [x] Rate limiting de 1–2 secunde între cereri, user-agent identificabil

### Faza 2: Parsare și chunking *(~2 zile)*
- [x] Extragerea ierarhiei: Titlu → Capitol → Secțiune → Articol → Alineat
- [x] Un chunk pe articol. Dacă articolul depășește ~800 de tokeni, se împarte pe alineate
- [x] Fiecare chunk primește ca prefix contextul ierarhic, ca embedding-ul să „știe” unde se află:
  `Codul Muncii > Titlul III > Capitolul II > Art. 145`
- [x] Metadate: `lege`, `articol`, `alineat`, `capitol`, `data_versiunii`, `url`
- [x] Eliminarea zgomotului: note de subsol cu modificări, „(la data ... a fost modificat de ...)”
- [x] Teste: numărul de articole extrase corespunde realității pentru fiecare lege

### Faza 3: Indexare *(~1 zi)*
- [x] Embeddings cu `bge-m3` și salvare în Chroma
- [x] Index BM25 pe același text
- [x] Script `index.py` reluabil (re-indexează doar ce s-a schimbat)

### Faza 4: Retrieval *(~2 zile)*
- [x] Căutare vectorială (top 30) și BM25 (top 30)
- [x] Combinare prin **Reciprocal Rank Fusion**
- [x] **Shortcut pentru referințe explicite**: „art. 145 din Codul Muncii” → căutare directă după metadate
- [x] Reranking și păstrarea top 5–8
- [x] Filtrare opțională după lege (dacă utilizatorul o specifică)

### Faza 5: Generare *(~1 zi)*
- [ ] Prompt de sistem: răspunde **doar** din context, citează fiecare afirmație, spune „nu am găsit” când e cazul
- [ ] Format de citare uniform: *(Legea X/AAAA, art. N alin. M)*
- [ ] Afișarea surselor folosite, cu link către legislatie.just.ro
- [ ] Disclaimer: „Informație orientativă, nu consultanță juridică”

### Faza 6: Evaluare *(~2 zile, apoi continuu)*
- [ ] Set de **40–50 de întrebări** scrise manual, cu articolul corect pentru fiecare:
  - întrebări directe („Ce prevede art. 41 din Codul Muncii?”)
  - întrebări în limbaj natural („Mă poate concedia angajatorul cât sunt în concediu medical?”)
  - întrebări fără răspuns în date (pentru a testa refuzul)
- [ ] Metrici de retrieval: **Recall@5** și **MRR** (s-a găsit articolul corect?)
- [ ] Metrici de răspuns: corectitudine și citare corectă (LLM ca judecător, plus verificare manuală pe un eșantion)
- [ ] Un singur script `eval/run.py` care produce un raport, rulat după fiecare schimbare

### Faza 7: Interfață *(~1–2 zile)*
- [ ] CLI: `python -m src.app.cli "întrebarea mea"`
- [ ] Streamlit: chat, surse expandabile, filtru pe lege

### Faza 8: Îmbunătățiri *(după MVP, în funcție de rezultatele evaluării)*
- [ ] Reformularea întrebării (query rewriting) pentru întrebări vagi
- [ ] Conversație cu memorie (întrebări de follow-up)
- [ ] Versionare: actualizarea automată când se modifică o lege
- [ ] Rezolvarea trimiterilor („potrivit art. 52 alin. (1) lit. d)”) prin includerea articolului referit în context
- [ ] Adăugarea Codului Civil și a Codului Fiscal

---

## Calendar orientativ

| Săptămâna | Livrabil |
|---|---|
| 1 | Fazele 0–3: date descărcate, parsate, indexate |
| 2 | Fazele 4–5: răspunsuri cu citări în CLI |
| 3 | Fazele 6–7: evaluare și interfață Streamlit |
| 4 | Portofoliu: deploy pe HF Spaces, README cu GIF și diagramă, tabel cu rezultate, CI |
| 5+ | Faza 8: îmbunătățiri ghidate de metrici (opțional) |

---

## Riscuri și cum le gestionăm

| Risc | Măsură |
|---|---|
| HTML-ul de pe legislatie.just.ro e inconsistent | Teste pe numărul de articole; parsare robustă cu `BeautifulSoup` |
| Legile se modifică, iar datele devin vechi | Salvăm data consolidării și o afișăm în răspuns |
| Modelul inventează articole | Citări verificate automat: articolul citat trebuie să existe în context |
| Răspunsurile pot fi luate drept consultanță juridică | Disclaimer vizibil și fără formulări categorice |

---

## Notă legală

Conform **Legii 8/1996, art. 9**, textele oficiale (legi, hotărâri, decizii) **nu sunt protejate de drepturi de autor**, deci pot fi folosite liber. Atenție la:
- Folosește **portalul oficial** (legislatie.just.ro), nu baze comerciale (Lege5, Indaco, Sintact). Acolo, forma consolidată și adnotările pot fi protejate ca bază de date.
- Descarcă **politicos** (rate limiting) și respectă termenii site-ului.
- Aplicația oferă informații orientative, **nu consultanță juridică**.
