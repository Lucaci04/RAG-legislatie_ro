# Intreaba legea: un sistem RAG pe legislatia romaneasca

Pui o intrebare in limbaj normal, de exemplu *"ma poate concedia seful cand sunt in concediu medical?"*, si primesti un raspuns bazat strict pe textul oficial al legii, cu trimitere la articol si alineat. Daca legea nu spune nimic despre ce ai intrebat, sistemul iti spune asta in loc sa inventeze.

![Rezultatele cautarii](docs/img/retrieval.png)

## De ce am facut proiectul asta

Am citit destul despre RAG (Retrieval-Augmented Generation) ca sa stiu cum arata diagrama: documente, embeddings, baza vectoriala, LLM. Dar simteam ca nu inteleg cu adevarat ce se intampla intre sagetile alea. De ce un chunk e mai bun decat altul? Cand ajuta cautarea pe cuvinte cheie si cand incurca? Cum stii daca sistemul chiar functioneaza sau doar pare ca functioneaza?

Asa ca mi-am propus sa construiesc unul de la zero, fara LangChain sau alte framework-uri care ascund pasii, si sa masor fiecare decizie in loc sa o iau dupa tutoriale.

Am ales legislatia romaneasca din trei motive:

- **Textele sunt publice si gratuite.** Actele normative nu sunt protejate de drepturi de autor (Legea 8/1996, art. 9), iar portalul legislatie.just.ro are forma consolidata la zi.
- **Raspunsurile se pot verifica.** Fie ai gasit articolul corect, fie nu. Asta face evaluarea mult mai onesta decat la un chatbot generic.
- **E o problema reala.** Oricine a cautat vreodata cate zile de concediu are sau ce amenda ia pentru o contraventie stie ca textul legii nu e prea prietenos.

## Ce stie sa faca

- Raspunde pe baza a **6 acte normative**, in forma consolidata cea mai recenta:
  - Constitutia Romaniei
  - Codul muncii (Legea 53/2003)
  - Codul rutier (OUG 195/2002)
  - Legea societatilor (Legea 31/1990)
  - Codul penal (Legea 286/2009)
  - Regimul juridic al contraventiilor (OG 2/2001)
- **Citeaza fiecare afirmatie** si iti arata textul complet al articolului, cu link la sursa oficiala.
- La intrebari despre **pedepse si sanctiuni**, le ordoneaza dupa gravitate: furt (6 luni - 3 ani sau amenda), furt calificat (1-5, 2-7 sau 3-10 ani), talharie si asa mai departe.
- Intelege intrebari scrise **fara diacritice** si informal ("cat concediu am pe an").
- Recunoaste referintele directe ("ce spune art. 41 din codul muncii") si merge direct la articol.
- **Refuza** cand informatia nu e in legile indexate, in loc sa ghiceasca.

## Cum functioneaza

```
legislatie.just.ro
      |
      v
 Descarcare ---> Parsare HTML ---> Chunking pe articole ---> Indexare
 (ultima          (titlu > capitol    (un articol = un chunk,    - vectori bge-m3 (Chroma)
  consolidare)     > articol >         cele lungi se impart       - BM25 pentru romana
                   alineat)            pe alineate)
                                                                      |
 Intrebare --> referinte explicite? --> BM25 + vectori --> RRF --> reranker --> top 5 articole
                                                                                    |
                                                                                    v
                                                          Gemini: raspuns doar din surse, cu citari
```

Cateva decizii de care sunt multumit:

- **Chunking pe articol, nu pe numar de caractere.** Legile au deja o structura naturala. Fiecare chunk primeste si "calea" in lege (de ex. `Codul muncii > Titlul III > Capitolul III > Art. 145`), ca embedding-ul sa stie unde se afla textul.
- **Curatarea textului.** Paginile oficiale sunt pline de note de tipul "(la 22-10-2022, ... a fost modificat de ...)". Le-am scos, plus articolele si literele abrogate. Doar asta a urcat articolul corect pe primul loc de la 87,2% la 89,4% din cazuri.
- **BM25 adaptat pentru romana:** fara diacritice (oamenii scriu "odihna", legea scrie cu diacritice), cu stemming si fara cuvinte de legatura.
- **Totul ruleaza local, mai putin LLM-ul.** Embeddings si reranker pe GPU-ul Mac-ului, gratuit. Doar formularea raspunsului merge la Gemini.

## Ce am invatat (cu cifre)

Am scris un set de **59 de intrebari de test**, fiecare cu articolul corect verificat de mana in textul legii: intrebari normale, informale, fara diacritice, capcane si intrebari la care legile indexate nu au raspuns. Apoi am masurat fiecare varianta.

| Varianta de cautare | Articolul corect pe locul 1 | In primele 5 |
|---|---|---|
| BM25 (cuvinte cheie) | 51,9% | 75,9% |
| Vectori (bge-m3) | 83,3% | 94,4% |
| Hibrid simplu (RRF) | 66,7% | 85,2% |
| Vectori + reranker | 87,0% | 98,1% |
| Hibrid + reranker | 88,9% | 100% |
| **Hibrid + reranker + referinte explicite** | **90,7%** | **100%** |

Lucrurile care m-au surprins:

1. **Cautarea hibrida "clasica" a iesit mai proasta decat vectorii singuri.** Peste tot citesti ca hibrid e mai bun. La mine, BM25 era atat de slab la intrebari formulate natural incat, pus la egalitate cu vectorii, ii tragea in jos. Exemplul meu preferat: la *"ce amenda iau daca trec pe rosu"*, BM25 gasea articolul din Constitutie despre **drapel** (rosu, galben, albastru).
2. **BM25 conteaza, dar abia impreuna cu reranker-ul.** Aduce candidati pe care vectorii nu ii gasesc, iar reranker-ul ii alege pe cei buni. Fara reranker strica, cu reranker ajunge la 100% in top 5. Fara sa masor, as fi ajuns la concluzia gresita in oricare dintre directii.
3. **Modelul de embeddings face cea mai mare diferenta.** bge-m3 a gasit articolul corect pe locul 1 in 83% din cazuri. Urmatorul model testat, de 2 ori mai mic, a ajuns la 67%.

![Modele de embeddings](docs/img/embeddings.png)

Rezultatele complete sunt in [docs/results.md](docs/results.md).

Si cateva lucruri invatate pe pielea mea, care nu apar in tutoriale:

- **Un LLM bun poate repara un retrieval imperfect, dar si invers.** La *"care e capitalul social minim pentru un SRL?"* cautarea aduce art. 10 (90.000 lei), care e despre societatile pe actiuni. Modelul a observat singur ca articolul nu se aplica la SRL si nu a raspuns gresit. Am pastrat intrebarea in setul de test, ca sa prind regresii.
- **Tier-ul gratuit Gemini are 20 de cereri pe zi pe model.** Am aflat in mijlocul evaluarii. De atunci aplicatia trece automat pe un model de rezerva cand limita e atinsa, iar daca nici acela nu raspunde, iti arata macar articolele gasite.
- **iCloud si mediile virtuale Python nu se inteleg.** Desktop-ul meu se sincronizeaza cu iCloud, care marca fisierele din `.venv` ca ascunse, iar Python le ignora. Solutia: orice folder terminat in `.nosync` e ignorat de iCloud.

## Stack

- **Python 3.12**, gestionat cu [uv](https://github.com/astral-sh/uv)
- **Embeddings:** BAAI/bge-m3, rulat local (MPS pe Apple Silicon)
- **Baza vectoriala:** Chroma, cu indexare incrementala (recalculeaza doar articolele modificate)
- **Cautare lexicala:** BM25 cu stemmer Snowball pentru romana
- **Reranker:** BAAI/bge-reranker-v2-m3
- **LLM:** Google Gemini, cu model de rezerva si reincercari automate
- **Interfata:** Streamlit
- **Calitate:** pytest (53 de teste), ruff

## Cum il rulezi

Ai nevoie de Python 3.12+, [uv](https://github.com/astral-sh/uv) si o cheie Gemini gratuita de pe [aistudio.google.com](https://aistudio.google.com).

```bash
git clone https://github.com/Lucaci04/RAG-legislatie_ro.git
cd RAG-legislatie_ro
cp .env.example .env        # pune cheia Gemini in .env
uv sync
```

Construiesti datele si indexul (prima data descarca si modelele, cam 4,6 GB):

```bash
uv run python -m legislatie_rag.ingest.download
uv run python -m legislatie_rag.ingest.parse
uv run python -m legislatie_rag.ingest.chunk
uv run python -m legislatie_rag.index.vector
```

Pornesti aplicatia:

```bash
uv run streamlit run src/legislatie_rag/app/streamlit_app.py
```

Sau intrebi direct din terminal:

```bash
uv run python -m legislatie_rag.generate.answer "cat concediu am pe an"
```

Evaluarea si testele:

```bash
uv run python -m legislatie_rag.eval.retrieval
uv run python -m legislatie_rag.eval.embeddings
uv run pytest
```

## Structura

```
src/legislatie_rag/
  ingest/     descarcare, parsare HTML, chunking
  index/      embeddings, Chroma, BM25, tokenizare pentru romana
  retrieve/   cautare hibrida, referinte explicite, reranker
  generate/   prompt, apelul la Gemini, verificarea citarilor
  eval/       setul de test, metrici, raport cu grafice
  app/        interfata Streamlit
data/eval/    cele 59 de intrebari de test si rezultatele
docs/         raportul de evaluare
```

## Limite si ce urmeaza

- Fiecare intrebare e tratata independent, fara memoria conversatiei.
- Doar 6 acte normative. Codul civil si Codul fiscal sunt urmatoarele pe lista.
- Comparatia intre modelele Gemini e facuta pe jumatate: scriptul exista, dar limita gratuita de 20 de cereri pe zi a oprit-o. Il termin cand am rabdare sau un buget mic.
- Trimiterile dintre articole ("potrivit art. 52 alin. (1) lit. d)") sunt extrase, dar inca nu le folosesc ca sa aduc automat si articolul referit.

## Disclaimer

Aplicatia ofera informatii orientative pe baza textului legii, nu consultanta juridica. Pentru o situatie concreta, vorbeste cu un avocat.

## Licenta

Codul e sub licenta [MIT](LICENSE). Textele legilor sunt preluate de pe portalul oficial [legislatie.just.ro](https://legislatie.just.ro).
