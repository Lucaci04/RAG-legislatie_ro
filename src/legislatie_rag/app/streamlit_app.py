"""Interfața web: întrebări despre legislație, cu sursele afișate sub fiecare răspuns.

Utilizare:
    uv run streamlit run src/legislatie_rag/app/streamlit_app.py
"""

from html import escape
from pathlib import Path

import streamlit as st

from legislatie_rag.config import LAWS
from legislatie_rag.generate.answer import DISCLAIMER, RAG, Answer
from legislatie_rag.retrieve.hybrid import ArticleHit

EXAMPLES = [
    "Câte zile de concediu de odihnă am minim pe an?",
    "Mă poate concedia angajatorul cât sunt în concediu medical?",
    "Ce pedeapsă primești pentru furt, în funcție de gravitate?",
    "Ce amendă iau dacă trec pe roșu?",
    "Ce vârstă trebuie să ai ca să candidezi la președinție?",
]
LAW_NAMES = {law.slug: law.name for law in LAWS}

st.set_page_config(
    page_title="Legislație RO",
    page_icon=":material/balance:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def format_date(iso: str | None) -> str:
    if not iso:
        return "forma republicată"
    year, month, day = iso.split("-")
    return f"{day}.{month}.{year}"


@st.cache_resource(show_spinner="Se încarcă modelele de căutare…")
def load_rag() -> RAG:
    rag = RAG()
    rag.retriever.search("încălzire")  # prima căutare pe GPU e lentă; o facem la pornire
    return rag


def source_label(n: int, hit: ArticleHit) -> str:
    """„[1]  Codul penal · art. 228 · Furtul”"""
    label_line = hit.chunks[0].header.split("\n")[-1]
    title = label_line.split(" - ", 1)[1] if " - " in label_line else ""
    parts = [LAW_NAMES.get(hit.law_slug, hit.chunks[0].law_ref), f"art. {hit.article}"]
    if title:
        parts.append(title)
    return f"[{n}]  " + " · ".join(parts)


def render_source(n: int, hit: ArticleHit) -> None:
    first = hit.chunks[0]
    with st.expander(source_label(n, hit)):
        crumb = " / ".join(first.header.split("\n")[0].split(" > ")[1:])
        if crumb:
            html(f'<div class="lr-crumb">{escape(crumb)}</div>')
        st.markdown("\n\n".join(c.body.replace("\n", "  \n") for c in hit.chunks))
        html(
            f'<div class="lr-source-foot">{escape(first.law_ref)} · versiune consolidată '
            f"{format_date(first.consolidation_date)} · "
            f'<a href="{escape(first.url)}" target="_blank">text oficial</a></div>'
        )


def render_answer(answer: Answer) -> None:
    if answer.llm_error:
        html(f'<div class="lr-notice">{escape(answer.text)}</div>')
    else:
        st.markdown(answer.text)

    cited = set(answer.cited)
    numbers = range(1, len(answer.sources) + 1)
    primary = [n for n in numbers if n in cited or answer.llm_error]
    others = [n for n in numbers if n not in primary]

    if primary:
        html('<div class="lr-label">Surse</div>')
        for n in primary:
            render_source(n, answer.sources[n - 1])
    if others:
        with st.expander(f"Alte articole găsite ({len(others)})"):
            for n in others:
                html(
                    f'<div class="lr-other">{escape(source_label(n, answer.sources[n - 1]))}</div>'
                )

    meta = [f"Căutare {answer.retrieval_ms / 1000:.1f} s"]
    if answer.model:
        meta += [f"generare {answer.generation_ms / 1000:.1f} s", answer.model]
    if answer.invalid_citations:
        meta.append(f"citări fără sursă: {answer.invalid_citations}")
    html(f'<div class="lr-meta">{escape(" · ".join(meta))}</div>')


def sidebar(rag: RAG) -> list[str]:
    versions: dict[str, tuple[str | None, str]] = {}
    for chunk in rag.retriever.chunks:
        versions.setdefault(chunk.law_slug, (chunk.consolidation_date, chunk.url))

    with st.sidebar:
        html('<div class="lr-overline">Asistent juridic</div>')
        html('<div class="lr-side-title">Legislație RO</div>')
        html(
            '<div class="lr-side-text">Răspunsuri formulate exclusiv pe baza textului oficial '
            "al legilor, cu trimitere la articol.</div>"
        )
        html('<div class="lr-label">Acte indexate</div>')
        for law in LAWS:
            date, url = versions.get(law.slug, (None, ""))
            html(
                f'<div class="lr-law"><a href="{escape(url)}" target="_blank">'
                f"{escape(law.name)}</a>"
                f"<span>{escape(law.short_ref)} · {format_date(date)}</span></div>"
            )
        selected = st.multiselect(
            "Restrânge căutarea",
            options=[law.slug for law in LAWS],
            format_func=LAW_NAMES.get,
            placeholder="Toate actele",
        )
        if st.button("Conversație nouă", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        html(
            f'<div class="lr-disclaimer">{DISCLAIMER} '
            "Fiecare întrebare este tratată independent.</div>"
        )
    return selected


def main() -> None:
    rag = load_rag()
    laws = sidebar(rag)

    html('<div class="lr-overline">Legislația României · text oficial</div>')
    html('<h1 class="lr-title">Întreabă legea</h1>')
    html(
        '<div class="lr-lead">Pune o întrebare în limbaj obișnuit. Răspunsul citează articolele '
        "de lege pe care se bazează, iar textul lor complet este disponibil sub răspuns.</div>"
    )

    # Întrebarea nouă se citește înainte de afișare, ca exemplele să dispară imediat.
    # Câmpul rămâne fixat jos indiferent de poziția apelului.
    question = st.chat_input("Scrie o întrebare despre legislație")
    question = question or st.session_state.pop("pending", None)

    messages = st.session_state.setdefault("messages", [])
    if not messages and not question:
        html('<div class="lr-label">Exemple</div>')
        with st.container(key="examples"):
            for example in EXAMPLES:
                if st.button(example, use_container_width=True):
                    st.session_state.pending = example
                    st.rerun()

    for message in messages:
        html(f'<div class="lr-question">{escape(message["question"])}</div>')
        render_answer(message["answer"])

    if not question:
        return

    html(f'<div class="lr-question">{escape(question)}</div>')
    with st.spinner("Se caută în legi și se formulează răspunsul…"):
        answer = rag.ask(question, laws=laws or None)
    render_answer(answer)
    messages.append({"question": question, "answer": answer})


main()
