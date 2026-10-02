"""Interfața web: chat cu legislația, cu sursele afișate sub fiecare răspuns.

Utilizare:
    uv run streamlit run src/legislatie_rag/app/streamlit_app.py
"""

import re

import streamlit as st

from legislatie_rag.config import LAWS
from legislatie_rag.generate.answer import DISCLAIMER, RAG, Answer
from legislatie_rag.retrieve.hybrid import ArticleHit

EXAMPLES = [
    "Câte zile de concediu de odihnă am minim pe an?",
    "Mă poate concedia angajatorul cât sunt în concediu medical?",
    "Ce amendă iau dacă trec pe roșu?",
    "Ce pedeapsă primești pentru furt, în funcție de gravitate?",
    "Ce vârstă trebuie să ai ca să candidezi la președinție?",
    "Care este capitalul social minim pentru un SRL?",
]

st.set_page_config(page_title="Legislație RO · asistent", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Se încarcă modelele de căutare (prima pornire durează ~30 s)…")
def load_rag() -> RAG:
    rag = RAG()
    rag.retriever.search("încălzire")  # prima căutare pe GPU e lentă; o facem la pornire
    return rag


def law_versions(rag: RAG) -> dict[str, tuple[str | None, str]]:
    versions = {}
    for chunk in rag.retriever.chunks:
        versions.setdefault(chunk.law_slug, (chunk.consolidation_date, chunk.url))
    return versions


def emphasize_markers(text: str) -> str:
    """[1] → **[1]**, ca trimiterile la surse să iasă în evidență."""
    return re.sub(r"\[(\d+)\]", r"**[\1]**", text)


def render_source(n: int, hit: ArticleHit) -> None:
    first = hit.chunks[0]
    version = first.consolidation_date or "forma republicată"
    with st.expander(f"**[{n}]** {hit.citation}"):
        st.caption(first.header.replace("\n", " · "))
        st.markdown("\n\n".join(c.body.replace("\n", "  \n") for c in hit.chunks))
        st.caption(f"Versiune consolidată: {version} · [textul oficial]({first.url})")


def render_answer(answer: Answer) -> None:
    if answer.llm_error:
        st.warning(answer.text, icon="⚠️")
    else:
        st.markdown(emphasize_markers(answer.text))
    if answer.invalid_citations:
        st.caption(f"⚠️ Citări fără sursă detectate: {answer.invalid_citations}")

    cited = set(answer.cited)
    primary = [n for n in range(1, len(answer.sources) + 1) if n in cited or answer.llm_error]
    others = [n for n in range(1, len(answer.sources) + 1) if n not in primary]
    if primary:
        st.markdown("**Surse**")
        for n in primary:
            render_source(n, answer.sources[n - 1])
    if others:
        with st.expander(f"Alte {len(others)} articole găsite, necitate în răspuns"):
            for n in others:
                hit = answer.sources[n - 1]
                st.markdown(f"**[{n}]** {hit.citation} — {hit.chunks[0].body[:140]}…")

    timing = f"căutare {answer.retrieval_ms / 1000:.1f} s"
    if answer.model:
        timing += f" · generare {answer.generation_ms / 1000:.1f} s · {answer.model}"
    st.caption(timing)


def sidebar(rag: RAG) -> list[str]:
    with st.sidebar:
        st.header("⚖️ Legislație RO")
        st.markdown(
            "Asistent RAG care răspunde **doar pe baza textului oficial** al legilor, "
            "cu trimitere la articol."
        )
        st.subheader("Legi indexate")
        versions = law_versions(rag)
        for law in LAWS:
            date, url = versions.get(law.slug, (None, ""))
            st.markdown(
                f"- [{law.name}]({url}) · {law.short_ref}  \n"
                f"  <small>versiune: {date or 'forma republicată'}</small>",
                unsafe_allow_html=True,
            )
        selected = st.multiselect(
            "Caută doar în",
            options=[law.slug for law in LAWS],
            format_func=lambda slug: next(law.name for law in LAWS if law.slug == slug),
            placeholder="Toate legile",
        )
        st.divider()
        st.caption(
            f"Fiecare întrebare e tratată independent (fără memoria conversației). {DISCLAIMER}"
        )
        if st.button("Conversație nouă", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return selected


def main() -> None:
    rag = load_rag()
    laws = sidebar(rag)

    st.title("Întreabă legea")
    st.caption(" · ".join(law.name for law in LAWS))

    messages = st.session_state.setdefault("messages", [])
    if not messages:
        st.markdown("**Exemple de întrebări:**")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["content"])

    question = st.chat_input("Scrie o întrebare despre legislație…")
    question = question or st.session_state.pop("pending", None)
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Caut în legi și formulez răspunsul…"):
            answer = rag.ask(question, laws=laws or None)
        render_answer(answer)
    messages.append({"role": "assistant", "content": answer})


main()
