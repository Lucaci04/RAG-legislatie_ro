from types import SimpleNamespace

import pytest
from google.genai import errors

from legislatie_rag.generate.answer import RAG, check_citations
from legislatie_rag.generate.llm import Completion, GeminiLLM, LLMUnavailableError
from legislatie_rag.generate.prompt import NO_SOURCES_ANSWER, SYSTEM_PROMPT, build_prompt
from legislatie_rag.retrieve.hybrid import ArticleHit
from tests.test_text_bm25 import make_chunk


def make_hit(article: str, body: str) -> ArticleHit:
    return ArticleHit("t", article, [make_chunk(article, body)], score=1.0)


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, prompt: str) -> Completion:
        self.calls.append((system, prompt))
        return Completion(self.text, model="fake")


class FakeRetriever:
    def __init__(self, hits: list[ArticleHit]):
        self.hits = hits

    def search(self, query, k=5, laws=None):
        return self.hits[:k]


def test_check_citations_separates_valid_and_invented():
    valid, invalid = check_citations("Da [2]. Și [1], iar [2] din nou. Dar [7].", n_sources=3)
    assert valid == [2, 1]
    assert invalid == [7]


def test_build_prompt_numbers_sources_with_context_header():
    prompt = build_prompt("Întrebare?", [make_hit("145", "(1) 20 de zile."), make_hit("60", "x")])

    assert prompt.startswith("SURSE:\n\n[1] Legea de test\n(1) 20 de zile.")
    assert "\n\n[2] Legea de test\nx" in prompt
    assert prompt.endswith("ÎNTREBARE: Întrebare?")


def test_system_prompt_lists_indexed_laws():
    assert "Codul muncii (Legea 53/2003)" in SYSTEM_PROMPT


def test_rag_answer_with_citations():
    llm = FakeLLM("Minimum 20 de zile [1].")
    rag = RAG(retriever=FakeRetriever([make_hit("145", "20 de zile")]), llm=llm)

    answer = rag.ask("Câte zile de concediu?")

    assert answer.cited == [1]
    assert answer.cited_sources[0].article == "145"
    assert answer.invalid_citations == []
    assert llm.calls[0][0] == SYSTEM_PROMPT


def test_rag_without_sources_does_not_call_llm():
    llm = FakeLLM("nu ar trebui apelat")
    answer = RAG(retriever=FakeRetriever([]), llm=llm).ask("Ceva?")

    assert answer.text == NO_SOURCES_ANSWER
    assert llm.calls == []


def test_rag_keeps_sources_when_llm_unavailable():
    class DownLLM:
        def generate(self, system, prompt):
            raise LLMUnavailableError("indisponibil")

    hits = [make_hit("145", "20 de zile")]
    answer = RAG(retriever=FakeRetriever(hits), llm=DownLLM()).ask("Câte zile?")

    assert answer.llm_error
    assert answer.sources == hits
    assert answer.cited == []


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "status": "X", "message": "x"}})


def _gemini_with(responses: list, monkeypatch) -> tuple[GeminiLLM, list[str]]:
    """GeminiLLM cu clientul înlocuit: fiecare apel consumă următorul răspuns/eroare."""
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr("time.sleep", lambda _: None)
    llm = GeminiLLM(model="principal", fallback_model="rezerva", max_retries=2)
    models_called: list[str] = []

    def generate_content(model, contents, config):
        models_called.append(model)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, usage_metadata=None)

    llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return llm, models_called


def test_gemini_retries_then_falls_back(monkeypatch):
    llm, called = _gemini_with([_api_error(503), _api_error(429), "răspuns"], monkeypatch)

    completion = llm.generate("sistem", "prompt")

    assert completion.text == "răspuns"
    assert completion.model == "rezerva"
    assert called == ["principal", "principal", "rezerva"]


def test_gemini_skips_retries_on_daily_quota(monkeypatch):
    daily = errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            }
        },
    )
    llm, called = _gemini_with([daily, "răspuns"], monkeypatch)

    assert llm.generate("sistem", "prompt").model == "rezerva"
    assert called == ["principal", "rezerva"]


def test_gemini_does_not_retry_client_errors(monkeypatch):
    llm, called = _gemini_with([_api_error(400)], monkeypatch)

    with pytest.raises(errors.APIError):
        llm.generate("sistem", "prompt")
    assert called == ["principal"]


def test_gemini_gives_up_when_all_models_unavailable(monkeypatch):
    llm, _ = _gemini_with([_api_error(503)] * 4, monkeypatch)

    with pytest.raises(LLMUnavailableError):
        llm.generate("sistem", "prompt")
