import pytest

from legislatie_rag.config import LAWS
from legislatie_rag.eval.dataset import load_questions
from legislatie_rag.eval.generation import thinking_for
from legislatie_rag.eval.retrieval import first_gold_rank, summarize


def test_first_gold_rank():
    assert first_gold_rank(["a:1", "b:2", "c:3"], ["b:2", "c:3"]) == 2
    assert first_gold_rank(["a:1"], ["z:9"]) is None


def test_summarize_recall_and_mrr():
    recall, mrr = summarize([1, 3, None, 2])

    assert recall[1] == pytest.approx(0.25)
    assert recall[3] == pytest.approx(0.75)
    assert recall[10] == pytest.approx(0.75)
    assert mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)


def test_thinking_level_per_model():
    assert thinking_for("gemini-3.8-flash") == "low"
    assert thinking_for("gemini-3.1-flash-lite") is None


def test_eval_dataset_is_well_formed():
    questions = load_questions()
    law_slugs = {law.slug for law in LAWS}

    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.type in {"natural", "informal", "explicit_ref", "trap", "unanswerable"}
        assert bool(q.gold) == (q.type != "unanswerable")
        assert all(g.split(":")[0] in law_slugs for g in q.gold)
        assert q.answer_key
