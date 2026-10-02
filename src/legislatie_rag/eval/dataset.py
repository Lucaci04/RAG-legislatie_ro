"""Setul de întrebări de evaluare (data/eval/questions.jsonl)."""

import json
from dataclasses import dataclass

from legislatie_rag.config import DATA_DIR

QUESTIONS_PATH = DATA_DIR / "eval" / "questions.jsonl"
RESULTS_DIR = DATA_DIR / "eval" / "results"


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    """natural | informal | explicit_ref | trap | unanswerable"""
    question: str
    gold: list[str]
    """Articolele corecte, ca „lege:articol”. Gol = răspunsul nu există în legile indexate."""
    answer_key: str
    """Ce trebuie să conțină un răspuns corect (folosit de judecătorul LLM)."""


def load_questions() -> list[Question]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f]
