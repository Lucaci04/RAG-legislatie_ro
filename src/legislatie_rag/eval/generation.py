"""Compară modele LLM pe același context, cu un judecător LLM „orb”.

1. Pentru fiecare întrebare, căutarea rulează o dată; toate modelele primesc aceleași surse.
2. Fiecare model răspunde (fără model de rezervă: o eroare rămâne eroare).
3. Judecătorul primește întrebarea, sursele, răspunsul de referință și răspunsurile
   anonimizate (A, B, C, ... în ordine aleatorie) și le notează pe toate într-un singur apel.

Rezultatele se salvează pe măsură ce vin, deci scriptul se poate relua după o limită de cereri.

Utilizare:
    uv run python -m legislatie_rag.eval.generation            # răspunsuri + judecată + raport
    uv run python -m legislatie_rag.eval.generation --report   # doar raportul
"""

import json
import random
import statistics
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google.genai import errors

from legislatie_rag.config import ROOT_DIR
from legislatie_rag.eval.dataset import RESULTS_DIR, Question, load_questions
from legislatie_rag.generate.answer import check_citations
from legislatie_rag.generate.llm import GeminiLLM, LLMUnavailableError
from legislatie_rag.generate.prompt import SYSTEM_PROMPT, build_prompt, format_source
from legislatie_rag.retrieve.hybrid import ArticleHit, Retriever

# gemini-2.5-flash nu mai e disponibil pentru conturi noi (404), deși apare în listă.
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
JUDGE_MODEL = "gemini-3.8-flash"
PACE_SECONDS = 4.0  # pauză între apeluri, pentru limitele tier-ului gratuit

ANSWERS_PATH = RESULTS_DIR / "answers.jsonl"
JUDGMENTS_PATH = RESULTS_DIR / "judgments.jsonl"

JUDGE_SYSTEM = """Ești evaluator pentru un asistent juridic bazat pe legislația din România.

Primești: întrebarea, sursele (articolele de lege) date asistentului, un răspuns de referință
și mai multe răspunsuri anonime (A, B, ...). Evaluează FIECARE răspuns independent.

correctness (0-2):
- 2 = conține faptele esențiale din răspunsul de referință, fără erori
- 1 = parțial corect: lipsește ceva important sau are o imprecizie minoră
- 0 = greșit, conține o eroare importantă sau nu răspunde la întrebare
Pentru întrebările fără răspuns în surse, 2 = refuză clar și nu inventează; 0 = inventează.

faithful: true dacă TOATE afirmațiile juridice din răspuns sunt susținute de surse.
refused: true dacă răspunsul spune, în esență, că informația nu se află în surse.

Fii strict și consecvent. Lungimea sau stilul nu contează, doar corectitudinea."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "correctness": {"type": "integer", "enum": [0, 1, 2]},
                    "faithful": {"type": "boolean"},
                    "refused": {"type": "boolean"},
                    "comment": {"type": "string"},
                },
                "required": ["label", "correctness", "faithful", "refused", "comment"],
            },
        }
    },
    "required": ["evaluations"],
}


def thinking_for(model: str) -> str | None:
    """Gemini 3.x non-lite: „low” (ca în aplicație). Lite și 2.5: setarea implicită."""
    return "low" if model.startswith("gemini-3") and "lite" not in model else None


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Context:
    hits: list[ArticleHit]
    prompt: str

    @property
    def keys(self) -> list[str]:
        return [f"{h.law_slug}:{h.article}" for h in self.hits]


def build_contexts(questions: list[Question]) -> dict[str, Context]:
    retriever = Retriever()
    contexts = {}
    for q in questions:
        hits = retriever.search(q.question, k=5)
        contexts[q.id] = Context(hits, build_prompt(q.question, hits))
    return contexts


def generate_answers(questions: list[Question], contexts: dict[str, Context]) -> None:
    done = {(r["id"], r["model"]) for r in load_jsonl(ANSWERS_PATH) if not r.get("error")}
    llms = {
        m: GeminiLLM(
            model=m,
            fallback_model=None,
            thinking_level=thinking_for(m),
            max_retries=3,
            retry_base_delay=10,
        )
        for m in MODELS
    }
    todo = [(q, m) for q in questions for m in MODELS if (q.id, m) not in done]
    print(f"Răspunsuri: {len(done)} existente, {len(todo)} de generat")

    for i, (q, model) in enumerate(todo, start=1):
        ctx = contexts[q.id]
        row: dict = {"id": q.id, "model": model}
        try:
            completion = llms[model].generate(SYSTEM_PROMPT, ctx.prompt)
            cited, invalid = check_citations(completion.text, len(ctx.hits))
            row |= {
                "text": completion.text,
                "latency_ms": round(completion.latency_ms),
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "cited": [ctx.keys[n - 1] for n in cited],
                "invalid_citations": invalid,
            }
        except (LLMUnavailableError, errors.APIError) as e:
            row["error"] = str(getattr(e, "__cause__", None) or e)[:300]
        append_jsonl(ANSWERS_PATH, row)
        status = "eroare" if row.get("error") else f"{row['latency_ms'] / 1000:.1f}s"
        print(f"  [{i}/{len(todo)}] {q.id:<24} {model:<24} {status}", flush=True)
        time.sleep(PACE_SECONDS)


def judge_answers(questions: list[Question], contexts: dict[str, Context]) -> None:
    answers: dict[str, dict[str, dict]] = {}
    for r in load_jsonl(ANSWERS_PATH):
        if not r.get("error"):
            answers.setdefault(r["id"], {})[r["model"]] = r
    # Ultima judecată pe întrebare; dacă între timp au apărut răspunsuri noi, se rejudecă tot.
    judged = {r["id"]: set(r["scores"]) for r in load_jsonl(JUDGMENTS_PATH)}
    todo = [q for q in questions if answers.get(q.id) and judged.get(q.id) != set(answers[q.id])]
    print(f"Judecată: {len(judged)} existente, {len(todo)} de evaluat")

    judge = GeminiLLM(
        model=JUDGE_MODEL,
        fallback_model=None,
        thinking_level=None,
        temperature=0.0,
        max_retries=4,
        retry_base_delay=15,
    )
    for i, q in enumerate(todo, start=1):
        models = [m for m in MODELS if m in answers[q.id]]
        random.Random(q.id).shuffle(models)  # ordine aleatorie, dar reproductibilă
        labels = {chr(ord("A") + j): m for j, m in enumerate(models)}
        sources = "\n\n".join(
            format_source(n, h) for n, h in enumerate(contexts[q.id].hits, start=1)
        )
        responses = "\n\n".join(
            f"### Răspunsul {label}\n{answers[q.id][m]['text']}" for label, m in labels.items()
        )
        prompt = (
            f"ÎNTREBARE: {q.question}\n\nSURSE:\n{sources}\n\n"
            f"RĂSPUNS DE REFERINȚĂ: {q.answer_key}\n\nRĂSPUNSURI DE EVALUAT:\n\n{responses}"
        )
        try:
            completion = judge.generate(JUDGE_SYSTEM, prompt, json_schema=JUDGE_SCHEMA)
        except LLMUnavailableError as e:
            print(f"  [{i}/{len(todo)}] {q.id}: judecătorul indisponibil ({e.__cause__})")
            continue
        evaluations = json.loads(completion.text)["evaluations"]
        scores = {labels[e["label"]]: e for e in evaluations if e["label"] in labels}
        append_jsonl(JUDGMENTS_PATH, {"id": q.id, "labels": labels, "scores": scores})
        print(f"  [{i}/{len(todo)}] {q.id:<24} judecat", flush=True)
        time.sleep(PACE_SECONDS)


def report(questions: list[Question]) -> list[dict]:
    by_id = {q.id: q for q in questions}
    answers = {(r["id"], r["model"]): r for r in load_jsonl(ANSWERS_PATH) if not r.get("error")}
    errors = {(r["id"], r["model"]) for r in load_jsonl(ANSWERS_PATH) if r.get("error")}
    judgments = {r["id"]: r["scores"] for r in load_jsonl(JUDGMENTS_PATH)}

    rows = []
    for model in MODELS:
        judged = [(qid, s[model]) for qid, s in judgments.items() if model in s]
        if not judged:
            continue
        answerable = [(qid, s) for qid, s in judged if by_id[qid].gold]
        unanswerable = [(qid, s) for qid, s in judged if not by_id[qid].gold]
        traps = [(qid, s) for qid, s in judged if by_id[qid].type == "trap"]
        model_answers = [answers[(qid, model)] for qid, _ in judged]
        cites_gold = [
            any(c in by_id[qid].gold for c in answers[(qid, model)]["cited"])
            for qid, _ in answerable
        ]
        latencies = sorted(a["latency_ms"] for a in model_answers)
        rows.append(
            {
                "model": model,
                "n": len(judged),
                "score": statistics.mean(s["correctness"] for _, s in judged) / 2,
                "fully_correct": statistics.mean(s["correctness"] == 2 for _, s in judged),
                "faithful": statistics.mean(s["faithful"] for _, s in judged),
                "cites_gold": statistics.mean(cites_gold) if cites_gold else None,
                "correct_refusals": (
                    statistics.mean(s["refused"] for _, s in unanswerable) if unanswerable else None
                ),
                "false_refusals": (
                    statistics.mean(
                        s["refused"] for qid, s in answerable if by_id[qid].type != "trap"
                    )
                ),
                "traps": statistics.mean(s["correctness"] == 2 for _, s in traps)
                if traps
                else None,
                "invalid_citations": sum(len(a["invalid_citations"]) for a in model_answers),
                "latency_p50_s": latencies[len(latencies) // 2] / 1000,
                "latency_p90_s": latencies[int(len(latencies) * 0.9)] / 1000,
                "output_tokens": statistics.mean(a["output_tokens"] or 0 for a in model_answers),
                "errors": sum(1 for (_, m) in errors if m == model),
            }
        )

    def pct(x):
        return "—" if x is None else f"{x:.0%}"

    print(
        f"\n{'Model':<24} {'n':>3} {'Scor':>6} {'Corect':>7} {'Fidel':>6} {'Citează':>8} "
        f"{'Refuz ok':>9} {'Refuz greșit':>13} {'Capcane':>8} {'p50':>6} {'p90':>6} {'tok':>5}"
    )
    for r in rows:
        print(
            f"{r['model']:<24} {r['n']:>3} {pct(r['score']):>6} {pct(r['fully_correct']):>7} "
            f"{pct(r['faithful']):>6} {pct(r['cites_gold']):>8} {pct(r['correct_refusals']):>9} "
            f"{pct(r['false_refusals']):>13} {pct(r['traps']):>8} {r['latency_p50_s']:>5.1f}s "
            f"{r['latency_p90_s']:>5.1f}s {r['output_tokens']:>5.0f}"
        )
    (RESULTS_DIR / "generation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    if "--report" not in sys.argv:
        contexts = build_contexts(questions)
        generate_answers(questions, contexts)
        judge_answers(questions, contexts)
    report(questions)


if __name__ == "__main__":
    main()
