"""Strat de abstractizare peste furnizorul de LLM.

Restul aplicației depinde doar de protocolul `LLM`; implicit folosim Gemini.
Un alt furnizor (Claude, Ollama) înseamnă o clasă nouă cu metoda `generate`.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from legislatie_rag.config import ROOT_DIR

log = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK = "gemini-3.1-flash-lite"
RETRYABLE_CODES = {429, 500, 503, 504}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> Completion: ...


class LLMUnavailableError(RuntimeError):
    """Furnizorul nu a răspuns nici după reîncercări, pe niciun model."""


class GeminiLLM:
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_model: str | None = DEFAULT_GEMINI_FALLBACK,
        temperature: float = 0.2,
        max_retries: int = 3,
        thinking_level: str | None = "low",
    ):
        from google import genai

        self._client = genai.Client()
        self.models = [m for m in (model, fallback_model) if m]
        self.temperature = temperature
        self.max_retries = max_retries
        # Răspunsul vine din câteva articole primite în context, deci un raționament lung
        # nu ajută, doar crește latența. Se aplică doar modelului principal: modelele „lite”
        # nu gândesc implicit.
        self.thinking_level = thinking_level

    def generate(self, system: str, prompt: str) -> Completion:
        from google.genai import errors, types

        last_error: Exception | None = None
        for i, model in enumerate(self.models):
            thinking = (
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if i == 0 and self.thinking_level
                else None
            )
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                thinking_config=thinking,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            for attempt in range(self.max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    usage = response.usage_metadata
                    return Completion(
                        text=(response.text or "").strip(),
                        model=model,
                        input_tokens=usage.prompt_token_count if usage else None,
                        output_tokens=usage.candidates_token_count if usage else None,
                    )
                except errors.APIError as e:
                    if e.code not in RETRYABLE_CODES:
                        raise
                    last_error = e
                    delay = 2**attempt
                    log.warning("%s: %s %s, reîncerc în %ss", model, e.code, e.status, delay)
                    time.sleep(delay)
            log.warning("%s indisponibil, trec pe modelul de rezervă", model)
        raise LLMUnavailableError(f"Niciun model disponibil ({self.models})") from last_error


def get_llm() -> LLM:
    load_dotenv(ROOT_DIR / ".env")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Lipsește GEMINI_API_KEY în .env (vezi .env.example)")
    return GeminiLLM(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        fallback_model=os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK),
    )
