"""LLM-переписывание зависимого вопроса (шаг [4])."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from assistant_api.normalization.schema import Entities, LEVELS
from assistant_api.settings import PROJECT_ROOT, get_settings

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "contextualize_ru.txt"
_PROMPT_FIELDS = ("state", "glossary_terms", "history", "question")
_PROMPT_FIELD_RE = re.compile(r"\{(" + "|".join(_PROMPT_FIELDS) + r")\}")


def render_contextualize_prompt(template: str, **kwargs: Any) -> str:
    """Подставляет поля промпта, не трогая JSON-пример со скобками."""
    missing = [name for name in _PROMPT_FIELDS if name not in kwargs]
    if missing:
        raise KeyError(f"нет полей промпта: {missing}")
    return _PROMPT_FIELD_RE.sub(lambda m: str(kwargs[m.group(1)]), template)


class ContextualizeResult:
    def __init__(self, data: dict, source: str = "llm", error: str | None = None):
        self.data = data
        self.source = source
        self.error = error


def contextualize(
    question: str,
    history: list[dict],
    state,
    glossary_terms: list[str],
    openai_client,
    settings=None,
) -> ContextualizeResult:
    settings = settings or get_settings()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    hist = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    prompt = render_contextualize_prompt(
        template,
        state=state.model_dump_json() if hasattr(state, "model_dump_json") else str(state),
        glossary_terms=", ".join(glossary_terms[:50]),
        history=hist or "(пусто)",
        question=question,
    )
    last_err = None
    attempts = 1 + int(settings.NORMALIZER_RETRIES)
    for _ in range(attempts):
        try:
            resp = openai_client.chat.completions.create(
                model=settings.NORMALIZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=settings.NORMALIZER_TIMEOUT_S,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            if data.get("maintenance_level") == "ТО-2" or (
                isinstance(data.get("entities"), dict)
                and data["entities"].get("maintenance_level") == "ТО-2"
            ):
                if isinstance(data.get("entities"), dict):
                    data["entities"]["maintenance_level"] = None
            return ContextualizeResult(data, "llm")
        except TimeoutError as exc:
            return ContextualizeResult({}, "fallback", "timeout")
        except Exception as exc:
            last_err = str(exc)
            if "timeout" in last_err.lower() or "timed out" in last_err.lower():
                return ContextualizeResult({}, "fallback", "timeout")
    return ContextualizeResult({}, "fallback", last_err or "invalid_json")
