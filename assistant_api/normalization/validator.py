"""Проверка переписывания (шаг [5])."""

from __future__ import annotations

import re

from assistant_api.normalization.text_clean import numbers_in, tokenize
from assistant_api.settings import get_settings

NEG = {"не", "нельзя", "без", "кроме", "только"}


def validate_rewrite(
    original: str,
    standalone: str,
    assistant_answers: list[str] | None = None,
    history_text: str = "",
    settings=None,
) -> tuple[bool, str | None]:
    settings = settings or get_settings()
    orig_nums = numbers_in(original)
    stand_nums = numbers_in(standalone)
    for n in orig_nums:
        if n not in stand_nums:
            return False, "missing_number"
    orig_toks = set(tokenize(original))
    stand_toks = set(tokenize(standalone))
    if orig_toks & NEG:
        if not (stand_toks & NEG) and not any(t.startswith("запрещ") for t in stand_toks):
            return False, "missing_negation"
    answers = " ".join(assistant_answers or [])
    ans_nums = set(numbers_in(answers))
    for n in stand_nums:
        if n not in orig_nums and n in ans_nums:
            return False, "answer_leak_number"
    if "ТО-2" in standalone or "то-2" in standalone.lower():
        hist = history_text + " " + original
        if "ТО-2" not in hist and "то-2" not in hist.lower():
            return False, "unknown_to_level"
    max_len = settings.REWRITE_MAX_RATIO * max(len(original), 30)
    if len(standalone) > max_len:
        return False, "too_long"
    if standalone.strip().startswith("Ответ:") or standalone.count(".") > 2:
        return False, "looks_like_answer"
    if re.search(r"^\s*\d+[.)]\s", standalone, re.M):
        return False, "looks_like_answer"
    return True, None
