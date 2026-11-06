"""Нормализация периода (11.3 п. 6)."""

from __future__ import annotations

import re

from assistant_api.normalization.text_clean import fold_yo

WORD_NUM = {
    "один": "1",
    "одна": "1",
    "одно": "1",
    "два": "2",
    "две": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
}

QUESTION_MARKERS = (
    "сколько раз",
    "раз в сколько",
    "через сколько",
    "как часто",
)

PERIOD_OK = re.compile(
    r"^(\d+(?:-\d+)? (месяц|ч|год|день|неделя)|\d+(?:-\d+)?/(год|месяц|неделя|день))$"
)


def _digits(text: str) -> str:
    s = fold_yo(text).lower()
    for w, n in WORD_NUM.items():
        s = re.sub(rf"\b{w}\b", n, s)
    s = s.replace("–", "-").replace("—", "-")
    return s


def normalize_period(value: str | None) -> str | None:
    if not value:
        return None
    raw = fold_yo(str(value)).strip()
    if PERIOD_OK.match(raw):
        return raw
    s = _digits(raw)
    if any(m in s for m in QUESTION_MARKERS):
        return None
    if re.search(r"ежедневн|каждый день", s):
        return "1/день"
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*раз[а]?\s*в\s*недел", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}/неделя"
    m = re.search(r"(\d+)\s*раз[а]?\s*в\s*год", s)
    if m:
        n = int(m.group(1))
        if n > 1:
            return f"{n}/год"
        return "1 год"
    if re.search(r"ежегодн|\bраз в год\b|\bодин раз в год\b", s):
        return "1 год"
    m = re.search(r"раз в (\d+) год", s)
    if m:
        return f"{m.group(1)} год"
    m = re.search(r"(\d+)\s*месяц", s)
    if m:
        return f"{m.group(1)} месяц"
    m = re.search(r"(\d+)\s*(?:ч|час|моточас)", s)
    if m:
        return f"{m.group(1)} ч"
    m = re.search(r"(\d+)\s*год", s)
    if m and "раз" in s:
        n = int(m.group(1))
        return f"{n} год"
    if PERIOD_OK.match(s):
        return s
    return None


def extract_period_from_question(text: str) -> str | None:
    return normalize_period(text)
