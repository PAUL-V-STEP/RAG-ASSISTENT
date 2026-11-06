"""Техническая очистка текста (шаг [1])."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

import yaml

from assistant_api.settings import get_settings

UNIT_MAP = [
    (re.compile(r"\bмиллилитров\b", re.I), "мл"),
    (re.compile(r"\bлитров\b", re.I), "л"),
    (re.compile(r"\bградусов\b|\bград\.?\b|°с", re.I), "°C"),
    (re.compile(r"\bминут\.?\b|\bмин\.\b", re.I), "мин"),
    (re.compile(r"\bчасов\b|\bчас\.\b|\bмоточасов\b", re.I), "ч"),
    (re.compile(r"\bмпа\b", re.I), "МПа"),
    (re.compile(r"\bн·м\b|\bнм\b", re.I), "Н·м"),
]


@lru_cache(maxsize=1)
def _to_variants() -> list[tuple[re.Pattern[str], str]]:
    settings = get_settings()
    path = settings.resolve_path("GLOSSARY_EXTRA_FILE")
    mapping: list[tuple[re.Pattern[str], str]] = []
    if path and path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("to_variants") or []:
            for src in item.get("from") or []:
                mapping.append(
                    (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.I), item["to"])
                )
    # длинные варианты первыми
    mapping.sort(key=lambda p: -len(p[0].pattern))
    return mapping


def fold_yo(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\u00a0", " ").replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    s = fold_yo(s)
    for pat, repl in UNIT_MAP:
        s = pat.sub(repl, s)
    for pat, repl in _to_variants():
        s = pat.sub(repl, s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def tokenize(text: str) -> list[str]:
    s = fold_yo(text).lower()
    return [t for t in re.split(r"[^\w-]+", s, flags=re.UNICODE) if t]


def numbers_in(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text or "")
