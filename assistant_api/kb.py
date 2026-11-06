"""Путь к базе знаний и kb_version."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from assistant_api.settings import PROJECT_ROOT, Settings, get_settings

QA_MARKERS = ("[вопрос-ответ]", "ВОПРОСЫ И ОТВЕТЫ")


def kb_version_for(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def load_kb_text(settings: Settings | None = None) -> tuple[str, Path, str]:
    settings = settings or get_settings()
    path = settings.data_file()
    if not path.exists():
        raise FileNotFoundError(f"DATA_FILE не найден: {path}")
    text = path.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("[вопрос-ответ]") or line.startswith("ВОПРОСЫ И ОТВЕТЫ"):
            raise ValueError(
                f"DATA_FILE содержит блок вопросов-ответов ({path.name}, строка {i})"
            )
    return text, path, kb_version_for(path)
