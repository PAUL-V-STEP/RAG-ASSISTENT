from __future__ import annotations

from pathlib import Path

import yaml


def load_qa_base(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("qa_base.yaml: ожидается mapping")
    return data


def load_qa_tests(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def fact_norm(s: str) -> str:
    import unicodedata
    import re

    t = unicodedata.normalize("NFKC", str(s)).replace("\u00a0", " ")
    t = t.replace("−", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t).lower()
    return t
