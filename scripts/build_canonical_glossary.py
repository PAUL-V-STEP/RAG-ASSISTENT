#!/usr/bin/env python3
"""Сборка glossary_canonical.yaml из RAG V3 + glossary_extra.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant_api.kb import load_kb_text
from assistant_api.normalization.glossary import parse_kb_glossary
from assistant_api.settings import load_settings


def main():
    settings = load_settings()
    text, path, _ = load_kb_text(settings)
    terms = parse_kb_glossary(text)
    extra_path = settings.resolve_path("GLOSSARY_EXTRA_FILE")
    extra = yaml.safe_load(extra_path.read_text(encoding="utf-8")) if extra_path.exists() else {}
    out = []
    for t in terms:
        out.append(
            {
                "canonical": t.canonical,
                "type": t.type,
                "node": t.node,
                "abbr": t.abbr,
                "variants": t.variants,
            }
        )
    for item in extra.get("terms") or []:
        out.append(item)
    dest = settings.resolve_path("GLOSSARY_CANONICAL_FILE")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"записано {len(out)} терминов → {dest}")
    if any(t.canonical == "ТО-2" for t in terms):
        raise SystemExit("в глоссарии есть ТО-2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
