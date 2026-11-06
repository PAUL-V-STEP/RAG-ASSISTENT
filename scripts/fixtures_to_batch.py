#!/usr/bin/env python3
"""Конвертер фикстур приложения A в batch YAML (17.7)."""

from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/input_fix.yaml")
    data = yaml.safe_load(src.read_text(encoding="utf-8")) or []
    items = []
    for i, row in enumerate(data, 1):
        items.append({"id": row.get("id") or f"x{i}", "turns": [{"q": row.get("raw") or row.get("question")}]})
    print(yaml.safe_dump({"schema_version": 1, "items": items}, allow_unicode=True))


if __name__ == "__main__":
    main()
