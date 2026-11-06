#!/usr/bin/env python3
"""Выгрузка qa_tool list в читаемый UTF-8 файл (с BOM для Блокнота).

Запуск из корня проекта:
    .\\venv_py311\\Scripts\\python.exe scripts\\qa_list_to_file.py
    .\\venv_py311\\Scripts\\python.exe scripts\\qa_list_to_file.py reports\\qa\\list.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from assistant_api.qa_layer.loader import load_qa_base
from assistant_api.settings import load_settings


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "qa" / "list.txt"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    data = load_qa_base(settings.resolve_path("QA_BASE_FILE"))
    records = data.get("records") or []

    lines = [
        "id     intent           node             level    verified     вопрос",
        "-" * 120,
    ]
    for r in records:
        q = (r.get("question") or "").replace("\n", " ")
        lines.append(
            f"{r['id']:6} {(r.get('intent') or '-'):16} "
            f"{str(r.get('node') or '-'):16} {str(r.get('maintenance_level') or '-'):8} "
            f"{str(r.get('verified_by') or '-'):12} {q}"
        )
    lines.append("")
    lines.append(f"Всего записей: {len(records)}")
    lines.append(f"Файл базы: {settings.QA_BASE_FILE}")

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8-sig")
    print(f"Записано {len(records)} строк → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
