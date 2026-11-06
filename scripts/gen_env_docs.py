#!/usr/bin/env python3
"""Таблица переменных для README из Settings."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_api.settings import SPECS

print("| Переменная | Тип | По умолчанию |")
print("|---|---|---|")
for s in SPECS:
    print(f"| `{s.name}` | {s.kind} | `{s.default}` |")
