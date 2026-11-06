#!/usr/bin/env python3
"""Оценка нормализации (без обязательной LLM)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant_api.normalization.intent_rules import detect_intent
from assistant_api.settings import PROJECT_ROOT

q = [
    ("Как часто проводится ТО-1 автолифта АЛ 3У-М6?", "periodicity"),
    ("Какую температуру рабочей жидкости гидросистемы нельзя превышать?", "parameter_value"),
    ("При каком уровне масла в баке запрещено эксплуатировать автолифт?", "safety"),
]
ok = 0
for text, intent in q:
    got, _ = detect_intent(text)
    ok += got == intent
rep = PROJECT_ROOT / "reports"
rep.mkdir(exist_ok=True)
(rep / "eval_normalization.json").write_text(
    json.dumps({"intent_ok": ok, "n": len(q)}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("intent", ok, "/", len(q))
