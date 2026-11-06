from __future__ import annotations

import re

from assistant_api.qa_layer.matcher import QADecision


def format_direct(decision: QADecision) -> str:
    rec = decision.record or {}
    src = rec.get("source") or []
    src_s = ", ".join("разд. 4" if x == "4" else str(x) for x in src)
    return f"{rec.get('answer','')}\n\nИсточник: п. {src_s}. Проверенный ответ ({decision.qa_id})."


def numbers_with_units(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?\s*(?:мин|ч|л|мл|%|°C|м/с|месяц(?:а|ев)?)", text or "", flags=re.I)


def hybrid_numbers_ok(answer: str, verified: str, chunks: str, question: str) -> bool:
    allowed = verified + " " + chunks + " " + question
    for item in numbers_with_units(answer):
        token = item.split()[0]
        if token not in allowed and item not in allowed:
            return False
    return True
