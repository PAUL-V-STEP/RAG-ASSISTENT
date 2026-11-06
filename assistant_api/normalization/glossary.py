"""Глоссарий канонизации."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from assistant_api.settings import PROJECT_ROOT, get_settings

PRODUCT_CANONICALS = {
    "ал",
    "ал 3у-м6",
    "автолифт",
    "оборудование",
    "надстройка",
    "спецоборудование",
    "изделие",
}

GLOSSARY_LINE_RE = None


@dataclass
class GlossaryTerm:
    canonical: str
    type: str
    node: Optional[str] = None
    abbr: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    parameter: Optional[str] = None


def parse_glossary_line(line: str) -> GlossaryTerm:
    text = line.strip()
    if text.startswith("[глоссарий]"):
        text = text[len("[глоссарий]") :].strip()
    if "—" in text:
        head, rest = text.split("—", 1)
    elif " - " in text:
        head, rest = text.split(" - ", 1)
    else:
        raise ValueError(f"не разобрана строка глоссария: {line}")
    canonical = head.strip()
    variants: list[str] = []
    if "Синонимы и варианты запроса:" in rest:
        syn = rest.split("Синонимы и варианты запроса:", 1)[1]
        syn = syn.split("Где встречается:", 1)[0]
        variants = [v.strip().rstrip(".") for v in syn.split(",") if v.strip()]
    gtype = _infer_type(canonical)
    node = _infer_node(canonical, rest)
    abbr = [canonical] if canonical.isupper() or canonical in {"ТО-1", "ЕО", "СО", "КОМ", "АКБ", "ГОСТ"} else []
    if canonical.lower() in PRODUCT_CANONICALS:
        gtype = "product"
    return GlossaryTerm(canonical=canonical, type=gtype, node=node, abbr=abbr, variants=variants)


def _infer_type(canonical: str) -> str:
    c = canonical.lower()
    if canonical in {"ЕО", "ТО-1", "СО"}:
        return "level"
    if canonical == "ТО":
        return "topic"
    nodes = {
        "шасси": "node",
        "гидросистема": "node",
        "подъемный механизм": "node",
        "кузов": "node",
        "выдвижные опоры": "node",
        "передняя площадка": "node",
        "передвижная и выдвижная площадки": "node",
    }
    if canonical in nodes:
        return "node"
    if c in PRODUCT_CANONICALS or canonical in {"АЛ", "АЛ 3У-М6"}:
        return "product"
    if canonical in {"трудоемкость"}:
        return "parameter"
    if canonical in {"санитарная обработка", "прошприцевать", "техническое освидетельствование"}:
        return "operation"
    if canonical in {
        "гидрожидкость",
        "гидравлическая насосная установка",
        "гидроцилиндр",
        "фильтроэлемент",
        "мерное стекло",
        "пресс-масленка",
        "рабочая платформа",
        "стрела пантографа",
        "задний трап",
        "проблесковый маяк",
        "Магос Унидем",
        "флаундер",
        "МОП",
        "формуляр",
        "КОМ",
    }:
        return "part"
    return "topic"


def _infer_node(canonical: str, rest: str) -> Optional[str]:
    mapping = {
        "шасси": "шасси",
        "гидросистема": "гидросистема",
        "гидрожидкость": "гидросистема",
        "гидравлическая насосная установка": "гидросистема",
        "гидроцилиндр": "гидросистема",
        "фильтроэлемент": "гидросистема",
        "мерное стекло": "гидросистема",
        "подъемный механизм": "подъемный механизм",
        "стрела пантографа": "подъемный механизм",
        "кузов": "фургон",
        "передняя площадка": "площадки",
        "передвижная и выдвижная площадки": "площадки",
        "выдвижные опоры": "опоры",
        "рабочая платформа": "площадки",
    }
    return mapping.get(canonical)


def load_glossary(settings=None) -> list[GlossaryTerm]:
    settings = settings or get_settings()
    terms: list[GlossaryTerm] = []
    can_path = settings.resolve_path("GLOSSARY_CANONICAL_FILE")
    extra_path = settings.resolve_path("GLOSSARY_EXTRA_FILE")
    if can_path and can_path.exists():
        data = yaml.safe_load(can_path.read_text(encoding="utf-8")) or []
        if isinstance(data, dict):
            data = data.get("terms") or []
        for item in data:
            terms.append(_term_from_dict(item))
    if extra_path and extra_path.exists():
        extra = yaml.safe_load(extra_path.read_text(encoding="utf-8")) or {}
        for item in extra.get("terms") or []:
            terms.append(_term_from_dict(item))
    # дедуп по canonical+type, extra позже перекрывает
    by_key: dict[tuple[str, str], GlossaryTerm] = {}
    for t in terms:
        by_key[(t.canonical.lower(), t.type)] = t
    return list(by_key.values())


def _term_from_dict(item: dict) -> GlossaryTerm:
    return GlossaryTerm(
        canonical=item["canonical"],
        type=item.get("type") or "topic",
        node=item.get("node"),
        abbr=list(item.get("abbr") or []),
        variants=list(item.get("variants") or []),
        parameter=item.get("parameter"),
    )


def all_surface_forms(term: GlossaryTerm) -> list[str]:
    forms = [term.canonical, *term.abbr, *term.variants]
    return [f for f in forms if f]


def subject_terms(terms: list[GlossaryTerm]) -> list[GlossaryTerm]:
    return [t for t in terms if t.type in {"node", "level", "operation", "part", "topic"}]


def parse_kb_glossary(kb_text: str) -> list[GlossaryTerm]:
    out = []
    for i, line in enumerate(kb_text.splitlines(), 1):
        if line.startswith("[глоссарий]"):
            try:
                out.append(parse_glossary_line(line))
            except ValueError as exc:
                raise ValueError(f"строка {i}: {exc}") from exc
    return out
