"""Канонизация вопроса (шаг [6])."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Optional

from rapidfuzz import fuzz, process

from assistant_api.normalization.glossary import GlossaryTerm, all_surface_forms, load_glossary
from assistant_api.normalization.intent_rules import detect_intent
from assistant_api.normalization.period import extract_period_from_question
from assistant_api.normalization.schema import Entities, LEVELS, NODES
from assistant_api.normalization.text_clean import fold_yo, tokenize
from assistant_api.settings import get_settings

STOP = {
    "и", "в", "во", "на", "с", "со", "к", "по", "для", "от", "до", "из", "у", "о", "об",
    "а", "но", "или", "ли", "же", "бы", "как", "что", "какой", "какая", "какие", "какое",
    "каков", "какова", "этот", "эта", "это", "при", "без", "над", "под", "после", "перед",
    "автолифт", "ал", "3у-м6", "ал-3у-м6",
}

ABBR_EXACT = {"ео", "со", "то-1", "ком", "акб", "гост"}

NUMBER_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(мин|ч|л|мл|%|°c|м/с|месяц(?:а|ев)?|час(?:ов)?)",
    re.I,
)


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


@lru_cache(maxsize=4096)
def lemmatize_token(token: str) -> str:
    morph = _morph()
    t = fold_yo(token).lower()
    if not morph:
        return t
    try:
        p = morph.parse(t)[0]
        return fold_yo(p.normal_form)
    except Exception:
        return t


def canonicalize(
    text: str,
    llm_entities: Entities | None = None,
    llm_intent: str | None = None,
    kb_version: str = "",
    normalizer_version: str | None = None,
    settings=None,
) -> dict:
    settings = settings or get_settings()
    terms = load_glossary(settings)
    intent, src = detect_intent(text)
    if intent == "other" and llm_intent:
        intent = llm_intent
        src = "llm"
    else:
        src = "rules" if src.startswith("rules") else src

    tokens = tokenize(text)
    lemmas = [lemmatize_token(t) for t in tokens]
    used = [False] * len(lemmas)
    found: list[GlossaryTerm] = []

    surfaces: list[tuple[tuple[str, ...], GlossaryTerm]] = []
    for term in terms:
        for form in all_surface_forms(term):
            parts = tuple(lemmatize_token(x) for x in tokenize(form) if x)
            if parts:
                surfaces.append((parts, term))
    surfaces.sort(key=lambda x: -len(x[0]))

    for parts, term in surfaces:
        n = len(parts)
        for i in range(len(lemmas) - n + 1):
            if any(used[i : i + n]):
                continue
            window = tuple(lemmas[i : i + n])
            if window == parts:
                for j in range(i, i + n):
                    used[j] = True
                found.append(term)
                break

    cutoff = settings.FUZZY_CUTOFF
    leftover_idx = [i for i, u in enumerate(used) if not u]
    catalog = []
    for term in terms:
        for form in all_surface_forms(term):
            if tokenize(form) and len(form) >= 5:
                catalog.append((form, term))
    for i in leftover_idx:
        tok = lemmas[i]
        if len(tok) < 5 or tok in ABBR_EXACT:
            continue
        names = [c[0] for c in catalog]
        hit = process.extractOne(tok, names, scorer=fuzz.ratio, score_cutoff=cutoff)
        if hit:
            _, _, idx = hit
            found.append(catalog[idx][1])
            used[i] = True

    ent = Entities()
    for term in found:
        if term.type == "product":
            continue
        if term.type == "node" and term.canonical.lower() in {n.lower() for n in NODES} or (
            term.node and term.node in NODES
        ):
            node_val = term.node or term.canonical
            if node_val == "кузов":
                node_val = "фургон"
            if node_val in NODES and not ent.node:
                ent.node = node_val  # type: ignore
        if term.type == "level" and term.canonical in LEVELS and not ent.maintenance_level:
            ent.maintenance_level = term.canonical  # type: ignore
        if term.type == "operation" and not ent.operation:
            ent.operation = term.canonical
        if term.type == "part" and not ent.part:
            ent.part = term.canonical
        if term.type == "parameter" and not ent.parameter:
            ent.parameter = term.canonical
        if term.type == "condition" and not ent.condition:
            ent.condition = term.canonical
        if term.node and not ent.node and term.node in NODES:
            ent.node = term.node  # type: ignore

    if llm_entities:
        for slot in ("node", "maintenance_level", "operation", "part", "parameter", "condition"):
            val = getattr(llm_entities, slot)
            if not val:
                continue
            ok = any(val == t.canonical or val in t.variants for t in terms)
            if getattr(ent, slot):
                continue
            if ok:
                setattr(ent, slot, val)

    ent.period = extract_period_from_question(text)
    numbers = []
    for m in NUMBER_UNIT.finditer(text):
        unit = m.group(2).lower().replace("°c", "°C")
        if unit.startswith("месяц"):
            unit = "месяц"
        elif unit.startswith("час"):
            unit = "ч"
        numbers.append(f"{m.group(1)} {unit}")
    ent.numbers = sorted(set(numbers))

    neg = any(t in tokenize(text) for t in ("не", "нельзя", "без", "запрещается", "запрещено"))
    neg_flag = "neg" if neg else "-"

    residual_parts = []
    if not (ent.operation or ent.part or ent.parameter):
        for i, lemma in enumerate(lemmas):
            if used[i]:
                continue
            if lemma in STOP or lemma.isdigit() or len(lemma) < 3:
                continue
            residual_parts.append(lemma)
        # topic (ТО, …) не заполняет слоты, но должен различать «что такое ТО?» и «что такое АЛ?»
        for term in found:
            if term.type == "topic":
                residual_parts.append(fold_yo(term.canonical).lower())
    residual = ",".join(sorted(set(residual_parts))) if residual_parts else "-"

    period = ent.period or "-"
    numbers_s = ",".join(ent.numbers) if ent.numbers else "-"
    canonical = "|".join(
        [
            intent,
            ent.node or "-",
            ent.maintenance_level or "-",
            ent.operation or "-",
            ent.part or "-",
            ent.parameter or "-",
            ent.condition or "-",
            period,
            numbers_s,
            neg_flag,
            residual,
        ]
    )
    nv = normalizer_version or settings.NORMALIZER_VERSION
    cache_key = hashlib.sha256(f"{nv}|{kb_version}|{canonical}".encode()).hexdigest()
    return {
        "intent": intent,
        "intent_source": "rules" if src != "llm" and src != "default" else src,
        "entities": ent,
        "canonical": canonical,
        "cache_key": cache_key,
        "neg_flag": neg_flag,
    }
