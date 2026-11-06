"""Нормализация шагов [0]–[6]."""

from __future__ import annotations

import time
from typing import Any

from assistant_api.normalization.canonicalizer import canonicalize
from assistant_api.normalization.contextualizer import contextualize
from assistant_api.normalization.glossary import all_surface_forms, load_glossary
from assistant_api.normalization.history import DialogMemory
from assistant_api.normalization.history_gate import needs_history
from assistant_api.normalization.input_fix import fix_input, is_confirm_reply
from assistant_api.normalization.schema import DialogState, Entities, InputFix, NormalizedQuery
from assistant_api.normalization.state import topic_switch_by_slots, update_state
from assistant_api.normalization.text_clean import clean_text
from assistant_api.normalization.validator import validate_rewrite
from assistant_api.settings import get_settings


def normalize_query(
    raw: str,
    memory: DialogMemory,
    openai_client=None,
    kb_version: str = "",
    settings=None,
    tracer=None,
) -> NormalizedQuery:
    settings = settings or get_settings()
    t_all = time.perf_counter()
    cleaned = clean_text(raw)
    if tracer:
        tracer.add(0, "raw", f"«{raw}»")
        tracer.add(1, "clean_text", f"«{cleaned}»")

    gloss = load_glossary(settings)
    gloss_names = []
    for t in gloss:
        gloss_names.extend(all_surface_forms(t))

    pending = memory.state.pending_input_fix
    if pending and is_confirm_reply(clean_text(raw)):
        # подтверждение раскладки: обрабатываем сохранённый вопрос
        nq = normalize_query(
            pending,
            _memory_without_pending(memory),
            openai_client,
            kb_version,
            settings,
            tracer,
        )
        nq.input_fix.action = "auto"
        memory.state.pending_input_fix = None
        return nq
    if pending and not is_confirm_reply(clean_text(raw)):
        memory.state.pending_input_fix = None

    ifx = fix_input(
        cleaned if settings.INPUT_FIX_ENABLED else raw,
        history_text=memory.history_text(),
        settings=settings,
        glossary_terms=gloss_names,
    )
    if not settings.INPUT_FIX_ENABLED:
        ifx = InputFix(action="none", text=clean_text(raw))
    if tracer:
        tracer.add(
            2,
            "fix_input",
            f"kinds={ifx.kinds}, layout_ratio {ifx.layout_ratio}, action={ifx.action} → «{ifx.text}»",
        )

    nq = NormalizedQuery(
        raw=raw,
        cleaned=cleaned,
        input_fix=ifx,
        standalone=ifx.text,
        normalizer_version=settings.NORMALIZER_VERSION,
        kb_version=kb_version,
    )

    if ifx.action == "confirm":
        nq.needs_clarification = True
        nq.clarification_kind = "input_fix"
        nq.clarification_question = ifx.confirm_question
        memory.state.pending_input_fix = ifx.text
        return nq

    hist = memory.history_for_llm()
    gate = False
    rule = None
    if settings.NORMALIZATION_ENABLED:
        gate, rule = needs_history(ifx.text, hist, settings)
    nq.history_gate = gate
    nq.history_gate_rule = rule
    nq.depends_on_history = gate
    if tracer:
        tracer.add(3, "needs_history", f"{gate} (правило {rule})")

    if (
        settings.NORMALIZATION_ENABLED
        and settings.CONTEXTUALIZATION_ENABLED
        and gate
        and openai_client is not None
    ):
        res = contextualize(ifx.text, hist, memory.state, gloss_names, openai_client, settings)
        if res.error:
            nq.rewrite_source = "fallback"
            nq.fallback_reason = res.error
            nq.standalone = ifx.text
            nq.topic_switch = False
            nq.topic_switch_source = "none"
            if tracer:
                tracer.add(4, "contextualize", f"fallback {res.error}", skipped=False)
        else:
            data = res.data
            nq.standalone = data.get("standalone") or ifx.text
            nq.depends_on_history = bool(data.get("depends_on_history", True))
            nq.needs_clarification = bool(data.get("needs_clarification"))
            nq.clarification_question = data.get("clarification_question")
            if nq.needs_clarification:
                nq.clarification_kind = "reference"
            ok, reason = validate_rewrite(
                ifx.text,
                nq.standalone,
                [t["content"] for t in hist if t["role"] == "assistant"],
                " ".join(t["content"] for t in hist),
                settings,
            )
            if tracer:
                tracer.add(5, "validate_rewrite", "ok" if ok else f"fail {reason}")
            if not ok:
                nq.standalone = ifx.text
                nq.rewrite_source = "fallback"
                nq.fallback_reason = reason
                nq.topic_switch = False
                nq.topic_switch_source = "none"
            else:
                nq.rewrite_source = "llm"
                nq.topic_switch = bool(data.get("topic_switch"))
                nq.topic_switch_source = "llm"
                ent_raw = data.get("entities") or {}
                try:
                    nq.entities = Entities(**{k: v for k, v in ent_raw.items() if k in Entities.model_fields})
                except Exception:
                    pass
    else:
        nq.standalone = ifx.text
        nq.rewrite_source = "raw"
        if tracer:
            tracer.add(4, "contextualize", "пропущен", skipped=True)
            tracer.add(5, "validate_rewrite", "пропущен", skipped=True)

    if nq.needs_clarification and nq.clarification_kind == "reference":
        return nq

    if settings.CANONICALIZATION_ENABLED or settings.NORMALIZATION_ENABLED:
        can = canonicalize(
            nq.standalone,
            llm_entities=nq.entities,
            kb_version=kb_version,
            settings=settings,
        )
        nq.intent = can["intent"]
        nq.intent_source = can["intent_source"]
        nq.entities = can["entities"]
        nq.canonical = can["canonical"]
        nq.cache_key = can["cache_key"]
        if tracer:
            tracer.add(6, "canonicalize", f"intent={nq.intent} canonical «{nq.canonical}»")
    else:
        can = canonicalize(nq.standalone, kb_version=kb_version, settings=settings)
        nq.canonical = can["canonical"]
        nq.cache_key = can["cache_key"]
        nq.intent = can["intent"]
        nq.entities = can["entities"]

    if nq.topic_switch_source != "llm":
        if hist and (not gate or nq.rewrite_source == "fallback" or not settings.CONTEXTUALIZATION_ENABLED):
            nq.topic_switch = topic_switch_by_slots(memory.state, nq.entities)
            nq.topic_switch_source = "slots" if hist else "none"
        elif not hist:
            nq.topic_switch = False
            nq.topic_switch_source = "none"

    return nq


def _memory_without_pending(memory: DialogMemory) -> DialogMemory:
    clone = DialogMemory(memory.settings)
    clone.turns = list(memory.turns)
    st = memory.state.model_copy()
    st.pending_input_fix = None
    clone.state = st
    return clone


def run_pipeline_steps(*args, **kwargs):
    return normalize_query(*args, **kwargs)
