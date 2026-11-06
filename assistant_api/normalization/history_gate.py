"""Фильтр needs_history без LLM (шаг [3])."""

from __future__ import annotations

from assistant_api.normalization.glossary import load_glossary, subject_terms, all_surface_forms
from assistant_api.normalization.text_clean import fold_yo, tokenize

FIRST = {"а", "и", "еще", "тогда", "также"}
PRONOUNS = {
    "он", "она", "оно", "они", "его", "ее", "их", "ему", "ей", "им", "ним", "ней", "них",
    "этот", "эта", "это", "эти", "этого", "этой", "этим",
    "тот", "та", "те", "там", "туда", "здесь", "тут", "аналогично",
}
PAIRS = {("такой", "же"), ("то", "же"), ("тот", "же")}


def needs_history(text: str, history: list | None, settings=None) -> tuple[bool, int | None]:
    if not history:
        return False, 1
    tokens = tokenize(text)
    if tokens and tokens[0] in FIRST:
        return True, 2
    if any(t in PRONOUNS for t in tokens):
        return True, 3
    for i in range(len(tokens) - 1):
        if (tokens[i], tokens[i + 1]) in PAIRS:
            return True, 3
    if not _has_subject_term(text, settings):
        return True, 4
    return False, 5


def _has_subject_term(text: str, settings=None) -> bool:
    from assistant_api.normalization.canonicalizer import lemmatize_token

    terms = subject_terms(load_glossary(settings))
    low = fold_yo(text).lower()
    toks = set(tokenize(text))
    lemmas = {lemmatize_token(t) for t in toks}
    for term in terms:
        for form in all_surface_forms(term):
            f = fold_yo(form).lower()
            if " " in f or any(ch.isdigit() for ch in f):
                if f in low:
                    return True
            else:
                fl = lemmatize_token(f)
                if f in toks or f.replace("-", "") in toks or fl in lemmas:
                    return True
    return False
