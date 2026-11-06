"""Правила intent (раздел 11.2)."""

from __future__ import annotations

from assistant_api.normalization.schema import Intent
from assistant_api.normalization.text_clean import fold_yo, tokenize


def detect_intent(text: str) -> tuple[Intent, str]:
    raw = fold_yo(text or "")
    low = raw.lower()
    tokens = tokenize(raw)
    pairs = [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]

    def has_prefix(prefixes: tuple[str, ...]) -> bool:
        return any(any(t.startswith(p) for p in prefixes) for t in tokens)

    def has_tokens(*words: str) -> bool:
        return any(w in tokens for w in words)

    def has_phrase(*phrases: str) -> bool:
        return any(p in low for p in phrases)

    qty = (
        has_tokens("сколько", "скольки")
        or has_phrase("какую температуру", "какая температура", "какое значение", "до какой", "до каких")
    )
    limit = (
        has_tokens("нельзя")
        or has_prefix(("запрещ", "превыша"))
        or has_phrase("не более", "не менее", "не меньше", "не выше", "не ниже")
        or has_tokens("выше", "ниже")
    )
    if qty and limit:
        return "parameter_value", "rules:0"

    last_word = ""
    stripped = low.rstrip(" ?!.")
    last_tokens = tokenize(stripped)
    if last_tokens:
        last_word = last_tokens[-1]
    if (
        has_prefix(("запрещ", "опасн"))
        or has_tokens("нельзя")
        or has_phrase("можно ли", "допускается ли", "разрешается ли")
        or last_word in {"можно", "нельзя"}
    ):
        return "safety", "rules:1"

    if has_phrase("нужно ли", "надо ли", "следует ли", "обязательно ли", "необходимо ли"):
        return "procedure", "rules:2"

    if (
        has_phrase("как часто", "периодичн", "через сколько", "раз в", "когда проводится", "когда нужно проводить")
        or has_tokens("интервал", "периодичность", "наработка")
    ):
        return "periodicity", "rules:3"

    if (
        has_tokens("сколько", "какова", "каков", "норма", "трудоемкость", "пропорция")
        or has_prefix(("максимальн",))
        or has_phrase("какая норма", "до скольки", "какую температуру")
    ):
        return "parameter_value", "rules:4"

    if has_phrase("какие детали", "какие запчасти", "что заменить"):
        return "parts", "rules:5"

    if (
        has_tokens("как", "где")
        or has_phrase(
            "что нужно",
            "что делать",
            "в каком порядке",
            "какие работы",
            "что входит",
            "кто допускается",
        )
    ):
        return "procedure", "rules:6"

    return "other", "default"
