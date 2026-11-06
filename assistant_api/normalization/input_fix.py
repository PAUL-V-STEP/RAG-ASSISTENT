"""Коррекция раскладки и регистра (шаг [2])."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from assistant_api.normalization.schema import InputFix
from assistant_api.normalization.text_clean import clean_text, fold_yo
from assistant_api.settings import Settings, get_settings

EN = "`qwertyuiop[]asdfghjkl;'zxcvbnm,./"
RU = "ёйцукенгшщзхъфывапролджэячсмитьбю."
EN_UP = '~QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?'
RU_UP = 'ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,'

EN2RU = str.maketrans(EN + EN_UP, RU + RU_UP)
RU2EN = str.maketrans(RU + RU_UP, EN + EN_UP)

HOMO_LATIN = set("ABCEHKMOPTXaceopxy")
HOMO_TO_CYR = str.maketrans(
    "ABCEHKMOPTXaceopxy",
    "АВСЕНКМОРТХасеорху",
)
HOMO_TO_LAT = str.maketrans(
    "АВСЕНКМОРТХасеорху",
    "ABCEHKMOPTXaceopxy",
)

ABBR = ["ЕО", "ТО-1", "СО", "КОМ", "АКБ", "ГОСТ", "ATEGO", "ATEGO"]
CONFIRM_WORDS = {"да", "верно", "ага", "yes", "+"}

_LATIN_RE = re.compile(r"[A-Za-z]")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")


@lru_cache(maxsize=1)
def _whitelist(path: str) -> set[str]:
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(w) for w in (p.get("words") or [])}


def to_cyrillic(text: str) -> str:
    return text.translate(EN2RU)


def to_latin(text: str) -> str:
    return text.translate(RU2EN)


def _is_whitelist(token: str, wl: set[str]) -> bool:
    low = token.lower()
    return any(w.lower() == low for w in wl)


def _restore_abbr(text: str, wl: set[str]) -> str:
    out = text
    for abbr in list(ABBR) + list(wl):
        out = re.sub(rf"(?i)\b{re.escape(abbr)}\b", abbr, out)
    # ТО-1 variants
    out = re.sub(r"(?i)\bто-1\b", "ТО-1", out)
    out = re.sub(r"(?i)\bего\b", lambda m: m.group(0), out)  # no-op keep
    return out


def _fix_homoglyph(token: str, wl: set[str]) -> Optional[str]:
    has_cyr = bool(_CYR_RE.search(token))
    has_lat = bool(_LATIN_RE.search(token))
    if not (has_cyr and has_lat):
        return None
    latin_chars = [c for c in token if c.isalpha() and c in HOMO_LATIN or (c.isascii() and c.isalpha())]
    cyr_chars = [c for c in token if _CYR_RE.match(c or "")]
    lat_in_token = [c for c in token if c.isascii() and c.isalpha()]
    if lat_in_token and all(c in HOMO_LATIN for c in lat_in_token) and has_cyr:
        return token.translate(HOMO_TO_CYR)
    if len(lat_in_token) >= len(cyr_chars) and has_cyr:
        lat = token.translate(HOMO_TO_LAT)
        if _is_whitelist(lat, wl):
            return lat
    return None


def _capslock_inverted(text: str) -> bool:
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text)
    if not words:
        return False
    inverted = 0
    lockish = 0
    for w in words:
        if w[0].islower() and w[1:].isupper():
            inverted += 1
            lockish += 1
        elif w.isupper():
            lockish += 1
    if inverted / len(words) >= 0.6:
        return True
    return inverted >= 1 and lockish / len(words) >= 0.6


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.9


def _split_keep(text: str) -> list[tuple[str, bool]]:
    parts = re.split(r"(\s+)", text)
    out = []
    for p in parts:
        out.append((p, bool(p.strip()) and not p.isspace()))
    return out


def fix_input(
    text: str,
    history_text: str = "",
    settings: Settings | None = None,
    glossary_terms: list[str] | None = None,
    morph=None,
) -> InputFix:
    settings = settings or get_settings()
    if not settings.INPUT_FIX_ENABLED:
        cleaned = clean_text(text)
        return InputFix(action="none", text=cleaned)

    wl_path = settings.resolve_path("LAYOUT_LATIN_WHITELIST_FILE")
    wl = _whitelist(str(wl_path)) if wl_path and wl_path.exists() else set()
    kinds: list[str] = []
    work = text
    changed: list[tuple[str, str]] = []

    if settings.HOMOGLYPH_FIX_ENABLED:
        tokens = re.findall(r"\S+|\s+", work)
        new_tokens = []
        for tok in tokens:
            if tok.isspace():
                new_tokens.append(tok)
                continue
            core = tok.strip(".,;:!?")
            fixed = _fix_homoglyph(core, wl)
            if fixed and fixed != core:
                kinds.append("homoglyph")
                changed.append((core, fixed))
                new_tokens.append(tok.replace(core, fixed, 1))
            else:
                new_tokens.append(tok)
        work = "".join(new_tokens)

    # Caps Lock on Latin first if inverted latin, then layout (f14)
    if settings.CAPSLOCK_FIX_ENABLED and _capslock_inverted(work) and _LATIN_RE.search(work) and not _CYR_RE.search(work):
        work = work.swapcase()
        kinds.append("capslock")

    unresolved: list[str] = []
    layout_ratio = None
    latin_tokens = 0
    action = "none"
    confirm_q = None

    if settings.LAYOUT_FIX_ENABLED:
        tokens = re.findall(r"\S+|\s+", work)
        candidates = []
        mapped = []
        for tok in tokens:
            if tok.isspace():
                mapped.append(tok)
                continue
            punct_end = ""
            core = tok
            if core.endswith("&") or core.endswith("?"):
                punct_end = "?"
                core = core[:-1]
            # reverse whitelist: russian layout of latin term
            lat_guess = to_latin(core)
            if _is_whitelist(lat_guess, wl) and _CYR_RE.search(core) and not _LATIN_RE.search(core):
                mapped.append(lat_guess + punct_end)
                changed.append((core, lat_guess))
                kinds.append("layout")
                continue
            if _is_whitelist(core, wl) or re.fullmatch(r"[\d°%.,;:()\-]+", core) or "°C" in core:
                mapped.append(tok)
                continue
            if _LATIN_RE.search(core):
                cyr = to_cyrillic(core)
                # trailing punctuation dictionary
                if core.endswith(".") and not cyr.endswith("ю"):
                    pass
                candidates.append((core, cyr))
                mapped.append(cyr + punct_end)
            else:
                mapped.append(tok)
        if candidates:
            latin_tokens = len(candidates)
            recognized = 0
            glossary_hits = 0
            gloss = [fold_yo(g).lower() for g in (glossary_terms or [])]
            hist = fold_yo(history_text).lower()
            for orig, cyr in candidates:
                ok = _recognized(cyr, morph, gloss, hist, wl)
                if ok:
                    recognized += 1
                    cyr_l = fold_yo(cyr).lower().strip(".,;:!?")
                    if any(cyr_l == g or cyr_l in g.split() for g in gloss) or cyr_l in hist:
                        glossary_hits += 1
                else:
                    unresolved.append(cyr)
            ratio = recognized / latin_tokens if latin_tokens else 0.0
            bonus = settings.LAYOUT_CONTEXT_BONUS * glossary_hits / latin_tokens if latin_tokens else 0
            ratio = min(1.0, ratio + bonus)
            layout_ratio = round(ratio, 3)
            auto_ok = ratio >= settings.LAYOUT_AUTO_MIN_RATIO and (
                latin_tokens >= settings.LAYOUT_MIN_LATIN_TOKENS or glossary_hits > 0
            )
            if auto_ok:
                action = "auto"
                kinds.append("layout")
                work = "".join(mapped)
                for orig, cyr in candidates:
                    if orig != cyr:
                        changed.append((orig, cyr))
            elif ratio >= settings.LAYOUT_CONFIRM_MIN_RATIO:
                unclear = settings.LAYOUT_UNCLEAR_ACTION
                kinds.append("layout")
                work = "".join(mapped)
                for orig, cyr in candidates:
                    changed.append((orig, cyr))
                if unclear == "confirm":
                    action = "confirm"
                    preview = clean_text(work)
                    confirm_q = (
                        "Похоже, вопрос набран в латинской раскладке. Вы имели в виду:\n"
                        f"«{preview}»\n"
                        "Ответьте «да» или задайте вопрос заново."
                    )
                elif unclear == "answer_with_note":
                    action = "answer_with_note"
                else:
                    action = "none"
                    work = text
            elif ratio > 0:
                action = "none"
            else:
                action = "none"

    if settings.CAPSLOCK_FIX_ENABLED:
        if _capslock_inverted(work):
            work = work.swapcase()
            if "capslock" not in kinds:
                kinds.append("capslock")
            if action == "none":
                action = "auto"
        elif _all_caps(work):
            work = work.lower()
            if work:
                work = work[0].upper() + work[1:]
            if "capslock" not in kinds:
                kinds.append("capslock")
            if action == "none":
                action = "auto"

    # unique kinds preserve order
    seen = []
    for k in kinds:
        if k not in seen:
            seen.append(k)
    work = _restore_abbr(work, wl)
    cleaned = clean_text(work)
    if action == "none" and not seen:
        cleaned = clean_text(text)
    return InputFix(
        action=action if seen or action != "none" else "none",
        text=cleaned,
        kinds=seen,  # type: ignore
        layout_ratio=layout_ratio,
        latin_tokens=latin_tokens,
        changed_tokens=changed,
        unresolved_tokens=unresolved,
        confirm_question=confirm_q,
    )


_FREQ_RU = {
    "как", "часто", "проводится", "сколько", "на", "воды", "для", "что",
    "запрещается", "при", "техническом", "обслуживании", "рабочую", "жидкость",
    "гидросистемы", "зимой", "прогреть", "минут", "прогревать", "проверять",
    "ли", "пальцы", "раз", "в", "три", "года", "трудоемкость", "да",
}


def _recognized(word: str, morph, gloss: list[str], hist: str, wl: set[str]) -> bool:
    w = fold_yo(word).strip(".,;:!?")
    low = w.lower()
    if morph is not None:
        try:
            if morph.word_is_known(low):
                return True
        except Exception:
            pass
    if any(low == g or low in g.split() or g == low for g in gloss):
        return True
    if low in hist.split():
        return True
    if low in {"то-1", "ео", "со", "ком", "акб", "магос", "унидем"}:
        return True
    if low in _FREQ_RU:
        return True
    return False


def is_confirm_reply(text: str) -> bool:
    t = fold_yo(clean_text(text)).lower().strip(" .!?")
    return t in CONFIRM_WORDS
