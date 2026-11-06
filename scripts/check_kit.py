#!/usr/bin/env python3
"""Проверка согласованности комплекта ТЗ-RAG-AL.

Запуск:
    python check_kit.py                 # файлы комплекта в текущем каталоге
    python check_kit.py --root <путь>   # каталог комплекта или корень проекта

Если файл не найден в каталоге, он ищется по месту в проекте
(таблица «Комплект файлов» ТЗ-RAG-AL). Код выхода: 0 — комплект согласован, 1 — есть ошибки.
Зависимости: pyyaml.
"""
import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

KB_NAME = "ТО автолифта АЛ 3У-М6 — версия для RAG V3.txt"
FILES = {
    "tz": ("ТЗ-RAG-AL.md", "docs"),
    "readme": ("README_RAG-AL.md", "."),
    "env": ("env.example", "."),
    "kb": (KB_NAME, "."),
    "qa_base": ("qa_base.yaml", "assistant_api/data/qa"),
    "qa_tests": ("qa_tests.yaml", "assistant_api/data/qa"),
    "batch": ("batch_example.yaml", "tests/fixtures"),
    "manifest": ("MANIFEST.md", "docs"),
}
PERIOD_RE = re.compile(r"^(\d+(?:-\d+)? (месяц|ч|год|день|неделя)|\d+(?:-\d+)?/(год|месяц|неделя|день))$")
FORBIDDEN_IN_ANSWERS = ["ТО-2", "МПа", "Н·м"]
STALE_MARKERS = {
    "без вопросов-ответов (TXT)": "старое имя базы знаний",
    "generate_with_history_ru": "старый шаблон (нужен generate_extended_ru.txt)",
    "generate_with_verified_ru": "старый шаблон (нужен generate_extended_ru.txt)",
    "docs/plan.md": "план этапа 0 — docs/normalization_plan.md",
    "SEMANTIC_CACHE_THRESHOLD": "порог кэша — SEMANTIC_CACHE_MIN_SIMILARITY",
    "[1]…[11]": "нумерация шагов — [0]–[11]",
    "задание v4": "ссылка на старый документ",
}

errors, notes = [], []


def err(msg):
    errors.append(msg)


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00a0", " ").replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).lower()


def find(root, key):
    name, place = FILES[key]
    for p in (root / name, root / place / name):
        if p.exists():
            return p
    err(f"нет файла {name} (искали в {root} и {root / place})")
    return None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    root = Path(ap.parse_args().root).resolve()
    P = {k: find(root, k) for k in FILES}

    # 1. База знаний
    kb = ""
    if P["kb"]:
        kb = P["kb"].read_text(encoding="utf-8")
        g = sum(1 for l in kb.splitlines() if l.startswith("[глоссарий]"))
        qa_lines = [i + 1 for i, l in enumerate(kb.splitlines()) if l.startswith("[вопрос-ответ]") or l.startswith("ВОПРОСЫ И ОТВЕТЫ")]
        if g != 38:
            err(f"база знаний: строк [глоссарий] {g}, ожидается 38")
        if qa_lines:
            err(f"база знаний содержит блок вопросов-ответов (строка {qa_lines[0]}): подключён не тот файл")
        notes.append(f"база знаний: [глоссарий] {g}, [вопрос-ответ] {len(qa_lines)}")
    kbn = norm(kb)

    # 2. База Q-A
    ids, groups = set(), {}
    if P["qa_base"]:
        b = yaml.safe_load(P["qa_base"].read_text(encoding="utf-8"))
        if b.get("schema_version") != 2:
            err(f"qa_base.yaml: schema_version {b.get('schema_version')}, нужна 2 (устаревший файл)")
        if b.get("kb_file") != KB_NAME:
            err(f"qa_base.yaml: kb_file «{b.get('kb_file')}», ожидается «{KB_NAME}»")
        recs = b.get("records", [])
        ids = {r["id"] for r in recs}
        groups = {r["id"]: r.get("answer_group") for r in recs}
        npar = sum(len(r.get("paraphrases") or []) for r in recs)
        if len(recs) != 50 or npar != 100:
            err(f"qa_base.yaml: записей {len(recs)}, перефразировок {npar}; ожидается 50 и 100")
        byid = {r["id"]: r for r in recs}
        if not (groups.get("qa002") == groups.get("qa003") == "to1_periodicity"):
            err("qa_base.yaml: qa002 и qa003 должны быть в answer_group to1_periodicity")
        if byid.get("qa005", {}).get("facts") != ["25 мин"]:
            err(f"qa_base.yaml: qa005.facts {byid.get('qa005', {}).get('facts')}, ожидается ['25 мин'] (правило 13.6)")
        for r in recs:
            per = r.get("period")
            for v in (per if isinstance(per, list) else [per] if per else []):
                if not PERIOD_RE.match(v):
                    err(f"qa_base.yaml: {r['id']}.period «{v}» не в формате нормализатора (11.3)")
            for w in FORBIDDEN_IN_ANSWERS:
                if w in r.get("answer", ""):
                    err(f"qa_base.yaml: {r['id']}: в ответе «{w}»")
            if kb:
                for f in r.get("facts") or []:
                    if norm(f) not in kbn:
                        err(f"qa_base.yaml: {r['id']}: факт «{f}» не найден в базе знаний")
                    if norm(f) not in norm(r.get("answer", "")):
                        err(f"qa_base.yaml: {r['id']}: факт «{f}» не найден в ответе")
        notes.append(f"qa_base.yaml: schema 2, записей {len(recs)}, перефразировок {npar}")

    # 3. Тесты Q-A
    if P["qa_tests"]:
        t = yaml.safe_load(P["qa_tests"].read_text(encoding="utf-8"))
        if t.get("schema_version") != 2:
            err(f"qa_tests.yaml: schema_version {t.get('schema_version')}, нужна 2 (устаревший файл)")
        tests = t.get("tests", [])
        c = Counter(x.get("type") for x in tests)
        want = {"match": 50, "contrast": 26, "no_match": 12, "dialog": 6}
        if len(tests) != 94 or dict(c) != want:
            err(f"qa_tests.yaml: {len(tests)} тестов {dict(c)}, ожидается 94 {want}")
        tid = {x["id"]: x for x in tests}
        if not {"c13a", "c13b"} <= set(tid):
            err("qa_tests.yaml: нет пары c13")
        for k, g in (("n02", "unit_out_of_corpus"), ("n04", "level_out_of_corpus")):
            if tid.get(k, {}).get("guard") != g:
                err(f"qa_tests.yaml: {k}.guard должен быть {g}")
        for k in ("t002", "t003"):
            if not tid.get(k, {}).get("accept_ids"):
                err(f"qa_tests.yaml: {k} без accept_ids")
        for x in tests:
            for f in ("expected_id", "accept_ids", "must_not", "near"):
                v = x.get(f)
                for i in (v if isinstance(v, list) else [v] if v else []):
                    if ids and i not in ids and i != "none":
                        err(f"qa_tests.yaml: {x['id']}.{f} ссылается на несуществующую {i}")
            for i in x.get("accept_ids") or []:
                if groups and (groups.get(i) is None or groups.get(i) != groups.get(x.get("expected_id"))):
                    err(f"qa_tests.yaml: {x['id']}: accept_ids {i} не из группы expected_id")
            if x.get("type") == "dialog" and x.get("stage") != "B":
                err(f"qa_tests.yaml: {x['id']}: dialog без stage: B")
        notes.append(f"qa_tests.yaml: schema 2, тестов {len(tests)} {dict(c)}")

    # 4. Пакетный пример
    if P["batch"]:
        bt = yaml.safe_load(P["batch"].read_text(encoding="utf-8"))
        items = bt.get("items", [])
        bids = {i["id"] for i in items}
        for it in items:
            for turn in it.get("turns", []):
                e = turn.get("expect") or {}
                for q in e.get("qa_id") or []:
                    if ids and q not in ids:
                        err(f"batch_example.yaml: {it['id']}: qa_id {q} нет в базе")
                ref = e.get("same_cache_key_as")
                if ref and ref not in bids:
                    err(f"batch_example.yaml: {it['id']}: same_cache_key_as {ref} не найден")
        notes.append(f"batch_example.yaml: элементов {len(items)}")

    # 5. Переменные окружения: ТЗ 6.3 == env.example
    if P["tz"] and P["env"]:
        tz = P["tz"].read_text(encoding="utf-8")
        sec = tz[tz.index("### 6.3."):tz.index("\n## 7.")]
        tz_vars = set(re.findall(r"^\| `([A-Z0-9_]+)` \|", sec, re.M))
        env_txt = P["env"].read_text(encoding="utf-8")
        env_vars = set(re.findall(r"^([A-Z0-9_]+)=", env_txt, re.M))
        if tz_vars != env_vars:
            err(f"env.example и ТЗ 6.3 расходятся: только в ТЗ {sorted(tz_vars - env_vars)}, только в env {sorted(env_vars - tz_vars)}")
        m = re.search(r"^DATA_FILE=(.*)$", env_txt, re.M)
        if not m or m.group(1).strip() != KB_NAME:
            err(f"env.example: DATA_FILE должен быть «{KB_NAME}»")
        bad = [v for v in tz_vars if v.endswith("_THRESHOLD")]
        if bad:
            err(f"ТЗ 6.3: пороги без метрики в имени: {bad}")
        notes.append(f"переменных окружения: {len(tz_vars)}")

    # 6. Устаревшие ссылки и нумерация шагов
    for key in ("tz", "readme", "batch", "qa_base", "qa_tests", "env"):
        if not P[key]:
            continue
        txt = P[key].read_text(encoding="utf-8")
        for mk, why in STALE_MARKERS.items():
            if mk in txt and not (key == "tz" and mk == "задание v4"):
                err(f"{P[key].name}: найдено «{mk}» — {why}")
    if P["readme"]:
        rd = P["readme"].read_text(encoding="utf-8")
        missing = [n for n in range(12) if f"[{n}]" not in rd]
        if missing:
            err(f"README_RAG-AL.md: нет шагов {missing}")

    # 7. Контрольные суммы
    if P["manifest"]:
        man = P["manifest"].read_text(encoding="utf-8")
        for key, (name, _) in FILES.items():
            if key == "manifest" or not P[key]:
                continue
            m = re.search(r"\| `" + re.escape(name) + r"` \|.*?\| `([0-9a-f]{64})` \|", man)
            if not m:
                err(f"MANIFEST.md: нет sha256 для {name}")
            elif m.group(1) != sha(P[key]):
                err(f"MANIFEST.md: sha256 {name} не совпадает (файл изменён или устарел)")

    for n in notes:
        print("·", n)
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for e in errors:
            print("  ✗", e)
        return 1
    print("\nКомплект согласован.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
