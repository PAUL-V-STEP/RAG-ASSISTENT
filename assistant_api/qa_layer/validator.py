from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from assistant_api.normalization.period import normalize_period
from assistant_api.normalization.schema import CONDITIONS, INTENTS, LEVELS, NODES
from assistant_api.qa_layer.loader import fact_norm, load_qa_base, load_qa_tests

ID_RE = re.compile(r"^qa\d{3}$")
FORBIDDEN = ["ТО-2", "МПа", "Н·м"]
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def validate_qa(settings, kb_text: str, kb_name: str) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warns: list[str] = []
    base_path = settings.resolve_path("QA_BASE_FILE")
    tests_path = settings.resolve_path("QA_TESTS_FILE")
    data = load_qa_base(base_path)
    if data.get("schema_version") != 2:
        errors.append(
            "устаревший файл: нет answer_group / accept_ids / guard, возьмите версию из комплекта (MANIFEST.md)"
        )
        return errors, warns, data
    recs = data.get("records") or []
    ids = []
    kbn = fact_norm(kb_text)
    by_id = {}
    questions = {}
    for r in recs:
        rid = r.get("id")
        ids.append(rid)
        if not rid or not ID_RE.match(str(rid)):
            errors.append(f"{rid}: id не qaNNN")
        for field in ("question", "answer", "intent", "facts", "source"):
            if field not in r:
                errors.append(f"{rid}: нет поля {field}")
        if r.get("intent") not in INTENTS:
            errors.append(f"{rid}: intent {r.get('intent')}")
        node = r.get("node")
        if node not in NODES and node is not None:
            errors.append(f"{rid}: node {node}")
        lvl = r.get("maintenance_level")
        if lvl not in LEVELS and lvl is not None:
            errors.append(f"{rid}: maintenance_level {lvl}")
        cond = r.get("condition")
        if cond not in CONDITIONS and cond is not None:
            errors.append(f"{rid}: condition {cond}")
        qn = fact_norm(r.get("question", ""))
        if qn in questions:
            errors.append(f"{rid}: дубль вопроса с {questions[qn]}")
        questions[qn] = rid
        for p in r.get("paraphrases") or []:
            pn = fact_norm(p)
            if pn in questions:
                errors.append(f"{rid}: дубль paraphrase с {questions[pn]}")
            questions[pn] = rid
        ans = r.get("answer") or ""
        for w in FORBIDDEN:
            if w in ans:
                errors.append(f"{rid}: в ответе «{w}»")
        for f in r.get("facts") or []:
            if fact_norm(f) not in kbn:
                errors.append(f"{rid}: факт «{f}» не найден в DATA_FILE")
            if fact_norm(f) not in fact_norm(ans):
                errors.append(f"{rid}: факт «{f}» не найден в ответе")
        for n in NUM_RE.findall(ans):
            if fact_norm(n) not in kbn and n not in kb_text:
                if n not in kb_text.replace(",", "."):
                    errors.append(f"{rid}: число {n} нет в DATA_FILE")
        for src in r.get("source") or []:
            if src == "4":
                continue
            if not re.search(rf"(?m)^{re.escape(str(src))}[\.\s]", kb_text):
                errors.append(f"{rid}: пункт {src} не найден")
        per = r.get("period")
        for v in per if isinstance(per, list) else ([per] if per else []):
            if normalize_period(v) != v:
                errors.append(f"{rid}: period «{v}» не канонический")
        by_id[rid] = r
    if len(ids) != len(set(ids)):
        errors.append("дубликаты id")

    groups = defaultdict(list)
    for r in recs:
        groups[(r.get("intent"), r.get("maintenance_level"))].append(r)
    for key, items in groups.items():
        general = [x for x in items if x.get("condition") is None]
        conds = [x for x in items if x.get("condition")]
        for g in general:
            gf = set(g.get("facts") or [])
            other_g = [x for x in general if x["id"] != g["id"]]
            other_facts = set()
            for x in other_g:
                other_facts |= set(x.get("facts") or [])
            for c in conds:
                cf = set(c.get("facts") or [])
                for f in gf:
                    if f in cf and f not in other_facts and len(general) == 1:
                        errors.append(f"{g['id']}: правило 13.6 факт «{f}»")

    for i, a in enumerate(recs):
        for b in recs[i + 1 :]:
            ga = a.get("answer_group") or a["id"]
            gb = b.get("answer_group") or b["id"]
            if ga == gb:
                continue
            if (
                a.get("intent") == b.get("intent")
                and a.get("maintenance_level") == b.get("maintenance_level")
                and a.get("condition") == b.get("condition")
                and set(a.get("facts") or []) == set(b.get("facts") or [])
                and a.get("facts")
            ):
                if fuzz.token_set_ratio(a.get("answer", ""), b.get("answer", "")) >= 85:
                    errors.append(f"{a['id']}/{b['id']}: скрытый дубль")

    if data.get("kb_file") and data.get("kb_file") != kb_name:
        warns.append(f"kb_file «{data.get('kb_file')}» ≠ DATA_FILE «{kb_name}»")

    tests = load_qa_tests(tests_path) if tests_path and tests_path.exists() else {}
    if tests.get("schema_version") not in (None, 2):
        errors.append("qa_tests.yaml schema_version не 2")
    idset = set(by_id)
    groups_map = {r["id"]: r.get("answer_group") for r in recs}
    for t in tests.get("tests") or []:
        for field in ("expected_id", "accept_ids", "must_not", "near"):
            v = t.get(field)
            for i in v if isinstance(v, list) else ([v] if v else []):
                if i not in idset and i != "none":
                    errors.append(f"{t.get('id')}: {field} неизвестный {i}")
        for i in t.get("accept_ids") or []:
            if groups_map.get(i) != groups_map.get(t.get("expected_id")):
                errors.append(f"{t.get('id')}: accept_ids не из группы")
        qn = fact_norm(t.get("question", ""))
        if qn in questions:
            warns.append(f"{t.get('id')}: утечка теста в базу ({questions[qn]})")

    draft = sum(1 for r in recs if r.get("verified_by") == "draft-ai")
    if draft:
        warns.append(f"непроверенных записей: {draft}")
    return errors, warns, data
