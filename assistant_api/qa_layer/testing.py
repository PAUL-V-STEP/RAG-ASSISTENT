from __future__ import annotations

from assistant_api.normalization.history import DialogMemory
from assistant_api.normalization.pipeline import normalize_query
from assistant_api.qa_layer.loader import load_qa_tests
from assistant_api.qa_layer.runtime import QARuntime


def run_tests(settings, runtime: QARuntime, openai_client=None, types=None, ignore_verified=False):
    path = settings.resolve_path("QA_TESTS_FILE")
    data = load_qa_tests(path)
    types = types or [x.strip() for x in str(settings.QA_TOOL_TEST_TYPES).split(",") if x.strip()]
    results = []
    for t in data.get("tests") or []:
        if t.get("type") not in types:
            continue
        if t.get("type") == "dialog" and t.get("stage") == "B" and not settings.CONTEXTUALIZATION_ENABLED:
            results.append({**t, "status": "skipped", "pass": None})
            continue
        mem = DialogMemory(settings)
        if t.get("type") == "dialog":
            for h in t.get("history") or []:
                if h["role"] == "user":
                    nq0 = normalize_query(h["content"], mem, None, runtime.kb_version, settings)
                    mem.add_user(nq0.input_fix.text, raw=h["content"])
                    from assistant_api.normalization.state import update_state

                    mem.state = update_state(mem.state, nq0.entities, nq0.topic_switch, 0)
                else:
                    mem.add_assistant(h["content"])
        nq = normalize_query(t["question"], mem, openai_client if t.get("type") == "dialog" else None, runtime.kb_version, settings)
        if nq.needs_clarification and t.get("expected_clarification"):
            results.append({**t, "status": "pass", "pass": True, "decision": "none", "clarification": True})
            continue
        dec = runtime.lookup(nq, ignore_verified=ignore_verified)
        ok = _eval(t, dec, nq)
        results.append(
            {
                **t,
                "status": "pass" if ok else "fail",
                "pass": ok,
                "decision": dec.decision,
                "got_id": dec.qa_id,
                "reason": dec.reason,
                "d1": dec.d1,
                "candidates": dec.candidates,
            }
        )
    return results


def _eval(t, dec, nq) -> bool:
    accept = set(t.get("accept_ids") or [])
    accept.add(t.get("expected_id"))
    typ = t["type"]
    if typ == "match":
        if dec.decision not in ("direct", "hybrid"):
            return False
        if t.get("expected_mode") and dec.decision != t["expected_mode"]:
            return False
        return dec.qa_id in accept
    if typ == "contrast":
        if dec.qa_id and dec.qa_id in (t.get("must_not") or []):
            return False
        return dec.decision == "none" or dec.qa_id in accept
    if typ == "no_match":
        if dec.decision != "none":
            return False
        if t.get("guard") and dec.reason != t["guard"]:
            return False
        return True
    if typ == "dialog":
        if t.get("expected_clarification"):
            return bool(nq.needs_clarification)
        return dec.qa_id == t.get("expected_id") or (
            t.get("expected_id") == "none" and dec.decision == "none"
        )
    return False
