from __future__ import annotations


def check_expect(expect: dict, nq, result: dict, qa_id) -> list[str]:
    fails = []
    if not expect:
        return fails
    ifx = expect.get("input_fix")
    if ifx:
        if ifx.get("action") and nq.input_fix.action != ifx["action"]:
            fails.append(f"input_fix.action {nq.input_fix.action} != {ifx['action']}")
        if ifx.get("text") and nq.input_fix.text != ifx["text"]:
            fails.append("input_fix.text")
        if ifx.get("kinds") and list(nq.input_fix.kinds) != list(ifx["kinds"]):
            fails.append(f"kinds {nq.input_fix.kinds}")
    if expect.get("intent") and nq.intent != expect["intent"]:
        fails.append(f"intent {nq.intent}")
    if expect.get("maintenance_level") and nq.entities.maintenance_level != expect["maintenance_level"]:
        fails.append("maintenance_level")
    if expect.get("decision") and (result.get("qa_decision") or "none") != expect["decision"]:
        fails.append("decision")
    want_ids = expect.get("qa_id")
    if want_ids:
        got = result.get("qa_id")
        if got not in want_ids:
            fails.append(f"qa_id {got} not in {want_ids}")
    text = (result.get("answer") or "") + " " + (nq.standalone or "")
    for s in expect.get("expected_contains") or []:
        if s.lower() not in text.lower() and s.lower() not in (nq.standalone or "").lower():
            # facts may appear after generation; for normalize stage check standalone+canonical
            if s not in (nq.canonical or "") and s not in (nq.standalone or ""):
                fails.append(f"missing {s}")
    for s in expect.get("not_contains") or []:
        if s in (result.get("answer") or ""):
            fails.append(f"forbidden {s}")
    for s in expect.get("standalone_contains") or []:
        st = (nq.standalone or "").lower()
        if s.lower() not in st:
            fails.append(f"standalone missing {s}")
    for s in expect.get("standalone_not_contains") or []:
        if s.lower() in (nq.standalone or "").lower():
            fails.append(f"standalone has {s}")
    if expect.get("depends_on_history") is True and not nq.depends_on_history:
        fails.append("depends_on_history")
    if expect.get("clarification") and not result.get("clarification") and not nq.needs_clarification:
        fails.append("clarification")
    return fails
