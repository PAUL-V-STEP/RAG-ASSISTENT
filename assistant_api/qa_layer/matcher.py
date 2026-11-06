from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from assistant_api.normalization.schema import NormalizedQuery
from assistant_api.normalization.text_clean import fold_yo, tokenize


@dataclass
class QADecision:
    decision: str = "none"
    qa_id: Optional[str] = None
    record: Optional[dict] = None
    d1: Optional[float] = None
    margin: Optional[float] = None
    reason: Optional[str] = None
    candidates: list[dict] = field(default_factory=list)
    skipped: bool = False


def stop_rules(text: str, out_of_corpus_units: list[str]) -> Optional[str]:
    low = fold_yo(text)
    if re.search(r"ТО-\d", text, re.I):
        m = re.search(r"ТО-(\d+)", text, re.I)
        if m and m.group(1) not in {"1"}:
            return "level_out_of_corpus"
    for u in out_of_corpus_units:
        if u.lower() in low.lower():
            return "unit_out_of_corpus"
    return None


def _overlap(query: str, rec: dict) -> float:
    q = set(tokenize(query or ""))
    if not q:
        return 0.0
    texts = [rec.get("question") or ""] + list(rec.get("paraphrases") or [])
    best = 0.0
    for text in texts:
        s = set(tokenize(text))
        if not s:
            continue
        best = max(best, len(q & s) / len(q | s))
    return best


def slots_ok(nq: NormalizedQuery, rec: dict) -> tuple[bool, str]:
    e = nq.entities
    if e.maintenance_level and rec.get("maintenance_level") and e.maintenance_level != rec.get("maintenance_level"):
        return False, "level"
    if e.node and rec.get("node") and e.node != rec.get("node"):
        return False, "node"
    rec_period = rec.get("period")
    periods = rec_period if isinstance(rec_period, list) else ([rec_period] if rec_period else [])
    if e.period and periods and e.period not in periods:
        return False, "period"
    rc = rec.get("condition")
    qc = e.condition
    if (rc or None) != (qc or None):
        return False, "condition"
    return True, "ok"


def decide(
    nq: NormalizedQuery,
    records: list[dict],
    distances: dict[str, float],
    settings,
    ignore_verified: bool = False,
    out_units: list[str] | None = None,
) -> QADecision:
    if settings.QA_STOP_RULES_ENABLED:
        reason = stop_rules(nq.standalone or nq.raw, out_units or [])
        if reason:
            return QADecision(decision="none", reason=reason)
    by_id = {r["id"]: r for r in records if r.get("status") != "disabled"}
    cands = []
    for rec in by_id.values():
        ok, why = slots_ok(nq, rec)
        d = distances.get(rec["id"])
        if nq.canonical and rec.get("_canonical") == nq.canonical and ok:
            d = 0.0
        if d is None:
            continue
        overlap = _overlap(nq.standalone or nq.raw, rec)
        if not ok:
            cands.append(
                {
                    "qa_id": rec["id"],
                    "d": d,
                    "slots_ok": False,
                    "fail": why,
                    "group": rec.get("answer_group") or rec["id"],
                    "overlap": overlap,
                }
            )
            continue
        cands.append(
            {
                "qa_id": rec["id"],
                "d": d,
                "slots_ok": True,
                "group": rec.get("answer_group") or rec["id"],
                "overlap": overlap,
            }
        )
    ok_cands = [c for c in cands if c["slots_ok"]]
    ok_cands.sort(key=lambda x: (x["d"], -x["overlap"]))
    if not ok_cands:
        return QADecision(decision="none", reason="no_candidates", candidates=cands)
    best = ok_cands[0]
    others = [c for c in ok_cands if c["group"] != best["group"]]
    margin = (others[0]["d"] - best["d"]) if others else math.inf
    rec = by_id[best["qa_id"]]
    d1 = best["d"]
    verified = rec.get("verified_by") not in (None, "draft-ai") or ignore_verified or not settings.QA_REQUIRE_VERIFIED
    intent_ok = rec.get("intent") == nq.intent
    # numbers in query
    nums_block_direct = False
    blob = " ".join([rec.get("question", ""), " ".join(rec.get("paraphrases") or []), rec.get("answer", "")])
    for n in nq.entities.numbers:
        if n.split()[0] not in blob and n not in blob:
            nums_block_direct = True
    decision = "none"
    reason = "above_hybrid"
    if d1 <= settings.QA_DIRECT_MAX_DISTANCE and margin >= settings.QA_MIN_MARGIN and intent_ok and verified and not nums_block_direct:
        decision = "direct"
        reason = "within_direct"
    elif d1 <= settings.QA_HYBRID_MAX_DISTANCE:
        decision = "hybrid"
        reason = "within_hybrid" if (verified or not settings.QA_REQUIRE_VERIFIED) else "draft-ai"
        if not intent_ok:
            reason = "intent_mismatch"
        if nums_block_direct:
            reason = "query_number"
    mode = settings.QA_MODE
    if mode == "hybrid_only" and decision == "direct":
        decision = "hybrid"
        reason = "hybrid_only"
    return QADecision(
        decision=decision,
        qa_id=best["qa_id"] if decision != "none" else None,
        record=rec if decision != "none" else None,
        d1=d1,
        margin=None if margin is math.inf else margin,
        reason=reason,
        candidates=cands[: settings.TRACE_SHOW_CANDIDATES],
    )
