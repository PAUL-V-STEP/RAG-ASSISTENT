from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from assistant_api.qa_layer.loader import load_qa_base
from assistant_api.qa_layer.matcher import QADecision, decide
from assistant_api.qa_layer.store import QAStore
from assistant_api.normalization.canonicalizer import canonicalize


class QARuntime:
    def __init__(self, settings, vector_store, kb_version: str):
        self.settings = settings
        self.vs = vector_store
        self.kb_version = kb_version
        self.store = QAStore(settings, vector_store, kb_version)
        self.records = []
        self.manifest = {}
        self.stale = False
        self._load()

    def _load(self):
        mpath = self.settings.resolve_path("QA_MANIFEST_FILE")
        if mpath and mpath.exists():
            self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
            if self.manifest.get("kb_version") != self.kb_version:
                self.stale = True
        bpath = self.settings.resolve_path("QA_BASE_FILE")
        if bpath and bpath.exists():
            data = load_qa_base(bpath)
            self.records = data.get("records") or []
            for r in self.records:
                can = canonicalize(r["question"], kb_version=self.kb_version, settings=self.settings)
                r["_canonical"] = can["canonical"]

    def lookup(self, nq, ignore_verified=False) -> QADecision:
        if not self.settings.QA_LAYER_ENABLED or self.settings.QA_MODE == "off":
            return QADecision(decision="none", skipped=True, reason="disabled")
        if self.stale:
            return QADecision(decision="none", skipped=True, reason="stale")
        units = self.manifest.get("out_of_corpus_units") or ["МПа", "Н·м"]
        hits = self.store.query(nq.standalone, self.vs._create_embedding, self.settings.QA_TOP_K)
        dist = {}
        for h in hits:
            qid = h["qa_id"]
            d = h["distance"] if h["distance"] is not None else 1.0
            dist[qid] = min(dist.get(qid, 1.0), d)
        dec = decide(nq, self.records, dist, self.settings, ignore_verified=ignore_verified, out_units=units)
        self._log(nq, dec)
        return dec

    def _log(self, nq, dec: QADecision):
        path = self.settings.resolve_path("QA_LOG_FILE")
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now().isoformat(),
            "mode": self.settings.QA_MODE,
            "raw": nq.raw,
            "standalone": nq.standalone,
            "canonical": nq.canonical,
            "decision": dec.decision,
            "qa_id": dec.qa_id,
            "d1": dec.d1,
            "margin": dec.margin,
            "reason": dec.reason,
            "step_id": 7,
            "kb_version": self.kb_version,
            "candidates": dec.candidates,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
