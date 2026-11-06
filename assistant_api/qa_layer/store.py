from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import chromadb

from assistant_api.normalization.canonicalizer import canonicalize
from assistant_api.qa_layer.loader import load_qa_base


def record_hash(rec: dict) -> str:
    blob = json.dumps(
        {k: rec.get(k) for k in ("id", "question", "paraphrases", "answer", "facts", "status")},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def collection_name(settings, kb_version: str) -> str:
    return f"{settings.QA_COLLECTION_PREFIX}{kb_version}"


class QAStore:
    def __init__(self, settings, vector_store, kb_version: str):
        self.settings = settings
        self.vs = vector_store
        self.kb_version = kb_version
        self.client = vector_store.client
        self.name = collection_name(settings, kb_version)

    def get_or_create(self, rebuild=False):
        if rebuild:
            try:
                self.client.delete_collection(self.name)
            except Exception:
                pass
        try:
            return self.client.get_collection(self.name)
        except Exception:
            return self.client.create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def build(self, records: list[dict], rebuild=False) -> dict:
        coll = self.get_or_create(rebuild=rebuild)
        existing = {}
        try:
            got = coll.get(include=["metadatas"])
            for i, mid in enumerate(got.get("ids") or []):
                existing[mid] = (got["metadatas"][i] or {}).get("rec_hash")
        except Exception:
            existing = {}
        wanted_ids = []
        added = updated = 0
        docs, ids, embs, metas = [], [], [], []
        for rec in records:
            if rec.get("status") == "disabled":
                continue
            rh = record_hash(rec)
            texts = [rec["question"], *(rec.get("paraphrases") or [])]
            kinds = ["question"] + ["paraphrase"] * len(rec.get("paraphrases") or [])
            can = canonicalize(rec["question"], kb_version=self.kb_version, settings=self.settings)
            rec["_canonical"] = can["canonical"]
            period = rec.get("period")
            period_s = ";".join(period) if isinstance(period, list) else (period or "")
            for i, (text, kind) in enumerate(zip(texts, kinds)):
                vid = f"{rec['id']}:{kind}:{i}"
                wanted_ids.append(vid)
                if existing.get(vid) == rh and not rebuild:
                    continue
                if vid in existing:
                    updated += 1
                else:
                    added += 1
                docs.append(text)
                ids.append(vid)
                embs.append(self.vs._create_embedding(text))
                metas.append(
                    {
                        "qa_id": rec["id"],
                        "answer_group": rec.get("answer_group") or rec["id"],
                        "kind": kind,
                        "intent": rec.get("intent") or "",
                        "node": rec.get("node") or "",
                        "maintenance_level": rec.get("maintenance_level") or "",
                        "condition": rec.get("condition") or "",
                        "period": period_s,
                        "canonical": rec["_canonical"],
                        "verified": rec.get("verified_by") not in (None, "draft-ai"),
                        "kb_version": self.kb_version,
                        "rec_hash": rh,
                    }
                )
        stale = [i for i in existing if i not in wanted_ids]
        if stale:
            coll.delete(ids=stale)
        if docs:
            # delete updated ids first
            try:
                coll.delete(ids=ids)
            except Exception:
                pass
            coll.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        unchanged = len(wanted_ids) - added - updated
        manifest = {
            "kb_version": self.kb_version,
            "embedding_model": "text-embedding-3-small",
            "normalizer_version": self.settings.NORMALIZER_VERSION,
            "built_at": datetime.now().isoformat(),
            "records": len(records),
            "vectors": len(wanted_ids),
            "hashes": {r["id"]: record_hash(r) for r in records},
            "out_of_corpus_units": ["МПа", "Н·м"],
        }
        mpath = self.settings.resolve_path("QA_MANIFEST_FILE")
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"added": added, "updated": updated, "deleted": len(stale), "unchanged": unchanged, "vectors": len(wanted_ids)}

    def query(self, text: str, embed, top_k: int) -> list[dict]:
        try:
            coll = self.client.get_collection(self.name)
        except Exception:
            return []
        qe = embed(text)
        res = coll.query(query_embeddings=[qe], n_results=top_k)
        out = []
        if not res.get("ids"):
            return out
        for i, vid in enumerate(res["ids"][0]):
            meta = (res.get("metadatas") or [[{}]])[0][i] or {}
            dist = (res.get("distances") or [[None]])[0][i]
            out.append({"id": vid, "qa_id": meta.get("qa_id"), "distance": dist, "meta": meta})
        return out
