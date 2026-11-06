"""Кэш RAG: точный ключ и косинусное сходство (раздел 12)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from assistant_api.settings import get_settings


class RAGCache:
    def __init__(
        self,
        db_path: str = "rag_cache.db",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        settings=None,
        kb_version: str = "",
    ):
        self.settings = settings or get_settings()
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.kb_version = kb_version
        self.exact_threshold = self.settings.SEMANTIC_CACHE_STRICT_MIN_SIMILARITY
        self.semantic_threshold = self.settings.SEMANTIC_CACHE_MIN_SIMILARITY
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT,
                kb_version TEXT,
                standalone TEXT,
                canonical TEXT,
                intent TEXT,
                node TEXT,
                maintenance_level TEXT,
                normalizer_version TEXT,
                cache_key TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(cache)")
        cols = {row[1] for row in cur.fetchall()}
        for col, typ in [
            ("embedding", "TEXT"),
            ("kb_version", "TEXT"),
            ("standalone", "TEXT"),
            ("canonical", "TEXT"),
            ("intent", "TEXT"),
            ("node", "TEXT"),
            ("maintenance_level", "TEXT"),
            ("normalizer_version", "TEXT"),
            ("cache_key", "TEXT"),
        ]:
            if col not in cols:
                cur.execute(f"ALTER TABLE cache ADD COLUMN {col} {typ}")
        conn.commit()
        conn.close()

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.lower().strip().split())

    def _get_query_hash(self, query: str) -> str:
        return hashlib.sha256(self._normalize_query(query).encode()).hexdigest()

    def _embed(self, query: str) -> Optional[List[float]]:
        if not self.embed_fn:
            return None
        return self.embed_fn(self._normalize_query(query))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _ttl_ok(self, created_at: str | None) -> bool:
        ttl = self.settings.SEMANTIC_CACHE_TTL_S
        if not ttl:
            return True
        if not created_at:
            return True
        try:
            dt = datetime.fromisoformat(created_at)
        except ValueError:
            return True
        return datetime.now() - dt <= timedelta(seconds=ttl)

    def get(
        self,
        query: str,
        cache_key: str | None = None,
        standalone: str | None = None,
        slots: dict | None = None,
        canonicalization: bool | None = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.settings.EXACT_CACHE_ENABLED and not self.settings.SEMANTIC_CACHE_ENABLED:
            return None
        use_canon = (
            self.settings.CANONICALIZATION_ENABLED
            if canonicalization is None
            else canonicalization
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        def _hit(row, match_type="exact"):
            return {
                "query": row[0],
                "answer": row[1],
                "context": json.loads(row[2]) if row[2] else None,
                "created_at": row[3],
                "from_cache": True,
                "match_type": match_type,
                "similarity": 1.0,
                "exact_hit": True,
            }

        if self.settings.EXACT_CACHE_ENABLED:
            qh = self._get_query_hash(query)
            cur.execute(
                "SELECT query, answer, context, created_at FROM cache WHERE query_hash=? AND (kb_version=? OR kb_version IS NULL OR kb_version='')",
                (qh, self.kb_version),
            )
            row = cur.fetchone()
            if row and self._ttl_ok(row[3]):
                conn.close()
                return _hit(row, "exact")

        if self.settings.EXACT_CACHE_ENABLED and use_canon and cache_key:
            cur.execute(
                "SELECT query, answer, context, created_at, standalone FROM cache WHERE cache_key=? AND (kb_version=? OR kb_version IS NULL)",
                (cache_key, self.kb_version),
            )
            row = cur.fetchone()
            if row and self._ttl_ok(row[3]):
                conn.close()
                return _hit(row, "exact")
        conn.close()
        if not self.settings.SEMANTIC_CACHE_ENABLED:
            return None
        return self._find_semantic_match(standalone or query, slots or {}, use_canon)

    def _find_semantic_match(self, text: str, slots: dict, use_canon: bool) -> Optional[Dict[str, Any]]:
        query_embedding = self._embed(text)
        if query_embedding is None:
            return None
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT query, answer, context, created_at, embedding, intent, node, maintenance_level, kb_version, normalizer_version
            FROM cache WHERE embedding IS NOT NULL
            """
        )
        rows = cur.fetchall()
        conn.close()
        best = None
        best_sim = -1.0
        cands = []
        for row in rows:
            cached_query, answer, context, created_at, embedding_json, intent, node, level, kbv, nv = row
            if not self._ttl_ok(created_at):
                continue
            if kbv and self.kb_version and kbv != self.kb_version:
                continue
            if self.settings.SEMANTIC_CACHE_FILTERS_ENABLED and use_canon:
                if (intent or "-") != (slots.get("intent") or "-"):
                    continue
                if (node or "-") != (slots.get("node") or "-"):
                    continue
                if (level or "-") != (slots.get("maintenance_level") or "-"):
                    continue
                if nv and nv != self.settings.NORMALIZER_VERSION:
                    continue
            try:
                cached_embedding = json.loads(embedding_json)
            except (TypeError, json.JSONDecodeError):
                continue
            sim = self._cosine_similarity(query_embedding, cached_embedding)
            cands.append((sim, cached_query, answer, context, created_at))
        cands.sort(reverse=True)
        cands = cands[: self.settings.SEMANTIC_CACHE_TOP_K]
        if not cands:
            return None
        best_sim, cached_query, answer, context, created_at = cands[0]
        match_type = None
        if best_sim >= self.exact_threshold:
            match_type = "exact"
        elif best_sim >= self.semantic_threshold:
            match_type = "semantic"
        if not match_type:
            return None
        return {
            "query": cached_query,
            "answer": answer,
            "context": json.loads(context) if context else None,
            "created_at": created_at,
            "from_cache": True,
            "similarity": round(best_sim, 4),
            "match_type": match_type,
            "semantic_hit": True,
        }

    def set(
        self,
        query: str,
        answer: str,
        context: list = None,
        standalone: str | None = None,
        canonical: str | None = None,
        intent: str | None = None,
        node: str | None = None,
        maintenance_level: str | None = None,
        cache_key: str | None = None,
    ):
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context) if context else None
        embedding_json = None
        try:
            embedding = self._embed(standalone or query)
            if embedding is not None:
                embedding_json = json.dumps(embedding)
        except Exception as e:
            print(f"[!] Не удалось сохранить embedding вопроса: {e}")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO cache (
                query_hash, query, answer, context, created_at, embedding,
                kb_version, standalone, canonical, intent, node, maintenance_level,
                normalizer_version, cache_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_hash,
                query,
                answer,
                context_json,
                datetime.now().isoformat(),
                embedding_json,
                self.kb_version,
                standalone,
                canonical,
                intent,
                node,
                maintenance_level,
                self.settings.NORMALIZER_VERSION,
                cache_key,
            ),
        )
        conn.commit()
        conn.close()

    def clear(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        conn.close()
        return n

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cache")
        count = cur.fetchone()[0]
        cur.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
        dates = cur.fetchone()
        conn.close()
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            "exact_similarity_threshold": self.exact_threshold,
            "semantic_similarity_threshold": self.semantic_threshold,
            "kb_version": self.kb_version,
        }
