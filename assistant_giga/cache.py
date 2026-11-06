"""
Модуль кеширования для RAG ассистента.
Использует SQLite для хранения пар вопрос-ответ и embeddings вопросов.
Попадание в кеш: сначала точный хеш, затем cosine similarity.
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
import os

import numpy as np


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class RAGCache:
    """Кеш для хранения результатов RAG запросов с семантическим поиском."""

    def __init__(
        self,
        db_path: str = "rag_cache.db",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Args:
            db_path: путь к файлу базы данных SQLite
            embed_fn: функция текст -> embedding (та же модель, что у векторного хранилища)
        """
        self.db_path = db_path
        self.embed_fn = embed_fn
        # >=: формулировки совпали полностью (почти идентичный вектор)
        self.exact_threshold = _env_float("CACHE_EXACT_SIMILARITY_THRESHOLD", 0.97)
        # >=: смысл вопроса тот же, формулировка может отличаться
        self.semantic_threshold = _env_float("CACHE_SEMANTIC_SIMILARITY_THRESHOLD", 0.88)
        if self.semantic_threshold > self.exact_threshold:
            raise ValueError(
                "CACHE_SEMANTIC_SIMILARITY_THRESHOLD не должен быть выше "
                "CACHE_EXACT_SIMILARITY_THRESHOLD"
            )
        self._init_db()

    def _init_db(self):
        """Создание таблицы кеша и колонки embedding при необходимости."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT
            )
        """)

        cursor.execute("PRAGMA table_info(cache)")
        columns = {row[1] for row in cursor.fetchall()}
        if "embedding" not in columns:
            cursor.execute("ALTER TABLE cache ADD COLUMN embedding TEXT")

        conn.commit()
        conn.close()

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.lower().strip().split())

    def _get_query_hash(self, query: str) -> str:
        """SHA-256 нормализованного запроса — ключ полного текстового совпадения."""
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

    def _match_type_for_similarity(self, similarity: float) -> Optional[str]:
        """
        exact — вопросы совпали полностью (хеш или cosine >= exact_threshold).
        semantic — совпадает смысл (semantic_threshold <= cosine < exact_threshold).
        """
        if similarity >= self.exact_threshold:
            return "exact"
        if similarity >= self.semantic_threshold:
            return "semantic"
        return None

    def _find_semantic_match(self, query: str) -> Optional[Dict[str, Any]]:
        query_embedding = self._embed(query)
        if query_embedding is None:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, answer, context, created_at, embedding
            FROM cache
            WHERE embedding IS NOT NULL
        """)
        rows = cursor.fetchall()
        conn.close()

        best = None
        best_sim = -1.0

        for cached_query, answer, context, created_at, embedding_json in rows:
            try:
                cached_embedding = json.loads(embedding_json)
            except (TypeError, json.JSONDecodeError):
                continue

            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_sim:
                best_sim = similarity
                best = {
                    "query": cached_query,
                    "answer": answer,
                    "context": json.loads(context) if context else None,
                    "created_at": created_at,
                    "from_cache": True,
                    "similarity": round(best_sim, 4),
                    "match_type": None,
                }

        if best is None:
            return None

        match_type = self._match_type_for_similarity(best_sim)
        if not match_type:
            return None

        best["match_type"] = match_type
        best["similarity"] = round(best_sim, 4)
        return best

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Получение ответа из кеша.

        1) Точный хеш нормализованного текста (полное совпадение формулировки).
        2) Cosine similarity embeddings:
           - >= CACHE_EXACT_SIMILARITY_THRESHOLD → exact
           - >= CACHE_SEMANTIC_SIMILARITY_THRESHOLD → semantic
        """
        query_hash = self._get_query_hash(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query, answer, context, created_at
            FROM cache
            WHERE query_hash = ?
        """, (query_hash,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "query": result[0],
                "answer": result[1],
                "context": json.loads(result[2]) if result[2] else None,
                "created_at": result[3],
                "from_cache": True,
                "match_type": "exact",
                "similarity": 1.0,
            }

        return self._find_semantic_match(query)

    def set(self, query: str, answer: str, context: list = None):
        """Сохранение ответа и embedding вопроса в кеш."""
        query_hash = self._get_query_hash(query)
        context_json = json.dumps(context) if context else None
        embedding_json = None
        try:
            embedding = self._embed(query)
            if embedding is not None:
                embedding_json = json.dumps(embedding)
        except Exception as e:
            print(f"[!] Не удалось сохранить embedding вопроса: {e}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query_hash, query, answer, context_json, datetime.now().isoformat(), embedding_json))
        conn.commit()
        conn.close()

    def clear(self):
        """Очистка всего кеша."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика кеша и текущие пороги."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM cache")
        dates = cursor.fetchone()
        conn.close()

        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates[0] else None,
            "newest_entry": dates[1] if dates[1] else None,
            "db_size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0,
            "exact_similarity_threshold": self.exact_threshold,
            "semantic_similarity_threshold": self.semantic_threshold,
        }


if __name__ == "__main__":
    cache = RAGCache("test_cache.db")

    cache.set(
        query="Что такое машинное обучение?",
        answer="Машинное обучение - это раздел искусственного интеллекта...",
        context=["doc1", "doc2"]
    )

    result = cache.get("Что такое машинное обучение?")
    print("Результат из кеша:", result)

    stats = cache.get_stats()
    print("Статистика кеша:", stats)

    if os.path.exists("test_cache.db"):
        os.remove("test_cache.db")
