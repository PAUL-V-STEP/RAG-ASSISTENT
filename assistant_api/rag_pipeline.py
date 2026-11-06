"""RAG pipeline АЛ 3У-М6: нормализация, Q-A, кэш, ретривер, LLM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from assistant_api.cache import RAGCache
from assistant_api.kb import load_kb_text
from assistant_api.normalization.history import DialogMemory
from assistant_api.normalization.pipeline import normalize_query
from assistant_api.normalization.state import update_state
from assistant_api.qa_layer.responder import format_direct, hybrid_numbers_ok
from assistant_api.qa_layer.runtime import QARuntime
from assistant_api.settings import PROJECT_ROOT, get_settings
from assistant_api.trace import Tracer
from assistant_api.vector_store import VectorStore

EXTENDED_PROMPT = Path(__file__).resolve().parent / "prompts" / "generate_extended_ru.txt"


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "rag_collection",
        cache_db_path: str = "rag_cache.db",
        data_file: str = "data/docs.txt",
        model: str = "gpt-4o-mini",
        settings=None,
        rebuild_kb: bool = False,
    ):
        self.settings = settings or get_settings()
        if not __import__("os").getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не установлен")
        self.model = self.settings.LLM_MODEL or model
        self.openai_client = OpenAI()
        kb_text, kb_path, self.kb_version = load_kb_text(self.settings)
        coll = f"{self.settings.KB_COLLECTION_PREFIX}{self.kb_version}"
        persist = str(self.settings.chroma_dir())
        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=coll,
            persist_directory=persist,
            kb_version=self.kb_version,
            rebuild=rebuild_kb,
        )
        data_path = str(kb_path)
        if self.vector_store.collection.count() == 0:
            print(f"Загрузка документов из {data_path}...")
            self.vector_store.load_documents(data_path)
        cache_name = f"{self.settings.CACHE_INDEX_PREFIX}{self.kb_version}.db"
        cache_path = Path(cache_db_path)
        if cache_path.name in {"rag_cache.db", "api_rag_cache.db"}:
            cache_path = PROJECT_ROOT / "assistant_api" / cache_name
        print("Инициализация кеша...")
        self.cache = RAGCache(
            db_path=str(cache_path),
            embed_fn=self.vector_store._create_embedding,
            settings=self.settings,
            kb_version=self.kb_version,
        )
        self.memory = DialogMemory(self.settings)
        self.qa = QARuntime(self.settings, self.vector_store, self.kb_version)
        self.request_id = 0
        self.last_trace = None
        print("RAG Pipeline инициализирован (API mode)")

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Документ {i}:\n{doc['text']}\n")
        context = "\n".join(context_parts)
        return f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Будь точным и кратким
- Отвечай на русском языке

Ответ:"""

    def _extended_prompt(self, raw, standalone, chunks, history, verified=None) -> str:
        tpl = EXTENDED_PROMPT.read_text(encoding="utf-8")
        verified_intro = verified_or = verified_rules = verified_block = ""
        if verified:
            verified_intro = " и проверенный ответ на близкий вопрос"
            verified_or = " или проверенного ответа"
            verified_rules = (
                "- Если проверенный ответ отвечает на вопрос — опирайся на него.\n"
                "- Если вопрос отличается (другое условие, другое число), ответь по смыслу "
                "проверенного ответа и фрагментов, явно указав норму из документа.\n"
            )
            rec = verified
            src = ", ".join(str(x) for x in rec.get("source") or [])
            verified_block = f"Проверенный ответ ({rec.get('id')}, пп. {src}): {rec.get('answer')}"
        history_block = f"История: {history}" if history else ""
        standalone_line = f"Уточнённый вопрос: {standalone}" if standalone and standalone != raw else ""
        return tpl.format(
            verified_intro=verified_intro,
            verified_or=verified_or,
            verified_rules=verified_rules,
            verified_block=verified_block,
            chunks=chunks,
            history_block=history_block,
            raw=raw,
            standalone_line=standalone_line,
        )

    def _generate_answer(self, prompt: str, messages: list | None = None) -> tuple[str, list]:
        msgs = messages or [
            {
                "role": "system",
                "content": "Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.settings.LLM_TEMPERATURE,
            max_tokens=500,
            timeout=self.settings.LLM_TIMEOUT_S,
        )
        return response.choices[0].message.content.strip(), msgs

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        self.request_id += 1
        tracer = Tracer(self.settings)
        nq = normalize_query(
            user_query,
            self.memory,
            self.openai_client,
            self.kb_version,
            self.settings,
            tracer,
        )
        note = ""
        if nq.input_fix.action == "answer_with_note":
            note = f"Вопрос распознан как: «{nq.input_fix.text}»\n"
        if nq.needs_clarification:
            answer = nq.clarification_question or "Уточните вопрос."
            self.memory.add_user(nq.input_fix.text, raw=nq.raw)
            self.memory.add_assistant(answer)
            self.memory.last_normalized = nq
            block = tracer.emit(self.request_id)
            self.last_trace = block
            return {
                "query": user_query,
                "answer": note + answer,
                "from_cache": False,
                "normalized": nq,
                "clarification": True,
            }

        qa_dec = None
        if self.settings.QA_LAYER_ENABLED and self.settings.QA_MODE != "off":
            qa_dec = self.qa.lookup(nq)
            tracer.add(7, "qa_layer", f"decision={qa_dec.decision} {qa_dec.qa_id} {qa_dec.reason}")
        else:
            tracer.add(
                7,
                "qa_layer",
                f"пропущен (QA_LAYER_ENABLED={self.settings.QA_LAYER_ENABLED} QA_MODE={self.settings.QA_MODE})",
                skipped=True,
            )

        shadow = self.settings.QA_MODE == "shadow"
        if qa_dec and qa_dec.decision == "direct" and not shadow:
            answer = format_direct(qa_dec)
            self._finish(nq, answer, cache=False, tracer=tracer)
            return {
                "query": user_query,
                "answer": note + answer,
                "from_cache": False,
                "qa_decision": qa_dec.decision,
                "qa_id": qa_dec.qa_id,
                "normalized": nq,
                "model": self.model,
                "mode": "API",
            }

        flags_on = (
            self.settings.NORMALIZATION_ENABLED
            or self.settings.INPUT_FIX_ENABLED
            or self.settings.QA_LAYER_ENABLED
        )
        search_text = nq.standalone if flags_on and self.settings.NORMALIZATION_ENABLED else nq.raw
        if not self.settings.NORMALIZATION_ENABLED and self.settings.INPUT_FIX_ENABLED:
            search_text = nq.input_fix.text

        cached = None
        if use_cache and not (qa_dec and qa_dec.decision in ("direct", "hybrid") and not shadow):
            cached = self.cache.get(
                nq.raw,
                cache_key=nq.cache_key,
                standalone=nq.standalone,
                slots={
                    "intent": nq.intent,
                    "node": nq.entities.node,
                    "maintenance_level": nq.entities.maintenance_level,
                },
            )
        if cached:
            tracer.add(8, "cache", f"hit {cached.get('match_type')}")
            self._finish(nq, cached["answer"], cache=False, tracer=tracer)
            return {
                "query": user_query,
                "matched_query": cached.get("query"),
                "answer": note + cached["answer"],
                "from_cache": True,
                "match_type": cached.get("match_type"),
                "similarity": cached.get("similarity"),
                "normalized": nq,
            }
        tracer.add(8, "cache", "miss")

        context_docs = self.vector_store.search(search_text, top_k=self.settings.RETRIEVER_TOP_K)
        tracer.add(9, "retrieval", f"k={len(context_docs)}")
        chunks = "\n".join(d["text"] for d in context_docs)
        hist = ""
        if flags_on and self.settings.NORMALIZATION_ENABLED:
            hist = " | ".join(f"{t['role']}: {t['content']}" for t in self.memory.history_for_llm())

        verified_rec = None
        if qa_dec and qa_dec.decision == "hybrid" and not shadow:
            verified_rec = dict(qa_dec.record or {})
            verified_rec["id"] = qa_dec.qa_id

        if flags_on:
            prompt = self._extended_prompt(nq.raw, nq.standalone, chunks, hist, verified_rec)
        else:
            prompt = self._create_prompt(nq.raw, context_docs)
        answer, msgs = self._generate_answer(prompt)
        tracer.add(10, "LLM request", f"model={self.model}", messages=msgs)

        if (
            verified_rec
            and self.settings.QA_HYBRID_NUMBER_CHECK
            and not hybrid_numbers_ok(answer, verified_rec.get("answer", ""), chunks, nq.raw + nq.standalone)
        ):
            prompt2 = self._extended_prompt(nq.raw, nq.standalone, chunks, hist, None)
            answer, msgs = self._generate_answer(prompt2)
            tracer.add(10, "LLM request", "hybrid_number_mismatch retry", messages=msgs)

        store = not (qa_dec and qa_dec.decision in ("direct", "hybrid") and not shadow)
        store_key = nq.cache_key
        # Fallback зависимого вопроса не кладём в канонический ключ (ТЗ 12.5),
        # но сохраняем по хешу сырой строки — повтор «Чем мыть?» должен попадать в кэш.
        if nq.rewrite_source == "fallback" and nq.depends_on_history and not self.settings.CACHE_STORE_FALLBACK:
            store_key = None
        if "нет информации" in answer.lower() or "документ отсутствует" in answer.lower():
            store = False
        self._finish(
            nq,
            answer,
            cache=store and use_cache,
            tracer=tracer,
            context_docs=context_docs,
            cache_key=store_key,
        )
        return {
            "query": user_query,
            "answer": note + answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
            "mode": "API",
            "normalized": nq,
            "qa_decision": qa_dec.decision if qa_dec else None,
            "qa_id": qa_dec.qa_id if qa_dec else None,
            "retriever_query": search_text,
        }

    def _finish(self, nq, answer, cache=True, tracer=None, context_docs=None, cache_key=...):
        if nq.clarification_kind != "input_fix":
            self.memory.add_user(nq.input_fix.text, raw=nq.raw)
            self.memory.add_assistant(answer)
            self.memory.state = update_state(
                self.memory.state, nq.entities, nq.topic_switch, self.request_id
            )
        self.memory.last_normalized = nq
        if cache:
            self.cache.set(
                nq.raw,
                answer,
                [d["text"] for d in (context_docs or [])],
                standalone=nq.standalone,
                canonical=nq.canonical,
                intent=nq.intent,
                node=nq.entities.node,
                maintenance_level=nq.entities.maintenance_level,
                cache_key=nq.cache_key if cache_key is ... else cache_key,
            )
        if tracer:
            tracer.add(11, "answer", answer[:120])
            self.last_trace = tracer.emit(self.request_id)
        self._log(nq, answer)

    def _log(self, nq, answer):
        path = self.settings.resolve_path("PIPELINE_LOG_FILE")
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "request_id": self.request_id,
            "raw": nq.raw,
            "standalone": nq.standalone,
            "canonical": nq.canonical,
            "intent": nq.intent,
            "kb_version": self.kb_version,
            "ts": datetime.now().isoformat(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
            "mode": "API",
            "kb_version": self.kb_version,
        }
