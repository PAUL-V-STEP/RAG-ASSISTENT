"""
Оценка RAG (без уровня Q-A) через RAGAS.

Запуск из корня проекта:
  .\\venv_py311\\Scripts\\python.exe -m assistant_api.evaluate_ragas

Нужны OPENAI_API_KEY и проиндексированная Chroma (как у консоли).
Q-A слой на время прогона выключается: иначе часть ответов уйдёт в hybrid/direct
без чанков, и Faithfulness считать будет не по чему.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def _compat_ragas_langchain():
    """ragas 0.3.1 импортирует VertexAI из langchain_community; в 0.4+ модуля нет."""
    import types

    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        try:
            __import__(name)
        except ModuleNotFoundError:
            dummy = types.ModuleType(name)

            class ChatVertexAI:  # noqa: N801
                pass

            dummy.ChatVertexAI = ChatVertexAI
            sys.modules[name] = dummy
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        import langchain_community.llms as llms_mod

        class VertexAI:  # noqa: N801
            pass

        llms_mod.VertexAI = VertexAI


_compat_ragas_langchain()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness

from assistant_api.rag_pipeline import RAGPipeline
from assistant_api.settings import load_settings

# Вопросы и эталоны — факты из DATA_FILE (RAG V3), не из базы Q-A.
EVALUATION_CASES = [
    (
        "Какие виды технического обслуживания предусмотрены для автолифта АЛ 3У-М6?",
        "Предусмотрены ежедневное техническое обслуживание (ЕО), техническое обслуживание № 1 (ТО-1) и сезонное техническое обслуживание (СО).",
    ),
    (
        "Как часто проводится ТО-1 автолифта АЛ 3У-М6?",
        "ТО-1 проводится через 2 месяца работы или через 250 ч работы гидросистемы.",
    ),
    (
        "Какова трудоемкость ЕО автолифта АЛ 3У-М6 зимой?",
        "Зимой трудоемкость ЕО — 50 мин, в остальное время — 25 мин.",
    ),
    (
        "Сколько Магос Унидем нужно на 10 л воды для мойки?",
        "Моющий раствор: 50 мл Магос Унидем на 10 л теплой воды.",
    ),
    (
        "Когда проводится сезонное техническое обслуживание (СО)?",
        "СО проводится два раза в год, весной и осенью.",
    ),
]


def _contexts(pipeline: RAGPipeline, result: dict, question: str) -> list[str]:
    docs = result.get("context_docs") or []
    texts = [d.get("text", "") for d in docs if d.get("text")]
    if texts:
        return texts
    search = result.get("retriever_query") or question
    found = pipeline.vector_store.search(search, top_k=pipeline.settings.RETRIEVER_TOP_K)
    return [d["text"] for d in found if d.get("text")]


def prepare_dataset(pipeline: RAGPipeline) -> Dataset:
    questions, answers, contexts, refs = [], [], [], []
    print("[*] Ответы RAG (QA_LAYER выключен, кэш не используется)\n")
    for i, (question, reference) in enumerate(EVALUATION_CASES, 1):
        print(f"  {i}/{len(EVALUATION_CASES)}: {question}")
        pipeline.memory.reset()
        result = pipeline.query(question, use_cache=False)
        ctx = _contexts(pipeline, result, question)
        questions.append(question)
        answers.append(result["answer"])
        contexts.append(ctx)
        refs.append(reference)
        print(f"     чанков: {len(ctx)}, ответ: {result['answer'][:80].replace(chr(10), ' ')}…")
    print()
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": refs,
        }
    )


def _mean(values) -> float:
    nums = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(nums) / len(nums) if nums else 0.0


def evaluate_rag_system() -> int:
    print("=" * 70)
    print("ОЦЕНКА RAG АЛ 3У-М6 ЧЕРЕЗ RAGAS (Faithfulness, Context Precision)")
    print("=" * 70)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен (.env в корне проекта)")
        return 1

    settings = load_settings().with_overrides(
        {"QA_LAYER_ENABLED": False, "QA_MODE": "off"}
    )
    try:
        print("[*] Инициализация RAGPipeline…\n")
        pipeline = RAGPipeline(settings=settings)
        print("\n[OK] готов, kb_version =", pipeline.kb_version, "\n")
    except Exception as e:
        print(f"[ОШИБКА] инициализация: {e}")
        return 1

    dataset = prepare_dataset(pipeline)
    print("[*] RAGAS (OpenAI, 1–3 мин)…\n")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision],
            raise_exceptions=False,
        )
    except Exception as e:
        print(f"[ОШИБКА] RAGAS: {e}")
        return 1

    avg_f = _mean(result["faithfulness"])
    avg_p = _mean(result["context_precision"])
    avg = (avg_f + avg_p) / 2

    print("=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"   Faithfulness:       {avg_f:.4f}  (ответ опирается на чанки)")
    print(f"   Context Precision:  {avg_p:.4f}  (чанки полезны для эталона)")
    print(f"   Среднее:            {avg:.4f}")
    if avg >= 0.7:
        print("   Оценка: хорошее качество ретривера и генерации")
    elif avg >= 0.5:
        print("   Оценка: приемлемо, смотрите слабые вопросы ниже")
    else:
        print("   Оценка: слабо — чанки или промпт не покрывают вопросы")

    print("\nПо вопросам:")
    for i, (q, _) in enumerate(EVALUATION_CASES):
        f = result["faithfulness"][i]
        p = result["context_precision"][i]
        fs = f"{f:.4f}" if not (isinstance(f, float) and math.isnan(f)) else "n/a"
        ps = f"{p:.4f}" if not (isinstance(p, float) and math.isnan(p)) else "n/a"
        print(f"  {i + 1}. F={fs}  P={ps}  {q}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate_rag_system())
