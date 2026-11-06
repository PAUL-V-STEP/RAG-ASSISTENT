"""Консольное приложение RAG-AL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from assistant_api.rag_pipeline import RAGPipeline
from assistant_api.settings import get_settings, load_settings, setting_names


def print_banner():
    print(
        """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент АЛ 3У-М6 (API)                     ║
╚══════════════════════════════════════════════════════════╝
"""
    )
    print("Команды: /reset /state /trace /config /qa  |  exit для выхода\n")


def print_response(result: dict):
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    if result.get("from_cache"):
        print(f"💾 Источник: КЕШ ({result.get('match_type')})")
    elif result.get("qa_decision") == "direct":
        print(f"📘 Источник: Q-A {result.get('qa_id')}")
    elif result.get("qa_decision") == "hybrid":
        print(
            f"📘 Источник: Q-A {result.get('qa_id')} (hybrid) + OpenAI API ({result.get('model', 'LLM')})"
        )
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
    print(f"\n💬 Ответ:\n{result['answer']}")
    print(f"{'─'*60}\n")


def print_config(settings):
    print(f"{'переменная':32} {'значение':40} источник")
    for name, val, src in settings.as_table():
        print(f"{name:32} {str(val)[:40]:40} {src}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--trace", choices=["off", "summary", "full"])
    parser.add_argument("--rebuild-kb", action="store_true")
    args = parser.parse_args(argv)
    cli = {}
    if args.trace:
        cli["TRACE_MODE"] = args.trace
    settings = load_settings(cli_overrides=cli)
    if args.print_config:
        print_config(settings)
        return 0

    print_banner()
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не установлен")
        return 1
    pipeline = RAGPipeline(settings=settings, rebuild_kb=args.rebuild_kb)
    print("✅ Система готова\n")

    while True:
        try:
            user_input = input("💭 Ваш вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if user_input == "/reset":
            pipeline.memory.reset()
            print("история очищена")
            continue
        if user_input == "/state":
            print(pipeline.memory.state.model_dump())
            if pipeline.memory.last_normalized:
                print(pipeline.memory.last_normalized.model_dump())
            continue
        if user_input.startswith("/trace"):
            parts = user_input.split()
            if len(parts) == 1 or parts[1] == "last":
                print(pipeline.last_trace or "(нет)")
            else:
                pipeline.settings = pipeline.settings.with_overrides({"TRACE_MODE": parts[1]})
                print("TRACE_MODE", parts[1])
            continue
        if user_input == "/config":
            print_config(pipeline.settings)
            continue
        if user_input.startswith("/qa"):
            parts = user_input.split()
            if len(parts) > 1:
                pipeline.settings = pipeline.settings.with_overrides({"QA_MODE": parts[1], "QA_LAYER_ENABLED": parts[1] != "off"})
                pipeline.qa.settings = pipeline.settings
                print("QA_MODE", parts[1])
            continue
        if user_input.lower() == "stats":
            print(pipeline.get_stats())
            continue
        if user_input.lower() == "clear":
            pipeline.cache.clear()
            print("кеш очищен")
            continue
        result = pipeline.query(user_input)
        print_response(result)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
