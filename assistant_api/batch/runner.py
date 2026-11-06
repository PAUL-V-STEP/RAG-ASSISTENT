from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from assistant_api.batch.checks import check_expect
from assistant_api.batch.readers import read_input
from assistant_api.settings import PROJECT_ROOT, load_settings


def main(argv=None):
    load_dotenv(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--stage")
    p.add_argument("--out")
    p.add_argument("--format", default="auto")
    args = p.parse_args(argv)
    cli = {}
    if args.input:
        cli["BATCH_INPUT_FILE"] = args.input
    if args.stage:
        cli["BATCH_STAGE"] = args.stage
    if args.out:
        cli["BATCH_OUTPUT_DIR"] = args.out
    settings = load_settings(cli_overrides=cli)
    inp = settings.resolve_path("BATCH_INPUT_FILE")
    if not inp or not inp.exists():
        print("нет BATCH_INPUT_FILE")
        return 2
    data = read_input(inp, settings.BATCH_FORMAT if settings.BATCH_FORMAT != "auto" else "auto")
    from assistant_api.rag_pipeline import RAGPipeline

    pipe = RAGPipeline(settings=settings)
    outdir = settings.resolve_path("BATCH_OUTPUT_DIR") / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    fails = 0
    n = 0
    for item in data.get("items") or []:
        pipe.memory.reset()
        for turn in item.get("turns") or []:
            n += 1
            q = turn.get("q")
            expect = (turn.get("expect") or {})
            stage = settings.BATCH_STAGE
            if stage in {"input_fix", "normalize"}:
                from assistant_api.normalization.pipeline import normalize_query

                nq = normalize_query(q, pipe.memory, None if stage == "input_fix" else pipe.openai_client, pipe.kb_version, settings)
                result = {"query": q, "answer": "", "qa_id": None, "qa_decision": None, "clarification": nq.needs_clarification}
                if nq.needs_clarification:
                    pipe.memory.add_user(nq.input_fix.text, raw=nq.raw)
                    pipe.memory.add_assistant(nq.clarification_question or "")
                    pipe.memory.last_normalized = nq
                else:
                    pipe.memory.add_user(nq.input_fix.text, raw=nq.raw)
                    if turn.get("a"):
                        pipe.memory.add_assistant(turn["a"])
                    pipe.memory.last_normalized = nq
            else:
                result = pipe.query(q)
                nq = result.get("normalized")
                if turn.get("a") and settings.BATCH_HISTORY_MODE == "fixed":
                    if pipe.memory.turns and pipe.memory.turns[-1].role == "assistant":
                        pipe.memory.turns[-1].content = turn["a"]
            ferr = check_expect(expect, nq, result, result.get("qa_id")) if nq else ["no nq"]
            if ferr:
                fails += 1
            results.append({"id": item.get("id"), "q": q, "fails": ferr, "standalone": getattr(nq, "standalone", None)})
    summary = {"n": n, "fails": fails, "pass_rate": 1 - fails / n if n else 1}
    (outdir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8"
    )
    (outdir / "summary.md").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = {k: v for k, v, _ in settings.as_table()}
    (outdir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    print("отчёт", outdir, summary)
    if summary["pass_rate"] < settings.BATCH_FAIL_UNDER:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
