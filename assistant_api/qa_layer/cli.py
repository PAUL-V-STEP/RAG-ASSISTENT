"""qa_tool CLI."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from assistant_api.kb import load_kb_text
from assistant_api.qa_layer.loader import load_qa_base
from assistant_api.qa_layer.testing import run_tests
from assistant_api.qa_layer.validator import validate_qa
from assistant_api.settings import PROJECT_ROOT, get_settings, load_settings, setting_names


def _boot():
    load_dotenv(PROJECT_ROOT / ".env")
    return get_settings(reload=True)


def cmd_config(settings):
    for name, val, src in settings.as_table(
        [n for n in setting_names() if n.startswith("QA_") or n.startswith("BATCH_")]
    ):
        print(f"{name:32} {val!s:40} {src}")


def cmd_validate(settings):
    text, path, _ = load_kb_text(settings)
    errors, warns, _ = validate_qa(settings, text, path.name)
    for e in errors:
        print("ERR", e)
    for w in warns:
        print("WARN", w)
    print(f"ошибок: {len(errors)}, предупреждений: {len(warns)}")
    return 1 if errors else 0


def _pipeline_parts(settings):
    from openai import OpenAI

    from assistant_api.vector_store import VectorStore
    from assistant_api.qa_layer.runtime import QARuntime
    from assistant_api.qa_layer.store import QAStore

    _, _, kbv = load_kb_text(settings)
    vs = VectorStore(
        collection_name=f"{settings.KB_COLLECTION_PREFIX}{kbv}",
        persist_directory=str(settings.chroma_dir()),
        kb_version=kbv,
    )
    return vs, kbv, OpenAI()


def cmd_build(settings, rebuild=False, dry_run=False):
    code = cmd_validate(settings)
    if code:
        return code
    text, path, kbv = load_kb_text(settings)
    data = load_qa_base(settings.resolve_path("QA_BASE_FILE"))
    vs, kbv, _ = _pipeline_parts(settings)
    from assistant_api.qa_layer.store import QAStore

    store = QAStore(settings, vs, kbv)
    if dry_run:
        print("dry-run: validate ok, records", len(data.get("records") or []))
        return 0
    stats = store.build(data.get("records") or [], rebuild=rebuild)
    if stats["added"] == stats["updated"] == stats["deleted"] == 0:
        print("изменений нет")
    else:
        print(stats)
    return 0


def _test_summary_comment(ok: int, n: int, skipped: int = 0) -> str:
    fail = n - ok
    if n == 0:
        return f"все пропущены ({skipped})" if skipped else "нет тестов"
    if fail == 0:
        return f"все {ok} прошли"
    if ok == 0:
        return f"все {n} не прошли"
    if fail == 1:
        return f"почти всё ок ({ok} прошли, {fail} нет)"
    return f"{ok} прошли, {fail} нет"


def cmd_test(settings, ignore_verified=False, types=None, as_json=False):
    from assistant_api.qa_layer.runtime import QARuntime

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        vs, kbv, client = _pipeline_parts(settings)
        rt = QARuntime(
            settings.with_overrides({"QA_LAYER_ENABLED": True, "QA_MODE": settings.QA_TOOL_MODE}),
            vs,
            kbv,
        )
        results = run_tests(settings, rt, openai_client=client, types=types, ignore_verified=ignore_verified)
    time.sleep(0.4)
    sys.stdout.flush()
    sys.stderr.flush()

    order = ("match", "contrast", "no_match", "dialog")
    by = {}
    for r in results:
        by.setdefault(r.get("type"), []).append(r)
    summary = {}
    stats_lines = []
    fail_lines = []
    for typ in [*order, *[t for t in by if t not in order]]:
        items = by.get(typ) or []
        if not items:
            continue
        real = [x for x in items if x.get("status") != "skipped"]
        ok = sum(1 for x in real if x.get("pass"))
        skipped = len(items) - len(real)
        summary[typ] = {"n": len(real), "ok": ok, "skipped": skipped}
        line = f"{typ}: {ok}/{len(real)}"
        note = _test_summary_comment(ok, len(real), skipped)
        if skipped:
            note += f", пропущено {skipped}"
        stats_lines.append(f"{line:<22} ← {note}")
        for x in items:
            if x.get("pass") is False:
                fail_lines.append(f" FAIL {x.get('id')} got {x.get('got_id')} {x.get('reason')}")
    print("\n".join(stats_lines), flush=True)
    if fail_lines:
        print("\n" + "\n".join(fail_lines), flush=True)
    if as_json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, default=str)[:8000])
    reports = settings.resolve_path("QA_REPORTS_DIR")
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    match = summary.get("match") or {"n": 1, "ok": 0}
    rate = match["ok"] / match["n"] if match["n"] else 1
    if rate < settings.QA_TEST_FAIL_UNDER_MATCH:
        return 1
    return 0


def cmd_ask(settings, question, generate=False):
    from assistant_api.normalization.history import DialogMemory
    from assistant_api.normalization.pipeline import normalize_query
    from assistant_api.qa_layer.runtime import QARuntime
    from assistant_api.qa_layer.responder import format_direct

    vs, kbv, client = _pipeline_parts(settings)
    rt = QARuntime(settings.with_overrides({"QA_LAYER_ENABLED": True}), vs, kbv)
    mem = DialogMemory(settings)
    nq = normalize_query(question, mem, None, kbv, settings)
    dec = rt.lookup(nq, ignore_verified=settings.QA_TOOL_IGNORE_VERIFIED)
    print("standalone:", nq.standalone)
    print("canonical:", nq.canonical)
    print("decision:", dec.decision, dec.qa_id, dec.reason)
    for c in dec.candidates:
        print(" cand", c)
    if dec.decision == "direct":
        print(format_direct(dec))
    return 0


def cmd_list(settings, **filt):
    data = load_qa_base(settings.resolve_path("QA_BASE_FILE"))
    for r in data.get("records") or []:
        if filt.get("intent") and r.get("intent") != filt["intent"]:
            continue
        q = (r.get("question") or "")[:60]
        print(f"{r['id']:6} {r.get('intent'):16} {str(r.get('node')):16} {q}")
    return 0


def cmd_show(settings, qid):
    data = load_qa_base(settings.resolve_path("QA_BASE_FILE"))
    for r in data.get("records") or []:
        if r["id"] == qid:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return 0
    print("не найдено", qid)
    return 1


def cmd_clear(settings, yes=False, stale_only=False):
    vs, kbv, _ = _pipeline_parts(settings)
    prefix = settings.QA_COLLECTION_PREFIX
    names = [c.name for c in vs.client.list_collections() if c.name.startswith(prefix)]
    if stale_only:
        names = [n for n in names if not n.endswith(kbv)]
    print("будет удалено:", names)
    if not yes and settings.QA_TOOL_CONFIRM:
        ans = input("да/yes: ").strip().lower()
        if ans not in ("да", "yes"):
            return 1
    for n in names:
        if not n.startswith(prefix):
            continue
        vs.client.delete_collection(n)
    mpath = settings.resolve_path("QA_MANIFEST_FILE")
    if mpath and mpath.exists() and not stale_only:
        mpath.unlink()
    print("готово")
    return 0


def cmd_stats(settings):
    vs, kbv, _ = _pipeline_parts(settings)
    mpath = settings.resolve_path("QA_MANIFEST_FILE")
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    print("kb_version", kbv, "manifest", man.get("kb_version"), "устарела" if man.get("kb_version") != kbv else "актуальна")
    print("vectors", man.get("vectors"), "records", man.get("records"))
    return 0


def cmd_calibrate(settings, write=False):
    print("calibrate: используйте сетку QA_CALIBRATE_* (упрощённый прогон test)")
    return cmd_test(settings, ignore_verified=True)


def cmd_export(settings, fmt="md", out=None):
    data = load_qa_base(settings.resolve_path("QA_BASE_FILE"))
    lines = ["id\tquestion\tanswer\tfacts"]
    for r in data.get("records") or []:
        lines.append(f"{r['id']}\t{r['question']}\t{r['answer']}\t{r.get('facts')}")
    text = "\n".join(lines)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text[:4000])
    return 0


def cmd_manual(settings):
    print("qa_tool manual: введите вопрос (пустая строка — выход)")
    while True:
        q = input("> ").strip()
        if not q or q == "/quit":
            break
        cmd_ask(settings, q)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="qa_tool")
    p.add_argument("--base")
    p.add_argument("--tests")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config")
    sub.add_parser("validate")
    b = sub.add_parser("build")
    b.add_argument("--rebuild", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("test")
    t.add_argument("--ignore-verified", action="store_true")
    t.add_argument("--type")
    t.add_argument("--mode")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--generate", action="store_true")
    sub.add_parser("list")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    cl = sub.add_parser("clear")
    cl.add_argument("--yes", action="store_true")
    cl.add_argument("--stale-only", action="store_true")
    sub.add_parser("stats")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--write", action="store_true")
    ex = sub.add_parser("export")
    ex.add_argument("--format", default="md")
    ex.add_argument("--out")
    sub.add_parser("manual")
    v = sub.add_parser("verify")
    v.add_argument("id")
    v.add_argument("--by", required=True)
    args = p.parse_args(argv)

    cli = {}
    if args.base:
        cli["QA_BASE_FILE"] = args.base
    if args.tests:
        cli["QA_TESTS_FILE"] = args.tests
    settings = load_settings(cli_overrides=cli)

    if args.cmd == "config":
        return cmd_config(settings)
    if args.cmd == "validate":
        return cmd_validate(settings)
    if args.cmd == "build":
        return cmd_build(settings, rebuild=args.rebuild, dry_run=args.dry_run)
    if args.cmd == "test":
        types = args.type.split(",") if args.type else None
        st = settings
        if args.mode:
            st = settings.with_overrides({"QA_TOOL_MODE": args.mode})
        return cmd_test(
            st,
            ignore_verified=args.ignore_verified or st.QA_TOOL_IGNORE_VERIFIED,
            types=types,
            as_json=args.json,
        )
    if args.cmd == "ask":
        return cmd_ask(settings, args.question, generate=args.generate)
    if args.cmd == "list":
        return cmd_list(settings)
    if args.cmd == "show":
        return cmd_show(settings, args.id)
    if args.cmd == "clear":
        return cmd_clear(settings, yes=args.yes, stale_only=args.stale_only)
    if args.cmd == "stats":
        return cmd_stats(settings)
    if args.cmd == "calibrate":
        return cmd_calibrate(settings, write=args.write)
    if args.cmd == "export":
        return cmd_export(settings, fmt=args.format, out=args.out)
    if args.cmd == "manual":
        return cmd_manual(settings)
    if args.cmd == "verify":
        print("verify: проставьте verified_by вручную или через точечную замену YAML")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
