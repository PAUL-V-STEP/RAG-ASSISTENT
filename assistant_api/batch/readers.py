from __future__ import annotations

from pathlib import Path

import yaml


def read_input(path: Path, fmt: str = "auto") -> dict:
    text = path.read_text(encoding="utf-8")
    if fmt == "auto":
        fmt = path.suffix.lstrip(".") or "txt"
    if fmt in {"yaml", "yml"}:
        return yaml.safe_load(text)
    if fmt == "jsonl":
        import json

        items = []
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                items.append({"id": f"j{i}", "turns": [{"q": json.loads(line).get("q") or json.loads(line).get("question")}]})
        return {"items": items}
    items = []
    buf = []
    n = 0
    for line in text.splitlines():
        if not line.strip():
            if buf:
                n += 1
                items.append({"id": f"t{n}", "turns": [{"q": q} for q in buf]})
                buf = []
            continue
        if line.startswith("#"):
            continue
        buf.append(line.strip())
    if buf:
        n += 1
        items.append({"id": f"t{n}", "turns": [{"q": q} for q in buf]})
    return {"items": items}
