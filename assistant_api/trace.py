"""Трассировка шагов пайплайна (раздел 16)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant_api.settings import Settings, mask_secret


@dataclass
class TraceStep:
    step_id: int
    name: str
    detail: str = ""
    skipped: bool = False
    ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.steps: list[TraceStep] = []
        self.request_id = 0
        self._t0 = time.perf_counter()

    def add(self, step_id: int, name: str, detail: str = "", skipped: bool = False, ms: float | None = None, **extra):
        self.steps.append(TraceStep(step_id, name, detail, skipped, ms, extra))

    def timed(self, step_id: int, name: str):
        t0 = time.perf_counter()

        class _Ctx:
            def __enter__(_self):
                return _self

            def __exit__(_self, *exc):
                _self.ms = (time.perf_counter() - t0) * 1000

        return _Ctx()

    def render_summary(self) -> str:
        lines = [f"════════════════ ТРАССИРОВКА ЗАПРОСА #{self.request_id} ════════════════"]
        for s in self.steps:
            mark = "пропущен" if s.skipped else s.detail
            lines.append(f"[{s.step_id}] {s.name:<18} {mark}")
        return "\n".join(lines)

    def render_full(self) -> str:
        lines = [f"════════════════ ТРАССИРОВКА ЗАПРОСА #{self.request_id} ════════════════"]
        for s in self.steps:
            ms = f"{s.ms:.0f} мс" if s.ms is not None else ""
            skip = " (пропущен)" if s.skipped else ""
            lines.append(f"[{s.step_id}] {s.name:<18} {s.detail}{skip}  {ms}".rstrip())
            prompt = s.extra.get("messages")
            if prompt and self.settings.TRACE_SHOW_PROMPT:
                lines.append("     ┌ messages")
                text = json.dumps(prompt, ensure_ascii=False)
                maxc = self.settings.TRACE_MAX_CHARS
                if maxc and len(text) > maxc:
                    text = text[:maxc] + "…"
                for ln in text.splitlines()[:40]:
                    lines.append(f"     │ {ln}")
                lines.append("     └")
        return "\n".join(lines)

    def emit(self, request_id: int) -> str | None:
        self.request_id = request_id
        mode = self.settings.TRACE_MODE
        if mode == "off":
            return None
        block = self.render_full() if mode == "full" else self.render_summary()
        if "sk-" in block:
            block = block.replace("sk-", "sk-…")
        out = self.settings.TRACE_OUTPUT
        if out in ("console", "both"):
            print(block, file=sys.stderr)
        if out in ("file", "both"):
            path = Path(self.settings.TRACE_FILE)
            if not path.is_absolute():
                from assistant_api.settings import PROJECT_ROOT

                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            rec = {"ts": datetime.now().isoformat(), "request_id": request_id, "block": block}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            md = path.with_suffix(".md")
            with md.open("a", encoding="utf-8") as f:
                f.write(block + "\n\n")
        return block
