"""История диалога в памяти процесса."""

from __future__ import annotations

from assistant_api.normalization.schema import DialogState, NormalizedQuery, Turn
from assistant_api.settings import get_settings


class DialogMemory:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.turns: list[Turn] = []
        self.state = DialogState()
        self.last_normalized: NormalizedQuery | None = None

    def reset(self) -> None:
        self.turns = []
        self.state = DialogState()
        self.last_normalized = None

    def add_user(self, content: str, raw: str | None = None) -> None:
        self.turns.append(Turn(role="user", content=content, raw=raw))

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content))

    def recent_pairs(self) -> list[Turn]:
        n = self.settings.HISTORY_TURNS * 2
        return self.turns[-n:] if n else []

    def history_for_llm(self) -> list[dict]:
        max_chars = self.settings.HISTORY_ANSWER_MAX_CHARS
        out = []
        for t in self.recent_pairs():
            content = t.content
            if t.role == "assistant" and len(content) > max_chars:
                content = content[:max_chars]
            out.append({"role": t.role, "content": content})
        return out

    def history_text(self) -> str:
        return " ".join(t.content for t in self.recent_pairs())
