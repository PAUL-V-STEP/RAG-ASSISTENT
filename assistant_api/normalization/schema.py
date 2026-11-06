"""Схема данных нормализации (ТЗ-RAG-AL, раздел 7)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

MaintenanceLevel = Literal["ЕО", "ТО-1", "СО"]
Node = Literal[
    "гидросистема",
    "подъемный механизм",
    "фургон",
    "электрооборудование",
    "шасси",
    "площадки",
    "опоры",
]
Intent = Literal[
    "procedure",
    "parameter_value",
    "periodicity",
    "safety",
    "parts",
    "fault_diagnosis",
    "other",
]

NODES = set(Node.__args__)
LEVELS = set(MaintenanceLevel.__args__)
INTENTS = set(Intent.__args__)
CONDITIONS = {"зима", "сильное загрязнение"}


class InputFix(BaseModel):
    action: Literal["none", "auto", "confirm", "answer_with_note"] = "none"
    text: str = ""
    kinds: list[Literal["homoglyph", "capslock", "layout"]] = Field(default_factory=list)
    layout_ratio: Optional[float] = None
    latin_tokens: int = 0
    changed_tokens: list[tuple[str, str]] = Field(default_factory=list)
    unresolved_tokens: list[str] = Field(default_factory=list)
    confirm_question: Optional[str] = None


class DialogState(BaseModel):
    node: Optional[Node] = None
    operation: Optional[str] = None
    part: Optional[str] = None
    maintenance_level: Optional[MaintenanceLevel] = None
    pending_input_fix: Optional[str] = None
    updated_turn: int = 0


class Entities(BaseModel):
    node: Optional[Node] = None
    maintenance_level: Optional[MaintenanceLevel] = None
    operation: Optional[str] = None
    part: Optional[str] = None
    parameter: Optional[str] = None
    condition: Optional[str] = None
    period: Optional[str] = None
    numbers: list[str] = Field(default_factory=list)


class NormalizedQuery(BaseModel):
    raw: str
    cleaned: str
    input_fix: InputFix
    standalone: str
    canonical: str = ""
    cache_key: str = ""
    intent: Intent = "other"
    intent_source: Literal["rules", "llm", "default"] = "default"
    entities: Entities = Field(default_factory=Entities)
    depends_on_history: bool = False
    topic_switch: bool = False
    topic_switch_source: Literal["llm", "slots", "none"] = "none"
    needs_clarification: bool = False
    clarification_kind: Optional[Literal["reference", "input_fix"]] = None
    clarification_question: Optional[str] = None
    rewrite_source: Literal["llm", "raw", "fallback"] = "raw"
    fallback_reason: Optional[str] = None
    normalizer_version: str = "1"
    kb_version: str = ""
    history_gate: bool = False
    history_gate_rule: Optional[int] = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    raw: Optional[str] = None
