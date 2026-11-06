"""Единая конфигурация RAG-AL (ТЗ-RAG-AL, раздел 6)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")

BOOL_TRUE = {"true", "1", "yes"}
BOOL_FALSE = {"false", "0", "no"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    default: Any
    enum: tuple[str, ...] | None = None
    min_v: float | None = None
    max_v: float | None = None


# Имена совпадают с env.example и таблицей 6.3.
SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("APP_MODE", "enum", "console", enum=("console", "batch")),
    FieldSpec("ENV_FILE", "path", ""),
    FieldSpec("DATA_FILE", "path", "ТО автолифта АЛ 3У-М6 — версия для RAG V3.txt"),
    FieldSpec("CHROMA_DIR", "path", ""),
    FieldSpec("KB_COLLECTION_PREFIX", "str", "kb__"),
    FieldSpec("RETRIEVER_TOP_K", "int", 3, min_v=1, max_v=20),
    FieldSpec("LLM_MODEL", "str", "gpt-4o-mini"),
    FieldSpec("LLM_TIMEOUT_S", "float", 60.0, min_v=1, max_v=120),
    FieldSpec("LLM_TEMPERATURE", "float", 0.3, min_v=0, max_v=2),
    FieldSpec("LOG_LEVEL", "enum", "INFO", enum=("DEBUG", "INFO", "WARNING", "ERROR")),
    FieldSpec("PIPELINE_LOG_FILE", "path", "logs/pipeline.jsonl"),
    FieldSpec("INPUT_FIX_ENABLED", "bool", True),
    FieldSpec("HOMOGLYPH_FIX_ENABLED", "bool", True),
    FieldSpec("CAPSLOCK_FIX_ENABLED", "bool", True),
    FieldSpec("LAYOUT_FIX_ENABLED", "bool", True),
    FieldSpec("LAYOUT_AUTO_MIN_RATIO", "float", 0.8, min_v=0, max_v=1),
    FieldSpec("LAYOUT_CONFIRM_MIN_RATIO", "float", 0.5, min_v=0, max_v=1),
    FieldSpec("LAYOUT_MIN_LATIN_TOKENS", "int", 2, min_v=1, max_v=10),
    FieldSpec("LAYOUT_UNCLEAR_ACTION", "enum", "confirm", enum=("confirm", "answer_with_note", "ignore")),
    FieldSpec("LAYOUT_CONTEXT_BONUS", "float", 0.1, min_v=0, max_v=0.5),
    FieldSpec("LAYOUT_LATIN_WHITELIST_FILE", "path", "assistant_api/data/latin_whitelist.yaml"),
    FieldSpec("NORMALIZATION_ENABLED", "bool", True),
    FieldSpec("CONTEXTUALIZATION_ENABLED", "bool", True),
    FieldSpec("CANONICALIZATION_ENABLED", "bool", True),
    FieldSpec("NORMALIZER_MODEL", "str", "gpt-4o-mini"),
    FieldSpec("NORMALIZER_TIMEOUT_S", "float", 6.0, min_v=1, max_v=30),
    FieldSpec("NORMALIZER_RETRIES", "int", 1, min_v=0, max_v=3),
    FieldSpec("NORMALIZER_VERSION", "str", "2"),
    FieldSpec("HISTORY_TURNS", "int", 3, min_v=0, max_v=10),
    FieldSpec("HISTORY_ANSWER_MAX_CHARS", "int", 500, min_v=100, max_v=5000),
    FieldSpec("REWRITE_MAX_RATIO", "float", 4.0, min_v=1, max_v=10),
    FieldSpec("FUZZY_CUTOFF", "int", 88, min_v=50, max_v=100),
    FieldSpec("GLOSSARY_EXTRA_FILE", "path", "assistant_api/data/glossary_extra.yaml"),
    FieldSpec("GLOSSARY_CANONICAL_FILE", "path", "assistant_api/data/glossary_canonical.yaml"),
    FieldSpec("EXACT_CACHE_ENABLED", "bool", True),
    FieldSpec("SEMANTIC_CACHE_ENABLED", "bool", True),
    FieldSpec("SEMANTIC_CACHE_MIN_SIMILARITY", "float", 0.88, min_v=0, max_v=1),
    FieldSpec("SEMANTIC_CACHE_STRICT_MIN_SIMILARITY", "float", 0.97, min_v=0, max_v=1),
    FieldSpec("SEMANTIC_CACHE_TOP_K", "int", 20, min_v=1, max_v=20),
    FieldSpec("SEMANTIC_CACHE_TTL_S", "int", 0, min_v=0),
    FieldSpec("SEMANTIC_CACHE_FILTERS_ENABLED", "bool", True),
    FieldSpec("CACHE_INDEX_PREFIX", "str", "cache__"),
    FieldSpec("CACHE_STORE_FALLBACK", "bool", False),
    FieldSpec("QA_LAYER_ENABLED", "bool", False),
    FieldSpec("QA_MODE", "enum", "shadow", enum=("off", "shadow", "hybrid_only", "full")),
    FieldSpec("QA_BASE_FILE", "path", "assistant_api/data/qa/qa_base.yaml"),
    FieldSpec("QA_TESTS_FILE", "path", "assistant_api/data/qa/qa_tests.yaml"),
    FieldSpec("QA_MANIFEST_FILE", "path", "assistant_api/data/qa/qa_manifest.json"),
    FieldSpec("QA_COLLECTION_PREFIX", "str", "qa_verified__"),
    FieldSpec("QA_TOP_K", "int", 5, min_v=1, max_v=20),
    FieldSpec("QA_DIRECT_MAX_DISTANCE", "float", 0.12, min_v=0, max_v=1),
    FieldSpec("QA_HYBRID_MAX_DISTANCE", "float", 0.25, min_v=0, max_v=1),
    FieldSpec("QA_MIN_MARGIN", "float", 0.03, min_v=0, max_v=1),
    FieldSpec("QA_REQUIRE_VERIFIED", "bool", True),
    FieldSpec("QA_STOP_RULES_ENABLED", "bool", True),
    FieldSpec("QA_HYBRID_NUMBER_CHECK", "bool", True),
    FieldSpec("QA_LOG_FILE", "path", "logs/qa_layer.jsonl"),
    FieldSpec("QA_REPORTS_DIR", "path", "reports/qa"),
    FieldSpec("QA_TOOL_MODE", "enum", "full", enum=("off", "shadow", "hybrid_only", "full")),
    FieldSpec("QA_TOOL_IGNORE_VERIFIED", "bool", False),
    FieldSpec("QA_TOOL_TEST_TYPES", "list", "match,contrast,no_match,dialog"),
    FieldSpec("QA_TOOL_WITH_GENERATION", "bool", False),
    FieldSpec("QA_TOOL_OUTPUT", "enum", "table", enum=("table", "json")),
    FieldSpec("QA_TOOL_QUIET", "bool", False),
    FieldSpec("QA_TOOL_CONFIRM", "bool", True),
    FieldSpec("QA_TEST_FAIL_UNDER_MATCH", "float", 0.9, min_v=0, max_v=1),
    FieldSpec("QA_TEST_MIN_DIRECT_SHARE", "float", 0.6, min_v=0, max_v=1),
    FieldSpec("QA_TEST_MAX_CONTRAST_ERRORS", "int", 0, min_v=0),
    FieldSpec("QA_TEST_MAX_NOMATCH_ERRORS", "int", 0, min_v=0),
    FieldSpec("QA_CALIBRATE_DIRECT_RANGE", "range", "0.04:0.20:0.01"),
    FieldSpec("QA_CALIBRATE_HYBRID_RANGE", "range", "0.10:0.40:0.02"),
    FieldSpec("QA_CALIBRATE_MARGIN_RANGE", "range", "0:0.08:0.01"),
    FieldSpec("QA_CALIBRATE_SPLIT", "float", 0.7, min_v=0.5, max_v=0.9),
    FieldSpec("QA_CALIBRATE_SEED", "int", 42),
    FieldSpec("QA_MANUAL_SHUFFLE", "bool", False),
    FieldSpec("QA_MANUAL_LIMIT", "int", 0, min_v=0),
    FieldSpec("TRACE_MODE", "enum", "off", enum=("off", "summary", "full")),
    FieldSpec("TRACE_OUTPUT", "enum", "console", enum=("console", "file", "both")),
    FieldSpec("TRACE_FILE", "path", "logs/trace.jsonl"),
    FieldSpec("TRACE_SHOW_PROMPT", "bool", True),
    FieldSpec("TRACE_MAX_CHARS", "int", 4000, min_v=0),
    FieldSpec("TRACE_SHOW_CANDIDATES", "int", 5, min_v=0, max_v=20),
    FieldSpec("TRACE_COLOR", "enum", "auto", enum=("auto", "on", "off")),
    FieldSpec("BATCH_INPUT_FILE", "path", ""),
    FieldSpec("BATCH_FORMAT", "enum", "auto", enum=("auto", "txt", "yaml", "jsonl", "csv")),
    FieldSpec("BATCH_STAGE", "enum", "full", enum=("input_fix", "normalize", "qa", "retrieve", "full")),
    FieldSpec("BATCH_OUTPUT_DIR", "path", "reports/batch"),
    FieldSpec("BATCH_PROFILES", "list", ""),
    FieldSpec("BATCH_COMPARE_WITH", "path", ""),
    FieldSpec("BATCH_HISTORY_MODE", "enum", "fixed", enum=("fixed", "live")),
    FieldSpec("BATCH_STOP_ON_ERROR", "bool", False),
    FieldSpec("BATCH_REPEAT", "int", 1, min_v=1, max_v=10),
    FieldSpec("BATCH_TRACE", "bool", True),
    FieldSpec("BATCH_FAIL_UNDER", "float", 1.0, min_v=0, max_v=1),
    FieldSpec("BATCH_LIMIT", "int", 0, min_v=0),
    FieldSpec("BATCH_FILTER_TAGS", "list", ""),
)

SPEC_BY_NAME = {s.name: s for s in SPECS}


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = dotenv_values(path)
    return {k: v for k, v in data.items() if k and v is not None}


def mask_secret(name: str, value: Any) -> Any:
    if not any(name.endswith(s) for s in SECRET_SUFFIXES):
        return value
    text = str(value or "")
    if len(text) <= 4:
        return "…"
    return f"{text[:2]}…{text[-4:]}"


def _parse_bool(name: str, raw: str, source: str) -> bool:
    key = raw.strip().lower()
    if key in BOOL_TRUE:
        return True
    if key in BOOL_FALSE:
        return False
    raise ConfigError(
        f"{name}={raw!r} ({source}): допустимо true/false/1/0/yes/no"
    )


def _parse_value(spec: FieldSpec, raw: str, source: str) -> Any:
    text = raw.strip()
    if spec.kind == "bool":
        return _parse_bool(spec.name, text, source)
    if spec.kind == "int":
        try:
            val = int(text)
        except ValueError as exc:
            raise ConfigError(f"{spec.name}={raw!r} ({source}): ожидается int") from exc
        if spec.min_v is not None and val < spec.min_v:
            raise ConfigError(f"{spec.name}={val} ({source}): минимум {spec.min_v}")
        if spec.max_v is not None and val > spec.max_v:
            raise ConfigError(f"{spec.name}={val} ({source}): максимум {spec.max_v}")
        return val
    if spec.kind == "float":
        try:
            val = float(text.replace(",", "."))
        except ValueError as exc:
            raise ConfigError(f"{spec.name}={raw!r} ({source}): ожидается float") from exc
        if spec.min_v is not None and val < spec.min_v:
            raise ConfigError(f"{spec.name}={val} ({source}): минимум {spec.min_v}")
        if spec.max_v is not None and val > spec.max_v:
            raise ConfigError(f"{spec.name}={val} ({source}): максимум {spec.max_v}")
        return val
    if spec.kind == "enum":
        if text not in spec.enum:
            raise ConfigError(
                f"{spec.name}={raw!r} ({source}): допустимо {list(spec.enum)}"
            )
        return text
    if spec.kind == "range":
        parts = text.split(":")
        if len(parts) != 3:
            raise ConfigError(f"{spec.name}={raw!r} ({source}): формат начало:конец:шаг")
        try:
            start, end, step = (float(p) for p in parts)
        except ValueError as exc:
            raise ConfigError(f"{spec.name}={raw!r} ({source}): числа в range") from exc
        if step <= 0:
            raise ConfigError(f"{spec.name}={raw!r} ({source}): шаг > 0")
        return text
    if spec.kind == "list":
        return text
    return text


@dataclass
class Settings:
    values: dict[str, Any]
    sources: dict[str, str]

    def __getattr__(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)

    def get(self, name: str) -> Any:
        return self.values[name]

    def source_of(self, name: str) -> str:
        return self.sources.get(name, "default")

    def resolve_path(self, name: str) -> Optional[Path]:
        raw = self.values.get(name) or ""
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def data_file(self) -> Path:
        path = self.resolve_path("DATA_FILE")
        if path is None:
            raise ConfigError("DATA_FILE пуст")
        return path

    def chroma_dir(self) -> Path:
        raw = self.values.get("CHROMA_DIR") or ""
        if raw:
            path = Path(raw)
            return path if path.is_absolute() else PROJECT_ROOT / path
        return Path(__file__).resolve().parent / "chroma_db"

    def as_table(self, names: list[str] | None = None) -> list[tuple[str, Any, str]]:
        keys = names or [s.name for s in SPECS]
        rows = []
        for key in keys:
            rows.append((key, mask_secret(key, self.values[key]), self.sources[key]))
        return rows

    def with_overrides(self, overrides: dict[str, Any], source: str = "cli") -> "Settings":
        values = dict(self.values)
        sources = dict(self.sources)
        for key, raw in overrides.items():
            if key not in SPEC_BY_NAME:
                continue
            if raw is None:
                continue
            spec = SPEC_BY_NAME[key]
            values[key] = _parse_value(spec, str(raw), source)
            sources[key] = source
        _cross_validate(values, sources)
        return Settings(values=values, sources=sources)


def _cross_validate(values: dict[str, Any], sources: dict[str, str]) -> None:
    confirm = values["LAYOUT_CONFIRM_MIN_RATIO"]
    auto = values["LAYOUT_AUTO_MIN_RATIO"]
    if confirm > auto:
        raise ConfigError(
            f"LAYOUT_CONFIRM_MIN_RATIO={confirm} ({sources['LAYOUT_CONFIRM_MIN_RATIO']}) "
            f"> LAYOUT_AUTO_MIN_RATIO={auto}"
        )
    if values["SEMANTIC_CACHE_MIN_SIMILARITY"] > values["SEMANTIC_CACHE_STRICT_MIN_SIMILARITY"]:
        raise ConfigError(
            "SEMANTIC_CACHE_MIN_SIMILARITY не должен быть выше "
            "SEMANTIC_CACHE_STRICT_MIN_SIMILARITY"
        )


def load_settings(
    cli_overrides: dict[str, Any] | None = None,
    env_file: str | None = None,
) -> Settings:
    dotenv_path = PROJECT_ROOT / ".env"
    dotenv_map = _parse_env_file(dotenv_path)

    process_env_file = env_file or os.environ.get("ENV_FILE") or dotenv_map.get("ENV_FILE") or ""
    profile_map: dict[str, str] = {}
    profile_source_label = "ENV_FILE"
    if process_env_file:
        p = Path(process_env_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise ConfigError(f"ENV_FILE={process_env_file!r}: файл не найден")
        profile_map = _parse_env_file(p)

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    cli_overrides = cli_overrides or {}

    for spec in SPECS:
        name = spec.name
        if name in cli_overrides and cli_overrides[name] is not None and str(cli_overrides[name]) != "":
            values[name] = _parse_value(spec, str(cli_overrides[name]), "cli")
            sources[name] = "cli"
            continue
        if name in os.environ and os.environ[name] != "":
            values[name] = _parse_value(spec, os.environ[name], "env")
            sources[name] = "env"
            continue
        if name in profile_map and profile_map[name] != "":
            values[name] = _parse_value(spec, profile_map[name], profile_source_label)
            sources[name] = profile_source_label
            continue
        if name in dotenv_map and dotenv_map[name] != "":
            values[name] = _parse_value(spec, dotenv_map[name], ".env")
            sources[name] = ".env"
            continue
        values[name] = spec.default
        sources[name] = "default"

    if process_env_file and sources.get("ENV_FILE") == "default":
        values["ENV_FILE"] = process_env_file
        sources["ENV_FILE"] = "env" if "ENV_FILE" in os.environ else profile_source_label

    _cross_validate(values, sources)
    return Settings(values=values, sources=sources)


_SETTINGS: Settings | None = None


def get_settings(reload: bool = False, **kwargs) -> Settings:
    global _SETTINGS
    if _SETTINGS is None or reload:
        _SETTINGS = load_settings(**kwargs)
    return _SETTINGS


def setting_names() -> list[str]:
    return [s.name for s in SPECS]
