#!/usr/bin/env python3
"""Заготовка отчёта порогов кэша: метрика — косинусное сходство."""

import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rep = root / "reports"
rep.mkdir(exist_ok=True)
(rep / "eval_cache_threshold.json").write_text(
    json.dumps(
        {
            "metric": "cosine_similarity",
            "SEMANTIC_CACHE_MIN_SIMILARITY": 0.88,
            "SEMANTIC_CACHE_STRICT_MIN_SIMILARITY": 0.97,
            "note": "значения из текущего assistant_api/cache.py на этапе 0",
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print("written reports/eval_cache_threshold.json")
