"""
Lightweight JSONL logging helpers for training/evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

