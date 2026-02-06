from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rationale:
    rule_id: str
    message: str
    details: dict[str, Any]


def fmt_rationale(r: Rationale) -> str:
    # Compact, analyst-friendly single-line summary
    parts = [f"{r.rule_id}: {r.message}"]
    if r.details:
        kv = ", ".join(f"{k}={v}" for k, v in r.details.items())
        parts.append(f"({kv})")
    return " ".join(parts)
