from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSpec:
    seconds: int = 5

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("WindowSpec.seconds must be > 0")
