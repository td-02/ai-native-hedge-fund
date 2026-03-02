from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_hex(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchSignal:
    symbol: str
    sentiment: float
    confidence: float
    summary: str
    source_urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategySignal:
    strategy: str
    symbol: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionCycle:
    run_id: str
    timestamp_utc: str
    symbols: list[str]
    target_weights: dict[str, float]
    risk_flags: list[str]
    model_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
