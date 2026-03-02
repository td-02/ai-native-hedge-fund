from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3
    recovery_timeout_sec: int = 120
    state: str = "closed"
    failure_count: int = 0
    opened_at_utc: datetime | None = None

    def allow_call(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.opened_at_utc is None:
                return False
            if datetime.now(timezone.utc) >= self.opened_at_utc + timedelta(
                seconds=self.recovery_timeout_sec
            ):
                self.state = "half_open"
                return True
            return False
        return True

    def on_success(self) -> None:
        self.state = "closed"
        self.failure_count = 0
        self.opened_at_utc = None

    def on_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.opened_at_utc = datetime.now(timezone.utc)


@dataclass
class BreakerRegistry:
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    default_failure_threshold: int = 3
    default_recovery_timeout_sec: int = 120

    def get(self, name: str) -> CircuitBreaker:
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.default_failure_threshold,
                recovery_timeout_sec=self.default_recovery_timeout_sec,
            )
        return self.breakers[name]


def with_retries(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay_sec: float = 0.4,
    max_delay_sec: float = 3.0,
    jitter_sec: float = 0.1,
) -> T:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - explicit runtime fallback
            last_err = exc
            if attempt >= max_retries:
                break
            delay = min(max_delay_sec, base_delay_sec * (2**attempt))
            delay += random.uniform(0.0, jitter_sec)
            time.sleep(delay)
    assert last_err is not None
    raise last_err
