from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .contracts import sha256_hex
from .db import SessionLocal, engine
from .models import AuditEvent
from .models import Base


@dataclass
class AuditLedger:
    session_factory: type[Session] = SessionLocal

    def __post_init__(self) -> None:
        Base.metadata.create_all(bind=engine)

    @staticmethod
    def _canonical_timestamp(value: datetime) -> str:
        """
        Canonicalize timestamps before hashing so DB timezone round-trips
        (especially SQLite) do not break hash verification.
        """
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat(timespec="microseconds")

    def _last_hash(self, session: Session) -> str:
        stmt = select(AuditEvent.event_hash).order_by(AuditEvent.id.desc()).limit(1)
        row = session.execute(stmt).scalar_one_or_none()
        return row or ("0" * 64)

    def append(self, event_type: str, run_id: str, payload: dict[str, Any]) -> str:
        with self.session_factory() as session:
            prev_hash = self._last_hash(session)
            now_utc = datetime.now(timezone.utc)
            body = {
                "event_type": event_type,
                "run_id": run_id,
                "timestamp_utc": self._canonical_timestamp(now_utc),
                "payload": payload,
                "prev_hash": prev_hash,
            }
            event_hash = sha256_hex(body)
            event = AuditEvent(
                run_id=run_id,
                event_type=event_type,
                event_hash=event_hash,
                prev_hash=prev_hash,
                payload=payload,
                timestamp_utc=now_utc,
            )
            session.add(event)
            session.commit()
            return event_hash

    def verify_tail(self, last_n: int = 20) -> bool:
        with self.session_factory() as session:
            stmt = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(max(1, last_n))
            events = list(reversed(session.execute(stmt).scalars().all()))
            prev = None
            for event in events:
                body = {
                    "event_type": event.event_type,
                    "run_id": event.run_id,
                    "timestamp_utc": self._canonical_timestamp(event.timestamp_utc),
                    "payload": event.payload,
                    "prev_hash": event.prev_hash,
                }
                expected = sha256_hex(body)
                if expected != event.event_hash:
                    return False
                if prev is not None and event.prev_hash != prev:
                    return False
                prev = event.event_hash
            return True
