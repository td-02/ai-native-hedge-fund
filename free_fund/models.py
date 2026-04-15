from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    event_hash: Mapped[str] = mapped_column(String(64), index=True)
    prev_hash: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Ohlcv(Base):
    __tablename__ = "ohlcv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), default="yfinance")
    meta: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        Index("ix_ohlcv_symbol_ts", "symbol", "ts_utc"),
    )

