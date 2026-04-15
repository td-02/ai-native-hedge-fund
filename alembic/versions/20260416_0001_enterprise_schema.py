"""enterprise schema with audit and ohlcv tables

Revision ID: 20260416_0001
Revises:
Create Date: 2026-04-16 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260416_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_audit_events_run_id", "audit_events", ["run_id"])
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_event_hash", "audit_events", ["event_hash"])

    op.create_table(
        "ohlcv",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("ts_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="yfinance"),
        sa.Column("meta", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_ohlcv_symbol_ts", "ohlcv", ["symbol", "ts_utc"])
    op.execute("SELECT create_hypertable('ohlcv', 'ts_utc', if_not_exists => TRUE)")


def downgrade() -> None:
    op.drop_index("ix_ohlcv_symbol_ts", table_name="ohlcv")
    op.drop_table("ohlcv")
    op.drop_index("ix_audit_events_event_hash", table_name="audit_events")
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_index("ix_audit_events_run_id", table_name="audit_events")
    op.drop_table("audit_events")

