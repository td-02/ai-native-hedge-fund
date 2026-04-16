from __future__ import annotations

import importlib


def test_audit_hash_chain_detects_tamper(monkeypatch, tmp_path):
    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("DATABASE__URL", f"sqlite+pysqlite:///{db_path}")

    import free_fund.db as db_mod
    import free_fund.audit as audit_mod
    import free_fund.models as models_mod

    importlib.reload(db_mod)
    importlib.reload(models_mod)
    importlib.reload(audit_mod)

    ledger = audit_mod.AuditLedger()
    ledger.append("a", "r1", {"x": 1})
    ledger.append("b", "r1", {"y": 2})
    assert ledger.verify_tail(last_n=10)

    with db_mod.SessionLocal() as session:
        row = session.query(models_mod.AuditEvent).order_by(models_mod.AuditEvent.id.desc()).first()
        row.payload = {"y": 999}
        session.add(row)
        session.commit()

    assert not ledger.verify_tail(last_n=10)

