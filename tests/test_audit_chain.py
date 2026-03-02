from __future__ import annotations

import json
from pathlib import Path

from free_fund.audit import AuditLedger


def test_audit_hash_chain_detects_tamper(tmp_path: Path):
    ledger = AuditLedger(tmp_path / "audit")
    ledger.append("a", "r1", {"x": 1})
    ledger.append("b", "r1", {"y": 2})
    assert ledger.verify_tail(last_n=10)

    path = tmp_path / "audit" / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[-1])
    tampered["payload"]["y"] = 999
    lines[-1] = json.dumps(tampered, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not ledger.verify_tail(last_n=10)
