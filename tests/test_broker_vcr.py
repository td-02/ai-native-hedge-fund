from __future__ import annotations

from pathlib import Path

import requests
import vcr


def test_alpaca_account_cassette_replay():
    cassette = Path(__file__).parent / "cassettes" / "alpaca_account.yaml"
    recorder = vcr.VCR(record_mode="none")
    with recorder.use_cassette(str(cassette)):
        resp = requests.get("https://paper-api.alpaca.markets/v2/account", timeout=2)
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_value"] == "100000"

