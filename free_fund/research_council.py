from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd
import requests


@dataclass
class LLMResearchCouncil:
    enable: bool = False
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/generate"

    def _ask(self, task: str, symbol: str) -> dict[str, Any]:
        if not self.enable:
            return {"conviction": 0.0, "summary": "disabled"}
        prompt = {
            "task": task,
            "symbol": symbol,
            "output": {"conviction": "float[-1,1]", "summary": "short text"},
        }
        payload = {
            "model": self.ollama_model,
            "prompt": json.dumps(prompt),
            "stream": False,
            "format": "json",
        }
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=20)
            r.raise_for_status()
            out = json.loads(r.json().get("response", "{}"))
            return {
                "conviction": float(max(-1.0, min(1.0, out.get("conviction", 0.0)))),
                "summary": str(out.get("summary", "na")),
            }
        except Exception:
            return {"conviction": 0.0, "summary": "fallback"}

    def run(self, symbols: list[str]) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
        details: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for symbol in symbols:
            researcher = self._ask("Analyze transcript/guidance momentum", symbol)
            news = self._ask("Find catalysts for this symbol", symbol)
            peer = self._ask("Compare symbol vs sector peers", symbol)
            synth = (researcher["conviction"] + news["conviction"] + peer["conviction"]) / 3.0
            details[symbol] = {
                "researcher": researcher,
                "news": news,
                "peer": peer,
                "synthesis": synth,
            }
            scores[symbol] = synth
        return pd.Series(scores, dtype=float), details
