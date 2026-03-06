from __future__ import annotations

from dataclasses import dataclass
import math
import json
from typing import Any

import feedparser
import pandas as pd
import requests
import yfinance as yf


@dataclass
class LLMResearchCouncil:
    enable: bool = False
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/generate"
    max_rounds: int = 2
    request_timeout_sec: int = 8

    def _tool_price_snapshot(self, symbol: str) -> dict[str, float]:
        try:
            data = yf.download(symbol, period="90d", auto_adjust=True, progress=False)
            if data.empty:
                return {"ret_5d": 0.0, "ret_20d": 0.0, "vol_20d": 0.0}
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"] if "Close" in data else data.iloc[:, 0]
            ret_5d = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else 0.0
            ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 21 else 0.0
            vol_20d = float(close.pct_change().tail(20).std(ddof=0)) if len(close) > 21 else 0.0
            return {"ret_5d": ret_5d, "ret_20d": ret_20d, "vol_20d": vol_20d}
        except Exception:
            return {"ret_5d": 0.0, "ret_20d": 0.0, "vol_20d": 0.0}

    def _tool_headline_snapshot(self, symbol: str) -> dict[str, Any]:
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            parsed = feedparser.parse(url)
            titles = [str(e.get("title", "")) for e in parsed.entries[:5]]
            return {"headline_count": len(titles), "headlines": titles}
        except Exception:
            return {"headline_count": 0, "headlines": []}

    def _tool_context(self, symbol: str) -> dict[str, Any]:
        return {
            "price_snapshot": self._tool_price_snapshot(symbol),
            "headline_snapshot": self._tool_headline_snapshot(symbol),
        }

    def _ask(
        self,
        task: str,
        symbol: str,
        tool_context: dict[str, Any],
        previous_notes: str = "",
    ) -> dict[str, Any]:
        if not self.enable:
            return {"conviction": 0.0, "summary": "disabled"}
        prompt = {
            "task": task,
            "symbol": symbol,
            "tool_context": tool_context,
            "previous_notes": previous_notes,
            "output": {"conviction": "float[-1,1]", "summary": "short text"},
        }
        payload = {
            "model": self.ollama_model,
            "prompt": json.dumps(prompt),
            "stream": False,
            "format": "json",
        }
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=self.request_timeout_sec)
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
            tool_context = self._tool_context(symbol)

            notes = ""
            researcher = {"conviction": 0.0, "summary": "fallback"}
            news = {"conviction": 0.0, "summary": "fallback"}
            peer = {"conviction": 0.0, "summary": "fallback"}
            for round_idx in range(max(1, self.max_rounds)):
                researcher = self._ask(
                    f"[round {round_idx+1}] Analyze transcript/guidance momentum",
                    symbol,
                    tool_context,
                    previous_notes=notes,
                )
                news = self._ask(
                    f"[round {round_idx+1}] Find catalysts for this symbol",
                    symbol,
                    tool_context,
                    previous_notes=notes,
                )
                peer = self._ask(
                    f"[round {round_idx+1}] Compare symbol vs sector peers on 5 metrics",
                    symbol,
                    tool_context,
                    previous_notes=notes,
                )
                notes = f"researcher={researcher['summary']} | news={news['summary']} | peer={peer['summary']}"

            synth = (researcher["conviction"] + news["conviction"] + peer["conviction"]) / 3.0
            if math.isnan(synth) or math.isinf(synth):
                synth = 0.0
            details[symbol] = {
                "tools": tool_context,
                "researcher": researcher,
                "news": news,
                "peer": peer,
                "synthesis": synth,
            }
            scores[symbol] = synth
        return pd.Series(scores, dtype=float), details
