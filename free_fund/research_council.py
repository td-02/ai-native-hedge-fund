from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TypedDict

import feedparser
from langgraph.graph import END, START, StateGraph
import pandas as pd
import requests
import yfinance as yf


class CouncilState(TypedDict, total=False):
    symbol: str
    tool_context: dict[str, Any]
    researcher: dict[str, Any]
    news_analyst: dict[str, Any]
    peer_reviewer: dict[str, Any]
    synthesizer: dict[str, Any]


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
            close = data["Close"].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
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

    def _ask(self, role: str, symbol: str, tool_context: dict[str, Any], notes: str = "") -> dict[str, Any]:
        if not self.enable:
            return {"conviction": 0.0, "summary": "disabled"}
        prompt = {
            "role": role,
            "symbol": symbol,
            "tool_context": tool_context,
            "notes": notes,
            "output": {"conviction": "float[-1,1]", "summary": "short text"},
        }
        payload = {"model": self.ollama_model, "prompt": json.dumps(prompt), "stream": False, "format": "json"}
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

    def _build_graph(self):
        graph = StateGraph(CouncilState)

        def researcher(state: CouncilState) -> CouncilState:
            out = self._ask("researcher", state["symbol"], state["tool_context"])
            return {"researcher": out}

        def news_analyst(state: CouncilState) -> CouncilState:
            notes = state.get("researcher", {}).get("summary", "")
            out = self._ask("news_analyst", state["symbol"], state["tool_context"], notes=notes)
            return {"news_analyst": out}

        def peer_reviewer(state: CouncilState) -> CouncilState:
            notes = state.get("news_analyst", {}).get("summary", "")
            out = self._ask("peer_reviewer", state["symbol"], state["tool_context"], notes=notes)
            return {"peer_reviewer": out}

        def synthesizer(state: CouncilState) -> CouncilState:
            c1 = float(state.get("researcher", {}).get("conviction", 0.0))
            c2 = float(state.get("news_analyst", {}).get("conviction", 0.0))
            c3 = float(state.get("peer_reviewer", {}).get("conviction", 0.0))
            synth = (c1 + c2 + c3) / 3.0
            return {
                "synthesizer": {
                    "conviction": float(max(-1.0, min(1.0, synth))),
                    "summary": "graph synthesis complete",
                }
            }

        graph.add_node("researcher", researcher)
        graph.add_node("news_analyst", news_analyst)
        graph.add_node("peer_reviewer", peer_reviewer)
        graph.add_node("synthesizer", synthesizer)
        graph.add_edge(START, "researcher")
        graph.add_edge("researcher", "news_analyst")
        graph.add_edge("news_analyst", "peer_reviewer")
        graph.add_edge("peer_reviewer", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    def run(self, symbols: list[str]) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
        details: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        app = self._build_graph()
        for symbol in symbols:
            state: CouncilState = {
                "symbol": symbol,
                "tool_context": {
                    "price_snapshot": self._tool_price_snapshot(symbol),
                    "headline_snapshot": self._tool_headline_snapshot(symbol),
                },
            }
            out = app.invoke(state)
            synth = float(out.get("synthesizer", {}).get("conviction", 0.0))
            details[symbol] = {
                "tools": state["tool_context"],
                "researcher": out.get("researcher", {"conviction": 0.0, "summary": "missing"}),
                "news_analyst": out.get("news_analyst", {"conviction": 0.0, "summary": "missing"}),
                "peer_reviewer": out.get("peer_reviewer", {"conviction": 0.0, "summary": "missing"}),
                "synthesizer": out.get("synthesizer", {"conviction": 0.0, "summary": "missing"}),
            }
            scores[symbol] = synth
        return pd.Series(scores, dtype=float), details

