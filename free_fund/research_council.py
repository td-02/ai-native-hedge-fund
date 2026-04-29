from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TypedDict
import re

import feedparser
from langgraph.graph import END, START, StateGraph
import pandas as pd
import requests
import yfinance as yf
from requests.exceptions import RequestException

from .logging import get_logger

logger = get_logger(__name__)


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

    def __post_init__(self) -> None:
        self._resolved_model: str | None = None

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
        model = self._resolve_model()
        prompt = {
            "role": role,
            "symbol": symbol,
            "tool_context": tool_context,
            "notes": notes,
            "output": {"conviction": "float[-1,1]", "summary": "short text"},
        }
        payload = {
            "model": model,
            "prompt": json.dumps(prompt),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 200},
        }
        timeout = max(15, int(self.request_timeout_sec))
        last_err = ""
        for attempt in range(2):
            try:
                r = requests.post(self.ollama_url, json=payload, timeout=timeout)
                r.raise_for_status()
                raw = r.json().get("response", "{}")
                out = self._extract_json(raw)
                return {
                    "conviction": float(max(-1.0, min(1.0, out.get("conviction", 0.0)))),
                    "summary": str(out.get("summary", "na")),
                }
            except Exception as exc:
                last_err = str(exc)
                continue
        logger.warning(
            "research_council.fallback",
            role=role,
            symbol=symbol,
            model=model,
            error=last_err[:280],
        )
        return {"conviction": 0.0, "summary": f"fallback:{last_err[:120]}"}

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        try:
            base = self.ollama_url.replace("/api/generate", "/api/tags")
            r = requests.get(base, timeout=5)
            r.raise_for_status()
            models = [str(m.get("name", "")) for m in (r.json().get("models", []) or []) if m.get("name")]
            if self.ollama_model in models:
                self._resolved_model = self.ollama_model
                return self._resolved_model
            if models:
                self._resolved_model = models[0]
                logger.info(
                    "research_council.model_fallback",
                    requested=self.ollama_model,
                    selected=self._resolved_model,
                )
                return self._resolved_model
        except RequestException as exc:
            logger.warning("research_council.model_resolve_failed", error=str(exc))
        self._resolved_model = self.ollama_model
        return self._resolved_model

    @staticmethod
    def _extract_json(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

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
