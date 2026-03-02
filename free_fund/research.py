from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import feedparser
import requests

from .contracts import ResearchSignal


@dataclass
class ResearchAgent:
    enable_llm: bool = False
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/generate"
    max_headlines: int = 8

    def _fetch_headlines(self, symbol: str) -> list[dict[str, str]]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        parsed = feedparser.parse(url)
        entries: list[dict[str, str]] = []
        for item in parsed.entries[: self.max_headlines]:
            entries.append({"title": item.get("title", ""), "url": item.get("link", "")})
        return entries

    @staticmethod
    def _keyword_sentiment(text: str) -> float:
        positive = ["beat", "growth", "upgrade", "profit", "strong", "record", "bullish"]
        negative = ["miss", "downgrade", "loss", "weak", "lawsuit", "bearish", "decline"]
        t = text.lower()
        p = sum(1 for w in positive if w in t)
        n = sum(1 for w in negative if w in t)
        if p + n == 0:
            return 0.0
        return float((p - n) / (p + n))

    def _llm_overlay(self, symbol: str, headlines: list[dict[str, str]]) -> dict[str, Any]:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda

            prompt = ChatPromptTemplate.from_template(
                "Return strict JSON with keys sentiment (-1..1), confidence (0..1), summary (<=35 words). "
                "Ticker: {symbol}. Headlines: {headlines}"
            )

            def render(inp: dict[str, Any]) -> str:
                return prompt.invoke(inp).to_string()

            def call_ollama(rendered_prompt: str) -> str:
                payload = {
                    "model": self.ollama_model,
                    "prompt": rendered_prompt,
                    "stream": False,
                    "format": "json",
                }
                resp = requests.post(self.ollama_url, json=payload, timeout=20)
                resp.raise_for_status()
                return resp.json().get("response", "{}")

            def parse_json(raw: str) -> dict[str, Any]:
                parsed = json.loads(raw)
                return {
                    "sentiment": float(parsed.get("sentiment", 0.0)),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "summary": str(parsed.get("summary", "LLM summary unavailable.")),
                }

            chain = RunnableLambda(render) | RunnableLambda(call_ollama) | RunnableLambda(parse_json)
            out = chain.invoke({"symbol": symbol, "headlines": headlines})
            out["sentiment"] = max(-1.0, min(1.0, out["sentiment"]))
            out["confidence"] = max(0.0, min(1.0, out["confidence"]))
            return out
        except Exception:
            return {
                "sentiment": 0.0,
                "confidence": 0.5,
                "summary": "LLM unavailable, deterministic fallback used.",
            }

    def run(self, symbols: list[str]) -> dict[str, ResearchSignal]:
        results: dict[str, ResearchSignal] = {}
        for symbol in symbols:
            headlines = self._fetch_headlines(symbol)
            joined = " ".join(h["title"] for h in headlines)
            det_sentiment = self._keyword_sentiment(joined)
            source_urls = [h["url"] for h in headlines if h["url"]]

            summary = "Deterministic headline sentiment."
            sentiment = det_sentiment
            confidence = 0.45 if headlines else 0.2

            if self.enable_llm and headlines:
                llm = self._llm_overlay(symbol, headlines)
                # Fixed blend keeps behavior deterministic when inputs are the same.
                sentiment = 0.7 * det_sentiment + 0.3 * llm["sentiment"]
                confidence = 0.7 * confidence + 0.3 * llm["confidence"]
                summary = llm["summary"]

            results[symbol] = ResearchSignal(
                symbol=symbol,
                sentiment=float(max(-1.0, min(1.0, sentiment))),
                confidence=float(max(0.0, min(1.0, confidence))),
                summary=summary,
                source_urls=source_urls[: self.max_headlines],
            )
        return results
