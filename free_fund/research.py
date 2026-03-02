from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests

from .contracts import ResearchSignal
from .resilience import with_retries


@dataclass
class ResearchAgent:
    enable_llm: bool = False
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/generate"
    max_headlines: int = 8
    max_news_age_minutes: int = 15
    max_retries: int = 2
    retry_base_delay_sec: float = 0.4

    def _fetch_headlines(self, symbol: str) -> list[dict[str, str]]:
        def _call() -> Any:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            return feedparser.parse(url)

        parsed = with_retries(
            _call,
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
        )
        entries: list[dict[str, str]] = []
        for item in parsed.entries[: self.max_headlines]:
            published = item.get("published_parsed") or item.get("updated_parsed")
            published_iso = ""
            if published is not None:
                published_iso = datetime(*published[:6], tzinfo=timezone.utc).isoformat()
            entries.append(
                {"title": item.get("title", ""), "url": item.get("link", ""), "published_utc": published_iso}
            )
        return entries

    def _is_stale(self, headlines: list[dict[str, str]]) -> bool:
        if not headlines:
            return True
        valid_times: list[datetime] = []
        for h in headlines:
            ts = h.get("published_utc", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                valid_times.append(dt.astimezone(timezone.utc))
            except Exception:
                continue
        if not valid_times:
            return True
        newest = max(valid_times)
        age_min = (datetime.now(timezone.utc) - newest).total_seconds() / 60.0
        return age_min > self.max_news_age_minutes

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
                def _call() -> str:
                    payload = {
                        "model": self.ollama_model,
                        "prompt": rendered_prompt,
                        "stream": False,
                        "format": "json",
                    }
                    resp = requests.post(self.ollama_url, json=payload, timeout=20)
                    resp.raise_for_status()
                    return resp.json().get("response", "{}")

                return with_retries(
                    _call,
                    max_retries=self.max_retries,
                    base_delay_sec=self.retry_base_delay_sec,
                )

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
            is_stale = self._is_stale(headlines)
            joined = " ".join(h["title"] for h in headlines)
            det_sentiment = self._keyword_sentiment(joined)
            source_urls = [h["url"] for h in headlines if h["url"]]

            summary = "Deterministic headline sentiment."
            sentiment = det_sentiment
            confidence = 0.45 if headlines else 0.2

            if is_stale:
                summary = "Research skipped due to stale news feed."
                sentiment = 0.0
                confidence = 0.1
            elif self.enable_llm and headlines:
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
