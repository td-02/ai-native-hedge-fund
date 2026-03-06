from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from mcp.server.fastmcp import FastMCP
import pandas as pd
import yfinance as yf

from .config import load_config
from .orchestrator import CentralizedHedgeFundSystem


def _download_close(symbols: list[str], period: str = "6mo") -> pd.DataFrame:
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()
        close.columns = symbols[:1]
    return close.dropna(how="all").ffill().dropna(how="any")


def build_research_mcp_server(
    config_path: str = "configs/default.yaml",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    mcp = FastMCP(
        name="AI Native Hedge Fund Research MCP",
        instructions=(
            "Research tools for market analysis, signal context, and decision previews. "
            "Outputs are structured and deterministic where possible."
        ),
        host=host,
        port=port,
        sse_path="/sse",
        message_path="/messages/",
        streamable_http_path="/mcp",
    )

    @mcp.tool(description="Fetch latest headline snapshot for a symbol from Yahoo Finance RSS.")
    def news_snapshot(symbol: str, max_items: int = 8) -> dict[str, Any]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        parsed = feedparser.parse(url)
        items = []
        for e in parsed.entries[: max(1, min(max_items, 20))]:
            items.append(
                {
                    "title": str(e.get("title", "")),
                    "url": str(e.get("link", "")),
                    "published": str(e.get("published", "")),
                }
            )
        return {"symbol": symbol, "count": len(items), "items": items}

    @mcp.tool(description="Compute basic return/volatility stats for a symbol.")
    def price_stats(symbol: str, period: str = "6mo") -> dict[str, Any]:
        close = _download_close([symbol], period=period)
        if close.empty:
            return {"symbol": symbol, "error": "no_data"}
        s = close.iloc[:, 0]
        rets = s.pct_change().dropna()
        out = {
            "symbol": symbol,
            "period": period,
            "last_price": float(s.iloc[-1]),
            "ret_5d": float(s.pct_change(5).iloc[-1]) if len(s) > 6 else 0.0,
            "ret_20d": float(s.pct_change(20).iloc[-1]) if len(s) > 21 else 0.0,
            "ann_vol_20d": float(rets.tail(20).std(ddof=0) * (252**0.5)) if len(rets) >= 20 else 0.0,
        }
        return out

    @mcp.tool(description="Compare multiple symbols by return and volatility.")
    def peer_compare(symbols: list[str], period: str = "6mo") -> dict[str, Any]:
        symbols = [s.strip().upper() for s in symbols if s and s.strip()]
        close = _download_close(symbols, period=period)
        if close.empty:
            return {"symbols": symbols, "error": "no_data"}
        rows = []
        for col in close.columns:
            s = close[col]
            rets = s.pct_change().dropna()
            rows.append(
                {
                    "symbol": str(col),
                    "ret_20d": float(s.pct_change(20).iloc[-1]) if len(s) > 21 else 0.0,
                    "ret_60d": float(s.pct_change(60).iloc[-1]) if len(s) > 61 else 0.0,
                    "ann_vol_20d": float(rets.tail(20).std(ddof=0) * (252**0.5)) if len(rets) >= 20 else 0.0,
                }
            )
        rows = sorted(rows, key=lambda r: r["ret_20d"], reverse=True)
        return {"period": period, "rows": rows}

    @mcp.tool(description="Macro snapshot: VIX, India VIX, USDINR, crude and gold.")
    def macro_snapshot() -> dict[str, Any]:
        symbols = ["^VIX", "^INDIAVIX", "INR=X", "CL=F", "GC=F"]
        close = _download_close(symbols, period="3mo")
        if close.empty:
            return {"error": "no_data"}
        latest = {str(c): float(close[c].iloc[-1]) for c in close.columns}
        ret_5d = {
            str(c): (float(close[c].pct_change(5).iloc[-1]) if len(close[c]) > 6 else 0.0)
            for c in close.columns
        }
        return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "latest": latest, "ret_5d": ret_5d}

    @mcp.tool(description="Run one dry decision cycle and return target weights and flags.")
    def decision_preview(config_override: str | None = None) -> dict[str, Any]:
        cfg_file = config_override or config_path
        cfg = load_config(cfg_file)
        decision = CentralizedHedgeFundSystem(cfg).run_cycle(execute=False)
        return {
            "run_id": decision.run_id,
            "timestamp_utc": decision.timestamp_utc,
            "target_weights": decision.target_weights,
            "risk_flags": decision.risk_flags,
            "symbols": decision.symbols,
        }

    @mcp.resource("research://readme")
    def research_readme() -> str:
        path = Path("README.md")
        if not path.exists():
            return "README.md not found."
        return path.read_text(encoding="utf-8")

    return mcp
