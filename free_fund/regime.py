from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import yfinance as yf


@dataclass
class RegimeSnapshot:
    regime: str
    confidence: float
    leverage_cap: float
    risk_multiplier: float


@dataclass
class MacroRegimeAgent:
    india_vix_symbol: str = "^INDIAVIX"
    us_vix_symbol: str = "^VIX"
    fx_symbol: str = "INR=X"

    def _latest_close(self, symbol: str) -> float:
        data = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        if data.empty:
            return 0.0
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[-1]
            return float(close.iloc[0] if hasattr(close, "iloc") else close)
        return float(data["Close"].iloc[-1])

    def run(self, prefer_india: bool = False) -> RegimeSnapshot:
        vix_symbol = self.india_vix_symbol if prefer_india else self.us_vix_symbol
        vix = self._latest_close(vix_symbol)
        fx = self._latest_close(self.fx_symbol)

        if vix >= 30:
            return RegimeSnapshot("stress", 0.8, 1.0, 0.5)
        if vix >= 20:
            return RegimeSnapshot("meanrev", 0.6, 2.5, 0.8)
        # Slight fx volatility penalty proxy for EM risk.
        if fx > 0 and prefer_india and fx > 90:
            return RegimeSnapshot("meanrev", 0.55, 2.0, 0.7)
        return RegimeSnapshot("trend", 0.7, 3.0, 1.0)
