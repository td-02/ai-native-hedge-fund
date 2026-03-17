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

    def _price_based_fallback(self, prices_window: pd.DataFrame | None = None) -> RegimeSnapshot:
        if prices_window is None or prices_window.empty or len(prices_window) < 30:
            return RegimeSnapshot("trend", 0.55, 2.0, 0.85)

        px = prices_window.dropna(how="all").copy()
        if px.empty:
            return RegimeSnapshot("trend", 0.55, 2.0, 0.85)
        rets = px.pct_change().dropna()
        if rets.empty:
            return RegimeSnapshot("trend", 0.55, 2.0, 0.85)

        horizon = min(63, len(px) - 1)
        r63 = px.pct_change(horizon).iloc[-1].fillna(0.0)
        market_ret = float(r63.mean())
        market_curve = px.mean(axis=1).dropna()
        dd = float((market_curve / market_curve.cummax() - 1.0).iloc[-1]) if len(market_curve) > 1 else 0.0
        breadth = float((r63 > 0).mean())
        ann_vol = float(rets.mean(axis=1).std(ddof=0) * (252 ** 0.5))

        if dd < -0.12 or ann_vol > 0.28 or breadth < 0.30:
            return RegimeSnapshot("stress", 0.78, 1.0, 0.50)
        if market_ret < 0.0 or breadth < 0.45:
            return RegimeSnapshot("meanrev", 0.62, 2.5, 0.80)
        return RegimeSnapshot("trend", 0.70, 3.0, 1.0)

    def run(self, prefer_india: bool = False, prices_window: pd.DataFrame | None = None) -> RegimeSnapshot:
        vix_symbol = self.india_vix_symbol if prefer_india else self.us_vix_symbol
        try:
            vix = self._latest_close(vix_symbol)
            fx = self._latest_close(self.fx_symbol)
        except Exception:
            return self._price_based_fallback(prices_window)

        if vix <= 0:
            return self._price_based_fallback(prices_window)

        if vix >= 30:
            return RegimeSnapshot("stress", 0.8, 1.0, 0.5)
        if vix >= 20:
            return RegimeSnapshot("meanrev", 0.6, 2.5, 0.8)
        # Slight fx volatility penalty proxy for EM risk.
        if fx > 0 and prefer_india and fx > 90:
            return RegimeSnapshot("meanrev", 0.55, 2.0, 0.7)
        return RegimeSnapshot("trend", 0.7, 3.0, 1.0)
