from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf


def _zscore(series: pd.Series) -> pd.Series:
    s = series.fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - float(s.mean())) / std).clip(-3, 3)


@dataclass
class AlphaPipelineAgents:
    """High-conviction alpha proxies with free/public inputs and deterministic fallbacks."""

    def earnings_momentum(self, prices: pd.DataFrame) -> pd.Series:
        # Proxy for post-earnings drift: short-term continuation after gap-like move.
        r1 = prices.pct_change(1).iloc[-1]
        r3 = prices.pct_change(3).iloc[-1]
        return _zscore(0.6 * r1 + 0.4 * r3)

    def fii_dii_flow(self, prices: pd.DataFrame) -> pd.Series:
        # Free proxy for institutional flow pressure: return persistence vs realized volatility.
        rets = prices.pct_change().dropna()
        momentum = rets.tail(5).mean()
        vol = rets.tail(20).std(ddof=0).replace(0, np.nan)
        flow_proxy = momentum / vol
        return _zscore(flow_proxy.fillna(0.0))

    def options_flow(self, prices: pd.DataFrame) -> pd.Series:
        # Free proxy for gamma/PCR-like tension using vol-of-vol and directional skew.
        rets = prices.pct_change().dropna()
        vol_short = rets.tail(5).std(ddof=0)
        vol_long = rets.tail(30).std(ddof=0).replace(0, np.nan)
        skew_proxy = rets.tail(10).mean()
        signal = skew_proxy - (vol_short / vol_long).fillna(0.0)
        return _zscore(signal)

    def short_interest(self, prices: pd.DataFrame) -> pd.Series:
        # Free proxy for squeeze risk: strong positive returns + elevated volatility.
        rets = prices.pct_change().dropna()
        r5 = rets.tail(5).mean()
        vol = rets.tail(10).std(ddof=0)
        return _zscore(r5 * vol)

    def block_deals(self, prices: pd.DataFrame) -> pd.Series:
        # Free proxy for block-deal continuation via abnormal move persistence.
        r1 = prices.pct_change(1).iloc[-1]
        r10 = prices.pct_change(10).iloc[-1]
        return _zscore(0.5 * r1 + 0.5 * r10)

    def run(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "earnings_momentum": self.earnings_momentum(prices),
            "fii_dii_flow": self.fii_dii_flow(prices),
            "options_flow": self.options_flow(prices),
            "short_interest": self.short_interest(prices),
            "block_deals": self.block_deals(prices),
        }


@dataclass
class CrossAssetArbitrageAgents:
    """Arbitrage layer with deterministic proxies for markets lacking full feeds."""

    def nse_bse_price_arb(self, prices: pd.DataFrame) -> pd.Series:
        # Placeholder for cross-exchange same-asset spread; zero unless dedicated feeds added.
        return pd.Series(0.0, index=prices.columns)

    def cash_futures_basis(self, prices: pd.DataFrame) -> pd.Series:
        # Proxy: short-term vs medium-term return spread.
        basis_proxy = prices.pct_change(5).iloc[-1] - prices.pct_change(20).iloc[-1]
        return _zscore(basis_proxy)

    def etf_nav_arb(self, prices: pd.DataFrame) -> pd.Series:
        # Proxy: deviation from rolling fair value.
        fair = prices.rolling(20).mean().iloc[-1].replace(0, np.nan)
        prem = (prices.iloc[-1] / fair - 1).fillna(0.0)
        return _zscore(-prem)

    def adr_arb(self, prices: pd.DataFrame) -> pd.Series:
        # Placeholder until ADR/local mapping is provided.
        return pd.Series(0.0, index=prices.columns)

    def run(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        return {
            "nse_bse_price_arb": self.nse_bse_price_arb(prices),
            "cash_futures_basis": self.cash_futures_basis(prices),
            "etf_nav_arb": self.etf_nav_arb(prices),
            "adr_arb": self.adr_arb(prices),
        }


@dataclass
class MacroIntelligenceAgents:
    """Macro regime intelligence layer."""

    def _safe_close(self, symbol: str) -> float:
        try:
            data = yf.download(symbol, period="20d", auto_adjust=True, progress=False)
            if data.empty:
                return 0.0
            if isinstance(data.columns, pd.MultiIndex):
                c = data["Close"].iloc[-1]
                return float(c.iloc[0] if hasattr(c, "iloc") else c)
            return float(data["Close"].iloc[-1])
        except Exception:
            return 0.0

    def rbi_policy_agent(self) -> float:
        # Placeholder policy stance score.
        return 0.0

    def global_carry(self) -> float:
        us10y = self._safe_close("^TNX")
        in10y = self._safe_close("^INDIAVIX")  # fallback proxy when free IN yield series unavailable
        if us10y <= 0 or in10y <= 0:
            return 0.0
        return float((in10y - us10y) / 100.0)

    def crude_gold_corr(self) -> float:
        try:
            data = yf.download(["CL=F", "GC=F"], period="90d", auto_adjust=True, progress=False)
            if data.empty:
                return 0.0
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            returns = close.pct_change().dropna()
            if returns.shape[1] < 2:
                return 0.0
            return float(returns.iloc[:, 0].corr(returns.iloc[:, 1]))
        except Exception:
            return 0.0

    def rupee_regime(self) -> float:
        try:
            data = yf.download("INR=X", period="60d", auto_adjust=True, progress=False)
            if data.empty:
                return 0.0
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"] if "Close" in data else data.iloc[:, 0]
            vol = float(close.pct_change().dropna().std(ddof=0))
            return vol
        except Exception:
            return 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "rbi_policy": self.rbi_policy_agent(),
            "global_carry": self.global_carry(),
            "crude_gold_corr": self.crude_gold_corr(),
            "rupee_regime": self.rupee_regime(),
        }


@dataclass
class MicrostructureAgents:
    """Real-time microstructure placeholders for future tick/order-book feeds."""

    def order_book_agent(self, symbols: list[str]) -> pd.Series:
        return pd.Series(0.0, index=symbols)

    def tape_reading(self, symbols: list[str]) -> pd.Series:
        return pd.Series(0.0, index=symbols)

    def latency_arb(self, symbols: list[str]) -> pd.Series:
        return pd.Series(0.0, index=symbols)

    def flash_crash_detector(self, prices: pd.DataFrame) -> bool:
        returns = prices.pct_change().dropna()
        if returns.empty:
            return False
        return bool(float(returns.tail(1).abs().max().max()) > 0.08)


@dataclass
class AdaptiveLearningLayer:
    """Online deterministic adaptation for strategy weights."""

    decay: float = 0.95
    min_weight: float = 0.05

    def update_weights(
        self,
        current: dict[str, float],
        realized_returns: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not realized_returns:
            return current
        updated: dict[str, float] = {}
        for k, v in current.items():
            perf = float(realized_returns.get(k, 0.0))
            updated[k] = max(self.min_weight, self.decay * v + (1.0 - self.decay) * max(0.0, perf))
        s = sum(updated.values())
        if s <= 1e-12:
            return current
        return {k: v / s for k, v in updated.items()}


@dataclass
class PrivateDataAlphaAgents:
    """Hooks for private alpha feeds (CSV/API). Defaults to neutral."""

    def run(self, symbols: list[str]) -> dict[str, pd.Series]:
        neutral = pd.Series(0.0, index=symbols)
        return {
            "bulk_parking_deals": neutral.copy(),
            "margin_data": neutral.copy(),
            "fo_lot_changes": neutral.copy(),
            "corporate_action_arb": neutral.copy(),
        }
