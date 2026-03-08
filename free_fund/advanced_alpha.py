from __future__ import annotations

from dataclasses import dataclass
import math
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
    enabled_signals: list[str] | None = None

    def _has_signal(self, name: str) -> bool:
        if not self.enabled_signals:
            return True
        return name in set(self.enabled_signals)

    def _safe_close(self, symbol: str, period: str = "9mo") -> pd.Series:
        try:
            data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            if data.empty:
                return pd.Series(dtype=float)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"] if "Close" in data else data.iloc[:, 0]
            return close.dropna().astype(float)
        except Exception:
            return pd.Series(dtype=float)

    def earnings_momentum(self, prices: pd.DataFrame) -> pd.Series:
        # Post-earnings drift proxy: combine known earnings surprise with recent event-window momentum.
        out: dict[str, float] = {}
        for s in prices.columns:
            surprise = 0.0
            try:
                t = yf.Ticker(s)
                ed = t.earnings_dates
                if isinstance(ed, pd.DataFrame) and not ed.empty:
                    row = ed.head(1).iloc[0]
                    est = float(row.get("EPS Estimate", 0.0) or 0.0)
                    rep = float(row.get("Reported EPS", 0.0) or 0.0)
                    if abs(est) > 1e-12:
                        surprise = (rep - est) / abs(est)
            except Exception:
                surprise = 0.0
            r1 = float(prices[s].pct_change(1).iloc[-1])
            r3 = float(prices[s].pct_change(3).iloc[-1])
            out[s] = 0.6 * surprise + 0.25 * r1 + 0.15 * r3
        return _zscore(pd.Series(out, dtype=float))

    def analyst_revisions(self, prices: pd.DataFrame) -> pd.Series:
        # Analyst revision signal from recommendation trend and net upgrade pressure.
        out: dict[str, float] = {}
        for s in prices.columns:
            score = 0.0
            try:
                t = yf.Ticker(s)
                rec = t.recommendations
                if isinstance(rec, pd.DataFrame) and not rec.empty:
                    recent = rec.tail(12)
                    to_grade = recent.get("To Grade")
                    from_grade = recent.get("From Grade")
                    if to_grade is not None and from_grade is not None:
                        upgrades = 0
                        downgrades = 0
                        for tg, fg in zip(to_grade.fillna(""), from_grade.fillna("")):
                            tgt = str(tg).lower()
                            frm = str(fg).lower()
                            if any(k in tgt for k in ("buy", "overweight", "outperform")) and not any(
                                k in frm for k in ("buy", "overweight", "outperform")
                            ):
                                upgrades += 1
                            if any(k in tgt for k in ("sell", "underperform")) and not any(
                                k in frm for k in ("sell", "underperform")
                            ):
                                downgrades += 1
                        score = float(upgrades - downgrades)
            except Exception:
                score = 0.0
            # Deterministic fallback if no recommendation feed.
            if abs(score) <= 1e-12:
                score = float(prices[s].pct_change(20).iloc[-1] - prices[s].pct_change(5).iloc[-1])
            out[s] = score
        return _zscore(pd.Series(out, dtype=float))

    def options_iv_term_structure(self, prices: pd.DataFrame) -> pd.Series:
        # Term structure: near IV minus far IV; positive suggests event/tension risk.
        out: dict[str, float] = {}
        for s in prices.columns:
            value = 0.0
            try:
                t = yf.Ticker(s)
                expiries = list(t.options or [])
                if len(expiries) >= 2:
                    c0 = t.option_chain(expiries[0]).calls
                    c1 = t.option_chain(expiries[min(2, len(expiries) - 1)]).calls
                    if not c0.empty and not c1.empty:
                        iv0 = float(c0["impliedVolatility"].dropna().mean())
                        iv1 = float(c1["impliedVolatility"].dropna().mean())
                        if math.isfinite(iv0) and math.isfinite(iv1):
                            value = iv0 - iv1
            except Exception:
                value = 0.0
            # Fallback to realized vol slope.
            if abs(value) <= 1e-12:
                rets = prices[s].pct_change().dropna()
                rv_short = float(rets.tail(5).std(ddof=0)) if len(rets) >= 5 else 0.0
                rv_long = float(rets.tail(30).std(ddof=0)) if len(rets) >= 30 else 0.0
                value = rv_short - rv_long
            out[s] = value
        return _zscore(pd.Series(out, dtype=float))

    def short_interest(self, prices: pd.DataFrame) -> pd.Series:
        # Short squeeze proxy: positive momentum with high vol.
        rets = prices.pct_change().dropna()
        r5 = rets.tail(5).mean()
        vol = rets.tail(10).std(ddof=0)
        return _zscore(r5 * vol)

    def block_deals(self, prices: pd.DataFrame) -> pd.Series:
        # Block-deal continuation proxy via abnormal move persistence.
        r1 = prices.pct_change(1).iloc[-1]
        r10 = prices.pct_change(10).iloc[-1]
        return _zscore(0.5 * r1 + 0.5 * r10)

    def volume_liquidity_shock(self, prices: pd.DataFrame) -> pd.Series:
        # Use volume shock when available; fallback to return shock.
        out: dict[str, float] = {}
        for s in prices.columns:
            shock = 0.0
            try:
                hist = yf.download(s, period="6mo", auto_adjust=True, progress=False)
                if isinstance(hist, pd.DataFrame) and not hist.empty and "Volume" in hist.columns:
                    v = hist["Volume"].dropna().astype(float)
                    if len(v) >= 30:
                        adv20 = float(v.tail(20).mean())
                        adv60 = float(v.tail(60).mean()) if len(v) >= 60 else adv20
                        if adv60 > 1e-12:
                            shock = (adv20 / adv60) - 1.0
            except Exception:
                shock = 0.0
            if abs(shock) <= 1e-12:
                r = prices[s].pct_change().dropna()
                if len(r) >= 20:
                    shock = float(r.tail(5).std(ddof=0) - r.tail(20).std(ddof=0))
            out[s] = shock
        return _zscore(pd.Series(out, dtype=float))

    def run(self, prices: pd.DataFrame) -> dict[str, pd.Series]:
        signals: dict[str, pd.Series] = {}
        if self._has_signal("earnings_momentum"):
            signals["earnings_momentum"] = self.earnings_momentum(prices)
        if self._has_signal("analyst_revisions"):
            signals["analyst_revisions"] = self.analyst_revisions(prices)
        if self._has_signal("options_iv_term_structure"):
            signals["options_iv_term_structure"] = self.options_iv_term_structure(prices)
        if self._has_signal("volume_liquidity_shock"):
            signals["volume_liquidity_shock"] = self.volume_liquidity_shock(prices)
        if self._has_signal("short_interest"):
            signals["short_interest"] = self.short_interest(prices)
        if self._has_signal("block_deals"):
            signals["block_deals"] = self.block_deals(prices)
        return signals


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
