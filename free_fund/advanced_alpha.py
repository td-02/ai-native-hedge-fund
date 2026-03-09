from __future__ import annotations

from dataclasses import dataclass
import math
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from probabilistic_core import BlackScholes, BayesianSignalAggregator
except Exception:  # pragma: no cover
    BlackScholes = None  # type: ignore
    BayesianSignalAggregator = None  # type: ignore
from llm_router import llm_chat


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


def _safe_llm_json(prompt: str, fallback: dict) -> dict:
    try:
        raw = llm_chat(prompt=prompt, system="", json_mode=True, timeout=15)
        data = json.loads(raw) if raw else {}
        if isinstance(data, dict):
            return data
        return dict(fallback)
    except Exception:
        return dict(fallback)


def get_options_alpha(ticker: str, prices_df: pd.DataFrame) -> dict:
    fallback = {"options_signal": 0.0, "atm_iv": 0.2, "skew": 0.0, "llm_interpretation": {}}
    try:
        t = yf.Ticker(ticker)
        expiries = list(t.options or [])
        if not expiries:
            return fallback

        spot = float(prices_df[ticker].dropna().iloc[-1]) if ticker in prices_df.columns and not prices_df[ticker].dropna().empty else 100.0
        near_expiry = None
        near_chain = None
        for e in expiries:
            chain = t.option_chain(e)
            calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
            puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()
            oi = float(calls.get("openInterest", pd.Series(dtype=float)).fillna(0.0).sum()) + float(
                puts.get("openInterest", pd.Series(dtype=float)).fillna(0.0).sum()
            )
            if oi > 100:
                near_expiry = e
                near_chain = chain
                break
        if near_chain is None:
            return fallback

        calls = near_chain.calls.copy()
        puts = near_chain.puts.copy()
        if calls.empty or puts.empty:
            return fallback

        atm_idx = (calls["strike"] - spot).abs().idxmin()
        atm_strike = float(calls.loc[atm_idx, "strike"])
        atm_market = float(calls.loc[atm_idx, "lastPrice"]) if "lastPrice" in calls.columns else float(calls.loc[atm_idx, "bid"])
        expiry_ts = pd.Timestamp(near_expiry)
        ttm = max(1 / 365.0, float((expiry_ts - pd.Timestamp.utcnow()).total_seconds()) / (365.0 * 24.0 * 3600.0))

        atm_iv = 0.2
        if BlackScholes is not None and np.isfinite(atm_market) and atm_market > 0:
            try:
                bs = BlackScholes(S=spot, K=atm_strike, T=ttm, r=0.05, sigma=0.2)
                iv = bs.implied_vol(market_price=atm_market, option_type="call")
                if np.isfinite(iv):
                    atm_iv = float(iv)
            except Exception:
                atm_iv = 0.2

        low = 0.95 * spot
        high = 1.05 * spot
        otm_put = puts[(puts["strike"] < spot) & (puts["strike"] >= low)]
        otm_call = calls[(calls["strike"] > spot) & (calls["strike"] <= high)]
        put_iv = float(otm_put.get("impliedVolatility", pd.Series(dtype=float)).dropna().mean()) if not otm_put.empty else np.nan
        call_iv = float(otm_call.get("impliedVolatility", pd.Series(dtype=float)).dropna().mean()) if not otm_call.empty else np.nan
        if not np.isfinite(put_iv):
            put_iv = atm_iv
        if not np.isfinite(call_iv):
            call_iv = atm_iv
        skew = float(put_iv - call_iv)

        term = 0.0
        if len(expiries) >= 2:
            far_chain = t.option_chain(expiries[-1])
            far_calls = far_chain.calls.copy()
            if not far_calls.empty and "strike" in far_calls.columns:
                far_idx = (far_calls["strike"] - spot).abs().idxmin()
                near_iv = float(calls.loc[atm_idx, "impliedVolatility"]) if "impliedVolatility" in calls.columns else atm_iv
                far_iv = float(far_calls.loc[far_idx, "impliedVolatility"]) if "impliedVolatility" in far_calls.columns else near_iv
                if np.isfinite(near_iv) and np.isfinite(far_iv):
                    term = float(near_iv - far_iv)

        llm_fb = {
            "options_sentiment": 0.0,
            "tail_risk_elevated": False,
            "smart_money_direction": "neutral",
            "iv_regime": "fair",
        }
        llm_data = _safe_llm_json(
            (
                f"Options market for {ticker}: ATM IV={atm_iv:.2%}, put/call skew={skew:.3f}, term structure={term:.3f}. "
                "Return JSON: {'options_sentiment': float -1 to 1, 'tail_risk_elevated': bool, "
                "'smart_money_direction': 'bullish|bearish|neutral', 'iv_regime': 'cheap|fair|expensive'}"
            ),
            llm_fb,
        )
        options_signal = float(max(-1.0, min(1.0, llm_data.get("options_sentiment", 0.0))))
        return {
            "options_signal": options_signal,
            "atm_iv": float(atm_iv),
            "skew": float(skew),
            "llm_interpretation": llm_data,
        }
    except Exception:
        return fallback


def get_earnings_alpha(ticker: str) -> dict:
    fallback = {"earnings_signal": 0.0, "days_to_earnings": None, "llm_forecast": {}}
    try:
        tk = yf.Ticker(ticker)
        days_to_earnings = None
        try:
            cal = tk.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                vals = cal.values.flatten().tolist()
                for v in vals:
                    try:
                        ts = pd.Timestamp(v)
                        delta = int((ts - pd.Timestamp.utcnow()).days)
                        if delta >= 0:
                            days_to_earnings = delta
                            break
                    except Exception:
                        continue
        except Exception:
            days_to_earnings = None

        surprise_mean = 0.0
        surprise_std = 0.0
        try:
            ed = tk.earnings_dates
            if isinstance(ed, pd.DataFrame) and not ed.empty:
                ed2 = ed.head(4)
                surprises = []
                for _, r in ed2.iterrows():
                    est = float(r.get("EPS Estimate", np.nan))
                    rep = float(r.get("Reported EPS", np.nan))
                    if np.isfinite(est) and np.isfinite(rep) and abs(est) > 1e-12:
                        surprises.append((rep - est) / abs(est))
                if surprises:
                    surprise_mean = float(np.mean(surprises))
                    surprise_std = float(np.std(surprises, ddof=0))
        except Exception:
            pass

        llm_fb = {
            "pre_earnings_bias": 0.0,
            "surprise_probability": 0.5,
            "surprise_direction": "inline",
            "confidence": 0.0,
            "recommended_position_size_scalar": 1.0,
        }
        llm_data = _safe_llm_json(
            (
                f"Company {ticker}: days to next earnings={days_to_earnings}, "
                f"historical EPS surprise mean={surprise_mean:.2%}, std={surprise_std:.2%}. "
                "Return JSON: {'pre_earnings_bias': float -1 to 1, 'surprise_probability': float 0-1, "
                "'surprise_direction': 'beat|miss|inline', 'confidence': float 0-1, "
                "'recommended_position_size_scalar': float 0.5-1.5}"
            ),
            llm_fb,
        )
        sig = float(max(-1.0, min(1.0, llm_data.get("pre_earnings_bias", 0.0))))
        conf = float(max(0.0, min(1.0, llm_data.get("confidence", 0.0))))
        earnings_signal = float(sig * max(0.5, conf))
        return {
            "earnings_signal": earnings_signal,
            "days_to_earnings": days_to_earnings,
            "llm_forecast": llm_data,
        }
    except Exception:
        return fallback


def get_macro_alpha(macro_snapshot: dict) -> dict:
    fallback = {
        "macro_signal": 0.0,
        "sector_tilts": {},
        "macro_regime": "unknown",
        "yield_curve_slope": 0.0,
    }
    try:
        vix = float(macro_snapshot.get("vix", 20.0))
        y10 = float(macro_snapshot.get("yield_10y", 0.04))
        y2 = float(macro_snapshot.get("yield_2y", 0.05))
        dxy = float(macro_snapshot.get("dxy", 103.0))
        slope = float(y10 - y2)
        if vix < 15:
            vix_regime = "low"
        elif vix < 20:
            vix_regime = "normal"
        elif vix <= 30:
            vix_regime = "high"
        else:
            vix_regime = "extreme"

        llm_fb = {
            "risk_appetite": 0.0,
            "equity_headwind": 0.0,
            "sector_rotation_signal": {"defensive": 0.0, "cyclical": 0.0, "growth": 0.0, "value": 0.0},
            "macro_regime": "unknown",
        }
        llm_data = _safe_llm_json(
            (
                f"Macro: VIX={vix} ({vix_regime}), yield curve={slope:.2%}, DXY={dxy}. "
                "Return JSON: {'risk_appetite': float -1 to 1, 'equity_headwind': float -1 to 1, "
                "'sector_rotation_signal': {'defensive': float, 'cyclical': float, 'growth': float, 'value': float}, "
                "'macro_regime': 'goldilocks|stagflation|recession|recovery|overheating'}"
            ),
            llm_fb,
        )
        risk_appetite = float(max(-1.0, min(1.0, llm_data.get("risk_appetite", 0.0))))
        headwind = float(max(-1.0, min(1.0, llm_data.get("equity_headwind", 0.0))))
        macro_signal = float(0.6 * risk_appetite - 0.4 * headwind)
        return {
            "macro_signal": macro_signal,
            "sector_tilts": dict(llm_data.get("sector_rotation_signal", {})),
            "macro_regime": str(llm_data.get("macro_regime", "unknown")),
            "yield_curve_slope": slope,
        }
    except Exception:
        return fallback


def combine_advanced_alpha(ticker: str, prices_df: pd.DataFrame, macro_snapshot: dict) -> dict:
    logger = logging.getLogger(__name__)
    opt_fallback = {"options_signal": 0.0, "atm_iv": 0.2, "skew": 0.0, "llm_interpretation": {}}
    earn_fallback = {"earnings_signal": 0.0, "days_to_earnings": None, "llm_forecast": {}}
    macro_fallback = {"macro_signal": 0.0, "sector_tilts": {}, "macro_regime": "unknown", "yield_curve_slope": 0.0}

    try:
        options_result = get_options_alpha(ticker, prices_df)
    except Exception as e:
        logger.warning("options alpha failed for %s: %s", ticker, e)
        options_result = dict(opt_fallback)

    try:
        earnings_result = get_earnings_alpha(ticker)
    except Exception as e:
        logger.warning("earnings alpha failed for %s: %s", ticker, e)
        earnings_result = dict(earn_fallback)

    try:
        macro_result = get_macro_alpha(macro_snapshot)
    except Exception as e:
        logger.warning("macro alpha failed for %s: %s", ticker, e)
        macro_result = dict(macro_fallback)

    if BayesianSignalAggregator is not None:
        agg = BayesianSignalAggregator(["options", "earnings", "macro"])
        agg.update_signal("options", float(options_result.get("options_signal", 0.0)), std=0.3)
        agg.update_signal("earnings", float(earnings_result.get("earnings_signal", 0.0)), std=0.4)
        agg.update_signal("macro", float(macro_result.get("macro_signal", 0.0)), std=0.2)
        combined = agg.get_combined_signal()
    else:
        vals = [
            float(options_result.get("options_signal", 0.0)),
            float(earnings_result.get("earnings_signal", 0.0)),
            float(macro_result.get("macro_signal", 0.0)),
        ]
        combined = {"combined": float(np.mean(vals)), "uncertainty": 1.0}

    return {
        "combined_alpha": float(combined.get("combined", 0.0)),
        "uncertainty": float(combined.get("uncertainty", 1.0)),
        "breakdown": {
            "options": float(options_result.get("options_signal", 0.0)),
            "earnings": float(earnings_result.get("earnings_signal", 0.0)),
            "macro": float(macro_result.get("macro_signal", 0.0)),
        },
        "macro_regime": str(macro_result.get("macro_regime", "unknown")),
        "days_to_earnings": earnings_result.get("days_to_earnings", None),
    }
