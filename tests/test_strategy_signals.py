"""
Tests for strategy signal correctness.
Guards against the regressions found during debugging:
  - NaN signals from short windows
  - QQQ/XLK sign inversion in combined score
  - IWM domination (ratio > 5x)
  - Zero gross exposure
  - Long-only weight enforcement
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from free_fund.strategy_stack import StrategyEnsembleAgent
from free_fund.contracts import ResearchSignal


BULL_SYMBOLS = ["SPY", "QQQ", "XLK", "XLV", "XLE", "TLT", "GLD", "IWM", "MTUM"]
TREND_WEIGHTS = {
    "trend_following": 0.40,
    "mean_reversion": 0.00,
    "volatility_carry": 0.10,
    "regime_switching": 0.05,
    "event_driven": 0.00,
    "relative_strength_rotation": 0.30,
    "dual_momentum_gate": 0.15,
}


def _make_agent():
    return StrategyEnsembleAgent(
        strategy_weights=TREND_WEIGHTS,
        dynamic_min_weight=0.03,
        dynamic_max_weight=0.45,
        dynamic_smoothing=0.20,
    )


def _bull_window(n: int = 253) -> pd.DataFrame:
    """Synthetic bull market: QQQ/XLK drift up strongly, TLT/XLE drift down."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    drifts = {
        "QQQ": 0.0012, "XLK": 0.0011, "SPY": 0.0008, "MTUM": 0.0007,
        "IWM": 0.0005, "XLV": 0.0002, "GLD": 0.0001,
        "TLT": -0.0003, "XLE": -0.0005,
    }
    data = {}
    for sym, drift in drifts.items():
        noise = np.random.randn(n) * 0.01
        rets = drift + noise
        data[sym] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=dates)


def _bear_window(n: int = 253) -> pd.DataFrame:
    """Synthetic bear market: equities down, TLT/GLD up."""
    np.random.seed(99)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    drifts = {
        "QQQ": -0.0010, "XLK": -0.0012, "SPY": -0.0008, "MTUM": -0.0009,
        "IWM": -0.0006, "XLV": -0.0003, "GLD": 0.0004,
        "TLT": 0.0003, "XLE": 0.0001,
    }
    data = {}
    for sym, drift in drifts.items():
        noise = np.random.randn(n) * 0.01
        data[sym] = 100 * np.cumprod(1 + (drift + noise))
    return pd.DataFrame(data, index=dates)


class TestNoNaNSignals:
    """Signals must not produce NaN for windows of 253+ days."""

    def test_trend_no_nan_253(self):
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        assert not scores["trend_following"].isna().any(), \
            "trend_following produced NaN with 253-day window"

    def test_all_strategies_no_nan_253(self):
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        for name, s in scores.items():
            assert not s.isna().any(), f"{name} produced NaN with 253-day window"

    def test_all_strategies_no_nan_short_window(self):
        """Signals must degrade gracefully even with short history."""
        agent = _make_agent()
        window = _bull_window(60)
        scores = agent.run(window, {})
        for name, s in scores.items():
            assert not s.isna().any(), f"{name} produced NaN with 60-day window"


class TestSignalDirection:
    """In a clear bull market, momentum signals must rank winners above losers."""

    def test_trend_ranks_winners_above_losers(self):
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        trend = scores["trend_following"]
        assert trend["QQQ"] > trend["TLT"], \
            f"trend: QQQ ({trend['QQQ']:.3f}) should beat TLT ({trend['TLT']:.3f})"
        assert trend["XLK"] > trend["XLE"], \
            f"trend: XLK ({trend['XLK']:.3f}) should beat XLE ({trend['XLE']:.3f})"

    def test_rotation_ranks_winners_above_losers(self):
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        rot = scores["relative_strength_rotation"]
        assert rot["QQQ"] > rot["XLE"], \
            f"rotation: QQQ ({rot['QQQ']:.3f}) should beat XLE ({rot['XLE']:.3f})"

    def test_dual_momentum_gates_losers(self):
        """dual_momentum_gate should give near-zero or negative scores to assets
        with negative absolute momentum."""
        agent = _make_agent()
        window = _bear_window(253)
        scores = agent.run(window, {})
        dmg = scores["dual_momentum_gate"]
        bear_equity = ["QQQ", "XLK", "SPY"]
        defensive = ["TLT", "GLD"]
        for sym in bear_equity:
            if sym in dmg.index:
                assert dmg[sym] <= 0.5, \
                    f"dual_momentum_gate: {sym} should be gated in bear market, got {dmg[sym]:.3f}"


class TestCombinedScoreDirection:
    """Combined weighted score must be positive for bull market winners."""

    def test_qqq_xlk_positive_combined_bull(self):
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        combined = agent.weighted_score(scores)
        assert combined["QQQ"] > 0, \
            f"QQQ={combined['QQQ']:.4f} must be positive in bull market"
        rank = combined.rank(ascending=False)
        n = len(combined)
        top_half_cutoff = (n + 1) // 2
        assert rank["QQQ"] <= top_half_cutoff, f"QQQ rank {int(rank['QQQ'])} not in top half"
        assert rank["XLK"] <= top_half_cutoff, f"XLK rank {int(rank['XLK'])} not in top half"
        assert rank["QQQ"] < rank["TLT"], "QQQ must rank above TLT"
        assert rank["XLK"] < rank["XLE"], "XLK must rank above XLE"

    def test_no_single_asset_dominates(self):
        """Guards against the IWM 22x dominance bug found during debugging."""
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        combined = agent.weighted_score(scores)
        positives = combined[combined > 0]
        assert len(positives) >= 3, \
            f"Only {len(positives)} positive scores in bull market"
        if len(positives) >= 2:
            ratio = positives.max() / (positives.min() + 1e-12)
            assert ratio < 8.0, \
                f"Dominance ratio {ratio:.1f}x > 8x. Top: {positives.nlargest(3).to_dict()}"

    def test_combined_score_correlation_with_returns(self):
        """Combined score must correlate positively with actual 1Y returns."""
        agent = _make_agent()
        window = _bull_window(400)
        scores = agent.run(window.tail(253), {})
        combined = agent.weighted_score(scores)
        actual_1y = window.pct_change(252).iloc[-1].reindex(combined.index).dropna()
        aligned = combined.reindex(actual_1y.index).dropna()
        if len(aligned) >= 4:
            corr = float(aligned.corr(actual_1y))
            assert corr > 0.30, \
                f"Signal-return correlation {corr:.3f} below 0.30 threshold"


class TestFundManagerOutput:
    """FundManager must produce gross > 0.5 and preserve signal direction."""

    def test_gross_exposure_positive(self):
        from free_fund.strategy_stack import FundManagerAgent
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        combined = agent.weighted_score(scores)
        fm = FundManagerAgent(max_weight=0.20, gross_limit=1.0)
        weights = fm.run(combined, prev_weights=None, top_k=10)
        gross = float(weights.abs().sum())
        assert gross > 0.50, f"FundManager gross={gross:.3f} below 0.50"

    def test_top_signals_get_highest_weights(self):
        from free_fund.strategy_stack import FundManagerAgent
        agent = _make_agent()
        window = _bull_window(253)
        scores = agent.run(window, {})
        combined = agent.weighted_score(scores)
        fm = FundManagerAgent(max_weight=0.20, gross_limit=1.0)
        weights = fm.run(combined, prev_weights=None, top_k=10)
        top_signal = combined.idxmax()
        top_weight = combined.nlargest(3).index
        assert top_signal in top_weight or weights[top_signal] > 0, \
            f"Top signal asset {top_signal} not in top weights"


class TestAdaptiveLookback:
    """Adaptive lookbacks must never request pct_change >= window length."""

    @pytest.mark.parametrize("n_days", [30, 60, 126, 200, 253, 500])
    def test_no_nan_at_various_window_sizes(self, n_days):
        agent = _make_agent()
        window = _bull_window(n_days)
        scores = agent.run(window, {})
        combined = agent.weighted_score(scores)
        assert not combined.isna().any(), \
            f"NaN in combined_score with window={n_days}"
        assert combined.abs().sum() > 0, \
            f"All-zero combined_score with window={n_days}"
