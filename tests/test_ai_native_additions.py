from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


def make_prices(n=100, tickers=["AAPL", "MSFT"]):
    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame(np.random.lognormal(0, 0.01, (n, len(tickers))), index=dates, columns=tickers).cumprod()


MOCK_TREND_RESPONSE = '{"trend_strength": 0.4, "trend_continuation_prob": 0.7, "reversal_risk": "low", "conviction": 0.8}'
MOCK_PAIRS_RESPONSE = '{"causal_link_strength": 0.8, "link_type": "fundamental", "trade_signal": 0.3, "stop_if_zscore_exceeds": 3.0}'
MOCK_REGIME_RESPONSE = '{"confirmed_regime": "bull", "regime_confidence": 0.75, "recommended_gross_exposure": 1.1, "favored_sectors": ["tech"], "risk_off_signal": false}'
MOCK_SENTIMENT_RESPONSE = '{"sentiment_score": 0.3, "confidence": 0.7, "key_catalysts": ["earnings beat"], "time_horizon": "short_term", "surprising_factor": 0.2}'


def test_black_scholes_greeks():
    from probabilistic_core import BlackScholes

    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    greeks = bs.greeks()
    assert 0.5 < greeks["delta"] < 0.7, "ATM call delta should be 0.5-0.7"
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    price = bs.price("call")
    assert price > 0


def test_black_scholes_implied_vol():
    from probabilistic_core import BlackScholes

    bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    market_price = bs.price("call")
    iv = bs.implied_vol(market_price, "call")
    assert abs(iv - 0.2) < 0.01, "IV should recover input sigma"


def test_kalman_convergence():
    from probabilistic_core import KalmanSignalTracker

    tracker = KalmanSignalTracker()
    t = np.linspace(0, 4 * np.pi, 100)
    noisy = np.sin(t) + np.random.normal(0, 0.3, 100)
    filtered = [tracker.update(x)["filtered_signal"] for x in noisy]
    assert np.std(filtered) < np.std(noisy), "Kalman should smooth signal"


def test_bayesian_aggregator():
    from probabilistic_core import BayesianSignalAggregator

    agg = BayesianSignalAggregator(["a", "b", "c"])
    agg.update_signal("a", 0.5, 0.1)
    agg.update_signal("b", 0.3, 0.2)
    agg.update_signal("c", 0.7, 0.3)
    result = agg.get_combined_signal()
    assert "combined" in result
    assert "uncertainty" in result
    assert result["uncertainty"] < 0.1, "Combined uncertainty should be less than smallest input"


def test_hmm_fit_predict():
    from probabilistic_core import RegimeHMM

    np.random.seed(42)
    regime1 = np.random.normal(0.001, 0.01, 100)
    regime2 = np.random.normal(-0.002, 0.025, 100)
    returns = np.concatenate([regime1, regime2])
    hmm = RegimeHMM(n_regimes=4)
    hmm.fit(returns)
    proba = hmm.predict_proba(returns)
    assert abs(sum([proba["bull"], proba["bear"], proba["crisis"], proba["sideways"]]) - 1.0) < 0.01


def test_bl_optimizer():
    from probabilistic_core import BayesianPortfolioOptimizer

    prices = make_prices(120, ["AAPL", "MSFT", "GOOG"])
    opt = BayesianPortfolioOptimizer()
    opt.set_market_prior(prices)
    opt.add_llm_view({"assets": ["AAPL"], "outperformance": 0.05, "confidence": 0.8})
    result = opt.optimize()
    assert "weights" in result
    weights = result["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1"
    assert weights["AAPL"] > weights["MSFT"], "Viewed asset should get higher weight"


def test_llm_router_fallback():
    with patch("llm_router.llm_chat") as mock_chat:
        mock_chat.return_value = '{"sentiment_score": 0.0, "confidence": 0.0, "key_catalysts": [], "time_horizon": "short_term", "surprising_factor": 0.0}'
        from research import get_llm_sentiment

        result = get_llm_sentiment(["Apple reports earnings"], "AAPL")
        assert "sentiment_score" in result
        assert -1.0 <= result["sentiment_score"] <= 1.0


@patch("llm_router.llm_chat", return_value=MOCK_TREND_RESPONSE)
def test_llm_trend_strategy(mock_llm):
    from strategy_stack import LLMTrendStrategy

    prices = make_prices(60)
    strategy = LLMTrendStrategy(tickers=["AAPL", "MSFT"], config={})
    signals = strategy.generate_signals(prices, headlines_by_ticker={"AAPL": ["strong quarter"], "MSFT": ["cloud growth"]})
    for ticker in ["AAPL", "MSFT"]:
        assert ticker in signals
        assert "signal" in signals[ticker]
        assert "kalman_uncertainty" in signals[ticker]


@patch("llm_router.llm_chat", return_value=MOCK_PAIRS_RESPONSE)
def test_llm_pairs_strategy(mock_llm):
    from strategy_stack import LLMPairsTradingStrategy

    prices = make_prices(60)
    strategy = LLMPairsTradingStrategy(pairs=[("AAPL", "MSFT")], config={})
    signals = strategy.generate_signals(prices)
    assert "AAPL_MSFT" in signals
    assert "signal" in signals["AAPL_MSFT"]


def test_options_alpha_fallback():
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.option_chain.side_effect = Exception("No options data")
        from advanced_alpha import get_options_alpha

        prices = make_prices(60, ["AAPL"])
        result = get_options_alpha("AAPL", prices)
        assert result["options_signal"] == 0.0, "Should return neutral on failure"


def test_combine_advanced_alpha_no_crash():
    with patch("advanced_alpha.get_options_alpha", return_value={"options_signal": 0.1, "atm_iv": 0.2, "skew": 0.0, "llm_interpretation": {}}):
        with patch("advanced_alpha.get_earnings_alpha", return_value={"earnings_signal": 0.2, "days_to_earnings": 15, "llm_forecast": {}}):
            with patch("advanced_alpha.get_macro_alpha", return_value={"macro_signal": -0.1, "sector_tilts": {}, "macro_regime": "goldilocks", "yield_curve_slope": 0.01}):
                from advanced_alpha import combine_advanced_alpha

                prices = make_prices(60, ["AAPL"])
                result = combine_advanced_alpha("AAPL", prices, {"vix": 18, "yield_10y": 0.04, "yield_2y": 0.05, "dxy": 103})
                assert "combined_alpha" in result
                assert "uncertainty" in result
                assert "breakdown" in result
