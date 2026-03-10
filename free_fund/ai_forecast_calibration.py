from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np
import pandas as pd

from llm_router import llm_chat


@dataclass
class ForecastRecord:
    pred: float
    realized: float
    confidence: float


@dataclass
class AIForecastCalibrator:
    """Tracks forecast error and converts raw confidence into calibrated confidence."""

    decay: float = 0.95
    min_history: int = 20
    max_history: int = 500
    records: dict[str, list[ForecastRecord]] = field(default_factory=dict)

    def update(self, ticker: str, pred: float, realized: float, confidence: float) -> None:
        recs = self.records.setdefault(ticker, [])
        recs.append(ForecastRecord(float(pred), float(realized), float(confidence)))
        if len(recs) > self.max_history:
            del recs[: len(recs) - self.max_history]

    def calibration_score(self, ticker: str) -> float:
        recs = self.records.get(ticker, [])
        if len(recs) < self.min_history:
            return 0.60
        weights = np.array([self.decay ** (len(recs) - 1 - i) for i in range(len(recs))], dtype=float)
        weights = weights / float(weights.sum() + 1e-12)
        abs_err = np.array([abs(r.pred - r.realized) for r in recs], dtype=float)
        mae = float((weights * abs_err).sum())
        # Map MAE to [0.1, 0.95] confidence multiplier.
        score = 1.0 / (1.0 + 25.0 * mae)
        return float(min(0.95, max(0.10, score)))

    def calibrated_confidence(self, ticker: str, raw_conf: float) -> float:
        c = float(max(0.0, min(1.0, raw_conf)))
        return float(c * self.calibration_score(ticker))


def _llm_forecast_for_ticker(
    ticker: str,
    returns_tail: pd.Series,
    use_llm: bool,
    timeout: int,
) -> dict:
    # Deterministic fallback is always available.
    mom5 = float(returns_tail.tail(5).mean()) if len(returns_tail) >= 5 else 0.0
    mom20 = float(returns_tail.tail(20).mean()) if len(returns_tail) >= 20 else mom5
    vol20 = float(returns_tail.tail(20).std(ddof=0)) if len(returns_tail) >= 20 else 0.01
    base_pred = 0.7 * mom5 + 0.3 * mom20
    fallback = {
        "expected_return": float(np.clip(base_pred, -0.05, 0.05)),
        "uncertainty": float(max(0.005, min(0.20, vol20))),
        "confidence": 0.4,
        "horizon": "short_term",
        "source": "deterministic_fallback",
    }
    if not use_llm:
        return fallback

    prompt = (
        f"Asset: {ticker}. Last 20 daily returns summary: "
        f"mean={mom20:.5f}, short_mean={mom5:.5f}, vol={vol20:.5f}. "
        "Return JSON with keys expected_return (float -0.05 to 0.05), uncertainty (float 0.005 to 0.2), "
        "confidence (float 0 to 1), horizon (intraday|short_term|medium_term)."
    )
    try:
        raw = llm_chat(prompt=prompt, system="You are a quant forecaster.", json_mode=True, timeout=timeout)
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return fallback
        er = float(data.get("expected_return", fallback["expected_return"]))
        un = float(data.get("uncertainty", fallback["uncertainty"]))
        cf = float(data.get("confidence", fallback["confidence"]))
        hz = str(data.get("horizon", "short_term"))
        out = {
            "expected_return": float(np.clip(er, -0.05, 0.05)),
            "uncertainty": float(np.clip(un, 0.005, 0.20)),
            "confidence": float(np.clip(cf, 0.0, 1.0)),
            "horizon": hz,
            "source": "llm",
        }
        return out
    except Exception:
        return fallback


def generate_ai_forecasts(
    prices_window: pd.DataFrame,
    tickers: list[str],
    calibrator: AIForecastCalibrator | None = None,
    use_llm: bool = False,
    timeout: int = 8,
) -> dict[str, dict[str, float | str]]:
    returns = prices_window.pct_change().dropna()
    out: dict[str, dict[str, float | str]] = {}
    for t in tickers:
        r = returns[t].dropna() if t in returns.columns else pd.Series(dtype=float)
        fc = _llm_forecast_for_ticker(t, r, use_llm=use_llm, timeout=timeout)
        raw_conf = float(fc.get("confidence", 0.0))
        if calibrator is not None:
            fc["confidence"] = calibrator.calibrated_confidence(t, raw_conf)
        out[t] = fc
    return out


__all__ = ["AIForecastCalibrator", "ForecastRecord", "generate_ai_forecasts"]
