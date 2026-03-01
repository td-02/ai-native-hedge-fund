from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests


@dataclass
class SignalAgent:
    lookback_days: int
    enable_ollama_overlay: bool = False
    ollama_url: str = 'http://localhost:11434/api/generate'
    ollama_model: str = 'llama3.1:8b'

    def _base_scores(self, window_prices: pd.DataFrame) -> pd.Series:
        ret_21 = window_prices.pct_change(21).iloc[-1]
        ret_63 = window_prices.pct_change(63).iloc[-1]
        ma_20 = window_prices.rolling(20).mean().iloc[-1]
        ma_100 = window_prices.rolling(100).mean().iloc[-1]
        trend = (ma_20 > ma_100).astype(float) * 2 - 1

        raw = 0.35 * ret_21 + 0.45 * ret_63 + 0.20 * trend
        z = (raw - raw.mean()) / (raw.std(ddof=0) + 1e-9)
        return z.clip(-2, 2)

    def _ollama_overlay(self, scores: pd.Series, recent_returns: pd.Series) -> pd.Series:
        prompt = {
            'task': 'Given ETF tickers with current normalized scores and 5-day returns, return small tilt adjustments between -0.15 and 0.15.',
            'format': {'TICKER': 'float'},
            'scores': scores.round(4).to_dict(),
            'returns_5d': recent_returns.round(4).to_dict(),
        }
        payload = {
            'model': self.ollama_model,
            'prompt': json.dumps(prompt),
            'stream': False,
            'format': 'json',
        }
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=20)
            r.raise_for_status()
            out = r.json().get('response', '{}')
            parsed = json.loads(out)
            tilt = pd.Series(parsed, dtype=float).reindex(scores.index).fillna(0.0)
            return tilt.clip(-0.15, 0.15)
        except Exception:
            return pd.Series(0.0, index=scores.index)

    def run(self, window_prices: pd.DataFrame) -> pd.Series:
        scores = self._base_scores(window_prices)
        if not self.enable_ollama_overlay:
            return scores
        recent_returns = window_prices.pct_change(5).iloc[-1]
        tilt = self._ollama_overlay(scores, recent_returns)
        return scores + tilt


@dataclass
class RiskAgent:
    target_annual_vol: float = 0.15
    min_annual_vol_floor: float = 0.10

    def run(self, scores: pd.Series, window_prices: pd.DataFrame) -> pd.Series:
        rets = window_prices.pct_change().dropna()
        ann_vol = rets.std(ddof=0) * np.sqrt(252)
        inv_vol = 1.0 / ann_vol.clip(lower=self.min_annual_vol_floor)
        risk_adjusted = scores * inv_vol
        if risk_adjusted.abs().sum() == 0:
            return risk_adjusted
        return risk_adjusted / risk_adjusted.abs().sum()


@dataclass
class AllocatorAgent:
    max_weight: float = 0.30
    gross_limit: float = 1.00

    def run(self, risk_scores: pd.Series) -> pd.Series:
        w = risk_scores.copy()
        if w.abs().sum() == 0:
            return w

        w = w / w.abs().sum() * self.gross_limit
        w = w.clip(lower=-self.max_weight, upper=self.max_weight)

        gross = w.abs().sum()
        if gross > self.gross_limit and gross > 0:
            w = w / gross * self.gross_limit
        return w.fillna(0.0)
