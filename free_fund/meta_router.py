from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd


@dataclass
class RegimeRoutingResult:
    regime: str
    strategy_weights: dict[str, float]
    gross_scale: float
    notes: list[str]


class MetaAgentRouter:
    """Deterministic regime-aware router for strategy activation and risk budget."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.default_regime = str(cfg.get("default_regime", "sideways"))
        self.gross_scale_by_regime = dict(
            cfg.get(
                "gross_scale_by_regime",
                {
                    "bull": 1.0,
                    "bear": 0.75,
                    "crisis": 0.45,
                    "sideways": 0.85,
                    "trend": 1.0,
                    "meanrev": 0.85,
                    "stress": 0.50,
                },
            )
        )
        self.strategy_weights_by_regime = dict(
            cfg.get(
                "strategy_weights_by_regime",
                {
                    "bull": {
                        "trend_following": 0.40,
                        "event_driven": 0.20,
                        "regime_switching": 0.20,
                        "mean_reversion": 0.05,
                        "volatility_carry": 0.05,
                        "relative_strength_rotation": 0.05,
                        "dual_momentum_gate": 0.05,
                    },
                    "bear": {
                        "trend_following": 0.20,
                        "event_driven": 0.10,
                        "regime_switching": 0.30,
                        "mean_reversion": 0.15,
                        "volatility_carry": 0.20,
                        "relative_strength_rotation": 0.00,
                        "dual_momentum_gate": 0.05,
                    },
                    "crisis": {
                        "trend_following": 0.05,
                        "event_driven": 0.05,
                        "regime_switching": 0.35,
                        "mean_reversion": 0.20,
                        "volatility_carry": 0.30,
                        "relative_strength_rotation": 0.00,
                        "dual_momentum_gate": 0.05,
                    },
                    "sideways": {
                        "trend_following": 0.18,
                        "event_driven": 0.12,
                        "regime_switching": 0.20,
                        "mean_reversion": 0.25,
                        "volatility_carry": 0.15,
                        "relative_strength_rotation": 0.05,
                        "dual_momentum_gate": 0.05,
                    },
                },
            )
        )
        self.min_strategy_weight = float(cfg.get("min_strategy_weight", 0.03))
        self.max_strategy_weight = float(cfg.get("max_strategy_weight", 0.60))
        self.quality_tilt_strength = float(cfg.get("quality_tilt_strength", 0.35))

    @staticmethod
    def _normalize(weights: dict[str, float], names: Iterable[str] | None = None) -> dict[str, float]:
        out = {k: max(0.0, float(v)) for k, v in weights.items()}
        if names is not None:
            for name in names:
                out.setdefault(str(name), 0.0)
        s = float(sum(out.values()))
        if s <= 1e-12:
            n = max(1, len(out))
            return {k: 1.0 / n for k in out}
        return {k: float(v) / s for k, v in out.items()}

    def _resolve_regime(self, regime: str | None) -> str:
        r = str(regime or "").strip().lower()
        if not r:
            return self.default_regime
        mapping = {
            "trend": "bull",
            "meanrev": "sideways",
            "stress": "crisis",
        }
        return mapping.get(r, r)

    def route(
        self,
        regime: str | None,
        strategy_names: list[str],
        quality_scores: dict[str, float] | None = None,
    ) -> RegimeRoutingResult:
        resolved = self._resolve_regime(regime)
        notes: list[str] = []
        base = self.strategy_weights_by_regime.get(resolved)
        if not isinstance(base, dict):
            base = {n: 1.0 / max(1, len(strategy_names)) for n in strategy_names}
            notes.append("fallback_uniform_strategy_weights")
        w = self._normalize(base, names=strategy_names)

        if quality_scores:
            q = {k: float(quality_scores.get(k, 0.0)) for k in strategy_names}
            q_mean = sum(q.values()) / max(1, len(q))
            q_std = math.sqrt(sum((v - q_mean) ** 2 for v in q.values()) / max(1, len(q)))
            if q_std > 1e-12:
                z = {k: (v - q_mean) / q_std for k, v in q.items()}
                tilt = {k: max(0.0, 1.0 + self.quality_tilt_strength * z[k]) for k in strategy_names}
                merged = {k: w[k] * tilt[k] for k in strategy_names}
                w = self._normalize(merged, names=strategy_names)
                notes.append("quality_tilt_applied")

        w = {k: min(self.max_strategy_weight, max(self.min_strategy_weight, v)) for k, v in w.items()}
        w = self._normalize(w, names=strategy_names)

        gross_scale = float(self.gross_scale_by_regime.get(resolved, self.gross_scale_by_regime.get("sideways", 0.85)))
        gross_scale = min(1.5, max(0.30, gross_scale))
        return RegimeRoutingResult(
            regime=resolved,
            strategy_weights=w,
            gross_scale=gross_scale,
            notes=notes,
        )


__all__ = ["MetaAgentRouter", "RegimeRoutingResult"]
