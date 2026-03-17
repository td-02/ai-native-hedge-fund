from __future__ import annotations

from collections import defaultdict, deque
import math


class SignalQualityGate:
    """Tracks OOS excess return quality and gates signal activation deterministically."""

    def __init__(self, window: int = 20, min_sharpe_lift: float = 0.15):
        self.window = max(5, int(window))
        self.min_sharpe_lift = float(min_sharpe_lift)
        self._hist: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self._enabled: dict[str, bool] = defaultdict(lambda: True)

    def update(self, signal_name: str, excess_return: float) -> None:
        name = str(signal_name)
        x = float(excess_return)
        self._hist[name].append(x)
        if len(self._hist[name]) < self.window:
            return

        vals = list(self._hist[name])
        mean_ex = sum(vals) / len(vals)
        std_ex = math.sqrt(sum((v - mean_ex) ** 2 for v in vals) / max(1, len(vals)))
        sharpe_lift = mean_ex / (std_ex + 1e-12) * math.sqrt(252)

        if mean_ex < 0.0:
            self._enabled[name] = False
        elif sharpe_lift > self.min_sharpe_lift:
            self._enabled[name] = True

    def is_enabled(self, signal_name: str) -> bool:
        return bool(self._enabled[str(signal_name)])

    def blend_weights(self, base_weights: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in dict(base_weights).items():
            out[k] = float(v) if self.is_enabled(k) else 0.0
        s = sum(max(0.0, x) for x in out.values())
        if s <= 1e-12:
            n = max(1, len(out))
            return {k: 1.0 / n for k in out}
        return {k: max(0.0, x) / s for k, x in out.items()}

    def snapshot(self) -> dict[str, dict[str, float | bool]]:
        snap: dict[str, dict[str, float | bool]] = {}
        for name, vals in self._hist.items():
            arr = list(vals)
            m = float(sum(arr) / len(arr)) if arr else 0.0
            snap[name] = {
                "enabled": bool(self._enabled[name]),
                "n": float(len(arr)),
                "rolling_mean_excess": m,
            }
        return snap


__all__ = ["SignalQualityGate"]
