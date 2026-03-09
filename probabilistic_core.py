from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def ledoit_wolf_shrinkage(returns_df: pd.DataFrame) -> np.ndarray:
    x = returns_df.dropna().to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        n = returns_df.shape[1]
        return np.eye(max(1, n), dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    t, n = x.shape
    sample = (x.T @ x) / max(1, t - 1)
    mu = float(np.trace(sample) / n)
    target = mu * np.eye(n)
    diff = sample - target
    denom = float(np.sum(diff * diff))
    if denom <= 1e-12:
        return target
    y = x * x
    phi_mat = (y.T @ y) / max(1, t - 1) - 2.0 * (x.T @ x) * sample / max(1, t - 1) + sample * sample
    phi = float(np.sum(phi_mat))
    kappa = max(0.0, min(1.0, phi / (denom * t)))
    shrunk = kappa * target + (1.0 - kappa) * sample
    return (shrunk + shrunk.T) / 2.0


def rolling_realized_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    returns = prices.pct_change()
    return returns.rolling(window).std(ddof=0) * math.sqrt(252.0)


def var_cvar(returns: np.ndarray, confidence: float = 0.95) -> dict:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"var": 0.0, "cvar": 0.0, "confidence": confidence}
    q = float(np.quantile(arr, 1.0 - confidence))
    tail = arr[arr <= q]
    cvar = float(tail.mean()) if tail.size > 0 else q
    return {"var": q, "cvar": cvar, "confidence": confidence}


@dataclass
class BlackScholes:
    S: float
    K: float
    T: float
    r: float
    sigma: float

    def _d1_d2(self, sigma: float | None = None) -> tuple[float, float]:
        s = max(1e-12, float(self.S))
        k = max(1e-12, float(self.K))
        t = max(1e-12, float(self.T))
        v = max(1e-12, float(self.sigma if sigma is None else sigma))
        d1 = (math.log(s / k) + (self.r + 0.5 * v * v) * t) / (v * math.sqrt(t))
        d2 = d1 - v * math.sqrt(t)
        return d1, d2

    def price(self, option_type: str = "call") -> float:
        d1, d2 = self._d1_d2()
        disc_k = self.K * math.exp(-self.r * self.T)
        if option_type.lower() == "put":
            return float(disc_k * norm.cdf(-d2) - self.S * norm.cdf(-d1))
        return float(self.S * norm.cdf(d1) - disc_k * norm.cdf(d2))

    def greeks(self) -> dict:
        d1, d2 = self._d1_d2()
        t = max(1e-12, float(self.T))
        pdf_d1 = norm.pdf(d1)
        delta = float(norm.cdf(d1))
        gamma = float(pdf_d1 / (self.S * max(1e-12, self.sigma) * math.sqrt(t)))
        vega = float(self.S * pdf_d1 * math.sqrt(t))
        theta = float(
            -(self.S * pdf_d1 * max(1e-12, self.sigma)) / (2.0 * math.sqrt(t))
            - self.r * self.K * math.exp(-self.r * t) * norm.cdf(d2)
        )
        rho = float(self.K * t * math.exp(-self.r * t) * norm.cdf(d2))
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def implied_vol(self, market_price: float, option_type: str = "call") -> float:
        target = float(market_price)
        v = max(1e-4, float(self.sigma))
        for _ in range(100):
            d1, _ = self._d1_d2(sigma=v)
            bs = BlackScholes(self.S, self.K, self.T, self.r, v)
            price = bs.price(option_type=option_type)
            vega = max(1e-8, float(bs.greeks()["vega"]))
            diff = price - target
            if abs(diff) < 1e-8:
                return float(v)
            v = v - diff / vega
            if not np.isfinite(v) or v <= 0:
                return float("nan")
        return float("nan")

    @staticmethod
    def vol_surface(options_chain_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        now = pd.Timestamp.utcnow()
        for _, r in options_chain_df.iterrows():
            try:
                strike = float(r["strike"])
                expiry = pd.Timestamp(r["expiry"])
                t = max(1 / 365.0, float((expiry - now).total_seconds()) / (365.0 * 24.0 * 3600.0))
                px = float(r["market_price"])
                opt_type = str(r["option_type"])
                s = float(r.get("spot", strike))
                rr = float(r.get("r", 0.05))
                bs = BlackScholes(S=s, K=strike, T=t, r=rr, sigma=0.2)
                iv = bs.implied_vol(px, option_type=opt_type)
                if np.isfinite(iv):
                    rec = dict(r)
                    rec["implied_vol"] = float(iv)
                    rows.append(rec)
            except Exception:
                continue
        return pd.DataFrame(rows)


@dataclass
class HestonModel:
    S: float
    K: float
    T: float
    r: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def _simulate_paths(self, n_paths: int = 10000, steps_per_year: int = 252) -> np.ndarray:
        steps = max(1, int(self.T * steps_per_year))
        dt = max(1e-8, self.T / steps)
        half = max(1, n_paths // 2)
        s = np.full((2 * half,), float(self.S), dtype=float)
        v = np.full((2 * half,), max(1e-8, float(self.v0)), dtype=float)
        for _ in range(steps):
            z1 = np.random.normal(size=half)
            z2 = np.random.normal(size=half)
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
            w1 = z1
            w2 = self.rho * z1 + math.sqrt(max(1e-12, 1.0 - self.rho * self.rho)) * z2
            v = np.maximum(1e-12, v + self.kappa * (self.theta - v) * dt + self.sigma_v * np.sqrt(v * dt) * w2)
            s = s * np.exp((self.r - 0.5 * v) * dt + np.sqrt(v * dt) * w1)
        return s

    def price(self, option_type: str = "call") -> float:
        terminal = self._simulate_paths(n_paths=10000, steps_per_year=252)
        if option_type.lower() == "put":
            payoff = np.maximum(self.K - terminal, 0.0)
        else:
            payoff = np.maximum(terminal - self.K, 0.0)
        return float(np.exp(-self.r * self.T) * payoff.mean())

    def calibrate_from_surface(self, vol_surface_df: pd.DataFrame) -> dict:
        if vol_surface_df.empty:
            return {
                "kappa": self.kappa,
                "theta": self.theta,
                "sigma_v": self.sigma_v,
                "rho": self.rho,
                "v0": self.v0,
                "calibration_error": float("nan"),
            }

        required = {"strike", "expiry", "market_price", "option_type"}
        if not required.issubset(set(vol_surface_df.columns)):
            return {
                "kappa": self.kappa,
                "theta": self.theta,
                "sigma_v": self.sigma_v,
                "rho": self.rho,
                "v0": self.v0,
                "calibration_error": float("nan"),
            }

        now = pd.Timestamp.utcnow()
        rows = []
        for _, r in vol_surface_df.iterrows():
            try:
                t = max(1 / 365.0, float((pd.Timestamp(r["expiry"]) - now).total_seconds()) / (365.0 * 24.0 * 3600.0))
                rows.append(
                    {
                        "K": float(r["strike"]),
                        "T": t,
                        "price": float(r["market_price"]),
                        "option_type": str(r["option_type"]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return {
                "kappa": self.kappa,
                "theta": self.theta,
                "sigma_v": self.sigma_v,
                "rho": self.rho,
                "v0": self.v0,
                "calibration_error": float("nan"),
            }

        def objective(x: np.ndarray) -> float:
            kappa, theta, sigma_v, rho, v0 = x
            if theta <= 0 or sigma_v <= 0 or v0 <= 0 or abs(rho) >= 0.999:
                return 1e6
            err = 0.0
            for item in rows[:40]:
                model = HestonModel(
                    S=self.S,
                    K=item["K"],
                    T=item["T"],
                    r=self.r,
                    v0=v0,
                    kappa=kappa,
                    theta=theta,
                    sigma_v=sigma_v,
                    rho=rho,
                )
                mp = model.price(option_type=item["option_type"])
                err += (mp - item["price"]) ** 2
            return float(err / max(1, min(40, len(rows))))

        x0 = np.array([self.kappa, self.theta, self.sigma_v, self.rho, self.v0], dtype=float)
        bounds = [(1e-3, 10.0), (1e-4, 2.0), (1e-3, 5.0), (-0.99, 0.99), (1e-4, 2.0)]
        res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        x = res.x if res.success else x0
        return {
            "kappa": float(x[0]),
            "theta": float(x[1]),
            "sigma_v": float(x[2]),
            "rho": float(x[3]),
            "v0": float(x[4]),
            "calibration_error": float(objective(x)),
        }


class KalmanSignalTracker:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.reset()

    def update(self, measurement: float) -> dict:
        z = float(measurement)
        A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = np.eye(2, dtype=float) * self.process_noise
        R = np.array([[self.measurement_noise]], dtype=float)
        x_pred = A @ self.x
        p_pred = A @ self.P @ A.T + Q
        y = np.array([[z]], dtype=float) - H @ x_pred
        S = H @ p_pred @ H.T + R
        K = p_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ H) @ p_pred
        return {
            "filtered_signal": float(self.x[0, 0]),
            "signal_velocity": float(self.x[1, 0]),
            "uncertainty": float(max(0.0, self.P[0, 0])),
        }

    def reset(self) -> None:
        self.x = np.zeros((2, 1), dtype=float)
        self.P = np.eye(2, dtype=float)


class BayesianSignalAggregator:
    def __init__(self, signal_names: list):
        self.signal_names = list(signal_names)
        self.values: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def update_signal(self, name: str, value: float, std: float) -> None:
        self.values[name] = float(value)
        self.stds[name] = float(max(1e-8, std))

    def get_combined_signal(self) -> dict:
        keys = [k for k in self.values.keys() if k in self.stds]
        if not keys:
            return {"combined": 0.0, "uncertainty": 1.0, "weights": {}, "n_signals": 0}
        inv_vars = np.array([1.0 / (self.stds[k] ** 2) for k in keys], dtype=float)
        w = inv_vars / max(1e-12, float(inv_vars.sum()))
        vals = np.array([self.values[k] for k in keys], dtype=float)
        combined = float(np.dot(w, vals))
        uncertainty = float(math.sqrt(1.0 / max(1e-12, float(inv_vars.sum()))))
        weights = {k: float(wi) for k, wi in zip(keys, w)}
        return {"combined": combined, "uncertainty": uncertainty, "weights": weights, "n_signals": len(keys)}


class RegimeHMM:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = int(n_regimes)
        self.pi = np.full(self.n_regimes, 1.0 / self.n_regimes, dtype=float)
        self.A = np.full((self.n_regimes, self.n_regimes), 1.0 / self.n_regimes, dtype=float)
        self.means = np.linspace(-0.01, 0.01, self.n_regimes)
        self.vars = np.full(self.n_regimes, 0.02**2, dtype=float)

    def _emission(self, x: np.ndarray) -> np.ndarray:
        b = np.zeros((len(x), self.n_regimes), dtype=float)
        for k in range(self.n_regimes):
            var = max(1e-12, float(self.vars[k]))
            b[:, k] = (1.0 / math.sqrt(2.0 * math.pi * var)) * np.exp(-0.5 * ((x - self.means[k]) ** 2) / var)
        return np.clip(b, 1e-300, None)

    def _forward_backward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        T = len(x)
        B = self._emission(x)
        alpha = np.zeros((T, self.n_regimes), dtype=float)
        scales = np.zeros(T, dtype=float)
        alpha[0] = self.pi * B[0]
        scales[0] = max(1e-300, float(alpha[0].sum()))
        alpha[0] /= scales[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scales[t] = max(1e-300, float(alpha[t].sum()))
            alpha[t] /= scales[t]

        beta = np.zeros((T, self.n_regimes), dtype=float)
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / max(1e-300, scales[t + 1])
        loglik = float(np.sum(np.log(scales + 1e-300)))
        return alpha, beta, loglik

    def fit(self, returns: np.ndarray) -> None:
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 20:
            return
        prev_loglik = -1e99
        for _ in range(100):
            alpha, beta, loglik = self._forward_backward(x)
            gamma = alpha * beta
            gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)
            xi = np.zeros((len(x) - 1, self.n_regimes, self.n_regimes), dtype=float)
            B = self._emission(x)
            for t in range(len(x) - 1):
                numer = alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None, :]
                denom = max(1e-300, float(numer.sum()))
                xi[t] = numer / denom

            self.pi = np.clip(gamma[0], 1e-12, None)
            self.pi /= self.pi.sum()
            A_num = xi.sum(axis=0)
            A_den = np.clip(gamma[:-1].sum(axis=0), 1e-12, None)
            self.A = A_num / A_den[:, None]
            self.A = np.clip(self.A, 1e-12, None)
            self.A /= self.A.sum(axis=1, keepdims=True)
            for k in range(self.n_regimes):
                w = gamma[:, k]
                denom = max(1e-12, float(w.sum()))
                mu = float(np.sum(w * x) / denom)
                var = float(np.sum(w * (x - mu) ** 2) / denom)
                self.means[k] = mu
                self.vars[k] = max(1e-8, var)
            if abs(loglik - prev_loglik) < 1e-4:
                break
            prev_loglik = loglik

    def predict_proba(self, returns: np.ndarray) -> dict:
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return {"bull": 0.25, "bear": 0.25, "crisis": 0.25, "sideways": 0.25, "most_likely": "sideways"}
        alpha, beta, _ = self._forward_backward(x)
        p = alpha[-1] * beta[-1]
        p = p / max(1e-12, float(p.sum()))
        labels = ["bull", "bear", "crisis", "sideways"][: self.n_regimes]
        out = {labels[i]: float(p[i]) for i in range(len(labels))}
        out["most_likely"] = labels[int(np.argmax(p))]
        return out

    def viterbi(self, returns: np.ndarray) -> np.ndarray:
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([], dtype=int)
        B = self._emission(x)
        T = len(x)
        logA = np.log(np.clip(self.A, 1e-300, None))
        logB = np.log(np.clip(B, 1e-300, None))
        logpi = np.log(np.clip(self.pi, 1e-300, None))
        dp = np.zeros((T, self.n_regimes), dtype=float)
        ptr = np.zeros((T, self.n_regimes), dtype=int)
        dp[0] = logpi + logB[0]
        for t in range(1, T):
            for j in range(self.n_regimes):
                vals = dp[t - 1] + logA[:, j]
                ptr[t, j] = int(np.argmax(vals))
                dp[t, j] = float(np.max(vals)) + logB[t, j]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]
        return states


class BayesianPortfolioOptimizer:
    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        self.risk_aversion = float(risk_aversion)
        self.tau = float(tau)
        self.mu_prior: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.assets: list[str] = []
        self.views: list[dict[str, Any]] = []

    def set_market_prior(self, returns_df: pd.DataFrame, market_caps: dict = None) -> None:
        clean = returns_df.dropna(how="any")
        self.assets = list(clean.columns)
        n = len(self.assets)
        if n == 0:
            self.mu_prior = np.array([], dtype=float)
            self.sigma = np.zeros((0, 0), dtype=float)
            return
        self.sigma = ledoit_wolf_shrinkage(clean)
        if market_caps:
            w = np.array([float(market_caps.get(a, 1.0)) for a in self.assets], dtype=float)
            w = w / max(1e-12, float(w.sum()))
        else:
            w = np.full(n, 1.0 / n, dtype=float)
        self.mu_prior = self.risk_aversion * (self.sigma @ w)

    def add_llm_view(self, view: dict) -> None:
        self.views.append(dict(view))

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mu_prior is None or self.sigma is None or self.sigma.size == 0:
            return np.array([], dtype=float), np.zeros((0, 0), dtype=float)
        n = len(self.assets)
        tau_sigma = self.tau * self.sigma
        if not self.views:
            return self.mu_prior.copy(), self.sigma.copy()
        P_rows = []
        Q = []
        O_diag = []
        for v in self.views:
            vec = np.zeros(n, dtype=float)
            assets = v.get("assets", [])
            if not assets:
                continue
            coeff = 1.0 / max(1, len(assets))
            for a in assets:
                if a in self.assets:
                    vec[self.assets.index(a)] = coeff
            conf = float(v.get("confidence", 0.5))
            conf = min(0.999, max(1e-3, conf))
            outperf = float(v.get("outperformance", 0.0))
            P_rows.append(vec)
            Q.append(outperf)
            O_diag.append(max(1e-6, (1.0 - conf) ** 2))
        if not P_rows:
            return self.mu_prior.copy(), self.sigma.copy()

        P = np.vstack(P_rows)
        Qv = np.array(Q, dtype=float)
        Omega = np.diag(np.array(O_diag, dtype=float))
        inv_tau_sigma = np.linalg.pinv(tau_sigma)
        inv_omega = np.linalg.pinv(Omega)
        middle = np.linalg.pinv(inv_tau_sigma + P.T @ inv_omega @ P)
        mu_post = middle @ (inv_tau_sigma @ self.mu_prior + P.T @ inv_omega @ Qv)
        sigma_post = self.sigma + middle
        return mu_post, sigma_post

    def optimize(self) -> dict:
        mu, sigma = self._posterior()
        n = len(mu)
        if n == 0:
            return {"weights": {}, "posterior_returns": {}, "posterior_cov": np.zeros((0, 0))}

        def obj(w: np.ndarray) -> float:
            ret = float(mu @ w)
            var = float(w.T @ sigma @ w)
            return -(ret - 0.5 * self.risk_aversion * var)

        x0 = np.full(n, 1.0 / n, dtype=float)
        bounds = [(0.0, 0.4)] * n
        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        w = res.x if res.success else x0
        return {
            "weights": {a: float(w[i]) for i, a in enumerate(self.assets)},
            "posterior_returns": {a: float(mu[i]) for i, a in enumerate(self.assets)},
            "posterior_cov": sigma,
        }

    def optimize_with_uncertainty(self, n_samples: int = 500) -> dict:
        mu, sigma = self._posterior()
        n = len(mu)
        if n == 0:
            return {"mean_weights": {}, "std_weights": {}, "p5_weights": {}, "p95_weights": {}}
        ws = []
        base_views = list(self.views)
        for _ in range(max(1, int(n_samples))):
            perturbed = []
            for v in base_views:
                vv = dict(v)
                conf = float(vv.get("confidence", 0.5))
                std = max(1e-6, 1.0 - min(0.999, max(1e-3, conf)))
                vv["outperformance"] = float(vv.get("outperformance", 0.0)) + np.random.normal(0.0, std * 0.01)
                perturbed.append(vv)
            self.views = perturbed
            out = self.optimize()
            ws.append([out["weights"].get(a, 0.0) for a in self.assets])
        self.views = base_views
        arr = np.asarray(ws, dtype=float)
        mean_w = np.mean(arr, axis=0)
        std_w = np.std(arr, axis=0, ddof=0)
        p5 = np.quantile(arr, 0.05, axis=0)
        p95 = np.quantile(arr, 0.95, axis=0)
        return {
            "mean_weights": {a: float(mean_w[i]) for i, a in enumerate(self.assets)},
            "std_weights": {a: float(std_w[i]) for i, a in enumerate(self.assets)},
            "p5_weights": {a: float(p5[i]) for i, a in enumerate(self.assets)},
            "p95_weights": {a: float(p95[i]) for i, a in enumerate(self.assets)},
        }

