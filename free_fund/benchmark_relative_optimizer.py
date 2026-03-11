from __future__ import annotations

import numpy as np
import pandas as pd


def _project_constraints(
    w: pd.Series,
    max_weight: float,
    gross_limit: float,
    net_limit: float,
) -> pd.Series:
    out = w.clip(lower=-max_weight, upper=max_weight).copy()
    net = float(out.sum())
    if abs(net) > net_limit and abs(net) > 1e-12:
        out = out - (net - np.sign(net) * net_limit) / len(out)
    gross = float(out.abs().sum())
    if gross > gross_limit and gross > 1e-12:
        out = out / gross * gross_limit
    return out


def optimize_benchmark_relative_weights(
    base_weights: pd.Series,
    alpha_views: dict[str, dict[str, float | str]],
    returns_window: pd.DataFrame,
    benchmark_returns: pd.Series,
    max_weight: float = 0.30,
    gross_limit: float = 1.0,
    net_limit: float = 0.30,
    max_turnover: float = 0.25,
    alpha_tilt_strength: float = 0.35,
    bab_tilt_strength: float = 0.15,
    uncertainty_penalty: float = 0.50,
    tracking_error_penalty: float = 0.60,
) -> tuple[pd.Series, dict[str, float]]:
    symbols = list(base_weights.index)
    ret = returns_window.reindex(columns=symbols).fillna(0.0)
    b = benchmark_returns.reindex(ret.index).fillna(0.0)

    raw_alpha = pd.Series({s: float(alpha_views.get(s, {}).get("expected_return", 0.0)) for s in symbols}, dtype=float)
    raw_unc = pd.Series({s: float(alpha_views.get(s, {}).get("uncertainty", 0.05)) for s in symbols}, dtype=float)
    raw_conf = pd.Series({s: float(alpha_views.get(s, {}).get("confidence", 0.0)) for s in symbols}, dtype=float)

    # Confidence and uncertainty adjusted alpha score.
    adj_alpha = raw_alpha * raw_conf / (1.0 + uncertainty_penalty * raw_unc)
    if float(adj_alpha.abs().sum()) <= 1e-12:
        return base_weights.copy(), {
            "active_alpha": 0.0,
            "tracking_error": 0.0,
            "turnover": 0.0,
            "objective": 0.0,
        }

    z = (adj_alpha - float(adj_alpha.mean()))
    std = float(z.std(ddof=0))
    if std > 1e-12:
        z = z / std
    alpha_tilt = z.clip(-2.5, 2.5)

    # BAB overlay: overweight lower-beta assets, underweight high-beta assets.
    beta = pd.Series(0.0, index=symbols, dtype=float)
    if len(ret) >= 40:
        for s in symbols:
            rs = ret[s].astype(float)
            rb = b.astype(float)
            aligned = pd.concat([rs, rb], axis=1).dropna()
            if len(aligned) < 30:
                beta.loc[s] = 1.0
                continue
            var_b = float(np.var(aligned.iloc[:, 1], ddof=0))
            if var_b <= 1e-12:
                beta.loc[s] = 1.0
                continue
            cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1])
            beta.loc[s] = cov / var_b
    if float(beta.std(ddof=0)) > 1e-12:
        beta_z = (beta - float(beta.mean())) / float(beta.std(ddof=0))
    else:
        beta_z = pd.Series(0.0, index=symbols)
    bab_tilt = (-beta_z).clip(-2.5, 2.5)

    tilt = alpha_tilt_strength * alpha_tilt + bab_tilt_strength * bab_tilt
    target = base_weights + tilt / max(1.0, float(tilt.abs().sum()))
    target = _project_constraints(target, max_weight=max_weight, gross_limit=gross_limit, net_limit=net_limit)

    # Turnover cap.
    delta = target - base_weights
    turnover = float(delta.abs().sum())
    if turnover > max_turnover and turnover > 1e-12:
        target = base_weights + delta * (max_turnover / turnover)
        target = _project_constraints(target, max_weight=max_weight, gross_limit=gross_limit, net_limit=net_limit)
        turnover = float((target - base_weights).abs().sum())

    # Risk diagnostics.
    active_alpha = float((target * raw_alpha).sum() - (base_weights * raw_alpha).sum())
    if len(ret) >= 20:
        p = (ret * target).sum(axis=1)
        pb = (ret * base_weights).sum(axis=1)
        active = p - b
        active_base = pb - b
        te = float(active.std(ddof=0) * np.sqrt(252))
        te_base = float(active_base.std(ddof=0) * np.sqrt(252))
    else:
        te = 0.0
        te_base = 0.0

    objective = active_alpha - tracking_error_penalty * max(0.0, te - te_base)
    if objective <= 0:
        # Deterministic no-harm fallback: do not override baseline when risk-adjusted edge is negative.
        return base_weights.copy(), {
            "active_alpha": active_alpha,
            "tracking_error": te_base,
            "tracking_error_base": te_base,
            "turnover": 0.0,
            "objective": float(objective),
        }
    diag = {
        "active_alpha": active_alpha,
        "tracking_error": te,
        "tracking_error_base": te_base,
        "turnover": turnover,
        "objective": float(objective),
        "avg_beta": float(beta.mean()) if len(beta) else 0.0,
    }
    return target.fillna(0.0), diag


__all__ = ["optimize_benchmark_relative_weights"]
