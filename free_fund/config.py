from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class SystemConfig(BaseModel):
    output_dir: str = "outputs"


class PortfolioConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "TLT", "GLD"])
    start_date: str = "2020-01-01"
    end_date: str | None = None
    lookback_days: int = 126
    max_weight: float = 0.30
    gross_limit: float = 1.0
    rebalance_every_n_days: int = 21


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="allow",
    )

    system: SystemConfig = Field(default_factory=SystemConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    agent: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] = Field(default_factory=dict)
    risk_hard_limits: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    resilience: dict[str, Any] = Field(default_factory=dict)
    health: dict[str, Any] = Field(default_factory=dict)
    alerts: dict[str, Any] = Field(default_factory=dict)
    tracing: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    learning: dict[str, Any] = Field(default_factory=dict)
    alpha_pipeline: dict[str, Any] = Field(default_factory=dict)
    arbitrage: dict[str, Any] = Field(default_factory=dict)
    microstructure: dict[str, Any] = Field(default_factory=dict)
    research_council: dict[str, Any] = Field(default_factory=dict)
    backtest: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] = Field(default_factory=dict)
    fund_manager: dict[str, Any] = Field(default_factory=dict)
    execution_controls: dict[str, Any] = Field(default_factory=dict)
    database: dict[str, Any] = Field(default_factory=dict)
    redis: dict[str, Any] = Field(default_factory=dict)
    celery: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Keep YAML/input values but let environment variables win.
        return env_settings, dotenv_settings, init_settings, file_secret_settings


def load_settings(path: str | Path = "configs/default.yaml") -> AppSettings:
    cfg_path = Path(path)
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    return AppSettings.model_validate(raw)


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Backward-compatible loader returning dicts for legacy callsites.
    Environment variables override nested values using `__` delimiter.
    Example: `PORTFOLIO__LOOKBACK_DAYS=252`.
    """
    settings = load_settings(path)
    return settings.model_dump(mode="python")

