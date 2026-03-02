from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .paper import AlpacaPaperBroker, PaperBrokerStub


class BrokerClient(Protocol):
    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None: ...


@dataclass
class ZerodhaKiteBroker:
    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        raise NotImplementedError("Zerodha Kite integration hook. Add credentials + SDK wiring.")


@dataclass
class UpstoxBroker:
    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        raise NotImplementedError("Upstox integration hook. Add credentials + SDK wiring.")


@dataclass
class AngelOneBroker:
    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        raise NotImplementedError("Angel One integration hook. Add credentials + SDK wiring.")


@dataclass
class BrokerRouter:
    clients: list[BrokerClient]

    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        for client in self.clients:
            try:
                client.submit_target_weights(weights, latest_prices, run_id=run_id)
                return type(client).__name__
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("No brokers configured")


def build_broker_router(cfg: dict) -> BrokerRouter:
    ecfg = cfg.get("execution", {})
    primary = str(ecfg.get("primary_broker", ecfg.get("broker", "stub")))
    backups = list(ecfg.get("backup_brokers", []))
    order = [primary] + backups

    clients: list[BrokerClient] = []
    for name in order:
        if name == "alpaca_paper":
            clients.append(
                AlpacaPaperBroker(
                    base_url=ecfg.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
                    min_order_notional=float(ecfg.get("min_order_notional", 10.0)),
                    order_style=str(ecfg.get("order_style", "twap")),
                    twap_slices=int(ecfg.get("twap_slices", 3)),
                    max_participation_adv=float(ecfg.get("max_participation_adv", 0.10)),
                    adv_notional_default=float(ecfg.get("adv_notional_default", 2_000_000.0)),
                    market_mode=str(ecfg.get("market_mode", "us")),
                )
            )
        elif name == "zerodha_kite":
            clients.append(ZerodhaKiteBroker())
        elif name == "upstox":
            clients.append(UpstoxBroker())
        elif name == "angel_one":
            clients.append(AngelOneBroker())
        elif name == "stub":
            clients.append(PaperBrokerStub())

    if not clients:
        clients = [PaperBrokerStub()]
    return BrokerRouter(clients=clients)
