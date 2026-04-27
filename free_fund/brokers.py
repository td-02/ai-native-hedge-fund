from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .logging import get_logger
from .paper import AlpacaPaperBroker, PaperBrokerStub

logger = get_logger(__name__)


class BrokerAdapter(ABC):
    @abstractmethod
    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _target_quantities(weights: pd.Series, latest_prices: pd.Series, capital: float) -> pd.Series:
        aligned_prices = latest_prices.reindex(weights.index).astype(float)
        aligned_prices = aligned_prices.where(aligned_prices > 0.0).dropna()
        aligned_weights = weights.reindex(aligned_prices.index).fillna(0.0).astype(float)
        target = (aligned_weights * float(capital)) / aligned_prices
        return target.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


@dataclass
class ZerodhaAdapter(BrokerAdapter):
    api_key: str
    access_token: str
    capital: float = 100_000.0

    def __post_init__(self) -> None:
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except Exception as exc:
            raise RuntimeError("kiteconnect dependency is required for ZerodhaAdapter") from exc
        self.client = KiteConnect(api_key=self.api_key)
        self.client.set_access_token(self.access_token)

    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        if latest_prices is None:
            raise ValueError("latest_prices is required for ZerodhaAdapter")
        positions_resp = self.client.positions() or {}
        net_positions = positions_resp.get("net", []) if isinstance(positions_resp, dict) else []
        current_qty = {
            str(p.get("tradingsymbol", "")): float(p.get("quantity", 0.0))
            for p in net_positions
            if str(p.get("tradingsymbol", ""))
        }
        target_qty = self._target_quantities(weights, latest_prices, capital=self.capital)
        for symbol, tgt in target_qty.items():
            delta = int(round(float(tgt) - float(current_qty.get(symbol, 0.0))))
            if delta == 0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            self.client.place_order(
                variety="regular",
                exchange="NSE",
                tradingsymbol=str(symbol),
                transaction_type=side,
                quantity=abs(delta),
                product="CNC",
                order_type="MARKET",
            )
            logger.info("zerodha.order_submitted", run_id=run_id, symbol=symbol, side=side, qty=abs(delta))


@dataclass
class AngelOneAdapter(BrokerAdapter):
    api_key: str
    client_id: str
    password: str
    totp_token: str | None = None
    capital: float = 100_000.0
    symbol_token_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        try:
            from SmartApi.smartConnect import SmartConnect  # type: ignore
        except Exception as exc:
            raise RuntimeError("smartapi-python dependency is required for AngelOneAdapter") from exc
        self.client = SmartConnect(api_key=self.api_key)

    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        if latest_prices is None:
            raise ValueError("latest_prices is required for AngelOneAdapter")
        token_map = self.symbol_token_map or {}
        positions = self.client.position() or {}
        pos_data = positions.get("data", []) if isinstance(positions, dict) else []
        current_qty = {
            str(p.get("tradingsymbol", "")): float(p.get("netqty", 0.0))
            for p in pos_data
            if str(p.get("tradingsymbol", ""))
        }
        target_qty = self._target_quantities(weights, latest_prices, capital=self.capital)
        for symbol, tgt in target_qty.items():
            symbol_token = token_map.get(symbol)
            if not symbol_token:
                logger.warning("angelone.token_missing", symbol=symbol, run_id=run_id)
                continue
            delta = int(round(float(tgt) - float(current_qty.get(symbol, 0.0))))
            if delta == 0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            order = {
                "variety": "NORMAL",
                "tradingsymbol": str(symbol),
                "symboltoken": str(symbol_token),
                "transactiontype": side,
                "exchange": "NSE",
                "ordertype": "MARKET",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "quantity": abs(delta),
            }
            self.client.placeOrder(order)
            logger.info("angelone.order_submitted", run_id=run_id, symbol=symbol, side=side, qty=abs(delta))


@dataclass
class BrokerRouter:
    clients: list[BrokerAdapter]

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
                logger.warning(
                    "broker.submit_failed",
                    broker=type(client).__name__,
                    error=str(exc),
                    run_id=run_id,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("No brokers configured")


def _adapter_from_name(name: str, ecfg: dict[str, Any]) -> BrokerAdapter | None:
    if name == "alpaca_paper":
        return AlpacaPaperBroker(
            base_url=ecfg.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
            min_order_notional=float(ecfg.get("min_order_notional", 10.0)),
            order_style=str(ecfg.get("order_style", "twap")),
            twap_slices=int(ecfg.get("twap_slices", 3)),
            max_participation_adv=float(ecfg.get("max_participation_adv", 0.10)),
            adv_notional_default=float(ecfg.get("adv_notional_default", 2_000_000.0)),
            market_mode=str(ecfg.get("market_mode", "us")),
        )
    if name == "zerodha_kite":
        key = str(ecfg.get("kite_api_key", ""))
        token = str(ecfg.get("kite_access_token", ""))
        if key and token:
            return ZerodhaAdapter(
                api_key=key,
                access_token=token,
                capital=float(ecfg.get("target_capital", 100_000.0)),
            )
        logger.warning("broker.zerodha_missing_credentials")
        return None
    if name == "angel_one":
        key = str(ecfg.get("angel_api_key", ""))
        cid = str(ecfg.get("angel_client_id", ""))
        pwd = str(ecfg.get("angel_password", ""))
        if key and cid and pwd:
            return AngelOneAdapter(
                api_key=key,
                client_id=cid,
                password=pwd,
                totp_token=str(ecfg.get("angel_totp_token", "")) or None,
                capital=float(ecfg.get("target_capital", 100_000.0)),
                symbol_token_map=dict(ecfg.get("angel_symbol_token_map", {}) or {}),
            )
        logger.warning("broker.angelone_missing_credentials")
        return None
    if name == "stub":
        return PaperBrokerStub()
    return None


def build_broker_router(cfg: dict) -> BrokerRouter:
    ecfg = cfg.get("execution", {})
    primary = str(ecfg.get("primary_broker", ecfg.get("broker", "stub")))
    backups = list(ecfg.get("backup_brokers", []))
    order = [primary] + backups

    clients: list[BrokerAdapter] = []
    for name in order:
        adapter = _adapter_from_name(name, ecfg)
        if adapter is not None:
            clients.append(adapter)

    if not clients:
        clients = [PaperBrokerStub()]
    return BrokerRouter(clients=clients)
