from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import os
from time import perf_counter
import pandas as pd
import requests

from .logging import get_logger
from .metrics import order_latency_seconds, orders_total

logger = get_logger(__name__)


@dataclass
class PaperBrokerStub:
    """Free local stub."""

    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series | None = None,
        run_id: str | None = None,
    ) -> None:
        printable = {k: round(float(v), 4) for k, v in weights.items() if abs(v) > 1e-6}
        logger.info("broker.stub_targets", run_id=run_id, targets=printable)


@dataclass
class AlpacaPaperBroker:
    """Alpaca paper trading broker (free paper account)."""

    base_url: str = 'https://paper-api.alpaca.markets'
    min_order_notional: float = 10.0
    timeout_sec: int = 20
    order_style: str = "twap"
    twap_slices: int = 3
    max_participation_adv: float = 0.10
    adv_notional_default: float = 2_000_000.0
    market_mode: str = "us"

    def __post_init__(self) -> None:
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('APCA_API_SECRET_KEY')
        env_base = os.getenv('APCA_PAPER_BASE_URL')
        if env_base:
            self.base_url = env_base.rstrip('/')

        if not self.api_key or not self.api_secret:
            raise ValueError(
                'Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.'
            )

    def _headers(self) -> dict[str, str]:
        return {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json',
        }

    def _request(self, method: str, path: str, json_payload: dict | None = None) -> dict | list:
        url = f'{self.base_url}{path}'
        resp = requests.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=json_payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        if not resp.text.strip():
            return {}
        return resp.json()

    def _account_equity(self) -> float:
        account = self._request('GET', '/v2/account')
        return float(account.get('portfolio_value', account.get('equity', 0.0)))

    def _positions(self) -> dict[str, float]:
        raw = self._request('GET', '/v2/positions')
        out: dict[str, float] = {}
        for p in raw:
            out[p['symbol']] = float(p['qty'])
        return out

    def _close_symbol(self, symbol: str) -> None:
        self._request('DELETE', f'/v2/positions/{symbol}')
        logger.info("broker.closed_position", symbol=symbol)

    def _submit_market_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> None:
        start = perf_counter()
        payload = {
            'symbol': symbol,
            'qty': f'{qty:.6f}',
            'side': side,
            'type': 'market',
            'time_in_force': 'day',
            'client_order_id': client_order_id[:48],
        }
        self._request('POST', '/v2/orders', json_payload=payload)
        elapsed = perf_counter() - start
        orders_total.labels(broker="alpaca_paper", status="submitted").inc()
        order_latency_seconds.labels(broker="alpaca_paper").observe(elapsed)
        logger.info("broker.order_submitted", symbol=symbol, side=side, qty=qty, order_id=client_order_id[:16], latency=elapsed)

    @staticmethod
    def _is_us_holiday(today: date) -> bool:
        # Minimal holiday gate for safety; can be replaced with exchange calendars.
        fixed = {(1, 1), (7, 4), (12, 25)}
        return (today.month, today.day) in fixed

    @staticmethod
    def _market_is_session_open(market_mode: str) -> bool:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        if weekday >= 5:
            return False
        if market_mode == "india":
            # NSE approx 09:15-15:30 IST => 03:45-10:00 UTC.
            return (now.hour, now.minute) >= (3, 45) and (now.hour, now.minute) <= (10, 0)
        # US regular approx 09:30-16:00 ET. Rough UTC window without DST handling: 14:30-21:00.
        return (now.hour, now.minute) >= (14, 30) and (now.hour, now.minute) <= (21, 0)

    @staticmethod
    def _client_order_id(run_id: str, symbol: str, side: str, slice_idx: int) -> str:
        raw = f"{run_id}:{symbol}:{side}:{slice_idx}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _slice_qty(self, total_qty: float) -> list[float]:
        n = max(1, int(self.twap_slices))
        base = total_qty / n
        out = [base] * n
        out[-1] = total_qty - sum(out[:-1])
        return [abs(q) for q in out if abs(q) > 1e-9]

    def _impact_allowed(self, delta_notional: float) -> bool:
        return delta_notional <= (self.max_participation_adv * self.adv_notional_default)

    def submit_target_weights(
        self,
        weights: pd.Series,
        latest_prices: pd.Series,
        run_id: str | None = None,
    ) -> None:
        weights = weights.fillna(0.0).astype(float)
        latest_prices = latest_prices.reindex(weights.index).astype(float)
        latest_prices = latest_prices[latest_prices > 0]
        weights = weights.reindex(latest_prices.index).fillna(0.0)
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        if self.market_mode == "us":
            today = datetime.now(timezone.utc).date()
            if self._is_us_holiday(today):
                logger.info("broker.execution_skipped", reason="us_holiday")
                return
        if not self._market_is_session_open(self.market_mode):
            logger.info("broker.execution_skipped", reason="market_closed")
            return

        equity = self._account_equity()
        current_positions = self._positions()
        target_symbols = set(weights.index)

        for symbol in current_positions:
            if symbol not in target_symbols:
                self._close_symbol(symbol)

        deltas: list[tuple[str, float, float]] = []
        for symbol in weights.index:
            px = float(latest_prices[symbol])
            target_qty = float(weights[symbol] * equity / px)
            current_qty = float(current_positions.get(symbol, 0.0))
            delta_qty = target_qty - current_qty
            delta_notional = abs(delta_qty) * px
            if delta_notional < self.min_order_notional:
                continue
            if not self._impact_allowed(delta_notional):
                logger.info("broker.order_skipped", symbol=symbol, reason="impact_adv_threshold")
                continue
            if self.market_mode == "india" and delta_qty < 0:
                # Conservative compliance mode for India.
                logger.info("broker.order_skipped", symbol=symbol, reason="india_short_blocked")
                continue
            deltas.append((symbol, delta_qty, delta_notional))

        # Execute sells before buys to reduce risk of buying-power rejections.
        sells = [x for x in deltas if x[1] < 0]
        buys = [x for x in deltas if x[1] > 0]
        ordered = sorted(sells, key=lambda x: x[2], reverse=True) + sorted(
            buys, key=lambda x: x[2], reverse=True
        )

        for symbol, delta_qty, _ in ordered:
            side = 'buy' if delta_qty > 0 else 'sell'
            qty_total = abs(delta_qty)
            if self.order_style in {"twap", "vwap"} and qty_total > 0:
                for i, qty_slice in enumerate(self._slice_qty(qty_total)):
                    oid = self._client_order_id(run_id, symbol, side, i)
                    self._submit_market_order(symbol=symbol, qty=qty_slice, side=side, client_order_id=oid)
            else:
                oid = self._client_order_id(run_id, symbol, side, 0)
                self._submit_market_order(symbol=symbol, qty=qty_total, side=side, client_order_id=oid)
