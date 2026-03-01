from __future__ import annotations

from dataclasses import dataclass
import os
import pandas as pd
import requests


@dataclass
class PaperBrokerStub:
    """Free local stub."""

    def submit_target_weights(self, weights: pd.Series) -> None:
        printable = {k: round(float(v), 4) for k, v in weights.items() if abs(v) > 1e-6}
        print('Paper order targets:', printable)


@dataclass
class AlpacaPaperBroker:
    """Alpaca paper trading broker (free paper account)."""

    base_url: str = 'https://paper-api.alpaca.markets'
    min_order_notional: float = 10.0
    timeout_sec: int = 20

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
        print(f'Closed position: {symbol}')

    def _submit_market_order(self, symbol: str, qty: float, side: str) -> None:
        payload = {
            'symbol': symbol,
            'qty': f'{qty:.6f}',
            'side': side,
            'type': 'market',
            'time_in_force': 'day',
        }
        self._request('POST', '/v2/orders', json_payload=payload)
        print(f'Submitted {side}: {symbol} qty={qty:.6f}')

    def submit_target_weights(self, weights: pd.Series, latest_prices: pd.Series) -> None:
        weights = weights.fillna(0.0).astype(float)
        latest_prices = latest_prices.reindex(weights.index).astype(float)
        latest_prices = latest_prices[latest_prices > 0]
        weights = weights.reindex(latest_prices.index).fillna(0.0)

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
            deltas.append((symbol, delta_qty, delta_notional))

        # Execute sells before buys to reduce risk of buying-power rejections.
        sells = [x for x in deltas if x[1] < 0]
        buys = [x for x in deltas if x[1] > 0]
        ordered = sorted(sells, key=lambda x: x[2], reverse=True) + sorted(
            buys, key=lambda x: x[2], reverse=True
        )

        for symbol, delta_qty, _ in ordered:
            side = 'buy' if delta_qty > 0 else 'sell'
            self._submit_market_order(symbol=symbol, qty=abs(delta_qty), side=side)
