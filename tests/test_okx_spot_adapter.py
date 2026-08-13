"""Tests unitaires de la frontière OKX, tous avec transport simulé."""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from okx_spot_adapter import BUY, SELL, OkxApiError, OkxSpotAdapter


INST = "ANY-QUOTE"


def envelope(data, code="0", msg=""):
    return {"code": code, "msg": msg, "data": data}


def order(*, order_id="1", client_id="opaque-id", side="buy", px="1.23", sz="4.5", acc="0", state="live", ord_type="post_only"):
    return {"ordId": order_id, "clOrdId": client_id, "instId": INST, "side": side, "px": px, "sz": sz, "accFillSz": acc, "state": state, "ordType": ord_type, "cTime": "1", "uTime": "2"}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, path, params, body, headers):
        self.calls.append((method, path, params, body, headers))
        return self.responses.pop(0)


def adapter(transport, websocket_factory=None):
    return OkxSpotAdapter("key", "secret", "pass", rest_transport=transport, websocket_factory=websocket_factory, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))


class TestInstrumentAndAccountNormalization:
    def test_instrument_is_normalized_to_decimal_without_hidden_rounding(self):
        transport = FakeTransport([envelope([{"instId": INST, "baseCcy": "ANY", "quoteCcy": "QUOTE", "tickSz": "0.0001", "lotSz": "0.01", "minSz": "0.02", "pxPrecision": "4", "szPrecision": "2", "state": "live"}])])
        result = adapter(transport).get_instrument(INST)
        assert result.tick_size == Decimal("0.0001")
        assert result.lot_size == Decimal("0.01")
        assert result.min_notional is None

    def test_invalid_decimal_is_rejected(self):
        transport = FakeTransport([envelope([{"instId": INST, "baseCcy": "ANY", "quoteCcy": "QUOTE", "tickSz": "no", "lotSz": "1", "minSz": "1", "state": "live"}])])
        with pytest.raises(OkxApiError):
            adapter(transport).get_instrument(INST)

    def test_fee_rates_are_account_values(self):
        transport = FakeTransport([envelope([{"feeGroup": [{"maker": "-0.0008", "taker": "-0.001"}]}])])
        fees = adapter(transport).get_fee_rates(INST)
        assert fees.maker == Decimal("-0.0008")
        assert fees.taker == Decimal("-0.001")

    def test_balances_return_available_and_total_base_quote(self):
        transport = FakeTransport([envelope([{"details": [{"ccy": "ANY", "availBal": "3.1", "cashBal": "4.2"}, {"ccy": "QUOTE", "availBal": "5.3", "cashBal": "6.4"}]}])])
        balances = adapter(transport).get_balances("ANY", "QUOTE")
        assert balances.base_available == Decimal("3.1")
        assert balances.quote_total == Decimal("6.4")


class TestPlacementAndOrders:
    @pytest.mark.parametrize(("side", "okx_side"), [(BUY, "buy"), (SELL, "sell")])
    def test_post_only_placement_preserves_opaque_client_id(self, side, okx_side):
        transport = FakeTransport([envelope([{"ordId": "42", "clOrdId": "opaque-42", "sCode": "0", "sMsg": ""}])])
        result = adapter(transport).place_post_only_limit(INST, side, Decimal("1.2345"), Decimal("6.7"), "opaque-42")
        body = transport.calls[0][3]
        assert result.accepted is True
        assert result.client_order_id == "opaque-42"
        assert body == {"instId": INST, "tdMode": "cash", "side": okx_side, "ordType": "post_only", "px": "1.2345", "sz": "6.7", "clOrdId": "opaque-42"}

    def test_post_only_rejection_is_explicit(self):
        transport = FakeTransport([envelope([{"ordId": "", "clOrdId": "opaque", "sCode": "51008", "sMsg": "post only rejected"}])])
        result = adapter(transport).place_post_only_limit(INST, BUY, Decimal("1"), Decimal("2"), "opaque")
        assert result.accepted is False
        assert result.rejection_code == "51008"

    def test_rejection_never_uses_market_fallback(self):
        transport = FakeTransport([envelope([{"ordId": "", "clOrdId": "opaque", "sCode": "51008", "sMsg": "rejected"}])])
        adapter(transport).place_post_only_limit(INST, BUY, Decimal("1"), Decimal("2"), "opaque")
        assert len(transport.calls) == 1
        assert transport.calls[0][3]["ordType"] == "post_only"

    def test_open_orders_and_opaque_id_search(self):
        transport = FakeTransport([envelope([order(client_id="a"), order(order_id="2", client_id="b")])])
        orders = adapter(transport).list_open_orders(INST, "b")
        assert len(orders) == 1
        assert orders[0].order_id == "2"
        assert orders[0].client_order_id == "b"

    def test_get_order_returns_normalized_snapshot(self):
        transport = FakeTransport([envelope([order(state="filled", acc="4.5")])])
        snapshot = adapter(transport).get_order(INST, order_id="1")
        assert snapshot.state == "filled"
        assert snapshot.accumulated_fill_quantity == Decimal("4.5")

    def test_cancel_ack_is_not_final_state(self):
        transport = FakeTransport([envelope([{"ordId": "1", "clOrdId": "opaque", "sCode": "0", "sMsg": ""}])])
        result = adapter(transport).cancel_order(INST, order_id="1")
        assert result.accepted is True
        assert not hasattr(result, "state")


class TestFills:
    def test_partial_fill_is_observable_without_rearming(self):
        fill = {**order(state="partially_filled", acc="2", sz="4"), "tradeId": "t1", "fillPx": "1.1", "fillSz": "2", "fillTime": "3", "fee": "-0.01", "feeCcy": "QUOTE", "rebate": "", "rebateCcy": ""}
        transport = FakeTransport([envelope([fill])])
        result = adapter(transport).list_fills(INST)
        assert result[0].order_state == "partially_filled"
        assert result[0].accumulated_fill_quantity < Decimal("4")

    def test_complete_fill_and_actual_fee_are_normalized(self):
        fill = {**order(state="filled", acc="4", sz="4", side="sell"), "tradeId": "t2", "fillPx": "1.2", "fillSz": "2", "fillTime": "3", "fee": "-0.02", "feeCcy": "ANY", "rebate": "0.003", "rebateCcy": "ANY"}
        transport = FakeTransport([envelope([fill])])
        result = adapter(transport).list_fills(INST)[0]
        assert result.order_state == "filled"
        assert result.fee == Decimal("-0.02")
        assert result.rebate == Decimal("0.003")


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        message = next(self.messages)
        if isinstance(message, Exception):
            raise message
        return json.dumps(message)

    async def close(self):
        self.closed = True


class TestOrdersWebSocket:
    def test_orders_channel_normalizes_update(self):
        ws = FakeWebSocket([{"data": [order(state="filled", acc="4.5")]}])
        transport = FakeTransport([])
        async def factory(_): return ws
        seen = []
        async def receive(update):
            seen.append(update)
            raise StopAsyncIteration
        with pytest.raises(StopAsyncIteration):
            asyncio.run(adapter(transport, factory).listen_orders(INST, receive, reconnect_attempts=0))
        assert seen[0].state == "filled"
        assert [message["op"] for message in ws.sent] == ["login", "subscribe"]

    def test_reconnect_catches_up_open_orders_via_rest(self):
        first = FakeWebSocket([ConnectionError("lost")])
        second = FakeWebSocket([StopAsyncIteration()])
        sockets = iter([first, second])
        transport = FakeTransport([envelope([order(order_id="caught")])])
        async def factory(_): return next(sockets)
        seen = []
        async def receive(update): seen.append(update)
        with pytest.raises(StopAsyncIteration):
            asyncio.run(adapter(transport, factory).listen_orders(INST, receive, reconnect_attempts=1))
        assert [item.order_id for item in seen] == ["caught"]
