"""Tests unitaires pour OkxSpotAdapter.place_market_buy().

Aucun appel réseau réel : le transport REST est simulé (RestTransport
Protocol), suivant exactement le même patron que les tests existants de
place_post_only_limit / get_ticker / get_instrument.

place_market_buy est réservé à l'initialisation patrimoniale (achat
initial de spot, S0 du PDF) : jamais utilisé pour les ordres de grille,
qui restent LIMIT POST-ONLY (inchangé).
"""

from datetime import datetime, timezone
from decimal import Decimal

from okx_spot_adapter import OkxSpotAdapter, PlacementResult


FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RecordingTransport:
    """Transport simulé qui enregistre exactement l'appel reçu."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, method, path, params, body, headers):
        self.calls.append(
            {"method": method, "path": path, "params": params, "body": body, "headers": headers}
        )
        return self.response


def _adapter(transport):
    return OkxSpotAdapter(
        api_key="key", secret_key="secret", passphrase="pass",
        rest_transport=transport, now=lambda: FIXED_NOW,
    )


# ---------------------------------------------------------------------------
# Payload — conformité au contrat OKX (tgtCcy=base_ccy, ordType=market)
# ---------------------------------------------------------------------------

class TestPlaceMarketBuyPayload:
    def test_uses_correct_endpoint(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "X")

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call["method"] == "POST"
        assert call["path"] == "/api/v5/trade/order"

    def test_payload_uses_market_order_type(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "X")

        body = transport.calls[0]["body"]
        assert body["ordType"] == "market"
        assert body["side"] == "buy"

    def test_payload_sets_tgt_ccy_base(self):
        """
        Point critique : sans tgtCcy=base_ccy, sz serait interprété en
        devise de cotation par défaut pour un achat MARKET (doc OKX),
        obligeant à une conversion — exactement l'incertitude de prix
        que l'usage direct de required_spot doit éviter.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "X")

        body = transport.calls[0]["body"]
        assert body["tgtCcy"] == "base_ccy"

    def test_payload_quantity_is_exact_required_spot_value(self):
        """sz doit porter exactement la quantité transmise, sans arrondi ni recalcul."""
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.504321"), "X")

        assert transport.calls[0]["body"]["sz"] == "106.504321"

    def test_payload_has_no_price_field(self):
        """Un ordre MARKET n'a pas de prix limite — px absent du payload."""
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "X")

        assert "px" not in transport.calls[0]["body"]

    def test_payload_uses_cash_trade_mode(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "X")

        assert transport.calls[0]["body"]["tdMode"] == "cash"

    def test_client_order_id_passed_through(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "INITBUYabc", "sCode": "0", "sMsg": ""}]})
        _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "INITBUYabc")

        assert transport.calls[0]["body"]["clOrdId"] == "INITBUYabc"


# ---------------------------------------------------------------------------
# Résultat — même convention que place_post_only_limit (jamais d'exception
# sur un rejet d'ordre ; accepted=False + rejection_code/message)
# ---------------------------------------------------------------------------

class TestPlaceMarketBuyResult:
    def test_accepted_order_returns_placement_result(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": "998877", "clOrdId": "INITBUY1", "sCode": "0", "sMsg": ""}]})
        result = _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "INITBUY1")

        assert result == PlacementResult(True, "998877", "INITBUY1", None, None)

    def test_rejected_order_does_not_raise(self):
        """
        Convention identique à place_post_only_limit : un rejet (sCode!=0)
        ne lève jamais d'exception — allow_item_error=True. C'est
        l'appelant (InitializationController) qui décide de la suite
        (transition ERROR), jamais l'adaptateur.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"ordId": None, "clOrdId": "INITBUY1", "sCode": "51008", "sMsg": "Insufficient balance"}]})
        result = _adapter(transport).place_market_buy("XRP-USDC", Decimal("106.5"), "INITBUY1")

        assert result.accepted is False
        assert result.rejection_code == "51008"
        assert result.rejection_message == "Insufficient balance"
        assert result.order_id is None
