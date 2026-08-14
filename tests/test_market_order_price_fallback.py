"""Tests de _order() / _order_price() — repli px vide -> avgPx (ordres MARKET).

Contexte : un ordre MARKET rempli renvoyé par OKX (get_order,
list_open_orders, listen_orders) a px="" (aucun prix limite pour un
ordre au marché) et renseigne avgPx à la place. _required_decimal(item,
"px") levait alors OkxApiError("Valeur décimale OKX absente") pour tout
ordre MARKET rempli — notamment l'achat initial S0
(InitializationController._buy_deficit -> place_market_buy), rendant
son propre statut illisible après exécution.

Ce fichier utilise le transport RestTransport factice existant
(cohérent avec test_okx_spot_adapter.py) : le défaut corrigé ici est
dans le parsing de la réponse, pas dans le transport, donc pas besoin
d'intercepter urlopen pour ces tests.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from okx_spot_adapter import OkxApiError, OkxSpotAdapter


INST = "XRP-USDC"


def envelope(data, code="0", msg=""):
    return {"code": code, "msg": msg, "data": data}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, path, params, body, headers):
        self.calls.append((method, path, params, body, headers))
        return self.responses.pop(0)


def adapter(transport):
    return OkxSpotAdapter("key", "secret", "pass", rest_transport=transport, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))


class TestMarketOrderWithEmptyPxFallsBackToAvgPx:
    def test_filled_market_order_parses_using_avg_px(self):
        """
        Cas exact rapporté par le serveur réel : clOrdId=INITBUYead951f3f4601bef1447d00d,
        state=filled, ordType=market, px="", avgPx="1.001", fillPx="1.001",
        sz="101.525", accFillSz="101.525".
        """
        item = {
            "ordId": "1", "clOrdId": "INITBUYead951f3f4601bef1447d00d",
            "instId": INST, "side": "buy",
            "px": "", "avgPx": "1.001", "fillPx": "1.001",
            "sz": "101.525", "accFillSz": "101.525",
            "state": "filled", "ordType": "market",
            "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        snapshot = adapter(transport).get_order(INST, client_order_id="INITBUYead951f3f4601bef1447d00d")

        assert snapshot.price == Decimal("1.001")
        assert snapshot.quantity == Decimal("101.525")
        assert snapshot.accumulated_fill_quantity == Decimal("101.525")
        assert snapshot.state == "filled"
        assert snapshot.order_type == "market"

    def test_market_order_with_missing_px_key_entirely_also_falls_back(self):
        """Repli également valide si px est absent du dict, pas seulement vide."""
        item = {
            "ordId": "1", "clOrdId": "opaque", "instId": INST, "side": "buy",
            "avgPx": "2.5", "sz": "10", "accFillSz": "10",
            "state": "filled", "ordType": "market", "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        snapshot = adapter(transport).get_order(INST, order_id="1")

        assert snapshot.price == Decimal("2.5")

    def test_market_order_with_empty_px_and_missing_avg_px_still_raises(self):
        """
        Si ni px ni avgPx ne sont exploitables, l'erreur explicite doit
        être conservée — aucun masquage silencieux d'une réponse OKX
        réellement incomplète.
        """
        item = {
            "ordId": "1", "clOrdId": "opaque", "instId": INST, "side": "buy",
            "px": "", "sz": "10", "accFillSz": "10",
            "state": "filled", "ordType": "market", "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        with pytest.raises(OkxApiError):
            adapter(transport).get_order(INST, order_id="1")


class TestLimitOrderStillUsesPx:
    def test_limit_order_with_px_populated_uses_px_not_avg_px(self):
        """
        Ordre LIMIT (le cas des ordres de grille, place_post_only_limit) :
        px est toujours renseigné et doit primer, même si avgPx est
        également présent (ordre partiellement ou totalement rempli).
        """
        item = {
            "ordId": "1", "clOrdId": "GRIDabc", "instId": INST, "side": "sell",
            "px": "1.2345", "avgPx": "1.2340", "sz": "6.7", "accFillSz": "6.7",
            "state": "filled", "ordType": "post_only", "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        snapshot = adapter(transport).get_order(INST, client_order_id="GRIDabc")

        assert snapshot.price == Decimal("1.2345")  # px, pas avgPx

    def test_limit_order_live_without_avg_px_still_works(self):
        """
        Ordre LIMIT non exécuté (live) : avgPx absent, px seul doit
        suffire — comportement inchangé par rapport à avant ce correctif.
        """
        item = {
            "ordId": "1", "clOrdId": "GRIDabc", "instId": INST, "side": "buy",
            "px": "0.95", "sz": "10", "accFillSz": "0",
            "state": "live", "ordType": "post_only", "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        snapshot = adapter(transport).get_order(INST, client_order_id="GRIDabc")

        assert snapshot.price == Decimal("0.95")


class TestNoRegressionOnOtherFields:
    def test_all_other_fields_unaffected_for_market_order(self):
        """
        Non-régression explicite : seul le champ price est concerné par
        le repli — tous les autres champs d'OrderSnapshot restent extraits
        exactement comme avant.
        """
        item = {
            "ordId": "999", "clOrdId": "INITBUYxyz", "instId": INST, "side": "buy",
            "px": "", "avgPx": "1.001", "sz": "101.525", "accFillSz": "101.525",
            "state": "filled", "ordType": "market", "cTime": "111", "uTime": "222",
        }
        transport = FakeTransport([envelope([item])])

        snapshot = adapter(transport).get_order(INST, order_id="999")

        assert snapshot.order_id == "999"
        assert snapshot.client_order_id == "INITBUYxyz"
        assert snapshot.inst_id == INST
        assert snapshot.side == "BUY"
        assert snapshot.created_at == 111
        assert snapshot.updated_at == 222

    def test_list_open_orders_also_benefits_from_fallback(self):
        """
        _order() est partagé par get_order, list_open_orders et
        listen_orders : le repli doit s'appliquer uniformément, sans
        modification spécifique à chacune de ces méthodes.
        """
        item = {
            "ordId": "1", "clOrdId": "INITBUYabc", "instId": INST, "side": "buy",
            "px": "", "avgPx": "1.001", "sz": "10", "accFillSz": "10",
            "state": "filled", "ordType": "market", "cTime": "1", "uTime": "2",
        }
        transport = FakeTransport([envelope([item])])

        orders = adapter(transport).list_open_orders(INST)

        assert orders[0].price == Decimal("1.001")
