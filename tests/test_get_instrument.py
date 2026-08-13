"""Tests unitaires pour OkxSpotAdapter.get_instrument() — endpoint public.

Contexte : GET /api/v5/account/instruments filtre par éligibilité de compte
(documenté OKX : peut renvoyer data=[] pour un compte ne "possédant" pas
la paire, indépendamment de son existence réelle sur l'exchange). Ce n'est
pas l'endpoint pertinent pour lire les caractéristiques génériques d'un
instrument. GET /api/v5/public/instruments est l'endpoint public standard,
sans dépendance au type de compte, sans authentification.

Aucun appel réseau réel : transport simulé (RestTransport Protocol).
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from okx_spot_adapter import InstrumentRules, OkxApiError, OkxSpotAdapter


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


_VALID_ITEM = {
    "instId": "XRP-USDT", "baseCcy": "XRP", "quoteCcy": "USDT",
    "tickSz": "0.0001", "lotSz": "0.1", "minSz": "0.1",
    "minNotional": None, "pxPrecision": "4", "szPrecision": "1",
    "state": "live",
}


# ---------------------------------------------------------------------------
# Cas nominal — reproduction et correction du bug rapporté
# ---------------------------------------------------------------------------

class TestGetInstrumentNominal:
    def test_returns_instrument_rules(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM]})
        rules = _adapter(transport).get_instrument("XRP-USDT")

        assert rules == InstrumentRules(
            inst_id="XRP-USDT", base_ccy="XRP", quote_ccy="USDT",
            tick_size=Decimal("0.0001"), lot_size=Decimal("0.1"), min_size=Decimal("0.1"),
            min_notional=None, price_precision=4, size_precision=1, state="live",
        )

    def test_account_scoped_empty_data_no_longer_reproduced(self):
        """
        Régression ciblée : le bug rapporté était account/instruments
        renvoyant data=[] pour un compte n'ayant pas "l'éligibilité" sur
        la paire, provoquant OkxApiError alors que l'instrument existe
        réellement. Avec l'endpoint public, un data=[] valide et légitime
        (transport factice) doit produire le même InstrumentRules attendu,
        prouvant que le chemin n'est plus soumis au filtrage de compte.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM]})
        rules = _adapter(transport).get_instrument("XRP-USDT")
        assert rules.inst_id == "XRP-USDT"


# ---------------------------------------------------------------------------
# Endpoint et absence d'authentification
# ---------------------------------------------------------------------------

class TestGetInstrumentEndpointAndAuth:
    def test_uses_public_endpoint_not_account_endpoint(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM]})
        _adapter(transport).get_instrument("XRP-USDT")

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call["method"] == "GET"
        assert call["path"] == "/api/v5/public/instruments"
        assert call["path"] != "/api/v5/account/instruments"
        assert call["params"] == {"instType": "SPOT", "instId": "XRP-USDT"}

    def test_sends_no_authentication_headers(self):
        """
        Endpoint public OKX : get_instrument ne doit jamais passer par le
        mécanisme de signature (_request), comme get_ticker.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM]})
        _adapter(transport).get_instrument("XRP-USDT")

        headers = transport.calls[0]["headers"]
        assert headers == {}
        assert "OK-ACCESS-KEY" not in headers
        assert "OK-ACCESS-SIGN" not in headers
        assert "OK-ACCESS-PASSPHRASE" not in headers

    def test_sends_no_request_body(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM]})
        _adapter(transport).get_instrument("XRP-USDT")

        assert transport.calls[0]["body"] is None


# ---------------------------------------------------------------------------
# Erreurs — cohérentes avec le mécanisme existant (OkxApiError)
# ---------------------------------------------------------------------------

class TestGetInstrumentErrors:
    def test_empty_data_raises_okx_api_error(self):
        """
        Cas où l'instrument n'existe réellement pas sur ce venue (ex.
        paire absente du catalogue EEA) : data=[] reste une erreur légitime
        sur l'endpoint public — contrairement à account/instruments, ce
        n'est plus un faux-négatif dû à l'éligibilité de compte.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_instrument("XRP-USDT")

    def test_multiple_items_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [_VALID_ITEM, _VALID_ITEM]})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_instrument("XRP-USDT")

    def test_top_level_error_code_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "51001", "msg": "Instrument ID does not exist", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_instrument("NOPE-USDT")

    def test_missing_tick_size_raises_okx_api_error(self):
        item = dict(_VALID_ITEM)
        del item["tickSz"]
        transport = RecordingTransport({"code": "0", "msg": "", "data": [item]})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_instrument("XRP-USDT")
