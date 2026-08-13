"""Tests unitaires pour OkxSpotAdapter.get_ticker().

Aucun appel réseau réel : le transport REST est simulé (RestTransport Protocol).
Ces tests ne couvrent que l'ajout de get_ticker() ; ils ne se substituent pas
à la suite existante de tests de l'adaptateur (non disponible dans cet
environnement — voir le rapport joint).
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from okx_spot_adapter import OkxApiError, OkxSpotAdapter, Ticker


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
# Cas nominal
# ---------------------------------------------------------------------------

class TestGetTickerNominal:
    def test_returns_ticker_with_last_price(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instType": "SPOT", "instId": "XRP-USDT", "last": "2.1534", "ts": "1672841403093"}],
        })
        ticker = _adapter(transport).get_ticker("XRP-USDT")

        assert ticker == Ticker(inst_id="XRP-USDT", last=Decimal("2.1534"), ts=1672841403093)

    def test_last_is_decimal_not_float(self):
        """La conversion doit passer directement str -> Decimal, jamais par float."""
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "0.1", "ts": "1672841403093"}],
        })
        ticker = _adapter(transport).get_ticker("XRP-USDT")

        # 0.1 n'est pas représentable exactement en binaire ; un aller-retour
        # par float introduirait une erreur de précision détectable ici.
        assert ticker.last == Decimal("0.1")
        assert isinstance(ticker.last, Decimal)


# ---------------------------------------------------------------------------
# Endpoint et absence d'authentification
# ---------------------------------------------------------------------------

class TestGetTickerEndpointAndAuth:
    def test_uses_correct_public_endpoint(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "2.15", "ts": "1"}],
        })
        _adapter(transport).get_ticker("XRP-USDT")

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call["method"] == "GET"
        assert call["path"] == "/api/v5/market/ticker"
        assert call["params"] == {"instId": "XRP-USDT"}

    def test_sends_no_authentication_headers(self):
        """
        Endpoint public OKX : get_ticker ne doit jamais passer par le
        mécanisme de signature (_request). Aucune clé, signature, passphrase
        ou timestamp d'authentification ne doit apparaître dans les headers.
        """
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "2.15", "ts": "1"}],
        })
        _adapter(transport).get_ticker("XRP-USDT")

        headers = transport.calls[0]["headers"]
        assert headers == {}
        assert "OK-ACCESS-KEY" not in headers
        assert "OK-ACCESS-SIGN" not in headers
        assert "OK-ACCESS-PASSPHRASE" not in headers

    def test_sends_no_request_body(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "2.15", "ts": "1"}],
        })
        _adapter(transport).get_ticker("XRP-USDT")

        assert transport.calls[0]["body"] is None


# ---------------------------------------------------------------------------
# Erreurs — cohérentes avec le mécanisme existant (OkxApiError)
# ---------------------------------------------------------------------------

class TestGetTickerErrors:
    def test_empty_data_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("XRP-USDT")

    def test_multiple_items_raises_okx_api_error(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [
                {"instId": "XRP-USDT", "last": "2.15", "ts": "1"},
                {"instId": "XRP-USDT", "last": "2.16", "ts": "2"},
            ],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("XRP-USDT")

    def test_top_level_error_code_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "51001", "msg": "Instrument ID does not exist", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("NOPE-USDT")

    def test_missing_last_raises_okx_api_error(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "ts": "1672841403093"}],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("XRP-USDT")

    def test_missing_ts_raises_okx_api_error(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "2.15"}],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("XRP-USDT")

    def test_invalid_last_raises_okx_api_error(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"instId": "XRP-USDT", "last": "not-a-number", "ts": "1"}],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_ticker("XRP-USDT")
