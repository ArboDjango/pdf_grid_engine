"""Tests unitaires verrouillant le domaine régional EEA d'OkxSpotAdapter.

Contexte : sous MiCA, les comptes enregistrés sur my.okx.com doivent utiliser
les hôtes eea.okx.com (REST) et wseea.okx.com (WebSocket privé) — les hôtes
globaux (www.okx.com, ws.okx.com) renvoient une erreur d'authentification
("API key doesn't exist") pour ces comptes, même avec des identifiants valides.

Ces tests ne couvrent que le domaine utilisé, jamais la logique de
signature, les endpoints métier, ni le mode démo (sujet distinct).
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

from okx_spot_adapter import OkxSpotAdapter


FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeHttpResponse:
    """Simule le context manager retourné par urlopen(), sans réseau."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


def _adapter_using_default_transport():
    return OkxSpotAdapter(
        api_key="key", secret_key="secret", passphrase="pass",
        now=lambda: FIXED_NOW,
    )


class TestEeaRegionalDomain:
    def test_rest_url_is_eea_domain(self):
        """Déclaration de classe : REST_URL doit pointer vers eea.okx.com."""
        assert OkxSpotAdapter.REST_URL == "https://eea.okx.com"

    def test_private_ws_url_is_eea_domain(self):
        """Déclaration de classe : PRIVATE_WS_URL doit pointer vers wseea.okx.com."""
        assert OkxSpotAdapter.PRIVATE_WS_URL == "wss://wseea.okx.com:8443/ws/v5/private"

    def test_rest_url_no_longer_targets_global_domain(self):
        """Verrou négatif : l'ancien domaine global ne doit plus être utilisé."""
        assert "www.okx.com" not in OkxSpotAdapter.REST_URL

    def test_private_ws_url_no_longer_targets_global_domain(self):
        assert OkxSpotAdapter.PRIVATE_WS_URL != "wss://ws.okx.com:8443/ws/v5/private"

    def test_urllib_transport_actually_targets_eea_domain_at_runtime(self):
        """
        Vérification runtime, pas seulement déclarative : la requête HTTP
        réellement construite par _urllib_transport doit cibler eea.okx.com,
        pas seulement la constante de classe.
        """
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({"code": "0", "msg": "", "data": [{"instId": "XRP-USDT", "last": "1.0", "ts": "1"}]})

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.get_ticker("XRP-USDT")

        assert len(captured_requests) == 1
        assert captured_requests[0].full_url.startswith("https://eea.okx.com")

    def test_signed_request_also_targets_eea_domain_at_runtime(self):
        """
        Même vérification pour un appel authentifié (get_instrument),
        afin de couvrir le chemin _request() en plus du chemin direct
        emprunté par get_ticker().
        """
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({
                "code": "0", "msg": "",
                "data": [{"instId": "XRP-USDT", "baseCcy": "XRP", "quoteCcy": "USDT",
                          "tickSz": "0.0001", "lotSz": "0.1", "minSz": "0.1",
                          "minNotional": None, "pxPrecision": "4", "szPrecision": "1",
                          "state": "live"}],
            })

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.get_instrument("XRP-USDT")

        assert len(captured_requests) == 1
        assert captured_requests[0].full_url.startswith("https://eea.okx.com")


class TestEeaChangeDoesNotAffectInjectedTransportTests:
    """
    Vérifie explicitement (point 3 de la demande) que le changement de
    domaine n'a aucun effet sur les appels utilisant un rest_transport
    injecté — puisque REST_URL et PRIVATE_WS_URL ne sont utilisés que
    par les fabriques par défaut (_urllib_transport, _default_websocket_factory),
    jamais par un transport fourni explicitement au constructeur.
    """

    def test_injected_transport_ignores_rest_url_entirely(self):
        calls = []

        def recording_transport(method, path, params, body, headers):
            calls.append((method, path, params, body, headers))
            return {"code": "0", "msg": "", "data": [{"instId": "XRP-USDT", "last": "1.0", "ts": "1"}]}

        adapter = OkxSpotAdapter(
            api_key="key", secret_key="secret", passphrase="pass",
            rest_transport=recording_transport, now=lambda: FIXED_NOW,
        )
        ticker = adapter.get_ticker("XRP-USDT")

        assert ticker.last == __import__("decimal").Decimal("1.0")
        assert len(calls) == 1
        # Le transport injecté ne référence jamais REST_URL : aucune URL
        # absolue n'entre en jeu dans ce chemin, seul le path relatif compte.
        assert calls[0][1] == "/api/v5/market/ticker"
