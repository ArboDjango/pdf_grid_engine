"""Tests du transport urllib réel de OkxSpotAdapter.

Contrairement aux tests existants de ce fichier (qui injectent tous un
RestTransport factice, court-circuitant _urllib_transport), ces tests
exercent directement _urllib_transport en interceptant urlopen — aucun
appel réseau réel, mais le vrai code de conversion HTTPError -> OkxApiError
est exécuté.

Contexte : lors du premier test LIVE, un HTTP 401 (et potentiellement
400/403/500) provoquait une urllib.error.HTTPError brute au lieu d'une
OkxApiError, court-circuitant l'abstraction d'erreur du reste du code
(_request, get_ticker, get_instrument, InitializationController,
GridActivationController, run_grid_trading.py — qui capturent tous
OkxApiError, jamais HTTPError).
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from okx_spot_adapter import OkxApiError, OkxSpotAdapter


FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _adapter_using_default_transport():
    """Adaptateur sans rest_transport injecté : force l'usage de _urllib_transport."""
    return OkxSpotAdapter(
        api_key="key", secret_key="secret", passphrase="pass",
        now=lambda: FIXED_NOW,
    )


def _http_error(status: int, body: bytes, reason: str = "error") -> HTTPError:
    """Construit une HTTPError réaliste, avec un corps lisible via .read()
    exactement comme le fait urllib pour une vraie réponse HTTP en erreur."""
    return HTTPError(
        url="https://eea.okx.com/api/v5/trade/order",
        code=status,
        msg=reason,
        hdrs=None,
        fp=BytesIO(body),
    )


class FakeHttpResponse:
    """Simule une réponse HTTP 200 réussie, sans réseau."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


class TestHttpErrorConvertedToOkxApiError:
    def test_http_400_with_okx_json_body_raises_okx_api_error_with_code_and_msg(self):
        """
        Cas exact rapporté : HTTP 400 avec
        {"code":"51000","data":[],"msg":"Parameter clOrdId error"}
        doit produire OkxApiError("Erreur OKX 51000: Parameter clOrdId error").
        """
        body = json.dumps({"code": "51000", "data": [], "msg": "Parameter clOrdId error"}).encode()

        def raising_urlopen(request, timeout=15):
            raise _http_error(400, body)

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")

        assert "51000" in str(exc_info.value)
        assert "Parameter clOrdId error" in str(exc_info.value)

    def test_http_401_with_okx_json_body_raises_okx_api_error_not_http_error(self):
        """
        Cas exact rapporté en LIVE : HTTP 401 doit produire OkxApiError,
        jamais urllib.error.HTTPError brute.
        """
        body = json.dumps({"code": "50113", "data": [], "msg": "Invalid signature"}).encode()

        def raising_urlopen(request, timeout=15):
            raise _http_error(401, body)

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError):
                adapter.get_order("XRP-USDC", order_id="1")
            # Vérification négative explicite : ce n'est jamais HTTPError
            # qui remonte, même indirectement.
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")
            assert not isinstance(exc_info.value, HTTPError)

    @pytest.mark.parametrize("status", [403, 500])
    def test_http_error_with_non_json_body_raises_okx_api_error_with_status(self, status):
        """
        HTTP 403/500 avec un corps non-JSON (ex. page d'erreur HTML d'un
        proxy/load-balancer) doit tout de même produire OkxApiError, avec
        le statut HTTP visible, sans jamais lever d'exception JSON parasite
        (JSONDecodeError) qui masquerait l'erreur réelle.
        """
        body = b"<html><body>502 Bad Gateway</body></html>"

        def raising_urlopen(request, timeout=15):
            raise _http_error(status, body)

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")

        assert str(status) in str(exc_info.value)

    def test_http_error_with_empty_body_raises_okx_api_error(self):
        """Corps de réponse vide (aucun octet) : ne doit jamais lever autre chose qu'OkxApiError."""
        def raising_urlopen(request, timeout=15):
            raise _http_error(500, b"")

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")

        assert "500" in str(exc_info.value)


class TestHttp200BehaviorUnchanged:
    def test_http_200_success_unchanged(self):
        """
        Non-régression explicite : une réponse HTTP 200 avec code="0"
        continue de fonctionner exactement comme avant ce correctif.
        """
        def fake_urlopen(request, timeout=15):
            return FakeHttpResponse({
                "code": "0", "msg": "",
                "data": [{
                    "ordId": "1", "clOrdId": "INITBUY786494c6ce8b3407758bae29",
                    "instId": "XRP-USDC", "side": "buy", "px": "1.0", "sz": "10",
                    "accFillSz": "0", "state": "live", "ordType": "market",
                    "cTime": "1", "uTime": "2",
                }],
            })

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            snapshot = adapter.get_order("XRP-USDC", order_id="1")

        assert snapshot.state == "live"
        assert snapshot.client_order_id == "INITBUY786494c6ce8b3407758bae29"

    def test_http_200_with_okx_error_code_still_raises_okx_api_error(self):
        """
        Non-régression explicite : HTTP 200 contenant
        {"code":"51603","msg":"Order does not exist",...} — cas exact
        rapporté avec un ordre inexistant — continue d'être transformé
        par _request() en OkxApiError("Erreur OKX 51603: Order does not
        exist"), exactement comme avant ce correctif.
        """
        def fake_urlopen(request, timeout=15):
            return FakeHttpResponse({"code": "51603", "msg": "Order does not exist", "data": []})

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", client_order_id="INITBUY786494c6ce8b3407758bae29")

        assert "51603" in str(exc_info.value)
        assert "Order does not exist" in str(exc_info.value)


class TestNoSecretsInErrorMessages:
    def test_error_message_never_contains_api_credentials(self):
        """
        Le message d'erreur final ne doit jamais contenir la clé API, le
        secret ou la passphrase utilisés pour la connexion — même en cas
        d'échec HTTP portant sur une requête authentifiée.
        """
        body = json.dumps({"code": "50113", "data": [], "msg": "Invalid signature"}).encode()

        def raising_urlopen(request, timeout=15):
            # Vérifie, comme le ferait un vrai serveur, que les secrets
            # transitent bien dans les en-têtes de la requête (donc dans
            # request.headers), jamais dans le corps de la réponse d'erreur.
            assert "OK-ACCESS-KEY" in request.headers or "Ok-access-key" in request.headers
            raise _http_error(401, body)

        adapter = OkxSpotAdapter(
            api_key="SUPER_SECRET_API_KEY",
            secret_key="SUPER_SECRET_SECRET_KEY",
            passphrase="SUPER_SECRET_PASSPHRASE",
            now=lambda: FIXED_NOW,
        )
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")

        message = str(exc_info.value)
        assert "SUPER_SECRET_API_KEY" not in message
        assert "SUPER_SECRET_SECRET_KEY" not in message
        assert "SUPER_SECRET_PASSPHRASE" not in message

    def test_fallback_envelope_never_contains_request_headers(self):
        """
        L'enveloppe de repli (corps non-JSON) est construite uniquement à
        partir du statut HTTP — jamais des en-têtes de la requête d'origine,
        qui contiennent la signature et les identifiants.
        """
        def raising_urlopen(request, timeout=15):
            raise _http_error(500, b"internal error, no json")

        adapter = OkxSpotAdapter(
            api_key="SUPER_SECRET_API_KEY",
            secret_key="SUPER_SECRET_SECRET_KEY",
            passphrase="SUPER_SECRET_PASSPHRASE",
            now=lambda: FIXED_NOW,
        )
        with patch("okx_spot_adapter.urlopen", raising_urlopen):
            with pytest.raises(OkxApiError) as exc_info:
                adapter.get_order("XRP-USDC", order_id="1")

        message = str(exc_info.value)
        assert "SUPER_SECRET_API_KEY" not in message
        assert "SUPER_SECRET_SECRET_KEY" not in message
        assert "SUPER_SECRET_PASSPHRASE" not in message
