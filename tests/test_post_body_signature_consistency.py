"""Tests de cohérence signature/body pour les requêtes POST avec corps.

Contexte : la signature HMAC (_request) sérialise le body en JSON compact
(separators=(",", ":")), et le transport par défaut (_urllib_transport)
doit désormais utiliser exactement la même sérialisation — sinon OKX
recalcule une signature différente de celle envoyée et rejette avec
"50113: Invalid Sign", quel que soit le contenu réel de la requête.

Ce défaut est invisible pour :
  - toute requête GET (body=None des deux côtés, aucune divergence possible) ;
  - tous les tests de test_okx_spot_adapter.py qui injectent un
    RestTransport factice (celui-ci intercepte AVANT toute sérialisation,
    donc ne peut jamais révéler une divergence de sérialisation).

Ces tests ciblent donc explicitement le transport par défaut
(_urllib_transport), en interceptant urlopen — jamais en injectant un
rest_transport de substitution.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

from okx_spot_adapter import OkxSpotAdapter, _sign


FIXED_NOW = datetime(2026, 1, 1, 12, 0, 0, 123000, tzinfo=timezone.utc)


class FakeHttpResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


def _adapter_using_default_transport(secret_key="secret"):
    return OkxSpotAdapter(
        api_key="key", secret_key=secret_key, passphrase="pass",
        now=lambda: FIXED_NOW,
    )


def _expected_compact_body(body: dict) -> str:
    """Réplique indépendante de la sérialisation utilisée par _request()
    pour la signature — jamais le code testé lui-même."""
    return json.dumps(body, separators=(",", ":"))


def _ok_response():
    return FakeHttpResponse({"code": "0", "msg": "", "data": [{"ordId": "1", "clOrdId": "X", "sCode": "0", "sMsg": ""}]})


# ---------------------------------------------------------------------------
# 3. place_market_buy() — body compact
# ---------------------------------------------------------------------------

class TestPlaceMarketBuyProducesCompactBody:
    def test_body_bytes_match_compact_json(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.place_market_buy("XRP-USDC", Decimal("106.5"), "INITBUYabc")

        sent_data = captured_requests[0].data
        expected_body = {
            "instId": "XRP-USDC", "tdMode": "cash", "side": "buy",
            "ordType": "market", "sz": "106.5", "tgtCcy": "base_ccy",
            "clOrdId": "INITBUYabc",
        }
        assert sent_data.decode() == _expected_compact_body(expected_body)
        assert " " not in sent_data.decode()

    def test_signature_matches_actual_sent_bytes(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        secret_key = "my-secret-key"
        adapter = _adapter_using_default_transport(secret_key=secret_key)
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.place_market_buy("XRP-USDC", Decimal("106.5"), "INITBUYabc")

        request = captured_requests[0]
        actual_body_bytes_sent = request.data.decode()
        timestamp = FIXED_NOW.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        expected_prehash = f"{timestamp}POST/api/v5/trade/order{actual_body_bytes_sent}"
        expected_signature = _sign(secret_key, expected_prehash)

        assert request.get_header("Ok-access-sign") == expected_signature


# ---------------------------------------------------------------------------
# 4. place_post_only_limit() — body compact
# ---------------------------------------------------------------------------

class TestPlacePostOnlyLimitProducesCompactBody:
    def test_body_bytes_match_compact_json(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.place_post_only_limit("XRP-USDC", "BUY", Decimal("1.2345"), Decimal("6.7"), "GRIDabc")

        sent_data = captured_requests[0].data
        expected_body = {
            "instId": "XRP-USDC", "tdMode": "cash", "side": "buy",
            "ordType": "post_only", "px": "1.2345", "sz": "6.7",
            "clOrdId": "GRIDabc",
        }
        assert sent_data.decode() == _expected_compact_body(expected_body)
        assert " " not in sent_data.decode()

    def test_signature_matches_actual_sent_bytes(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        secret_key = "my-secret-key"
        adapter = _adapter_using_default_transport(secret_key=secret_key)
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.place_post_only_limit("XRP-USDC", "SELL", Decimal("110.0"), Decimal("1.5"), "GRIDxyz")

        request = captured_requests[0]
        actual_body_bytes_sent = request.data.decode()
        timestamp = FIXED_NOW.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        expected_prehash = f"{timestamp}POST/api/v5/trade/order{actual_body_bytes_sent}"
        expected_signature = _sign(secret_key, expected_prehash)

        assert request.get_header("Ok-access-sign") == expected_signature


# ---------------------------------------------------------------------------
# 5. cancel_order() — body compact
# ---------------------------------------------------------------------------

class TestCancelOrderProducesCompactBody:
    def test_body_bytes_match_compact_json(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.cancel_order("XRP-USDC", order_id="42")

        sent_data = captured_requests[0].data
        expected_body = {"instId": "XRP-USDC", "ordId": "42"}
        assert sent_data.decode() == _expected_compact_body(expected_body)
        assert " " not in sent_data.decode()

    def test_signature_matches_actual_sent_bytes(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return _ok_response()

        secret_key = "my-secret-key"
        adapter = _adapter_using_default_transport(secret_key=secret_key)
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.cancel_order("XRP-USDC", client_order_id="GRIDxyz")

        request = captured_requests[0]
        actual_body_bytes_sent = request.data.decode()
        timestamp = FIXED_NOW.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        expected_prehash = f"{timestamp}POST/api/v5/trade/cancel-order{actual_body_bytes_sent}"
        expected_signature = _sign(secret_key, expected_prehash)

        assert request.get_header("Ok-access-sign") == expected_signature


# ---------------------------------------------------------------------------
# 7. GET — non affecté
# ---------------------------------------------------------------------------

class TestGetRequestsUnaffected:
    def test_get_order_sends_no_body(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({
                "code": "0", "msg": "",
                "data": [{"ordId": "1", "clOrdId": "X", "instId": "XRP-USDC", "side": "buy",
                          "px": "1.0", "sz": "1.0", "accFillSz": "0", "state": "live",
                          "ordType": "post_only", "cTime": "1", "uTime": "2"}],
            })

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.get_order("XRP-USDC", order_id="1")

        assert captured_requests[0].data is None

    def test_get_order_signature_still_matches_empty_body(self):
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({
                "code": "0", "msg": "",
                "data": [{"ordId": "1", "clOrdId": "X", "instId": "XRP-USDC", "side": "buy",
                          "px": "1.0", "sz": "1.0", "accFillSz": "0", "state": "live",
                          "ordType": "post_only", "cTime": "1", "uTime": "2"}],
            })

        secret_key = "my-secret-key"
        adapter = _adapter_using_default_transport(secret_key=secret_key)
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.get_order("XRP-USDC", order_id="1")

        request = captured_requests[0]
        timestamp = FIXED_NOW.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        expected_prehash = f"{timestamp}GET/api/v5/trade/order?instId=XRP-USDC&ordId=1"
        expected_signature = _sign(secret_key, expected_prehash)

        assert request.get_header("Ok-access-sign") == expected_signature
