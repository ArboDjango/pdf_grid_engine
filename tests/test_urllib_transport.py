"""Tests unitaires pour OkxSpotAdapter._urllib_transport().

Aucun appel réseau réel : urllib.request.urlopen est intercepté par une
factice qui capture l'objet Request construit, sans jamais ouvrir de
connexion. Ces tests ciblent exclusivement le transport par défaut, qui
n'est exercé par aucun autre test de la suite (tous les autres tests
injectent un rest_transport factice et court-circuitent _urllib_transport).
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

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
    """Adapter sans rest_transport injecté : force l'usage de _urllib_transport."""
    return OkxSpotAdapter(
        api_key="key", secret_key="secret", passphrase="pass",
        now=lambda: FIXED_NOW,
    )


class TestUrllibTransportUserAgent:
    def test_sets_default_user_agent_when_none_provided(self):
        """
        Cas get_ticker() : headers={} fourni par l'appelant.
        _urllib_transport doit tout de même fixer un User-Agent explicite,
        pour éviter le User-Agent par défaut de urllib (Python-urllib/x.y),
        filtré par le WAF d'OKX (HTTP 403 constaté en conditions réelles).
        """
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({"code": "0", "msg": "", "data": [{"instId": "XRP-USDT", "last": "1.0", "ts": "1"}]})

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter.get_ticker("XRP-USDT")

        assert len(captured_requests) == 1
        user_agent = captured_requests[0].get_header("User-agent")
        assert user_agent is not None
        assert user_agent != ""
        assert "python-urllib" not in user_agent.lower()

    def test_preserves_caller_provided_headers(self):
        """
        Cas _request() : headers contient OK-ACCESS-KEY, OK-ACCESS-SIGN, etc.
        Ces en-têtes doivent survivre intégralement à côté du User-Agent
        par défaut — la signature ne doit jamais être altérée ou perdue.
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
        sent = captured_requests[0]
        assert sent.get_header("Ok-access-key") == "key"
        assert sent.get_header("Ok-access-passphrase") == "pass"
        assert sent.get_header("Ok-access-sign") is not None
        assert sent.get_header("Ok-access-timestamp") is not None
        assert sent.get_header("Content-type") == "application/json"

    def test_caller_user_agent_takes_priority_over_default(self):
        """
        Si l'appelant fournit explicitement un User-Agent, il ne doit pas
        être écrasé par le défaut (comportement de dict.update).
        """
        captured_requests = []

        def fake_urlopen(request, timeout=15):
            captured_requests.append(request)
            return FakeHttpResponse({"code": "0", "msg": "", "data": [{"instId": "XRP-USDT", "last": "1.0", "ts": "1"}]})

        adapter = _adapter_using_default_transport()
        with patch("okx_spot_adapter.urlopen", fake_urlopen):
            adapter._transport("GET", "/api/v5/market/ticker", {"instId": "XRP-USDT"}, None, {"User-Agent": "custom-caller/9.9"})

        assert len(captured_requests) == 1
        assert captured_requests[0].get_header("User-agent") == "custom-caller/9.9"
