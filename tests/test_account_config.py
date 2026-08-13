"""Tests unitaires pour OkxSpotAdapter.get_account_config().

Aucun appel réseau réel : le transport REST est simulé (RestTransport
Protocol), suivant exactement le même patron que get_fee_rates /
get_balances (endpoint privé, signé, via _one).

get_account_config sert exclusivement à vérifier la précondition
feeType == "1" avant tout achat MARKET (voir InitializationController) ;
ce module n'appelle jamais set-fee-type (aucune écriture).
"""

from datetime import datetime, timezone

from okx_spot_adapter import AccountConfig, OkxApiError, OkxSpotAdapter

import pytest


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

class TestGetAccountConfigNominal:
    def test_returns_account_config_with_fee_type(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"acctLv": "2", "feeType": "1", "posMode": "net_mode"}],
        })
        config = _adapter(transport).get_account_config()

        assert config == AccountConfig(fee_type="1")

    def test_extracts_fee_type_exactly_as_returned(self):
        """feeType est transmis tel quel (str), sans conversion ni interprétation."""
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"feeType": "0"}],
        })
        config = _adapter(transport).get_account_config()

        assert config.fee_type == "0"

    def test_uses_correct_private_endpoint(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"feeType": "1"}]})
        _adapter(transport).get_account_config()

        assert len(transport.calls) == 1
        call = transport.calls[0]
        assert call["method"] == "GET"
        assert call["path"] == "/api/v5/account/config"

    def test_sends_authentication_headers(self):
        """
        Endpoint privé (account/*) : contrairement à get_ticker/get_instrument,
        cet appel passe par _one/_request et doit donc porter la signature.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"feeType": "1"}]})
        _adapter(transport).get_account_config()

        headers = transport.calls[0]["headers"]
        assert "OK-ACCESS-KEY" in headers
        assert "OK-ACCESS-SIGN" in headers


# ---------------------------------------------------------------------------
# Erreurs — cohérentes avec le mécanisme existant (OkxApiError via _one)
# ---------------------------------------------------------------------------

class TestGetAccountConfigErrors:
    def test_empty_data_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "0", "msg": "", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_account_config()

    def test_multiple_items_raises_okx_api_error(self):
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"feeType": "1"}, {"feeType": "0"}],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_account_config()

    def test_top_level_error_code_raises_okx_api_error(self):
        transport = RecordingTransport({"code": "50001", "msg": "Service unavailable", "data": []})

        with pytest.raises(OkxApiError):
            _adapter(transport).get_account_config()

    def test_missing_fee_type_raises_key_error(self):
        """
        feeType est un champ chaîne simple, extrait par accès direct
        (item["feeType"]), exactement comme instId/baseCcy/state dans
        get_instrument — convention déjà en place dans ce fichier pour
        les champs chaîne obligatoires (par opposition à _required_decimal
        pour les champs numériques). Une absence lève donc KeyError,
        pas OkxApiError, par cohérence avec cette convention existante.
        """
        transport = RecordingTransport({"code": "0", "msg": "", "data": [{"acctLv": "2"}]})

        with pytest.raises(KeyError):
            _adapter(transport).get_account_config()

    def test_item_level_error_raises_okx_api_error(self):
        """
        get_account_config n'utilise pas allow_item_error=True (ce n'est
        pas une écriture pouvant être légitimement rejetée) : un item
        en erreur (sCode != 0) doit donc lever, pas être toléré.
        """
        transport = RecordingTransport({
            "code": "0", "msg": "",
            "data": [{"feeType": "1", "sCode": "1", "sMsg": "erreur interne"}],
        })

        with pytest.raises(OkxApiError):
            _adapter(transport).get_account_config()
