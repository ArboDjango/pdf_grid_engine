"""Tests pour le correctif du parsing des fills OKX -- accFillSz absent.

Réponses OKX synthétiques/fixées, aucun appel réseau réel. Ne place, n'annule
ni ne modifie aucun ordre -- teste exclusivement la fonction de parsing _fill().
"""

from decimal import Decimal

import pytest

from okx_spot_adapter import Fill, OkxApiError, _fill


# Fill réel XRP-USDC fourni pour ce chantier -- accFillSz absent, exactement
# comme retourné par le vrai endpoint GET /api/v5/trade/fills.
REAL_XRP_FILL = {
    "fillSz": "21.044",
    "fee": "-0.0415703176",
    "feeRate": "-0.002",
    "ordId": "3839040208425013248",
    "clOrdId": "GRIDe910f6b329b6232c473553b5",
    "side": "buy",
    "fillPx": "0.9877",
    "fillPnl": "0",
    "instId": "XRP-USDC",
    "feeCcy": "USDC",
    "fillTime": "1786926350744",
    "tradeId": "838447",
}


class TestAccFillSzAbsent:
    def test_a_parsing_succeeds_when_accfillsz_absent(self):
        """A. fill OKX avec accFillSz absent -> parsing réussi."""
        fill = _fill(REAL_XRP_FILL)
        assert isinstance(fill, Fill)

    def test_a_accumulated_fill_quantity_is_none_when_absent(self):
        """accFillSz absent -> accumulated_fill_quantity=None, jamais une valeur inventée."""
        fill = _fill(REAL_XRP_FILL)
        assert fill.accumulated_fill_quantity is None


class TestAccFillSzPresent:
    def test_b_accfillsz_preserved_when_present(self):
        """B. fill OKX avec accFillSz présent -> comportement actuel conservé (valeur exacte)."""
        item = dict(REAL_XRP_FILL)
        item["accFillSz"] = "21.044"
        fill = _fill(item)
        assert fill.accumulated_fill_quantity == Decimal("21.044")

    def test_b_accfillsz_distinct_value_preserved_exactly(self):
        """Valeur distincte de fillSz, pour vérifier qu'aucune confusion entre les deux champs n'existe."""
        item = dict(REAL_XRP_FILL)
        item["accFillSz"] = "99.999"
        fill = _fill(item)
        assert fill.accumulated_fill_quantity == Decimal("99.999")
        assert fill.fill_quantity == Decimal("21.044")  # fillSz, jamais confondu avec accFillSz


class TestFillSzAbsent:
    def test_c_missing_fillsz_raises_explicit_error(self):
        """C. fillSz absent -> échec propre et explicite (OkxApiError), jamais une valeur inventée."""
        item = dict(REAL_XRP_FILL)
        del item["fillSz"]
        with pytest.raises(OkxApiError):
            _fill(item)


class TestFeeAndFeeCurrency:
    def test_d_fee_preserved_correctly(self):
        """D. fee présent -> conservé correctement."""
        fill = _fill(REAL_XRP_FILL)
        assert fill.fee == Decimal("-0.0415703176")

    def test_d_fee_currency_preserved_correctly(self):
        """D. feeCcy présent -> conservé correctement."""
        fill = _fill(REAL_XRP_FILL)
        assert fill.fee_currency == "USDC"


class TestClientOrderId:
    def test_e_client_order_id_preserved_when_present(self):
        """E. clOrdId présent -> conservé."""
        fill = _fill(REAL_XRP_FILL)
        assert fill.client_order_id == "GRIDe910f6b329b6232c473553b5"

    def test_f_client_order_id_absent_falls_back_to_empty_string(self):
        """F. clOrdId absent -> comportement actuel conservé (chaîne vide, pas une exception)."""
        item = dict(REAL_XRP_FILL)
        del item["clOrdId"]
        fill = _fill(item)
        assert fill.client_order_id == ""


class TestRealXrpFillFromThisChantier:
    def test_g_real_fill_parses_completely_and_correctly(self):
        """G. Vérification spécifique du fill réel XRP-USDC fourni pour ce chantier."""
        fill = _fill(REAL_XRP_FILL)

        assert fill.trade_id == "838447"
        assert fill.order_id == "3839040208425013248"
        assert fill.client_order_id == "GRIDe910f6b329b6232c473553b5"
        assert fill.inst_id == "XRP-USDC"
        assert fill.side == "BUY"
        assert fill.fill_price == Decimal("0.9877")
        assert fill.fill_quantity == Decimal("21.044")
        assert fill.accumulated_fill_quantity is None  # absent côté OKX, jamais inventé
        assert fill.fee == Decimal("-0.0415703176")
        assert fill.fee_currency == "USDC"
        assert fill.rebate is None  # absent de la réponse fournie
        assert fill.rebate_currency is None
        assert fill.filled_at == 1786926350744

    def test_g_no_network_call_no_order_action(self):
        """
        Vérification de sécurité explicite : _fill() est une fonction de
        parsing pure -- aucun appel réseau, aucun placement, aucune
        annulation, aucune modification de compte n'est possible en
        l'appelant, par construction (aucun paramètre adapter/transport).
        """
        import inspect
        sig = inspect.signature(_fill)
        assert list(sig.parameters.keys()) == ["item"]  # aucune dépendance réseau possible
