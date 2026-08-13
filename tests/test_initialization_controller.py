"""Tests pour InitializationController et derive_client_order_id.

Aucun appel réseau réel : adaptateur simulé complet (lectures + écritures),
distinct du FakeSource read-only utilisé par tests/test_grid_preflight.py
— ce composant a explicitement besoin d'écriture, à la différence du
preflight.
"""

from decimal import Decimal

import pytest

from grid_preflight import PreflightConfig
from initialization_controller import (
    InitializationController,
    InitState,
    derive_client_order_id,
)
from okx_spot_adapter import Balances, FeeRates, InstrumentRules, OkxApiError, OrderSnapshot, PlacementResult
from trellis_calculator import GridGeometry


def config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("110"), gll=Decimal("90"), nu=2, nl=2,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


# ---------------------------------------------------------------------------
# derive_client_order_id — déterminisme, exclusion des soldes, format OKX
# ---------------------------------------------------------------------------

class TestDeriveClientOrderId:
    def test_deterministic_same_config_same_id(self):
        c = config()
        id_1 = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        id_2 = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        assert id_1 == id_2

    @pytest.mark.parametrize("field,value", [
        ("gul", Decimal("111")), ("gll", Decimal("91")), ("nu", 3), ("nl", 3),
        ("p0", Decimal("101")), ("spacing_h_pct", Decimal("0.002")),
        ("alpha", Decimal("0.6")), ("inst_id", "BTC-USDC"),
    ])
    def test_changes_when_any_economic_field_changes(self, field, value):
        base_id = derive_client_order_id(config(), Decimal("0.1"), Decimal("0.1"))
        changed_id = derive_client_order_id(config(**{field: value}), Decimal("0.1"), Decimal("0.1"))
        assert base_id != changed_id

    def test_excludes_operational_margin(self):
        """
        operational_margin n'influence jamais required_spot (il n'entre
        que dans _cycle_check, la vérification de rentabilité — voir
        trace exacte de grid_preflight.run_preflight). L'inclure dans le
        hash produirait des ID différents pour une intention d'achat
        strictement identique, cassant inutilement la détection de
        continuité au redémarrage si ce seul paramètre était ajusté.
        """
        id_1 = derive_client_order_id(config(operational_margin=Decimal("0.5")), Decimal("0.1"), Decimal("0.1"))
        id_2 = derive_client_order_id(config(operational_margin=Decimal("0.9")), Decimal("0.1"), Decimal("0.1"))
        assert id_1 == id_2

    def test_changes_when_geometry_changes(self):
        base_id = derive_client_order_id(config(), Decimal("0.1"), Decimal("0.1"))
        changed_id = derive_client_order_id(config(geometry=GridGeometry.EQUIRATIO), Decimal("0.1"), Decimal("0.1"))
        assert base_id != changed_id

    def test_excludes_balances_by_construction(self):
        """
        derive_client_order_id ne prend même pas balances en paramètre :
        ce test documente et verrouille explicitement cette exclusion
        architecturale (voir docstring du module) — deux soldes
        distincts n'ont structurellement aucun moyen d'influencer l'ID.
        """
        c = config()
        id_with_context_a = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        # Aucune balance n'entre dans le calcul : rejouer le même appel,
        # quelle que soit l'évolution patrimoniale entre deux instants,
        # produit forcément le même résultat.
        id_with_context_b = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        assert id_with_context_a == id_with_context_b

    def test_changes_when_tick_size_changes(self):
        c = config()
        id_1 = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        id_2 = derive_client_order_id(c, Decimal("0.01"), Decimal("0.1"))
        assert id_1 != id_2

    def test_changes_when_lot_size_changes(self):
        c = config()
        id_1 = derive_client_order_id(c, Decimal("0.1"), Decimal("0.1"))
        id_2 = derive_client_order_id(c, Decimal("0.1"), Decimal("0.01"))
        assert id_1 != id_2

    def test_format_valid_for_okx(self):
        derived = derive_client_order_id(config(), Decimal("0.1"), Decimal("0.1"))
        assert len(derived) <= 32
        assert derived[0].isalpha()
        assert derived.isalnum()
        assert derived.startswith("INITBUY")


# ---------------------------------------------------------------------------
# Adaptateur simulé — lectures + écritures, complet
# ---------------------------------------------------------------------------

class FakeAdapter:
    """Simule OkxSpotAdapter : lectures (OkxPreflightSource) + écritures."""

    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.placed_orders = []
        self.existing_order = None  # OrderSnapshot | None, retourné par get_order
        self.placement_result = None  # PlacementResult à retourner par place_market_buy

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        if self.existing_order is None:
            raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")
        return self.existing_order

    def place_market_buy(self, inst_id, quantity, client_order_id):
        self.placed_orders.append((inst_id, quantity, client_order_id))
        if self.placement_result is not None:
            return self.placement_result
        return PlacementResult(True, "ORD1", client_order_id, None, None)


def snapshot(state, client_order_id="INITBUYtest", accumulated="0"):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", "BUY", Decimal("0"), Decimal("106.5"), Decimal(accumulated), state, "market", None, None)


# ---------------------------------------------------------------------------
# Cas nominal — patrimoine déjà suffisant
# ---------------------------------------------------------------------------

class TestAlreadyCovered:
    def test_spot_covered_skips_purchase_entirely(self):
        adapter = FakeAdapter(base="10", quote="1000")
        result = InitializationController().run(adapter, config())

        assert result.state == InitState.SPOT_READY
        assert adapter.placed_orders == []  # aucun achat déclenché


# ---------------------------------------------------------------------------
# Cas A — 0 spot, premier achat
# ---------------------------------------------------------------------------

class TestFirstPurchase:
    def test_needs_spot_triggers_single_market_buy(self):
        adapter = FakeAdapter(base="0", quote="1000")
        result = InitializationController().run(adapter, config())

        assert result.state == InitState.BUYING
        assert len(adapter.placed_orders) == 1

    def test_buy_quantity_is_exact_deficit_not_full_required_spot(self):
        """
        Achat = required_spot - base_available, jamais required_spot brut
        si une partie du spot est déjà détenue (cas C : XRP partiel).
        """
        adapter = FakeAdapter(base="0", quote="1000")
        result = InitializationController().run(adapter, config())
        report = InitializationController()._preflight(adapter, config())

        _, quantity, _ = adapter.placed_orders[0]
        assert quantity == report.required_spot  # base_available=0 => déficit = required_spot entier

    def test_partial_holdings_buy_only_the_deficit(self):
        """
        Le déficit est calculé sur le report FRAIS (recalculé avec les
        balances courantes), jamais sur un required_spot figé d'un appel
        antérieur — total_capital inclut base_total*p0, donc required_spot
        peut varier légèrement selon les soldes réels. C'est le
        comportement correct (toujours relire l'état frais), pas une
        instabilité.
        """
        report_source = FakeAdapter(base="0", quote="1000")
        full_report = InitializationController()._preflight(report_source, config())
        partial_base = full_report.required_spot / 2
        spent_quote = partial_base * config().p0

        adapter = FakeAdapter(base=str(partial_base), quote=str(Decimal("1000") - spent_quote))
        InitializationController().run(adapter, config())

        fresh_report = InitializationController()._preflight(adapter, config())
        _, quantity, _ = adapter.placed_orders[0]
        assert quantity == fresh_report.required_spot - partial_base

    def test_market_buy_uses_derived_deterministic_id(self):
        adapter = FakeAdapter(base="0", quote="1000")
        InitializationController().run(adapter, config())

        expected_id = derive_client_order_id(config(), adapter.instrument.tick_size, adapter.instrument.lot_size)
        _, _, used_id = adapter.placed_orders[0]
        assert used_id == expected_id


# ---------------------------------------------------------------------------
# Idempotence — zéro double achat après redémarrage
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_filled_order_found_skips_new_purchase(self):
        """Point 8 de la validation : crash après FILLED, avant GRID_PLACING."""
        adapter = FakeAdapter(base="0", quote="1000")
        # Simule un solde après un achat FILLED économiquement cohérent :
        # le quote dépensé correspond exactement au spot acquis à p0.
        full_report = InitializationController()._preflight(FakeAdapter(base="0", quote="1000"), config())
        spent_quote = full_report.required_spot * config().p0
        adapter.balances = Balances(full_report.required_spot, Decimal("1000") - spent_quote, full_report.required_spot, Decimal("1000") - spent_quote)
        adapter.existing_order = snapshot("filled", accumulated=str(full_report.required_spot))

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.SPOT_READY
        assert adapter.placed_orders == []  # AUCUN second achat

    def test_partially_filled_order_found_waits_no_new_purchase(self):
        """Point 9 : PARTIALLY_FILLED => attendre, jamais un second ordre."""
        adapter = FakeAdapter(base="0", quote="1000")
        adapter.existing_order = snapshot("partially_filled", accumulated="50")

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.PARTIALLY_FILLED
        assert adapter.placed_orders == []

    def test_live_order_found_waits_no_new_purchase(self):
        adapter = FakeAdapter(base="0", quote="1000")
        adapter.existing_order = snapshot("live")

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.BUYING
        assert adapter.placed_orders == []

    def test_order_not_found_rechecks_real_balances_before_deciding(self):
        """
        Point 10 : ordre introuvable => reconsulter les soldes réels
        avant toute nouvelle décision (jamais racheter aveuglément).
        Ici, les soldes réels (économiquement cohérents avec un achat
        déjà exécuté à p0) montrent spot_covered=True => aucun nouvel achat.
        """
        full_report = InitializationController()._preflight(FakeAdapter(base="0", quote="1000"), config())
        spent_quote = full_report.required_spot * config().p0
        adapter = FakeAdapter(base=str(full_report.required_spot), quote=str(Decimal("1000") - spent_quote))
        adapter.existing_order = None  # introuvable

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.SPOT_READY
        assert adapter.placed_orders == []

    def test_order_not_found_and_still_uncovered_buys_real_deficit(self):
        adapter = FakeAdapter(base="0", quote="1000")
        adapter.existing_order = None

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.BUYING
        assert len(adapter.placed_orders) == 1


# ---------------------------------------------------------------------------
# Cas D — capital insuffisant
# ---------------------------------------------------------------------------

class TestInsufficientCapital:
    def test_insufficient_capital_leads_to_error_not_purchase(self):
        adapter = FakeAdapter(base="0", quote="1", minimum="1")  # capital dérisoire
        result = InitializationController().run(adapter, config())

        assert result.state == InitState.ERROR
        assert adapter.placed_orders == []


# ---------------------------------------------------------------------------
# Rejet d'achat — jamais de retry automatique
# ---------------------------------------------------------------------------

class TestPurchaseRejection:
    def test_rejected_purchase_leads_to_error(self):
        adapter = FakeAdapter(base="0", quote="1000")
        adapter.placement_result = PlacementResult(False, None, "INITBUYx", "51008", "Insufficient balance")

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.ERROR
        assert "51008" in result.detail
        assert len(adapter.placed_orders) == 1  # une seule tentative, aucun retry automatique


# ---------------------------------------------------------------------------
# Interdiction des SELL avant SPOT_READY — invariant de sécurité
# ---------------------------------------------------------------------------

class TestNoSellBeforeSpotReady:
    def test_buying_state_never_reaches_grid_placing_implicitly(self):
        """
        InitializationController s'arrête strictement à SPOT_READY (ou
        ERROR/BUYING/PARTIALLY_FILLED) : il ne place jamais lui-même
        d'ordre de grille. Le placement des SELL est une responsabilité
        de l'appelant, déclenchée uniquement sur InitState.SPOT_READY.
        """
        adapter = FakeAdapter(base="0", quote="1000")
        result = InitializationController().run(adapter, config())

        assert result.state != InitState.GRID_PLACING
        assert result.state != InitState.ACTIVE

    def test_spot_ready_state_carries_confirmed_report_for_caller(self):
        """
        L'appelant doit pouvoir vérifier base_available >= required_spot
        depuis un rapport FRAIS avant de placer les SELL — jamais un
        rapport périmé d'avant l'achat.
        """
        full_report = InitializationController()._preflight(FakeAdapter(base="0", quote="1000"), config())
        spent_quote = full_report.required_spot * config().p0
        adapter = FakeAdapter(base=str(full_report.required_spot), quote=str(Decimal("1000") - spent_quote))

        result = InitializationController().run(adapter, config())

        assert result.state == InitState.SPOT_READY
        assert result.report is not None
        assert result.report.spot_covered is True
        assert result.report.balances.base_available >= result.report.required_spot
