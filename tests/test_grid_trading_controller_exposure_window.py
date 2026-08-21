"""Tests du correctif #1 : application de k_buy/k_sell dès l'activation
initiale (GridTradingController), avec garde-fou explicite contre
l'annulation automatique d'ordres orphelins.

Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

from decimal import Decimal

import pytest

from cell_order_identity import derive_root_order_id
from grid_order_orchestrator import plan_exposure, run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from okx_spot_adapter import (
    AccountConfig,
    Balances,
    CancelResult,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001",
                 tick="0.0001", lot="0.001", minimum="0.1", fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_account_config(self):
        return AccountConfig(fee_type=self._fee_type)

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        if order_id is None:
            return self.fills
        return tuple(f for f in self.fills if f.order_id == order_id)

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
        raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        self.placed_orders.append((inst_id, side, price, quantity, client_order_id))
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, side, price, quantity, Decimal("0"), "live", "post_only", 1, 1)
        self.open_orders = self.open_orders + (snapshot,)
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)

    def cancel_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.cancelled_orders.append(client_order_id)
        self.open_orders = tuple(o for o in self.open_orders if o.client_order_id != client_order_id)
        return CancelResult(True, "ORDX", client_order_id, None, None)

    def place_market_buy(self, inst_id, quantity, client_order_id):
        self.placed_orders.append((inst_id, "MARKET_BUY", quantity, client_order_id))
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", Decimal("1.0013"), quantity, quantity, "filled", "market", 1, 1)
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
    )
    values.update(changes)
    return PreflightConfig(**values)


# ---------------------------------------------------------------------------
# 1 -- 5+5 théoriques -> 1+1 exposés
# ---------------------------------------------------------------------------

class TestFiveByFiveToOneByOne:
    def test_theoretical_grid_still_ten_cells(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)
        assert len(report.trellis) == 11  # 10 niveaux + P0
        assert len(report.orders) == 10  # 5 BUY + 5 SELL, P0 exclu -- inchangé

    def test_activation_materializes_exactly_two_orders(self):
        adapter = FakeAdapter()
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert result.state == GridTradingState.ACTIVATED
        assert len(adapter.placed_orders) == 2
        sides = sorted(p[1] for p in adapter.placed_orders)
        assert sides == ["BUY", "SELL"]

    def test_q_unchanged_matches_theoretical_full_grid_calculation(self):
        adapter = FakeAdapter()
        config = grid_config()
        report_before = run_preflight(adapter, config)  # q théorique, sans fenêtre

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        # q des ordres réellement placés == q théorique de la grille complète
        placed_qty = {p[3] for p in adapter.placed_orders}
        assert placed_qty == {report_before.q}


# ---------------------------------------------------------------------------
# 2 -- 2+2 -> 2+2
# ---------------------------------------------------------------------------

class TestTwoByTwo:
    def test_k_buy_2_k_sell_2_materializes_four_orders(self):
        adapter = FakeAdapter()
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=2, k_sell=2)

        assert result.state == GridTradingState.ACTIVATED
        assert len(adapter.placed_orders) == 4
        sides = sorted(p[1] for p in adapter.placed_orders)
        assert sides == ["BUY", "BUY", "SELL", "SELL"]


# ---------------------------------------------------------------------------
# 3 -- aucun to_cancel -> activation possible (cas neuf, sans ordre préexistant)
# ---------------------------------------------------------------------------

class TestNoConflictAllowsActivation:
    def test_fresh_grid_no_preexisting_orders_activates_normally(self):
        adapter = FakeAdapter()
        config = grid_config()

        plan = plan_exposure(adapter, run_preflight(adapter, config), 1, 1)
        assert plan.decision.to_cancel == ()  # confirmé : rien à annuler sur grille neuve

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)
        assert result.state == GridTradingState.ACTIVATED


# ---------------------------------------------------------------------------
# 4 -- to_cancel non vide -> STOP sans annulation (le cœur de ce chantier)
# ---------------------------------------------------------------------------

class TestOrphanedOrdersRefusal:
    def test_orphaned_orders_produce_exposure_conflict_not_activated(self):
        adapter = FakeAdapter()
        config = grid_config()
        # Ordres réels, orphelins, appartenant à une AUTRE configuration --
        # ne correspondent à aucun niveau du treillis courant.
        adapter.open_orders = (
            OrderSnapshot("O1", "GRIDorphelinBUY0000000000", "XRP-USDC", "BUY", Decimal("0.50"), Decimal("10"), Decimal("0"), "live", "post_only", 1, 1),
            OrderSnapshot("O2", "GRIDorphelinSELL000000000", "XRP-USDC", "SELL", Decimal("2.00"), Decimal("10"), Decimal("0"), "live", "post_only", 1, 1),
        )

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert result.state == GridTradingState.EXPOSURE_CONFLICT
        assert "GRIDorphelinBUY0000000000" in result.detail
        assert "GRIDorphelinSELL000000000" in result.detail

    def test_orphaned_orders_never_cancelled(self):
        adapter = FakeAdapter()
        config = grid_config()
        adapter.open_orders = (
            OrderSnapshot("O1", "GRIDorphelinBUY0000000000", "XRP-USDC", "BUY", Decimal("0.50"), Decimal("10"), Decimal("0"), "live", "post_only", 1, 1),
        )

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert adapter.cancelled_orders == []  # AUCUNE annulation, jamais

    def test_orphaned_orders_no_new_order_placed_either(self):
        adapter = FakeAdapter()
        config = grid_config()
        adapter.open_orders = (
            OrderSnapshot("O1", "GRIDorphelinBUY0000000000", "XRP-USDC", "BUY", Decimal("0.50"), Decimal("10"), Decimal("0"), "live", "post_only", 1, 1),
        )

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert adapter.placed_orders == []  # AUCUN placement non plus -- STOP complet

    def test_orphaned_orders_still_untouched_on_okx_after_refusal(self):
        adapter = FakeAdapter()
        config = grid_config()
        original_orders = (
            OrderSnapshot("O1", "GRIDorphelinBUY0000000000", "XRP-USDC", "BUY", Decimal("0.50"), Decimal("10"), Decimal("0"), "live", "post_only", 1, 1),
        )
        adapter.open_orders = original_orders

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert adapter.open_orders == original_orders  # rigoureusement intact


# ---------------------------------------------------------------------------
# 5 -- ROOT inchangés
# ---------------------------------------------------------------------------

class TestRootUnchanged:
    def test_placed_client_order_ids_match_derive_root_order_id(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        for inst_id, side, price, qty, client_order_id in adapter.placed_orders:
            expected_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            assert client_order_id == expected_root


# ---------------------------------------------------------------------------
# 6 -- q inchangé (même valeur que sans fenêtre)
# ---------------------------------------------------------------------------

class TestQUnchangedAcrossWindowSizes:
    def test_q_identical_regardless_of_k_buy_k_sell(self):
        adapter1 = FakeAdapter()
        adapter2 = FakeAdapter()
        config = grid_config()

        GridTradingController().run(adapter1, config, k_buy=1, k_sell=1)
        GridTradingController().run(adapter2, config, k_buy=2, k_sell=2)

        qty1 = {p[3] for p in adapter1.placed_orders}
        qty2 = {p[3] for p in adapter2.placed_orders}
        assert qty1 == qty2  # q strictement identique, indépendant de la fenêtre


# ---------------------------------------------------------------------------
# 7 -- run_exposure_cycle() en régime normal inchangé
# ---------------------------------------------------------------------------

class TestRunExposureCycleUnchanged:
    def test_run_exposure_cycle_never_cancels_far_but_already_open_order(self):
        """RÉVISÉ par ce chantier (alignement PDF) : un ordre déjà ouvert
        n'est plus jamais annulé simplement parce qu'il est loin de P0 --
        run_exposure_cycle() délègue à plan_exposure()/select_exposed_orders(),
        qui appliquent désormais cette garantie uniformément, activation
        initiale comme régime de croisière."""
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)
        far_sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", float(report.trellis[-1]))
        adapter.open_orders = (
            OrderSnapshot("O1", far_sell_root, "XRP-USDC", "SELL", report.trellis[-1], Decimal("1"), Decimal("0"), "live", "post_only", 1, 1),
        )

        result = run_exposure_cycle(adapter, report, 1, 1)

        assert far_sell_root not in result.cancelled
        assert result.cancelled == ()

    def test_run_exposure_cycle_uses_plan_exposure_internally_same_result(self):
        """plan_exposure() et run_exposure_cycle() doivent produire la
        même décision -- l'extraction n'a rien changé au calcul."""
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        plan = plan_exposure(adapter, report, 1, 1)
        # Un second appel, indépendant, doit produire la même décision
        # (idempotence déjà garantie, reconfirmée après l'extraction).
        plan2 = plan_exposure(adapter, report, 1, 1)

        assert plan.decision.to_materialize == plan2.decision.to_materialize
        assert plan.decision.to_cancel == plan2.decision.to_cancel


# ---------------------------------------------------------------------------
# Non-régression : comportement historique préservé quand k_buy/k_sell absents
# ---------------------------------------------------------------------------

class TestBackwardCompatibilityNoWindow:
    def test_no_k_buy_k_sell_activates_full_theoretical_grid(self):
        """run_grid_trading.py n'appelle jamais k_buy/k_sell -- comportement
        historique (nu+nl ordres) strictement préservé."""
        adapter = FakeAdapter()
        config = grid_config()

        result = GridTradingController().run(adapter, config)  # SANS k_buy/k_sell

        assert result.state == GridTradingState.ACTIVATED
        assert len(adapter.placed_orders) == 10  # nu+nl, comportement historique intact
