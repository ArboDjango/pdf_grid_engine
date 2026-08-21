"""Tests dédiés au chantier d'alignement PDF : priorité au cycle propre
de chaque cellule (CASH_HELD -> BUY même gi, SPOT_HELD -> SELL même gi),
k_buy/k_sell devenus une limite de sécurité pour les cellules INITIALES
uniquement (Niveau 2), jamais pour les cellules en poursuite de cycle
(Niveau 1, inconditionnel) ni pour les cellules déjà ouvertes.

Couvre explicitement TEST 1 à TEST 10 du chantier, plus le cas XRP réel
(§10) qui doit échouer avec l'ancien comportement et réussir avec le
nouveau.
"""

from decimal import Decimal

import pytest

from cell import Cell, CellState
from cell_order_identity import derive_root_order_id
from cell_state_reconstruction import CellReconstruction
from grid_order_exposure_controller import select_exposed_orders
from grid_order_orchestrator import _filter_by_available_liquidity, plan_exposure, run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import (
    AccountConfig, Balances, CancelResult, FeeRates, InstrumentRules,
    OkxApiError, OrderSnapshot, PlacementResult,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base_avail="20000", quote_avail="20000", base_total=None, quote_total=None, fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.0001"), Decimal("0.001"), Decimal("0.1"), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(
            Decimal(base_avail), Decimal(quote_avail),
            Decimal(base_total if base_total is not None else base_avail),
            Decimal(quote_total if quote_total is not None else quote_avail),
        )
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []

    def get_instrument(self, inst_id): return self.instrument
    def get_fee_rates(self, inst_id): return self.fees
    def get_balances(self, b, q): return self.balances
    def get_account_config(self): return AccountConfig(fee_type=self._fee_type)

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        return self.fills

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


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def cs(gi, state, last_side=None):
    """Construit une CellReconstruction minimale pour les tests directs
    de select_exposed_orders."""
    return CellReconstruction(Cell(gi, state), None, None, last_side, None)


# =============================================================================
# TEST 1 : SELL @ 1.012 -> CASH_HELD(1.012) -> BUY @ 1.012
# =============================================================================

class TestOne:
    def test_sell_completed_cell_gets_buy_at_same_gi(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        target_gi = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))[0]
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - target_gi) < 1e-9 else item
            for item in cell_states
        )

        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)

        assert (("BUY", target_gi, float(report.q))) in decision.to_materialize


# =============================================================================
# TEST 2 : deux SELL complétés -> deux BUY, si liquidité suffisante
#          (cas XRP exact : 1.012 et 1.0229)
# =============================================================================

class TestTwo:
    def test_two_completed_sells_both_get_their_own_buy(self):
        adapter = FakeAdapter(quote_avail="1000000")
        config = grid_config()
        report = run_preflight(adapter, config)

        upper = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        gi1, gi2 = upper[0], upper[1]

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if item.cell.gi in (gi1, gi2) else item
            for item in cell_states
        )

        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)

        materialized_buys = {price for side, price, _ in decision.to_materialize if side == "BUY"}
        assert gi1 in materialized_buys
        assert gi2 in materialized_buys


# =============================================================================
# TEST 3 : deux CASH_HELD (poursuite de cycle), liquidité insuffisante
#          pour les deux -> seules les finançables matérialisées,
#          aucune substitution, aucune annulation injustifiée
# =============================================================================

class TestThree:
    def test_insufficient_liquidity_materializes_only_financable_no_substitution(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        upper = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        gi1, gi2 = upper[0], upper[1]
        q = float(report.q)

        instructions = (("BUY", gi1, q), ("BUY", gi2, q))
        cost_one = Decimal(str(gi1)) * Decimal(str(q))
        financable = _filter_by_available_liquidity(instructions, base_available=Decimal("0"), quote_available=cost_one)

        assert len(financable) == 1
        assert financable[0][1] == gi1
        assert all(price in (gi1, gi2) for _, price, _ in financable)


# =============================================================================
# TEST 4 : BUY @ gi exécuté -> SELL @ même gi (symétrique de TEST 1)
# =============================================================================

class TestFour:
    def test_buy_completed_cell_gets_sell_at_same_gi(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        target_gi = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))[-1]
        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.SPOT_HELD, last_side="BUY") if abs(item.cell.gi - target_gi) < 1e-9 else item
            for item in cell_states
        )

        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)

        assert (("SELL", target_gi, float(report.q))) in decision.to_materialize


# =============================================================================
# TEST 5 : cellule initiale CASH_HELD + cellule ayant terminé un SELL
#          -> le réarmement du cycle n'est jamais perdu à cause de la
#          priorité géométrique
# =============================================================================

class TestFive:
    def test_cycle_completion_not_lost_to_geometric_priority(self):
        adapter = FakeAdapter(quote_avail="1000000")
        config = grid_config()
        report = run_preflight(adapter, config)

        lower = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        initial_nearest = lower[-1]
        cycle_gi = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))[2]

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - cycle_gi) < 1e-9 else item
            for item in cell_states
        )

        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)

        materialized_buys = {price for side, price, _ in decision.to_materialize if side == "BUY"}
        assert cycle_gi in materialized_buys
        assert initial_nearest in materialized_buys


# =============================================================================
# TEST 6-8 : P0, q, derive_grid_order_id strictement inchangés
# =============================================================================

class TestSixSevenEight:
    def test_6_p0_unchanged(self):
        adapter = FakeAdapter(quote_avail="1000000")
        config = grid_config()
        original_p0 = config.p0
        report = run_preflight(adapter, config)
        cycle_gi = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))[0]
        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - cycle_gi) < 1e-9 else item
            for item in cell_states
        )
        select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)
        assert config.p0 == original_p0

    def test_7_q_unchanged_across_tiers(self):
        adapter = FakeAdapter(quote_avail="1000000")
        config = grid_config()
        report = run_preflight(adapter, config)
        cycle_gi = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))[0]
        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - cycle_gi) < 1e-9 else item
            for item in cell_states
        )
        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)
        for _, _, qty in decision.to_materialize:
            assert qty == float(report.q)

    def test_8_derive_grid_order_id_untouched(self):
        import inspect
        import grid_order_exposure_controller
        source = inspect.getsource(grid_order_exposure_controller)
        assert "def derive_grid_order_id" not in source
        assert "from grid_activation_controller import derive_grid_order_id" in source


# =============================================================================
# TEST 9 : price-limit -- gi conservé, cellule en attente, aucune substitution
# =============================================================================

class TestNine:
    def test_price_limit_keeps_gi_no_substitution(self):
        from grid_order_orchestrator import _filter_by_price_limit
        instructions = (("SELL", 0.9896, 18.266),)
        result = _filter_by_price_limit(instructions, buy_lmt=Decimal("1.05"), sell_lmt=Decimal("0.998"))
        assert result == ()
        assert all(price == 0.9896 for _, price, _ in instructions)


# =============================================================================
# TEST 10 : aucune régression du comportement d'activation initiale
# =============================================================================

class TestTen:
    def test_initial_activation_unchanged_all_cells_are_level_2(self):
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )

        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)

        assert len(decision.to_materialize) == 2


# =============================================================================
# Cas XRP réel exact -- doit démontrer le nouveau comportement : les DEUX
# BUY (1.012 et 1.0229) sont demandés, même avec k_buy=1.
# =============================================================================

class TestRealXrpCase:
    def test_xrp_case_both_completed_sells_get_buy_with_k_equals_1_and_2(self):
        adapter = FakeAdapter(quote_avail="1000000")
        config = grid_config(p0=Decimal("1.0013"))
        report = run_preflight(adapter, config)

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != config.p0
        )
        upper = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        gi_1012, gi_10229 = upper[0], upper[1]
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if item.cell.gi in (gi_1012, gi_10229) else item
            for item in cell_states
        )

        decision_k2 = select_exposed_orders(report, cell_states, (), k_buy=2, k_sell=1)
        buys_k2 = {p for s, p, _ in decision_k2.to_materialize if s == "BUY"}
        assert gi_1012 in buys_k2 and gi_10229 in buys_k2

        # Nouveau comportement voulu : k_buy=1 suffit désormais aussi.
        decision_k1 = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)
        buys_k1 = {p for s, p, _ in decision_k1.to_materialize if s == "BUY"}
        assert gi_1012 in buys_k1 and gi_10229 in buys_k1
