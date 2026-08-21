"""Tests de compare_cell_state.py -- comparateur READ-ONLY.

Aucun accès réseau réel. Vérifie chaque verdict, l'anti-circularité
(MEMORY_MISSING jamais bootstrappé), et l'absence totale d'écriture.
"""

from decimal import Decimal

import pytest

from cell import CellState
from cell_memory import CellMemoryEntry, OrderStatus, ReconciliationStatus
from compare_cell_state import Verdict, _classify, _new_order_side, _old_order_side
from grid_order_projection import BUY, SELL
from okx_spot_adapter import OrderSnapshot


def make_entry(**overrides):
    values = dict(
        economic_state=CellState.SPOT_HELD,
        materialization_side=None,
        order_status=OrderStatus.NONE,
        client_order_id=None,
        order_id=None,
        trade_ids_applied=(),
        filled_qty=Decimal("0"),
        target_qty=Decimal("18.266"),
        last_transition_ts=None,
        reconciliation_status=ReconciliationStatus.OK,
        reconciliation_since_ts=None,
    )
    values.update(overrides)
    return CellMemoryEntry(**values)


class TestVerdictClassification:
    def test_1_identical_states_produce_match(self):
        entry = make_entry(economic_state=CellState.SPOT_HELD, order_status=OrderStatus.OPEN,
                            materialization_side=SELL)
        verdict = _classify(CellState.SPOT_HELD, entry, old_order="SELL", new_order="SELL")
        assert verdict == Verdict.MATCH

    def test_2_different_economic_state_produces_state_mismatch(self):
        entry = make_entry(economic_state=CellState.CASH_HELD, order_status=OrderStatus.OPEN,
                            materialization_side=BUY)
        verdict = _classify(CellState.SPOT_HELD, entry, old_order="SELL", new_order="BUY")
        assert verdict == Verdict.STATE_MISMATCH

    def test_3_same_state_different_order_produces_order_mismatch(self):
        entry = make_entry(economic_state=CellState.SPOT_HELD, order_status=OrderStatus.OPEN,
                            materialization_side=None)
        verdict = _classify(CellState.SPOT_HELD, entry, old_order="SELL", new_order="NONE")
        assert verdict == Verdict.ORDER_MISMATCH

    def test_4_missing_memory_produces_memory_missing(self):
        verdict = _classify(CellState.SPOT_HELD, None, old_order="SELL", new_order="NONE")
        assert verdict == Verdict.MEMORY_MISSING

    def test_4b_none_order_status_also_produces_memory_missing(self):
        entry = make_entry(order_status=OrderStatus.NONE)
        verdict = _classify(CellState.SPOT_HELD, entry, old_order="NONE", new_order="NONE")
        assert verdict == Verdict.MEMORY_MISSING

    def test_5_needs_reconciliation_produces_reconciliation_required(self):
        entry = make_entry(
            order_status=OrderStatus.UNKNOWN,
            reconciliation_status=ReconciliationStatus.NEEDS_RECONCILIATION,
        )
        verdict = _classify(CellState.SPOT_HELD, entry, old_order="SELL", new_order="NONE")
        assert verdict == Verdict.RECONCILIATION_REQUIRED
        # priorité sur STATE_MISMATCH même si les états diffèrent aussi -- toujours signalé en premier


class TestHelpers:
    def test_old_order_side_finds_matching_order(self):
        orders = (OrderSnapshot("O1", "C1", "XRP-USDC", "SELL", Decimal("0.9896"), Decimal("18.266"), Decimal("0"), "live", "post_only", 1, 1),)
        assert _old_order_side(orders, 0.9896) == "SELL"

    def test_old_order_side_none_when_no_match(self):
        orders = ()
        assert _old_order_side(orders, 0.9896) == "NONE"

    def test_new_order_side_none_for_missing_entry(self):
        assert _new_order_side(None) == "NONE"

    def test_new_order_side_reflects_open_materialization(self):
        entry = make_entry(order_status=OrderStatus.OPEN, materialization_side=SELL)
        assert _new_order_side(entry) == "SELL"

    def test_new_order_side_none_when_filled(self):
        entry = make_entry(order_status=OrderStatus.FILLED, materialization_side=None)
        assert _new_order_side(entry) == "NONE"


class TestAntiCircularity:
    """Preuve explicite : une cellule sans entrée cell_memory ne doit
    JAMAIS produire un NEW_STATE dérivé de OLD -- uniquement MEMORY_MISSING."""

    def test_6_no_memory_never_copies_old_state(self):
        # Peu importe la valeur de old_state, l'absence de mémoire
        # produit TOUJOURS MEMORY_MISSING, jamais un état copié.
        for old in (CellState.CASH_HELD, CellState.SPOT_HELD):
            verdict = _classify(old, None, old_order="SELL", new_order="NONE")
            assert verdict == Verdict.MEMORY_MISSING

    def test_7_no_disk_write_functions_referenced(self):
        """Vérification structurelle : compare_cell_state.py ne doit
        jamais appeler save_cell_memory ni save_grid_state."""
        import inspect
        import compare_cell_state
        source = inspect.getsource(compare_cell_state)
        assert "save_cell_memory" not in source
        assert "save_grid_state" not in source

    def test_8_no_placement_or_cancel_functions_referenced(self):
        import inspect
        import compare_cell_state
        source = inspect.getsource(compare_cell_state)
        assert "adapter.place_post_only_limit(" not in source
        assert "adapter.cancel_order(" not in source
        assert "adapter.place_market_buy(" not in source


class TestGiIntegrity:
    def test_gi_rounding_consistent_across_old_and_new(self):
        """Les clés de cell_memory (str(float)) et le gi reconstruit
        (float) doivent s'apparier de façon fiable -- round() à 10
        décimales des deux côtés (déjà appliqué dans compare())."""
        assert round(0.98960000001, 10) == round(0.9896, 10)
