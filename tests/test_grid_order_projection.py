"""Tests CD-004 — projection de grille en instructions d'ordres.

Les instructions sont volontairement de simples tuples ``(side, price, q)`` :
CD-004 ne crée ni ordre exchange ni structure d'ordre persistante.
"""

import pytest

from cell import Cell, CellState
from grid_order_projection import BUY, SELL, GridOrderProjection


TRELLIS = (90.0, 95.0, 100.0, 105.0, 110.0)
P0 = 100.0
Q = 7.25


@pytest.fixture
def projection() -> GridOrderProjection:
    return GridOrderProjection()


@pytest.fixture
def cells() -> tuple[Cell, ...]:
    return (
        Cell(90.0, CellState.CASH_HELD),
        Cell(95.0, CellState.CASH_HELD),
        Cell(105.0, CellState.SPOT_HELD),
        Cell(110.0, CellState.SPOT_HELD),
    )


class TestInitialProjection:
    def test_cash_held_projects_buy_at_cell_level(self, projection, cells):
        orders = projection.project_initial(cells, Q)
        assert (BUY, 90.0, Q) in orders
        assert (BUY, 95.0, Q) in orders

    def test_spot_held_projects_sell_at_cell_level(self, projection, cells):
        orders = projection.project_initial(cells, Q)
        assert (SELL, 105.0, Q) in orders
        assert (SELL, 110.0, Q) in orders

    def test_no_initial_order_is_projected_at_p0(self, projection, cells):
        orders = projection.project_initial(cells, Q)
        assert all(price != P0 for _, price, _ in orders)

    def test_q_is_preserved_exactly(self, projection, cells):
        assert all(quantity == Q for _, _, quantity in projection.project_initial(cells, Q))

    def test_cells_are_not_modified(self, projection, cells):
        before = tuple((cell.gi, cell.state) for cell in cells)
        projection.project_initial(cells, Q)
        assert tuple((cell.gi, cell.state) for cell in cells) == before


class TestRearmingAfterCompletedBuy:
    def test_buy_projects_sell_at_immediately_higher_level(self, projection):
        assert projection.after_buy_completed(TRELLIS, 95.0, Q) == ((SELL, 100.0, Q),)

    def test_buy_below_p0_projects_sell_at_p0(self, projection):
        assert projection.after_buy_completed(TRELLIS, 95.0, Q) == ((SELL, P0, Q),)

    def test_buy_at_p0_projects_sell_at_first_higher_level(self, projection):
        assert projection.after_buy_completed(TRELLIS, P0, Q) == ((SELL, 105.0, Q),)

    def test_buy_at_gll_projects_sell_at_next_level(self, projection):
        assert projection.after_buy_completed(TRELLIS, 90.0, Q) == ((SELL, 95.0, Q),)

    def test_buy_at_gul_has_no_projection(self, projection):
        assert projection.after_buy_completed(TRELLIS, 110.0, Q) == ()


class TestRearmingAfterCompletedSell:
    def test_sell_projects_buy_at_immediately_lower_level(self, projection):
        assert projection.after_sell_completed(TRELLIS, 105.0, Q) == ((BUY, 100.0, Q),)

    def test_sell_above_p0_projects_buy_at_p0(self, projection):
        assert projection.after_sell_completed(TRELLIS, 105.0, Q) == ((BUY, P0, Q),)

    def test_sell_at_p0_projects_buy_at_first_lower_level(self, projection):
        assert projection.after_sell_completed(TRELLIS, P0, Q) == ((BUY, 95.0, Q),)

    def test_sell_at_gul_projects_buy_at_previous_level(self, projection):
        assert projection.after_sell_completed(TRELLIS, 110.0, Q) == ((BUY, 105.0, Q),)

    def test_sell_at_gll_has_no_projection(self, projection):
        assert projection.after_sell_completed(TRELLIS, 90.0, Q) == ()


class TestPartialAndMultipleFills:
    def test_partial_fill_has_no_projection(self, projection):
        assert projection.after_partial_fill() == ()

    def test_successive_fills_rearm_level_by_level(self, projection):
        first = projection.after_buy_completed(TRELLIS, 95.0, Q)
        second = projection.after_buy_completed(TRELLIS, 90.0, Q)
        assert first == ((SELL, 100.0, Q),)
        assert second == ((SELL, 95.0, Q),)

    def test_multi_level_crossing_does_not_create_catch_up_orders(self, projection, cells):
        initial = projection.project_initial(cells, Q)
        first = projection.after_sell_completed(TRELLIS, 105.0, Q)
        second = projection.after_sell_completed(TRELLIS, 110.0, Q)
        assert initial == ((BUY, 90.0, Q), (BUY, 95.0, Q), (SELL, 105.0, Q), (SELL, 110.0, Q))
        assert first == ((BUY, 100.0, Q),)
        assert second == ((BUY, 105.0, Q),)


class TestReconciliation:
    def test_does_not_duplicate_successor_already_open(self, projection):
        open_orders = ((SELL, 100.0, Q),)
        assert projection.after_buy_completed(TRELLIS, 95.0, Q, open_orders) == ()

    def test_projects_successor_when_it_is_absent_from_open_orders(self, projection):
        open_orders = ((BUY, 90.0, Q),)
        assert projection.after_buy_completed(TRELLIS, 95.0, Q, open_orders) == ((SELL, 100.0, Q),)


class TestTrellisIntegrity:
    def test_rearming_does_not_modify_trellis(self, projection):
        trellis = list(TRELLIS)
        before = tuple(trellis)
        projection.after_sell_completed(trellis, 105.0, Q)
        assert tuple(trellis) == before
