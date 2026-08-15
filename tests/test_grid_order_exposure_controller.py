"""Tests pour grid_order_exposure_controller.select_exposed_orders.

Fonction pure : aucun appel réseau, cells et open_orders fournis
directement par les tests.
"""

from decimal import Decimal

import pytest

from cell import Cell, CellState
from grid_activation_controller import derive_grid_order_id
from grid_order_exposure_controller import ExposureDecision, select_exposed_orders
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import Balances, FeeRates, InstrumentRules, OrderSnapshot
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances


def grid_config(**changes):
    # nu=3, nl=3 : plusieurs niveaux de chaque côté, pour tester "les k plus proches".
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("115"), gll=Decimal("85"), nu=3, nl=3,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def open_order_snapshot(client_order_id, side="BUY", price="90.0"):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("0"), "live", "post_only", None, None)


def _cells_all_initial(report):
    """Reconstruit les cellules dans leur état initial pur (aucun historique),
    dans l'ordre attendu par select_exposed_orders (treillis, P0 exclu)."""
    p0 = float(report.config.p0)
    cells = []
    for gi_decimal in report.trellis:
        gi = float(gi_decimal)
        if gi == p0:
            continue
        cells.append(Cell(gi, CellState.SPOT_HELD if gi > p0 else CellState.CASH_HELD))
    return tuple(cells)


class TestKeepKNearestToP0:
    def test_keep_k_nearest_to_p0_selects_correct_levels_each_side(self):
        """
        nu=3, nl=3 (6 cellules), k_buy=1, k_sell=1 : seuls les niveaux
        immédiatement adjacents à P0 (index nl-1 et nl+1... en tenant
        compte de l'exclusion de P0) doivent être matérialisés.
        """
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        decision = select_exposed_orders(report, cells, open_orders=(), k_buy=1, k_sell=1)

        assert len(decision.to_materialize) == 2
        sides = {side for side, _, _ in decision.to_materialize}
        assert sides == {"BUY", "SELL"}
        # Le BUY matérialisé doit être le niveau juste en-dessous de P0 :
        # le plus proche par index, donc le plus grand des niveaux CASH_HELD.
        buy_prices = [price for side, price, _ in decision.to_materialize if side == "BUY"]
        cash_prices = sorted(float(g) for g in report.trellis if float(g) < float(report.config.p0))
        assert buy_prices[0] == cash_prices[-1]  # le plus proche de P0 côté BUY

    def test_k_buy_2_exposes_two_nearest_buy_levels(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        decision = select_exposed_orders(report, cells, open_orders=(), k_buy=2, k_sell=1)

        buy_count = sum(1 for side, _, _ in decision.to_materialize if side == "BUY")
        assert buy_count == 2

    def test_k_zero_exposes_nothing_on_that_side(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        decision = select_exposed_orders(report, cells, open_orders=(), k_buy=0, k_sell=1)

        assert all(side != "BUY" for side, _, _ in decision.to_materialize)

    def test_selection_never_uses_market_price_only_index_distance(self):
        """
        Vérification structurelle : select_exposed_orders ne prend aucun
        paramètre de prix de marché courant — seule la fonction elle-même
        et sa signature en attestent.
        """
        import inspect
        sig = inspect.signature(select_exposed_orders)
        assert "current_price" not in sig.parameters
        assert "market_price" not in sig.parameters


class TestExcessBeyondKIsCancelled:
    def test_excess_beyond_k_is_designated_for_cancellation(self):
        """
        Un ordre ouvert au-delà des k plus proches doit être désigné à
        l'annulation — jamais silencieusement ignoré.
        """
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        far_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        far_gi = far_cash_prices[0]  # le plus ÉLOIGNÉ de P0 côté BUY
        far_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", far_gi)
        open_orders = (open_order_snapshot(far_id, side="BUY", price=str(far_gi)),)

        decision = select_exposed_orders(report, cells, open_orders, k_buy=1, k_sell=1)

        assert far_id in decision.to_cancel

    def test_kept_order_never_appears_in_cancel_list(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        near_gi = near_cash_prices[-1]  # le plus proche de P0
        near_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", near_gi)
        open_orders = (open_order_snapshot(near_id, side="BUY", price=str(near_gi)),)

        decision = select_exposed_orders(report, cells, open_orders, k_buy=1, k_sell=1)

        assert near_id not in decision.to_cancel
        # Déjà ouvert et conservé : pas de nouvel ordre à matérialiser pour ce niveau.
        assert not any(price == near_gi for _, price, _ in decision.to_materialize)


class TestNoDuplicateOrders:
    def test_no_duplicate_orders_when_already_open(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        near_gi = near_cash_prices[-1]
        near_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", near_gi)
        open_orders = (open_order_snapshot(near_id, side="BUY", price=str(near_gi)),)

        decision = select_exposed_orders(report, cells, open_orders, k_buy=1, k_sell=1)

        # Le niveau déjà ouvert n'est jamais redemandé en matérialisation.
        assert near_id not in [
            derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            for side, price, _ in decision.to_materialize
        ]


class TestP0NeverACell:
    def test_p0_never_appears_in_materialized_or_cancelled(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)
        p0 = float(report.config.p0)

        decision = select_exposed_orders(report, cells, open_orders=(), k_buy=3, k_sell=3)

        assert all(price != p0 for _, price, _ in decision.to_materialize)


class TestValidation:
    def test_negative_k_raises(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        cells = _cells_all_initial(report)

        with pytest.raises(ValueError):
            select_exposed_orders(report, cells, open_orders=(), k_buy=-1, k_sell=1)

    def test_mismatched_cells_length_raises(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        with pytest.raises(ValueError):
            select_exposed_orders(report, cells=(), open_orders=(), k_buy=1, k_sell=1)

from cell_order_identity import derive_next_order_id


def test_open_non_root_occurrence_is_preserved():
    """
    Une occurrence non-ROOT déjà ouverte à un gi exposé doit être
    conservée par la politique d'exposition.

    Exemple :
        BUY#1 rempli
        -> SELL#1 ouvert

    SELL#1 ne doit ni être annulé, ni provoquer un doublon.
    """
    adapter = FakeAdapter(base="1000", quote="1000")
    report = run_preflight(adapter, grid_config())

    cells = _cells_all_initial(report)

    near_cash_prices = sorted(
        float(g)
        for g in report.trellis
        if float(g) < float(report.config.p0)
    )
    gi = near_cash_prices[-1]

    buy_root = derive_grid_order_id(
        report.config,
        report.instrument.tick_size,
        report.instrument.lot_size,
        "BUY",
        gi,
    )

    sell_occurrence = derive_next_order_id(
        report.config,
        report.instrument.tick_size,
        report.instrument.lot_size,
        "SELL",
        gi,
        buy_root,
    )

    open_orders = (
        open_order_snapshot(
            sell_occurrence,
            side="SELL",
            price=str(gi),
        ),
    )

    # La cellule doit être SPOT_HELD pour que SELL soit l'ordre exposé.
    cells = tuple(
        Cell(
            c.gi,
            CellState.SPOT_HELD if c.gi == gi else c.state,
        )
        for c in cells
    )

    decision = select_exposed_orders(
        report,
        cells,
        open_orders,
        k_buy=1,
        k_sell=1,
    )

    assert sell_occurrence not in decision.to_cancel

    # Aucun nouvel SELL au même gi.
    assert not any(
        side == "SELL" and price == gi
        for side, price, _ in decision.to_materialize
    )
