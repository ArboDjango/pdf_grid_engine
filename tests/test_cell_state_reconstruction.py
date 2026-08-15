"""Tests pour cell_state_reconstruction.reconstruct_cells.

Aucun appel réseau réel : adaptateur simulé, cohérent avec le patron déjà
établi dans tests/test_grid_activation_controller.py, étendu ici avec
cancel_order (absent du FakeAdapter existant, nécessaire pour ce chantier).
"""

from decimal import Decimal

from cell import CellState
from cell_state_reconstruction import reconstruct_cells
from grid_activation_controller import derive_grid_order_id
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import (
    Balances,
    CancelResult,
    FeeRates,
    Fill,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    """Simule OkxSpotAdapter, étendu avec cancel_order pour ce chantier."""

    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.placement_results = {}
        self.cancelled_orders = []
        self.cancel_results = {}
        self.fills = ()

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
        raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        self.placed_orders.append((inst_id, side, price, quantity, client_order_id))
        if client_order_id in self.placement_results:
            return self.placement_results[client_order_id]
        return PlacementResult(True, "ORDX", client_order_id, None, None)

    def cancel_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.cancelled_orders.append((inst_id, client_order_id))
        if client_order_id in self.cancel_results:
            return self.cancel_results[client_order_id]
        return CancelResult(True, "ORDX", client_order_id, None, None)

    def list_fills(self, inst_id, *, order_id=None):
        if order_id is None:
            return self.fills
        return tuple(
            fill for fill in self.fills
            if fill.order_id == order_id
        )

def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("110"), gll=Decimal("90"), nu=2, nl=2,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def open_order_snapshot(client_order_id, side="BUY", price="90.0", state="live"):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("0"), state, "post_only", None, None)


def filled_snapshot(client_order_id, side="BUY", price="90.0", updated_at=100):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("1.4"), "filled", "post_only", None, updated_at)


class TestReconstructCellsNeverTouched:
    def test_never_touched_cell_falls_back_to_instantiator_initial_state(self):
        """
        Aucun ordre ouvert, aucun historique : l'état retombe sur la règle
        de CellInstantiator (gi>P0 -> SPOT_HELD, gi<P0 -> CASH_HELD).
        """
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        cells = reconstruct_cells(adapter, report, open_orders=())

        # nu=2 niveaux au-dessus de P0 (SPOT_HELD), nl=2 en-dessous (CASH_HELD)
        spot_count = sum(1 for c in cells if c.state is CellState.SPOT_HELD)
        cash_count = sum(1 for c in cells if c.state is CellState.CASH_HELD)
        assert spot_count == 2
        assert cash_count == 2

    def test_p0_never_appears_as_a_cell(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        cells = reconstruct_cells(adapter, report, open_orders=())

        assert len(cells) == len(report.trellis) - 1  # P0 exclu
        assert report.config.p0 not in [Decimal(str(c.gi)) for c in cells]


class TestReconstructCellsFromOpenOrders:
    def test_open_buy_order_means_cash_held(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])  # premier niveau, en-dessous de P0
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        adapter.open_orders = (open_order_snapshot(buy_id, side="BUY", price=str(gi)),)

        cells = reconstruct_cells(adapter, report, adapter.open_orders)

        matching = [c for c in cells if c.gi == gi]
        assert len(matching) == 1
        assert matching[0].state is CellState.CASH_HELD

    def test_open_sell_order_means_spot_held(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[-1])  # dernier niveau, au-dessus de P0
        sell_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "SELL", gi)
        adapter.open_orders = (open_order_snapshot(sell_id, side="SELL", price=str(gi)),)

        cells = reconstruct_cells(adapter, report, adapter.open_orders)

        matching = [c for c in cells if c.gi == gi]
        assert matching[0].state is CellState.SPOT_HELD

    def test_open_order_check_uses_provided_open_orders_no_extra_network_call(self):
        """
        Un niveau déjà présent dans open_orders (fourni par l'appelant) ne
        doit déclencher aucun appel get_order supplémentaire.
        """
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        adapter.open_orders = (open_order_snapshot(buy_id, side="BUY", price=str(gi)),)

        calls = []
        original = adapter.get_order
        adapter.get_order = lambda *a, **k: (calls.append(k), original(*a, **k))[1]

        reconstruct_cells(adapter, report, adapter.open_orders)

        # Aucun appel get_order pour le niveau déjà dans open_orders — mais
        # les 3 autres niveaux (hors fenêtre) déclenchent bien 2 appels
        # chacun (BUY puis SELL) : vérifie qu'aucun de ces appels ne porte
        # sur buy_id (déjà résolu sans réseau).
        called_ids = {k.get("client_order_id") for k in calls}
        assert buy_id not in called_ids


class TestReconstructCellsFromHistory:
    def test_filled_buy_in_history_means_spot_held(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        adapter.get_order_results[buy_id] = filled_snapshot(buy_id, side="BUY", price=str(gi))

        cells = reconstruct_cells(adapter, report, open_orders=())

        matching = [c for c in cells if c.gi == gi]
        assert matching[0].state is CellState.SPOT_HELD

    def test_filled_sell_in_history_means_cash_held(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[-1])
        sell_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "SELL", gi)
        adapter.get_order_results[sell_id] = filled_snapshot(sell_id, side="SELL", price=str(gi))

        cells = reconstruct_cells(adapter, report, open_orders=())

        matching = [c for c in cells if c.gi == gi]
        assert matching[0].state is CellState.CASH_HELD

    def test_both_sides_filled_uses_most_recent(self):
        """
        Les deux côtés ont déjà été remplis au cours de la vie de la
        grille : le plus récent (updated_at le plus grand) détermine
        l'état courant.
        """
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        sell_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "SELL", gi)
        adapter.get_order_results[buy_id] = filled_snapshot(buy_id, side="BUY", price=str(gi), updated_at=100)
        adapter.get_order_results[sell_id] = filled_snapshot(sell_id, side="SELL", price=str(gi), updated_at=200)

        cells = reconstruct_cells(adapter, report, open_orders=())

        matching = [c for c in cells if c.gi == gi]
        # sell plus récent (200 > 100) -> dernier événement = vente -> cash
        assert matching[0].state is CellState.CASH_HELD

    def test_non_filled_history_falls_back_to_initial_state(self):
        """Un ordre trouvé mais canceled/expired ne compte pas comme un basculement."""
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])  # < P0 -> CASH_HELD attendu par défaut
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        adapter.get_order_results[buy_id] = OrderSnapshot("ORD1", buy_id, "XRP-USDC", "BUY", Decimal(str(gi)), Decimal("1.4"), Decimal("0"), "canceled", "post_only", None, None)

        cells = reconstruct_cells(adapter, report, open_orders=())

        matching = [c for c in cells if c.gi == gi]
        assert matching[0].state is CellState.CASH_HELD  # état initial, pas un basculement

from cell_order_identity import (
    derive_root_order_id,
    derive_next_order_id,
)


def test_occurrence_chain_buy_sell_buy_same_gi():
    """La reconstruction suit BUY#1 -> SELL#1 -> BUY#2."""

    adapter = FakeAdapter(base="1000", quote="1000")
    report = run_preflight(adapter, grid_config())
    gi = 95.0
    tick = report.instrument.tick_size
    lot = report.instrument.lot_size

    buy_1 = derive_root_order_id(
        report.config,
        tick,
        lot,
        "BUY",
        gi,
    )

    sell_1 = derive_next_order_id(
        report.config,
        tick,
        lot,
        "SELL",
        gi,
        buy_1,
    )

    buy_2 = derive_next_order_id(
        report.config,
        tick,
        lot,
        "BUY",
        gi,
        sell_1,
    )

    adapter.fills = (
        Fill(
            trade_id="T1",
            order_id="O1",
            client_order_id=buy_1,
            inst_id=report.config.inst_id,
            side="BUY",
            fill_price=Decimal(str(gi)),
            fill_quantity=Decimal("1"),
            accumulated_fill_quantity=Decimal("1"),
            order_state="filled",
            filled_at=100,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
        Fill(
            trade_id="T2",
            order_id="O2",
            client_order_id=sell_1,
            inst_id=report.config.inst_id,
            side="SELL",
            fill_price=Decimal(str(gi)),
            fill_quantity=Decimal("1"),
            accumulated_fill_quantity=Decimal("1"),
            order_state="filled",
            filled_at=200,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
        Fill(
            trade_id="T3",
            order_id="O3",
            client_order_id=buy_2,
            inst_id=report.config.inst_id,
            side="BUY",
            fill_price=Decimal(str(gi)),
            fill_quantity=Decimal("1"),
            accumulated_fill_quantity=Decimal("1"),
            order_state="filled",
            filled_at=300,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
    )

    states = reconstruct_cell_states(
        adapter,
        report,
        open_orders=(),
    )

    state = next(item for item in states if item.cell.gi == gi)

    assert state.cell.state is CellState.SPOT_HELD
    assert state.last_client_order_id == buy_2
    assert state.last_side == "BUY"
    assert state.last_filled_at == 300

from cell_order_identity import derive_root_order_id, derive_next_order_id
from cell_state_reconstruction import reconstruct_cell_states


def test_buy_sell_buy_same_gi_reconstructs_latest_occurrence():
    """
    Vérifie le cycle économique complet :

        BUY #1 @ gi
             ↓
        SELL #1 @ gi
             ↓
        BUY #2 @ gi

    La reconstruction doit retenir BUY #2 comme dernière occurrence,
    donc la cellule doit être SPOT_HELD.
    """

    adapter = FakeAdapter(base="1000", quote="1000")
    report = run_preflight(adapter, grid_config())

    gi = 95.0

    tick = report.instrument.tick_size
    lot = report.instrument.lot_size

    buy_1 = derive_root_order_id(
        report.config,
        tick,
        lot,
        "BUY",
        gi,
    )

    sell_1 = derive_next_order_id(
        report.config,
        tick,
        lot,
        "SELL",
        gi,
        buy_1,
    )

    buy_2 = derive_next_order_id(
        report.config,
        tick,
        lot,
        "BUY",
        gi,
        sell_1,
    )

    adapter.fills = (
        Fill(
            trade_id="T1",
            order_id="O1",
            client_order_id=buy_1,
            inst_id="XRP-USDC",
            side="BUY",
            fill_price=Decimal("95.0"),
            fill_quantity=Decimal("1.0"),
            accumulated_fill_quantity=Decimal("1.0"),
            order_state="filled",
            filled_at=100,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
        Fill(
            trade_id="T2",
            order_id="O2",
            client_order_id=sell_1,
            inst_id="XRP-USDC",
            side="SELL",
            fill_price=Decimal("95.0"),
            fill_quantity=Decimal("1.0"),
            accumulated_fill_quantity=Decimal("1.0"),
            order_state="filled",
            filled_at=200,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
        Fill(
            trade_id="T3",
            order_id="O3",
            client_order_id=buy_2,
            inst_id="XRP-USDC",
            side="BUY",
            fill_price=Decimal("95.0"),
            fill_quantity=Decimal("1.0"),
            accumulated_fill_quantity=Decimal("1.0"),
            order_state="filled",
            filled_at=300,
            fee=None,
            fee_currency=None,
            rebate=None,
            rebate_currency=None,
        ),
    )

    states = reconstruct_cell_states(
        adapter,
        report,
        open_orders=(),
    )

    state = next(
        item
        for item in states
        if item.cell.gi == gi
    )

    assert state.cell.state is CellState.SPOT_HELD
    assert state.last_client_order_id == buy_2
    assert state.last_side == "BUY"
    assert state.last_filled_at == 300
