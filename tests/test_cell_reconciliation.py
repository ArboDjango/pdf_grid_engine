"""Tests de cell_reconciliation.py -- Phase 2-3.

Adaptateur simulé, aucun accès réseau réel. Couvre la liste minimale
demandée : transitions FULL/PARTIAL/CANCELED, plusieurs fills,
duplicate trade_id, erreur réseau, order not found, NEEDS_RECONCILIATION,
résolution ultérieure, seuil d'alerte, isolation d'une cellule bloquée.
"""

from decimal import Decimal

from cell import CellState
from cell_memory import CellMemoryEntry, OrderStatus, ReconciliationStatus, initial_entry
from cell_reconciliation import RECONCILIATION_ALERT_CYCLES, needs_operator_alert, reconcile_cell
from grid_order_projection import BUY, SELL
from okx_spot_adapter import Fill, OkxApiError, OrderSnapshot


def make_fill(trade_id, client_order_id, side, quantity, order_id="O1"):
    return Fill(trade_id, order_id, client_order_id, "XRP-USDC", side, Decimal("0.9896"), Decimal(quantity), None, "", 1000, None, None, None, None)


def open_order(client_order_id, side, price, state="live"):
    return OrderSnapshot("O1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("18.266"), Decimal("0"), state, "post_only", 1, 1)


class FakeAdapter:
    def __init__(self):
        self.fills = ()
        self.order_result = None  # OrderSnapshot, ou OkxApiError, ou Exception à lever
        self.list_fills_calls = 0
        self.get_order_calls = 0

    def list_fills(self, inst_id, *, order_id=None):
        self.list_fills_calls += 1
        return self.fills

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.get_order_calls += 1
        if isinstance(self.order_result, Exception):
            raise self.order_result
        if self.order_result is None:
            raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")
        return self.order_result


BUY_ROOT = "GRIDbuy0000000000000000000"
SELL_ROOT = "GRIDsell000000000000000000"


class TestNoOrderInProgress:
    def test_status_none_returns_unchanged(self):
        adapter = FakeAdapter()
        entry = initial_entry(CellState.CASH_HELD, Decimal("18.266"))
        result = reconcile_cell(adapter, "XRP-USDC", None, None, entry, now_ts=1000)
        assert result is entry  # objet identique, aucun appel réseau
        assert adapter.list_fills_calls == 0


class TestFullFill:
    def test_cash_held_buy_full_becomes_spot_held(self):
        adapter = FakeAdapter()
        adapter.fills = (make_fill("840664", BUY_ROOT, BUY, "18.266"),)
        entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.OPEN, BUY_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.SPOT_HELD
        assert result.order_status == OrderStatus.FILLED
        assert result.trade_ids_applied == ("840664",)
        assert result.last_transition_ts == 2000

    def test_spot_held_sell_full_becomes_cash_held(self):
        adapter = FakeAdapter()
        adapter.fills = (make_fill("840665", SELL_ROOT, SELL, "18.266"),)
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.CASH_HELD
        assert result.order_status == OrderStatus.FILLED


class TestPartialFill:
    def test_buy_partial_does_not_change_economic_state(self):
        adapter = FakeAdapter()
        adapter.fills = (make_fill("840664", BUY_ROOT, BUY, "10.000"),)
        entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.OPEN, BUY_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.CASH_HELD  # INCHANGÉ
        assert result.order_status == OrderStatus.PARTIAL
        assert result.filled_qty == Decimal("10.000")

    def test_sell_partial_does_not_change_economic_state(self):
        adapter = FakeAdapter()
        adapter.fills = (make_fill("840665", SELL_ROOT, SELL, "5.000"),)
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.SPOT_HELD  # INCHANGÉ

    def test_partial_then_completion_multiple_fills_summed(self):
        """Plusieurs fills successifs : jamais seulement le dernier."""
        adapter = FakeAdapter()
        entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.OPEN, BUY_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        adapter.fills = (make_fill("840664", BUY_ROOT, BUY, "10.000"),)
        entry = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=2000)
        assert entry.order_status == OrderStatus.PARTIAL

        adapter.fills = (
            make_fill("840664", BUY_ROOT, BUY, "10.000"),
            make_fill("840666", BUY_ROOT, BUY, "8.266"),
        )
        entry = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=3000)
        assert entry.economic_state == CellState.SPOT_HELD
        assert entry.filled_qty == Decimal("18.266")
        assert set(entry.trade_ids_applied) == {"840664", "840666"}


class TestDuplicateTradeId:
    def test_already_applied_trade_id_never_counted_twice(self):
        adapter = FakeAdapter()
        adapter.fills = (make_fill("840664", BUY_ROOT, BUY, "18.266"),)
        entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.PARTIAL, BUY_ROOT, "O1",
            ("840664",),  # déjà appliqué
            Decimal("18.266"), Decimal("18.266"), 1000, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=2000)
        # Le même trade_id ne doit pas être recompté : filled_qty ne double pas.
        assert result.filled_qty == Decimal("18.266")
        assert result.trade_ids_applied == ("840664",)


class TestCanceled:
    def test_buy_canceled_leaves_economic_state_unchanged(self):
        adapter = FakeAdapter()
        adapter.fills = ()
        adapter.order_result = open_order(BUY_ROOT, BUY, "0.9896", state="canceled")
        entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.OPEN, BUY_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", BUY_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.CASH_HELD  # INCHANGÉ
        assert result.order_status == OrderStatus.CANCELED

    def test_sell_canceled_leaves_economic_state_unchanged(self):
        adapter = FakeAdapter()
        adapter.fills = ()
        adapter.order_result = open_order(SELL_ROOT, SELL, "0.9896", state="canceled")
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result.economic_state == CellState.SPOT_HELD  # INCHANGÉ


class TestOrderNotFound:
    def test_no_fill_no_order_produces_needs_reconciliation(self):
        adapter = FakeAdapter()
        adapter.fills = ()
        adapter.order_result = None  # OkxApiError levée par get_order
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result.reconciliation_status == ReconciliationStatus.NEEDS_RECONCILIATION
        assert result.economic_state == CellState.SPOT_HELD  # INCHANGÉ
        assert result.reconciliation_since_ts == 2000  # première détection


class TestNetworkError:
    def test_network_failure_on_list_fills_leaves_entry_untouched(self):
        adapter = FakeAdapter()

        def raise_network_error(inst_id, *, order_id=None):
            raise ConnectionError("panne réseau réelle")
        adapter.list_fills = raise_network_error

        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result is entry  # objet strictement identique, rien modifié

    def test_network_failure_on_get_order_leaves_entry_untouched(self):
        adapter = FakeAdapter()
        adapter.fills = ()
        adapter.order_result = TimeoutError("panne réseau réelle")
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=2000)
        assert result is entry  # jamais NEEDS_RECONCILIATION suite à une panne réseau


class TestNeedsReconciliationResolutionLater:
    def test_needs_reconciliation_later_resolved_by_fill(self):
        adapter = FakeAdapter()
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.UNKNOWN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), 1000, ReconciliationStatus.NEEDS_RECONCILIATION, 1000,
        )
        adapter.fills = (make_fill("999999", SELL_ROOT, SELL, "18.266"),)
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=5000)
        assert result.reconciliation_status == ReconciliationStatus.OK
        assert result.economic_state == CellState.CASH_HELD

    def test_needs_reconciliation_preserves_original_since_ts_while_unresolved(self):
        adapter = FakeAdapter()
        adapter.fills = ()
        adapter.order_result = None
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.UNKNOWN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), 1000, ReconciliationStatus.NEEDS_RECONCILIATION, 1000,
        )
        result = reconcile_cell(adapter, "XRP-USDC", SELL_ROOT, "O1", entry, now_ts=5000)
        assert result.reconciliation_since_ts == 1000  # PAS 5000 -- première détection préservée


class TestAlertThreshold:
    def test_no_alert_before_threshold(self):
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.UNKNOWN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), 1000, ReconciliationStatus.NEEDS_RECONCILIATION,
            reconciliation_since_ts=1000,
        )
        # 5 cycles de 60s = 300s < 600s (seuil)
        assert needs_operator_alert(entry, now_ts=1000 + 300, cycle_interval_seconds=60) is False

    def test_alert_at_threshold(self):
        entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.UNKNOWN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), 1000, ReconciliationStatus.NEEDS_RECONCILIATION,
            reconciliation_since_ts=1000,
        )
        assert RECONCILIATION_ALERT_CYCLES == 10
        # 10 cycles de 60s = 600s
        assert needs_operator_alert(entry, now_ts=1000 + 600, cycle_interval_seconds=60) is True

    def test_no_alert_when_status_ok(self):
        entry = initial_entry(CellState.CASH_HELD, Decimal("18.266"))
        assert needs_operator_alert(entry, now_ts=999999, cycle_interval_seconds=60) is False


class TestCellIsolation:
    """Une cellule NEEDS_RECONCILIATION ne doit jamais empêcher la
    réconciliation indépendante des autres cellules -- vérifié en
    appelant reconcile_cell séparément sur plusieurs cellules, aucun
    état partagé entre les appels."""

    def test_blocked_cell_does_not_affect_reconciliation_of_another(self):
        adapter_blocked = FakeAdapter()
        adapter_blocked.fills = ()
        adapter_blocked.order_result = None
        blocked_entry = CellMemoryEntry(
            CellState.SPOT_HELD, SELL, OrderStatus.OPEN, SELL_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        blocked_result = reconcile_cell(adapter_blocked, "XRP-USDC", SELL_ROOT, "O1", blocked_entry, now_ts=2000)
        assert blocked_result.reconciliation_status == ReconciliationStatus.NEEDS_RECONCILIATION

        adapter_normal = FakeAdapter()
        adapter_normal.fills = (make_fill("111111", BUY_ROOT, BUY, "18.266"),)
        normal_entry = CellMemoryEntry(
            CellState.CASH_HELD, BUY, OrderStatus.OPEN, BUY_ROOT, "O1", (),
            Decimal("0"), Decimal("18.266"), None, ReconciliationStatus.OK, None,
        )
        normal_result = reconcile_cell(adapter_normal, "XRP-USDC", BUY_ROOT, "O1", normal_entry, now_ts=2000)
        assert normal_result.economic_state == CellState.SPOT_HELD
        assert normal_result.reconciliation_status == ReconciliationStatus.OK


class TestImmutabilityInvariants:
    def test_gi_never_a_field_of_cell_memory_entry(self):
        """gi n'est jamais un champ de CellMemoryEntry -- c'est
        exclusivement la clé du dictionnaire cell_memory (cell_memory.py)."""
        entry = initial_entry(CellState.CASH_HELD, Decimal("18.266"))
        assert not hasattr(entry, "gi")

    def test_preflight_config_never_imported_by_reconciliation(self):
        """cell_reconciliation.py ne doit jamais toucher PreflightConfig
        -- vérifié structurellement, pas seulement par convention."""
        import cell_reconciliation
        import inspect
        source = inspect.getsource(cell_reconciliation)
        assert "PreflightConfig" not in source
