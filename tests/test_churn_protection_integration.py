"""Tests d'intégration -- protection anti-churn N=3 câblée dans
select_exposed_orders (grid_order_exposure_controller.py) et dans
plan_exposure/run_exposure_cycle (grid_order_orchestrator.py), chantier
feature/churn-protection-n3.

Complète tests/test_churn_protection.py (logique pure) : ici, on vérifie
le CÂBLAGE -- guard de matérialisation, non-annulation des ordres déjà
ouverts, persistance à travers plusieurs cycles (restart simulé), et
absence de tout effet de bord sur les appelants qui ne fournissent pas
churn_state_path (rétrocompatibilité stricte).
"""

from decimal import Decimal

from cell import Cell, CellState
from cell_state_reconstruction import CellReconstruction
from churn_protection import ChurnProtectionEntry, load_churn_protection, save_churn_protection
from grid_order_exposure_controller import select_exposed_orders
from grid_order_orchestrator import plan_exposure, run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import Balances, CancelResult, FeeRates, Fill, InstrumentRules, OrderSnapshot, PlacementResult
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base="1000", quote="1000", tick="0.0001", lot="0.001", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.fills = ()
        self.placed_orders = []
        self.cancelled_orders = []

    def get_instrument(self, inst_id): return self.instrument
    def get_fee_rates(self, inst_id): return self.fees
    def get_balances(self, base_ccy, quote_ccy): return self.balances
    def list_open_orders(self, inst_id, client_order_id=None): return self.open_orders
    def list_fills(self, inst_id, order_id=None): return self.fills

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        from okx_spot_adapter import OkxApiError
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
    return CellReconstruction(Cell(gi, state), None, None, last_side, None)


def grid_fill(gi, side, ts, coid="GRID-TEST"):
    return Fill(
        trade_id=f"t{ts}", order_id=f"o{ts}", client_order_id=coid, inst_id="XRP-USDC",
        side=side, fill_price=Decimal(str(gi)), fill_quantity=Decimal("10"),
        accumulated_fill_quantity=Decimal("10"), order_state="", filled_at=ts,
        fee=Decimal("-0.001"), fee_currency="USDC", rebate=None, rebate_currency=None,
    )


# =============================================================================
# select_exposed_orders : le guard s'applique UNIQUEMENT au Niveau 1, jamais
# aux ordres déjà ouverts, jamais à une autre cellule.
# =============================================================================

class TestExposureControllerGuard:
    def test_protected_cell_excluded_from_new_materialization(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        target_gi = sorted(float(g) for g in report.trellis if float(g) > float(report.config.p0))[0]

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(report.config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != report.config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - target_gi) < 1e-9 else item
            for item in cell_states
        )

        decision_unprotected = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)
        assert ("BUY", target_gi, float(report.q)) in decision_unprotected.to_materialize

        decision_protected = select_exposed_orders(
            report, cell_states, (), k_buy=1, k_sell=1,
            churn_protected_gis=frozenset({target_gi}),
        )
        assert ("BUY", target_gi, float(report.q)) not in decision_protected.to_materialize

    def test_already_open_order_at_protected_gi_never_cancelled(self):
        """La protection interdit uniquement la création d'un NOUVEAU
        cycle -- un ordre déjà ouvert à un gi protégé reste ouvert."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        target_gi = sorted(float(g) for g in report.trellis if float(g) > float(report.config.p0))[0]

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(report.config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != report.config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - target_gi) < 1e-9 else item
            for item in cell_states
        )
        already_open = OrderSnapshot("ORD1", "GRID-EXISTING", "XRP-USDC", "BUY", Decimal(str(target_gi)), Decimal("10"), Decimal("0"), "live", "post_only", None, None)

        decision = select_exposed_orders(
            report, cell_states, (already_open,), k_buy=1, k_sell=1,
            churn_protected_gis=frozenset({target_gi}),
        )
        assert "GRID-EXISTING" not in decision.to_cancel

    def test_protecting_one_cell_never_affects_another(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        buy_gis = sorted(float(g) for g in report.trellis if float(g) < float(report.config.p0))
        target_gi, other_gi = buy_gis[-1], buy_gis[-2]

        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(report.config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != report.config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL")
            if abs(item.cell.gi - target_gi) < 1e-9 or abs(item.cell.gi - other_gi) < 1e-9
            else item
            for item in cell_states
        )

        decision = select_exposed_orders(
            report, cell_states, (), k_buy=2, k_sell=2,
            churn_protected_gis=frozenset({target_gi}),
        )
        assert ("BUY", other_gi, float(report.q)) in decision.to_materialize
        assert ("BUY", target_gi, float(report.q)) not in decision.to_materialize

    def test_default_frozenset_preserves_prior_behavior(self):
        """Aucun argument churn_protected_gis fourni -- comportement
        strictement identique à avant l'introduction du mécanisme."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        target_gi = sorted(float(g) for g in report.trellis if float(g) > float(report.config.p0))[0]
        cell_states = tuple(
            cs(float(g), CellState.SPOT_HELD if float(g) > float(report.config.p0) else CellState.CASH_HELD)
            for g in report.trellis if g != report.config.p0
        )
        cell_states = tuple(
            cs(item.cell.gi, CellState.CASH_HELD, last_side="SELL") if abs(item.cell.gi - target_gi) < 1e-9 else item
            for item in cell_states
        )
        decision = select_exposed_orders(report, cell_states, (), k_buy=1, k_sell=1)
        assert ("BUY", target_gi, float(report.q)) in decision.to_materialize


# =============================================================================
# plan_exposure / run_exposure_cycle : câblage, persistance, redémarrage.
# =============================================================================

class TestOrchestratorWiring:
    def test_churn_state_path_none_means_feature_disabled_no_file_written(self, tmp_path):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        would_be_path = str(tmp_path / "active_grid_state_XRP-USDC.json")

        plan_exposure(adapter, report, k_buy=1, k_sell=1)  # churn_state_path=None (défaut)

        import os
        assert not os.path.exists(would_be_path)

    def test_three_consecutive_c_cycles_then_fourth_blocked(self, tmp_path):
        """4 fills consécutifs sur le même gi (aucun autre gi jamais
        touché) forment 3 cycles C chevauchants (fill1->fill2, fill2->
        fill3, fill3->fill4) -- la cellule doit être protégée dès la
        fermeture du 3e, c'est-à-dire dès le 4e fill traité. Chaque appel
        simule un cycle bot successif (les fills s'accumulent
        progressivement, comme le ferait adapter.list_fills() en réel)."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(sorted(g for g in report.trellis if g > report.config.p0)[0])

        # Cycle bot 1 : SELL puis BUY sur gi -- 1er fill (aucun cycle
        # encore formé) puis 2e fill (1 cycle C : fill1->fill2).
        adapter.fills = (grid_fill(gi, "SELL", 100), grid_fill(gi, "BUY", 200))
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        state = load_churn_protection(path)
        assert state[gi].consecutive_c_cycles == 1
        assert state[gi].churn_protected is False

        # Cycle bot 2 : un fill de plus -- 2e cycle C (fill2->fill3).
        adapter.fills = adapter.fills + (grid_fill(gi, "SELL", 300),)
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        state = load_churn_protection(path)
        assert state[gi].consecutive_c_cycles == 2
        assert state[gi].churn_protected is False

        # Cycle bot 3 : un fill de plus -- 3e cycle C (fill3->fill4) ->
        # protection déclenchée exactement à ce fill.
        adapter.fills = adapter.fills + (grid_fill(gi, "BUY", 400),)
        plan = plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        state = load_churn_protection(path)
        assert state[gi].consecutive_c_cycles == 3
        assert state[gi].churn_protected is True
        assert state[gi].protected_at == 400

        # Aucun ordre de continuation de cycle n'est proposé pour gi tant
        # que protégée -- même si un 5e fill (hypothétique) arrivait, il
        # ne serait de toute façon jamais matérialisé par CE mécanisme.
        assert not any(side == "SELL" and abs(price - gi) < 1e-9 for side, price, _ in plan.decision.to_materialize)

    def test_progression_releases_cell_and_materialization_resumes(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(sorted(g for g in report.trellis if g > report.config.p0)[0])
        other_gi = float(sorted(g for g in report.trellis if g > report.config.p0)[1])

        adapter.fills = (
            grid_fill(gi, "SELL", 100), grid_fill(gi, "BUY", 200),
            grid_fill(gi, "SELL", 300), grid_fill(gi, "BUY", 400),
            grid_fill(gi, "SELL", 500), grid_fill(gi, "BUY", 600),
        )
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        assert load_churn_protection(path)[gi].churn_protected is True

        # Progression : un fill sur un AUTRE gi de la grille.
        adapter.fills = adapter.fills + (grid_fill(other_gi, "SELL", 700),)
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        state = load_churn_protection(path)
        assert state[gi].churn_protected is False
        assert state[gi].consecutive_c_cycles == 0

    def test_restart_preserves_protection(self, tmp_path):
        """Persistance à travers un redémarrage simulé : un état déjà
        protégé, écrit sur disque, doit être relu identique par un appel
        totalement indépendant (aucune mémoire en RAM partagée)."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        pre_restart_state = {1.05: ChurnProtectionEntry(3, True, 999, 999)}
        save_churn_protection(pre_restart_state, path)

        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        adapter.fills = ()
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)

        reloaded = load_churn_protection(path)
        assert reloaded[1.05].churn_protected is True
        assert reloaded[1.05].protected_at == 999

    def test_restart_then_progression_releases_correctly(self, tmp_path):
        """Redémarrage avec une cellule déjà protégée, PUIS progression
        réelle observée après le redémarrage -- doit être libérée."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(sorted(g for g in report.trellis if g > report.config.p0)[0])
        other_gi = float(sorted(g for g in report.trellis if g > report.config.p0)[1])

        save_churn_protection({gi: ChurnProtectionEntry(3, True, 999, 999)}, path)

        adapter.fills = (grid_fill(other_gi, "SELL", 1500),)
        plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        assert load_churn_protection(path)[gi].churn_protected is False

    def test_run_exposure_cycle_threads_churn_state_path(self, tmp_path):
        """run_exposure_cycle (pas seulement plan_exposure) doit aussi
        câbler churn_state_path -- utilisé réellement par run()."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        import os
        assert os.path.exists(path)

    def test_no_extra_order_solely_because_a_cell_was_released(self, tmp_path):
        """Une fois libérée, la cellule reçoit EXACTEMENT l'ordre de
        continuation de cycle qu'elle aurait reçu sans jamais avoir été
        protégée -- rien de plus, aucune duplication."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(sorted(g for g in report.trellis if g > report.config.p0)[0])
        other_gi = float(sorted(g for g in report.trellis if g > report.config.p0)[1])

        adapter.fills = (
            grid_fill(gi, "SELL", 100), grid_fill(gi, "BUY", 200),
            grid_fill(gi, "SELL", 300), grid_fill(gi, "BUY", 400),
            grid_fill(gi, "SELL", 500), grid_fill(gi, "BUY", 600),
            grid_fill(other_gi, "SELL", 700),
        )
        plan = plan_exposure(adapter, report, k_buy=1, k_sell=1, churn_state_path=path)
        assert load_churn_protection(path)[gi].churn_protected is False

        gi_orders = [i for i in plan.decision.to_materialize if abs(i[1] - gi) < 1e-9]
        assert len(gi_orders) == 1
        side, price, _ = gi_orders[0]
        assert side == "SELL"  # dernier fill = BUY -> repost SELL, même gi
        assert price == gi
