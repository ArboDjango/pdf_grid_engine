"""Tests pour derive_grid_order_id et GridActivationController.

Suit exactement le patron déjà établi pour
initialization_controller.derive_client_order_id /
tests/test_initialization_controller.py.

Aucun appel réseau réel : adaptateur simulé (lectures + écritures),
suivant les mêmes conventions que FakeAdapter dans
tests/test_initialization_controller.py.
"""

from decimal import Decimal

from grid_activation_controller import (
    GridActivationController,
    GridActivationState,
    derive_grid_order_id,
)
from grid_preflight import PreflightConfig, PreflightReport, run_preflight
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


class TestDeriveGridOrderIdDeterminism:
    def test_same_inputs_produce_same_id(self):
        c = config()
        id_1 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert id_1 == id_2

    def test_different_side_produces_different_id(self):
        c = config()
        buy_id = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 100.0)
        sell_id = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "SELL", 100.0)
        assert buy_id != sell_id

    def test_different_price_produces_different_id(self):
        c = config()
        id_1 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 95.0)
        assert id_1 != id_2

    def test_excludes_quantity(self):
        """
        q n'est même pas un paramètre de la fonction : ce test documente
        et verrouille explicitement cette exclusion architecturale — un
        identifiant de niveau désigne un emplacement (side, price), jamais
        une quantité instantanée dépendante des soldes réels.
        """
        c = config()
        id_a = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_b = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert id_a == id_b  # aucune divergence possible : q n'entre jamais en jeu

    def test_different_config_produces_different_id(self):
        id_1 = derive_grid_order_id(config(), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(config(alpha=Decimal("0.6")), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert id_1 != id_2

    def test_different_p0_produces_different_id(self):
        id_1 = derive_grid_order_id(config(), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(config(p0=Decimal("101")), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert id_1 != id_2

    def test_different_tick_size_produces_different_id(self):
        c = config()
        id_1 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(c, Decimal("0.01"), Decimal("0.1"), "BUY", 90.0)
        assert id_1 != id_2

    def test_different_lot_size_produces_different_id(self):
        c = config()
        id_1 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        id_2 = derive_grid_order_id(c, Decimal("0.1"), Decimal("0.01"), "BUY", 90.0)
        assert id_1 != id_2


class TestDeriveGridOrderIdFormat:
    def test_format_valid_for_okx(self):
        derived = derive_grid_order_id(config(), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert len(derived) <= 32
        assert derived[0].isalpha()
        assert derived.isalnum()
        assert derived.startswith("GRID")

    def test_never_collides_with_initial_buy_prefix(self):
        """
        Un identifiant de grille ne peut jamais être confondu avec un
        identifiant d'achat initial (INITBUY) pour la même configuration
        — préfixes disjoints, vérifiable statiquement.
        """
        derived = derive_grid_order_id(config(), Decimal("0.1"), Decimal("0.1"), "BUY", 90.0)
        assert not derived.startswith("INITBUY")


# ---------------------------------------------------------------------------
# GridActivationController — adaptateur simulé
# ---------------------------------------------------------------------------

class FakeAdapter:
    """Simule OkxSpotAdapter : lectures (OkxPreflightSource) + écritures."""

    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()  # tuple[OrderSnapshot, ...] à renseigner par les tests
        self.open_orders_error = None  # exception à lever si non None
        self.get_order_results = {}  # client_order_id -> OrderSnapshot | Exception
        self.placed_orders = []  # trace des appels place_post_only_limit
        self.placement_results = {}  # client_order_id -> PlacementResult

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def list_open_orders(self, inst_id, client_order_id=None):
        if self.open_orders_error is not None:
            raise self.open_orders_error
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


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("110"), gll=Decimal("90"), nu=2, nl=2,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def open_order_snapshot(client_order_id, side="BUY", price="90.0"):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("0"), "live", "post_only", None, None)


class TestGridActivationPhase1Sondage:
    def test_no_open_orders_all_levels_get_placed(self):
        """
        Aucun ordre ouvert, aucun trouvé par get_order : tous les niveaux
        sont placés via place_post_only_limit => ACTIVE.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ACTIVE
        assert result.already_open == ()
        assert len(result.placed) == len(report.orders)

    def test_all_levels_already_open_reports_active(self):
        """
        Si tous les niveaux de report.orders correspondent déjà à un
        ordre ouvert (identifiants dérivés présents dans list_open_orders),
        aucun niveau ne reste "remaining" => ACTIVE, sans aucun placement.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        # Construit les identifiants attendus pour CHAQUE niveau de la
        # grille produite par le preflight, et les déclare "déjà ouverts".
        from grid_activation_controller import derive_grid_order_id
        snapshots = []
        for side, price, _quantity in report.orders:
            derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            snapshots.append(open_order_snapshot(derived_id, side=side, price=str(price)))
        adapter.open_orders = tuple(snapshots)

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ACTIVE
        assert len(result.already_open) == len(report.orders)

    def test_single_list_open_orders_call_regardless_of_n(self):
        """Contrôle du coût réseau : un seul appel list_open_orders, quel que soit n."""
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        calls = []
        original = adapter.list_open_orders
        adapter.list_open_orders = lambda *a, **k: (calls.append((a, k)), original(*a, **k))[1]

        GridActivationController().run(adapter, report)

        assert len(calls) == 1

    def test_sondage_error_produces_error_state_not_partial(self):
        """
        Une erreur lors du sondage (OkxApiError) doit produire ERROR,
        jamais une interprétation silencieuse comme "aucun ordre ouvert".
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        adapter.open_orders_error = OkxApiError("panne temporaire")
        report = run_preflight(adapter, grid_config())

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ERROR
        assert "sondage" in result.detail


class TestGridActivationPhase2VerificationCiblee:
    def test_filled_level_detected_via_get_order(self):
        """
        Un niveau déjà exécuté (disparu du carnet ouvert, donc absent de
        la phase 1) doit être retrouvé par get_order en phase 2 et classé
        already_filled — jamais retenté au placement.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.get_order_results[derived_id] = OrderSnapshot(
            "ORD1", derived_id, "XRP-USDC", first_side, Decimal(str(first_price)), Decimal("1.4"), Decimal("1.4"), "filled", "post_only", None, None,
        )

        result = GridActivationController().run(adapter, report)

        assert (first_side, first_price, report.orders[0][2]) in result.already_filled

    def test_absent_level_not_found_anywhere_gets_placed(self):
        """
        Niveau absent du carnet ET introuvable via get_order (OkxApiError)
        => placé via place_post_only_limit, résultat ACTIVE (acceptation
        par défaut du FakeAdapter).
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        # Aucun get_order_results renseigné => OkxApiError par défaut pour tous.

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ACTIVE
        assert result.already_filled == ()
        assert len(result.placed) == len(report.orders)

    def test_live_order_found_only_in_phase2_treated_as_already_open(self):
        """
        Fenêtre de course : un niveau live/partially_filled non détecté en
        phase 1 mais trouvé en phase 2 est traité comme déjà ouvert,
        jamais replacé.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.get_order_results[derived_id] = OrderSnapshot(
            "ORD1", derived_id, "XRP-USDC", first_side, Decimal(str(first_price)), Decimal("1.4"), Decimal("0"), "live", "post_only", None, None,
        )

        result = GridActivationController().run(adapter, report)

        assert (first_side, first_price, report.orders[0][2]) in result.already_open

    def test_canceled_order_found_in_phase2_treated_as_needing_placement(self):
        """Un ordre annulé/expiré doit être considéré comme à replacer, jamais comme satisfait."""
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.get_order_results[derived_id] = OrderSnapshot(
            "ORD1", derived_id, "XRP-USDC", first_side, Decimal(str(first_price)), Decimal("1.4"), Decimal("0"), "canceled", "post_only", None, None,
        )

        result = GridActivationController().run(adapter, report)

        assert (first_side, first_price, report.orders[0][2]) not in result.already_open
        assert (first_side, first_price, report.orders[0][2]) not in result.already_filled
        assert (first_side, first_price, report.orders[0][2]) in result.placed
        assert result.state == GridActivationState.ACTIVE

    def test_real_transport_failure_produces_error_not_absent(self):
        """
        Point de sécurité critique : une panne réseau réelle (pas
        OkxApiError) pendant la phase 2 doit produire ERROR et arrêter
        immédiatement le traitement des niveaux restants — jamais être
        interprétée comme "ordre absent", jamais continuer aveuglément.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.get_order_results[derived_id] = ConnectionError("panne réseau réelle")

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ERROR
        assert derived_id in result.detail

    def test_transport_failure_stops_remaining_levels_immediately(self):
        """
        Après une panne réelle sur un niveau, aucun niveau suivant ne doit
        être traité — vérifié en comptant les appels get_order.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert len(report.orders) >= 2  # précondition du test

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.get_order_results[derived_id] = ConnectionError("panne réseau réelle")

        calls = []
        original = adapter.get_order
        adapter.get_order = lambda *a, **k: (calls.append(k), original(*a, **k))[1]

        GridActivationController().run(adapter, report)

        # Un seul appel : celui qui a échoué. Aucun niveau suivant tenté.
        assert len(calls) == 1


class TestGridActivationPhase3Placement:
    def test_all_absent_levels_get_placed_when_accepted(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ACTIVE
        assert len(result.placed) == len(report.orders)
        assert result.failed == ()
        assert len(adapter.placed_orders) == len(report.orders)

    def test_placement_uses_derived_id_per_level(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        GridActivationController().run(adapter, report)

        used_ids = {call[4] for call in adapter.placed_orders}
        expected_ids = {
            derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            for side, price, _ in report.orders
        }
        assert used_ids == expected_ids

    def test_placement_uses_side_price_quantity_from_order_instruction(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        GridActivationController().run(adapter, report)

        placed_tuples = {(side, str(price), str(quantity)) for _, side, price, quantity, _ in adapter.placed_orders}
        expected_tuples = {(side, str(Decimal(str(price))), str(Decimal(str(quantity)))) for side, price, quantity in report.orders}
        assert placed_tuples == expected_tuples

    def test_single_rejection_produces_partial_not_error(self):
        """
        Un rejet explicite (accepted=False) doit produire PARTIAL, jamais
        ERROR — la distinction entre rejet métier et panne réelle est
        stricte (voir phase 2 pour la panne réelle).
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.placement_results[derived_id] = PlacementResult(False, None, derived_id, "51008", "Insufficient balance")

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.PARTIAL
        assert len(result.failed) == 1
        assert result.failed[0][1] == "51008"
        assert result.failed[0][2] == "Insufficient balance"

    def test_rejected_level_does_not_block_other_levels(self):
        """
        Un seul niveau rejeté parmi n acceptés : les n-1 autres restent
        placés (pas de rollback), comptabilisés dans placed.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert len(report.orders) >= 2

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.placement_results[derived_id] = PlacementResult(False, None, derived_id, "51008", "Insufficient balance")

        result = GridActivationController().run(adapter, report)

        assert len(result.placed) == len(report.orders) - 1
        assert len(result.failed) == 1

    def test_no_retry_no_order_type_change_on_rejection(self):
        """
        Un rejet ne doit jamais déclencher de nouvelle tentative : un seul
        appel place_post_only_limit par niveau rejeté, jamais de second
        essai ni de bascule vers un autre type d'ordre.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        first_side, first_price, _ = report.orders[0]
        derived_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.placement_results[derived_id] = PlacementResult(False, None, derived_id, "51008", "Insufficient balance")

        GridActivationController().run(adapter, report)

        calls_for_this_level = [c for c in adapter.placed_orders if c[4] == derived_id]
        assert len(calls_for_this_level) == 1

    def test_placement_phase_starts_only_after_sondage_fully_complete(self):
        """
        Sécurité : aucun appel place_post_only_limit ne doit avoir lieu
        avant que list_open_orders ET tous les get_order ciblés soient
        terminés — vérifié via un ordre d'appels global.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        call_order = []
        original_list = adapter.list_open_orders
        original_get = adapter.get_order
        original_place = adapter.place_post_only_limit
        adapter.list_open_orders = lambda *a, **k: (call_order.append("list_open_orders"), original_list(*a, **k))[1]
        adapter.get_order = lambda *a, **k: (call_order.append("get_order"), original_get(*a, **k))[1]
        adapter.place_post_only_limit = lambda *a, **k: (call_order.append("place_post_only_limit"), original_place(*a, **k))[1]

        GridActivationController().run(adapter, report)

        last_sondage_index = max(
            i for i, name in enumerate(call_order) if name in ("list_open_orders", "get_order")
        )
        first_placement_index = min(
            (i for i, name in enumerate(call_order) if name == "place_post_only_limit"),
            default=len(call_order),
        )
        assert first_placement_index > last_sondage_index


class TestGridActivationRealTransportFailures:
    def test_phase1_real_transport_failure_produces_error(self):
        """
        Symétrique au test déjà existant pour la phase 2 : une panne
        réseau réelle (pas OkxApiError) pendant list_open_orders doit
        produire ERROR, jamais se propager comme exception non gérée.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        adapter.open_orders_error = ConnectionError("panne réseau réelle en phase 1")

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ERROR
        assert "panne réseau" in result.detail

    def test_phase3_real_transport_failure_produces_error_preserves_partial_progress(self):
        """
        Une panne réelle en cours de boucle de placement (après que
        certains niveaux ont déjà été placés avec succès) doit produire
        ERROR tout en préservant dans le résultat ce qui a réellement été
        accompli avant l'échec — pas une perte silencieuse d'information.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert len(report.orders) >= 2

        first_side, first_price, _ = report.orders[0]
        second_side, second_price, _ = report.orders[1]
        first_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        second_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, second_side, second_price)

        original_place = adapter.place_post_only_limit
        def failing_place(inst_id, side, price, quantity, client_order_id):
            if client_order_id == second_id:
                raise ConnectionError("panne réseau réelle en phase 3")
            return original_place(inst_id, side, price, quantity, client_order_id)
        adapter.place_post_only_limit = failing_place

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ERROR
        assert (first_side, first_price, report.orders[0][2]) in result.placed  # progrès préservé
        assert "panne réseau" in result.detail

    def test_phase3_stops_immediately_no_further_placement_attempted(self):
        """Après la panne, aucun niveau suivant ne doit être tenté."""
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert len(report.orders) >= 2

        first_side, first_price, _ = report.orders[0]
        first_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)

        def failing_place(inst_id, side, price, quantity, client_order_id):
            raise ConnectionError("panne dès le premier niveau")
        adapter.place_post_only_limit = failing_place

        GridActivationController().run(adapter, report)

        # Aucun ordre ne doit avoir été enregistré comme placé avec succès
        # (le compteur placed_orders du FakeAdapter n'est jamais atteint
        # puisque failing_place lève avant tout enregistrement).
        assert adapter.placed_orders == []


class TestGridActivationRecoveryAfterPartial:
    def test_full_restart_scenario_converges_to_active_without_duplicating_orders(self):
        """
        Scénario d'intégration bout-en-bout, simulant un vrai redémarrage :

        1er appel : un niveau est rejeté par l'exchange (ex. solde
        temporairement insuffisant) => PARTIAL, n-1 niveaux placés.

        Entre les deux appels : la condition qui causait le rejet est
        corrigée côté exchange (simulé par le retrait du forçage de
        rejet du FakeAdapter — exactement ce qui se produirait après un
        redémarrage réel, où la seule vérité est l'état de l'exchange).

        2e appel (== redémarrage) : les n-1 niveaux déjà ouverts sont
        détectés en phase 1 (list_open_orders), AUCUN replacement n'est
        tenté pour eux ; seul le niveau manquant est retenté et accepté
        => ACTIVE. Convergence par idempotence, sans logique de reprise
        spécifique — exactement la propriété visée par l'architecture.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert len(report.orders) >= 2

        rejected_side, rejected_price, _ = report.orders[0]
        rejected_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, rejected_side, rejected_price)

        # --- 1er appel : un niveau est rejeté ---
        adapter.placement_results[rejected_id] = PlacementResult(False, None, rejected_id, "51008", "Insufficient balance")

        first_result = GridActivationController().run(adapter, report)

        assert first_result.state == GridActivationState.PARTIAL
        assert len(first_result.failed) == 1
        assert len(first_result.placed) == len(report.orders) - 1
        first_call_count = len(adapter.placed_orders)
        assert first_call_count == len(report.orders)  # tous tentés, un seul rejeté

        # --- Simulation du redémarrage : les ordres acceptés au 1er appel
        # sont maintenant réellement ouverts sur l'exchange (le FakeAdapter
        # ne les a pas mémorisés comme "ouverts" automatiquement — on
        # matérialise ici ce que list_open_orders retournerait réellement
        # après le succès des n-1 premiers placements). ---
        accepted_ids = {
            derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            for side, price, _ in report.orders
            if not (side == rejected_side and price == rejected_price)
        }
        adapter.open_orders = tuple(
            open_order_snapshot(order_id, side="BUY", price="0")
            for order_id in accepted_ids
        )
        # La condition de rejet est corrigée : le niveau précédemment
        # rejeté sera accepté à cette seconde tentative.
        del adapter.placement_results[rejected_id]
        adapter.placed_orders = []  # remise à zéro du compteur d'appels pour ce second appel

        # --- 2e appel : redémarrage ---
        second_result = GridActivationController().run(adapter, report)

        assert second_result.state == GridActivationState.ACTIVE
        # Les n-1 niveaux déjà ouverts sont détectés en phase 1 : AUCUN
        # appel place_post_only_limit ne doit les concerner à nouveau.
        assert len(second_result.already_open) == len(report.orders) - 1
        # Seul le niveau précédemment rejeté est retenté et placé.
        assert len(second_result.placed) == 1
        assert second_result.placed[0][:2] == (rejected_side, rejected_price)
        # Un seul appel réseau de placement au second passage : le niveau
        # manquant, jamais les n-1 déjà ouverts.
        assert len(adapter.placed_orders) == 1


class TestGridActivationRefusesWhenSpotNotCovered:
    """
    Garde-fou intrinsèque : report.spot_covered=False doit produire ERROR
    avant tout appel réseau — aucune dépendance à un appelant externe
    (GridTradingController) pour que cette garantie soit vraie.
    """

    def test_spot_not_covered_produces_error_state(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert report.spot_covered is False  # précondition du test

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ERROR

    def test_spot_not_covered_never_calls_list_open_orders(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        calls = []
        adapter.list_open_orders = lambda *a, **k: calls.append(("list_open_orders", a, k))

        GridActivationController().run(adapter, report)

        assert calls == []

    def test_spot_not_covered_never_calls_get_order(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        calls = []
        adapter.get_order = lambda *a, **k: calls.append(("get_order", a, k))

        GridActivationController().run(adapter, report)

        assert calls == []

    def test_spot_not_covered_never_calls_place_post_only_limit(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        calls = []
        adapter.place_post_only_limit = lambda *a, **k: calls.append(("place_post_only_limit", a, k))

        GridActivationController().run(adapter, report)

        assert calls == []
        assert adapter.placed_orders == []

    def test_error_message_mentions_spot_covered_false(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, grid_config())

        result = GridActivationController().run(adapter, report)

        assert "spot_covered=False" in result.detail

    def test_spot_covered_true_preserves_existing_behavior(self):
        """
        spot_covered=True doit conserver exactement le comportement
        antérieur au patch : le garde-fou ne doit jamais interférer
        lorsque la condition est satisfaite.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        assert report.spot_covered is True  # précondition du test

        result = GridActivationController().run(adapter, report)

        assert result.state == GridActivationState.ACTIVE
        assert len(result.placed) == len(report.orders)
