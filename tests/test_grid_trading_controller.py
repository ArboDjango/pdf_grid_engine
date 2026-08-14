"""Tests pour GridTradingController.

Aucun appel réseau réel : adaptateur simulé complet, couvrant les trois
niveaux orchestrés (get_account_config, InitializationController via
get_instrument/get_fee_rates/get_balances/get_order/place_market_buy,
GridActivationController via list_open_orders/get_order/
place_post_only_limit).

Ce fichier ne teste AUCUN calcul économique (q, required_spot, treillis) :
ces calculs sont déjà couverts par tests/test_grid_preflight.py,
tests/test_initialization_controller.py et
tests/test_grid_activation_controller.py. Ici, seul le séquencement est
vérifié.
"""

from decimal import Decimal

from grid_activation_controller import GridActivationState, derive_grid_order_id
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from initialization_controller import InitState, derive_client_order_id
from okx_spot_adapter import (
    AccountConfig,
    Balances,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
)
from trellis_calculator import GridGeometry


def config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("110"), gll=Decimal("90"), nu=2, nl=2,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


class FakeAdapter:
    """Adaptateur simulé complet : lectures + écritures des trois niveaux."""

    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1",
                 minimum="0.1", fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.account_config = AccountConfig(fee_type=fee_type)
        self.account_config_error = None

        self.get_order_results = {}  # client_order_id -> OrderSnapshot | Exception
        self.placement_results = {}  # client_order_id -> PlacementResult
        self.open_orders = ()

        self.calls = []  # trace de tous les appels d'écriture/lecture significatifs

    # --- OkxPreflightSource ---
    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    # --- précondition feeType ---
    def get_account_config(self):
        self.calls.append("get_account_config")
        if self.account_config_error is not None:
            raise self.account_config_error
        return self.account_config

    # --- InitializationController ---
    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.calls.append(("get_order", client_order_id))
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
        raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")

    def place_market_buy(self, inst_id, quantity, client_order_id):
        self.calls.append(("place_market_buy", client_order_id, quantity))
        if client_order_id in self.placement_results:
            return self.placement_results[client_order_id]
        return PlacementResult(True, "ORDX", client_order_id, None, None)

    # --- GridActivationController ---
    def list_open_orders(self, inst_id, client_order_id=None):
        self.calls.append("list_open_orders")
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        self.calls.append(("place_post_only_limit", client_order_id))
        if client_order_id in self.placement_results:
            return self.placement_results[client_order_id]
        return PlacementResult(True, "ORDG", client_order_id, None, None)


def had_call(adapter, name):
    return any(c == name or (isinstance(c, tuple) and c[0] == name) for c in adapter.calls)


# ---------------------------------------------------------------------------
# feeType — précondition
# ---------------------------------------------------------------------------

class TestFeeTypePrecondition:
    def test_fee_type_1_triggers_initialization(self):
        adapter = FakeAdapter(base="0", quote="1000", fee_type="1")

        result = GridTradingController().run(adapter, config())

        assert had_call(adapter, "get_account_config")
        assert result.state != GridTradingState.FEE_TYPE_REJECTED

    def test_fee_type_not_1_stops_immediately(self):
        adapter = FakeAdapter(base="0", quote="1000", fee_type="0")

        result = GridTradingController().run(adapter, config())

        assert result.state == GridTradingState.FEE_TYPE_REJECTED
        assert result.fee_type == "0"

    def test_fee_type_not_1_makes_no_trading_call(self):
        """
        Aucun appel de trading (place_market_buy, place_post_only_limit)
        ne doit avoir lieu si feeType != "1" — garanti par construction,
        pas seulement par convention.
        """
        adapter = FakeAdapter(base="0", quote="1000", fee_type="2")

        GridTradingController().run(adapter, config())

        assert not had_call(adapter, "place_market_buy")
        assert not had_call(adapter, "place_post_only_limit")
        assert not had_call(adapter, "get_order")  # InitializationController jamais atteint
        assert not had_call(adapter, "list_open_orders")  # GridActivationController jamais atteint

    def test_account_config_error_makes_no_purchase(self):
        adapter = FakeAdapter(base="0", quote="1000")
        adapter.account_config_error = OkxApiError("panne temporaire")

        result = GridTradingController().run(adapter, config())

        assert result.state == GridTradingState.ERROR
        assert not had_call(adapter, "place_market_buy")
        assert not had_call(adapter, "place_post_only_limit")

    def test_never_calls_set_fee_type(self):
        """
        Vérification structurelle : GridTradingController n'effectue
        jamais d'appel à set-fee-type. Une mention documentaire du nom
        (dans le message d'erreur expliquant la précondition) est
        légitime et attendue ; ce qui est interdit est un APPEL réel.
        """
        import grid_trading_controller
        import inspect
        source = inspect.getsource(grid_trading_controller)
        assert "set_fee_type(" not in source
        assert ".set_fee_type" not in source
        assert "adapter.set_fee_type" not in source


# ---------------------------------------------------------------------------
# Initialisation — branchement sur InitState
# ---------------------------------------------------------------------------

class TestInitializationBranching:
    def test_spot_ready_triggers_activation(self):
        adapter = FakeAdapter(base="1000", quote="1000")  # patrimoine largement suffisant

        result = GridTradingController().run(adapter, config())

        assert result.initialization_result.state == InitState.SPOT_READY
        assert had_call(adapter, "list_open_orders")  # GridActivationController atteint
        assert result.state == GridTradingState.ACTIVATED

    def test_buying_does_not_trigger_activation(self):
        adapter = FakeAdapter(base="0", quote="1000")  # déclenche un achat -> BUYING

        result = GridTradingController().run(adapter, config())

        assert result.initialization_result.state == InitState.BUYING
        assert not had_call(adapter, "list_open_orders")  # GridActivationController jamais atteint
        assert result.state == GridTradingState.INITIALIZATION_INCOMPLETE

    def test_partially_filled_does_not_trigger_activation(self):
        adapter = FakeAdapter(base="0", quote="1000")
        report = run_preflight(adapter, config())
        derived_id = derive_client_order_id(config(), report.instrument.tick_size, report.instrument.lot_size)
        adapter.get_order_results[derived_id] = OrderSnapshot(
            "ORD1", derived_id, "XRP-USDC", "BUY", Decimal("0"), Decimal("1.4"), Decimal("0.5"), "partially_filled", "market", None, None,
        )

        result = GridTradingController().run(adapter, config())

        assert result.initialization_result.state == InitState.PARTIALLY_FILLED
        assert not had_call(adapter, "list_open_orders")
        assert result.state == GridTradingState.INITIALIZATION_INCOMPLETE

    def test_error_does_not_trigger_activation(self):
        adapter = FakeAdapter(base="0", quote="1", minimum="1")  # capital dérisoire -> ERROR

        result = GridTradingController().run(adapter, config())

        assert result.initialization_result.state == InitState.ERROR
        assert not had_call(adapter, "list_open_orders")
        assert result.state == GridTradingState.INITIALIZATION_INCOMPLETE


# ---------------------------------------------------------------------------
# Rapport transmis — exactitude, jamais un rapport antérieur
# ---------------------------------------------------------------------------

class TestReportPropagation:
    def test_exact_initialization_report_passed_to_activation(self):
        """
        Le rapport transmis à GridActivationController doit être
        exactement InitializationResult.report (identité d'objet), jamais
        un rapport de preflight recalculé ou antérieur à l'achat.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        captured_reports = []
        from grid_activation_controller import GridActivationController as RealGAC
        original_run = RealGAC.run
        def spying_run(self, adapter_arg, report_arg):
            captured_reports.append(report_arg)
            return original_run(self, adapter_arg, report_arg)
        RealGAC.run = spying_run
        try:
            result = GridTradingController().run(adapter, config())
        finally:
            RealGAC.run = original_run

        assert len(captured_reports) == 1
        assert captured_reports[0] is result.initialization_result.report


# ---------------------------------------------------------------------------
# Propagation du résultat final — sans transformation
# ---------------------------------------------------------------------------

class TestActivationResultPropagation:
    def test_active_result_propagated_untransformed(self):
        adapter = FakeAdapter(base="1000", quote="1000")

        result = GridTradingController().run(adapter, config())

        assert result.state == GridTradingState.ACTIVATED
        assert result.activation_result.state == GridActivationState.ACTIVE

    def test_partial_result_propagated_untransformed(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, config())
        first_side, first_price, _ = report.orders[0]
        rejected_id = derive_grid_order_id(config(), report.instrument.tick_size, report.instrument.lot_size, first_side, first_price)
        adapter.placement_results[rejected_id] = PlacementResult(False, None, rejected_id, "51008", "Insufficient balance")

        result = GridTradingController().run(adapter, config())

        assert result.state == GridTradingState.ACTIVATED
        assert result.activation_result.state == GridActivationState.PARTIAL
        assert len(result.activation_result.failed) == 1

    def test_activation_internal_error_propagated_untransformed(self):
        """
        GridActivationController capture déjà ses propres pannes réseau
        en interne (voir grid_activation_controller.py, phases 1 et 3) et
        retourne un GridActivationResult(state=ERROR, ...) SANS lever —
        il ne s'agit donc pas d'une exception remontant à l'orchestrateur,
        mais d'un résultat normal à propager tel quel (constraint 5 :
        "retourner son résultat sans le transformer").
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        adapter.list_open_orders = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("panne réelle"))

        result = GridTradingController().run(adapter, config())

        # L'orchestrateur A BIEN atteint et complété l'étape 3 : il a reçu
        # un résultat (qui se trouve être porteur d'une erreur interne),
        # pas levé d'exception lui-même.
        assert result.state == GridTradingState.ACTIVATED
        assert result.activation_result.state == GridActivationState.ERROR
        assert "panne réelle" in result.activation_result.detail

    def test_orchestrator_level_exception_during_activation_produces_error(self):
        """
        Si GridActivationController.run() lève une exception NON gérée en
        son sein (cas distinct du précédent — ici on simule une panne au
        niveau de l'orchestrateur lui-même, avant tout traitement interne
        de GridActivationController), l'orchestrateur doit produire
        GridTradingState.ERROR, en conservant initialization_result pour
        diagnostic.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        from grid_activation_controller import GridActivationController as RealGAC
        original_run = RealGAC.run
        def raising_run(self, adapter_arg, report_arg):
            raise RuntimeError("panne au niveau orchestrateur")
        RealGAC.run = raising_run
        try:
            result = GridTradingController().run(adapter, config())
        finally:
            RealGAC.run = original_run

        assert result.state == GridTradingState.ERROR
        assert result.initialization_result is not None
        assert result.initialization_result.state == InitState.SPOT_READY


# ---------------------------------------------------------------------------
# Absence de toute logique économique dans l'orchestrateur
# ---------------------------------------------------------------------------

class TestNoEconomicLogic:
    def test_module_never_imports_trellis_calculator(self):
        """
        L'orchestrateur ne doit avoir aucune dépendance directe vers la
        construction géométrique — il ne consomme que des résultats déjà
        calculés par les composants qu'il séquence.
        """
        import grid_trading_controller
        import inspect
        source = inspect.getsource(grid_trading_controller)
        assert "TrellisCalculator" not in source
        assert "CellInstantiator" not in source
        assert "GridOrderProjection" not in source

    def test_module_never_computes_required_spot_or_q(self):
        """
        Aucun recalcul de q ou required_spot : ces symboles n'apparaissent
        légitimement que dans la documentation du module (pour expliquer
        ce qui n'est PAS fait) ; ce test vérifie l'absence de toute
        assignation ou calcul réel sur ces grandeurs, pas l'absence pure
        et simple du mot dans les commentaires.
        """
        import grid_trading_controller
        import inspect
        source = inspect.getsource(grid_trading_controller)
        assert "required_spot =" not in source
        assert "required_spot=" not in source
        assert " q = " not in source
        assert "q=Decimal" not in source
