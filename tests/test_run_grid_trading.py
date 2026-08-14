"""Tests pour run_grid_trading.py.

Aucun appel réseau réel. L'adaptateur OKX est simulé pour les tests qui
en ont besoin (build_config, propagation de résultat) ; les tests de
parsing d'arguments et de refus précoce n'instancient jamais d'adaptateur.
"""

from decimal import Decimal

import pytest

from grid_activation_controller import GridActivationResult, GridActivationState
from grid_preflight import PreflightConfig
from grid_trading_controller import GridTradingResult, GridTradingState
from initialization_controller import InitializationResult, InitState
from okx_spot_adapter import OkxApiError
from run_grid_trading import (
    EntrypointRefusal,
    build_adapter,
    build_config,
    check_live_confirmation,
    determine_exit_code,
    main,
    parse_args,
)
from trellis_calculator import GridGeometry


# ---------------------------------------------------------------------------
# Parsing des arguments
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_mode_absent_raises(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_mode_dry_run_parsed(self):
        args = parse_args(["--mode", "DRY_RUN"])
        assert args.mode == "DRY_RUN"
        assert args.confirm_live is False

    def test_mode_live_with_confirm(self):
        args = parse_args(["--mode", "LIVE", "--confirm-live"])
        assert args.mode == "LIVE"
        assert args.confirm_live is True

    def test_inst_id_default_is_xrp_usdc(self):
        args = parse_args(["--mode", "DRY_RUN"])
        assert args.inst_id == "XRP-USDC"

    def test_inst_id_override(self):
        args = parse_args(["--mode", "DRY_RUN", "--inst-id", "BTC-USDC"])
        assert args.inst_id == "BTC-USDC"

    def test_invalid_mode_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "PRODUCTION"])


# ---------------------------------------------------------------------------
# Confirmation LIVE — refus avant toute action
# ---------------------------------------------------------------------------

class TestLiveConfirmation:
    def test_live_without_confirm_raises_before_any_action(self):
        with pytest.raises(EntrypointRefusal):
            check_live_confirmation("LIVE", confirm_live=False)

    def test_live_with_confirm_does_not_raise(self):
        check_live_confirmation("LIVE", confirm_live=True)  # ne doit pas lever

    def test_demo_without_confirm_does_not_raise(self):
        """--confirm-live est sans effet hors LIVE : DEMO ne le requiert jamais."""
        check_live_confirmation("DEMO", confirm_live=False)

    def test_dry_run_without_confirm_does_not_raise(self):
        check_live_confirmation("DRY_RUN", confirm_live=False)

    def test_live_refusal_happens_before_env_read(self, monkeypatch):
        """
        Vérification bout-en-bout : sans --confirm-live, main() ne doit
        jamais tenter de lire les variables d'environnement — même
        absentes, aucun KeyError ne doit être levé (la vérification
        --confirm-live intervient avant).
        """
        monkeypatch.delenv("OKX_API_KEY", raising=False)
        monkeypatch.delenv("OKX_SECRET_KEY", raising=False)
        monkeypatch.delenv("OKX_PASSPHRASE", raising=False)

        exit_code = main(["--mode", "LIVE"])

        assert exit_code == 1


# ---------------------------------------------------------------------------
# Construction de l'adaptateur — cohérence mode/demo
# ---------------------------------------------------------------------------

class TestBuildAdapter:
    def test_demo_mode_produces_demo_true(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        adapter = build_adapter("DEMO")

        assert adapter._demo is True

    def test_live_mode_produces_demo_false(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        adapter = build_adapter("LIVE")

        assert adapter._demo is False

    def test_dry_run_mode_produces_demo_false(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        adapter = build_adapter("DRY_RUN")

        assert adapter._demo is False

    def test_missing_env_var_raises_key_error(self, monkeypatch):
        monkeypatch.delenv("OKX_API_KEY", raising=False)
        monkeypatch.delenv("OKX_SECRET_KEY", raising=False)
        monkeypatch.delenv("OKX_PASSPHRASE", raising=False)

        with pytest.raises(KeyError):
            build_adapter("DEMO")


# ---------------------------------------------------------------------------
# Construction de PreflightConfig — valeurs exactes de dry_run_xrp.py
# ---------------------------------------------------------------------------

class FakeAdapterForConfig:
    """Simule uniquement get_ticker/get_fee_rates, nécessaires à build_config."""

    def __init__(self, last="100", maker="-0.001"):
        self._last = Decimal(last)
        self._maker = Decimal(maker)

    def get_ticker(self, inst_id):
        from okx_spot_adapter import Ticker
        return Ticker(inst_id=inst_id, last=self._last, ts=1)

    def get_fee_rates(self, inst_id):
        from okx_spot_adapter import FeeRates
        return FeeRates(inst_id=inst_id, maker=self._maker, taker=Decimal("-0.002"))


class TestBuildConfig:
    def test_config_matches_dry_run_xrp_parameters_exactly(self):
        adapter = FakeAdapterForConfig(last="100", maker="-0.001")

        config = build_config(adapter, "XRP-USDC")

        assert config == PreflightConfig(
            inst_id="XRP-USDC",
            gul=Decimal("100") * Decimal("1.15"),
            gll=Decimal("100") * Decimal("0.85"),
            nu=5,
            nl=5,
            p0=Decimal("100"),
            geometry=GridGeometry.EQUIDISTANT,
            spacing_h_pct=Decimal("0.001"),  # abs(-0.001)
            alpha=Decimal("0.95"),
            operational_margin=Decimal("0.50"),
        )

    def test_config_uses_provided_inst_id(self):
        adapter = FakeAdapterForConfig()

        config = build_config(adapter, "ETH-USDC")

        assert config.inst_id == "ETH-USDC"

    def test_spacing_h_pct_takes_absolute_value_of_negative_maker_fee(self):
        """Convention de signe OKX : négatif = commission ; toujours positif ici."""
        adapter = FakeAdapterForConfig(maker="-0.0008")

        config = build_config(adapter, "XRP-USDC")

        assert config.spacing_h_pct == Decimal("0.0008")


# ---------------------------------------------------------------------------
# DRY_RUN n'appelle jamais GridTradingController
# ---------------------------------------------------------------------------

class TestDryRunNeverCallsGridTradingController:
    def test_dry_run_does_not_invoke_grid_trading_controller(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        calls = []

        class SpyGridTradingController:
            def run(self, adapter, config):
                calls.append("run")
                raise AssertionError("GridTradingController ne doit jamais être appelé en DRY_RUN")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "GridTradingController", SpyGridTradingController)
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FakeAdapterForConfig())
        monkeypatch.setattr(run_grid_trading, "run_preflight", lambda adapter, config: _fake_report())

        exit_code = main(["--mode", "DRY_RUN"])

        assert calls == []
        assert exit_code == 0


def _fake_report():
    """Rapport minimal suffisant pour format_report()."""
    from grid_preflight import PreflightReport
    from okx_spot_adapter import Balances, FeeRates, InstrumentRules
    instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.1"), Decimal("0.1"), Decimal("0.1"), None, 1, 1, "live")
    fees = FeeRates("XRP-USDC", Decimal("-0.001"), Decimal("-0.002"))
    balances = Balances(Decimal("0"), Decimal("1000"), Decimal("0"), Decimal("1000"))
    config = PreflightConfig("XRP-USDC", Decimal("115"), Decimal("85"), 5, 5, Decimal("100"), GridGeometry.EQUIDISTANT, Decimal("0.001"), Decimal("0.95"), Decimal("0.5"))
    return PreflightReport(instrument, fees, balances, config, Decimal("1000"), Decimal("950"), (Decimal("85"), Decimal("100"), Decimal("115")), Decimal("1"), Decimal("1"), Decimal("5"), Decimal("85"), False, True, (), (), ())


# ---------------------------------------------------------------------------
# Propagation des différents GridTradingResult
# ---------------------------------------------------------------------------

def _fee_type_rejected_result():
    return GridTradingResult(GridTradingState.FEE_TYPE_REJECTED, "0", None, None, "feeType doit valoir \"1\"")


def _initialization_incomplete_result():
    init_result = InitializationResult(InitState.BUYING, None, None, None)
    return GridTradingResult(GridTradingState.INITIALIZATION_INCOMPLETE, "1", init_result, None, "initialisation non terminée, état=BUYING")


def _activated_active_result():
    init_result = InitializationResult(InitState.SPOT_READY, None, None, None)
    activation_result = GridActivationResult(GridActivationState.ACTIVE, (), (), (), (), None)
    return GridTradingResult(GridTradingState.ACTIVATED, "1", init_result, activation_result, None)


def _activated_partial_result():
    init_result = InitializationResult(InitState.SPOT_READY, None, None, None)
    activation_result = GridActivationResult(GridActivationState.PARTIAL, (), (), (), (("BUY", 90.0, "51008", "Insufficient balance"),), "1 niveau(x) rejeté(s)")
    return GridTradingResult(GridTradingState.ACTIVATED, "1", init_result, activation_result, None)


def _activated_error_result():
    init_result = InitializationResult(InitState.SPOT_READY, None, None, None)
    activation_result = GridActivationResult(GridActivationState.ERROR, (), (), (), (), "panne réseau réelle")
    return GridTradingResult(GridTradingState.ACTIVATED, "1", init_result, activation_result, None)


def _grid_trading_error_result():
    return GridTradingResult(GridTradingState.ERROR, None, None, None, "panne réseau réelle")


class TestResultPropagation:
    @pytest.mark.parametrize("result_factory,expected_code", [
        (_fee_type_rejected_result, 1),
        (_initialization_incomplete_result, 2),
        (_activated_active_result, 0),
        (_activated_partial_result, 3),
        (_activated_error_result, 4),
        (_grid_trading_error_result, 2),
    ])
    def test_exit_code_matches_result(self, result_factory, expected_code):
        result = result_factory()
        assert determine_exit_code("DEMO", result) == expected_code

    def test_detail_never_reformulated(self, capsys, monkeypatch):
        """
        Le detail produit par GridTradingController doit apparaître tel
        quel dans la sortie console, jamais reformulé ou traduit.
        """
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        import run_grid_trading

        class FakeController:
            def run(self, adapter, config):
                return _fee_type_rejected_result()

        monkeypatch.setattr(run_grid_trading, "GridTradingController", FakeController)
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FakeAdapterForConfig())

        main(["--mode", "DEMO"])

        captured = capsys.readouterr()
        assert 'feeType doit valoir "1"' in captured.out


# ---------------------------------------------------------------------------
# Codes de sortie — table complète
# ---------------------------------------------------------------------------

class TestExitCodes:
    def test_dry_run_always_returns_zero(self):
        assert determine_exit_code("DRY_RUN", None) == 0

    def test_unexpected_state_returns_four_never_zero(self):
        """
        Un état inattendu ne doit jamais produire un succès (0) par
        défaut — toujours un échec technique (4), jamais une supposition
        optimiste.
        """
        class FakeActivation:
            state = "SOMETHING_UNEXPECTED"

        class FakeResult:
            state = GridTradingState.ACTIVATED
            activation_result = FakeActivation()

        assert determine_exit_code("DEMO", FakeResult()) == 4


# ---------------------------------------------------------------------------
# Absence de logique économique / d'appels directs de trading
# ---------------------------------------------------------------------------

class TestNoEconomicLogicOrDirectTradingCalls:
    def test_module_never_computes_q_or_required_spot(self):
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "required_spot =" not in source
        assert " q = " not in source

    def test_module_never_imports_trellis_calculator_computation_classes(self):
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "TrellisCalculator" not in source
        assert "CellInstantiator" not in source

    def test_module_never_calls_place_market_buy_directly(self):
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "place_market_buy(" not in source
        assert ".place_market_buy" not in source

    def test_module_never_calls_place_post_only_limit_directly(self):
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "place_post_only_limit(" not in source
        assert ".place_post_only_limit" not in source

    def test_module_never_imports_initialization_controller_class_directly(self):
        """
        InitializationController ne doit jamais être appelé directement
        par ce script — seul GridTradingController y accède. Le module
        peut importer InitState (pour lire un résultat), mais jamais
        InitializationController lui-même.
        """
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "InitializationController(" not in source

    def test_module_never_imports_grid_activation_controller_class_directly(self):
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "GridActivationController(" not in source

    def test_sole_business_call_in_demo_live_is_grid_trading_controller(self):
        """
        Vérification positive, symétrique aux vérifications négatives
        ci-dessus : GridTradingController().run(...) doit bien être
        l'unique appel métier du chemin DEMO/LIVE.
        """
        import run_grid_trading
        import inspect
        source = inspect.getsource(run_grid_trading)
        assert "GridTradingController().run(" in source


# ---------------------------------------------------------------------------
# Correction du finding bloquant : OkxApiError dans build_config() (get_ticker
# / get_fee_rates) doit produire le code 4, jamais un traceback non géré.
# ---------------------------------------------------------------------------

class FailingTickerAdapter:
    """get_ticker lève OkxApiError ; get_fee_rates ne doit jamais être appelé après."""

    def get_ticker(self, inst_id):
        raise OkxApiError("instrument introuvable")

    def get_fee_rates(self, inst_id):
        raise AssertionError("get_fee_rates ne doit pas être appelé si get_ticker a échoué avant")


class FailingFeeRatesAdapter:
    """get_ticker réussit ; get_fee_rates lève OkxApiError."""

    def get_ticker(self, inst_id):
        from okx_spot_adapter import Ticker
        return Ticker(inst_id=inst_id, last=Decimal("100"), ts=1)

    def get_fee_rates(self, inst_id):
        raise OkxApiError("service de frais indisponible")


class TestOkxApiErrorDuringConfigBuild:
    def test_get_ticker_failure_returns_exit_code_four(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FailingTickerAdapter())

        exit_code = main(["--mode", "DRY_RUN"])

        assert exit_code == 4

    def test_get_ticker_failure_produces_no_unhandled_traceback(self, monkeypatch):
        """
        Vérification directe de la régression : avant correction, cette
        exception remontait hors de main() sans être interceptée. Ici,
        l'appel doit retourner normalement (pas lever), preuve que
        l'exception est bien capturée à l'intérieur de main().
        """
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FailingTickerAdapter())

        try:
            exit_code = main(["--mode", "DEMO"])
        except OkxApiError:
            pytest.fail("OkxApiError ne doit jamais remonter hors de main() — régression du finding bloquant")

        assert exit_code == 4

    def test_get_fee_rates_failure_returns_exit_code_four(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FailingFeeRatesAdapter())

        exit_code = main(["--mode", "DRY_RUN"])

        assert exit_code == 4

    def test_get_fee_rates_failure_produces_no_unhandled_traceback(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "k")
        monkeypatch.setenv("OKX_SECRET_KEY", "s")
        monkeypatch.setenv("OKX_PASSPHRASE", "p")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FailingFeeRatesAdapter())

        try:
            exit_code = main(["--mode", "LIVE", "--confirm-live"])
        except OkxApiError:
            pytest.fail("OkxApiError ne doit jamais remonter hors de main() — régression du finding bloquant")

        assert exit_code == 4

    def test_error_message_contains_no_secrets(self, monkeypatch, capsys):
        """
        Le message d'erreur affiché doit contenir le détail OKX de
        l'erreur, mais jamais les identifiants réels utilisés pour la
        connexion (api_key/secret_key/passphrase).
        """
        monkeypatch.setenv("OKX_API_KEY", "SUPER_SECRET_API_KEY")
        monkeypatch.setenv("OKX_SECRET_KEY", "SUPER_SECRET_SECRET_KEY")
        monkeypatch.setenv("OKX_PASSPHRASE", "SUPER_SECRET_PASSPHRASE")

        import run_grid_trading
        monkeypatch.setattr(run_grid_trading, "build_adapter", lambda mode: FailingTickerAdapter())

        main(["--mode", "DRY_RUN"])

        captured = capsys.readouterr()
        assert "SUPER_SECRET_API_KEY" not in captured.out
        assert "SUPER_SECRET_SECRET_KEY" not in captured.out
        assert "SUPER_SECRET_PASSPHRASE" not in captured.out
        assert "instrument introuvable" in captured.out  # le détail OKX reste affiché

    def test_other_exit_codes_unchanged_by_this_fix(self):
        """
        Non-régression explicite : les codes 0, 1, 2, 3 ne doivent pas
        être affectés par l'ajout de la gestion d'OkxApiError.
        """
        assert determine_exit_code("DRY_RUN", None) == 0
        assert determine_exit_code("DEMO", _activated_active_result()) == 0
        assert determine_exit_code("DEMO", _fee_type_rejected_result()) == 1
        assert determine_exit_code("DEMO", _initialization_incomplete_result()) == 2
        assert determine_exit_code("DEMO", _activated_partial_result()) == 3

    def test_live_refusal_still_precedes_any_network_or_env_access(self, monkeypatch):
        """
        Non-régression explicite : le refus LIVE sans --confirm-live doit
        toujours intervenir avant toute lecture d'environnement et tout
        appel réseau — y compris maintenant que OkxApiError est capturée
        par ailleurs. Si build_adapter était appelé avant la vérification,
        l'absence des variables d'environnement lèverait KeyError (code 4),
        pas EntrypointRefusal (code 1) : ce test continue de discriminer
        correctement l'ordre des opérations.
        """
        monkeypatch.delenv("OKX_API_KEY", raising=False)
        monkeypatch.delenv("OKX_SECRET_KEY", raising=False)
        monkeypatch.delenv("OKX_PASSPHRASE", raising=False)

        exit_code = main(["--mode", "LIVE"])

        assert exit_code == 1
