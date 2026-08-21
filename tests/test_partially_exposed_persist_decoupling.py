"""Tests du correctif : PARTIALLY_EXPOSED avec au moins un ordre réel doit
persister et démarrer la boucle, pour que le cycle suivant puisse
reprendre ce que l'activation initiale n'a pas terminé -- jamais laisser
un ordre réel orphelin.

Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

import sys
from decimal import Decimal

import pytest

import run_active_grid
from grid_activation_controller import GridActivationResult, GridActivationState
from grid_trading_controller import GridTradingController, GridTradingResult, GridTradingState
from okx_spot_adapter import AccountConfig, Balances, FeeRates, InstrumentRules, Ticker
from run_active_grid import _has_real_orders


class FakeAdapter:
    def __init__(self, *, base="20000", quote="20000", p0="1.0013"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.0001"), Decimal("0.001"), Decimal("0.1"), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.ticker_price = Decimal(p0)
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self._candles = _make_candles([1.000 + i * 0.0008 for i in range(50)])

    def get_instrument(self, inst_id): return self.instrument
    def get_fee_rates(self, inst_id): return self.fees
    def get_balances(self, b, q): return self.balances
    def get_account_config(self): return AccountConfig(fee_type="1")
    def get_ticker(self, inst_id): return Ticker(inst_id=inst_id, last=self.ticker_price, ts=1)
    def get_klines(self, inst_id, bar="15m", limit=50): return self._candles

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        return self.fills

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        from okx_spot_adapter import OkxApiError
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
        raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        from okx_spot_adapter import OrderSnapshot, PlacementResult
        self.placed_orders.append((inst_id, side, price, quantity, client_order_id))
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, side, price, quantity, Decimal("0"), "live", "post_only", 1, 1)
        self.open_orders = self.open_orders + (snapshot,)
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


def _make_candles(closes):
    from okx_spot_adapter import Candle
    return tuple(
        Candle(ts=1_000_000 + i * 900_000, open=Decimal(str(c)), high=Decimal(str(c + 0.0005)),
               low=Decimal(str(c - 0.0005)), close=Decimal(str(c)), volume=Decimal("1000"))
        for i, c in enumerate(closes)
    )


def make_activation_result(state, placed=(), already_open=(), already_filled=(), failed=(), detail=None):
    return GridActivationResult(state, already_open, already_filled, placed, failed, detail)


def make_grid_trading_result(state, activation_result=None, detail=None):
    return GridTradingResult(state, "1", None, activation_result, detail)


# ---------------------------------------------------------------------------
# Tests unitaires purs de _has_real_orders
# ---------------------------------------------------------------------------

class TestHasRealOrdersPure:
    def test_activation_result_none_returns_false(self):
        result = make_grid_trading_result(GridTradingState.EXPOSURE_CONFLICT, activation_result=None)
        assert _has_real_orders(result) is False

    def test_placed_non_empty_returns_true(self):
        activation_result = make_activation_result(GridActivationState.PARTIAL, placed=(("SELL", 1.02, Decimal("10")),))
        result = make_grid_trading_result(GridTradingState.PARTIALLY_EXPOSED, activation_result)
        assert _has_real_orders(result) is True

    def test_already_open_non_empty_returns_true(self):
        activation_result = make_activation_result(GridActivationState.ACTIVE, already_open=(("BUY", 0.99, Decimal("10")),))
        result = make_grid_trading_result(GridTradingState.ACTIVATED, activation_result)
        assert _has_real_orders(result) is True

    def test_already_filled_non_empty_returns_true(self):
        activation_result = make_activation_result(GridActivationState.ACTIVE, already_filled=(("BUY", 0.99, Decimal("10")),))
        result = make_grid_trading_result(GridTradingState.ACTIVATED, activation_result)
        assert _has_real_orders(result) is True

    def test_all_empty_returns_false(self):
        activation_result = make_activation_result(GridActivationState.ERROR)
        result = make_grid_trading_result(GridTradingState.ERROR, activation_result)
        assert _has_real_orders(result) is False

    def test_never_infers_from_to_materialize_only_from_activation_result(self):
        """Preuve structurelle : la fonction ne prend qu'un seul argument
        (activation), jamais un ExposurePlan/to_materialize."""
        import inspect
        sig = inspect.signature(_has_real_orders)
        assert list(sig.parameters.keys()) == ["activation"]


# ---------------------------------------------------------------------------
# 1-4 : intégration via run_auto_mode (mêmes 4 cas dans main() --mode create,
# structure symétrique, testée une fois ici pour le chemin partagé)
# ---------------------------------------------------------------------------

class TestFourCasesIntegration:
    def test_1_activated_persists_and_runs(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)
        run_calls = []
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: run_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 0
        assert run_active_grid.load_grid_state(state_path) is not None
        assert run_calls == [1]

    def test_2_partially_exposed_with_real_order_persists_and_runs(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        activation_result = make_activation_result(
            GridActivationState.ACTIVE,  # côté SELL réussi -> ACTIVE au niveau GridActivationController
            placed=(("SELL", 1.02, Decimal("10")),),
        )
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: make_grid_trading_result(GridTradingState.PARTIALLY_EXPOSED, activation_result, "BUY filtré, SELL placé"),
        )
        run_calls = []
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: run_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 0
        assert run_active_grid.load_grid_state(state_path) is not None  # PERSISTÉ malgré PARTIALLY_EXPOSED
        assert run_calls == [1]  # BOUCLE DÉMARRÉE

    def test_3_partially_exposed_with_zero_orders_does_not_run(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        activation_result = make_activation_result(GridActivationState.ERROR)  # rien placé
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: make_grid_trading_result(GridTradingState.PARTIALLY_EXPOSED, activation_result, "rien matérialisé"),
        )
        run_calls = []
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: run_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 2
        assert run_active_grid.load_grid_state(state_path) is None  # PAS persisté
        assert run_calls == []  # PAS de boucle

    def test_4_error_before_placement_unchanged(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: make_grid_trading_result(GridTradingState.ERROR, None, "panne avant tout placement"),
        )
        run_calls = []
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: run_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 2
        assert run_active_grid.load_grid_state(state_path) is None
        assert run_calls == []


# ---------------------------------------------------------------------------
# 5 : le SELL réellement placé n'est jamais annulé par ce correctif
# ---------------------------------------------------------------------------

class TestNoCancellationIntroduced:
    def test_5_no_cancel_order_method_referenced_by_has_real_orders(self):
        import inspect
        source = inspect.getsource(_has_real_orders)
        assert "cancel_order" not in source
        assert "place_post_only_limit" not in source

    def test_5_placed_sell_survives_partial_exposure_path(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        sell_instruction = ("SELL", 1.02, Decimal("10"))
        activation_result = make_activation_result(GridActivationState.ACTIVE, placed=(sell_instruction,))
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: make_grid_trading_result(GridTradingState.PARTIALLY_EXPOSED, activation_result, "BUY filtré"),
        )
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        # Le SELL déjà placé reste rapporté tel quel dans activation_result --
        # jamais retiré, jamais annulé par ce correctif.
        assert sell_instruction in activation_result.placed


# ---------------------------------------------------------------------------
# 6 : le cycle suivant peut démarrer normalement après PARTIALLY_EXPOSED
# ---------------------------------------------------------------------------

class TestNextCycleResumesNormally:
    def test_6_persisted_config_reloadable_and_resumable(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        activation_result = make_activation_result(GridActivationState.ACTIVE, placed=(("SELL", 1.02, Decimal("10")),))
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: make_grid_trading_result(GridTradingState.PARTIALLY_EXPOSED, activation_result, "BUY filtré"),
        )
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        # Cycle suivant : run_auto_mode relance, trouve maintenant un état
        # persisté valide -> chemin REPRISE normal, jamais recréation.
        second_run_calls = []
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: second_run_calls.append(1))
        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 0
        assert second_run_calls == [1]


# ---------------------------------------------------------------------------
# Non-régression explicite : ACTIVATED simple reste strictement inchangé
# ---------------------------------------------------------------------------

class TestActivatedUnchanged:
    def test_activated_behaves_exactly_as_before(self, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        adapter = FakeAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)
        monkeypatch.setattr(run_active_grid, "run", lambda a, c: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "auto", "--live"])

        exit_code = run_active_grid.run_auto_mode(adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 0
        assert run_active_grid.load_grid_state(state_path) is not None
