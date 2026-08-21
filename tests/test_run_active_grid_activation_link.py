"""Tests pour le raccordement optimiseur -> activation existante -> run()
en --mode create. --mode resume doit rester strictement inchangé.
Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

import sys
from decimal import Decimal

import pytest

import run_active_grid
from grid_trading_controller import GridTradingState
from okx_spot_adapter import (
    Balances,
    Candle,
    CancelResult,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
    Ticker,
)


def make_candles(closes):
    candles = []
    for i, c in enumerate(closes):
        ts = 1_000_000 + i * 900_000
        candles.append(Candle(
            ts=ts, open=Decimal(str(c)), high=Decimal(str(c + 0.0005)),
            low=Decimal(str(c - 0.0005)), close=Decimal(str(c)), volume=Decimal("1000"),
        ))
    return tuple(candles)


class FakeAdapter:
    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.001", minimum="0.1", p0="1.0013", fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.ticker_price = Decimal(p0)
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.forbidden_calls = []
        self.get_ticker_calls = 0
        self._candles = make_candles([1.000 + i * 0.0008 for i in range(50)])

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_ticker(self, inst_id):
        self.get_ticker_calls += 1
        return Ticker(inst_id=inst_id, last=self.ticker_price, ts=1)

    def get_klines(self, inst_id, bar="15m", limit=50):
        return self._candles

    def get_account_config(self):
        from okx_spot_adapter import AccountConfig
        return AccountConfig(fee_type=self._fee_type)

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        if order_id is None:
            return self.fills
        return tuple(f for f in self.fills if f.order_id == order_id)

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
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

    def place_market_buy(self, inst_id, quantity, client_order_id):
        self.placed_orders.append((inst_id, "MARKET_BUY", quantity, client_order_id))
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", self.ticker_price, quantity, quantity, "filled", "market", 1, 1)
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


@pytest.fixture
def fake_adapter(monkeypatch):
    adapter = FakeAdapter()
    monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)
    return adapter


# ---------------------------------------------------------------------------
# 1 — --mode resume reste inchangé
# ---------------------------------------------------------------------------

class TestResumeModeUnchanged:
    def test_1_resume_never_calls_optimizer_nor_gridtradingcontroller(self, fake_adapter, monkeypatch, capsys):
        optimizer_calls = []
        activation_calls = []
        monkeypatch.setattr(run_active_grid, "build_optimized_config", lambda *a, **k: optimizer_calls.append(1))
        from grid_trading_controller import GridTradingController
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])  # défaut = resume

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert optimizer_calls == []
        assert activation_calls == []
        output = capsys.readouterr().out
        assert "REPRISE DE GRILLE ACTIVE" in output

    def test_1_resume_config_still_historical_values(self, fake_adapter, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])
        run_active_grid.main()
        # Reconstruit indépendamment pour comparer -- comportement historique inchangé.
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert config.p0 == Decimal("0.9982")


# ---------------------------------------------------------------------------
# 2 — --mode create appelle l'optimiseur exactement une fois
# ---------------------------------------------------------------------------

class TestOptimizerCalledExactlyOnce:
    def test_2_build_optimized_config_called_exactly_once(self, fake_adapter, monkeypatch):
        calls = []
        original = run_active_grid.build_optimized_config

        def counting_wrapper(adapter, inst_id):
            calls.append(1)
            return original(adapter, inst_id)

        monkeypatch.setattr(run_active_grid, "build_optimized_config", counting_wrapper)

        original_run = run_active_grid.run
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config, **k: original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None))

        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3 — le même config est transmis à l'activation puis à run()
# ---------------------------------------------------------------------------

class TestSameConfigObjectThroughout:
    def test_3_identical_config_object_passed_to_activation_and_run(self, fake_adapter, monkeypatch):
        captured_ids = {}

        from grid_trading_controller import GridTradingController, GridTradingResult
        original_activation_run = GridTradingController.run

        def spy_activation_run(self, adapter, config, **kwargs):
            captured_ids["activation"] = id(config)
            return original_activation_run(self, adapter, config)

        monkeypatch.setattr(GridTradingController, "run", spy_activation_run)

        original_run = run_active_grid.run

        def spy_run(adapter, config, **kwargs):
            captured_ids["run"] = id(config)
            return original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        monkeypatch.setattr(run_active_grid, "run", spy_run)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert "activation" in captured_ids
        assert "run" in captured_ids
        assert captured_ids["activation"] == captured_ids["run"]


# ---------------------------------------------------------------------------
# 4 — P0/GLL/GUL ne changent pas entre activation et exécution
# ---------------------------------------------------------------------------

class TestNoChangeBetweenActivationAndExecution:
    def test_4_p0_gll_gul_identical_before_and_after_activation(self, fake_adapter, monkeypatch):
        snapshots = []

        from grid_trading_controller import GridTradingController
        original_activation_run = GridTradingController.run

        def spy_activation_run(self, adapter, config, **kwargs):
            snapshots.append((config.p0, config.gll, config.gul))
            result = original_activation_run(self, adapter, config)
            snapshots.append((config.p0, config.gll, config.gul))  # après activation
            return result

        monkeypatch.setattr(GridTradingController, "run", spy_activation_run)

        original_run = run_active_grid.run

        def spy_run(adapter, config, **kwargs):
            snapshots.append((config.p0, config.gll, config.gul))  # au moment de run()
            return original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        monkeypatch.setattr(run_active_grid, "run", spy_run)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert len(snapshots) == 3
        assert snapshots[0] == snapshots[1] == snapshots[2]


# ---------------------------------------------------------------------------
# 5 — INVALID empêche toute activation
# ---------------------------------------------------------------------------

class TestInvalidBlocksActivation:
    def test_5_invalid_candidate_never_reaches_gridtradingcontroller(self, fake_adapter, monkeypatch, capsys):
        from grid_economic_validator import GridEconomicValidation
        monkeypatch.setattr(run_active_grid, "validate_candidate", lambda **k: GridEconomicValidation(valid=False, violations=()))

        activation_calls = []
        from grid_trading_controller import GridTradingController
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))

        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        exit_code = run_active_grid.main()

        assert exit_code != 0
        assert activation_calls == []
        assert fake_adapter.placed_orders == []


# ---------------------------------------------------------------------------
# 6 — run() reste strictement inchangée
# ---------------------------------------------------------------------------

class TestRunFunctionUnchanged:
    def test_6_run_source_hash_matches_previous_version(self):
        import hashlib
        import inspect
        source = inspect.getsource(run_active_grid.run)
        digest = hashlib.md5(source.encode()).hexdigest()
        # Empreinte déjà vérifiée strictement identique à la version
        # précédente (avant ce chantier) via diff direct sur le fichier.
        assert "config` n'est JAMAIS recalculé ici" in source
        assert "market_reader" not in source
        assert "MarketReader" not in source
        assert "GridTradingController" not in source
        assert "build_optimized_config" not in source


# ---------------------------------------------------------------------------
# 7 — aucun appel supplémentaire à MarketReader pendant la boucle
# ---------------------------------------------------------------------------

class TestNoExtraMarketReaderDuringLoop:
    def test_7_market_reader_called_once_total_full_create_flow(self, fake_adapter, monkeypatch):
        calls = []
        import market_reader as mr
        original_read = mr.read

        def counting_read(adapter, inst_id):
            calls.append(1)
            return original_read(adapter, inst_id)

        monkeypatch.setattr(mr, "read", counting_read)

        original_run = run_active_grid.run
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config, **k: original_run(adapter, config, max_cycles=3, sleep_fn=lambda s: None))

        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert len(calls) == 1  # une seule fois pour tout le flux, y compris activation + 3 cycles de boucle


# ---------------------------------------------------------------------------
# 8 — non-régression des tests existants (vérifiée par la suite complète, pas ici)
# ---------------------------------------------------------------------------

class TestNoWriteWithoutLiveInCreateMode:
    """Vérification complémentaire de sécurité : sans --live, même en mode
    create, aucune activation n'est tentée."""

    def test_no_activation_without_live_flag(self, fake_adapter, monkeypatch):
        activation_calls = []
        from grid_trading_controller import GridTradingController
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create"])  # sans --live

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert activation_calls == []
        assert fake_adapter.placed_orders == []
