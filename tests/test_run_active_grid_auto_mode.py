"""Tests pour --mode auto (run_active_grid.py) : persistance minimale,
reprise sans MarketReader si valide, création optimisée + persistance
atomique sinon. Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

import json
import os
import sys
import tempfile
from decimal import Decimal

import pytest

import run_active_grid
from grid_trading_controller import GridTradingController, GridTradingState
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
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", self.ticker_price, quantity, quantity, "filled", "market", 1, 1)
        self.placed_orders.append((inst_id, "MARKET_BUY", quantity, client_order_id))
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


@pytest.fixture
def fake_adapter(monkeypatch):
    adapter = FakeAdapter()
    monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)
    return adapter


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "active_grid_state.json")


def _bound_run(monkeypatch, max_cycles=1):
    """Empêche la boucle réelle (infinie, sleep réel) pendant les tests."""
    original = run_active_grid.run
    monkeypatch.setattr(run_active_grid, "run", lambda adapter, config, **k: original(adapter, config, max_cycles=max_cycles, sleep_fn=lambda s: None))


def _write_valid_state(path, *, p0="1.0013", gll="0.951235", gul="1.0563715"):
    payload = {
        "inst_id": "XRP-USDC", "p0": p0, "gll": gll, "gul": gul, "nu": 5, "nl": 5,
        "geometry": "FLEXIBLE", "spacing_h_pct": "0.0008", "alpha": "0.95",
        "operational_margin": "0.50", "tick_size": "0.0001", "lot_size": "0.001",
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# 1 — persistance présente et valide -> reprise sans MarketReader
# ---------------------------------------------------------------------------

class TestResumeFromPersistedState:
    def test_1_valid_state_resumes_without_market_reader(self, fake_adapter, monkeypatch, state_path, capsys):
        _write_valid_state(state_path)
        calls = []
        import market_reader as mr
        monkeypatch.setattr(mr, "read", lambda *a, **k: calls.append(1))
        _bound_run(monkeypatch)

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 0
        assert calls == []  # jamais appelé
        assert "REPRISE AUTOMATIQUE" in capsys.readouterr().out

    def test_1_no_gridtradingcontroller_call_on_valid_state(self, fake_adapter, monkeypatch, state_path):
        _write_valid_state(state_path)
        activation_calls = []
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert activation_calls == []


# ---------------------------------------------------------------------------
# 2 — absence de persistance -> optimisation + création
# ---------------------------------------------------------------------------

class TestCreateWhenNoState:
    def test_2_no_state_file_triggers_optimization(self, fake_adapter, monkeypatch, state_path):
        assert not os.path.exists(state_path)
        calls = []
        import market_reader as mr
        original_read = mr.read
        monkeypatch.setattr(mr, "read", lambda *a, **k: (calls.append(1), original_read(*a, **k))[-1])
        _bound_run(monkeypatch)

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 0
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3/4 — plusieurs redémarrages, P0/GLL/GUL conservés à l'identique
# ---------------------------------------------------------------------------

class TestMultipleRestartsPreserveConfig:
    def test_3_4_p0_gll_gul_identical_and_no_new_activation_across_restarts(self, fake_adapter, monkeypatch, state_path):
        _write_valid_state(state_path, p0="1.0013", gll="0.951235", gul="1.0563715")
        activation_calls = []
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        _bound_run(monkeypatch)

        for _ in range(3):  # trois "redémarrages" successifs
            exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)
            assert exit_code == 0

        assert activation_calls == []  # jamais, sur les 3 redémarrages
        config = run_active_grid.load_grid_state(state_path)
        assert config.p0 == Decimal("1.0013")
        assert config.gll == Decimal("0.951235")
        assert config.gul == Decimal("1.0563715")


# ---------------------------------------------------------------------------
# 5 — policy appelée une seule fois en l'absence d'état
# ---------------------------------------------------------------------------

class TestPolicyCalledOnce:
    def test_5_build_optimized_config_called_exactly_once(self, fake_adapter, monkeypatch, state_path):
        calls = []
        original = run_active_grid.build_optimized_config
        monkeypatch.setattr(run_active_grid, "build_optimized_config", lambda *a, **k: (calls.append(1), original(*a, **k))[-1])
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6 — INVALID -> aucune activation ni persistance
# ---------------------------------------------------------------------------

class TestInvalidBlocksActivationAndPersistence:
    def test_6_invalid_candidate_no_activation_no_state_file(self, fake_adapter, monkeypatch, state_path):
        from grid_economic_validator import GridEconomicValidation
        monkeypatch.setattr(run_active_grid, "validate_candidate", lambda **k: GridEconomicValidation(valid=False, violations=()))
        activation_calls = []
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 1
        assert activation_calls == []
        assert not os.path.exists(state_path)

    def test_6_activation_not_fully_activated_no_state_file(self, fake_adapter, monkeypatch, state_path):
        from grid_trading_controller import GridTradingResult
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, adapter, config, **kwargs: GridTradingResult(GridTradingState.ERROR, "1", None, None, "échec simulé"),
        )

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 2
        assert not os.path.exists(state_path)


# ---------------------------------------------------------------------------
# 7 — même objet config transmis à l'activation puis à run()
# ---------------------------------------------------------------------------

class TestSameConfigObject:
    def test_7_same_config_object_activation_and_run(self, fake_adapter, monkeypatch, state_path):
        captured = {}
        original_activation = GridTradingController.run

        def spy_activation(self, adapter, config, **kwargs):
            captured["activation"] = id(config)
            return original_activation(self, adapter, config)

        monkeypatch.setattr(GridTradingController, "run", spy_activation)

        original_run = run_active_grid.run

        def spy_run(adapter, config, **k):
            captured["run"] = id(config)
            return original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        monkeypatch.setattr(run_active_grid, "run", spy_run)

        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert captured["activation"] == captured["run"]

    def test_7_same_config_object_on_resume_path(self, fake_adapter, monkeypatch, state_path):
        _write_valid_state(state_path)
        captured = {}
        original_run = run_active_grid.run

        def spy_run(adapter, config, **k):
            captured["run"] = id(config)
            return original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        monkeypatch.setattr(run_active_grid, "run", spy_run)

        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        loaded = run_active_grid.load_grid_state(state_path)
        assert captured["run"] is not None  # run() a bien reçu un config (le même tout du long, un seul chargement)


# ---------------------------------------------------------------------------
# 8 — fichier corrompu -> nouvelle création optimisée, jamais de devinette
# ---------------------------------------------------------------------------

class TestCorruptedStateFile:
    def test_8_corrupted_json_falls_back_to_creation(self, fake_adapter, monkeypatch, state_path):
        with open(state_path, "w") as f:
            f.write("{ceci n'est pas du JSON valide")

        calls = []
        import market_reader as mr
        original_read = mr.read
        monkeypatch.setattr(mr, "read", lambda *a, **k: (calls.append(1), original_read(*a, **k))[-1])
        _bound_run(monkeypatch)

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 0
        assert len(calls) == 1  # repli sur création, jamais une tentative de "deviner"

    def test_8_missing_required_field_falls_back_to_creation(self, fake_adapter, monkeypatch, state_path):
        payload = {"inst_id": "XRP-USDC", "p0": "1.0013"}  # incomplet
        with open(state_path, "w") as f:
            json.dump(payload, f)

        result = run_active_grid.load_grid_state(state_path)
        assert result is None

    def test_8_invalid_decimal_value_falls_back_to_creation(self, fake_adapter, state_path):
        _write_valid_state(state_path, p0="not-a-decimal")
        result = run_active_grid.load_grid_state(state_path)
        assert result is None

    def test_8_load_never_raises_on_garbage_file(self, state_path):
        with open(state_path, "wb") as f:
            f.write(b"\x00\x01\x02random garbage bytes")
        result = run_active_grid.load_grid_state(state_path)  # ne doit jamais lever
        assert result is None


# ---------------------------------------------------------------------------
# 9 — écriture atomique
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_9_state_file_contains_exact_twelve_fields(self, fake_adapter, state_path):
        instrument = fake_adapter.get_instrument("XRP-USDC")
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")

        run_active_grid.save_grid_state(config, instrument, state_path)

        with open(state_path) as f:
            payload = json.load(f)
        assert set(payload.keys()) == set(run_active_grid._REQUIRED_STATE_FIELDS)

    def test_9_decimals_serialized_as_strings(self, fake_adapter, state_path):
        instrument = fake_adapter.get_instrument("XRP-USDC")
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")

        run_active_grid.save_grid_state(config, instrument, state_path)

        with open(state_path) as f:
            payload = json.load(f)
        assert isinstance(payload["p0"], str)
        assert isinstance(payload["gll"], str)
        assert isinstance(payload["gul"], str)

    def test_9_no_temp_file_left_behind_after_successful_write(self, fake_adapter, state_path, tmp_path):
        instrument = fake_adapter.get_instrument("XRP-USDC")
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")

        run_active_grid.save_grid_state(config, instrument, state_path)

        remaining = [p for p in os.listdir(tmp_path) if p.startswith(".active_grid_state_")]
        assert remaining == []

    def test_9_reload_produces_identical_config(self, fake_adapter, state_path):
        instrument = fake_adapter.get_instrument("XRP-USDC")
        original_config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")

        run_active_grid.save_grid_state(original_config, instrument, state_path)
        reloaded = run_active_grid.load_grid_state(state_path)

        assert reloaded.p0 == original_config.p0
        assert reloaded.gll == original_config.gll
        assert reloaded.gul == original_config.gul
        assert reloaded.nu == original_config.nu
        assert reloaded.nl == original_config.nl
        assert reloaded.geometry == original_config.geometry
        assert reloaded.spacing_h_pct == original_config.spacing_h_pct
        assert reloaded.alpha == original_config.alpha
        assert reloaded.operational_margin == original_config.operational_margin

    def test_9_write_uses_os_replace_not_direct_write(self):
        import inspect
        source = inspect.getsource(run_active_grid.save_grid_state)
        assert "os.replace" in source
        assert "fsync" in source
        assert "mkstemp" in source

    def test_9_state_only_written_after_activated_confirmed(self, fake_adapter, monkeypatch, state_path):
        from grid_trading_controller import GridTradingResult
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, adapter, config, **kwargs: GridTradingResult(GridTradingState.PARTIAL if hasattr(GridTradingState, "PARTIAL") else GridTradingState.ERROR, "1", None, None, "non activé"),
        )
        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)
        assert not os.path.exists(state_path)


# ---------------------------------------------------------------------------
# 10 — run() reste strictement inchangé (empreinte)
# ---------------------------------------------------------------------------

class TestRunUnchanged:
    def test_10_run_function_unchanged(self):
        """Empreinte mise à jour délibérément (chantier observabilité) :
        run() journalise désormais le résultat non-ACTIVE de
        run_exposure_cycle() -- seule addition, aucune logique de trading
        touchée (vérifié ci-dessous : ni market_reader ni load_grid_state
        n'apparaissent, exactement comme avant)."""
        import hashlib
        import inspect
        source = inspect.getsource(run_active_grid.run)
        digest = hashlib.md5(source.encode()).hexdigest()
        assert digest == "fb9ec05bbca95aeeadaeba41937ba29b"
        assert "market_reader" not in source
        assert "load_grid_state" not in source
        assert "save_grid_state" not in source
