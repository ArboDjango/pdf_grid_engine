"""Tests pour la réintégration de la policy V1 d'optimisation dans
run_active_grid.py (--mode create). Aucun accès réseau réel, aucun ordre
OKX pendant pytest.
"""

import sys
from decimal import Decimal

import pytest

import run_active_grid
from cell_order_identity import derive_root_order_id
from grid_preflight import run_preflight
from market_reader import MarketConditions, MarketReaderError
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
from trellis_calculator import GridGeometry
from grid_trading_controller import GridTradingController, GridTradingResult, GridTradingState


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
    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.001", minimum="0.1", p0="1.0013"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.ticker_price = Decimal(p0)
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.forbidden_calls = []
        self.get_ticker_calls = 0
        # Bougies légèrement haussières -> DI ratio > 1, atr_norm modéré (comme le cas réel).
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

    def get_account_config(self):
        from okx_spot_adapter import AccountConfig
        return AccountConfig(fee_type="1")

    def get_klines(self, inst_id, bar="15m", limit=50):
        return self._candles

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

    def place_market_buy(self, *a, **k):
        self.forbidden_calls.append("place_market_buy")
        raise AssertionError("ÉCRITURE INTERDITE : MARKET")


@pytest.fixture
def fake_adapter(monkeypatch):
    adapter = FakeAdapter()
    monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)
    return adapter


# ---------------------------------------------------------------------------
# 1 — la policy produit bien le PreflightConfig attendu
# ---------------------------------------------------------------------------

class TestPolicyProducesExpectedConfig:
    def test_1_optimized_config_has_flexible_geometry_and_p0_from_ticker(self, fake_adapter):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        assert config.geometry == GridGeometry.FLEXIBLE
        assert config.p0 == fake_adapter.ticker_price
        assert config.nu == 5
        assert config.nl == 5
        assert config.alpha == Decimal("0.95")
        assert config.operational_margin == Decimal("0.50")

    def test_1_gul_gll_derived_from_real_market_conditions_not_hardcoded(self, fake_adapter):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        # GUL/GLL doivent être calculés à partir de p0, pas des constantes
        # HISTORICAL_* (qui appartiennent au chemin resume, pas create).
        assert config.gul != run_active_grid.HISTORICAL_GUL
        assert config.gll != run_active_grid.HISTORICAL_GLL
        assert config.gul > config.p0 > config.gll


# ---------------------------------------------------------------------------
# 2 — P0/GLL/GUL calculés une seule fois
# ---------------------------------------------------------------------------

class TestComputedOnce:
    def test_2_get_ticker_called_exactly_once_during_creation(self, fake_adapter):
        run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        assert fake_adapter.get_ticker_calls == 1

    def test_2_get_ticker_never_called_again_during_subsequent_cycles(self, fake_adapter):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        calls_after_creation = fake_adapter.get_ticker_calls

        run_active_grid.run(fake_adapter, config, max_cycles=3, sleep_fn=lambda s: None)

        assert fake_adapter.get_ticker_calls == calls_after_creation  # inchangé pendant la boucle


# ---------------------------------------------------------------------------
# 3 — run() ne consulte jamais MarketReader
# ---------------------------------------------------------------------------

class TestRunNeverConsultsMarketReader:
    def test_3_run_never_imports_or_calls_market_reader(self, fake_adapter, monkeypatch):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")

        calls = []
        import market_reader as mr
        monkeypatch.setattr(mr, "read", lambda *a, **k: calls.append(1))

        run_active_grid.run(fake_adapter, config, max_cycles=3, sleep_fn=lambda s: None)

        assert calls == []  # jamais rappelé pendant la boucle

    def test_3_run_source_never_references_market_reader(self):
        import inspect
        source = inspect.getsource(run_active_grid.run)
        assert "market_reader" not in source
        assert "MarketReader" not in source


# ---------------------------------------------------------------------------
# 4 — plusieurs cycles conservent exactement le même config
# ---------------------------------------------------------------------------

class TestSameConfigAcrossCycles:
    def test_4_id_config_unchanged_across_cycles_create_mode(self, fake_adapter):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        config_id = id(config)

        run_active_grid.run(fake_adapter, config, max_cycles=4, sleep_fn=lambda s: None)

        assert id(config) == config_id


# ---------------------------------------------------------------------------
# 5 — le treillis reste immuable
# ---------------------------------------------------------------------------

class TestTrellisImmutable:
    def test_5_trellis_identical_across_multiple_preflight_calls(self, fake_adapter):
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        report1 = run_preflight(fake_adapter, config)
        report2 = run_preflight(fake_adapter, config)
        report3 = run_preflight(fake_adapter, config)
        assert report1.trellis == report2.trellis == report3.trellis


# ---------------------------------------------------------------------------
# 6 — validation économique INVALID empêche l'activation
# ---------------------------------------------------------------------------

class TestInvalidValidationBlocksCreation:
    def test_6_invalid_candidate_raises_grid_creation_refused(self, fake_adapter, monkeypatch):
        # Force une validation INVALID sans toucher à la formule économique
        # elle-même -- on intercepte uniquement le résultat retourné.
        from grid_economic_validator import GridEconomicValidation
        monkeypatch.setattr(run_active_grid, "validate_candidate", lambda **k: GridEconomicValidation(valid=False, violations=()))

        with pytest.raises(run_active_grid.GridCreationRefused):
            run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")

    def test_6_no_config_returned_and_no_side_effect_on_invalid(self, fake_adapter, monkeypatch):
        from grid_economic_validator import GridEconomicValidation
        monkeypatch.setattr(run_active_grid, "validate_candidate", lambda **k: GridEconomicValidation(valid=False, violations=()))

        try:
            run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        except run_active_grid.GridCreationRefused:
            pass

        assert fake_adapter.placed_orders == []
        assert fake_adapter.cancelled_orders == []

    def test_6_main_create_mode_returns_nonzero_and_no_write_on_invalid(self, fake_adapter, monkeypatch, capsys):
        from grid_economic_validator import GridEconomicValidation
        monkeypatch.setattr(run_active_grid, "validate_candidate", lambda **k: GridEconomicValidation(valid=False, violations=()))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create"])

        exit_code = run_active_grid.main()

        assert exit_code != 0
        assert fake_adapter.placed_orders == []
        assert "refusée" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# 7, 8, 9 — non-régression : repost même gi, tie-break, reconstruction OKX
# ---------------------------------------------------------------------------

class TestNonRegressionCoreMechanisms:
    def test_7_8_repost_same_gi_and_tiebreak_unchanged_with_optimized_config(self, fake_adapter):
        """
        Reproduit le cas réel déjà verrouillé (BUY 0.9877 FILLED, SELL
        1.0079 LIVE) mais avec un config produit par build_optimized_config
        au lieu de build_historical_config -- vérifie que le repost au
        même gi et le tie-break (préférence pour un ordre déjà ouvert)
        fonctionnent identiquement, quelle que soit l'origine de `config`.
        """
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        report = run_preflight(fake_adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi = lower_levels[-1]
        sell_gi = upper_levels[0]

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        fake_adapter.fills = (Fill(
            trade_id="T1", order_id="O1", client_order_id=buy_root,
            inst_id="XRP-USDC", side="BUY", fill_price=Decimal(str(buy_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=100,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        fake_adapter.open_orders = (
            OrderSnapshot("O2", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        run_active_grid.run(fake_adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        # Tie-break : l'ordre déjà ouvert (sell_gi) est conservé.
        assert sell_root not in fake_adapter.cancelled_orders
        assert fake_adapter.cancelled_orders == []

    def test_9_okx_reconstruction_unchanged_with_optimized_config(self, fake_adapter):
        """La reconstruction depuis l'état réel OKX fonctionne identiquement, config créé ou repris."""
        config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        report = run_preflight(fake_adapter, config)

        from cell_state_reconstruction import reconstruct_cell_states
        gi = float(report.trellis[0])
        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        fake_adapter.open_orders = (
            OrderSnapshot("O1", buy_root, "XRP-USDC", "BUY", Decimal(str(gi)), Decimal("1"), Decimal("0"), "live", "post_only", 1, 1),
        )

        states = reconstruct_cell_states(fake_adapter, report, fake_adapter.open_orders)
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        from cell import CellState
        assert matching.cell.state is CellState.CASH_HELD  # BUY ouvert -> cash
        assert matching.last_client_order_id == buy_root


# ---------------------------------------------------------------------------
# 10 — aucun ordre réel pendant les tests
# ---------------------------------------------------------------------------

class TestNoRealOrderDuringTests:
    def test_10_no_write_call_in_readonly_create_mode(self, fake_adapter, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create"])  # sans --live

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert fake_adapter.placed_orders == []
        assert fake_adapter.cancelled_orders == []
        assert fake_adapter.forbidden_calls == []
        output = capsys.readouterr().out
        assert "READ ONLY" in output

    def test_10_no_write_call_building_config_alone(self, fake_adapter):
        run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        assert fake_adapter.placed_orders == []
        assert fake_adapter.cancelled_orders == []
        assert fake_adapter.forbidden_calls == []


# ---------------------------------------------------------------------------
# 11 — persistance de la nouvelle configuration après --mode create --live
#      (correctif : save_grid_state() était absente de ce chemin)
# ---------------------------------------------------------------------------

class TestCreateLivePersistsAfterActivated:
    def test_11_persists_new_config_after_activated(self, fake_adapter, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        # run() lui-même (la boucle infinie) n'est pas l'objet de ce test --
        # neutralisé pour que main() retourne, sans jamais désactiver la
        # persistance elle-même (appelée AVANT ce point dans main()).
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        exit_code = run_active_grid.main()

        assert exit_code == 0
        loaded = run_active_grid.load_grid_state(state_path)
        assert loaded is not None
        assert loaded.p0 == fake_adapter.ticker_price
        assert loaded.inst_id == "XRP-USDC"

    def test_11_persisted_config_matches_activated_config_exactly(self, fake_adapter, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        expected_config = run_active_grid.build_optimized_config(fake_adapter, "XRP-USDC")
        run_active_grid.main()

        loaded = run_active_grid.load_grid_state(state_path)
        assert loaded.gul == expected_config.gul
        assert loaded.gll == expected_config.gll
        assert loaded.nu == expected_config.nu
        assert loaded.nl == expected_config.nl
        assert loaded.geometry == expected_config.geometry

    def test_11_no_persistence_when_activation_fails(self, fake_adapter, monkeypatch, tmp_path):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: GridTradingResult(GridTradingState.ERROR, "1", None, None, "échec simulé"),
        )
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        exit_code = run_active_grid.main()

        assert exit_code == 2
        assert run_active_grid.load_grid_state(state_path) is None
        import os
        assert not os.path.exists(state_path)

    def test_11_no_persistence_when_exposure_conflict(self, fake_adapter, monkeypatch, tmp_path):
        """PARTIAL/EXPOSURE_CONFLICT (tout état != ACTIVATED) : même garde-fou."""
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        monkeypatch.setattr(
            GridTradingController, "run",
            lambda self, *a, **k: GridTradingResult(GridTradingState.EXPOSURE_CONFLICT, "1", None, None, "conflit simulé"),
        )
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert run_active_grid.load_grid_state(state_path) is None

    def test_11_persistence_happens_before_run_loop_entry(self, fake_adapter, monkeypatch, tmp_path):
        """L'écriture doit être visible AVANT l'entrée dans run() -- vérifié
        en lisant le fichier depuis l'intérieur du stub de run() lui-même."""
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)

        observed = {}

        def spy_run(adapter, config):
            observed["loaded_at_run_entry"] = run_active_grid.load_grid_state(state_path)

        monkeypatch.setattr(run_active_grid, "run", spy_run)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        assert observed["loaded_at_run_entry"] is not None

    def test_11_atomic_write_no_tmp_file_left_behind(self, fake_adapter, monkeypatch, tmp_path):
        import os
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config: None)
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live"])

        run_active_grid.main()

        remaining = os.listdir(tmp_path)
        assert all(not name.startswith(".active_grid_state_") for name in remaining)

    def test_11_auto_mode_path_unaffected(self, fake_adapter, monkeypatch, tmp_path):
        """Non-régression explicite : --mode auto (déjà fonctionnel) reste
        strictement inchangé par ce correctif."""
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        monkeypatch.setattr(run_active_grid, "state_file_path_for", lambda inst_id: state_path)
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config: None)

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="XRP-USDC", state_path=state_path)

        assert exit_code == 0
        assert run_active_grid.load_grid_state(state_path) is not None
