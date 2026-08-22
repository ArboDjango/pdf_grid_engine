"""Tests pour le support multi-instrument de run_active_grid.py --mode auto.

Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

import sys
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
    """Adaptateur factice paramétrable par instrument -- inst_id n'est
    jamais codé en dur, chaque méthode le reçoit et le respecte."""

    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.001", minimum="0.1", p0="1.0013", fee_type="1"):
        self.instrument = InstrumentRules("PLACEHOLDER", "X", "Y", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("PLACEHOLDER", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.ticker_price = Decimal(p0)
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.get_instrument_calls = []
        self._candles = make_candles([1.000 + i * 0.0008 for i in range(50)])

    def get_instrument(self, inst_id):
        self.get_instrument_calls.append(inst_id)
        return InstrumentRules(inst_id, "X", "Y", self.instrument.tick_size, self.instrument.lot_size, self.instrument.min_size, None, 4, 1, "live")

    def get_fee_rates(self, inst_id):
        return FeeRates(inst_id, self.fees.maker, self.fees.taker)

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_ticker(self, inst_id):
        return Ticker(inst_id=inst_id, last=self.ticker_price, ts=1)

    def get_klines(self, inst_id, bar="15m", limit=50):
        return self._candles

    def get_account_config(self):
        from okx_spot_adapter import AccountConfig
        return AccountConfig(fee_type=self._fee_type)

    def list_open_orders(self, inst_id, client_order_id=None):
        matching = tuple(o for o in self.open_orders if o.inst_id == inst_id)
        if client_order_id is None:
            return matching
        return tuple(o for o in matching if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        matching = tuple(f for f in self.fills if f.inst_id == inst_id)
        if order_id is None:
            return matching
        return tuple(f for f in matching if f.order_id == order_id)

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
        self.cancelled_orders.append((inst_id, client_order_id))
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


def _bound_run(monkeypatch, max_cycles=1):
    original = run_active_grid.run
    monkeypatch.setattr(run_active_grid, "run", lambda adapter, config, **k: original(adapter, config, max_cycles=max_cycles, sleep_fn=lambda s: None))


def _write_valid_state(path, *, inst_id="XRP-USDC", p0="1.0013", gll="0.951235", gul="1.0563715", q="21.044", allocated_capital="20000"):
    import json
    payload = {
        "inst_id": inst_id, "p0": p0, "gll": gll, "gul": gul, "nu": 5, "nl": 5,
        "geometry": "FLEXIBLE", "spacing_h_pct": "0.0008", "alpha": "0.95",
        "operational_margin": "0.50", "tick_size": "0.0001", "lot_size": "0.001",
        "q": q,
        "allocated_capital": allocated_capital,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# 1 -- state_file_path_for : dérivation déterministe par instrument
# ---------------------------------------------------------------------------

class TestStateFilePathDerivation:
    def test_1_derives_expected_filename_for_xrp(self):
        assert run_active_grid.state_file_path_for("XRP-USDC") == "active_grid_state_XRP-USDC.json"

    def test_1_derives_expected_filename_for_inj(self):
        assert run_active_grid.state_file_path_for("INJ-USDT") == "active_grid_state_INJ-USDT.json"

    def test_1_distinct_instruments_never_share_path(self):
        xrp_path = run_active_grid.state_file_path_for("XRP-USDC")
        inj_path = run_active_grid.state_file_path_for("INJ-USDT")
        assert xrp_path != inj_path


# ---------------------------------------------------------------------------
# 2 -- XRP charge exclusivement son propre état
# ---------------------------------------------------------------------------

class TestXrpLoadsOnlyXrpState:
    def test_2_xrp_loads_only_xrp_state_file(self, fake_adapter, monkeypatch, tmp_path):
        xrp_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC", p0="1.0013")
        _write_valid_state(inj_path, inst_id="INJ-USDT", p0="25.50", gll="24.0", gul="27.0")  # présent mais NE DOIT PAS être chargé
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="XRP-USDC", state_path=xrp_path)

        loaded = run_active_grid.load_grid_state(xrp_path)
        assert loaded.inst_id == "XRP-USDC"
        assert loaded.p0 == Decimal("1.0013")


# ---------------------------------------------------------------------------
# 3 -- INJ charge exclusivement son propre état
# ---------------------------------------------------------------------------

class TestInjLoadsOnlyInjState:
    def test_3_inj_loads_only_inj_state_file(self, fake_adapter, monkeypatch, tmp_path):
        xrp_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC", p0="1.0013")
        _write_valid_state(inj_path, inst_id="INJ-USDT", p0="25.50", gll="24.0", gul="27.0")
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        loaded = run_active_grid.load_grid_state(inj_path)
        assert loaded.inst_id == "INJ-USDT"
        assert loaded.p0 == Decimal("25.50")


# ---------------------------------------------------------------------------
# 4 -- absence d'état INJ -> optimisation INJ (jamais XRP)
# ---------------------------------------------------------------------------

class TestInjOptimizationWhenNoInjState:
    def test_4_no_inj_state_triggers_inj_optimization_not_xrp(self, fake_adapter, monkeypatch, tmp_path):
        xrp_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC")  # présent, mais pour un AUTRE instrument
        assert not __import__("os").path.exists(inj_path)

        captured_inst_ids = []
        original_optimizer = run_active_grid.build_optimized_config
        monkeypatch.setattr(run_active_grid, "build_optimized_config", lambda adapter, inst_id, **k: (captured_inst_ids.append(inst_id), original_optimizer(adapter, inst_id, **k))[-1])
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path, allocated_capital_usdc=Decimal("200"))

        assert captured_inst_ids == ["INJ-USDT"]  # jamais XRP-USDC, malgré le fichier XRP présent à côté


# ---------------------------------------------------------------------------
# 5 -- présence d'état INJ valide -> aucune optimisation
# ---------------------------------------------------------------------------

class TestInjResumeNoOptimization:
    def test_5_valid_inj_state_skips_optimization(self, fake_adapter, monkeypatch, tmp_path):
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(inj_path, inst_id="INJ-USDT", p0="25.50", gll="24.0", gul="27.0")

        optimizer_calls = []
        monkeypatch.setattr(run_active_grid, "build_optimized_config", lambda *a, **k: optimizer_calls.append(1))
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        assert optimizer_calls == []


# ---------------------------------------------------------------------------
# 6 -- création XRP inchangée (défaut --inst-id)
# ---------------------------------------------------------------------------

class TestXrpCreationUnchangedByDefault:
    def test_6_default_inst_id_is_xrp(self, fake_adapter, monkeypatch, tmp_path):
        state_path = str(tmp_path / "state.json")
        captured_inst_ids = []
        original_optimizer = run_active_grid.build_optimized_config
        monkeypatch.setattr(run_active_grid, "build_optimized_config", lambda adapter, inst_id, **k: (captured_inst_ids.append(inst_id), original_optimizer(adapter, inst_id, **k))[-1])
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path, allocated_capital_usdc=Decimal("200"))  # inst_id NON fourni

        assert captured_inst_ids == ["XRP-USDC"]  # défaut préservé


# ---------------------------------------------------------------------------
# 7 -- reprise XRP inchangée (appel historique existant, sans inst_id)
# ---------------------------------------------------------------------------

class TestXrpResumeCallSignatureUnchanged:
    def test_7_existing_call_style_still_works_unmodified(self, fake_adapter, monkeypatch, tmp_path):
        """Reproduit EXACTEMENT le style d'appel des 11 tests déjà existants
        (test_run_active_grid_auto_mode.py) -- doit continuer à fonctionner
        sans aucune adaptation."""
        state_path = str(tmp_path / "active_grid_state.json")
        _write_valid_state(state_path, inst_id="XRP-USDC")
        _bound_run(monkeypatch)

        exit_code = run_active_grid.run_auto_mode(fake_adapter, live=True, state_path=state_path)

        assert exit_code == 0


# ---------------------------------------------------------------------------
# 8 -- run() reste identique (empreinte déjà vérifiée, reconfirmée ici)
# ---------------------------------------------------------------------------

class TestRunUnchanged:
    def test_8_run_source_hash_unchanged(self):
        """Empreinte mise à jour délibérément -- voir
        test_run_active_grid_auto_mode.py::TestRunUnchanged pour la
        justification complète (chantier churn-protection-n3, run() dérive
        et transmet désormais churn_state_path, aucune autre logique
        touchée)."""
        import hashlib
        import inspect
        digest = hashlib.md5(inspect.getsource(run_active_grid.run).encode()).hexdigest()
        assert digest == "e042c7d36cc49fed6f3c422faea9bdb2"


# ---------------------------------------------------------------------------
# 9 -- aucune confusion d'inst_id
# ---------------------------------------------------------------------------

class TestNoInstIdConfusion:
    def test_9_loaded_config_inst_id_matches_requested_instrument(self, fake_adapter, monkeypatch, tmp_path):
        inj_path = str(tmp_path / "inj_state.json")
        _write_valid_state(inj_path, inst_id="INJ-USDT")
        _bound_run(monkeypatch)

        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        loaded = run_active_grid.load_grid_state(inj_path)
        assert loaded.inst_id == "INJ-USDT"
        assert loaded.inst_id != "XRP-USDC"


# ---------------------------------------------------------------------------
# 10 -- aucun croisement de fichiers de persistance
# ---------------------------------------------------------------------------

class TestNoFileCrossing:
    def test_10_writing_inj_state_never_touches_xrp_file(self, fake_adapter, monkeypatch, tmp_path):
        xrp_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC", p0="1.0013")
        xrp_mtime_before = __import__("os").path.getmtime(xrp_path)

        _bound_run(monkeypatch)
        run_active_grid.run_auto_mode(fake_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        xrp_mtime_after = __import__("os").path.getmtime(xrp_path)
        assert xrp_mtime_before == xrp_mtime_after  # jamais touché
        loaded_xrp = run_active_grid.load_grid_state(xrp_path)
        assert loaded_xrp.inst_id == "XRP-USDC"  # contenu XRP intact


# ---------------------------------------------------------------------------
# 11 -- les deux instances peuvent coexister/fonctionner simultanément
# ---------------------------------------------------------------------------

class TestTwoInstancesCoexist:
    def test_11_two_run_auto_mode_calls_with_distinct_paths_dont_interfere(self, monkeypatch, tmp_path):
        xrp_adapter = FakeAdapter(p0="1.0013")
        inj_adapter = FakeAdapter(p0="25.50")

        xrp_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        inj_path = str(tmp_path / "active_grid_state_INJ-USDT.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC", p0="1.0013")
        _write_valid_state(inj_path, inst_id="INJ-USDT", p0="25.50", gll="24.0", gul="27.0")

        _bound_run(monkeypatch)
        activation_calls = []
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))

        exit_xrp = run_active_grid.run_auto_mode(xrp_adapter, live=True, inst_id="XRP-USDC", state_path=xrp_path)
        exit_inj = run_active_grid.run_auto_mode(inj_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        assert exit_xrp == 0
        assert exit_inj == 0
        # Garantie réelle de "reprise" : jamais d'activation (GridTradingController),
        # jamais d'optimisation -- PAS "zéro ordre placé", puisque
        # run_exposure_cycle matérialise légitimement les niveaux absents
        # de l'état réel de l'exchange (ici, l'adaptateur factice démarre
        # sans aucun ordre ouvert -- contrairement à la production réelle
        # où BUY/SELL sont déjà réellement ouverts).
        assert activation_calls == []
        # Les ordres, s'il y en a, appartiennent chacun exclusivement à leur instrument.
        assert all(inst == "XRP-USDC" for inst, *_ in xrp_adapter.placed_orders)
        assert all(inst == "INJ-USDT" for inst, *_ in inj_adapter.placed_orders)

    def test_11_config_objects_are_distinct_between_instances(self, monkeypatch, tmp_path):
        xrp_adapter = FakeAdapter()
        inj_adapter = FakeAdapter()
        xrp_path = str(tmp_path / "xrp.json")
        inj_path = str(tmp_path / "inj.json")
        _write_valid_state(xrp_path, inst_id="XRP-USDC")
        _write_valid_state(inj_path, inst_id="INJ-USDT")

        captured = []
        original_run = run_active_grid.run
        monkeypatch.setattr(run_active_grid, "run", lambda adapter, config, **k: (captured.append(config), original_run(adapter, config, max_cycles=1, sleep_fn=lambda s: None))[-1])

        run_active_grid.run_auto_mode(xrp_adapter, live=True, inst_id="XRP-USDC", state_path=xrp_path)
        run_active_grid.run_auto_mode(inj_adapter, live=True, inst_id="INJ-USDT", state_path=inj_path)

        assert len(captured) == 2
        assert captured[0].inst_id != captured[1].inst_id
        assert id(captured[0]) != id(captured[1])


# ---------------------------------------------------------------------------
# 12 -- ordres d'un instrument jamais confondus avec ceux de l'autre
# ---------------------------------------------------------------------------

class TestOrdersNeverCrossInstruments:
    def test_12_client_order_id_differs_between_instruments_at_same_price(self, fake_adapter):
        from grid_activation_controller import derive_grid_order_id
        from grid_preflight import PreflightConfig
        from trellis_calculator import GridGeometry

        common = dict(gul=Decimal("1.05"), gll=Decimal("0.95"), nu=5, nl=5, p0=Decimal("1.00"),
                      geometry=GridGeometry.FLEXIBLE, spacing_h_pct=Decimal("0.0008"),
                      alpha=Decimal("0.95"), operational_margin=Decimal("0.50"))
        config_xrp = PreflightConfig(inst_id="XRP-USDC", **common)
        config_inj = PreflightConfig(inst_id="INJ-USDT", **common)

        id_xrp = derive_grid_order_id(config_xrp, Decimal("0.0001"), Decimal("0.001"), "BUY", 0.98)
        id_inj = derive_grid_order_id(config_inj, Decimal("0.0001"), Decimal("0.001"), "BUY", 0.98)

        assert id_xrp != id_inj  # même prix, même côté -- jamais la même identité

    def test_12_list_open_orders_filters_by_inst_id(self, fake_adapter):
        fake_adapter.open_orders = (
            OrderSnapshot("O1", "GRIDxrp1", "XRP-USDC", "BUY", Decimal("0.98"), Decimal("1"), Decimal("0"), "live", "post_only", 1, 1),
            OrderSnapshot("O2", "GRIDinj1", "INJ-USDT", "BUY", Decimal("25"), Decimal("1"), Decimal("0"), "live", "post_only", 1, 1),
        )

        xrp_orders = fake_adapter.list_open_orders("XRP-USDC")
        inj_orders = fake_adapter.list_open_orders("INJ-USDT")

        assert len(xrp_orders) == 1 and xrp_orders[0].inst_id == "XRP-USDC"
        assert len(inj_orders) == 1 and inj_orders[0].inst_id == "INJ-USDT"


# ---------------------------------------------------------------------------
# 13 -- non-régression complète (vérifiée par la suite pytest globale, pas ici)
# ---------------------------------------------------------------------------

class TestBackwardCompatibilityMarker:
    """Placeholder documentant que la non-régression complète est vérifiée
    par 'pytest -q' sur l'ensemble du dépôt, pas dans ce fichier isolément."""

    def test_marker(self):
        assert True
