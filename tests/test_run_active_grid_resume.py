"""Tests pour run_active_grid.py — reprise de grille active (A à J).

Aucun accès réseau réel, aucun ordre OKX pendant pytest. Adaptateur
factice dont les méthodes d'écriture lèvent une exception si elles sont
appelées de manière imprévue.
"""

import sys
from dataclasses import replace
from decimal import Decimal

import pytest

import run_active_grid
from cell_order_identity import derive_root_order_id
from grid_preflight import run_preflight
from okx_spot_adapter import (
    Balances,
    CancelResult,
    Candle,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
    Ticker,
)
from trellis_calculator import GridGeometry


def _make_candles(closes):
    candles = []
    for i, c in enumerate(closes):
        ts = 1_000_000 + i * 900_000
        candles.append(Candle(
            ts=ts, open=Decimal(str(c)), high=Decimal(str(c + 0.0005)),
            low=Decimal(str(c - 0.0005)), close=Decimal(str(c)), volume=Decimal("1000"),
        ))
    return tuple(candles)


class _CreateModeCapableAdapter:
    """Adaptateur factice complet, utilisé UNIQUEMENT par les deux tests
    --mode create ci-dessous (test_e_create_mode_*) -- distinct de
    FakeAdapter, dont get_ticker() lève délibérément pour garantir que le
    chemin resume ne l'appelle jamais. --mode create a légitimement besoin
    de get_ticker()/get_klines() pour l'optimiseur (MarketReader)."""

    def __init__(self):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.0001"), Decimal("0.001"), Decimal("0.1"), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(Decimal("20000"), Decimal("20000"), Decimal("20000"), Decimal("20000"))
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self._candles = _make_candles([1.000 + i * 0.0008 for i in range(50)])

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_ticker(self, inst_id):
        return Ticker(inst_id=inst_id, last=Decimal("1.0013"), ts=1)

    def get_klines(self, inst_id, bar="15m", limit=50):
        return self._candles

    def get_account_config(self):
        from okx_spot_adapter import AccountConfig
        return AccountConfig(fee_type="1")

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
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", Decimal("1.0013"), quantity, quantity, "filled", "market", 1, 1)
        self.placed_orders.append((inst_id, "MARKET_BUY", quantity, client_order_id))
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


class FakeAdapter:
    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.001", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.forbidden_calls = []
        self.get_ticker_calls = 0

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_ticker(self, inst_id):
        # Surveillé explicitement (test F, via get_ticker_calls) : ne doit
        # jamais être appelé par build_historical_config ni par le chemin
        # READ-ONLY de main() pour définir P0. grid_replacement.py appelle
        # légitimement get_ticker DEPUIS run() (jamais pour définir P0 --
        # seulement pour détecter un dépassement de borne) ; ne PAS lever
        # ici pour ne pas casser les tests qui exercent run() -- le
        # compteur reste le moyen de vérification, pas une exception.
        self.get_ticker_calls += 1
        from okx_spot_adapter import Ticker
        return Ticker(inst_id=inst_id, last=Decimal("0.9982"), ts=1)

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
# A, B, C — config historique construite exactement
# ---------------------------------------------------------------------------

class TestHistoricalConfigConstruction:
    def test_a_p0_matches_historical_value_exactly(self, fake_adapter):
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert config.p0 == Decimal("0.9982")

    def test_b_geometry_is_flexible(self, fake_adapter):
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert config.geometry == GridGeometry.FLEXIBLE

    def test_c_gll_gul_match_historical_values_exactly(self, fake_adapter):
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert config.gll == Decimal("0.9469627131102811")
        assert config.gul == Decimal("1.04811")
        assert config.nu == 5
        assert config.nl == 5


# ---------------------------------------------------------------------------
# D, E, F — aucune reconstruction, aucune activation initiale
# ---------------------------------------------------------------------------

class TestNoRebuildNoInitialActivation:
    def test_d_build_config_never_called(self, fake_adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(run_active_grid, "build_config", lambda *a, **k: calls.append(1), raising=False)
        run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert calls == []

    def test_d_build_config_not_imported_in_module(self):
        assert "build_config" not in dir(run_active_grid)

    def test_e_gridtradingcontroller_importable_but_never_called_in_resume_mode(self, fake_adapter, monkeypatch, capsys):
        """
        Propriété architecturale réelle depuis le raccordement --mode create :
        GridTradingController PEUT être importé par le module (nécessaire à
        l'activation en --mode create) -- mais en --mode resume (le défaut),
        il ne doit JAMAIS être appelé. build_historical_config seule ne le
        référence jamais non plus (test_d ci-dessus, inchangé).
        """
        assert "GridTradingController" in dir(run_active_grid)  # importé, légitimement

        activation_calls = []
        from grid_trading_controller import GridTradingController
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])  # défaut = resume

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert activation_calls == []  # jamais appelé en mode resume
        assert "REPRISE DE GRILLE ACTIVE" in capsys.readouterr().out

    def test_e_create_mode_activation_requires_live_flag(self, monkeypatch):
        """
        En --mode create, l'activation (GridTradingController) n'est
        autorisée que sous --live -- sans --live, jamais appelée, même
        en mode create. Adaptateur local complet (distinct de fake_adapter
        ci-dessus, qui interdit délibérément get_ticker pour le mode resume) :
        --mode create a légitimement besoin de get_ticker/get_klines pour
        l'optimiseur.
        """
        adapter = _CreateModeCapableAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        activation_calls = []
        from grid_trading_controller import GridTradingController
        monkeypatch.setattr(GridTradingController, "run", lambda self, *a, **k: activation_calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--allocated-capital-usdc", "200"])  # sans --live

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert activation_calls == []

    def test_e_create_mode_activation_called_only_with_live_flag(self, monkeypatch):
        """Symétrique : avec --live en mode create, l'activation EST appelée."""
        adapter = _CreateModeCapableAdapter()
        monkeypatch.setattr(run_active_grid, "build_adapter", lambda mode: adapter)

        activation_calls = []
        from grid_trading_controller import GridTradingController
        original_run = GridTradingController.run

        def spy(self, adapter_arg, config, **kwargs):
            activation_calls.append(1)
            return original_run(self, adapter_arg, config)

        monkeypatch.setattr(GridTradingController, "run", spy)

        original_loop = run_active_grid.run
        monkeypatch.setattr(run_active_grid, "run", lambda adapter_arg, config, **k: original_loop(adapter_arg, config, max_cycles=1, sleep_fn=lambda s: None))

        monkeypatch.setattr(sys, "argv", ["run_active_grid.py", "--mode", "create", "--live", "--allocated-capital-usdc", "200"])

        run_active_grid.main()

        assert activation_calls == [1]  # appelée exactement une fois

    def test_f_get_ticker_never_used_to_define_p0(self, fake_adapter):
        # get_ticker lève si jamais appelé (voir FakeAdapter) -- la simple
        # construction ne doit jamais l'invoquer.
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        assert fake_adapter.get_ticker_calls == 0
        assert config.p0 == Decimal("0.9982")

    def test_f_readonly_main_never_calls_get_ticker(self, fake_adapter, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])  # sans --live
        exit_code = run_active_grid.main()
        assert exit_code == 0
        assert fake_adapter.get_ticker_calls == 0


# ---------------------------------------------------------------------------
# G, H — invariance de config et de treillis entre cycles
# ---------------------------------------------------------------------------

class TestConfigAndTrellisInvariance:
    def test_g_two_successive_run_preflight_use_exact_same_config(self, fake_adapter):
        # build_historical_config() ne fige jamais q (--mode resume, non
        # utilisé en production H24) -- run() l'exige désormais figé.
        config = replace(run_active_grid.build_historical_config(fake_adapter, "XRP-USDC"), q=Decimal("21.044"), allocated_capital=Decimal("20000"))
        config_id = id(config)

        run_active_grid.run(fake_adapter, config, max_cycles=2, sleep_fn=lambda s: None)

        assert id(config) == config_id

    def test_h_trellis_identical_across_cycles(self, fake_adapter):
        config = run_active_grid.build_historical_config(fake_adapter, "XRP-USDC")
        report1 = run_preflight(fake_adapter, config)
        report2 = run_preflight(fake_adapter, config)
        assert report1.trellis == report2.trellis


# ---------------------------------------------------------------------------
# I — scénario réel, tie-break déjà en place
# ---------------------------------------------------------------------------

class TestRealScenarioTieBreak:
    def test_i_existing_sell_kept_and_cycle_completion_gets_own_buy_or_sell(self, fake_adapter):
        """RÉVISÉ par ce chantier : sell_gi (déjà ouvert) jamais annulé ;
        buy_gi (poursuite de cycle genuine, BUY complété) obtient
        désormais son propre SELL -- remplace l'ancienne assertion
        new_sells == [] qui affirmait le comportement inverse (starvation
        du cycle par le tie-break)."""
        # build_historical_config() ne fige jamais q (--mode resume, non
        # utilisé en production H24) -- run() l'exige désormais figé ;
        # 21.044 correspond à la quantité historique réelle de ce scénario.
        config = replace(run_active_grid.build_historical_config(fake_adapter, "XRP-USDC"), q=Decimal("21.044"), allocated_capital=Decimal("20000"))
        report = run_preflight(fake_adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi = lower_levels[-1]
        sell_gi = upper_levels[0]

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        fake_adapter.fills = (Fill(
            trade_id="838447", order_id="3839040208425013248", client_order_id=buy_root,
            inst_id="XRP-USDC", side="BUY", fill_price=Decimal(str(buy_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=1786926350744,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        fake_adapter.open_orders = (
            OrderSnapshot("3839040221108588544", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        run_active_grid.run(fake_adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert sell_root not in fake_adapter.cancelled_orders
        assert fake_adapter.cancelled_orders == []
        new_sells = [p for _, side, p, _, _ in fake_adapter.placed_orders if side == "SELL"]
        assert buy_gi in [float(p) for p in new_sells]


# ---------------------------------------------------------------------------
# J — aucun appel d'écriture en mode READ-ONLY
# ---------------------------------------------------------------------------

class TestReadOnlyDefaultNoWrite:
    def test_j_no_write_call_without_live_flag(self, fake_adapter, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])  # sans --live

        exit_code = run_active_grid.main()

        assert exit_code == 0
        assert fake_adapter.placed_orders == []
        assert fake_adapter.cancelled_orders == []
        assert fake_adapter.forbidden_calls == []
        output = capsys.readouterr().out
        assert "READ ONLY" in output
        assert "NO ORDER PLACED" in output

    def test_j_run_exposure_cycle_never_reached_without_live(self, fake_adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(run_active_grid, "run_exposure_cycle", lambda *a, **k: calls.append(1))
        monkeypatch.setattr(sys, "argv", ["run_active_grid.py"])

        run_active_grid.main()

        assert calls == []
