"""Tests pour grid_replacement.py -- remplacement automatique de la grille
active (chantier feature/grid-replacement), validé par 4 rapports de
conception successifs avant implémentation.

Couvre directement les 13 scénarios obligatoires du chantier.
"""

import json
import os
from dataclasses import replace
from decimal import Decimal

import pytest

import grid_replacement
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingState
from okx_spot_adapter import (
    Balances, Candle, CancelResult, FeeRates, InstrumentRules, OkxApiError,
    OrderSnapshot, PlacementResult, Ticker,
)
from trellis_calculator import GridGeometry


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
    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001",
                 tick="0.0001", lot="0.001", minimum="0.1", ticker_price="1.0013",
                 candle_closes=None):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.ticker_price = Decimal(ticker_price)
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.get_ticker_calls = 0
        closes = candle_closes if candle_closes is not None else [1.000 + i * 0.0008 for i in range(50)]
        self._candles = make_candles(closes)

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
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", self.ticker_price, quantity, quantity, "filled", "market", 1, 1)
        self.placed_orders.append((inst_id, "MARKET_BUY", quantity, client_order_id))
        # Remplissage immédiat simulé (fake d'intégration, pas de carnet
        # réel) : le prochain get_order() sur cet identifiant doit refléter
        # l'exécution complète, et les soldes doivent réellement bouger --
        # sinon InitializationController relirait spot_covered=False
        # indéfiniment (achat "FILLED" mais patrimoine inchangé).
        self.get_order_results[client_order_id] = snapshot
        cost = quantity * self.ticker_price
        self.balances = Balances(
            self.balances.base_available + quantity, self.balances.quote_available - cost,
            self.balances.base_total + quantity, self.balances.quote_total - cost,
        )
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
        q=Decimal("21.044"), allocated_capital=Decimal("200"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def open_order(gi, side="BUY", coid="GRID-EXISTING"):
    return OrderSnapshot("ORD1", coid, "XRP-USDC", side, Decimal(str(gi)), Decimal("10"), Decimal("0"), "live", "post_only", None, None)


# =============================================================================
# 3 : moins de 3 observations -> aucun remplacement
# =============================================================================

class TestBelowThreshold:
    def test_two_consecutive_breaches_do_not_trigger(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")  # au-dessus de GUL=1.0563715

        for _ in range(2):
            result_config, materialize = grid_replacement.run_replacement_check(adapter, config, 1, 1, path)
            assert result_config == config
            assert materialize is True

        state = grid_replacement.load_replacement_state(path)
        assert state.lifecycle_status == "ACTIVE"
        assert state.above_streak == 2
        assert adapter.cancelled_orders == []

    def test_return_to_zone_resets_the_streak(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")

        grid_replacement.run_replacement_check(adapter, config, 1, 1, path)
        grid_replacement.run_replacement_check(adapter, config, 1, 1, path)
        adapter.ticker_price = Decimal("1.00")  # retour dans la zone
        grid_replacement.run_replacement_check(adapter, config, 1, 1, path)

        state = grid_replacement.load_replacement_state(path)
        assert state.lifecycle_status == "ACTIVE"
        assert state.above_streak == 0


# =============================================================================
# 1, 4, 5, 6, 7, 8, 9, 12 : dépassement haussier confirmé, bout en bout
# =============================================================================

MAX_ATTEMPTS = 8  # marge large : REPLACEMENT_N (déclenchement) + retries d'activation


def run_until_materialized(adapter, config, path, k_buy=1, k_sell=1, max_attempts=MAX_ATTEMPTS):
    """Boucle jusqu'à ce que le remplacement aboutisse (ou épuisement du
    budget d'essais). GridTradingController peut avoir besoin de plusieurs
    appels pour terminer l'achat patrimonial initial (BUYING -> SPOT_READY)
    -- ce n'est pas garanti en un seul appel de run_replacement_check."""
    result_config = config
    materialize = True
    for _ in range(max_attempts):
        result_config, materialize = grid_replacement.run_replacement_check(adapter, result_config, k_buy, k_sell, path)
        if materialize and result_config != config:
            return result_config, True
    return result_config, materialize


class TestUpwardReplacement:
    def _trigger(self, tmp_path, ticker_price="1.20"):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price=ticker_price)
        new_config, materialize = run_until_materialized(adapter, config, path)
        return adapter, config, new_config, materialize, path

    def test_1_upward_breach_confirmed_triggers_replacement(self, tmp_path):
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path)
        assert materialize is True
        assert new_config != old_config
        state = grid_replacement.load_replacement_state(path)
        assert state.lifecycle_status == "ACTIVE"  # remplacement abouti -> retour à ACTIVE

    def test_4_p0_new_exactly_equals_old_boundary(self, tmp_path):
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path)
        instrument = adapter.get_instrument("XRP-USDC")
        expected_p0 = grid_replacement._round_down(old_config.gul, instrument.tick_size)
        assert new_config.p0 == expected_p0

    def test_5_p0_new_differs_from_confirmation_price(self, tmp_path):
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path, ticker_price="1.60")
        assert new_config.p0 != Decimal("1.60")
        assert new_config.p0 == grid_replacement._round_down(old_config.gul, adapter.get_instrument("XRP-USDC").tick_size)

    def test_6_widening_applies_only_to_breached_side(self, tmp_path):
        """Côté franchi (GUL) : largeur = ATR + déplacement. Côté non
        franchi (GLL) : largeur ATR seule -- vérifié en comparant à une
        grille de référence construite SANS aucun remplacement (même
        marché, même ATR), qui isole la largeur ATR pure."""
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path, ticker_price="1.20")

        # Largeur GLL réellement obtenue (côté non franchi).
        gll_pct_obtained = (float(new_config.p0) - float(new_config.gll)) / float(new_config.p0)

        # Largeur ATR pure attendue de ce côté, recalculée indépendamment
        # avec les mêmes conditions de marché.
        import market_reader
        conditions = market_reader.read(adapter, "XRP-USDC")
        sim = grid_replacement._atr_width_pct(conditions.atr_norm_15m, conditions.di_ratio_15m)

        assert gll_pct_obtained == pytest.approx(sim["gll_pct"], rel=1e-6)

        # Côté franchi (GUL) : strictement plus large que l'ATR seul,
        # preuve que le déplacement a bien été ajouté.
        gul_pct_obtained = (float(new_config.gul) - float(new_config.p0)) / float(new_config.p0)
        assert gul_pct_obtained > sim["gul_pct"]

    def test_7_q_is_recomputed_not_copied(self, tmp_path):
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path)
        assert new_config.q is not None
        assert new_config.q != old_config.q
        # Vérifie que q correspond bien à un calcul frais via run_preflight
        # sur la config de la nouvelle grille (jamais une copie).
        report = run_preflight(adapter, replace(new_config, q=None))
        assert new_config.q == report.q

    def test_8_old_generation_orders_fully_cancelled(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")
        adapter.open_orders = (
            open_order(1.0013, "BUY", "GRID-A"),
            open_order(1.0451, "SELL", "GRID-B"),
        )
        for _ in range(grid_replacement.REPLACEMENT_N):
            grid_replacement.run_replacement_check(adapter, config, 1, 1, path)

        assert "GRID-A" in adapter.cancelled_orders
        assert "GRID-B" in adapter.cancelled_orders
        assert not any(o.client_order_id.startswith("GRID-A") or o.client_order_id.startswith("GRID-B") for o in adapter.open_orders)

    def test_9_new_generation_correctly_persisted(self, tmp_path):
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path)
        assert os.path.exists(path)
        with open(path) as f:
            payload = json.load(f)
        assert Decimal(payload["p0"]) == new_config.p0
        assert Decimal(payload["gul"]) == new_config.gul
        assert Decimal(payload["gll"]) == new_config.gll
        assert Decimal(payload["q"]) == new_config.q
        assert Decimal(payload["allocated_capital"]) == new_config.allocated_capital

    def test_12_new_width_avoids_immediate_re_breach(self, tmp_path):
        """La largeur du côté franchi doit couvrir une marge réelle
        au-delà du prix de confirmation -- pas seulement l'ATR seul (qui
        laisserait la nouvelle grille déjà dépassée, cf. rapport de
        conception validé)."""
        adapter, old_config, new_config, materialize, path = self._trigger(tmp_path, ticker_price="1.20")
        assert new_config.gul > Decimal("1.20")  # dépasse le prix de confirmation, pas seulement l'atteint


# =============================================================================
# 2 : dépassement baissier confirmé (symétrique)
# =============================================================================

class TestDownwardReplacement:
    def test_2_downward_breach_confirmed_triggers_replacement(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="0.80", candle_closes=[1.000 - i * 0.0008 for i in range(50)])

        new_config, materialize = run_until_materialized(adapter, config, path)

        assert materialize is True
        assert new_config != config
        instrument = adapter.get_instrument("XRP-USDC")
        expected_p0 = grid_replacement._round_down(config.gll, instrument.tick_size)
        assert new_config.p0 == expected_p0
        assert new_config.p0 != Decimal("0.80")
        assert new_config.gll < Decimal("0.80")  # dépasse le prix de confirmation vers le bas

    def test_6_widening_applies_only_to_breached_side_downward(self, tmp_path):
        """Symétrique de test_6 (TestUpwardReplacement) pour un dépassement
        BAS : côté franchi (GLL) : largeur = ATR + déplacement. Côté non
        franchi (GUL) : largeur ATR seule -- vérifié en comparant à une
        grille de référence construite SANS aucun remplacement (même
        marché, même ATR), qui isole la largeur ATR pure."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="0.80", candle_closes=[1.000 - i * 0.0008 for i in range(50)])

        new_config, materialize = run_until_materialized(adapter, config, path)
        assert materialize is True

        # Largeur GUL réellement obtenue (côté non franchi).
        gul_pct_obtained = (float(new_config.gul) - float(new_config.p0)) / float(new_config.p0)

        # Largeur ATR pure attendue de ce côté, recalculée indépendamment
        # avec les mêmes conditions de marché.
        import market_reader
        conditions = market_reader.read(adapter, "XRP-USDC")
        sim = grid_replacement._atr_width_pct(conditions.atr_norm_15m, conditions.di_ratio_15m)

        assert gul_pct_obtained == pytest.approx(sim["gul_pct"], rel=1e-6)

        # Côté franchi (GLL) : strictement plus étroit (plus bas) que
        # l'ATR seul ne le placerait, preuve que le déplacement a bien
        # été ajouté à la largeur de ce côté.
        gll_pct_obtained = (float(new_config.p0) - float(new_config.gll)) / float(new_config.p0)
        assert gll_pct_obtained > sim["gll_pct"]


# =============================================================================
# 11 : nouveau dépassement réel permettant G1 -> G2
# =============================================================================

class TestChainedReplacement:
    def test_11_second_real_breach_of_g1_triggers_g2(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")

        g1, materialize = run_until_materialized(adapter, config, path)
        assert materialize is True
        assert g1 != config

        # Le marché dépasse ensuite réellement GUL_G1 elle-même.
        adapter.ticker_price = g1.gul + Decimal("0.5")
        g2, materialize = run_until_materialized(adapter, g1, path)

        assert materialize is True
        assert g2 != g1
        assert g2 != config
        instrument = adapter.get_instrument("XRP-USDC")
        assert g2.p0 == grid_replacement._round_down(g1.gul, instrument.tick_size)


# =============================================================================
# 10 : reprise correcte après redémarrage
# =============================================================================

class TestRestartRecovery:
    def test_10_resumes_from_cancelling_without_redeciding(self, tmp_path):
        """Simule un redémarrage juste après la transition vers
        CANCELLING (avant toute annulation effective) -- l'appel suivant
        doit reprendre l'annulation, jamais redécider depuis zéro ni
        laisser les deux 'générations' coexister."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")
        adapter.open_orders = (open_order(1.0013, "BUY", "GRID-OLD"),)

        pre_restart_state = grid_replacement.ReplacementState("CANCELLING", "UP", 3, 0, None)
        grid_replacement.save_replacement_state(pre_restart_state, path)

        new_config, materialize = run_until_materialized(adapter, config, path)

        assert "GRID-OLD" in adapter.cancelled_orders
        assert materialize is True
        assert new_config != config

    def test_10_resumes_from_activating_with_frozen_config_never_recomputed(self, tmp_path):
        """Simule un redémarrage pendant ACTIVATING -- la configuration
        gelée doit être réutilisée TELLE QUELLE, jamais recalculée (sinon
        elle dériverait d'une tentative à l'autre)."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.60")  # prix TRÈS différent de celui utilisé pour geler pending_config

        frozen_draft = replace(
            grid_config(gul=Decimal("1.20"), gll=Decimal("0.99"), p0=Decimal("1.0563"), q=None, allocated_capital=Decimal("300")),
        )
        report = run_preflight(adapter, frozen_draft)
        frozen = replace(frozen_draft, q=report.q)

        pending_state = grid_replacement.ReplacementState("ACTIVATING", "UP", 0, 0, frozen)
        grid_replacement.save_replacement_state(pending_state, path)

        new_config, materialize = run_until_materialized(adapter, config, path)

        assert materialize is True
        assert new_config.p0 == Decimal("1.0563")  # config gelée réutilisée, PAS recalculée depuis le prix 1.60
        assert new_config.q == frozen.q

    def test_10_no_two_generations_active_simultaneously_across_restart(self, tmp_path):
        """Après reprise complète, un seul jeu d'ordres 'GRID' doit
        correspondre à la nouvelle génération -- jamais de coexistence."""
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")
        adapter.open_orders = (open_order(1.0013, "BUY", "GRID-OLD"),)

        pre_restart_state = grid_replacement.ReplacementState("CANCELLING", "UP", 3, 0, None)
        grid_replacement.save_replacement_state(pre_restart_state, path)

        grid_replacement.run_replacement_check(adapter, config, 1, 1, path)

        assert "GRID-OLD" not in [o.client_order_id for o in adapter.open_orders]


# =============================================================================
# 13 : interaction avec la protection anti-churn existante
# =============================================================================

class TestChurnProtectionInteraction:
    def test_13_churn_protection_reset_after_replacement(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        adapter = FakeAdapter(ticker_price="1.20")

        import churn_protection
        stale_state = {1.0013: churn_protection.ChurnProtectionEntry(3, True, 999, 999)}
        churn_protection.save_churn_protection(stale_state, path)
        assert churn_protection.load_churn_protection(path)  # bien présent avant remplacement

        new_config, materialize = run_until_materialized(adapter, config, path)

        assert materialize is True
        assert churn_protection.load_churn_protection(path) == {}  # repart à zéro (règle 11)
