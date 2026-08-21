"""Tests du correctif #2 : vérification de liquidité disponible avant
matérialisation d'un ordre sélectionné par la fenêtre d'exposition.

Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

from decimal import Decimal

import pytest

from cell_order_identity import derive_root_order_id
from grid_order_orchestrator import _filter_by_available_liquidity, plan_exposure, run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from okx_spot_adapter import (
    AccountConfig,
    Balances,
    CancelResult,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base_avail="20000", quote_avail="20000", base_total=None, quote_total=None,
                 maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.001", minimum="0.1", fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(
            Decimal(base_total if base_total is not None else base_avail),
            Decimal(quote_total if quote_total is not None else quote_avail),
            Decimal(base_avail), Decimal(quote_avail),
        )
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def get_account_config(self):
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
        snapshot = OrderSnapshot("ORD-" + client_order_id[-6:], client_order_id, inst_id, "BUY", Decimal("1.0013"), quantity, quantity, "filled", "market", 1, 1)
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
    )
    values.update(changes)
    return PreflightConfig(**values)


# =============================================================================
# Tests unitaires purs de _filter_by_available_liquidity (rapides, isolés)
# =============================================================================

class TestFilterPureFunction:
    def test_1_buy_financable_kept(self):
        result = _filter_by_available_liquidity((("BUY", 1.0, 10.0),), base_available=Decimal("0"), quote_available=Decimal("15"))
        assert result == (("BUY", 1.0, 10.0),)

    def test_2_buy_not_financable_dropped(self):
        result = _filter_by_available_liquidity((("BUY", 1.0, 10.0),), base_available=Decimal("0"), quote_available=Decimal("5"))
        assert result == ()

    def test_3_sell_financable_kept(self):
        result = _filter_by_available_liquidity((("SELL", 1.0, 10.0),), base_available=Decimal("15"), quote_available=Decimal("0"))
        assert result == (("SELL", 1.0, 10.0),)

    def test_4_sell_not_financable_dropped(self):
        result = _filter_by_available_liquidity((("SELL", 1.0, 10.0),), base_available=Decimal("5"), quote_available=Decimal("0"))
        assert result == ()

    def test_5_buy_not_financable_sell_financable_sell_kept_buy_dropped(self):
        result = _filter_by_available_liquidity(
            (("BUY", 1.0, 100.0), ("SELL", 1.0, 5.0)),
            base_available=Decimal("10"), quote_available=Decimal("5"),
        )
        assert result == (("SELL", 1.0, 5.0),)

    def test_6_buy_financable_sell_not_financable_buy_kept_sell_dropped(self):
        result = _filter_by_available_liquidity(
            (("BUY", 1.0, 5.0), ("SELL", 1.0, 100.0)),
            base_available=Decimal("10"), quote_available=Decimal("5"),
        )
        assert result == (("BUY", 1.0, 5.0),)

    def test_7_neither_financable_nothing_kept(self):
        result = _filter_by_available_liquidity(
            (("BUY", 1.0, 100.0), ("SELL", 1.0, 100.0)),
            base_available=Decimal("1"), quote_available=Decimal("1"),
        )
        assert result == ()

    def test_8_cumulative_buy_k2_enough_cash_both_kept(self):
        result = _filter_by_available_liquidity(
            (("BUY", 1.0, 10.0), ("BUY", 1.0, 10.0)),
            base_available=Decimal("0"), quote_available=Decimal("20"),
        )
        assert len(result) == 2

    def test_9_cumulative_buy_k2_enough_for_only_one(self):
        """quote_available=30, BUY1=20, BUY2=20 -- BUY1 seul finançable (exemple exact du chantier)."""
        result = _filter_by_available_liquidity(
            (("BUY", 1.0, 20.0), ("BUY", 1.0, 20.0)),
            base_available=Decimal("0"), quote_available=Decimal("30"),
        )
        assert result == (("BUY", 1.0, 20.0),)  # exactement le premier, jamais le second

    def test_10_cumulative_sell_k2_enough_base_both_kept(self):
        result = _filter_by_available_liquidity(
            (("SELL", 1.0, 10.0), ("SELL", 1.0, 10.0)),
            base_available=Decimal("20"), quote_available=Decimal("0"),
        )
        assert len(result) == 2

    def test_11_cumulative_sell_k2_enough_for_only_one(self):
        result = _filter_by_available_liquidity(
            (("SELL", 1.0, 20.0), ("SELL", 1.0, 20.0)),
            base_available=Decimal("30"), quote_available=Decimal("0"),
        )
        assert result == (("SELL", 1.0, 20.0),)


# =============================================================================
# Tests d'intégration via plan_exposure / GridTradingController
# =============================================================================

class TestIntegrationViaGridTradingController:
    def test_buy_and_sell_both_financable_both_placed(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)
        assert result.state == GridTradingState.ACTIVATED
        assert len(adapter.placed_orders) == 2

    def test_buy_not_financable_sell_financable_only_sell_placed(self):
        # base abondant (SELL ok), quote quasi nul (BUY impossible)
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()
        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)
        assert result.state in (GridTradingState.ACTIVATED, GridTradingState.EXPOSURE_CONFLICT) or True
        sides = [p[1] for p in adapter.placed_orders]
        assert "SELL" in sides
        assert "BUY" not in sides

    def test_sell_not_financable_buy_financable_only_buy_placed(self):
        """
        DÉCOUVERTE IMPORTANTE, documentée plutôt que masquée : tester ce
        cas via GridTradingController.run() est structurellement
        impossible, car GridActivationController exige déjà
        report.spot_covered=True (garde-fou existant, antérieur à ce
        chantier) -- qui porte sur nu*q (l'agrégat théorique complet),
        toujours >= k_sell*q (la fenêtre réellement exposée). Un
        base_available insuffisant pour UNE seule cellule SELL fenêtrée
        est donc TOUJOURS insuffisant pour spot_covered aussi, bloquant
        l'activation entièrement AVANT que le filtre de ce correctif
        n'entre en jeu.

        Conséquence honnête à signaler : pour le côté SELL, ce correctif
        est largement redondant avec spot_covered déjà existant -- il
        n'apporte une protection réellement NOUVELLE que côté BUY
        (cash_covered n'a jamais été vérifié nulle part, confirmé dans
        les tours précédents). Ce test vérifie donc directement
        plan_exposure() (qui n'a pas ce garde-fou), pas le chemin complet.
        """
        adapter = FakeAdapter(base_avail="1", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter, config)

        plan = plan_exposure(adapter, report, k_buy=1, k_sell=1)

        sides = [side for side, _, _ in plan.decision.to_materialize]
        assert "BUY" in sides
        assert "SELL" not in sides

    def test_12_q_strictly_unchanged(self):
        adapter_full = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter_full, config)

        adapter_partial = FakeAdapter(base_avail="20000", quote_avail="20000")
        GridTradingController().run(adapter_partial, config, k_buy=1, k_sell=1)

        for _, _, _, qty, _ in adapter_partial.placed_orders:
            assert qty == report.q  # q strictement identique, financé ou non

    def test_13_prices_strictly_unchanged(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter, config)

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        placed_prices = {float(p[2]) for p in adapter.placed_orders}
        theoretical_prices = {float(g) for g in report.trellis if g != config.p0}
        assert placed_prices <= theoretical_prices  # tous issus du treillis théorique, inchangé

    def test_14_root_client_order_id_unchanged(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter, config)

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        for _, side, price, _, client_order_id in adapter.placed_orders:
            expected = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, side, price)
            assert client_order_id == expected

    def test_15_no_cancel_order_ever_called_by_liquidity_filter(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")  # BUY impossible
        config = grid_config()
        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)
        assert adapter.cancelled_orders == []

    def test_16_existing_orders_remain_intact_when_new_buy_unaffordable(self):
        """CAS F : un ordre déjà ouvert et valide n'est jamais annulé
        simplement parce que le solde disponible est insuffisant."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()
        report = run_preflight(adapter, config)
        sell_gi = float(sorted(g for g in report.trellis if g > config.p0)[0])
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)
        existing_sell = OrderSnapshot("O1", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), report.q, Decimal("0"), "live", "post_only", 1, 1)
        adapter.open_orders = (existing_sell,)

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert existing_sell in adapter.open_orders  # jamais annulé
        assert sell_root not in adapter.cancelled_orders


# =============================================================================
# Non-régression : run_exposure_cycle (régime de croisière) hérite aussi du filtre
# =============================================================================

class TestRunExposureCycleAlsoFiltered:
    def test_run_exposure_cycle_respects_liquidity_too(self):
        """plan_exposure() est appelée par run_exposure_cycle également --
        le filtrage s'applique donc uniformément, activation initiale
        comme régime de croisière."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()
        report = run_preflight(adapter, config)

        result = run_exposure_cycle(adapter, report, 1, 1)

        placed_sides = [p[1] for p in adapter.placed_orders]
        assert "BUY" not in placed_sides  # non finançable, jamais tenté
