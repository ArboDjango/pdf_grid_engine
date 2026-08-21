"""Tests du correctif : une fenêtre d'exposition partiellement matérialisée
(filtre de liquidité ou price-limit) ne doit jamais être déclarée ACTIVATED.

Aucun accès réseau réel, aucun ordre OKX pendant pytest.
"""

from decimal import Decimal

import pytest

from grid_order_orchestrator import plan_exposure
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from okx_spot_adapter import (
    AccountConfig, Balances, CancelResult, FeeRates, InstrumentRules,
    OkxApiError, OrderSnapshot, PlacementResult,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base_avail="20000", quote_avail="20000", base_total=None, quote_total=None, fee_type="1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.0001"), Decimal("0.001"), Decimal("0.1"), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(
            Decimal(base_avail), Decimal(quote_avail),
            Decimal(base_total if base_total is not None else base_avail),
            Decimal(quote_total if quote_total is not None else quote_avail),
        )
        self._fee_type = fee_type
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []

    def get_instrument(self, inst_id): return self.instrument
    def get_fee_rates(self, inst_id): return self.fees
    def get_balances(self, base_ccy, quote_ccy): return self.balances
    def get_account_config(self): return AccountConfig(fee_type=self._fee_type)

    def list_open_orders(self, inst_id, client_order_id=None):
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def list_fills(self, inst_id, *, order_id=None):
        return self.fills

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
        return CancelResult(True, "ORDX", client_order_id, None, None)


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
    )
    values.update(changes)
    return PreflightConfig(**values)


class TestPartiallyExposedOnLiquidityFilter:
    def test_buy_eliminated_by_liquidity_is_not_activated(self):
        """Cas exact du diagnostic INJ : SELL finançable, BUY non
        finançable -- résultat NE DOIT PLUS être ACTIVATED."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")  # BUY impossible, SELL ok
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert result.state == GridTradingState.PARTIALLY_EXPOSED
        assert result.state != GridTradingState.ACTIVATED

    def test_sell_still_placed_despite_partial_exposure(self):
        """La correction ne touche jamais le comportement des ordres :
        le SELL finançable reste réellement placé."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        sides = [p[1] for p in adapter.placed_orders]
        assert "SELL" in sides
        assert "BUY" not in sides  # toujours filtré, comportement du filtre inchangé

    def test_both_sides_financable_still_activated(self):
        """Non-régression : quand la fenêtre complète est honorée,
        ACTIVATED reste le résultat correct."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert result.state == GridTradingState.ACTIVATED

    def test_detail_mentions_missing_instruction(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert "BUY" in result.detail
        assert "fenêtre" in result.detail.lower()

    def test_activation_result_still_attached_on_partial_exposure(self):
        """activation_result reste accessible -- rien n'est masqué,
        seule l'étiquette finale change."""
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()

        result = GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert result.activation_result is not None


class TestNoChangeToFilterGridQP0:
    def test_q_unchanged(self):
        adapter_full = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter_full, config)

        # total identique (20000/20000) -> q identique ; seul available diffère.
        adapter_partial = FakeAdapter(base_avail="20000", quote_avail="0.01", base_total="20000", quote_total="20000")
        GridTradingController().run(adapter_partial, config, k_buy=1, k_sell=1)

        for _, _, _, qty, _ in adapter_partial.placed_orders:
            assert qty == report.q  # q strictement inchangé, filtré ou non

    def test_p0_and_geometry_untouched(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()
        original_p0, original_gul, original_gll = config.p0, config.gul, config.gll

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert config.p0 == original_p0
        assert config.gul == original_gul
        assert config.gll == original_gll

    def test_no_cancel_triggered_by_this_correction(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()

        GridTradingController().run(adapter, config, k_buy=1, k_sell=1)

        assert adapter.cancelled_orders == []


class TestExposurePlanTracksRequestedWindow:
    def test_requested_to_materialize_captures_pre_filter_selection(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="0.01")
        config = grid_config()
        report = run_preflight(adapter, config)

        plan = plan_exposure(adapter, report, k_buy=1, k_sell=1)

        # Demandé : 2 instructions (BUY+SELL) -- avant tout filtre.
        assert len(plan.requested_to_materialize) == 2
        # Matérialisé après filtre : 1 seule (SELL).
        assert len(plan.decision.to_materialize) == 1

    def test_requested_equals_materialized_when_fully_funded(self):
        adapter = FakeAdapter(base_avail="20000", quote_avail="20000")
        config = grid_config()
        report = run_preflight(adapter, config)

        plan = plan_exposure(adapter, report, k_buy=1, k_sell=1)

        assert plan.requested_to_materialize == plan.decision.to_materialize
