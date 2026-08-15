"""Tests pour grid_order_orchestrator.run_exposure_cycle.

Couvre le cycle complet : reconstruction -> sélection -> annulation ->
délégation à GridActivationController (inchangé). Adaptateur simulé,
aucun appel réseau réel.
"""

from decimal import Decimal

from cell_state_reconstruction import reconstruct_cells
from grid_activation_controller import GridActivationState, derive_grid_order_id
from grid_order_orchestrator import run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import Balances, CancelResult, FeeRates, InstrumentRules, OkxApiError, OrderSnapshot, PlacementResult
from trellis_calculator import GridGeometry


class FakeAdapter:
    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 1, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.get_order_results = {}
        self.fills = ()
        self.placed_orders = []
        self.placement_results = {}
        self.cancelled_orders = []
        self.cancel_results = {}
        self.call_order = []  # trace globale de l'ordre d'appel des méthodes

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def list_open_orders(self, inst_id, client_order_id=None):
        self.call_order.append("list_open_orders")
        if client_order_id is None:
            return self.open_orders
        return tuple(o for o in self.open_orders if o.client_order_id == client_order_id)

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.call_order.append("get_order")
        if client_order_id in self.get_order_results:
            result = self.get_order_results[client_order_id]
            if isinstance(result, Exception):
                raise result
            return result
        raise OkxApiError("Réponse OKX : un élément attendu, 0 reçu")

    def list_fills(self, inst_id, *, order_id=None):
        self.call_order.append("list_fills")
        if order_id is None:
            return self.fills
        return tuple(fill for fill in self.fills if fill.order_id == order_id)

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        self.call_order.append("place_post_only_limit")
        self.placed_orders.append((inst_id, side, price, quantity, client_order_id))
        if client_order_id in self.placement_results:
            return self.placement_results[client_order_id]
        return PlacementResult(True, "ORDX", client_order_id, None, None)

    def cancel_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.call_order.append("cancel_order")
        self.cancelled_orders.append((inst_id, client_order_id))
        if client_order_id in self.cancel_results:
            return self.cancel_results[client_order_id]
        return CancelResult(True, "ORDX", client_order_id, None, None)


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("115"), gll=Decimal("85"), nu=3, nl=3,
        p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def filled_snapshot(client_order_id, side="BUY", price="90.0", updated_at=100):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("1.4"), "filled", "post_only", None, updated_at)


def open_order_snapshot(client_order_id, side="BUY", price="90.0"):
    return OrderSnapshot("ORD1", client_order_id, "XRP-USDC", side, Decimal(price), Decimal("1.4"), Decimal("0"), "live", "post_only", None, None)


class TestBuyFillRepostsAtSameGi:
    def test_buy_fill_reposts_at_same_gi(self):
        """
        Fidélité PDF centrale : un BUY complètement rempli à gi doit
        produire un SELL EXACTEMENT à ce même gi — jamais un voisin.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        touched_gi = near_cash_prices[-1]  # niveau le plus proche de P0, côté BUY

        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", touched_gi)
        # Le BUY a été rempli : plus dans open_orders, mais trouvable dans
        # l'historique comme filled.
        adapter.get_order_results[buy_id] = filled_snapshot(buy_id, side="BUY", price=str(touched_gi))
        adapter.open_orders = ()

        result = run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        sell_prices_placed = [price for _, side, price, _, _ in [
            (adapter.instrument.inst_id, s, p, q, cid) for (_, s, p, q, cid) in adapter.placed_orders
        ] if side == "SELL"]
        assert touched_gi in sell_prices_placed  # repost EXACTEMENT au même gi

    def test_reposted_sell_uses_exact_same_gi_not_a_neighbor(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        touched_gi = near_cash_prices[-1]
        neighbor_up = sorted(float(g) for g in report.trellis)[sorted(float(g) for g in report.trellis).index(touched_gi) + 1]

        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", touched_gi)
        adapter.get_order_results[buy_id] = filled_snapshot(buy_id, side="BUY", price=str(touched_gi))

        result = run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        sell_prices = [p for _, s, p, _, _ in adapter.placed_orders if s == "SELL"]
        assert neighbor_up not in sell_prices  # jamais le voisin (ancien mécanisme after_buy_completed rejeté)


class TestSellFillRepostsAtSameGi:
    def test_sell_fill_reposts_at_same_gi(self):
        """
        k_buy=2 : évite l'ambiguïté d'égalité de distance avec la cellule
        BUY naturelle immédiatement adjacente côté opposé (95.0, elle
        aussi à distance 1 de P0=100 — voir
        TestTieBreakOnEqualDistance ci-dessous pour cette égalité
        documentée explicitement). Avec k_buy=2, les deux candidates à
        distance 1 sont retenues, sans dépendre de l'ordre de départage.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        near_spot_prices = sorted((float(g) for g in report.trellis if float(g) > float(report.config.p0)))
        touched_gi = near_spot_prices[0]  # niveau le plus proche de P0, côté SELL

        sell_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "SELL", touched_gi)
        adapter.get_order_results[sell_id] = filled_snapshot(sell_id, side="SELL", price=str(touched_gi))

        result = run_exposure_cycle(adapter, report, k_buy=2, k_sell=1)

        buy_prices_placed = [p for _, s, p, _, _ in adapter.placed_orders if s == "BUY"]
        assert touched_gi in buy_prices_placed


class TestTieBreakOnEqualDistance:
    """
    Découverte lors de l'implémentation : une cellule qui bascule en
    CASH_HELD après un SELL rempli (côté géométriquement au-dessus de P0)
    peut se trouver à EXACTEMENT la même distance d'index que la cellule
    CASH_HELD naturelle immédiatement adjacente côté BUY. La politique
    "k plus proches de P0" ne compare jamais le côté géométrique
    d'origine après un basculement — seule la distance d'index compte,
    conformément à la fidélité PDF (l'état courant d'une cellule dépend
    de son historique, jamais de sa position initiale par rapport à P0).

    Le départage, en cas d'égalité stricte, est déterministe : tri stable
    par distance croissante, préservant l'ordre d'origine du treillis
    (prix croissant) — la cellule de PRIX LE PLUS BAS parmi les ex-aequo
    l'emporte. Ceci est un détail d'implémentation, documenté ici
    explicitement, jamais une règle économique nouvelle.
    """

    def test_equal_distance_tie_is_broken_by_ascending_trellis_order(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        # gi=95.0 (CASH_HELD naturel, index 2) et gi=105.0 (bascule après
        # SELL rempli, index 4) sont toutes deux à distance 1 de P0 (index 3).
        sell_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "SELL", 105.0)
        adapter.get_order_results[sell_id] = filled_snapshot(sell_id, side="SELL", price="105.0")

        result = run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        buy_prices_placed = [p for _, s, p, _, _ in adapter.placed_orders if s == "BUY"]
        assert buy_prices_placed == [Decimal("95.0")]  # le prix le plus bas des deux ex-aequo


class TestCancelPrecedesMaterialization:
    def test_cancel_precedes_materialization_in_call_order(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        # Un ordre trop éloigné, à annuler, plus un niveau proche à matérialiser.
        far_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        far_gi = far_cash_prices[0]
        far_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", far_gi)
        adapter.open_orders = (open_order_snapshot(far_id, side="BUY", price=str(far_gi)),)

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        assert "cancel_order" in adapter.call_order
        assert "place_post_only_limit" in adapter.call_order
        cancel_index = adapter.call_order.index("cancel_order")
        place_index = adapter.call_order.index("place_post_only_limit")
        assert cancel_index < place_index

    def test_cancelled_order_is_the_designated_one_not_the_fresh_repost(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        far_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        far_gi = far_cash_prices[0]
        far_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", far_gi)
        adapter.open_orders = (open_order_snapshot(far_id, side="BUY", price=str(far_gi)),)

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        cancelled_ids = [cid for _, cid in adapter.cancelled_orders]
        assert far_id in cancelled_ids
        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        near_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", near_cash_prices[-1])
        assert near_id not in cancelled_ids


class TestGridActivationControllerReceivesNarrowedReportUnmodified:
    def test_activation_result_reflects_only_the_narrowed_window(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        result = run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        # Au plus k_buy + k_sell placements — jamais la grille complète (nu+nl=6).
        assert len(adapter.placed_orders) <= 2
        assert result.activation_result.state == GridActivationState.ACTIVE


class TestPartialFillDoesNotRearm:
    def test_partial_fill_does_not_rearm(self):
        """
        Un ordre partiellement rempli (state=partially_filled) ne doit
        déclencher aucun repost — la cellule reste dans son état courant
        tant que l'exécution n'est pas complète (RN-101 §3, déjà acté).
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        near_cash_prices = sorted((float(g) for g in report.trellis if float(g) < float(report.config.p0)))
        touched_gi = near_cash_prices[-1]
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", touched_gi)

        # Toujours dans open_orders, mais partiellement rempli : ni filled,
        # ni absent -- la reconstruction doit le lire comme CASH_HELD
        # (l'ordre BUY est toujours "ouvert" au sens list_open_orders).
        adapter.open_orders = (OrderSnapshot("ORD1", buy_id, "XRP-USDC", "BUY", Decimal(str(touched_gi)), Decimal("1.4"), Decimal("0.5"), "partially_filled", "post_only", None, None),)

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        # Aucun SELL placé à ce niveau : le fill n'est pas complet.
        sell_prices = [p for _, s, p, _, _ in adapter.placed_orders if s == "SELL"]
        assert touched_gi not in sell_prices


class TestNoMarketOrder:
    def test_no_market_order_method_ever_called(self):
        """
        Vérification structurelle : aucune méthode d'achat/vente MARKET
        n'est appelée par l'orchestrateur — seul place_post_only_limit
        (et cancel_order) figurent dans le FakeAdapter et sont exercés.
        """
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        assert "place_market_buy" not in adapter.call_order
        # Le FakeAdapter n'implémente même pas place_market_buy : tout appel
        # aurait levé AttributeError, jamais atteint silencieusement.

    def test_orchestrator_module_never_references_market(self):
        import inspect
        import grid_order_orchestrator
        source = inspect.getsource(grid_order_orchestrator)
        assert "place_market_buy" not in source
        assert "MARKET" not in source


class TestNoPriceRecalculation:
    def test_no_price_recalculation_trellis_unchanged_after_cycle(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        original_trellis = report.trellis

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        assert report.trellis == original_trellis  # report est frozen, jamais muté

    def test_no_price_recalculation_placed_prices_are_trellis_values(self):
        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())
        trellis_prices = {float(g) for g in report.trellis}

        run_exposure_cycle(adapter, report, k_buy=1, k_sell=1)

        for _, _, price, _, _ in adapter.placed_orders:
            assert price in trellis_prices  # jamais un prix recalculé hors treillis


class TestRepeatedCellCycle:

    def test_buy_sell_buy_same_gi_creates_new_buy_order(self):
        """
        Le même gi peut effectuer plusieurs cycles économiques :

            BUY gi -> SELL gi -> BUY gi

        Le second BUY est une nouvelle occurrence d'ordre.
        Il ne doit pas être bloqué par le premier BUY déjà FILLED.
        """

        adapter = FakeAdapter(base="1000", quote="1000")
        report = run_preflight(adapter, grid_config())

        # Niveau BUY le plus proche de P0.
        buy_prices = sorted(
            float(g)
            for g in report.trellis
            if float(g) < float(report.config.p0)
        )
        gi = buy_prices[-1]

        buy_id = derive_grid_order_id(
            report.config,
            report.instrument.tick_size,
            report.instrument.lot_size,
            "BUY",
            gi,
        )

        sell_id = derive_grid_order_id(
            report.config,
            report.instrument.tick_size,
            report.instrument.lot_size,
            "SELL",
            gi,
        )

        # ------------------------------------------------------------
        # Cycle 1 : BUY gi déjà rempli
        # → la cellule doit devenir SPOT_HELD
        # → le moteur doit placer SELL gi
        # ------------------------------------------------------------

        adapter.get_order_results[buy_id] = filled_snapshot(
            buy_id,
            side="BUY",
            price=str(gi),
            updated_at=100,
        )

        result_1 = run_exposure_cycle(
            adapter,
            report,
            k_buy=1,
            k_sell=1,
        )

        sell_placements = [
            order
            for order in adapter.placed_orders
            if order[1] == "SELL" and order[2] == gi
        ]

        assert len(sell_placements) == 1

        # ------------------------------------------------------------
        # Cycle 2 : le SELL gi vient lui aussi d'être rempli
        # → la cellule doit redevenir CASH_HELD
        # → le moteur doit replacer BUY gi
        # ------------------------------------------------------------

        adapter.get_order_results[sell_id] = filled_snapshot(
            sell_id,
            side="SELL",
            price=str(gi),
            updated_at=200,
        )

        result_2 = run_exposure_cycle(
            adapter,
            report,
            k_buy=1,
            k_sell=1,
        )

        buy_placements = [
            order
            for order in adapter.placed_orders
            if order[1] == "BUY" and order[2] == gi
        ]

        print("\n--- DEBUG CYCLE 2 ---")
        print("gi =", gi)
        print("buy_id =", buy_id)
        print("sell_id =", sell_id)
        print("placed_orders =", adapter.placed_orders)
        print("get_order_results =", adapter.get_order_results)
        print("result_2 =", result_2)

        # Il doit maintenant exister un SECOND BUY à exactement gi.
        assert len(buy_placements) == 1
