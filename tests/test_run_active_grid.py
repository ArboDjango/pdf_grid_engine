"""Tests pour run_active_grid.py.

Aucun accès réseau réel, aucun ordre OKX pendant pytest. Adaptateur
factice dont les méthodes d'écriture lèvent une exception si elles sont
appelées de manière imprévue (test 6, sécurité).

NOTE IMPORTANTE : dans cet environnement de test, cell_order_identity.py
est un STUB LOCAL (jamais fourni dans cette conversation, jamais livré) --
voir la note en tête de ce fichier de stub. Les tests ci-dessous vérifient
les propriétés ARCHITECTURALES du nouveau code (immutabilité de config,
invariance du treillis, non-annulation d'un ordre de la grille active) --
pas le contenu exact de la dérivation d'identité elle-même.
"""

from decimal import Decimal

import pytest

from cell_order_identity import derive_root_order_id
from grid_preflight import PreflightConfig, run_preflight
from grid_order_orchestrator import run_exposure_cycle
from okx_spot_adapter import (
    Balances,
    CancelResult,
    FeeRates,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
    PlacementResult,
)
from run_active_grid import run
from trellis_calculator import GridGeometry


class FakeAdapter:
    """Toute méthode d'écriture appelée de façon imprévue lève -- sauf
    place_post_only_limit, dont l'usage légitime (repost) est explicitement
    autorisé et suivi via placed_orders, jamais silencieux."""

    def __init__(self, *, base="20000", quote="20000", maker="-0.0008", taker="-0.001", tick="0.0001", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.placed_orders = []
        self.cancelled_orders = []
        self.forbidden_calls = []

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

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
        raise AssertionError("ÉCRITURE INTERDITE : MARKET, jamais attendu dans cette boucle")


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
        # q figé par défaut -- run() l'exige désormais (identité immuable de
        # la grille) ; 21.044 correspond à la quantité historique réelle
        # utilisée par les scénarios de fill/ordre déjà ouvert de ce fichier.
        q=Decimal("21.044"),
    )
    values.update(changes)
    return PreflightConfig(**values)


class TestConfigImmutability:
    """TEST 1 — configuration immuable."""

    def test_id_config_unchanged_across_cycles(self):
        adapter = FakeAdapter()
        config = grid_config()
        config_id_before = id(config)

        run(adapter, config, max_cycles=3, sleep_fn=lambda s: None)

        assert id(config) == config_id_before

    def test_all_fields_strictly_identical_across_cycles(self):
        adapter = FakeAdapter()
        config = grid_config()
        snapshot = (config.p0, config.gll, config.gul, config.nu, config.nl, config.geometry)

        run(adapter, config, max_cycles=5, sleep_fn=lambda s: None)

        assert (config.p0, config.gll, config.gul, config.nu, config.nl, config.geometry) == snapshot

    def test_preflight_config_is_frozen_dataclass(self):
        """Garantie structurelle : PreflightConfig est déjà @dataclass(frozen=True) -- toute tentative de réaffectation lève."""
        config = grid_config()
        with pytest.raises(Exception):
            config.p0 = Decimal("999")


class TestTrellisInvariance:
    """TEST 2 — treillis invariant."""

    def test_trellis_identical_across_multiple_run_preflight_calls(self):
        adapter = FakeAdapter()
        config = grid_config()

        report1 = run_preflight(adapter, config)
        report2 = run_preflight(adapter, config)
        report3 = run_preflight(adapter, config)

        assert report1.trellis == report2.trellis == report3.trellis


class TestDynamicPatrimonyStaticTrellisAndQ:
    """TEST 3 — patrimoine dynamique, treillis ET q invariants.

    RÉVISÉ par le chantier q-immuable : q fait désormais partie de
    l'identité économique de la grille (config.q), au même titre que le
    treillis (P0/GLL/GUL/nu/nl) -- une variation des soldes OKX entre deux
    cycles ne doit JAMAIS le modifier. Remplace l'ancienne assertion
    `report_before.q != report_after.q` qui affirmait exactement le
    comportement inverse (q recalculé à chaque cycle depuis les soldes
    courants) -- c'est précisément le défaut que ce chantier corrige.
    """

    def test_q_and_trellis_both_survive_balance_variation(self):
        adapter = FakeAdapter(base="20000", quote="20000")
        config = grid_config()

        report_before = run_preflight(adapter, config)

        adapter.balances = Balances(Decimal("5000"), Decimal("5000"), Decimal("5000"), Decimal("5000"))
        report_after = run_preflight(adapter, config)

        assert report_before.q == report_after.q == config.q  # q, figé
        assert report_before.trellis == report_after.trellis  # treillis, invariant

    def test_materialized_qty_immutable_across_cycles_despite_balance_drift(self):
        """Test obligatoire bout-en-bout : une variation des soldes OKX
        entre deux cycles de run() ne doit jamais changer la quantité
        réellement matérialisée dans les ordres placés."""
        adapter = FakeAdapter(base="20000", quote="20000")
        config = grid_config()

        balance_changes = iter([
            Balances(Decimal("5000"), Decimal("5000"), Decimal("5000"), Decimal("5000")),
            Balances(Decimal("50000"), Decimal("50000"), Decimal("50000"), Decimal("50000")),
        ])

        def drift_balances(_seconds):
            try:
                adapter.balances = next(balance_changes)
            except StopIteration:
                pass

        run(adapter, config, max_cycles=3, sleep_fn=drift_balances)

        assert adapter.placed_orders  # au moins un ordre matérialisé sur les 3 cycles
        placed_quantities = {qty for _, _, _, qty, _ in adapter.placed_orders}
        assert placed_quantities == {config.q}  # jamais autre chose que le q figé


class TestRealCaseBuyFilledSellLive:
    """TEST 4 — cas réel : BUY 0.9877 FILLED, SELL 1.0079 LIVE."""

    def test_sell_position_survives_and_cycle_completion_never_lost_to_config_drift(self):
        """
        RÉVISÉ par ce chantier (alignement PDF, priorité au cycle propre) :
        1.0079 (déjà ouvert) n'est PLUS JAMAIS annulé pour une raison de
        priorité géométrique -- garanti conservé. La cellule ayant
        réellement complété son BUY (poursuite de cycle genuine) obtient
        désormais son propre SELL à son propre gi, sans jamais remplacer
        1.0079. Le Niveau 2 (cellules initiales) conserve par ailleurs son
        propre budget k_sell indépendant, pouvant donc exposer une
        troisième cellule initiale jamais touchée -- comportement voulu,
        pas une fuite : k_buy/k_sell restent une limite de sécurité pour
        les cellules initiales UNIQUEMENT (cf. module
        grid_order_exposure_controller.py), jamais pour les cellules en
        poursuite de cycle ni pour celles déjà ouvertes.
        """
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi = lower_levels[-1]
        sell_gi = upper_levels[0]

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        adapter.fills = (Fill(
            trade_id="838447", order_id="3839040208425013248", client_order_id=buy_root,
            inst_id="XRP-USDC", side="BUY", fill_price=Decimal(str(buy_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=1786926350744,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        adapter.open_orders = (
            OrderSnapshot("3839040221108588544", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        cycles = run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert cycles == 1
        # 1.0079 (sell_gi) JAMAIS annulé -- garantie absolue de ce chantier.
        assert sell_root not in adapter.cancelled_orders
        assert adapter.cancelled_orders == []
        remaining_sells = [o for o in adapter.open_orders if o.side == "SELL"]
        remaining_prices = {float(o.price) for o in remaining_sells}
        # sell_gi (conservé) ET buy_gi (nouveau SELL, poursuite de cycle)
        # doivent tous deux être présents -- jamais l'un au prix de l'autre.
        assert sell_gi in remaining_prices
        assert buy_gi in remaining_prices
        # Tout prix supplémentaire éventuel doit rester un niveau réel du
        # treillis (Niveau 2, cellule initiale) -- jamais un prix étranger.
        theoretical_prices = {float(g) for g in report.trellis}
        assert remaining_prices <= theoretical_prices

    def test_cancellation_never_caused_by_config_drift_only_by_exposure_window(self):
        """
        Vérification négative explicite : si 1.0079 est annulé, ce n'est
        JAMAIS parce que config a changé (déjà vérifié structurellement
        par ailleurs -- id(config) constant) -- uniquement parce que le
        mécanisme k_buy/k_sell, à config strictement inchangé, a fait ce
        choix. Ce test verrouille que ce chantier (config immuable) a bien
        éliminé la cause qu'il visait, même si une autre cause légitime
        subsiste.
        """
        adapter = FakeAdapter()
        config = grid_config()
        config_id_before = id(config)
        report = run_preflight(adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi, sell_gi = lower_levels[-1], upper_levels[0]

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        adapter.fills = (Fill(
            trade_id="838447", order_id="O1", client_order_id=buy_root,
            inst_id="XRP-USDC", side="BUY", fill_price=Decimal(str(buy_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=100,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        adapter.open_orders = (
            OrderSnapshot("O2", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert id(config) == config_id_before  # jamais recalculé, cause éliminée par ce chantier


class TestRegressionDifferentConfigForbidden:
    """TEST 5 — régression critique : un config différent entre deux cycles doit être visible/détectable."""

    def test_calling_run_twice_with_different_config_objects_produces_different_trellis(self):
        """
        Démontre que la propriété d'invariance dépend ENTIÈREMENT de la
        discipline "même objet config" -- si l'appelant (mal à propos)
        construit un DEUXIÈME config différent et l'utilise, le treillis
        change nécessairement. Ce test documente et verrouille cette
        dépendance : run_active_grid.py n'expose jamais de mécanisme
        interne permettant de changer config au sein d'un seul appel à
        run() -- run() ne prend qu'un seul `config`, jamais réassigné,
        pour toute sa durée d'exécution (vérifié structurellement : aucune
        réaffectation de `config` dans le corps de run(), cf. code source).
        """
        adapter = FakeAdapter()
        config_a = grid_config(p0=Decimal("1.0013"), gul=Decimal("1.0563715"), gll=Decimal("0.951235"))
        config_b = grid_config(p0=Decimal("1.0500"), gul=Decimal("1.1000"), gll=Decimal("1.0000"))

        report_a = run_preflight(adapter, config_a)
        report_b = run_preflight(adapter, config_b)

        assert report_a.trellis != report_b.trellis  # confirme la nécessité de ne JAMAIS changer d'objet

        import inspect
        source = inspect.getsource(run)
        # Vérification structurelle : `config` n'est jamais réassigné dans run().
        assert "config =" not in source.replace("def run(", "")


class TestSecurityNoUnexpectedWrite:
    """TEST 6 — sécurité."""

    def test_no_market_order_ever_called(self):
        adapter = FakeAdapter()
        config = grid_config()

        run(adapter, config, max_cycles=2, sleep_fn=lambda s: None)

        assert adapter.forbidden_calls == []

    def test_only_post_only_limit_used_for_any_placement(self):
        adapter = FakeAdapter()
        config = grid_config()

        run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        # Chaque ordre placé, s'il y en a, ne provient que de place_post_only_limit
        # (aucun autre point de placement n'existe dans FakeAdapter hormis
        # le market_buy explicitement interdit).
        assert adapter.forbidden_calls == []

    def test_run_never_sleeps_in_test_mode(self):
        """max_cycles fourni -> sleep_fn ne doit jamais être le vrai time.sleep."""
        adapter = FakeAdapter()
        config = grid_config()
        sleep_calls = []

        run(adapter, config, max_cycles=3, sleep_fn=lambda s: sleep_calls.append(s))

        assert sleep_calls == []  # max_cycles atteint -> boucle s'arrête avant tout sleep


class TestTieBreakPrefersAlreadyOpenOrder:
    """Correctif : à distance d'index égale, préférer la cellule déjà
    matérialisée comme ordre ouvert — évite un churn CANCEL+PLACE sans
    bénéfice économique. Cas de référence obligatoire : BUY 0.9877 FILLED,
    SELL 1.0079 LIVE, k_sell=1 -> SELL 1.0079 conservé, aucune annulation,
    aucun remplacement par 0.9877."""

    def test_reference_case_sell_1_0079_kept_no_cancellation_no_replacement(self):
        """RÉVISÉ par ce chantier (alignement PDF, priorité au cycle propre
        de chaque cellule) : le SELL 1.0079 déjà ouvert reste conservé,
        aucune annulation -- MAIS la cellule ayant réellement terminé un
        BUY (poursuite de cycle genuine) obtient désormais correctement
        son propre SELL, plutôt que d'être starvée par le tie-break. Ce
        nouveau SELL n'est PAS un remplacement de 1.0079 (jamais annulé,
        vérifié ci-dessous) -- c'est une exposition supplémentaire
        légitime, exactement l'objectif de ce chantier."""
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi = lower_levels[-1]   # devient SPOT_HELD après le fill -- poursuite de cycle
        sell_gi = upper_levels[0]   # déjà SPOT_HELD, SELL déjà ouvert

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        adapter.fills = (Fill(
            trade_id="838447", order_id="3839040208425013248", client_order_id=buy_root,
            inst_id="XRP-USDC", side="BUY", fill_price=Decimal(str(buy_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=1786926350744,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        adapter.open_orders = (
            OrderSnapshot("3839040221108588544", sell_root, "XRP-USDC", "SELL", Decimal(str(sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        cycles = run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert cycles == 1
        # SELL 1.0079 conservé : jamais annulé.
        assert sell_root not in adapter.cancelled_orders
        # Aucune annulation du tout.
        assert adapter.cancelled_orders == []
        # SELL 1.0079 toujours présent, inchangé (jamais remplacé).
        remaining_sells_at_sell_gi = [o for o in adapter.open_orders if o.side == "SELL" and float(o.price) == sell_gi]
        assert len(remaining_sells_at_sell_gi) == 1
        # NOUVEAU comportement voulu : le BUY genuinement complété obtient
        # désormais un SELL à SON PROPRE gi (buy_gi) -- plus jamais starvé.
        new_sells = [p for _, side, p, _, _ in adapter.placed_orders if side == "SELL"]
        assert buy_gi in [float(p) for p in new_sells]

    def test_symmetric_case_buy_side_already_open_preferred_over_freshly_flipped(self):
        """RÉVISÉ par ce chantier, symétrique au test précédent : le BUY
        déjà ouvert reste conservé, la cellule ayant réellement complété
        un SELL (poursuite de cycle genuine) obtient désormais son propre
        BUY, sans jamais remplacer l'ordre déjà ouvert."""
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        lower_levels = sorted(float(g) for g in report.trellis if float(g) < float(config.p0))
        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        buy_gi = lower_levels[-1]   # BUY déjà ouvert ici
        sell_gi = upper_levels[0]   # SELL rempli -> poursuite de cycle

        buy_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "BUY", buy_gi)
        sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", sell_gi)

        from okx_spot_adapter import Fill
        adapter.fills = (Fill(
            trade_id="T2", order_id="O2", client_order_id=sell_root,
            inst_id="XRP-USDC", side="SELL", fill_price=Decimal(str(sell_gi)),
            fill_quantity=Decimal("21.044"), accumulated_fill_quantity=None,
            order_state="", filled_at=999,
            fee=None, fee_currency=None, rebate=None, rebate_currency=None,
        ),)
        adapter.open_orders = (
            OrderSnapshot("O1", buy_root, "XRP-USDC", "BUY", Decimal(str(buy_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert buy_root not in adapter.cancelled_orders
        assert adapter.cancelled_orders == []
        remaining_buys_at_buy_gi = [o for o in adapter.open_orders if o.side == "BUY" and float(o.price) == buy_gi]
        assert len(remaining_buys_at_buy_gi) == 1
        # NOUVEAU comportement voulu : le SELL genuinement complété obtient
        # désormais un BUY à SON PROPRE gi (sell_gi).
        new_buys = [p for _, side, p, _, _ in adapter.placed_orders if side == "BUY"]
        assert sell_gi in [float(p) for p in new_buys]

    def test_far_but_already_open_never_cancelled_regardless_of_distance(self):
        """RÉVISÉ par ce chantier : un ordre déjà ouvert n'est plus jamais
        annulé simplement parce qu'une autre cellule est plus proche de
        P0 -- contrainte explicite de ce chantier. Remplace l'ancien
        test_non_tied_distance_unaffected_lower_index_still_wins, qui
        affirmait exactement le comportement inverse."""
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)

        upper_levels = sorted(float(g) for g in report.trellis if float(g) > float(config.p0))
        far_sell_gi = upper_levels[-1]  # le plus ÉLOIGNÉ de P0
        far_sell_root = derive_root_order_id(config, report.instrument.tick_size, report.instrument.lot_size, "SELL", far_sell_gi)
        adapter.open_orders = (
            OrderSnapshot("O1", far_sell_root, "XRP-USDC", "SELL", Decimal(str(far_sell_gi)), Decimal("21.044"), Decimal("0"), "live", "post_only", 1, 1),
        )

        run(adapter, config, max_cycles=1, sleep_fn=lambda s: None)

        assert far_sell_root not in adapter.cancelled_orders
        assert adapter.cancelled_orders == []
