"""Tests pour le correctif de reconstruction : reconnaissance des fills OKX
réels, dont le champ "state" est structurellement absent (confirmé :
GET /api/v5/trade/fills ne contient jamais ce champ, quel que soit
l'instrument -- "state" appartient exclusivement à GET /api/v5/trade/order).

Aucun appel réseau réel. Adaptateur simulé, aucune méthode d'écriture
n'est jamais appelée ni ne doit l'être -- vérifié explicitement.

N'utilise PAS cell_order_identity.py (derive_root_order_id/derive_next_order_id)
-- ce fichier n'a été fourni dans aucun lot de cette conversation. Les
client_order_id utilisés ici sont des chaînes "GRID..." construites
directement, suffisantes pour exercer reconstruct_cell_states (qui ne
connaît jamais comment un identifiant a été dérivé -- seulement son
préfixe "GRID" et sa correspondance side+gi).
"""

from decimal import Decimal

from cell import CellState
from cell_state_reconstruction import reconstruct_cell_states
from grid_activation_controller import derive_grid_order_id
from grid_preflight import PreflightConfig, run_preflight
from okx_spot_adapter import (
    Balances,
    FeeRates,
    Fill,
    InstrumentRules,
    OkxApiError,
    OrderSnapshot,
)
from trellis_calculator import GridGeometry


class FakeAdapter:
    """Toute méthode d'écriture lève si appelée -- vérifie l'absence
    structurelle d'appel d'écriture pendant la reconstruction (test 6)."""

    def __init__(self, *, base="1000", quote="1000", maker="0.001", tick="0.0001", lot="0.1", minimum="0.1"):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal(tick), Decimal(lot), Decimal(minimum), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.open_orders = ()
        self.fills = ()
        self.get_order_results = {}
        self.write_calls = []

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

    def place_post_only_limit(self, *a, **k):
        self.write_calls.append("place_post_only_limit")
        raise AssertionError("ÉCRITURE DÉTECTÉE — reconstruct_cell_states ne doit jamais écrire")

    def place_market_buy(self, *a, **k):
        self.write_calls.append("place_market_buy")
        raise AssertionError("ÉCRITURE DÉTECTÉE")

    def cancel_order(self, *a, **k):
        self.write_calls.append("cancel_order")
        raise AssertionError("ÉCRITURE DÉTECTÉE")


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.05"), gll=Decimal("0.95"), nu=2, nl=2,
        p0=Decimal("1.00"), geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def real_fill(client_order_id, order_id, trade_id, side, price, quantity, filled_at, order_state=""):
    """Reproduit exactement le contrat réel de /trade/fills -- order_state
    par défaut à '' (chaîne vide), comme le produit réellement _fill()
    face à un champ 'state' structurellement absent."""
    return Fill(
        trade_id=trade_id, order_id=order_id, client_order_id=client_order_id,
        inst_id="XRP-USDC", side=side, fill_price=Decimal(str(price)),
        fill_quantity=Decimal(str(quantity)), accumulated_fill_quantity=None,
        order_state=order_state, filled_at=filled_at,
        fee=None, fee_currency=None, rebate=None, rebate_currency=None,
    )


class TestRealXrpBuyFillRecognized:
    """Cas 1 et 2 : le BUY XRP-USDC réel fourni pour ce chantier."""

    def test_1_fill_without_state_field_is_recognized_as_execution_proof(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])  # niveau < P0

        client_order_id = "GRIDe910f6b329b6232c473553b5"
        adapter.fills = (real_fill(client_order_id, "3839040208425013248", "838447", "BUY", gi, "21.044", 1786926350744, order_state=""),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.SPOT_HELD  # test 2

    def test_2_full_identity_fields_match_the_real_fill_exactly(self):
        """Reproduction exacte du fill XRP-USDC réel fourni pour ce chantier."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])

        client_order_id = "GRIDe910f6b329b6232c473553b5"
        order_id = "3839040208425013248"
        adapter.fills = (real_fill(client_order_id, order_id, "838447", "BUY", gi, "21.044", 1786926350744, order_state=""),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.SPOT_HELD
        assert matching.last_order_id == order_id
        assert matching.last_client_order_id == client_order_id
        assert matching.last_side == "BUY"
        assert matching.last_filled_at == 1786926350744


class TestSellFillRecognizedSymmetrically:
    def test_3_sell_fill_without_state_recognized_as_cash_held(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[-1])  # niveau > P0

        client_order_id = "GRIDsell0000000000000000"
        adapter.fills = (real_fill(client_order_id, "ORD-SELL-1", "TRADE-1", "SELL", gi, "10.0", 500, order_state=""),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.CASH_HELD
        assert matching.last_side == "SELL"
        assert matching.last_client_order_id == client_order_id


class TestIncompleteOrInvalidFillNotArtificiallyRecognized:
    def test_4_fill_not_belonging_to_this_grid_is_ignored(self):
        """clOrdId ne commence pas par 'GRID' -- n'appartient pas à cette grille, jamais reconnu."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])

        adapter.fills = (real_fill("FOREIGNxxxx", "ORDX", "TRADEX", "BUY", gi, "5.0", 100, order_state=""),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        # Non reconnu comme fill de cette grille -> retombe sur l'état initial
        assert matching.last_client_order_id is None
        assert matching.cell.state is CellState.CASH_HELD  # état initial (gi < P0)

    def test_4_fill_wrong_instrument_is_ignored(self):
        """instId différent -- n'appartient pas à cette grille, jamais reconnu."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])

        wrong_inst_fill = Fill(
            trade_id="T1", order_id="O1", client_order_id="GRIDxxxx",
            inst_id="ETH-USDC", side="BUY", fill_price=Decimal(str(gi)),
            fill_quantity=Decimal("1"), accumulated_fill_quantity=None,
            order_state="", filled_at=100, fee=None, fee_currency=None,
            rebate=None, rebate_currency=None,
        )
        adapter.fills = (wrong_inst_fill,)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.last_client_order_id is None


class TestNoOpenOrderNeededForHistoricalFill:
    def test_5_historical_fill_recognized_without_any_open_order(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])

        adapter.fills = (real_fill("GRIDhist0000", "ORD1", "T1", "BUY", gi, "1.0", 100, order_state=""),)
        adapter.open_orders = ()  # explicitement vide

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.SPOT_HELD


class TestNoWriteMethodCalled:
    def test_6_no_write_method_called_during_reconstruction(self):
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        adapter.fills = (real_fill("GRIDwrite000", "ORD1", "T1", "BUY", gi, "1.0", 100, order_state=""),)

        reconstruct_cell_states(adapter, report, open_orders=())

        assert adapter.write_calls == []


class TestNonRegressionAllFourPaths:
    def test_7_open_order_path_unaffected_by_fix(self):
        """OPEN_ORDER : chemin totalement indépendant du filtre corrigé."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        buy_id = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        adapter.open_orders = (OrderSnapshot("ORD1", buy_id, "XRP-USDC", "BUY", Decimal(str(gi)), Decimal("1"), Decimal("0"), "live", "post_only", None, None),)

        states = reconstruct_cell_states(adapter, report, adapter.open_orders)
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.CASH_HELD  # BUY ouvert -> cash
        assert matching.last_client_order_id == buy_id

    def test_7_fill_path_still_works_with_order_state_filled_explicit(self):
        """FILL : non-régression -- un fill avec order_state='filled' explicite (ancien style de test) continue de fonctionner."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        adapter.fills = (real_fill("GRIDlegacy00", "ORD1", "T1", "BUY", gi, "1.0", 100, order_state="filled"),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.SPOT_HELD

    def test_7_get_order_path_still_reachable_when_no_fill_matches(self):
        """GET_ORDER : chemin de repli toujours atteint si aucun fill ne correspond."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])
        buy_root = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)

        from okx_spot_adapter import OrderSnapshot as OS
        adapter.get_order_results[buy_root] = OS("ORD1", buy_root, "XRP-USDC", "BUY", Decimal(str(gi)), Decimal("1"), Decimal("1"), "filled", "post_only", None, 999)
        adapter.fills = ()  # aucun fill -- doit retomber sur GET_ORDER, pas rester bloqué

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.cell.state is CellState.SPOT_HELD
        assert matching.last_client_order_id == buy_root

    def test_7_initial_state_path_still_reachable_when_nothing_matches(self):
        """INITIAL_STATE : aucun ordre, aucun fill, aucun historique -- état initial du PDF."""
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())

        states = reconstruct_cell_states(adapter, report, open_orders=())

        for s in states:
            assert s.last_client_order_id is None
            expected = CellState.SPOT_HELD if s.cell.gi > float(report.config.p0) else CellState.CASH_HELD
            assert s.cell.state is expected


class TestClientOrderIdNeverInventedOrRecalculated:
    def test_8_last_client_order_id_matches_fill_exactly_never_recalculated(self):
        """
        Vérifie qu'aucun identifiant n'est recalculé/dérivé pour remplacer
        celui réellement porté par le fill -- last_client_order_id doit
        être EXACTEMENT la valeur du champ clOrdId du fill, jamais un ROOT
        recalculé indépendamment (qui, ici, serait différent puisque
        client_order_id utilisé est délibérément arbitraire, pas dérivé
        de derive_grid_order_id).
        """
        adapter = FakeAdapter()
        report = run_preflight(adapter, grid_config())
        gi = float(report.trellis[0])

        # Client_order_id délibérément DIFFÉRENT de ce que derive_grid_order_id
        # produirait pour (BUY, gi) -- pour prouver qu'il n'est jamais recalculé.
        arbitrary_client_order_id = "GRIDzzzz9999888877776666"
        computed_root = derive_grid_order_id(report.config, report.instrument.tick_size, report.instrument.lot_size, "BUY", gi)
        assert arbitrary_client_order_id != computed_root  # précondition du test

        adapter.fills = (real_fill(arbitrary_client_order_id, "ORD1", "T1", "BUY", gi, "1.0", 100, order_state=""),)

        states = reconstruct_cell_states(adapter, report, open_orders=())
        matching = next(s for s in states if abs(s.cell.gi - gi) < 1e-9)

        assert matching.last_client_order_id == arbitrary_client_order_id
        assert matching.last_client_order_id != computed_root
