"""Reconstruction de l'état des cellules depuis l'état réel OKX.

La cellule économique reste définie uniquement par son gi et son CellState.
L'identité de la dernière occurrence d'ordre est conservée séparément afin
de permettre le chaînage :

    BUY @ gi #1 -> SELL @ gi #1 -> BUY @ gi #2 -> ...

Aucune persistance locale n'est utilisée.
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import Cell, CellState
from grid_activation_controller import derive_grid_order_id
from grid_order_projection import BUY, SELL
from grid_preflight import PreflightReport
from okx_spot_adapter import OkxApiError, Fill, OrderSnapshot


@dataclass(frozen=True)
class CellReconstruction:
    """État économique + dernière occurrence connue de la cellule."""

    cell: Cell
    last_order_id: str | None
    last_client_order_id: str | None
    last_side: str | None
    last_filled_at: int | None


def reconstruct_cells(
    adapter,
    report: PreflightReport,
    open_orders: tuple[OrderSnapshot, ...],
) -> tuple[Cell, ...]:
    """API historique : retourne uniquement les Cell."""

    return tuple(
        item.cell
        for item in reconstruct_cell_states(adapter, report, open_orders)
    )


def reconstruct_cell_states(
    adapter,
    report: PreflightReport,
    open_orders: tuple[OrderSnapshot, ...],
) -> tuple[CellReconstruction, ...]:
    """Reconstruit état et dernière occurrence de chaque cellule.

    Sources de vérité, dans l'ordre :

    1. ordre actuellement ouvert à gi ;
    2. fills OKX permettant de reconstruire la chaîne des occurrences ;
    3. historique ROOT via get_order(), pour compatibilité avec les ordres
       historiques encore accessibles mais sans fill retourné ;
    4. état initial si aucune information exploitable n'existe.

    P0 n'est jamais une cellule.
    """

    config = report.config
    instrument = report.instrument
    p0_index = config.nl
    p0 = float(config.p0)

    # Un seul sondage ouvert fourni par l'appelant.
    open_by_id = {
        order.client_order_id: order
        for order in open_orders
    }

    # Historique des fills.
    try:
        fills = adapter.list_fills(config.inst_id)
    except OkxApiError:
        fills = ()

    grid_fills = tuple(
        fill
        for fill in fills
        if _belongs_to_this_grid(fill, report)
        # NOTE : aucune condition sur fill.order_state ici.
        #
        # GET /api/v5/trade/fills ne contient structurellement JAMAIS de
        # champ "state" -- confirmé sur la documentation officielle OKX
        # (schéma réel : instType, instId, tradeId, ordId, clOrdId,
        # billId, tag, fillPx, fillSz, side, posSide, execType, feeCcy,
        # fee, ts). "state" appartient exclusivement à GET
        # /api/v5/trade/order (niveau ordre), jamais à /trade/fills
        # (niveau exécution). La présence même d'un enregistrement dans
        # list_fills() CONSTITUE déjà la preuve d'exécution -- il n'existe
        # pas de "fill non exécuté" retourné par cet endpoint.
        #
        # L'ancienne condition `and fill.order_state == "filled"` reposait
        # sur une hypothèse de contrat erronée : Fill.order_state, dérivé
        # de item.get("state", "") dans okx_spot_adapter.py::_fill, vaut
        # toujours "" pour un fill réel -- rendant cette voie de
        # reconstruction structurellement inatteignable en production,
        # confirmé concrètement sur un BUY XRP-USDC réel exécuté à 0.9877
        # (tradeId=838447), jamais reconnu avant ce correctif.
    )

    cells: list[CellReconstruction] = []

    for index, gi_decimal in enumerate(report.trellis):
        if index == p0_index:
            continue

        gi = float(gi_decimal)

        buy_root_id = derive_grid_order_id(
            config,
            instrument.tick_size,
            instrument.lot_size,
            BUY,
            gi,
        )
        sell_root_id = derive_grid_order_id(
            config,
            instrument.tick_size,
            instrument.lot_size,
            SELL,
            gi,
        )

        cells.append(
            _reconstruct_one_cell_state(
                adapter=adapter,
                inst_id=config.inst_id,
                gi=gi,
                p0=p0,
                buy_root_id=buy_root_id,
                sell_root_id=sell_root_id,
                open_orders=open_orders,
                open_by_id=open_by_id,
                fills=grid_fills,
            )
        )

    return tuple(cells)


def _reconstruct_one_cell_state(
    adapter,
    inst_id: str,
    gi: float,
    p0: float,
    buy_root_id: str,
    sell_root_id: str,
    open_orders: tuple[OrderSnapshot, ...],
    open_by_id: dict[str, OrderSnapshot],
    fills: tuple[Fill, ...],
) -> CellReconstruction:

    # ------------------------------------------------------------------
    # 1. Ordre actuellement ouvert.
    #
    # IMPORTANT : ne pas rechercher uniquement le ROOT.
    # Une occurrence suivante possède un nouvel ID.
    # L'identité économique est side + gi.
    # ------------------------------------------------------------------

    matching_open = [
        order
        for order in open_orders
        if order.side in (BUY, SELL)
        and _same_price(order.price, gi)
    ]

    if matching_open:
        # Une seule occurrence opérationnelle est normalement possible
        # à un même gi. Si plusieurs apparaissent, on conserve la première
        # mais l'ordre reste explicitement prioritaire sur l'historique.
        order = matching_open[0]

        state = (
            CellState.CASH_HELD
            if order.side == BUY
            else CellState.SPOT_HELD
        )

        return CellReconstruction(
            Cell(gi, state),
            order.order_id,
            order.client_order_id,
            order.side,
            order.updated_at,
        )

    # ------------------------------------------------------------------
    # 2. Historique des fills.
    #
    # Toutes les occurrences GRID au même gi participent à la chaîne.
    # Le dernier fill détermine l'état économique.
    # ------------------------------------------------------------------

    matching = [
        fill
        for fill in fills
        if fill.side in (BUY, SELL)
        and _same_price(fill.fill_price, gi)
    ]

    if matching:
        last = max(
            matching,
            key=lambda fill: (
                fill.filled_at if fill.filled_at is not None else 0,
                fill.trade_id,
            ),
        )

        state = (
            CellState.SPOT_HELD
            if last.side == BUY
            else CellState.CASH_HELD
        )

        return CellReconstruction(
            Cell(gi, state),
            last.order_id,
            last.client_order_id,
            last.side,
            last.filled_at,
        )

    # ------------------------------------------------------------------
    # 3. Compatibilité historique ROOT.
    #
    # Avant l'introduction des occurrences, les tests et certaines grilles
    # historiques n'ont pas nécessairement de Fill exploitable via
    # list_fills(). On conserve donc get_order() sur les deux ROOT.
    # ------------------------------------------------------------------

    buy_snapshot = _safe_get_order(
        adapter,
        inst_id,
        buy_root_id,
    )
    sell_snapshot = _safe_get_order(
        adapter,
        inst_id,
        sell_root_id,
    )

    buy_filled = (
        buy_snapshot is not None
        and buy_snapshot.state == "filled"
    )
    sell_filled = (
        sell_snapshot is not None
        and sell_snapshot.state == "filled"
    )

    if buy_filled and sell_filled:
        buy_time = buy_snapshot.updated_at or 0
        sell_time = sell_snapshot.updated_at or 0

        if buy_time > sell_time:
            return CellReconstruction(
                Cell(gi, CellState.SPOT_HELD),
                buy_snapshot.order_id,
                buy_snapshot.client_order_id,
                BUY,
                buy_snapshot.updated_at,
            )

        return CellReconstruction(
            Cell(gi, CellState.CASH_HELD),
            sell_snapshot.order_id,
            sell_snapshot.client_order_id,
            SELL,
            sell_snapshot.updated_at,
        )

    if buy_filled:
        return CellReconstruction(
            Cell(gi, CellState.SPOT_HELD),
            buy_snapshot.order_id,
            buy_snapshot.client_order_id,
            BUY,
            buy_snapshot.updated_at,
        )

    if sell_filled:
        return CellReconstruction(
            Cell(gi, CellState.CASH_HELD),
            sell_snapshot.order_id,
            sell_snapshot.client_order_id,
            SELL,
            sell_snapshot.updated_at,
        )

    # ------------------------------------------------------------------
    # 4. Aucun historique exploitable : état initial du PDF/CD-003.
    # ------------------------------------------------------------------

    state = (
        CellState.SPOT_HELD
        if gi > p0
        else CellState.CASH_HELD
    )

    return CellReconstruction(
        Cell(gi, state),
        None,
        None,
        None,
        None,
    )


def _same_price(value, gi: float) -> bool:
    """Comparaison numérique tolérante pour Decimal/float."""

    return abs(float(value) - gi) < 1e-12


def _belongs_to_this_grid(
    fill: Fill,
    report: PreflightReport,
) -> bool:
    """Reconnaît les fills appartenant à cette configuration."""

    return (
        fill.inst_id == report.config.inst_id
        and fill.client_order_id.startswith("GRID")
    )


def _safe_get_order(
    adapter,
    inst_id: str,
    client_order_id: str,
) -> OrderSnapshot | None:
    try:
        return adapter.get_order(
            inst_id,
            client_order_id=client_order_id,
        )
    except OkxApiError:
        return None
