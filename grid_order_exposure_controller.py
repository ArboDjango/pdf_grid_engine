"""GridOrderExposureController — politique d'exposition partielle du treillis.

Extension de ce projet, JAMAIS une règle du PDF : le PDF (Fig. 3, §3.1.2)
suppose que toutes les cellules sont actives dès l'origine — aucune notion
de fenêtre d'exposition n'y figure. Cette politique existe uniquement pour
limiter le capital immobilisé sur OKX ; elle ne modifie jamais le
mécanisme économique lui-même (repost à la position d'origine, gi
immuable, cell.py inchangé) — seulement QUELS niveaux du treillis fixe
sont actuellement matérialisés comme ordres réels.

Politique V1 : conserver exactement les k_buy niveaux CASH_HELD et les
k_sell niveaux SPOT_HELD les plus proches de P0 par position d'INDEX dans
le treillis fixe — jamais par une prédiction de marché, jamais par un
recalcul de prix. k_buy et k_sell sont des paramètres du moteur,
configurables, sans aucune signification économique tirée du PDF.
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import Cell, CellState
from grid_activation_controller import derive_grid_order_id
from grid_order_projection import BUY, SELL, OrderInstruction
from grid_preflight import PreflightReport
from okx_spot_adapter import OrderSnapshot


@dataclass(frozen=True)
class ExposureDecision:
    """Résultat de la politique d'exposition : ce qu'il faut matérialiser,
    ce qu'il faut annuler. Aucun effet de bord — cette fonction ne fait
    jamais elle-même d'appel réseau."""
    to_materialize: tuple[OrderInstruction, ...]
    to_cancel: tuple[str, ...]  # client_order_id


def select_exposed_orders(
    report: PreflightReport,
    cells: tuple[Cell, ...],
    open_orders: tuple[OrderSnapshot, ...],
    k_buy: int,
    k_sell: int,
) -> ExposureDecision:
    """Sélectionne les k_buy/k_sell niveaux les plus proches de P0.

    `cells` doit provenir de cell_state_reconstruction.reconstruct_cells
    appelée sur ce même `report` — dans le même ordre (treillis, P0
    exclu), condition nécessaire pour que l'appariement d'index ci-dessous
    soit correct.
    """
    if k_buy < 0 or k_sell < 0:
        raise ValueError("k_buy et k_sell doivent être positifs ou nuls")

    config = report.config
    instrument = report.instrument
    p0_index = config.nl
    q = float(report.q)

    indexed_cells = _pair_with_trellis_index(report, cells, p0_index)

    cash_cells = [(idx, cell) for idx, cell in indexed_cells if cell.state is CellState.CASH_HELD]
    spot_cells = [(idx, cell) for idx, cell in indexed_cells if cell.state is CellState.SPOT_HELD]

    # Les k les plus proches de P0 par distance d'INDEX — jamais par prix.
    cash_cells.sort(key=lambda pair: abs(pair[0] - p0_index))
    spot_cells.sort(key=lambda pair: abs(pair[0] - p0_index))

    open_ids = {order.client_order_id for order in open_orders}

    # Pour chaque cellule retenue, l'identité réellement ouverte prime sur
    # l'identité ROOT. Une occurrence peut être différente du ROOT :
    #
    #   BUY#1 -> SELL#1 -> BUY#2 -> ...
    #
    # La politique d'exposition ne doit jamais transformer une occurrence
    # légitime en ordre "orphelin" simplement parce que son client_order_id
    # n'est pas celui dérivé du ROOT.
    kept_buy_ids = _kept_open_ids(
        open_orders,
        config,
        instrument.tick_size,
        instrument.lot_size,
        BUY,
        cash_cells[:k_buy],
    )
    kept_sell_ids = _kept_open_ids(
        open_orders,
        config,
        instrument.tick_size,
        instrument.lot_size,
        SELL,
        spot_cells[:k_sell],
    )
    kept_ids = kept_buy_ids | kept_sell_ids

    to_materialize: list[OrderInstruction] = []

    for _, cell in cash_cells[:k_buy]:
        if not _has_open_order_at(
            open_orders,
            BUY,
            cell.gi,
        ):
            to_materialize.append((BUY, cell.gi, q))

    for _, cell in spot_cells[:k_sell]:
        if not _has_open_order_at(
            open_orders,
            SELL,
            cell.gi,
        ):
            to_materialize.append((SELL, cell.gi, q))

    to_cancel = tuple(
        order_id
        for order_id in open_ids
        if order_id not in kept_ids
    )

    return ExposureDecision(tuple(to_materialize), to_cancel)


def _has_open_order_at(
    open_orders: tuple[OrderSnapshot, ...],
    side: str,
    gi: float,
) -> bool:
    """Indique si une occurrence est déjà ouverte à ce niveau et côté.

    L'identité de l'occurrence est volontairement ignorée ici : la décision
    porte sur la cellule économique (side, gi), pas sur le ROOT.
    """
    return any(
        order.side == side
        and float(order.price) == gi
        for order in open_orders
    )


def _kept_open_ids(
    open_orders: tuple[OrderSnapshot, ...],
    config,
    tick_size,
    lot_size,
    side: str,
    selected_cells: list[tuple[int, Cell]],
) -> set[str]:
    """Retourne les identités réellement ouvertes sur les cellules retenues.

    Si une occurrence est déjà ouverte à (side, gi), son client_order_id réel
    est conservé. Sinon, le ROOT est retenu comme identité attendue pour que
    l'absence d'ordre correspondant reste matérialisable normalement.
    """
    kept_ids: set[str] = set()

    for _, cell in selected_cells:
        matching = [
            order
            for order in open_orders
            if order.side == side
            and float(order.price) == cell.gi
        ]

        if matching:
            kept_ids.add(matching[0].client_order_id)
        else:
            kept_ids.add(
                derive_grid_order_id(
                    config,
                    tick_size,
                    lot_size,
                    side,
                    cell.gi,
                )
            )

    return kept_ids

def _pair_with_trellis_index(report: PreflightReport, cells: tuple[Cell, ...], p0_index: int) -> list[tuple[int, Cell]]:
    """Ré-associe chaque cellule reconstruite à son index d'origine dans
    report.trellis — reconstruct_cells les produit dans cet ordre exact
    (treillis, P0 exclu), jamais réordonnées."""
    non_p0_indices = [i for i in range(len(report.trellis)) if i != p0_index]
    if len(non_p0_indices) != len(cells):
        raise ValueError(
            f"cells ({len(cells)}) ne correspond pas au treillis hors P0 "
            f"({len(non_p0_indices)}) — cells doit provenir de "
            "cell_state_reconstruction.reconstruct_cells sur ce même report"
        )
    return list(zip(non_p0_indices, cells))
