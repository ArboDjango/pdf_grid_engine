"""GridOrderExposureController — priorité au cycle propre de chaque cellule,
k_buy/k_sell comme limite de sécurité pour les cellules initiales.

Extension de ce projet, JAMAIS une règle du PDF : le PDF (Fig. 3, §3.1.2)
suppose que toutes les cellules sont actives dès l'origine — aucune notion
de fenêtre d'exposition n'y figure. Cette politique existe uniquement pour
limiter le capital immobilisé sur OKX ; elle ne modifie jamais le
mécanisme économique lui-même (repost à la position d'origine, gi
immuable, cell.py inchangé) — seulement QUELS niveaux du treillis fixe
sont actuellement matérialisés comme ordres réels.

Politique V2 (corrective) — DEUX NIVEAUX, jamais confondus :

  Niveau 1 — POURSUITE DE CYCLE (last_side connu, cf. CellReconstruction) :
    une cellule qui vient de terminer un BUY ou un SELL est fidèle au PDF
    (§3.1.2 : "place a buy/sell order AT THE ORIGINAL GRID POSITION") --
    elle doit pouvoir être réarmée à SON PROPRE gi, TOUJOURS, jamais
    limitée par k_buy/k_sell, jamais retardée parce qu'une autre cellule
    serait plus proche de P0. Ce niveau est donc INCONDITIONNEL quant à
    la distance à P0 -- seules les contraintes réelles de matérialisation
    (liquidité, price-limit, appliquées PLUS LOIN dans plan_exposure,
    jamais ici) peuvent encore différer sa matérialisation, sans jamais
    déplacer son gi ni la remplacer par une autre cellule.

  Niveau 2 — CELLULES INITIALES (last_side=None, jamais encore touchées) :
    comportement STRICTEMENT INCHANGÉ par rapport à la politique V1 --
    les k_buy/k_sell niveaux les plus proches de P0 par distance d'INDEX,
    jamais par prix. C'est ici, et ici seulement, que k_buy/k_sell
    continuent d'agir comme une limite de sécurité sur le capital
    immobilisé à l'activation initiale d'une grille neuve -- comportement
    d'activation non modifié par ce correctif.

Ces deux niveaux sont simplement concaténés (Niveau 1 puis Niveau 2) --
jamais de priorité arbitrée entre eux au-delà de cette distinction, jamais
de substitution d'une cellule par une autre, jamais de déplacement de gi.
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import Cell, CellState
from cell_state_reconstruction import CellReconstruction
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
    cell_states: tuple[CellReconstruction, ...],
    open_orders: tuple[OrderSnapshot, ...],
    k_buy: int,
    k_sell: int,
) -> ExposureDecision:
    """Sélectionne les niveaux à exposer -- poursuite de cycle (Niveau 1,
    inconditionnelle) puis cellules initiales les plus proches de P0
    (Niveau 2, plafonnées par k_buy/k_sell).

    `cell_states` doit provenir de
    cell_state_reconstruction.reconstruct_cell_states appelée sur ce même
    `report` — dans le même ordre (treillis, P0 exclu), condition
    nécessaire pour que l'appariement d'index ci-dessous soit correct.
    """
    if k_buy < 0 or k_sell < 0:
        raise ValueError("k_buy et k_sell doivent être positifs ou nuls")

    config = report.config
    instrument = report.instrument
    p0_index = config.nl
    q = float(report.q)

    indexed = _pair_with_trellis_index(report, cell_states, p0_index)

    cash_candidates = [(idx, cs) for idx, cs in indexed if cs.cell.state is CellState.CASH_HELD]
    spot_candidates = [(idx, cs) for idx, cs in indexed if cs.cell.state is CellState.SPOT_HELD]

    # Étape 0, AVANT toute classification par niveau : une cellule dont
    # l'ordre nécessaire est DÉJÀ ouvert (qu'il ait ou non déjà été rempli
    # une fois auparavant) n'a besoin d'AUCUNE nouvelle décision -- elle
    # est simplement conservée telle quelle. Sans cette étape, last_side
    # est ambigu : un ordre encore EN ATTENTE (jamais rempli) porte aussi
    # last_side non nul (cf. reconstruct_cell_states, branche "ordre
    # actuellement ouvert"), ce qui la ferait à tort libérer le budget du
    # Niveau 2 sans jamais consommer le sien -- une source réelle de
    # sur-exposition au-delà de k_buy/k_sell, découverte et corrigée ici.
    cash_already_open = [(idx, cs) for idx, cs in cash_candidates if _has_open_order_at(open_orders, BUY, cs.cell.gi)]
    spot_already_open = [(idx, cs) for idx, cs in spot_candidates if _has_open_order_at(open_orders, SELL, cs.cell.gi)]

    cash_needs_action = [(idx, cs) for idx, cs in cash_candidates if not _has_open_order_at(open_orders, BUY, cs.cell.gi)]
    spot_needs_action = [(idx, cs) for idx, cs in spot_candidates if not _has_open_order_at(open_orders, SELL, cs.cell.gi)]

    # Niveau 1 : poursuite de cycle -- last_side connu ET aucun ordre déjà
    # ouvert pour ce côté (donc un événement RÉELLEMENT terminé, jamais un
    # ordre simplement en attente) -- INCONDITIONNEL, jamais trié par
    # distance (aucune compétition entre elles : toutes incluses, fidèles
    # au PDF, "at the original grid position").
    cash_cycle = [(idx, cs) for idx, cs in cash_needs_action if cs.last_side is not None]
    spot_cycle = [(idx, cs) for idx, cs in spot_needs_action if cs.last_side is not None]

    # Niveau 2 : cellules initiales -- comportement V1 strictement inchangé,
    # tri par distance d'index à P0, départage par ordre déjà ouvert en cas
    # d'égalité, plafonné par k_buy/k_sell.
    cash_initial = [(idx, cs) for idx, cs in cash_needs_action if cs.last_side is None]
    spot_initial = [(idx, cs) for idx, cs in spot_needs_action if cs.last_side is None]
    cash_initial.sort(key=lambda pair: (abs(pair[0] - p0_index), 0 if _has_open_order_at(open_orders, BUY, pair[1].cell.gi) else 1))
    spot_initial.sort(key=lambda pair: (abs(pair[0] - p0_index), 0 if _has_open_order_at(open_orders, SELL, pair[1].cell.gi) else 1))

    selected_cash = cash_already_open + cash_cycle + cash_initial[:k_buy]
    selected_spot = spot_already_open + spot_cycle + spot_initial[:k_sell]

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
        [(idx, cs.cell) for idx, cs in selected_cash],
    )
    kept_sell_ids = _kept_open_ids(
        open_orders,
        config,
        instrument.tick_size,
        instrument.lot_size,
        SELL,
        [(idx, cs.cell) for idx, cs in selected_spot],
    )
    kept_ids = kept_buy_ids | kept_sell_ids

    to_materialize: list[OrderInstruction] = []

    for _, cs in selected_cash:
        if not _has_open_order_at(
            open_orders,
            BUY,
            cs.cell.gi,
        ):
            to_materialize.append((BUY, cs.cell.gi, q))

    for _, cs in selected_spot:
        if not _has_open_order_at(
            open_orders,
            SELL,
            cs.cell.gi,
        ):
            to_materialize.append((SELL, cs.cell.gi, q))

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

def _pair_with_trellis_index(report: PreflightReport, cell_states: tuple[CellReconstruction, ...], p0_index: int) -> list[tuple[int, CellReconstruction]]:
    """Ré-associe chaque cellule reconstruite à son index d'origine dans
    report.trellis — reconstruct_cell_states les produit dans cet ordre
    exact (treillis, P0 exclu), jamais réordonnées."""
    non_p0_indices = [i for i in range(len(report.trellis)) if i != p0_index]
    if len(non_p0_indices) != len(cell_states):
        raise ValueError(
            f"cell_states ({len(cell_states)}) ne correspond pas au treillis hors P0 "
            f"({len(non_p0_indices)}) — cell_states doit provenir de "
            "cell_state_reconstruction.reconstruct_cell_states sur ce même report"
        )
    return list(zip(non_p0_indices, cell_states))
