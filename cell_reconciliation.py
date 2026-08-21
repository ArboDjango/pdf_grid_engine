"""cell_reconciliation.py — réconciliation mémoire persistée <-> OKX réel
(Phase 2-3).

PROPOSITION ARCHITECTURALE, PAS ENCORE BRANCHÉE AU CHEMIN DE DÉCISION.
Composant séparé, testable indépendamment -- ne modifie ni
select_exposed_orders(), ni reconstruct_cell_states(), ni aucun fichier
existant. Aucun appel d'écriture (place_post_only_limit, cancel_order)
n'est jamais effectué ici -- uniquement des lectures (list_fills,
get_order) et des calculs purs.

ORDRE STRICT, jamais réordonné :
    1. list_fills()
    2. somme des fills NON DÉJÀ APPLIQUÉS (trade_ids_applied)
    3. FULL FILL confirmé -> transition économique
    4. sinon get_order()
    5. CANCELED confirmé -> état économique INCHANGÉ
    6. absence métier des deux côtés -> NEEDS_RECONCILIATION
    7. erreur réseau (pas une réponse métier) -> AUCUNE modification mémoire

Invariants garantis par construction, jamais par convention :
    - PARTIAL ne change jamais economic_state (cf. reconcile_cell, la
      seule branche qui modifie economic_state est le FULL FILL).
    - Un trade_id déjà dans trade_ids_applied n'est jamais recompté.
    - Une exception réseau (pas OkxApiError) retourne l'entry INCHANGÉE,
      objet identique, jamais une copie modifiée même partiellement.
"""

from __future__ import annotations

from decimal import Decimal

from cell import CellState
from cell_memory import CellMemoryEntry, OrderStatus, ReconciliationStatus
from grid_order_projection import BUY, SELL
from okx_spot_adapter import OkxApiError


class NetworkFailure(Exception):
    """Distincte de OkxApiError (erreur métier confirmée) -- signale à
    l'appelant qu'AUCUNE conclusion n'a pu être tirée, jamais une absence
    interprétée comme une donnée. reconcile_cell() ne lève jamais cette
    exception elle-même : elle catch les pannes de transport et retourne
    l'entry inchangée -- ce type existe pour la traçabilité/le
    diagnostic, en cas d'usage futur qui voudrait la distinguer."""


def reconcile_cell(
    adapter,
    inst_id: str,
    client_order_id: str | None,
    order_id: str | None,
    entry: CellMemoryEntry,
    now_ts: int,
) -> CellMemoryEntry:
    """Réconcilie une cellule unique contre l'état réel OKX, selon
    l'ordre strict documenté en tête de module.

    Si entry.order_status est NONE (aucun ordre en cours), retourne
    entry inchangée -- rien à réconcilier.

    Args:
        adapter: Source OKX (list_fills, get_order).
        inst_id: Instrument concerné.
        client_order_id/order_id: Identité de l'ordre en cours pour
            cette cellule (entry.client_order_id/entry.order_id,
            transmis explicitement par l'appelant pour garder cette
            fonction pure vis-à-vis de la structure de stockage).
        entry: Mémoire actuellement persistée pour cette cellule.
        now_ts: Horodatage courant (pour reconciliation_since_ts).

    Returns:
        Une NOUVELLE CellMemoryEntry (ou l'objet `entry` inchangé si
        aucune conclusion n'a pu être tirée -- réseau, ou ambiguïté déjà
        déclarée sans nouvelle preuve).
    """
    if entry.order_status == OrderStatus.NONE:
        return entry

    # ------------------------------------------------------------
    # Étape 1-2 : list_fills(), somme des fills non déjà appliqués.
    # ------------------------------------------------------------
    try:
        fills = adapter.list_fills(inst_id, order_id=order_id)
    except OkxApiError:
        fills = ()  # absence métier confirmée -- pas un échec réseau
    except Exception:
        # Panne de transport réelle -- AUCUNE conclusion, entry inchangée.
        return entry

    matching_fills = tuple(
        fill for fill in fills
        if fill.client_order_id == client_order_id
        and fill.trade_id not in entry.trade_ids_applied
    )

    new_filled_qty = entry.filled_qty + sum((f.fill_quantity for f in matching_fills), Decimal("0"))
    new_trade_ids = entry.trade_ids_applied + tuple(f.trade_id for f in matching_fills)

    # ------------------------------------------------------------
    # Étape 3 : FULL FILL confirmé -> transition économique.
    # ------------------------------------------------------------
    if new_filled_qty >= entry.target_qty:
        new_economic_state = (
            CellState.SPOT_HELD if entry.materialization_side == BUY else CellState.CASH_HELD
        )
        return CellMemoryEntry(
            economic_state=new_economic_state,
            materialization_side=None,
            order_status=OrderStatus.FILLED,
            client_order_id=entry.client_order_id,
            order_id=entry.order_id,
            trade_ids_applied=new_trade_ids,
            filled_qty=new_filled_qty,
            target_qty=entry.target_qty,
            last_transition_ts=now_ts,
            reconciliation_status=ReconciliationStatus.OK,
            reconciliation_since_ts=None,
        )

    # Fill partiel détecté (nouveaux trade_ids appliqués, mais pas FULL) :
    # persister l'avancement, jamais transitionner economic_state.
    if matching_fills:
        return CellMemoryEntry(
            economic_state=entry.economic_state,  # INCHANGÉ -- invariant absolu
            materialization_side=entry.materialization_side,
            order_status=OrderStatus.PARTIAL,
            client_order_id=entry.client_order_id,
            order_id=entry.order_id,
            trade_ids_applied=new_trade_ids,
            filled_qty=new_filled_qty,
            target_qty=entry.target_qty,
            last_transition_ts=now_ts,
            reconciliation_status=ReconciliationStatus.OK,
            reconciliation_since_ts=None,
        )

    # ------------------------------------------------------------
    # Étape 4 : sinon get_order().
    # ------------------------------------------------------------
    try:
        order = adapter.get_order(inst_id, client_order_id=client_order_id)
    except OkxApiError:
        order = None  # absence métier confirmée
    except Exception:
        return entry  # panne réseau -- AUCUNE conclusion

    # ------------------------------------------------------------
    # Étape 5 : CANCELED confirmé -> état économique INCHANGÉ.
    # ------------------------------------------------------------
    if order is not None and order.state == "canceled":
        return CellMemoryEntry(
            economic_state=entry.economic_state,  # INCHANGÉ
            materialization_side=None,
            order_status=OrderStatus.CANCELED,
            client_order_id=entry.client_order_id,
            order_id=entry.order_id,
            trade_ids_applied=entry.trade_ids_applied,
            filled_qty=entry.filled_qty,
            target_qty=entry.target_qty,
            last_transition_ts=now_ts,
            reconciliation_status=ReconciliationStatus.OK,
            reconciliation_since_ts=None,
        )

    if order is not None and order.state in ("live", "partially_filled"):
        # Toujours vivant sur OKX -- pas de fill nouveau détecté à
        # l'étape 1-2 (fenêtre de course possible), mais rien à changer.
        return entry

    # ------------------------------------------------------------
    # Étape 6 : absence métier des deux côtés -> NEEDS_RECONCILIATION.
    #
    # Confirmé et validé explicitement : aucune tentative de distinguer
    # "jamais existé" de "devenu inaccessible" -- toute absence de
    # preuve positive de FILLED/CANCELED va ici, sans exception.
    # ------------------------------------------------------------
    return CellMemoryEntry(
        economic_state=entry.economic_state,  # INCHANGÉ
        materialization_side=entry.materialization_side,
        order_status=OrderStatus.UNKNOWN,
        client_order_id=entry.client_order_id,
        order_id=entry.order_id,
        trade_ids_applied=entry.trade_ids_applied,
        filled_qty=entry.filled_qty,
        target_qty=entry.target_qty,
        last_transition_ts=entry.last_transition_ts,
        reconciliation_status=ReconciliationStatus.NEEDS_RECONCILIATION,
        reconciliation_since_ts=(
            entry.reconciliation_since_ts
            if entry.reconciliation_status == ReconciliationStatus.NEEDS_RECONCILIATION
            else now_ts
        ),
    )


# Seuil d'alerte validé explicitement : 10 cycles consécutifs,
# CYCLE_INTERVAL_SECONDS=60 (déjà la constante de run_active_grid.py,
# jamais dupliquée ici en valeur -- seul le nombre de cycles est fixé).
RECONCILIATION_ALERT_CYCLES = 10


def needs_operator_alert(entry: CellMemoryEntry, now_ts: int, cycle_interval_seconds: int) -> bool:
    """True si une cellule est NEEDS_RECONCILIATION depuis au moins
    RECONCILIATION_ALERT_CYCLES cycles consécutifs. Ne résout jamais
    l'ambiguïté elle-même -- seuil d'alerte uniquement, jamais une
    correction automatique."""
    if entry.reconciliation_status != ReconciliationStatus.NEEDS_RECONCILIATION:
        return False
    if entry.reconciliation_since_ts is None:
        return False
    elapsed = now_ts - entry.reconciliation_since_ts
    return elapsed >= RECONCILIATION_ALERT_CYCLES * cycle_interval_seconds
