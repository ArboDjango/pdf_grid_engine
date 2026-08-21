"""cell_memory.py — mémoire persistante des cellules (Phase 1).

PROPOSITION ARCHITECTURALE, PAS ENCORE BRANCHÉE AU CHEMIN DE DÉCISION.

Ce module ne modifie ni ne remplace reconstruct_cell_states()
(cell_state_reconstruction.py) : il définit un schéma de persistance
indépendant, additif, qui pourra ultérieurement devenir une source de
réconciliation plus rapide que la reconstruction complète -- jamais un
remplacement de la vérité OKX elle-même (celle-ci reste toujours
prioritaire en cas de divergence, cf. cell_reconciliation.py).

Persisté dans le MÊME fichier que la configuration de grille
(active_grid_state_<inst_id>.json), sous une nouvelle clé de premier
niveau "cell_memory" -- jamais un fichier séparé (cohérent avec la
décision déjà actée : un fichier par grille, pas un fichier par
cellule). Les champs de configuration existants (p0, gll, gul, etc.)
ne sont ni lus ni modifiés par ce module -- lecture/écriture par fusion
stricte, jamais un remplacement du fichier entier.

Dix champs par cellule, exactement ceux spécifiés et validés :
    economic_state, materialization_side, order_status, client_order_id,
    order_id, trade_ids_applied, filled_qty, target_qty,
    last_transition_ts, reconciliation_status, reconciliation_since_ts

AUCUN historique complet des transitions passées n'est conservé --
uniquement le dernier état connu par cellule (déjà tranché : "trop"
serait une nouvelle dépendance fragile).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum

from cell import CellState
from grid_order_projection import BUY, SELL


class OrderStatus(Enum):
    """État technique normalisé de l'ordre courant d'une cellule --
    distinct de CellState (état économique, cell.py, jamais modifié par
    ce module). NONE : aucun ordre en cours pour cette cellule."""
    NONE = "NONE"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ReconciliationStatus(Enum):
    """Méta-état de réconciliation -- JAMAIS une valeur de CellState
    (cell.py reste strictement binaire, CD-001 inchangé). Une cellule
    NEEDS_RECONCILIATION reste économiquement CASH_HELD ou SPOT_HELD ;
    ce champ est orthogonal, jamais un troisième état économique."""
    OK = "OK"
    NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"


@dataclass(frozen=True)
class CellMemoryEntry:
    """Mémoire persistée d'une cellule unique, identifiée par son gi
    (jamais inclus ici -- c'est la clé du dictionnaire cell_memory,
    cohérent avec l'immutabilité absolue de gi : il n'a pas sa place
    comme un champ modifiable de cette structure).
    """
    economic_state: CellState
    materialization_side: str | None  # BUY, SELL, ou None (aucun ordre en cours)
    order_status: OrderStatus
    client_order_id: str | None
    order_id: str | None
    trade_ids_applied: tuple[str, ...]
    filled_qty: Decimal
    target_qty: Decimal
    last_transition_ts: int | None
    reconciliation_status: ReconciliationStatus
    reconciliation_since_ts: int | None

    def __post_init__(self):
        if self.materialization_side not in (None, BUY, SELL):
            raise ValueError(f"materialization_side invalide : {self.materialization_side!r}")
        if not isinstance(self.trade_ids_applied, tuple):
            raise TypeError("trade_ids_applied doit être un tuple (immuable, jamais une liste mutable)")


def initial_entry(economic_state: CellState, target_qty: Decimal) -> CellMemoryEntry:
    """Construit l'entrée initiale d'une cellule n'ayant encore jamais
    eu de matérialisation -- économiquement déterminée exactement comme
    aujourd'hui (CellInstantiator, gi vs P0), jamais recalculée ici."""
    return CellMemoryEntry(
        economic_state=economic_state,
        materialization_side=None,
        order_status=OrderStatus.NONE,
        client_order_id=None,
        order_id=None,
        trade_ids_applied=(),
        filled_qty=Decimal("0"),
        target_qty=target_qty,
        last_transition_ts=None,
        reconciliation_status=ReconciliationStatus.OK,
        reconciliation_since_ts=None,
    )


def serialize_entry(entry: CellMemoryEntry) -> dict:
    """Traduction pure vers un dict JSON-compatible. Decimal -> chaîne,
    jamais un flottant (cohérent avec save_grid_state existant)."""
    return {
        "economic_state": entry.economic_state.name,
        "materialization_side": entry.materialization_side,
        "order_status": entry.order_status.value,
        "client_order_id": entry.client_order_id,
        "order_id": entry.order_id,
        "trade_ids_applied": list(entry.trade_ids_applied),
        "filled_qty": str(entry.filled_qty),
        "target_qty": str(entry.target_qty),
        "last_transition_ts": entry.last_transition_ts,
        "reconciliation_status": entry.reconciliation_status.value,
        "reconciliation_since_ts": entry.reconciliation_since_ts,
    }


def deserialize_entry(payload: dict) -> CellMemoryEntry:
    """Traduction pure inverse. Lève ValueError/KeyError/TypeError sur
    tout contenu invalide -- jamais une valeur de repli devinée."""
    return CellMemoryEntry(
        economic_state=CellState[payload["economic_state"]],
        materialization_side=payload["materialization_side"],
        order_status=OrderStatus(payload["order_status"]),
        client_order_id=payload["client_order_id"],
        order_id=payload["order_id"],
        trade_ids_applied=tuple(payload["trade_ids_applied"]),
        filled_qty=Decimal(payload["filled_qty"]),
        target_qty=Decimal(payload["target_qty"]),
        last_transition_ts=payload["last_transition_ts"],
        reconciliation_status=ReconciliationStatus(payload["reconciliation_status"]),
        reconciliation_since_ts=payload["reconciliation_since_ts"],
    )


def serialize_cell_memory(cell_memory: dict[float, CellMemoryEntry]) -> dict:
    """dict[gi, CellMemoryEntry] -> dict JSON-compatible {"<gi>": {...}}.
    Clé de gi sérialisée via repr(float) implicite de str() -- cohérent
    avec la comparaison tolérante déjà utilisée ailleurs (_same_price)."""
    return {str(gi): serialize_entry(entry) for gi, entry in cell_memory.items()}


def deserialize_cell_memory(payload: dict) -> dict[float, CellMemoryEntry]:
    return {float(gi_str): deserialize_entry(entry_payload) for gi_str, entry_payload in payload.items()}


def save_cell_memory(cell_memory: dict[float, CellMemoryEntry], path: str) -> None:
    """Écrit cell_memory dans le fichier de grille existant, PAR FUSION
    STRICTE -- lit d'abord le contenu actuel (s'il existe), ne modifie
    QUE la clé "cell_memory", préserve tous les autres champs
    (configuration de grille) strictement intacts. Écriture atomique,
    identique à save_grid_state (tempfile + fsync + os.replace) --
    aucun état partiel jamais visible au chemin final.

    Si le fichier n'existe pas encore, il est créé avec uniquement la
    clé "cell_memory" -- ce module ne construit jamais la section de
    configuration, qui reste l'entière responsabilité de
    run_active_grid.py::save_grid_state (jamais dupliquée ici).
    """
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing["cell_memory"] = serialize_cell_memory(cell_memory)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".active_grid_state_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_cell_memory(path: str) -> dict[float, CellMemoryEntry]:
    """Lit la section cell_memory du fichier de grille. Retourne un dict
    vide si le fichier est absent, invalide, ou dépourvu de cette
    section -- NE LÈVE JAMAIS, cohérent avec load_grid_state existant.
    Un contenu invalide est traité exactement comme une absence, jamais
    comme une erreur fatale."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        raw_cell_memory = payload.get("cell_memory")
        if raw_cell_memory is None:
            return {}
        return deserialize_cell_memory(raw_cell_memory)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, InvalidOperation, OSError):
        return {}
