"""compare_cell_state.py — comparateur READ-ONLY, ancienne vs nouvelle architecture.

STRICTEMENT READ-ONLY : n'appelle jamais place_post_only_limit,
cancel_order, ni aucune écriture de cell_memory ou de
active_grid_state_<inst_id>.json. Réutilise, sans les modifier :
    - run_preflight() (grid_preflight.py)
    - reconstruct_cell_states() (cell_state_reconstruction.py) -- OLD
    - load_cell_memory() (cell_memory.py) -- lecture seule
    - reconcile_cell() (cell_reconciliation.py) -- NEW, lecture seule

ANTI-CIRCULARITÉ, garantie structurelle : si une cellule n'a pas encore
d'entrée dans cell_memory (order_status == NONE ou absente), NEW_STATE
est explicitement MEMORY_MISSING -- jamais dérivé de OLD, jamais
bootstrappé silencieusement par une seconde reconstruction déguisée.
reconcile_cell() ne peut conclure que s'il existe déjà un ordre en
cours à réconcilier ; en son absence, aucune conclusion n'est inventée.

Aucune divergence détectée ici ne déclenche jamais de correction
automatique -- ce script affiche, il ne corrige jamais.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from cell import CellState
from cell_memory import CellMemoryEntry, OrderStatus, ReconciliationStatus, load_cell_memory
from cell_reconciliation import reconcile_cell
from cell_state_reconstruction import reconstruct_cell_states
from grid_order_projection import BUY, SELL
from grid_preflight import run_preflight
from run_active_grid import load_grid_state, state_file_path_for
from run_grid_trading import build_adapter


class Verdict(Enum):
    MATCH = "MATCH"
    STATE_MISMATCH = "STATE_MISMATCH"
    ORDER_MISMATCH = "ORDER_MISMATCH"
    MEMORY_MISSING = "MEMORY_MISSING"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


@dataclass(frozen=True)
class ComparisonRow:
    gi: float
    old_state: str
    new_state: str
    old_order: str
    new_order: str
    verdict: Verdict


def _old_order_side(open_orders, gi: float) -> str:
    """Ordre réellement ouvert à ce gi, tous côtés confondus -- NONE si
    aucun. Lecture seule (open_orders déjà obtenu par un appel unique et
    partagé, jamais dupliqué)."""
    for order in open_orders:
        if order.side in (BUY, SELL) and abs(float(order.price) - gi) < 1e-12:
            return order.side
    return "NONE"


def _new_order_side(entry: CellMemoryEntry | None) -> str:
    if entry is None:
        return "NONE"
    if entry.order_status in (OrderStatus.OPEN, OrderStatus.PARTIAL) and entry.materialization_side is not None:
        return entry.materialization_side
    return "NONE"


def _classify(
    old_state: CellState,
    new_entry: CellMemoryEntry | None,
    old_order: str,
    new_order: str,
) -> Verdict:
    if new_entry is None or new_entry.order_status == OrderStatus.NONE:
        return Verdict.MEMORY_MISSING
    if new_entry.reconciliation_status == ReconciliationStatus.NEEDS_RECONCILIATION:
        return Verdict.RECONCILIATION_REQUIRED
    if new_entry.economic_state != old_state:
        return Verdict.STATE_MISMATCH
    if old_order != new_order:
        return Verdict.ORDER_MISMATCH
    return Verdict.MATCH


def compare(adapter, inst_id: str) -> tuple[ComparisonRow, ...]:
    """Construit la comparaison complète OLD vs NEW pour cet instrument.

    STRICTEMENT READ-ONLY -- aucune des fonctions appelées n'écrit
    jamais (vérifié par lecture directe de chacune, cf. docstring de
    module). N'écrit jamais cell_memory, ni active_grid_state_*.json.
    """
    state_path = state_file_path_for(inst_id)
    config = load_grid_state(state_path)
    if config is None:
        raise RuntimeError(f"{state_path} absent ou invalide -- aucune grille active à comparer.")

    report = run_preflight(adapter, config)
    open_orders = adapter.list_open_orders(inst_id)  # UN SEUL appel, partagé OLD/NEW

    # --- OLD : reconstruction inchangée, réutilisée telle quelle ---
    old_cell_states = reconstruct_cell_states(adapter, report, open_orders)
    old_by_gi = {round(float(s.cell.gi), 10): s.cell.state for s in old_cell_states}

    # --- NEW : mémoire persistée (probablement vide/incomplète) + réconciliation ---
    persisted_memory = load_cell_memory(state_path)  # lecture seule

    rows = []
    for gi_key in sorted(old_by_gi.keys()):
        old_state = old_by_gi[gi_key]
        old_order = _old_order_side(open_orders, gi_key)

        persisted_entry = persisted_memory.get(gi_key)

        if persisted_entry is None or persisted_entry.order_status == OrderStatus.NONE:
            # ANTI-CIRCULARITÉ : aucune tentative de bootstrap ici.
            new_entry = persisted_entry
        else:
            # reconcile_cell() applique SA PROPRE procédure indépendante
            # (list_fills/get_order), jamais une copie de old_state.
            import time
            new_entry = reconcile_cell(
                adapter, inst_id,
                persisted_entry.client_order_id, persisted_entry.order_id,
                persisted_entry, now_ts=int(time.time() * 1000),
            )

        new_state_label = new_entry.economic_state.name if new_entry is not None else "UNINITIALIZED"
        new_order = _new_order_side(new_entry)

        verdict = _classify(old_state, new_entry, old_order, new_order)

        rows.append(ComparisonRow(gi_key, old_state.name, new_state_label, old_order, new_order, verdict))

    return tuple(rows)


def print_report(rows: tuple[ComparisonRow, ...]) -> None:
    print(f"{'GI':<10} {'OLD_STATE':<12} {'NEW_STATE':<14} {'OLD_ORDER':<10} {'NEW_ORDER':<10} {'VERDICT'}")
    print("-" * 76)
    for row in rows:
        print(f"{row.gi:<10} {row.old_state:<12} {row.new_state:<14} {row.old_order:<10} {row.new_order:<10} {row.verdict.value}")

    counts = {v: 0 for v in Verdict}
    for row in rows:
        counts[row.verdict] += 1

    print("\n========== RÉSUMÉ ==========")
    print(f"Cellules comparées      : {len(rows)}")
    print(f"MATCH                   : {counts[Verdict.MATCH]}")
    print(f"STATE_MISMATCH          : {counts[Verdict.STATE_MISMATCH]}")
    print(f"ORDER_MISMATCH          : {counts[Verdict.ORDER_MISMATCH]}")
    print(f"MEMORY_MISSING          : {counts[Verdict.MEMORY_MISSING]}")
    print(f"RECONCILIATION_REQUIRED : {counts[Verdict.RECONCILIATION_REQUIRED]}")

    divergences = [r for r in rows if r.verdict != Verdict.MATCH]
    if divergences:
        print("\n========== DIVERGENCES DÉTAILLÉES ==========")
        for row in divergences:
            print(f"  gi={row.gi} : {row.verdict.value} -- OLD=({row.old_state},{row.old_order}) NEW=({row.new_state},{row.new_order})")
            if row.verdict == Verdict.MEMORY_MISSING:
                print(f"    -> cell_memory ne possède pas encore d'entrée exploitable pour ce niveau ; "
                      f"bootstrap distinct nécessaire avant toute comparaison significative ici.")

    print("\n========== SAFETY ==========")
    print("READ ONLY -- aucune écriture cell_memory, aucune écriture active_grid_state_*.json")
    print("NO ORDER PLACED / CANCELLED / MODIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparateur READ-ONLY OLD vs NEW.")
    parser.add_argument("inst_id")
    args = parser.parse_args()

    adapter = build_adapter("LIVE")
    rows = compare(adapter, args.inst_id)
    print_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
