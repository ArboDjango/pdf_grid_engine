#!/usr/bin/env python3
"""initialize_fresh_cell_memory.py — initialise cell_memory pour une
grille NEUVE, déterministiquement, sans aucune dépendance à l'historique
OKX (list_fills, list_open_orders, get_order).

OUTIL OPÉRATEUR EXPLICITE -- jamais appelé automatiquement par
run_active_grid.py ni par systemd.

RÈGLE D'INITIALISATION : réutilise directement CellInstantiator (CD-003,
cell_instantiator.py, INCHANGÉ) -- gi < P0 -> CASH_HELD, gi > P0 ->
SPOT_HELD -- jamais réimplémentée une seconde fois ici.

STRICTEMENT INTERDIT dans ce script, vérifié structurellement par les
tests associés : place_post_only_limit, cancel_order, list_fills,
list_open_orders, get_order. Seuls get_instrument/get_fee_rates/
get_balances sont appelés, via run_preflight() -- nécessaires pour
obtenir le treillis réel et q, jamais pour interroger l'historique
d'ordres.

IDEMPOTENCE : si une cell_memory non vide existe déjà dans le fichier
cible, le script REFUSE par défaut (CELL_MEMORY_ALREADY_EXISTS) et
quitte sans écrire -- sauf --force explicite.

Ne modifie ni PreflightConfig, ni la grille, ni aucun champ de
configuration déjà présent dans active_grid_state_<inst_id>.json
(save_cell_memory fusionne strictement, cf. cell_memory.py, Phase 1).

Usage :
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... \\
        python3 initialize_fresh_cell_memory.py XRP-USDC
    ... --force   # écrase une cell_memory déjà existante, explicitement
"""

from __future__ import annotations

import argparse
import sys

from cell_instantiator import CellInstantiator
from cell_memory import initial_entry, load_cell_memory, save_cell_memory
from grid_preflight import run_preflight
from run_active_grid import load_grid_state, state_file_path_for
from run_grid_trading import build_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialise cell_memory pour une grille neuve.")
    parser.add_argument("inst_id")
    parser.add_argument("--force", action="store_true", help="Écrase une cell_memory déjà existante.")
    return parser.parse_args()


def build_fresh_cell_memory(adapter, config) -> dict:
    """Construit la mémoire initiale, exclusivement à partir de la
    géométrie -- AUCUN appel à list_fills/list_open_orders/get_order.

    Réutilise CellInstantiator (règle gi<P0/gi>P0 déjà établie, jamais
    réimplémentée) et initial_entry() (cell_memory.py, Phase 1, jamais
    réimplémenté).
    """
    report = run_preflight(adapter, config)  # get_instrument/get_fee_rates/get_balances uniquement

    trellis_floats = tuple(float(level) for level in report.trellis)
    cells = CellInstantiator().instantiate(trellis_floats, float(config.p0))

    return {
        cell.gi: initial_entry(cell.state, report.q)
        for cell in cells
    }


def main() -> int:
    args = parse_args()
    inst_id = args.inst_id

    state_path = state_file_path_for(inst_id)

    config = load_grid_state(state_path)
    if config is None:
        print(f"❌ {state_path} absent ou invalide -- aucune grille persistée à initialiser.")
        return 1

    existing_memory = load_cell_memory(state_path)
    if existing_memory and not args.force:
        print("CELL_MEMORY_ALREADY_EXISTS")
        print(f"  {len(existing_memory)} cellule(s) déjà présente(s) dans {state_path}.")
        print("  Aucune modification effectuée. Utiliser --force pour écraser explicitement.")
        return 1

    adapter = build_adapter("LIVE")
    fresh_memory = build_fresh_cell_memory(adapter, config)

    save_cell_memory(fresh_memory, state_path)

    print(f"✅ cell_memory initialisée : {len(fresh_memory)} cellule(s), écrite dans {state_path}.")
    for gi in sorted(fresh_memory.keys()):
        entry = fresh_memory[gi]
        print(f"  gi={gi:<12} economic_state={entry.economic_state.name:<10} target_qty={entry.target_qty}")

    print("\n========== SAFETY ==========")
    print("Aucun appel list_fills / list_open_orders / get_order effectué.")
    print("Aucun ordre placé ni annulé. active_grid_state_*.json : seule la clé cell_memory modifiée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
