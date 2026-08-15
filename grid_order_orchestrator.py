"""GridOrderOrchestrator — séquence annulation puis matérialisation.

L'orchestrateur reconstruit l'état économique des cellules, applique la
politique d'exposition, puis délègue le placement à GridActivationController.

L'identité d'une nouvelle occurrence est dérivée à partir de la dernière
occurrence remplie de la cellule :

    ROOT -> occurrence suivante -> occurrence suivante -> ...

GridActivationController reste responsable du sondage et du placement réel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from cell_order_identity import derive_next_order_id, derive_root_order_id
from cell_state_reconstruction import reconstruct_cell_states
from grid_activation_controller import GridActivationController, GridActivationResult
from grid_order_exposure_controller import select_exposed_orders
from grid_preflight import PreflightReport


@dataclass(frozen=True)
class OrchestrationResult:
    """Résultat d'un cycle complet de réconciliation/exposition."""

    cancelled: tuple[str, ...]
    activation_result: GridActivationResult


def run_exposure_cycle(
    adapter,
    report: PreflightReport,
    k_buy: int,
    k_sell: int,
) -> OrchestrationResult:
    """Exécute un cycle : reconstruction, sélection, annulation, délégation.

    L'identité de chaque ordre matérialisé est déterminée à partir de la
    dernière occurrence remplie de sa cellule.

    Si aucune occurrence précédente n'existe :
        derive_root_order_id(...)

    Sinon :
        derive_next_order_id(..., predecessor_id)
    """

    inst_id = report.config.inst_id
    open_orders = adapter.list_open_orders(inst_id)

    # Reconstruction enrichie : contrairement à reconstruct_cells(), nous
    # conservons ici la dernière occurrence connue de chaque cellule.
    cell_states = reconstruct_cell_states(
        adapter,
        report,
        open_orders,
    )

    cells = tuple(item.cell for item in cell_states)

    decision = select_exposed_orders(
        report,
        cells,
        open_orders,
        k_buy,
        k_sell,
    )

    cancelled: list[str] = []

    for client_order_id in decision.to_cancel:
        adapter.cancel_order(
            inst_id,
            client_order_id=client_order_id,
        )
        cancelled.append(client_order_id)

    # ------------------------------------------------------------------
    # Identité des occurrences
    # ------------------------------------------------------------------
    #
    # La politique d'exposition nous donne uniquement :
    #
    #     (side, gi, q)
    #
    # On retrouve donc la CellReconstruction correspondant à (gi), puis
    # on dérive l'identité de la prochaine occurrence.
    #
    # Important : le predecessor est la DERNIÈRE occurrence remplie,
    # pas simplement le ROOT correspondant au niveau.
    # ------------------------------------------------------------------

    order_ids = {}

    for instruction in decision.to_materialize:
        side, price, quantity = instruction

        state = _find_cell_state(
            cell_states,
            price,
        )

        if state is None:
            # Sécurité : une instruction doit toujours correspondre à une
            # cellule reconstruite du même treillis.
            raise RuntimeError(
                f"Instruction sans cellule correspondante : "
                f"{instruction!r}"
            )

        if state.last_client_order_id is None:
            client_order_id = derive_root_order_id(
                report.config,
                report.instrument.tick_size,
                report.instrument.lot_size,
                side,
                price,
            )
        else:
            client_order_id = derive_next_order_id(
                report.config,
                report.instrument.tick_size,
                report.instrument.lot_size,
                side,
                price,
                state.last_client_order_id,
            )

        order_ids[instruction] = client_order_id

    narrowed_report = replace(
        report,
        orders=decision.to_materialize,
    )

    activation_result = GridActivationController().run(
        adapter,
        narrowed_report,
        order_ids=order_ids,
    )

    return OrchestrationResult(
        tuple(cancelled),
        activation_result,
    )


def _find_cell_state(cell_states, gi: float):
    """Retourne la reconstruction correspondant exactement au niveau gi."""

    for state in cell_states:
        if abs(float(state.cell.gi) - float(gi)) < 1e-12:
            return state

    return None
