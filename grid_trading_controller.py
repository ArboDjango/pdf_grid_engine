"""Orchestrateur — chaînage feeType -> InitializationController -> GridActivationController.

Ce module ne contient AUCUNE logique économique : il ne recalcule jamais
q, required_spot, ou le treillis, et ne modifie jamais PreflightReport.
Sa seule responsabilité est de séquencer, dans l'ordre correct, les trois
lectures/décisions déjà implémentées ailleurs :

    get_account_config()  -> feeType == "1" ?
            |
    InitializationController.run(adapter, config)
            |
    état == SPOT_READY ?
            |
    GridActivationController.run(adapter, initialization_result.report)

Garantie de sécurité, assurée par construction (pas par convention) :
aucun ordre de grille ne peut être envoyé si feeType != "1" ou si
l'initialisation n'atteint pas SPOT_READY — dans les deux cas, la
fonction retourne avant même d'importer/référencer GridActivationController
dans le chemin d'exécution.

Aucun appel à set-fee-type n'est jamais effectué par ce module (ni par
aucun des composants qu'il orchestre).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from grid_activation_controller import GridActivationController, GridActivationResult
from grid_preflight import PreflightConfig
from initialization_controller import InitializationController, InitializationResult, InitState
from okx_spot_adapter import OkxApiError


class GridTradingState(Enum):
    FEE_TYPE_REJECTED = auto()
    INITIALIZATION_INCOMPLETE = auto()
    ACTIVATED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class GridTradingResult:
    """Résultat final de GridTradingController.run().

    Au plus un des trois champs *_result est renseigné, selon l'étape à
    laquelle la séquence s'est arrêtée — jamais une reconstruction ou une
    transformation de ces résultats, toujours les objets exacts retournés
    par les composants sous-jacents.
    """
    state: GridTradingState
    fee_type: str | None
    initialization_result: InitializationResult | None
    activation_result: GridActivationResult | None
    detail: str | None


class GridTradingController:
    """Séquence feeType -> InitializationController -> GridActivationController.

    Ne recalcule rien, ne décide rien économiquement. Chaque étape est
    déléguée intégralement à son composant existant ; ce contrôleur ne
    fait que lire l'état retourné et décider s'il poursuit ou s'arrête.
    """

    def run(self, adapter, config: PreflightConfig) -> GridTradingResult:
        # --- Étape 1 : précondition feeType, avant tout appel de trading ---
        try:
            account_config = adapter.get_account_config()
        except OkxApiError as error:
            return GridTradingResult(
                GridTradingState.ERROR, None, None, None,
                f"lecture de la configuration du compte a échoué : {error}",
            )
        except Exception as error:  # panne réelle de transport, jamais silencieuse
            return GridTradingResult(
                GridTradingState.ERROR, None, None, None,
                f"lecture de la configuration du compte : panne réseau réelle : {error}",
            )

        if account_config.fee_type != "1":
            return GridTradingResult(
                GridTradingState.FEE_TYPE_REJECTED, account_config.fee_type, None, None,
                "feeType doit valoir \"1\" (frais Spot BUY en devise de cotation) ; "
                "configuration manuelle requise via l'interface OKX ou set-fee-type — "
                "ce contrôleur ne le fait jamais automatiquement",
            )

        # --- Étape 2 : initialisation patrimoniale ---
        try:
            initialization_result = InitializationController().run(adapter, config)
        except Exception as error:  # panne réelle, jamais silencieuse
            return GridTradingResult(
                GridTradingState.ERROR, account_config.fee_type, None, None,
                f"initialisation a échoué : {error}",
            )

        if initialization_result.state != InitState.SPOT_READY:
            return GridTradingResult(
                GridTradingState.INITIALIZATION_INCOMPLETE, account_config.fee_type,
                initialization_result, None,
                f"initialisation non terminée, état={initialization_result.state.name} "
                "— aucun ordre de grille ne sera envoyé",
            )

        # --- Étape 3 : activation de la grille, avec le rapport EXACT de
        # l'initialisation (jamais un rapport de preflight antérieur) ---
        try:
            activation_result = GridActivationController().run(adapter, initialization_result.report)
        except Exception as error:  # panne réelle, jamais silencieuse
            return GridTradingResult(
                GridTradingState.ERROR, account_config.fee_type, initialization_result, None,
                f"activation de la grille a échoué : {error}",
            )

        return GridTradingResult(
            GridTradingState.ACTIVATED, account_config.fee_type, initialization_result, activation_result, None,
        )
