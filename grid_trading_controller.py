"""Orchestrateur — chaînage feeType -> InitializationController -> GridActivationController.

Ce module ne contient AUCUNE logique économique : il ne recalcule jamais
q, required_spot, ou le treillis, et ne modifie jamais PreflightReport.
Sa seule responsabilité est de séquencer, dans l'ordre correct, les
lectures/décisions déjà implémentées ailleurs :

    get_account_config()  -> feeType == "1" ?
            |
    InitializationController.run(adapter, config)
            |
    état == SPOT_READY ?
            |
    k_buy/k_sell fournis ?
            |
        NON -> GridActivationController.run(adapter, report COMPLET)
               (comportement historique, strictement inchangé)
            |
        OUI -> plan_exposure(adapter, report, k_buy, k_sell)  [PUR, aucune écriture]
               (grid_order_orchestrator.py, réutilisé, jamais dupliqué)
                    |
               plan.decision.to_cancel non vide ?
                    |
                OUI -> GridTradingState.EXPOSURE_CONFLICT, AUCUNE écriture,
                        AUCUNE annulation automatique -- des ordres réels
                        semblent appartenir à une autre configuration
                        (garde-fou contre les ordres orphelins, cf. analyse
                        dédiée : le fichier de persistance local peut être
                        absent alors que des ordres réels existent déjà)
                    |
                NON -> GridActivationController.run(adapter, plan.narrowed_report,
                        order_ids=plan.order_ids)  -- k_buy+k_sell ordres
                        SEULEMENT, jamais nu+nl

Garantie de sécurité, assurée par construction (pas par convention) :
aucun ordre de grille ne peut être envoyé si feeType != "1" ou si
l'initialisation n'atteint pas SPOT_READY — dans les deux cas, la
fonction retourne avant même d'importer/référencer GridActivationController
dans le chemin d'exécution.

Garantie de sécurité supplémentaire (ce chantier) : si k_buy/k_sell sont
fournis et qu'au moins un ordre réel devrait être annulé pour respecter
la fenêtre d'exposition, AUCUNE annulation n'est jamais tentée à
l'activation initiale -- la fonction retourne EXPOSURE_CONFLICT et
s'arrête, laissant la décision à l'opérateur.

k_buy/k_sell restent optionnels (défaut None) : tout appelant qui ne les
fournit pas conserve exactement le comportement historique (activation
de la grille théorique complète, nu+nl ordres) -- run_grid_trading.py
n'est jamais modifié et continue de bénéficier de ce comportement inchangé.

Aucun appel à set-fee-type n'est jamais effectué par ce module (ni par
aucun des composants qu'il orchestre).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from grid_activation_controller import GridActivationController, GridActivationResult, GridActivationState
from grid_order_orchestrator import plan_exposure
from grid_preflight import PreflightConfig
from initialization_controller import InitializationController, InitializationResult, InitState
from okx_spot_adapter import OkxApiError


class GridTradingState(Enum):
    FEE_TYPE_REJECTED = auto()
    INITIALIZATION_INCOMPLETE = auto()
    EXPOSURE_CONFLICT = auto()
    PARTIALLY_EXPOSED = auto()
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

    def run(self, adapter, config: PreflightConfig, k_buy: int | None = None, k_sell: int | None = None) -> GridTradingResult:
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
        if k_buy is None or k_sell is None:
            # Comportement historique, strictement inchangé : grille
            # théorique complète (nu+nl), aucune fenêtre d'exposition.
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

        # k_buy/k_sell fournis : appliquer la fenêtre d'exposition AVANT
        # toute activation -- plan_exposure() est PURE (aucune écriture),
        # réutilisée telle quelle depuis grid_order_orchestrator.py, jamais
        # dupliquée.
        try:
            plan = plan_exposure(adapter, initialization_result.report, k_buy, k_sell)
        except Exception as error:  # panne réelle, jamais silencieuse
            return GridTradingResult(
                GridTradingState.ERROR, account_config.fee_type, initialization_result, None,
                f"planification de l'exposition a échoué : {error}",
            )

        if plan.decision.to_cancel:
            # GARDE-FOU CONTRE LES ORDRES ORPHELINS : des ordres réels
            # existent et ne correspondent à aucun niveau exposé par cette
            # configuration -- possible grille déjà active sous une autre
            # configuration (le fichier de persistance local peut être
            # absent ou invalide sans que cela signifie qu'aucun ordre
            # réel n'existe). AUCUNE annulation automatique n'est jamais
            # tentée ici -- décision remontée explicitement à l'opérateur.
            return GridTradingResult(
                GridTradingState.EXPOSURE_CONFLICT, account_config.fee_type, initialization_result, None,
                f"activation refusée : {len(plan.decision.to_cancel)} ordre(s) existant(s) "
                f"devrai(en)t être annulé(s) pour respecter k_buy={k_buy}/k_sell={k_sell} "
                f"({plan.decision.to_cancel}) -- probable conflit avec une grille déjà "
                f"active sous une autre configuration. Aucune annulation automatique "
                f"n'est jamais tentée à l'activation initiale.",
            )

        try:
            activation_result = GridActivationController().run(
                adapter, plan.narrowed_report, order_ids=plan.order_ids,
            )
        except Exception as error:  # panne réelle, jamais silencieuse
            return GridTradingResult(
                GridTradingState.ERROR, account_config.fee_type, initialization_result, None,
                f"activation de la grille a échoué : {error}",
            )

        # Correctif sémantique ACTIVATED : ni le filtre de liquidité, ni le
        # filtre price-limit, ni la géométrie, ni q, ni P0 ne sont modifiés
        # ici -- ce bloc observe uniquement deux faits déjà produits
        # ailleurs, jamais recalculés :
        #   1. la fenêtre RÉELLEMENT demandée par select_exposed_orders
        #      (plan.requested_to_materialize, AVANT tout filtre) ;
        #   2. l'état réel retourné par GridActivationController
        #      (activation_result.state), jusqu'ici jamais consulté.
        if len(plan.decision.to_materialize) < len(plan.requested_to_materialize):
            missing = tuple(
                instr for instr in plan.requested_to_materialize
                if instr not in plan.decision.to_materialize
            )
            return GridTradingResult(
                GridTradingState.PARTIALLY_EXPOSED, account_config.fee_type, initialization_result, activation_result,
                f"fenêtre d'exposition demandée (k_buy={k_buy}/k_sell={k_sell}) non entièrement "
                f"matérialisée : {len(missing)} niveau(x) écarté(s) par un filtre de matérialisation "
                f"(liquidité et/ou price-limit) : {missing} -- jamais annulé, jamais recalculé, "
                f"retentera au cycle suivant.",
            )

        if activation_result.state != GridActivationState.ACTIVE:
            return GridTradingResult(
                GridTradingState.PARTIALLY_EXPOSED, account_config.fee_type, initialization_result, activation_result,
                f"activation partielle au sens de GridActivationController "
                f"(état={activation_result.state.name}) : {activation_result.detail}",
            )

        return GridTradingResult(
            GridTradingState.ACTIVATED, account_config.fee_type, initialization_result, activation_result, None,
        )
