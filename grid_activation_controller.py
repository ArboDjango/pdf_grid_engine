"""Contrôleur d'activation de grille — placement des ordres POST-ONLY.

Ce module transforme report.orders (PreflightReport, produit par
grid_preflight.run_preflight) en ordres réels sur l'exchange, de façon
idempotente : un redémarrage avec la même configuration ne replace jamais
un niveau déjà ouvert ou déjà exécuté.

Garde-fou intrinsèque : run() refuse toute activation si
report.spot_covered est False, avant tout appel réseau — cette garantie
ne dépend d'aucun appelant externe (elle vaut même en cas d'appel direct
à GridActivationController, hors de GridTradingController).

Contrat explicite : activation d'une configuration donnée uniquement.
GridActivationController ne détecte ni ne gère un changement de
configuration entre deux appels (P0, bornes, nu/nl, alpha, tick_size,
lot_size) : toute variation produit un jeu d'identifiants entièrement
différent (voir derive_grid_order_id), et ce module tente simplement
d'activer la configuration reçue, sans connaître ni fermer une éventuelle
grille antérieure sous une autre configuration. Empêcher l'activation
simultanée de deux configurations différentes est la responsabilité d'un
futur orchestrateur, non implémenté ici.

Frontière stricte, symétrique à InitializationController :
  InitializationController  -> achat initial S0 uniquement, jamais la grille
  GridActivationController  -> placement de la grille uniquement, jamais
                                 l'achat initial ni la vérification de
                                 précondition (feeType)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

from grid_preflight import PreflightConfig, PreflightReport
from grid_order_projection import OrderInstruction
from okx_spot_adapter import OkxApiError, OrderSnapshot


def derive_grid_order_id(
    config: PreflightConfig,
    tick_size: Decimal,
    lot_size: Decimal,
    side: str,
    price: float,
) -> str:
    """Identifiant déterministe d'un ordre de grille pour cette configuration.

    Même matériau que initialization_controller.derive_client_order_id
    (les neuf champs de PreflightConfig, plus tick_size et lot_size), avec
    side et price ajoutés pour distinguer chaque emplacement de la grille.

    q (la quantité) est délibérément EXCLU : il dépend de total_capital,
    donc des soldes réels au moment du calcul de run_preflight. L'inclure
    romprait l'idempotence à la moindre fluctuation patrimoniale entre
    deux appels — un identifiant de niveau de grille désigne un
    emplacement (side, price) dans une configuration donnée, jamais une
    quantité instantanée.

    Préfixe "GRID" distinct de "INITBUY" (initialization_controller) :
    aucune collision possible entre un identifiant d'achat initial et un
    identifiant d'ordre de grille pour la même configuration.

    Compatible OKX : alphanumérique, débute par une lettre, <= 32
    caractères ("GRID" + 24 caractères hex = 28, marge conservée par
    rapport à la limite de 32 imposée par OKX).
    """
    payload = "|".join([
        config.inst_id, str(config.gul), str(config.gll), str(config.nu), str(config.nl),
        str(config.p0), config.geometry.name, str(config.spacing_h_pct), str(config.alpha),
        str(tick_size), str(lot_size), side, str(price),
    ])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"GRID{digest}"


class GridActivationState(Enum):
    ACTIVE = auto()
    PARTIAL = auto()
    ERROR = auto()


@dataclass(frozen=True)
class GridActivationResult:
    """Résultat d'un appel à GridActivationController.run()."""
    state: GridActivationState
    already_open: tuple[OrderInstruction, ...]
    already_filled: tuple[OrderInstruction, ...]
    placed: tuple[OrderInstruction, ...]
    failed: tuple[tuple[OrderInstruction, str | None, str | None], ...]
    detail: str | None


class GridActivationController:
    """Place les ordres POST-ONLY de report.orders, idempotent.

    Contrat : activation d'une configuration donnée uniquement. Ne détecte
    ni ne gère un changement de configuration entre deux appels — voir
    docstring de module.
    """

    def run(self, adapter, report: PreflightReport) -> GridActivationResult:
        """Sonde, vérifie, puis place les niveaux manquants.

        Phase 1 : un seul appel list_open_orders(inst_id) — niveaux déjà
        ouverts (live/partially_filled), détectés en masse.
        Phase 2 : pour chaque niveau non trouvé en phase 1, un appel ciblé
        get_order(client_order_id=derived_id) — seule façon de retrouver
        un niveau déjà FILLED (disparu du carnet des ordres ouverts).
        Phase 3 : place_post_only_limit pour chaque niveau encore absent
        après les phases 1 et 2 — démarre uniquement une fois le sondage
        complet terminé pour tous les niveaux (jamais de placement avant
        la fin du sondage).

        Toute exception autre qu'OkxApiError (panne réelle de transport)
        pendant l'une quelconque des trois phases interrompt immédiatement
        le traitement des niveaux restants et produit ERROR — jamais une
        tentative à l'aveugle, jamais une interprétation silencieuse comme
        "absent". L'état déjà accumulé (already_open, already_filled,
        placed, failed) est préservé dans le résultat pour diagnostic, y
        compris en cas d'échec en cours de boucle de placement. Un rejet
        explicite au placement (phase 3, accepted=False) ne déclenche
        jamais de nouvelle tentative avec un autre type d'ordre : il est
        comptabilisé dans failed, et l'état final est PARTIAL.

        Garde-fou intrinsèque : si report.spot_covered est False, aucun
        appel réseau n'est effectué (ni sondage, ni vérification, ni
        placement) — ERROR immédiat. Cette vérification ne dépend
        d'aucun appelant externe (ex. GridTradingController) : elle est
        vraie quel que soit le point d'entrée utilisé pour appeler run().
        """
        if not report.spot_covered:
            return GridActivationResult(
                GridActivationState.ERROR, (), (), (), (),
                "activation refusée : report.spot_covered=False — "
                "le spot requis pour les niveaux SELL n'est pas disponible",
            )

        config = report.config
        instrument = report.instrument

        try:
            open_orders = adapter.list_open_orders(config.inst_id)
        except OkxApiError as error:
            return GridActivationResult(
                GridActivationState.ERROR, (), (), (), (),
                f"sondage des ordres ouverts a échoué : {error}",
            )
        except Exception as error:  # panne réelle de transport, jamais silencieuse
            return GridActivationResult(
                GridActivationState.ERROR, (), (), (), (),
                f"sondage des ordres ouverts : panne réseau réelle : {error}",
            )
        open_ids = {order.client_order_id for order in open_orders}

        already_open: list[OrderInstruction] = []
        after_phase1: list[tuple[OrderInstruction, str]] = []  # (instruction, derived_id)
        for instruction in report.orders:
            side, price, quantity = instruction
            derived_id = derive_grid_order_id(config, instrument.tick_size, instrument.lot_size, side, price)
            if derived_id in open_ids:
                already_open.append(instruction)
            else:
                after_phase1.append((instruction, derived_id))

        already_filled: list[OrderInstruction] = []
        to_place: list[tuple[OrderInstruction, str]] = []
        for instruction, derived_id in after_phase1:
            try:
                order = adapter.get_order(config.inst_id, client_order_id=derived_id)
            except OkxApiError:
                # Introuvable : ni ouvert (déjà écarté en phase 1), ni
                # jamais placé. Candidat à la mise en place (commit suivant).
                to_place.append((instruction, derived_id))
                continue
            except Exception as error:  # panne réelle de transport, jamais silencieuse
                return GridActivationResult(
                    GridActivationState.ERROR, tuple(already_open), tuple(already_filled), (), (),
                    f"vérification ciblée a échoué (niveau {derived_id}) : {error}",
                )

            if order.state == "filled":
                already_filled.append(instruction)
            elif order.state in ("live", "partially_filled"):
                # Non détecté en phase 1 (fenêtre de course possible) mais
                # bien présent et vivant : traité comme déjà ouvert.
                already_open.append(instruction)
            else:
                # État terminal non rempli (canceled/expired) : à replacer.
                to_place.append((instruction, derived_id))

        # Phase 3 — placement. Démarre uniquement une fois les phases 1 et 2
        # entièrement terminées pour TOUS les niveaux de report.orders
        # (contrainte de sécurité : aucun placement avant sondage complet).
        placed: list[OrderInstruction] = []
        failed: list[tuple[OrderInstruction, str | None, str | None]] = []
        for instruction, derived_id in to_place:
            side, price, quantity = instruction
            try:
                result = adapter.place_post_only_limit(
                    config.inst_id, side, Decimal(str(price)), Decimal(str(quantity)), derived_id,
                )
            except Exception as error:  # OkxApiError (erreur API réelle) ou panne de
                # transport : la boucle s'arrête immédiatement, sans tenter les
                # niveaux suivants. L'état accumulé jusqu'ici (already_open,
                # already_filled, placed, failed) est préservé pour diagnostic.
                return GridActivationResult(
                    GridActivationState.ERROR, tuple(already_open), tuple(already_filled), tuple(placed), tuple(failed),
                    f"placement a échoué (niveau {derived_id}) : {error}",
                )
            if result.accepted:
                placed.append(instruction)
            else:
                # Rejet explicite : jamais de nouvelle tentative avec un
                # autre type d'ordre, jamais de retry automatique.
                failed.append((instruction, result.rejection_code, result.rejection_message))

        if failed:
            return GridActivationResult(
                GridActivationState.PARTIAL, tuple(already_open), tuple(already_filled), tuple(placed), tuple(failed),
                f"{len(failed)} niveau(x) rejeté(s) au placement",
            )
        return GridActivationResult(
            GridActivationState.ACTIVE, tuple(already_open), tuple(already_filled), tuple(placed), (), None,
        )
