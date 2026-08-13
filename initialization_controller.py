"""Contrôleur d'initialisation patrimoniale — achat initial S0 du PDF.

Ce composant orchestre la séquence décrite au §3.1.1 du PDF source :
achat initial de spot (S0) avant activation de la grille, lorsque le
patrimoine réel du compte ne couvre pas encore les niveaux SELL au-dessus
de P0.

Frontière architecturale stricte :
  grid_preflight.py       -> OkxPreflightSource -> READ ONLY  (inchangé)
  InitializationController -> OkxSpotAdapter    -> READ + WRITE

Ce module ne recalcule jamais Gv/S0 : la quantité à acheter provient
exclusivement de report.required_spot (grid_preflight.run_preflight),
jamais d'un calcul parallèle.

L'identifiant d'ordre déterministe (client_order_id) dépend uniquement de
l'intention économique d'initialisation — les neuf champs de
PreflightConfig, plus les règles de quantification de l'instrument
(tick_size, lot_size) — jamais des soldes réels. Les soldes sont l'état
observé que l'achat cherche à modifier ; les inclure casserait
l'idempotence (un achat partiel changerait l'ID au prochain appel,
empêchant de retrouver l'ordre en cours). Aucun timestamp, aucun
compteur local, aucune persistance locale : la seule source de vérité
est l'état réel de l'exchange, relu à chaque appel.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

from grid_preflight import OkxPreflightSource, PreflightConfig, PreflightError, PreflightReport, run_preflight
from okx_spot_adapter import OkxApiError, OrderSnapshot


class InitState(Enum):
    IDLE = auto()
    NEEDS_SPOT = auto()
    BUYING = auto()
    PARTIALLY_FILLED = auto()
    SPOT_READY = auto()
    GRID_PLACING = auto()
    ACTIVE = auto()
    ERROR = auto()


_LIVE_ORDER_STATES = {"live", "partially_filled"}


class InitializationError(RuntimeError):
    """L'initialisation patrimoniale ne peut pas se poursuivre en sécurité."""


@dataclass(frozen=True)
class InitializationResult:
    """État final d'un appel à InitializationController.run()."""
    state: InitState
    report: PreflightReport | None
    order: OrderSnapshot | None
    detail: str | None


def derive_client_order_id(config: PreflightConfig, tick_size: Decimal, lot_size: Decimal) -> str:
    """Identifiant déterministe de l'achat initial pour cette configuration.

    Même configuration économique (les neuf champs de PreflightConfig) et
    mêmes règles de quantification de l'instrument (tick_size, lot_size)
    => même identifiant, à toute exécution, sans état local. Les soldes
    réels (balances) sont délibérément exclus : voir docstring de module.

    Compatible OKX : alphanumérique, débute par une lettre, <= 32
    caractères ("INITBUY" + 24 caractères hex = 31).
    """
    payload = "|".join([
        config.inst_id, str(config.gul), str(config.gll), str(config.nu), str(config.nl),
        str(config.p0), config.geometry.name, str(config.spacing_h_pct), str(config.alpha),
        str(tick_size), str(lot_size),
    ])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"INITBUY{digest}"


class InitializationController:
    """Orchestre l'achat initial de spot, idempotent, jusqu'à SPOT_READY.

    Ne place jamais d'ordre de grille (POST-ONLY) : s'arrête à SPOT_READY.
    Le placement des ordres de grille (GRID_PLACING -> ACTIVE) est
    entièrement délégué à l'appelant, qui reste seul responsable
    d'utiliser report.orders (déjà projeté par grid_preflight/CD-004).
    """

    def run(self, adapter, config: PreflightConfig) -> InitializationResult:
        """Exécute la séquence jusqu'à SPOT_READY (ou ERROR).

        `adapter` doit exposer les lectures de OkxPreflightSource ET les
        écritures place_market_buy / get_order. Un objet typé strictement
        OkxPreflightSource ne peut structurellement pas être passé ici
        pour la partie écriture — c'est la frontière de sécurité (voir
        docstring de module).
        """
        report = self._preflight(adapter, config)
        if report is None:
            return InitializationResult(InitState.ERROR, None, None, "preflight a échoué")

        if report.spot_covered:
            return InitializationResult(InitState.SPOT_READY, report, None, None)

        derived_id = derive_client_order_id(config, report.instrument.tick_size, report.instrument.lot_size)

        existing = self._find_existing_order(adapter, config.inst_id, derived_id)
        if existing is not None:
            return self._handle_existing_order(adapter, config, existing, derived_id)

        return self._buy_deficit(adapter, config, report, derived_id)

    # -------------------------------------------------------------------
    # Étapes internes
    # -------------------------------------------------------------------

    def _preflight(self, adapter, config: PreflightConfig) -> PreflightReport | None:
        try:
            return run_preflight(adapter, config)
        except (PreflightError, OkxApiError):
            return None

    def _find_existing_order(self, adapter, inst_id: str, derived_id: str) -> OrderSnapshot | None:
        """Recherche un achat initial déjà tenté sous cet identifiant déterministe.

        Aucun ordre trouvé (data vide, ou item en erreur) => OkxApiError,
        capturée uniformément : les deux cas signifient "aucun ordre
        valide n'existe sous cet ID", sans distinction nécessaire.
        """
        try:
            return adapter.get_order(inst_id, client_order_id=derived_id)
        except OkxApiError:
            return None

    def _handle_existing_order(self, adapter, config: PreflightConfig, order: OrderSnapshot, derived_id: str) -> InitializationResult:
        if order.state == "filled":
            report = self._preflight(adapter, config)
            if report is None:
                return InitializationResult(InitState.ERROR, None, order, "preflight de confirmation a échoué")
            if report.spot_covered:
                return InitializationResult(InitState.SPOT_READY, report, order, None)
            return InitializationResult(
                InitState.ERROR, report, order,
                "achat initial FILLED mais spot_covered=False après relecture — incohérence patrimoniale",
            )
        if order.state == "partially_filled":
            return InitializationResult(InitState.PARTIALLY_FILLED, None, order, None)
        if order.state == "live":
            return InitializationResult(InitState.BUYING, None, order, None)
        # canceled / expired / autre état terminal non rempli : reconsulter
        # les soldes réels avant toute nouvelle décision (jamais racheter
        # aveuglément required_spot entier).
        report = self._preflight(adapter, config)
        if report is None:
            return InitializationResult(InitState.ERROR, None, order, "preflight après ordre disparu a échoué")
        if report.spot_covered:
            return InitializationResult(InitState.SPOT_READY, report, order, None)
        return self._buy_deficit(adapter, config, report, derived_id)

    def _buy_deficit(self, adapter, config: PreflightConfig, report: PreflightReport, derived_id: str) -> InitializationResult:
        deficit = report.required_spot - report.balances.base_available
        if deficit <= 0:
            return InitializationResult(InitState.SPOT_READY, report, None, None)

        placement = adapter.place_market_buy(config.inst_id, deficit, derived_id)
        if not placement.accepted:
            return InitializationResult(
                InitState.ERROR, report, None,
                f"achat initial rejeté : {placement.rejection_code} {placement.rejection_message}",
            )
        return InitializationResult(InitState.BUYING, report, None, None)
