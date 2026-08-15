"""Identité déterministe des occurrences d'ordres d'une cellule.

La cellule économique est identifiée par son niveau gi.
Chaque occurrence d'ordre possède une identité distincte.

Occurrence racine :
    H(configuration, side, gi, ROOT)

Occurrence suivante :
    H(configuration, side, gi, predecessor_client_order_id)

Cette brique est pure :
- aucun accès réseau ;
- aucune persistance ;
- aucune connaissance des soldes ;
- q volontairement exclu de l'identité.
"""

from __future__ import annotations

import hashlib

from grid_activation_controller import derive_grid_order_id
from grid_preflight import PreflightConfig


ROOT = "ROOT"


def derive_grid_occurrence_id(
    config: PreflightConfig,
    tick_size,
    lot_size,
    side: str,
    price: float,
    predecessor_id: str,
) -> str:
    """Dérive l'identité déterministe d'une occurrence d'ordre.

    Le premier ordre d'une cellule utilise predecessor_id=ROOT.
    Chaque ordre suivant utilise comme prédécesseur le client_order_id
    de l'occurrence précédente.

    q est volontairement exclu : il dépend du capital instantané et ne
    doit jamais modifier l'identité économique d'une occurrence.
    """
    payload = "|".join([
        config.inst_id,
        str(config.gul),
        str(config.gll),
        str(config.nu),
        str(config.nl),
        str(config.p0),
        config.geometry.name,
        str(config.spacing_h_pct),
        str(config.alpha),
        str(tick_size),
        str(lot_size),
        side,
        str(price),
        predecessor_id,
    ])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"GRID{digest}"


def derive_root_order_id(
    config: PreflightConfig,
    tick_size,
    lot_size,
    side: str,
    price: float,
) -> str:
    """Identité de la première occurrence.

    Elle reste exactement compatible avec l'ancien système :
    le ROOT est l'ancien derive_grid_order_id().
    """
    return derive_grid_order_id(
        config,
        tick_size,
        lot_size,
        side,
        price,
    )


def derive_next_order_id(
    config: PreflightConfig,
    tick_size,
    lot_size,
    side: str,
    price: float,
    predecessor_id: str,
) -> str:
    """Identité de l'occurrence suivant un ordre déjà exécuté."""
    if not predecessor_id:
        raise ValueError("predecessor_id ne peut pas être vide")

    return derive_grid_occurrence_id(
        config,
        tick_size,
        lot_size,
        side,
        price,
        predecessor_id,
    )
