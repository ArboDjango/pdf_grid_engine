"""
cell.py

Implémentation de CD-001 -- Component Definition, Cell (version finale
validée). Traduction stricte du contrat, rien de plus : aucune méthode
sans un événement du §5 pour la justifier, aucune validation au-delà
des invariants explicitement listés au §4, aucune dépendance vers un
autre composant de ce dépôt.

Contrat (CD-001) :
  §1  -- Cell représente une position de prix fixe et son état
         courant. Elle ne décide jamais elle-même de son état
         initial -- celui-ci lui est fourni à la construction, par un
         composant qui n'est pas Cell.
  §2  -- Justification : RN-100 §1 (nature d'une cellule), §3 (deux
         états, jamais un troisième), §7 (bascule d'état), §12
         (invariants). SA-001 §1 (une entité se définit par ce qui
         l'individualise), §7 (Gv n'individualise rien, exclu).
  §3  -- Identité : gi et state, rien d'autre. Tous deux des
         attributs publics -- aucune méthode d'accès dédiée n'est
         nécessaire ni autorisée.
  §4  -- Invariants garantis, exactement deux : gi immuable après
         construction ; toujours exactement un état. Rien d'autre
         n'est un invariant de ce composant (en particulier, la
         positivité de gi n'est pas listée ici et n'est donc pas
         validée).
  §5  -- Événements reçus, exactement trois : construction (état
         donné directement, sans référence à P0), signal SELL
         complété, signal BUY complété. Aucune autre méthode
         publique n'existe.
  §6  -- Machine à états : SPOT_HELD <-> CASH_HELD, transitions
         symétriques, aucune autre.
  §7  -- Ne connaît jamais : P0, l'exchange, les frais, la glissade,
         le prix réellement exécuté, le mécanisme de persistance, le
         registre, le patrimoine agrégé, Gv, le prix de marché
         courant, le PnL, les autres cellules.
  §8  -- Seule elle-même a le droit de modifier son état, en réaction
         aux événements du §5 -- jamais par écriture directe externe.
"""

from __future__ import annotations

from enum import Enum


class CellState(Enum):
    """Les deux seuls états possibles d'une cellule (CD-001 §3, RN-100 §3)."""
    SPOT_HELD = "SPOT_HELD"
    CASH_HELD = "CASH_HELD"


class CellStateError(Exception):
    """
    Levée quand un signal d'exécution est reçu depuis un état qui ne
    correspond pas (CD-001 §6) -- une transition que le contrat ne
    décrit jamais.
    """


class Cell:
    """
    Une cellule (CD-001) : un prix nominal fixe, un état courant.
    Rien de plus -- voir la docstring de module pour ce qu'elle ne
    connaît jamais (CD-001 §7).

    Attributes:
        gi: Le prix nominal de la cellule. Fixé une seule fois à la
            construction, jamais modifié ensuite (CD-001 §4).
        state: L'état courant (SPOT_HELD ou CASH_HELD).
    """

    def __init__(self, gi: float, state: CellState):
        self.gi = gi
        self.state = state

    def on_sell_completed(self) -> None:
        """
        Réagit au signal binaire (Canal B, RN-101 §2) informant que
        l'ordre de vente en attente à cette cellule a été
        intégralement exécuté -- bascule vers CASH_HELD (CD-001 §6).

        La cellule ne "vend" jamais elle-même -- elle reçoit un fait
        accompli, et réagit en conséquence (CD-001 §7 : aucune
        connaissance de l'exchange ou de l'exécution elle-même).

        Raises:
            CellStateError: si la cellule n'est pas actuellement en
                SPOT_HELD.
        """
        if self.state is not CellState.SPOT_HELD:
            raise CellStateError(
                f"Cellule à gi={self.gi} : signal de vente complétée reçu, "
                f"mais état actuel = {self.state.value} (attendu SPOT_HELD)"
            )
        self.state = CellState.CASH_HELD

    def on_buy_completed(self) -> None:
        """
        Réagit au signal binaire (Canal B, RN-101 §2) informant que
        l'ordre d'achat en attente à cette cellule a été intégralement
        exécuté -- bascule vers SPOT_HELD (CD-001 §6).

        Raises:
            CellStateError: si la cellule n'est pas actuellement en
                CASH_HELD.
        """
        if self.state is not CellState.CASH_HELD:
            raise CellStateError(
                f"Cellule à gi={self.gi} : signal d'achat complété reçu, "
                f"mais état actuel = {self.state.value} (attendu CASH_HELD)"
            )
        self.state = CellState.SPOT_HELD
