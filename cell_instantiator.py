"""
CD-003 — CellInstantiator

Service : instancie les n cellules de la grille initiale.
Responsabilité unique : associer à chaque niveau du treillis (distinct de P0)
le niveau gi et l'état initial déterminé par sa position stricte par rapport
à P0.

Ce module est l'unique traduction de CD-003.
Il ne connaît jamais : les équations géométriques, Gul, Gll, nu, nl, h%,
F0, Gv, la position de P0 dans le treillis (nl), l'exchange, le réseau,
l'horloge, le système de fichiers, le mécanisme de réconciliation, ni le
destin des cellules produites.

Dépendance interne autorisée : cell.py uniquement (Cell et CellState).
Aucune dépendance vers trellis_calculator.py.
"""

from __future__ import annotations

from cell import Cell, CellState


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CellInstantiator:
    """
    Service : instancie les n cellules à partir du treillis et de P0.

    Catégorie : Service (SA-001 §2 et §10).
    Aucun attribut persistant entre deux appels.
    Aucune identité propre.

    Référence : CD-003.
    """

    def instantiate(
        self,
        trellis: tuple[float, ...],
        P0: float,
    ) -> tuple[Cell, ...]:
        """
        Produit la collection de n = len(trellis) - 1 cellules.

        Paramètres
        ----------
        trellis : tuple[float, ...]
            Treillis immuable de n+1 niveaux (sortie de CD-002).
            Contient P0 comme niveau-frontière (CD-002 I3, RN-103 §3).
            Ce composant s'appuie sur les garanties de TrellisCalculator
            sans les revérifier (CD-003 §4.1).
        P0 : float
            Niveau-frontière — scalaire naturalisé transmis séparément
            (SA-001 §9, CD-003 §4.2).
            Ce composant ne recherche pas P0 dans le treillis et ne
            connaît pas sa position (nl).
            P0 n'est pas conservé après l'appel.

        Retourne
        --------
        tuple[Cell, ...]
            Collection immuable de n = len(trellis) - 1 cellules :
            - gi > P0 → CellState.SPOT_HELD (RN-100, Fig. 3(a))
            - gi < P0 → CellState.CASH_HELD (RN-100, Fig. 3(a))
            - gi == P0 → niveau-frontière, ignoré (RN-103 §3, §6 ;
              CD-003 §6 : P0 appartient au treillis mais n'est adressé
              par aucune cellule)

        Note sur gi == P0
        -----------------
        Le niveau-frontière P0 est toujours présent dans le treillis
        (CD-002 I3). Lors du parcours, il est ignoré : P0 n'est pas un
        cas métier — il n'a pas d'état et ne peut pas être adressé par
        une cellule (RN-103 §6). Ce n'est pas une violation de CD-002.
        """
        cells: list[Cell] = []

        for gi in trellis:
            if gi > P0:
                cells.append(Cell(gi, CellState.SPOT_HELD))
            elif gi < P0:
                cells.append(Cell(gi, CellState.CASH_HELD))
            # else gi == P0 : niveau-frontière, ignoré sans erreur
            # (RN-103 §3, §6 ; CD-003 §6)

        # P0 n'est plus référencé après ce point.
        return tuple(cells)
