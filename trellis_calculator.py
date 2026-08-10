"""
CD-002 — TrellisCalculator

Service: produit le treillis immuable de n+1 niveaux de prix.
Responsabilité unique : produire le treillis immuable de n+1 niveaux dont la
géométrie est conforme aux équations de construction (Éq. 1, 11–14) et dont
chaque paire de niveaux adjacents satisfait gi+1 − gi > h% × gi+1 (Éq. 16).
Tous les paramètres d'entrée sont des scalaires naturalisés (SA-001 §9).

Ce module est l'unique traduction de CD-002.
Il ne connaît jamais : Cell, les états SPOT_HELD/CASH_HELD, les grandeurs
patrimoniales, l'exchange, le réseau, l'horloge, le système de fichiers,
la manière dont P0 et h% ont été déterminés, ni le mécanisme de réconciliation
(RN-102).
"""

from __future__ import annotations

from enum import Enum, auto


# ---------------------------------------------------------------------------
# Énumération des géométries
# ---------------------------------------------------------------------------

class GridGeometry(Enum):
    """Les trois géométries de grille définies dans le PDF."""
    EQUIDISTANT = auto()  # Éq. 1, 11
    EQUIRATIO   = auto()  # Éq. 3, 12
    FLEXIBLE    = auto()  # Éq. 13–14


# ---------------------------------------------------------------------------
# Exception contractuelle
# ---------------------------------------------------------------------------

class TrellisParameterError(Exception):
    """
    Levée lorsqu'un paramètre d'entrée est invalide, ou lorsque le treillis
    calculé viole un invariant de CD-002 (I1–I6).

    Signale une erreur dans les entrées de l'appelant. Ce service ne modifie
    jamais les paramètres pour tenter de satisfaire une contrainte violée
    (CD-002 §6, I6).
    """


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TrellisCalculator:
    """
    Service : produit le treillis immuable de n+1 niveaux de prix.

    Catégorie : Service (SA-001 §2).
    Aucun attribut persistant. Aucune identité propre. Aucun état conservé
    entre deux appels.

    Référence : CD-002.
    """

    def calculate(
        self,
        Gul: float,
        Gll: float,
        nu: int,
        nl: int,
        P0: float,
        geometry: GridGeometry,
        h_pct: float,
    ) -> tuple[float, ...]:
        """
        Produit le treillis immuable.

        Tous les paramètres sont des scalaires naturalisés (SA-001 §9).
        Ce service ne détient aucune référence active vers le monde réel
        et ne peut rafraîchir aucune valeur.

        Paramètres
        ----------
        Gul : float
            Borne supérieure du treillis. Doit être > Gll.
        Gll : float
            Borne inférieure du treillis. Doit être > 0.
        nu : int
            Nombre de niveaux strictement supérieurs à P0. Doit être >= 1.
        nl : int
            Nombre de niveaux strictement inférieurs à P0. Doit être >= 1.
        P0 : float
            Niveau-frontière (RN-103 §3). Doit satisfaire Gll < P0 < Gul.
            Pour EQUIDISTANT et EQUIRATIO : reçu pour uniformité d'interface
            et affirmation de son rôle architectural ; sa cohérence est
            vérifiée via I3 (CD-002 §4.1).
            Pour FLEXIBLE : ancre explicite des deux progressions (Éq. 13–14).
        geometry : GridGeometry
            EQUIDISTANT (Éq. 1, 11), EQUIRATIO (Éq. 3, 12)
            ou FLEXIBLE (Éq. 13–14).
        h_pct : float
            Coefficient de contrainte d'espacement minimal pour Éq. 16.
            Scalaire naturalisé : la manière dont il a été déterminé est
            inconnue de ce service.

        Retourne
        --------
        tuple[float, ...]
            Tuple immuable et ordonné de n+1 = nu+nl+1 niveaux de prix,
            satisfaisant les invariants I1–I6 de CD-002.

        Lève
        ----
        TrellisParameterError
            Si une entrée est invalide, ou si le treillis calculé viole
            l'un des invariants I1–I6. Ce service ne modifie jamais les
            paramètres pour satisfaire une contrainte ; les entrées invalides
            sont rejetées telles quelles.
        """
        self._validate_inputs(Gul, Gll, nu, nl, P0, h_pct)

        if geometry is GridGeometry.EQUIDISTANT:
            levels = self._compute_equidistant(Gul, Gll, nu, nl)
        elif geometry is GridGeometry.EQUIRATIO:
            levels = self._compute_equiratio(Gul, Gll, nu, nl)
        elif geometry is GridGeometry.FLEXIBLE:
            levels = self._compute_flexible(Gul, Gll, nu, nl, P0)
        else:
            raise TrellisParameterError(f"Géométrie inconnue : {geometry!r}")

        # I5 est garantie par construction (application directe des Éq. 1, 3, 11–14).
        # Vérifiée dans les tests plutôt qu'à l'exécution pour éviter les faux
        # positifs liés aux erreurs d'arrondi en virgule flottante.
        self._verify_invariants(levels, Gul, Gll, nu, nl, P0, h_pct)

        return tuple(levels)  # immuable — CD-002 §5

    # -------------------------------------------------------------------------
    # Validation des entrées
    # -------------------------------------------------------------------------

    def _validate_inputs(
        self,
        Gul: float,
        Gll: float,
        nu: int,
        nl: int,
        P0: float,
        h_pct: float,
    ) -> None:
        if not isinstance(nu, int) or nu < 1:
            raise TrellisParameterError(
                f"nu doit être un entier >= 1, reçu : {nu!r}"
            )
        if not isinstance(nl, int) or nl < 1:
            raise TrellisParameterError(
                f"nl doit être un entier >= 1, reçu : {nl!r}"
            )
        if not isinstance(Gll, (int, float)) or not (Gll > 0):
            raise TrellisParameterError(
                f"Gll doit être un réel strictement positif, reçu : {Gll!r}"
            )
        if not isinstance(Gul, (int, float)) or not (Gul > Gll):
            raise TrellisParameterError(
                f"Gul doit être strictement supérieur à Gll ({Gll}), reçu : {Gul!r}"
            )
        if not isinstance(P0, (int, float)) or not (Gll < P0 < Gul):
            raise TrellisParameterError(
                f"P0 doit satisfaire Gll < P0 < Gul "
                f"(Gll={Gll}, Gul={Gul}), reçu : P0={P0!r}"
            )
        if not isinstance(h_pct, (int, float)) or not (h_pct > 0):
            raise TrellisParameterError(
                f"h_pct doit être un réel strictement positif, reçu : {h_pct!r}"
            )

    # -------------------------------------------------------------------------
    # Calcul géométrique
    # -------------------------------------------------------------------------

    def _compute_equidistant(
        self, Gul: float, Gll: float, nu: int, nl: int
    ) -> list[float]:
        """
        Produit n+1 niveaux équidistants de Gll à Gul.
        Éq. 1 : Gs = (Gul − Gll) / n
        Éq. 11 : gi = Gll + Gs × i,  i ∈ 0..n  (adapté pour n+1 niveaux)
        P0 n'est pas utilisé dans le calcul ; sa cohérence est vérifiée via I3.
        """
        n = nu + nl
        Gs = (Gul - Gll) / n                          # Éq. 1
        return [Gll + Gs * i for i in range(n + 1)]   # Éq. 11 — n+1 niveaux

    def _compute_equiratio(
        self, Gul: float, Gll: float, nu: int, nl: int
    ) -> list[float]:
        """
        Produit n+1 niveaux en progression géométrique de Gll à Gul.
        Éq. 3  : Gs = (Gul / Gll)^(1/n)
        Éq. 12 : gi = Gll × Gs^i,  i ∈ 0..n  (adapté pour n+1 niveaux)
        P0 n'est pas utilisé dans le calcul ; sa cohérence est vérifiée via I3.
        """
        n = nu + nl
        Gs = (Gul / Gll) ** (1.0 / n)                    # Éq. 3
        return [Gll * (Gs ** i) for i in range(n + 1)]    # Éq. 12 — n+1 niveaux

    def _compute_flexible(
        self, Gul: float, Gll: float, nu: int, nl: int, P0: float
    ) -> list[float]:
        """
        Produit n+1 niveaux flexibles ancrés sur P0.
        Éq. 13 : Gsu = (Gul / P0)^(1/nu)  — ratio de la progression supérieure
        Éq. 14 : Gsl = (P0 / Gll)^(1/nl)  — ratio de la progression inférieure

        Niveaux inférieurs (nl+1 valeurs, de Gll à P0 inclus) :
            Gll × Gsl^k,  k ∈ 0..nl

        Niveaux supérieurs (nu valeurs, de P0×Gsu à Gul) :
            P0 × Gsu^k,   k ∈ 1..nu

        Total : (nl+1) + nu = n+1 niveaux.
        """
        Gsu = (Gul / P0) ** (1.0 / nu)                       # Éq. 13
        Gsl = (P0  / Gll) ** (1.0 / nl)                      # Éq. 14
        lower = [Gll * (Gsl ** k) for k in range(nl + 1)]    # Gll … P0
        upper = [P0  * (Gsu ** k) for k in range(1, nu + 1)] # P0×Gsu … Gul
        return lower + upper

    # -------------------------------------------------------------------------
    # Vérification des invariants I1–I4, I6
    # -------------------------------------------------------------------------

    def _verify_invariants(
        self,
        levels: list[float],
        Gul: float,
        Gll: float,
        nu: int,
        nl: int,
        P0: float,
        h_pct: float,
    ) -> None:
        """
        Vérifie I1–I4 et I6 sur le treillis calculé.
        I5 est garantie par construction et vérifiée dans les tests.
        """
        n = nu + nl

        # I1 — Cardinalité : exactement n+1 niveaux (CD-002 §6 I1 ; RN-103 §2)
        if len(levels) != n + 1:
            raise TrellisParameterError(
                f"I1 violé : {n + 1} niveaux attendus, {len(levels)} obtenus"
            )

        # I2 — Bornes : Gll en premier, Gul en dernier (CD-002 §6 I2)
        if not _approx_equal(levels[0], Gll):
            raise TrellisParameterError(
                f"I2 violé : premier niveau {levels[0]} ≉ Gll {Gll}"
            )
        if not _approx_equal(levels[-1], Gul):
            raise TrellisParameterError(
                f"I2 violé : dernier niveau {levels[-1]} ≉ Gul {Gul}"
            )

        # I3 — P0 à la position nl (index 0), inégalités strictes de part et d'autre
        #      (CD-002 §6 I3 ; RN-103 §3, §6)
        #      Pour EQUIDISTANT et EQUIRATIO, valide également la cohérence de P0
        #      (CD-002 §4.1).
        p0_idx = nl  # 0-indexé ; CD-002 désigne la position 1-indexée nl+1
        if not _approx_equal(levels[p0_idx], P0):
            raise TrellisParameterError(
                f"I3 violé : niveau à l'index {p0_idx} vaut {levels[p0_idx]}, "
                f"P0={P0} attendu. "
                f"Pour EQUIDISTANT/EQUIRATIO, cela indique un P0 incohérent "
                f"avec les autres paramètres (CD-002 §4.1)."
            )
        for i in range(p0_idx):
            if not levels[i] < P0:
                raise TrellisParameterError(
                    f"I3 violé : levels[{i}]={levels[i]} n'est pas "
                    f"strictement inférieur à P0={P0}"
                )
        for i in range(p0_idx + 1, n + 1):
            if not levels[i] > P0:
                raise TrellisParameterError(
                    f"I3 violé : levels[{i}]={levels[i]} n'est pas "
                    f"strictement supérieur à P0={P0}"
                )

        # I4 — Ordre strict : tous les niveaux strictement croissants (CD-002 §6 I4)
        for i in range(n):
            if not levels[i] < levels[i + 1]:
                raise TrellisParameterError(
                    f"I4 violé : levels[{i}]={levels[i]} >= "
                    f"levels[{i + 1}]={levels[i + 1]}"
                )

        # I6 — Éq. 16 : gi+1 − gi > h_pct × gi+1 pour toute paire adjacente
        #      (CD-002 §6 I6)
        #      Ce service ne modifie jamais les paramètres ; une violation est
        #      une erreur d'entrée.
        for i in range(n):
            g_i  = levels[i]
            g_i1 = levels[i + 1]
            if not (g_i1 - g_i > h_pct * g_i1):
                raise TrellisParameterError(
                    f"I6 violé (Éq. 16) sur la paire ({i}, {i + 1}) : "
                    f"g[{i + 1}] − g[{i}] = {g_i1 - g_i:.6g} "
                    f"n'est pas > h_pct × g[{i + 1}] = {h_pct * g_i1:.6g} "
                    f"(h_pct={h_pct}). "
                    f"Ce service ne modifie pas les paramètres pour satisfaire "
                    f"cette contrainte."
                )


# ---------------------------------------------------------------------------
# Utilitaire interne
# ---------------------------------------------------------------------------

def _approx_equal(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """
    Égalité en virgule flottante pour les vérifications de bornes I2 et I3.
    Utilise une tolérance relative pour accommoder les accumulations d'arrondi
    dans les progressions géométriques.
    Non exposé dans l'interface publique de ce module.
    """
    if b == 0.0:
        return abs(a) < rel_tol
    return abs(a - b) / abs(b) < rel_tol
