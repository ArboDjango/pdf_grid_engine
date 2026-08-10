"""
Tests unitaires pour CellInstantiator — CD-003.

Chaque test est explicitement rattaché à un invariant (I1–I4) ou à la
règle contractuelle de CD-003.
Aucune référence à trellis_calculator, à l'exchange, ni à une logique
patrimoniale.

Les treillis de test sont construits manuellement — CD-003 ne connaît
pas l'origine des niveaux, uniquement leur valeur.

Les invariants I1–I4 de ce composant sont distincts des invariants I1–I6
de CD-002.
"""

import pytest

from cell import Cell, CellState
from cell_instantiator import CellInstantiator


# ---------------------------------------------------------------------------
# Treillis de référence
# ---------------------------------------------------------------------------

# Treillis équidistant canonique : 11 niveaux, P0 = 100.0 au centre (nl=5)
# Gll=90, Gs=2 → niveaux : 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110
TRELLIS_EQ = (90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0)
P0_EQ = 100.0
N_EQ = 10   # nombre de cellules attendu (n+1 - 1)
NL_EQ = 5   # niveaux sous P0
NU_EQ = 5   # niveaux au-dessus de P0

# Treillis flexible minimal : 5 niveaux, P0 = 100.0, nl=2, nu=2
TRELLIS_FLEX = (80.0, 90.0, 100.0, 110.0, 120.0)
P0_FLEX = 100.0
N_FLEX = 4


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def inst() -> CellInstantiator:
    return CellInstantiator()


# ---------------------------------------------------------------------------
# I1 — Cardinalité : exactement n = len(trellis) − 1 cellules (CD-003 §6 I1)
# ---------------------------------------------------------------------------

class TestI1Cardinalite:
    def test_I1_treillis_symetrique(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        assert len(cells) == N_EQ

    def test_I1_treillis_flexible(self, inst):
        cells = inst.instantiate(TRELLIS_FLEX, P0_FLEX)
        assert len(cells) == N_FLEX

    def test_I1_treillis_minimal_trois_niveaux(self, inst):
        """Treillis minimal : nl=1, nu=1 → 2 cellules."""
        t = (80.0, 100.0, 120.0)
        cells = inst.instantiate(t, 100.0)
        assert len(cells) == 2

    def test_I1_nu_nl_asymetriques(self, inst):
        """nl ≠ nu : la cardinalité reste n+1-1 = n."""
        t = (80.0, 90.0, 100.0, 110.0, 115.0, 120.0)  # nl=2, nu=3
        cells = inst.instantiate(t, 100.0)
        assert len(cells) == 5

    def test_I1_P0_est_exclu_du_compte(self, inst):
        """Le niveau P0 n'est adressé par aucune cellule (RN-103 §6)."""
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        # Exactement n niveaux (pas n+1) : P0 est ignoré
        assert len(cells) == len(TRELLIS_EQ) - 1


# ---------------------------------------------------------------------------
# I2 — Adressage : chaque gi appartient au treillis et est différent de P0
#      (CD-003 §6 I2 ; RN-103 §4, §6)
# ---------------------------------------------------------------------------

class TestI2Adressage:
    def test_I2_tous_gi_dans_treillis(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        gi_values = {c.gi for c in cells}
        expected = set(TRELLIS_EQ) - {P0_EQ}
        assert gi_values == expected

    def test_I2_P0_absent_des_gi(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        assert all(c.gi != P0_EQ for c in cells)

    def test_I2_chaque_niveau_adresse_exactement_une_cellule(self, inst):
        """Exhaustivité : injection de n niveaux vers n cellules (RN-103 §4)."""
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        gi_list = [c.gi for c in cells]
        assert len(gi_list) == len(set(gi_list))  # pas de doublon
        assert set(gi_list) == set(TRELLIS_EQ) - {P0_EQ}

    def test_I2_gi_est_attribut_cell(self, inst):
        """Cell reçoit bien gi lors de sa construction (CD-001 §3)."""
        trellis = (80.0, 100.0, 120.0)
        cells = inst.instantiate(trellis, 100.0)
        gi_values = sorted(c.gi for c in cells)
        assert gi_values == [80.0, 120.0]


# ---------------------------------------------------------------------------
# I3 — SPOT_HELD : gi > P0 → CellState.SPOT_HELD (CD-003 §6 I3)
# ---------------------------------------------------------------------------

class TestI3SpotHeld:
    def test_I3_niveaux_superieurs_sont_spot_held(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        for c in cells:
            if c.gi > P0_EQ:
                assert c.state is CellState.SPOT_HELD, (
                    f"gi={c.gi} > P0={P0_EQ} mais état {c.state!r}"
                )

    def test_I3_exactement_nu_cellules_spot_held(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        spot = [c for c in cells if c.state is CellState.SPOT_HELD]
        assert len(spot) == NU_EQ

    def test_I3_asymetrique_un_niveau_superieur(self, inst):
        trellis = (80.0, 100.0, 120.0)
        cells = inst.instantiate(trellis, 100.0)
        spot = [c for c in cells if c.state is CellState.SPOT_HELD]
        assert len(spot) == 1
        assert spot[0].gi == 120.0

    def test_I3_cell_recoit_etat_spot_held_a_la_construction(self, inst):
        """L'état est fourni au moment de la construction — CD-001 §5."""
        trellis = (90.0, 100.0, 110.0)
        cells = inst.instantiate(trellis, 100.0)
        spot = next(c for c in cells if c.gi == 110.0)
        assert spot.state is CellState.SPOT_HELD


# ---------------------------------------------------------------------------
# I4 — CASH_HELD : gi < P0 → CellState.CASH_HELD (CD-003 §6 I4)
# ---------------------------------------------------------------------------

class TestI4CashHeld:
    def test_I4_niveaux_inferieurs_sont_cash_held(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        for c in cells:
            if c.gi < P0_EQ:
                assert c.state is CellState.CASH_HELD, (
                    f"gi={c.gi} < P0={P0_EQ} mais état {c.state!r}"
                )

    def test_I4_exactement_nl_cellules_cash_held(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        cash = [c for c in cells if c.state is CellState.CASH_HELD]
        assert len(cash) == NL_EQ

    def test_I4_asymetrique_un_niveau_inferieur(self, inst):
        trellis = (80.0, 100.0, 120.0)
        cells = inst.instantiate(trellis, 100.0)
        cash = [c for c in cells if c.state is CellState.CASH_HELD]
        assert len(cash) == 1
        assert cash[0].gi == 80.0

    def test_I4_cell_recoit_etat_cash_held_a_la_construction(self, inst):
        trellis = (90.0, 100.0, 110.0)
        cells = inst.instantiate(trellis, 100.0)
        cash = next(c for c in cells if c.gi == 90.0)
        assert cash.state is CellState.CASH_HELD


# ---------------------------------------------------------------------------
# Exhaustivité (CD-003 §6, corollaire de I2 + I3 + I4)
# ---------------------------------------------------------------------------

class TestExhaustivite:
    def test_chaque_niveau_non_P0_adresse_exactement_une_fois(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        attendu = set(TRELLIS_EQ) - {P0_EQ}
        produit = {c.gi for c in cells}
        assert attendu == produit

    def test_partition_spot_cash_couvre_toutes_les_cellules(self, inst):
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        spot = {c.gi for c in cells if c.state is CellState.SPOT_HELD}
        cash = {c.gi for c in cells if c.state is CellState.CASH_HELD}
        assert spot | cash == set(TRELLIS_EQ) - {P0_EQ}
        assert spot & cash == set()

    def test_tous_gi_strictement_differents_de_P0(self, inst):
        cells = inst.instantiate(TRELLIS_FLEX, P0_FLEX)
        for c in cells:
            assert c.gi != P0_FLEX


# ---------------------------------------------------------------------------
# gi == P0 : P0 est ignoré, jamais adressé par une cellule (CD-003 §6)
# RN-103 §3 : P0 appartient au treillis comme niveau-frontière.
# RN-103 §6 : aucune cellule n'est adressée par P0 lors de la construction.
# ---------------------------------------------------------------------------

class TestGiEgalP0EstExclu:
    def test_P0_n_est_adresse_par_aucune_cellule(self, inst):
        """P0 présent dans le treillis → ignoré, aucune cellule avec gi == P0."""
        trellis = (90.0, 100.0, 110.0)
        cells = inst.instantiate(trellis, 100.0)
        assert all(c.gi != 100.0 for c in cells)

    def test_I1_respecte_malgre_presence_de_P0_dans_treillis(self, inst):
        """n+1 niveaux dont P0 → n cellules (CD-003 §5 ; I1)."""
        trellis = (90.0, 100.0, 110.0)  # n+1 = 3 niveaux
        cells = inst.instantiate(trellis, 100.0)
        assert len(cells) == 2  # n = 2

    def test_P0_en_debut_de_treillis_est_ignore(self, inst):
        """P0 en première position → ignoré, 2 cellules au-dessus."""
        trellis = (100.0, 110.0, 120.0)
        cells = inst.instantiate(trellis, 100.0)
        assert len(cells) == 2
        assert all(c.gi > 100.0 for c in cells)
        assert all(c.state is CellState.SPOT_HELD for c in cells)

    def test_P0_en_fin_de_treillis_est_ignore(self, inst):
        """P0 en dernière position → ignoré, 2 cellules en dessous."""
        trellis = (80.0, 90.0, 100.0)
        cells = inst.instantiate(trellis, 100.0)
        assert len(cells) == 2
        assert all(c.gi < 100.0 for c in cells)
        assert all(c.state is CellState.CASH_HELD for c in cells)

    def test_niveau_frontiere_pas_un_cas_metier(self, inst):
        """
        P0 n'est pas un cas métier à traiter (CD-003 §6).
        Le service produit les n cellules attendues sans exception.
        """
        # Un treillis valide (CD-002) contient toujours P0 — ce n'est pas une erreur
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        assert len(cells) == N_EQ


# ---------------------------------------------------------------------------
# P0 transmis séparément — frontière (CD-003 §4.2, §7)
# ---------------------------------------------------------------------------

class TestP0TransmisSeparement:
    def test_P0_different_produit_repartition_differente(self, inst):
        """
        Le même treillis avec un P0 différent produit des états différents.
        CD-003 reçoit P0 séparément et le consomme lors de l'instanciation.
        """
        trellis = (80.0, 90.0, 95.0, 110.0, 120.0)
        # P0 = 100.0 : 80, 90, 95 sous P0 → CASH ; 110, 120 au-dessus → SPOT
        cells_a = inst.instantiate(trellis, 100.0)
        # P0 = 92.0 : 80, 90 sous P0 → CASH ; 95, 110, 120 au-dessus → SPOT
        cells_b = inst.instantiate(trellis, 92.0)

        spot_a = {c.gi for c in cells_a if c.state is CellState.SPOT_HELD}
        spot_b = {c.gi for c in cells_b if c.state is CellState.SPOT_HELD}
        assert spot_a != spot_b

    def test_sortie_est_tuple_immuable(self, inst):
        """La collection produite est immuable (CD-003 §5)."""
        cells = inst.instantiate(TRELLIS_EQ, P0_EQ)
        assert isinstance(cells, tuple)
        with pytest.raises(TypeError):
            cells[0] = None  # type: ignore[index]


# ---------------------------------------------------------------------------
# Respect des frontières (CD-003 §7)
# ---------------------------------------------------------------------------

class TestRespectDesFrontieres:
    def test_pas_de_dependance_trellis_calculator(self):
        """
        cell_instantiator ne doit pas importer trellis_calculator
        (CD-003 §7 ; note d'implémentation).
        """
        import cell_instantiator
        assert not hasattr(cell_instantiator, "TrellisCalculator")
        assert not hasattr(cell_instantiator, "GridGeometry")
        assert not hasattr(cell_instantiator, "TrellisParameterError")

    def test_pas_de_grandeurs_patrimoniales(self):
        """
        cell_instantiator n'expose aucun concept patrimonial (CD-003 §7).
        """
        import cell_instantiator
        assert not hasattr(cell_instantiator, "Gv")
        assert not hasattr(cell_instantiator, "F0")

    def test_pas_de_parametres_geometriques(self):
        """
        cell_instantiator n'expose aucun paramètre géométrique (CD-003 §7).
        """
        import cell_instantiator
        assert not hasattr(cell_instantiator, "Gul")
        assert not hasattr(cell_instantiator, "Gll")

    def test_seule_dependance_autorisee_est_cell(self):
        """La seule dépendance interne autorisée est cell.py (CD-003 note)."""
        import cell_instantiator
        assert hasattr(cell_instantiator, "Cell")
        assert hasattr(cell_instantiator, "CellState")

    def test_service_sans_etat_persistant(self, inst):
        """
        Deux appels successifs produisent des résultats indépendants
        — pas d'état persistant entre les appels (SA-001 §2).
        """
        cells_1 = inst.instantiate(TRELLIS_EQ, P0_EQ)
        cells_2 = inst.instantiate(TRELLIS_EQ, P0_EQ)
        assert len(cells_1) == len(cells_2)
        assert {c.gi for c in cells_1} == {c.gi for c in cells_2}
        # Objets distincts (pas partagés entre appels)
        assert cells_1[0] is not cells_2[0]
