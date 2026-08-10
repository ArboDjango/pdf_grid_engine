"""
Tests unitaires pour TrellisCalculator — CD-002.

Chaque test est explicitement rattaché à un ou plusieurs invariants (I1–I6)
ou à la validation des entrées. La nomenclature suit le schéma :
  test_I{invariant}_{géométrie}_{description}
ou
  test_invalid_{paramètre}_{description}

Aucun test n'introduit de logique au-delà de ce que CD-002 spécifie.
Aucune référence à Cell, aux états patrimoniaux, ou au monde réel.
"""

import math
import pytest

from trellis_calculator import (
    GridGeometry,
    TrellisCalculator,
    TrellisParameterError,
)


# ---------------------------------------------------------------------------
# Utilitaires de test
# ---------------------------------------------------------------------------

def _equiratio_p0(Gul: float, Gll: float, nu: int, nl: int) -> float:
    """P0 cohérent pour une géométrie EQUIRATIO — calculé depuis les paramètres."""
    n = nu + nl
    Gs = (Gul / Gll) ** (1.0 / n)
    return Gll * (Gs ** nl)


# ---------------------------------------------------------------------------
# Paramètres canoniques par géométrie
# ---------------------------------------------------------------------------

# EQUIDISTANT : P0 = Gll + nl × Gs = 90 + 5×2 = 100  (exact)
EQUIDIST = dict(
    Gul=110.0,
    Gll=90.0,
    nu=5,
    nl=5,
    P0=100.0,
    geometry=GridGeometry.EQUIDISTANT,
    h_pct=0.001,   # Gs=2 ; 2 > 0.001×102 ≈ 0.102 ✓ pour toute paire
)

# EQUIRATIO : P0 = sqrt(90×110) ≈ 99.499  (calculé)
_EQR_P0 = _equiratio_p0(110.0, 90.0, 5, 5)
EQUIRATIO = dict(
    Gul=110.0,
    Gll=90.0,
    nu=5,
    nl=5,
    P0=_EQR_P0,
    geometry=GridGeometry.EQUIRATIO,
    h_pct=0.001,
)

# FLEXIBLE : P0 ancre explicite
FLEXIBLE = dict(
    Gul=120.0,
    Gll=80.0,
    nu=4,
    nl=4,
    P0=100.0,
    geometry=GridGeometry.FLEXIBLE,
    h_pct=0.001,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def calc() -> TrellisCalculator:
    return TrellisCalculator()


# ---------------------------------------------------------------------------
# Immutabilité — CD-002 §5
# ---------------------------------------------------------------------------

class TestImmutabilite:
    def test_sortie_est_un_tuple(self, calc):
        result = calc.calculate(**EQUIDIST)
        assert isinstance(result, tuple)

    def test_tuple_non_modifiable(self, calc):
        result = calc.calculate(**EQUIDIST)
        with pytest.raises(TypeError):
            result[0] = 0.0  # type: ignore[index]

    def test_immutabilite_equiratio(self, calc):
        result = calc.calculate(**EQUIRATIO)
        assert isinstance(result, tuple)

    def test_immutabilite_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# I1 — Cardinalité : exactement nu+nl+1 niveaux (CD-002 §6 I1 ; RN-103 §2)
# ---------------------------------------------------------------------------

class TestI1Cardinalite:
    def test_I1_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        assert len(result) == EQUIDIST["nu"] + EQUIDIST["nl"] + 1

    def test_I1_equiratio(self, calc):
        result = calc.calculate(**EQUIRATIO)
        assert len(result) == EQUIRATIO["nu"] + EQUIRATIO["nl"] + 1

    def test_I1_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        assert len(result) == FLEXIBLE["nu"] + FLEXIBLE["nl"] + 1

    def test_I1_nu_nl_asymetriques(self, calc):
        """nu ≠ nl : la cardinalité doit rester nu+nl+1."""
        result = calc.calculate(
            Gul=130.0, Gll=90.0, nu=8, nl=4, P0=90.0 + 4 * (130.0 - 90.0) / 12,
            geometry=GridGeometry.EQUIDISTANT, h_pct=0.001,
        )
        assert len(result) == 8 + 4 + 1


# ---------------------------------------------------------------------------
# I2 — Bornes : Gll en premier, Gul en dernier (CD-002 §6 I2)
# ---------------------------------------------------------------------------

class TestI2Bornes:
    def test_I2_premier_niveau_Gll_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        assert result[0] == pytest.approx(EQUIDIST["Gll"])

    def test_I2_dernier_niveau_Gul_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        assert result[-1] == pytest.approx(EQUIDIST["Gul"])

    def test_I2_bornes_equiratio(self, calc):
        result = calc.calculate(**EQUIRATIO)
        assert result[0]  == pytest.approx(EQUIRATIO["Gll"])
        assert result[-1] == pytest.approx(EQUIRATIO["Gul"])

    def test_I2_bornes_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        assert result[0]  == pytest.approx(FLEXIBLE["Gll"])
        assert result[-1] == pytest.approx(FLEXIBLE["Gul"])


# ---------------------------------------------------------------------------
# I3 — P0 à l'index nl (0-indexé) ; inégalités strictes de part et d'autre
#      (CD-002 §6 I3 ; RN-103 §3, §6)
# ---------------------------------------------------------------------------

class TestI3PositionP0:
    def test_I3_P0_a_index_nl_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        assert result[EQUIDIST["nl"]] == pytest.approx(EQUIDIST["P0"])

    def test_I3_niveaux_inferieurs_strictement_sous_P0_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        nl, P0 = EQUIDIST["nl"], EQUIDIST["P0"]
        for level in result[:nl]:
            assert level < P0

    def test_I3_niveaux_superieurs_strictement_au_dessus_P0_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        nl, P0 = EQUIDIST["nl"], EQUIDIST["P0"]
        for level in result[nl + 1:]:
            assert level > P0

    def test_I3_P0_a_index_nl_equiratio(self, calc):
        result = calc.calculate(**EQUIRATIO)
        assert result[EQUIRATIO["nl"]] == pytest.approx(EQUIRATIO["P0"], rel=1e-9)

    def test_I3_P0_a_index_nl_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        assert result[FLEXIBLE["nl"]] == pytest.approx(FLEXIBLE["P0"], rel=1e-9)

    def test_I3_niveaux_inferieurs_strictement_sous_P0_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        nl, P0 = FLEXIBLE["nl"], FLEXIBLE["P0"]
        for level in result[:nl]:
            assert level < P0

    def test_I3_niveaux_superieurs_strictement_au_dessus_P0_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        nl, P0 = FLEXIBLE["nl"], FLEXIBLE["P0"]
        for level in result[nl + 1:]:
            assert level > P0

    def test_I3_P0_incoherent_equidistant_leve_erreur(self, calc):
        """I3 détecte un P0 incohérent pour EQUIDISTANT (CD-002 §4.1)."""
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "P0": 101.0})  # pas sur un niveau de grille

    def test_I3_P0_incoherent_equiratio_leve_erreur(self, calc):
        """I3 détecte un P0 incohérent pour EQUIRATIO (CD-002 §4.1)."""
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIRATIO, "P0": 100.0})  # pas la moyenne géométrique


# ---------------------------------------------------------------------------
# I4 — Ordre strict : tous les niveaux strictement croissants (CD-002 §6 I4)
# ---------------------------------------------------------------------------

class TestI4OrdreStrict:
    def test_I4_equidistant(self, calc):
        result = calc.calculate(**EQUIDIST)
        for a, b in zip(result, result[1:]):
            assert a < b

    def test_I4_equiratio(self, calc):
        result = calc.calculate(**EQUIRATIO)
        for a, b in zip(result, result[1:]):
            assert a < b

    def test_I4_flexible(self, calc):
        result = calc.calculate(**FLEXIBLE)
        for a, b in zip(result, result[1:]):
            assert a < b


# ---------------------------------------------------------------------------
# I5 — Conformité géométrique (CD-002 §6 I5)
# Vérifiée ici plutôt qu'à l'exécution (garantie par construction).
# ---------------------------------------------------------------------------

class TestI5ConformiteGeometrique:
    def test_I5_espacement_constant_equidistant(self, calc):
        """Tous les espacements valent Gs = (Gul−Gll)/n (Éq. 1)."""
        result = calc.calculate(**EQUIDIST)
        n = EQUIDIST["nu"] + EQUIDIST["nl"]
        Gs_attendu = (EQUIDIST["Gul"] - EQUIDIST["Gll"]) / n
        espacements = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        for esp in espacements:
            assert esp == pytest.approx(Gs_attendu)

    def test_I5_ratio_constant_equiratio(self, calc):
        """Tous les ratios gi+1/gi valent Gs = (Gul/Gll)^(1/n) (Éq. 3)."""
        result = calc.calculate(**EQUIRATIO)
        n = EQUIRATIO["nu"] + EQUIRATIO["nl"]
        Gs_attendu = (EQUIRATIO["Gul"] / EQUIRATIO["Gll"]) ** (1.0 / n)
        for i in range(len(result) - 1):
            assert result[i + 1] / result[i] == pytest.approx(Gs_attendu)

    def test_I5_ratio_superieur_flexible(self, calc):
        """Moitié supérieure : ratio Gsu constant (Éq. 13)."""
        result = calc.calculate(**FLEXIBLE)
        nu, nl = FLEXIBLE["nu"], FLEXIBLE["nl"]
        Gsu_attendu = (FLEXIBLE["Gul"] / FLEXIBLE["P0"]) ** (1.0 / nu)
        for k in range(1, nu + 1):
            assert result[nl + k] / result[nl + k - 1] == pytest.approx(Gsu_attendu)

    def test_I5_ratio_inferieur_flexible(self, calc):
        """Moitié inférieure : ratio Gsl constant (Éq. 14)."""
        result = calc.calculate(**FLEXIBLE)
        nl = FLEXIBLE["nl"]
        Gsl_attendu = (FLEXIBLE["P0"] / FLEXIBLE["Gll"]) ** (1.0 / nl)
        for k in range(1, nl + 1):
            assert result[k] / result[k - 1] == pytest.approx(Gsl_attendu)


# ---------------------------------------------------------------------------
# I6 — Éq. 16 : gi+1 − gi > h_pct × gi+1 pour toute paire (CD-002 §6 I6)
# ---------------------------------------------------------------------------

class TestI6ContrainteEspacement:
    def test_I6_satisfaite_equidistant(self, calc):
        calc.calculate(**EQUIDIST)  # ne doit pas lever

    def test_I6_satisfaite_equiratio(self, calc):
        calc.calculate(**EQUIRATIO)  # ne doit pas lever

    def test_I6_satisfaite_flexible(self, calc):
        calc.calculate(**FLEXIBLE)  # ne doit pas lever

    def test_I6_violation_leve_erreur(self, calc):
        """h_pct si grand qu'aucune paire ne peut satisfaire Éq. 16."""
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "h_pct": 0.5})

    def test_I6_message_mentionne_I6_ou_eq16(self, calc):
        """Le message d'erreur identifie I6 ou Éq. 16 (traçabilité CD-002)."""
        with pytest.raises(TrellisParameterError) as exc_info:
            calc.calculate(**{**EQUIDIST, "h_pct": 0.5})
        msg = str(exc_info.value)
        assert "I6" in msg or "16" in msg

    def test_I6_service_ne_modifie_pas_les_parametres(self, calc):
        """
        Le service lève une erreur sans tenter d'ajuster les paramètres
        (CD-002 §6 I6 : «ce composant ne modifie jamais les paramètres»).
        """
        params_originaux = {**EQUIDIST, "h_pct": 0.5}
        h_pct_avant = params_originaux["h_pct"]
        with pytest.raises(TrellisParameterError):
            calc.calculate(**params_originaux)
        assert params_originaux["h_pct"] == h_pct_avant

    def test_I6_toutes_les_paires_verifiees(self, calc):
        """
        Une paire violant I6 parmi les paires hautes suffit à lever une erreur.
        Éq. 16 sur la paire (108, 110) : 2 > 0.019×110=2.09 ? Non → violation.
        Les paires basses passent : 2 > 0.019×92=1.748 ? Oui.
        """
        # h_pct = 0.019 : violations sur les paires supérieures uniquement
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "h_pct": 0.019})


# ---------------------------------------------------------------------------
# Validation des entrées
# ---------------------------------------------------------------------------

class TestValidationEntrees:
    def test_invalid_nu_zero(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "nu": 0})

    def test_invalid_nu_negatif(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "nu": -1})

    def test_invalid_nu_flottant(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "nu": 5.0})

    def test_invalid_nl_zero(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "nl": 0})

    def test_invalid_nl_negatif(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "nl": -1})

    def test_invalid_Gll_zero(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "Gll": 0.0, "P0": 50.0})

    def test_invalid_Gll_negatif(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "Gll": -10.0, "P0": 50.0})

    def test_invalid_Gul_egal_Gll(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "Gul": 90.0})

    def test_invalid_Gul_inferieur_Gll(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "Gul": 80.0})

    def test_invalid_P0_egal_Gll(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "P0": 90.0})

    def test_invalid_P0_egal_Gul(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "P0": 110.0})

    def test_invalid_P0_sous_Gll(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "P0": 85.0})

    def test_invalid_P0_au_dessus_Gul(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "P0": 115.0})

    def test_invalid_h_pct_zero(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "h_pct": 0.0})

    def test_invalid_h_pct_negatif(self, calc):
        with pytest.raises(TrellisParameterError):
            calc.calculate(**{**EQUIDIST, "h_pct": -0.01})

    def test_geometrie_inconnue_leve_erreur(self, calc):
        with pytest.raises((TrellisParameterError, AttributeError, TypeError)):
            calc.calculate(**{**EQUIDIST, "geometry": "INCONNUE"})


# ---------------------------------------------------------------------------
# Respect des frontières — CD-002 §7
# Aucune dépendance envers Cell, les états patrimoniaux, ou le monde réel.
# ---------------------------------------------------------------------------

class TestRespectDesFrontieres:
    def test_pas_de_dependance_cell(self):
        """
        trellis_calculator ne doit pas importer Cell ni les états patrimoniaux
        (CD-002 §7 : «ne connaît jamais Cell, SPOT_HELD/CASH_HELD…»).
        """
        import trellis_calculator
        assert not hasattr(trellis_calculator, "Cell")
        assert not hasattr(trellis_calculator, "CellState")
        assert not hasattr(trellis_calculator, "SPOT_HELD")
        assert not hasattr(trellis_calculator, "CASH_HELD")

    def test_pas_de_grandeur_patrimoniale(self):
        """
        trellis_calculator n'expose aucun concept patrimonial (Gv, F0)
        dans son espace de noms public (CD-002 §7).
        """
        import trellis_calculator
        assert not hasattr(trellis_calculator, "Gv")
        assert not hasattr(trellis_calculator, "F0")
