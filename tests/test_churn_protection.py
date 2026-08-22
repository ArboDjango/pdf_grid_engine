"""Tests pour churn_protection.py -- mécanisme anti-churn N=3 + libération
par progression (chantier feature/churn-protection-n3).

Couvre directement les scénarios 1 à 8 et 13-18 du chantier (au niveau de
la logique PURE, update_churn_protection), plus la preuve explicite de
non-double-comptage exigée par la consigne ("cycles chevauchants").
"""

from decimal import Decimal

import pytest

from churn_protection import (
    ChurnProtectionEntry, GridFillRecord, fresh_entry, load_churn_protection,
    protected_gis, save_churn_protection, update_churn_protection,
)

GI_A = 1.02
GI_B = 1.03  # "autre gi" quelconque -- jamais restreint à un voisin immédiat
P0 = Decimal("1.00")


def f(gi, ts):
    return GridFillRecord(gi=gi, timestamp=ts)


# =============================================================================
# 1-3 : accumulation du compteur
# =============================================================================

class TestAccumulation:
    def test_1_single_c_cycle_not_protected(self):
        """1 cycle C (2 fills, aucun autre gi touché) -> cellule non protégée."""
        fills = (f(GI_A, 100), f(GI_A, 200))
        state = update_churn_protection({}, fills, P0)
        entry = state[GI_A]
        assert entry.consecutive_c_cycles == 1
        assert entry.churn_protected is False

    def test_2_two_consecutive_c_cycles_not_protected(self):
        """2 cycles C consécutifs -> toujours non protégée (seuil N=3 non atteint)."""
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300))
        state = update_churn_protection({}, fills, P0)
        entry = state[GI_A]
        assert entry.consecutive_c_cycles == 2
        assert entry.churn_protected is False

    def test_3_three_consecutive_c_cycles_protected(self):
        """3 cycles C consécutifs -> CHURN_PROTECTED, protected_at = fill
        qui a fait franchir le seuil."""
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        entry = state[GI_A]
        assert entry.consecutive_c_cycles == 3
        assert entry.churn_protected is True
        assert entry.protected_at == 400
        assert protected_gis(state) == frozenset({GI_A})


# =============================================================================
# 4 : aucune matérialisation de plus tant que protégée -- vérifié au niveau
# de select_exposed_orders (test_grid_order_exposure_controller.py) ; ici on
# vérifie que consecutive_c_cycles n'évolue PAS artificiellement au-delà de
# ce que de VRAIS fills produiraient (aucun 4e cycle simulé n'est compté
# sans un fill réel supplémentaire).
# =============================================================================

class TestNoPhantomFourthCycle:
    def test_4_no_extra_cycle_without_a_new_real_fill(self):
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        # Ré-appliquer avec les MÊMES fills (aucun nouveau) -- idempotent.
        state2 = update_churn_protection(state, fills, P0)
        assert state2[GI_A].consecutive_c_cycles == 3
        assert state2[GI_A].churn_protected is True


# =============================================================================
# 5, 7 : libération par progression, dans les deux sens
# =============================================================================

class TestReleaseByProgression:
    def test_5_progression_to_other_gi_releases_cell(self):
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].churn_protected is True

        fills_with_progression = fills + (f(GI_B, 500),)
        state2 = update_churn_protection(state, fills_with_progression, P0)
        assert state2[GI_A].churn_protected is False
        assert state2[GI_A].consecutive_c_cycles == 0
        assert state2[GI_A].protected_at is None

    def test_7_progression_in_the_other_direction_same_behavior(self):
        """Même principe, direction opposée (gi plus bas plutôt que plus
        haut) -- aucune notion de direction n'existe dans le mécanisme,
        seulement "un autre gi touché"."""
        gi_low = 0.98
        fills = (f(gi_low, 100), f(gi_low, 200), f(gi_low, 300), f(gi_low, 400))
        state = update_churn_protection({}, fills, P0)
        assert state[gi_low].churn_protected is True

        fills2 = fills + (f(GI_B, 500),)
        state2 = update_churn_protection(state, fills2, P0)
        assert state2[gi_low].churn_protected is False


# =============================================================================
# 6 : après libération, nouveau cycle sur l'ancienne cellule autorisé
# =============================================================================

class TestReeligibleAfterRelease:
    def test_6_cell_becomes_eligible_immediately_after_release(self):
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400), f(GI_B, 500))
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].churn_protected is False
        assert state[GI_A].consecutive_c_cycles == 0

    def test_6b_a_genuinely_fresh_c_cycle_after_release_counts_correctly(self):
        """Le cycle immédiatement suivant la libération (fenêtre 400->600)
        contient encore le fill libérateur (GI_B@500) dans SA PROPRE
        fenêtre -- il n'est donc PAS lui-même un cycle C, exactement comme
        classify_cycle le déterminerait rétrospectivement (une progression
        a eu lieu DURANT ce cycle). Seul le cycle suivant (600->700), dont
        la fenêtre ne contient plus aucun autre gi, est un C authentique."""
        fills = (
            f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400), f(GI_B, 500),
            f(GI_A, 600), f(GI_A, 700),
        )
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].churn_protected is False
        assert state[GI_A].consecutive_c_cycles == 1


# =============================================================================
# 8 : progression vers un niveau NON immédiatement voisin -- comportement de
# la mécanique existante (select_exposed_orders ne distingue jamais les
# niveaux par adjacence), donc AUCUNE règle de voisinage n'est inventée ici :
# un gi distant libère exactement comme un gi adjacent.
# =============================================================================

class TestNonAdjacentProgressionStillReleases:
    def test_8_far_away_gi_touch_releases_exactly_like_a_neighbor(self):
        gi_far = 1.50  # loin de GI_A, pas un "voisin" au sens treillis
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].churn_protected is True

        fills2 = fills + (f(gi_far, 401),)
        state2 = update_churn_protection(state, fills2, P0)
        assert state2[GI_A].churn_protected is False, (
            "un gi distant doit libérer exactement comme un voisin immédiat -- "
            "aucune restriction d'adjacence n'existe dans le mécanisme"
        )


# =============================================================================
# 11-12 : plusieurs cellules protégées simultanément, indépendance stricte
# =============================================================================

class TestIndependencePerCell:
    def test_11_each_cell_has_its_own_state(self):
        """gi_y touché AVANT le début des cycles de gi_x (jamais entre ses
        fills, jamais après son déclenchement) : ne casse pas ses fenêtres
        de cycle (elles ne le contiennent pas), et ne peut pas non plus le
        libérer rétroactivement (la libération ne regarde que les fills
        POSTÉRIEURS à protected_at). Si gi_y était touché après
        protected_at de gi_x, gi_x serait légitimement libéré -- ce n'est
        PAS un défaut d'indépendance, c'est exactement la définition de
        "libération par progression" (cf. test_5/test_7)."""
        gi_x, gi_y = 1.02, 1.10
        fills = (
            f(gi_y, 50), f(gi_y, 150),
            f(gi_x, 200), f(gi_x, 300), f(gi_x, 400), f(gi_x, 500),
        )
        state = update_churn_protection({}, fills, P0)
        assert state[gi_x].churn_protected is True
        assert state[gi_y].churn_protected is False
        assert state[gi_y].consecutive_c_cycles == 1

    def test_12_protecting_one_cell_never_touches_another(self):
        gi_x, gi_y = 1.02, 1.10
        fills = (f(gi_x, 100), f(gi_x, 200), f(gi_x, 300), f(gi_x, 400))
        state_before = update_churn_protection({}, (f(gi_y, 50),), P0)
        state_after = update_churn_protection(state_before, fills + (f(gi_y, 50),), P0)
        # gi_y : un seul fill jamais réobservé (déjà last_processed_fill_ts=50) -> inchangé
        assert state_after[gi_y] == state_before[gi_y]


# =============================================================================
# 17 : aucun cooldown temporel -- une protection ne s'arrête JAMAIS
# uniquement parce que du temps s'est écoulé, quelle que soit sa durée.
# =============================================================================

class TestNoTemporalCooldown:
    def test_17_protection_never_expires_from_time_alone(self):
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].churn_protected is True

        # Un temps arbitrairement long s'écoule, mais AUCUN nouveau fill,
        # ni sur GI_A ni ailleurs -- ré-application avec les MÊMES données.
        huge_later = update_churn_protection(state, fills, P0)
        assert huge_later[GI_A].churn_protected is True, (
            "la protection ne doit jamais expirer d'elle-même -- seule une "
            "progression réelle (fill sur un autre gi) peut la lever"
        )


# =============================================================================
# Non-double-comptage (cycles chevauchants) -- preuve directe
# =============================================================================

class TestNoDoubleCounting:
    def test_overlapping_pairs_are_never_double_counted(self):
        """4 fills consécutifs sur le même gi (aucun autre gi touché) ne
        doivent produire QUE 3 cycles C -- jamais 6 (paires chevauchantes
        comptées deux fois) ni aucune autre valeur."""
        fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state = update_churn_protection({}, fills, P0)
        assert state[GI_A].consecutive_c_cycles == 3

    def test_batch_of_new_fills_processed_incrementally_not_reprocessed(self):
        """Un premier appel traite les 2 premiers fills (1 cycle C) ; un
        second appel, avec 2 fills SUPPLÉMENTAIRES, doit ajouter EXACTEMENT
        2 cycles (total 3), jamais retraiter les fills déjà comptés."""
        state1 = update_churn_protection({}, (f(GI_A, 100), f(GI_A, 200)), P0)
        assert state1[GI_A].consecutive_c_cycles == 1

        all_fills = (f(GI_A, 100), f(GI_A, 200), f(GI_A, 300), f(GI_A, 400))
        state2 = update_churn_protection(state1, all_fills, P0)
        assert state2[GI_A].consecutive_c_cycles == 3
        assert state2[GI_A].last_processed_fill_ts == 400

    def test_first_ever_fill_forms_no_cycle(self):
        """Un seul fill (jamais de précédent) ne forme aucun cycle --
        exactement comme build_cycles() qui ne pair jamais un fill isolé."""
        state = update_churn_protection({}, (f(GI_A, 100),), P0)
        assert state[GI_A].consecutive_c_cycles == 0
        assert state[GI_A].last_processed_fill_ts == 100


# =============================================================================
# P0 n'est jamais une cellule
# =============================================================================

class TestP0NeverTracked:
    def test_p0_fills_are_ignored(self):
        p0 = 1.00
        fills = (f(p0, 100), f(p0, 200), f(p0, 300), f(p0, 400))
        state = update_churn_protection({}, fills, Decimal(str(p0)))
        assert p0 not in state


# =============================================================================
# ChurnProtectionEntry -- invariants
# =============================================================================

class TestEntryInvariants:
    def test_protected_requires_protected_at(self):
        with pytest.raises(ValueError):
            ChurnProtectionEntry(3, True, None, 100)

    def test_not_protected_requires_no_protected_at(self):
        with pytest.raises(ValueError):
            ChurnProtectionEntry(0, False, 100, 100)

    def test_negative_counter_rejected(self):
        with pytest.raises(ValueError):
            ChurnProtectionEntry(-1, False, None, None)

    def test_fresh_entry_is_the_default(self):
        assert fresh_entry() == ChurnProtectionEntry(0, False, None, None)


# =============================================================================
# Persistance -- fusion stricte avec le fichier de grille existant
# =============================================================================

class TestPersistence:
    def test_save_then_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        state = {GI_A: ChurnProtectionEntry(3, True, 400, 400)}
        save_churn_protection(state, path)
        loaded = load_churn_protection(path)
        assert loaded == state

    def test_save_preserves_existing_grid_config_keys(self, tmp_path):
        import json
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        with open(path, "w") as fh:
            json.dump({"inst_id": "XRP-USDC", "p0": "1.00"}, fh)

        save_churn_protection({GI_A: ChurnProtectionEntry(3, True, 400, 400)}, path)

        with open(path) as fh:
            payload = json.load(fh)
        assert payload["inst_id"] == "XRP-USDC"
        assert payload["p0"] == "1.00"
        assert "churn_protection" in payload

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_churn_protection(str(tmp_path / "absent.json")) == {}

    def test_load_corrupt_file_returns_empty_never_raises(self, tmp_path):
        path = tmp_path / "active_grid_state_XRP-USDC.json"
        path.write_text("{not valid json")
        assert load_churn_protection(str(path)) == {}

    def test_only_non_default_entries_are_persisted(self, tmp_path):
        """Une cellule jamais touchée par le mécanisme (état par défaut) ne
        produit aucune entrée dans le fichier -- "ne persister que ce qui
        est nécessaire"."""
        import json
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        save_churn_protection({GI_A: fresh_entry(), 1.10: ChurnProtectionEntry(1, False, None, 200)}, path)
        with open(path) as fh:
            payload = json.load(fh)
        assert str(GI_A) not in payload["churn_protection"]
        assert str(1.10) in payload["churn_protection"]
