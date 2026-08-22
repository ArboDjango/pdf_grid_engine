"""churn_protection.py — protection anti-churn par cellule (N=3 + libération
par progression), validée par simulation rétrospective sur les fills réels
du 21/08/2026 (17 épisodes, 17 SAFE, 0 POTENTIELLEMENT_DOMMAGEABLE).

Mécanisme, EXACTEMENT celui validé, rien de plus :

    3 cycles C consécutifs sur une cellule (même gi)
        -> CHURN_PROTECTED
        -> aucun nouveau cycle sur cette cellule
        -> libérée dès qu'un AUTRE gi de la grille est effectivement touché
           (n'importe lequel, pas seulement un voisin immédiat -- la
           mécanique existante ne distingue jamais les niveaux par
           adjacence, cf. select_exposed_orders)
        -> compteur remis à zéro, cellule à nouveau éligible

Un cycle C = un cycle (paire de fills consécutifs de côté opposé sur LE
MÊME gi) durant lequel AUCUN AUTRE gi de la grille n'a été touché --
exactement la définition déjà établie et validée par
analyze_local_churn.py / simulate_anti_churn_thresholds.py /
simulate_opportunity_cost_policy_c.py. Ce module ne réinvente PAS cette
définition -- il l'applique de façon incrémentale, événement par
événement, jamais par retraitement d'un historique complet.

--------------------------------------------------------------------------
POURQUOI UN ÉTAT PERSISTÉ EST NÉCESSAIRE (et pas un recalcul à chaque cycle)
--------------------------------------------------------------------------
Le chemin de décision live (cell_state_reconstruction.reconstruct_cell_states)
reconstruit l'état ÉCONOMIQUE de chaque cellule à chaque cycle SANS AUCUNE
persistance locale, en s'appuyant sur adapter.list_fills() -- mais
OkxSpotAdapter.list_fills() appelle GET /api/v5/trade/fills SANS
pagination : cet endpoint OKX ne couvre que les exécutions récentes (la
fenêtre exacte dépend du compte/de l'API, non garantie indéfiniment
longue), avec au plus un lot de résultats par appel. Un compteur "cycles C
consécutifs" ne peut donc PAS être recalculé de façon fiable depuis zéro à
chaque cycle à partir de ce seul appel -- une cellule protégée depuis
plusieurs heures verrait son historique fondateur potentiellement sorti de
la fenêtre retournée par OKX.

Ce module persiste donc, PAR CELLULE, exactement ce qui est nécessaire
pour continuer à compter correctement SANS jamais retraiter l'historique
complet :

    consecutive_c_cycles    -- compteur de cycles C consécutifs
    churn_protected          -- la cellule est-elle actuellement protégée
    protected_at              -- horodatage de déclenchement (None si non protégée)
    last_processed_fill_ts   -- horodatage du dernier fill DE CETTE CELLULE
                                 déjà pris en compte dans consecutive_c_cycles
                                 (nécessaire pour ne traiter chaque fill
                                 qu'UNE SEULE FOIS -- voir note chevauchement
                                 ci-dessous)

Chaque cycle du bot ne regarde donc que ce qui est NOUVEAU depuis le
dernier passage (un fill qui vient de se produire, à l'instant, est par
construction toujours dans la fenêtre retournée par OKX) -- jamais un
retraitement de fills anciens.

--------------------------------------------------------------------------
NOTE CHEVAUCHEMENT (cycles qui se chevauchent)
--------------------------------------------------------------------------
L'analyse rétrospective (build_cycles() dans analyze_local_churn.py) a
démontré que des cycles construits par appariement de fills consécutifs se
CHEVAUCHENT : le fill de fermeture d'un cycle est aussi le fill
d'ouverture du suivant. Ce module n'utilise PAS cette construction par
paires : chaque fill RÉEL de la cellule n'est traité QU'UNE SEULE FOIS,
au moment où il est détecté comme nouveau (last_processed_fill_ts avance
d'exactement un cran par fill traité, jamais recompté). Voir
test_churn_protection.py::TestNoDoubleCounting pour la preuve directe.

--------------------------------------------------------------------------
CE QUE CE MODULE NE FAIT JAMAIS
--------------------------------------------------------------------------
- Ne modifie jamais gi, q, allocated_capital, K_BUY/K_SELL, la géométrie,
  ni aucune formule du PDF.
- N'annule jamais un ordre déjà ouvert -- interdit uniquement la
  matérialisation d'un NOUVEAU cycle (appliqué dans
  grid_order_exposure_controller.select_exposed_orders).
- N'introduit aucun cooldown temporel, aucun timeout, aucun seuil ATR/
  volatilité -- la SEULE condition de libération est un fill réel sur un
  autre gi.
- Ne touche jamais cell_memory.py (proposition architecturale distincte,
  toujours non branchée au chemin de décision -- ce module ne la branche
  pas non plus) ni CellMemoryEntry.target_qty.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from decimal import Decimal

from grid_preflight import PreflightReport
from okx_spot_adapter import OkxApiError

# Seul paramètre du mécanisme -- validé par simulation rétrospective sur les
# fills réels du 21/08/2026 (17/17 épisodes SAFE). Aucun autre seuil
# (temporel, ATR, volatilité) n'existe dans ce module.
CHURN_PROTECTION_N = 3

_SAME_PRICE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ChurnProtectionEntry:
    """État anti-churn persisté d'une seule cellule, identifiée par son gi
    (jamais inclus ici -- clé du dictionnaire, gi restant immuable, cf.
    cell.py CD-001 §4)."""

    consecutive_c_cycles: int
    churn_protected: bool
    protected_at: int | None
    last_processed_fill_ts: int | None

    def __post_init__(self):
        if self.consecutive_c_cycles < 0:
            raise ValueError("consecutive_c_cycles doit être >= 0")
        if self.churn_protected and self.protected_at is None:
            raise ValueError("churn_protected=True exige protected_at non None")
        if not self.churn_protected and self.protected_at is not None:
            raise ValueError("churn_protected=False exige protected_at=None")


def fresh_entry() -> ChurnProtectionEntry:
    """État initial d'une cellule n'ayant encore jamais été observée par ce
    mécanisme -- jamais protégée, compteur à zéro."""
    return ChurnProtectionEntry(
        consecutive_c_cycles=0,
        churn_protected=False,
        protected_at=None,
        last_processed_fill_ts=None,
    )


@dataclass(frozen=True)
class GridFillRecord:
    """Fill minimal, réduit aux deux seuls champs nécessaires à ce module
    (gi apparié au treillis, horodatage) -- jamais un remplacement de
    okx_spot_adapter.Fill, seulement une projection locale."""

    gi: float
    timestamp: int


# ---------------------------------------------------------------------------
# Persistance -- MÊME fichier que la configuration de grille
# (active_grid_state_<inst_id>.json), sous une clé de premier niveau dédiée
# "churn_protection", par fusion stricte -- même schéma et même mécanisme
# atomique (tempfile + fsync + os.replace) que cell_memory.py, sans jamais
# importer ni brancher ce module dormant (hors périmètre de ce chantier).
# ---------------------------------------------------------------------------


def _serialize_entry(entry: ChurnProtectionEntry) -> dict:
    return {
        "consecutive_c_cycles": entry.consecutive_c_cycles,
        "churn_protected": entry.churn_protected,
        "protected_at": entry.protected_at,
        "last_processed_fill_ts": entry.last_processed_fill_ts,
    }


def _deserialize_entry(payload: dict) -> ChurnProtectionEntry:
    return ChurnProtectionEntry(
        consecutive_c_cycles=payload["consecutive_c_cycles"],
        churn_protected=payload["churn_protected"],
        protected_at=payload["protected_at"],
        last_processed_fill_ts=payload["last_processed_fill_ts"],
    )


def load_churn_protection(path: str) -> dict[float, ChurnProtectionEntry]:
    """Lit la section churn_protection du fichier de grille. Retourne un
    dict vide si le fichier est absent, invalide, ou dépourvu de cette
    section -- NE LÈVE JAMAIS (même doctrine que load_cell_memory /
    load_grid_state : un contenu invalide est traité comme une absence,
    jamais comme une erreur fatale qui interromprait la boucle active)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
        raw = payload.get("churn_protection")
        if raw is None:
            return {}
        return {float(gi_str): _deserialize_entry(entry) for gi_str, entry in raw.items()}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
        return {}


def save_churn_protection(state: dict[float, ChurnProtectionEntry], path: str) -> None:
    """Écrit l'état anti-churn dans le fichier de grille existant, PAR
    FUSION STRICTE -- lit d'abord le contenu actuel (s'il existe), ne
    modifie QUE la clé "churn_protection", préserve tous les autres champs
    (configuration de grille, cell_memory éventuel) strictement intacts.
    Écriture atomique (tempfile + fsync + os.replace), identique à
    save_grid_state/save_cell_memory.

    Seules les entrées NON par défaut sont persistées (ne persister que ce
    qui est nécessaire) -- une cellule jamais touchée par ce mécanisme ne
    produit aucune entrée dans le fichier."""
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    meaningful = {gi: entry for gi, entry in state.items() if entry != fresh_entry()}
    existing["churn_protection"] = {str(gi): _serialize_entry(entry) for gi, entry in meaningful.items()}

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".active_grid_state_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(existing, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Lecture des fills -- appel INDÉPENDANT de reconstruct_cell_states (jamais
# une modification de cell_state_reconstruction.py), même critère
# d'appartenance à la grille ("GRID" + inst_id) et même comparaison
# tolérante de prix -- pour ne jamais dévier de la définition déjà en
# production.
# ---------------------------------------------------------------------------


def fetch_grid_fills(adapter, report: PreflightReport) -> tuple[GridFillRecord, ...]:
    """Récupère les fills de cette grille et les apparie au gi du treillis
    correspondant. I/O SEULE -- aucun calcul de protection ici (séparation
    stricte lecture / logique pure, comme plan_exposure vis-à-vis de
    select_exposed_orders)."""
    inst_id = report.config.inst_id
    try:
        fills = adapter.list_fills(inst_id)
    except OkxApiError:
        fills = ()

    trellis_floats = [float(g) for g in report.trellis]

    records = []
    for fill in fills:
        if fill.inst_id != inst_id or not fill.client_order_id.startswith("GRID"):
            continue
        if fill.filled_at is None:
            continue
        price = float(fill.fill_price)
        matched_gi = next(
            (g for g in trellis_floats if abs(g - price) < _SAME_PRICE_TOLERANCE),
            None,
        )
        if matched_gi is None:
            continue
        records.append(GridFillRecord(gi=matched_gi, timestamp=fill.filled_at))

    return tuple(records)


# ---------------------------------------------------------------------------
# Logique pure -- aucun I/O, aucun appel réseau, aucun effet de bord.
# ---------------------------------------------------------------------------


def update_churn_protection(
    state: dict[float, ChurnProtectionEntry],
    grid_fills: tuple[GridFillRecord, ...],
    p0: Decimal,
    N: int = CHURN_PROTECTION_N,
) -> dict[float, ChurnProtectionEntry]:
    """Calcule le nouvel état anti-churn à partir de l'état persisté
    précédent et des fills de grille actuellement connus (fenêtre OKX
    récente). PURE : ne modifie ni `state` ni `grid_fills`, retourne un
    nouveau dict.

    P0 n'est jamais une cellule (cf. cell_state_reconstruction.py) --
    tout fill dont le gi apparié serait P0 est ignoré (ne devrait
    structurellement jamais se produire, aucun ordre n'étant jamais
    matérialisé à P0).
    """
    p0_float = float(p0)
    relevant_gis = {gi for gi in state} | {f.gi for f in grid_fills if abs(f.gi - p0_float) >= _SAME_PRICE_TOLERANCE}

    new_state: dict[float, ChurnProtectionEntry] = dict(state)

    # Phase 1 -- comptage des cycles C, un NOUVEAU fill de la cellule à la
    # fois, dans l'ordre chronologique -- jamais un retraitement de
    # l'historique déjà pris en compte (last_processed_fill_ts). Exécutée
    # AVANT la libération par progression (Phase 2, ci-dessous) : une
    # cellule qui franchit le seuil ET dont le fill libérateur figure déjà
    # dans CE MÊME lot de fills (les deux événements observés au même
    # cycle du bot) doit être libérée immédiatement, jamais retardée d'un
    # cycle supplémentaire en attendant le prochain appel.
    for gi in relevant_gis:
        entry = new_state.get(gi, fresh_entry())
        own_new_fills = sorted(
            (f for f in grid_fills if f.gi == gi and (
                entry.last_processed_fill_ts is None or f.timestamp > entry.last_processed_fill_ts
            )),
            key=lambda f: f.timestamp,
        )
        if not own_new_fills:
            continue

        consecutive_c = entry.consecutive_c_cycles
        churn_protected = entry.churn_protected
        protected_at = entry.protected_at
        previous_ts = entry.last_processed_fill_ts

        for fill in own_new_fills:
            if previous_ts is None:
                # Tout premier fill jamais observé pour cette cellule sous
                # ce mécanisme -- aucun cycle ne peut encore être formé
                # (il faut un fill PRÉCÉDENT pour fermer un cycle).
                previous_ts = fill.timestamp
                continue

            other_gi_touched = any(
                f.gi != gi and previous_ts < f.timestamp <= fill.timestamp
                for f in grid_fills
            )
            if other_gi_touched:
                # progression détectée -- même traitement que la Phase 1 :
                # remise à zéro complète, jamais un état intermédiaire.
                consecutive_c = 0
                churn_protected = False
                protected_at = None
            else:
                consecutive_c += 1
                if consecutive_c >= N and not churn_protected:
                    churn_protected = True
                    protected_at = fill.timestamp

            previous_ts = fill.timestamp

        new_state[gi] = ChurnProtectionEntry(
            consecutive_c_cycles=consecutive_c,
            churn_protected=churn_protected,
            protected_at=protected_at,
            last_processed_fill_ts=previous_ts,
        )

    # Phase 2 -- libération par progression : toute cellule ACTUELLEMENT
    # protégée (y compris une cellule qui vient tout juste d'être protégée
    # par la Phase 1 ci-dessus, dans ce même appel) est libérée dès qu'un
    # fill sur un AUTRE gi (n'importe lequel) est survenu après
    # protected_at. Indépendant du fait que la cellule elle-même ait ou
    # non un nouveau fill ce cycle-ci.
    for gi in relevant_gis:
        entry = new_state.get(gi, fresh_entry())
        if not entry.churn_protected:
            continue
        progressed = any(
            f.gi != gi and f.timestamp > entry.protected_at
            for f in grid_fills
        )
        if progressed:
            # Remet le compteur/la protection à zéro -- mais PRÉSERVE
            # last_processed_fill_ts : l'oublier ferait retraiter la
            # cellule, au prochain appel, comme "jamais vue", donc rejouer
            # ses fills déjà comptés depuis le tout début et la
            # re-protéger aussitôt à tort avec les mêmes fills historiques.
            new_state[gi] = replace(
                entry,
                consecutive_c_cycles=0,
                churn_protected=False,
                protected_at=None,
            )

    return new_state


def protected_gis(state: dict[float, ChurnProtectionEntry]) -> frozenset[float]:
    """Ensemble des gi actuellement CHURN_PROTECTED -- ce que
    select_exposed_orders a besoin de connaître, rien de plus."""
    return frozenset(gi for gi, entry in state.items() if entry.churn_protected)
