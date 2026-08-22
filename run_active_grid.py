"""run_active_grid.py — reprise/création et boucle d'exécution d'une grille active.

RÈGLE ARCHITECTURALE DÉFINITIVE : PreflightConfig = identité IMMUTABLE de
la grille active. Une fois construit -- par reprise (valeurs historiques
codées en dur) ou par création (optimiseur, ci-dessous) -- P0/GLL/GUL/nu/
nl/geometry ne sont plus jamais recalculés pendant la vie de cette
instance.

DEUX chemins de construction de `config`, tous deux EXCLUSIVEMENT avant
la boucle, jamais dans `run()` :

  1. build_historical_config -- reprise d'une grille déjà active
     (valeurs historiques codées en dur, inchangé par ce chantier).

  2. build_optimized_config -- création d'une NOUVELLE grille, via la
     policy V1 déjà validée dans ce projet (MarketReader 15 min -> ATR
     normalisé + DI ratio -> classification -> GLL/GUL -> validation
     économique -> PreflightConfig). Formule reprise VERBATIM de
     live_first_activation_xrp.py/diagnose_xrp_policy.py -- aucune
     formule réinventée. Duplication assumée (troisième occurrence) :
     aucune consolidation entre fichiers n'a été faite, conformément à
     la consigne de limiter le changement au point d'intégration et de
     ne pas faire de refactoring général.

`run()` reste STRICTEMENT INCHANGÉE : elle reçoit toujours un `config`
déjà construit, par n'importe lequel des deux chemins, et ne consulte
JAMAIS MarketReader, JAMAIS la policy, JAMAIS get_ticker() -- elle ne
sait même pas comment `config` a été obtenu.

Réutilise EXCLUSIVEMENT des briques existantes, toutes inchangées :
  - run_grid_trading.build_adapter                    (construction de l'adaptateur UNIQUEMENT)
  - market_reader.read                                 (build_optimized_config UNIQUEMENT, jamais dans run())
  - candidate_economic_validator.validate_candidate     (build_optimized_config UNIQUEMENT)
  - grid_preflight.run_preflight                        (à chaque cycle, même config)
  - grid_order_orchestrator.run_exposure_cycle           (à chaque cycle -- inclut le départage
                                                           déjà en place, préférant un ordre déjà
                                                           ouvert à distance d'index égale ; reçoit
                                                           désormais churn_state_path pour la
                                                           protection anti-churn N=3, cf.
                                                           churn_protection.py -- comportement de
                                                           sélection des ordres inchangé pour toute
                                                           cellule non CHURN_PROTECTED)

Aucune formule économique, aucune formule de treillis, aucun mécanisme de
client_order_id, aucune logique de repost n'est réimplémentée ici -- ce
fichier ne fait que séquencer des appels à des fonctions déjà existantes.

build_config() (run_grid_trading.py), GridTradingController : JAMAIS
appelés, ni au démarrage, ni dans la boucle, dans AUCUN des deux modes.

Mode READ-ONLY par défaut : sans --live, affiche l'identité de la grille
(reprise ou créée) et s'arrête AVANT toute entrée dans la boucle --
run_exposure_cycle n'est alors jamais appelé, donc aucune écriture n'est
possible. --live requis pour entrer réellement dans la boucle active.

--mode resume (défaut, comportement inchangé) : reprend la grille
historique déjà active.
--mode create : construit une NOUVELLE grille via l'optimiseur, PUIS
l'active via GridTradingController (moteur d'activation existant,
inchangé) -- le MÊME objet `config`, construit une seule fois par
build_optimized_config, est transmis successivement à
GridTradingController().run() puis à run() : aucune reconstruction, aucun
recalcul de P0/GLL/GUL entre l'activation et l'exécution. L'activation
n'est effectuée que sous --live (elle écrit réellement) -- à n'utiliser
qu'après arrêt complet de toute grille active existante (mécanisme
d'arrêt non traité ici, volontairement hors périmètre).

Arrêt : SIGINT (Ctrl-C) uniquement. Aucun recalcul de grille, aucune
modification d'ordre au moment de l'arrêt -- sortie propre. La logique de
récupération du capital n'est délibérément pas traitée ici.

Usage :
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... python3 run_active_grid.py                          # reprise, READ-ONLY
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... python3 run_active_grid.py --live                   # reprise, boucle réelle
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... python3 run_active_grid.py --mode create             # création, READ-ONLY
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... python3 run_active_grid.py --mode create --live      # création, boucle réelle
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import time
from dataclasses import replace
from decimal import Decimal, InvalidOperation

import grid_replacement
import market_reader
from candidate_economic_validator import validate_candidate
from grid_activation_controller import GridActivationState
from grid_order_orchestrator import run_exposure_cycle
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from run_grid_trading import build_adapter
from trellis_calculator import GridGeometry

INST_ID = "XRP-USDC"
MODE = "LIVE"
K_BUY = 1
K_SELL = 1

# ---------------------------------------------------------------------------
# Policy V1 -- reprise VERBATIM de live_first_activation_xrp.py /
# diagnose_xrp_policy.py, déjà validée dans ce projet. Aucune formule
# réinventée. Utilisée UNIQUEMENT par build_optimized_config, jamais par run().
# ---------------------------------------------------------------------------
K_ATR = 15.0
DI_BULLISH_THRESHOLD = 1.2
DI_BEARISH_THRESHOLD = 0.8
PCT_MIN = 0.05
PCT_MAX = 0.30

NU = 5
NL = 5
CREATE_GEOMETRY = GridGeometry.FLEXIBLE
ALPHA = Decimal("0.95")
OPERATIONAL_MARGIN = Decimal("0.50")

# ---------------------------------------------------------------------------
# Grille active à REPRENDRE — valeurs HISTORIQUES, exactes, immuables pour
# cette instance. Celles qui ont réellement produit BUY 0.9877 × 21.044
# (exécuté) et SELL 1.0079 × 21.044 (ouvert). Jamais recalculées.
# ---------------------------------------------------------------------------
HISTORICAL_P0 = Decimal("0.9982")
HISTORICAL_GLL = Decimal("0.9469627131102811")
HISTORICAL_GUL = Decimal("1.04811")
HISTORICAL_NU = 5
HISTORICAL_NL = 5
HISTORICAL_GEOMETRY = GridGeometry.FLEXIBLE
HISTORICAL_ALPHA = Decimal("0.95")
HISTORICAL_OPERATIONAL_MARGIN = Decimal("0.50")

# Décision d'orchestration, pas une règle économique (déjà établi dans ce
# projet : la boucle sert uniquement à réconcilier l'état et matérialiser
# les niveaux devenus nécessaires -- jamais un délai de trading).
CYCLE_INTERVAL_SECONDS = 60.0

# ---------------------------------------------------------------------------
# Persistance minimale --mode auto -- exactement les 12 champs nécessaires à
# reconstruire un PreflightConfig identique, jamais plus. Écriture atomique
# uniquement (fichier temporaire + fsync + os.replace) -- un crash ne peut
# jamais produire un fichier partiellement écrit.
# ---------------------------------------------------------------------------
STATE_FILE_PATH = "active_grid_state.json"
_REQUIRED_STATE_FIELDS = (
    "inst_id", "p0", "gll", "gul", "nu", "nl", "geometry",
    "spacing_h_pct", "alpha", "operational_margin", "tick_size", "lot_size",
)


def state_file_path_for(inst_id: str) -> str:
    """Chemin de persistance déterminé par l'instrument -- aucun partage
    d'état possible entre deux instruments distincts, par construction du
    nom lui-même. N'affecte PAS STATE_FILE_PATH (conservé tel quel comme
    valeur par défaut historique de save_grid_state/load_grid_state, pour
    compatibilité stricte avec les appels existants qui le référencent
    explicitement)."""
    return f"active_grid_state_{inst_id}.json"


class StopRequested(Exception):
    """Levée par le gestionnaire SIGINT pour interrompre proprement la boucle."""


def _install_sigint_handler() -> None:
    def _handler(signum, frame):
        raise StopRequested()
    signal.signal(signal.SIGINT, _handler)


def print_grid_identity(config: PreflightConfig, instrument) -> None:
    print("========== GRILLE ACTIVE — IDENTITÉ IMMUTABLE ==========")
    print(f"P0                 : {config.p0}")
    print(f"GLL                : {config.gll}")
    print(f"GUL                : {config.gul}")
    print(f"nu                 : {config.nu}")
    print(f"nl                 : {config.nl}")
    print(f"geometry           : {config.geometry.name}")
    print(f"spacing_h_pct      : {config.spacing_h_pct}")
    print(f"alpha              : {config.alpha}")
    print(f"operational_margin : {config.operational_margin}")
    if config.allocated_capital is not None:
        print(f"allocated_capital  : {config.allocated_capital} {instrument.quote_ccy}")
    if config.q is not None:
        print(f"q                  : {config.q}")
    print(f"tick_size          : {instrument.tick_size}")
    print(f"lot_size           : {instrument.lot_size}")
    print(f"id(config)         : {id(config)}  (référence à surveiller — ne doit jamais changer)")


def build_historical_config(adapter, inst_id: str) -> PreflightConfig:
    """Construit le PreflightConfig EXACT de la grille déjà active --
    P0/GLL/GUL/nu/nl/geometry historiques et IMMUTABLES pour cette
    instance, jamais recalculés, jamais lus via get_ticker().

    Seul spacing_h_pct est lu frais (frais réels courants), même
    convention que run_grid_trading.py::build_config /
    live_first_activation_xrp.py -- jamais l'identité géométrique de la
    grille elle-même, qui n'a plus aucun besoin d'être relue une fois la
    grille déjà active.
    """
    maker_fee_rate = abs(adapter.get_fee_rates(inst_id).maker)
    return PreflightConfig(
        inst_id=inst_id,
        gul=HISTORICAL_GUL,
        gll=HISTORICAL_GLL,
        nu=HISTORICAL_NU,
        nl=HISTORICAL_NL,
        p0=HISTORICAL_P0,
        geometry=HISTORICAL_GEOMETRY,
        spacing_h_pct=maker_fee_rate,
        alpha=HISTORICAL_ALPHA,
        operational_margin=HISTORICAL_OPERATIONAL_MARGIN,
    )


class GridCreationRefused(Exception):
    """Levée si la création d'une nouvelle grille doit être refusée --
    données de marché insuffisantes, ou candidat économiquement INVALID.
    Cohérent avec la discipline déjà établie dans ce projet : refuser
    plutôt que deviner, jamais d'activation sous incertitude."""


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _simulate_policy(atr_norm_15m: float, di_ratio_15m: float) -> dict:
    """Formule V1, reprise VERBATIM de live_first_activation_xrp.py --
    aucune formule réinventée. k_atr=15, seuils DI 1.2/0.8, force bornée à
    1, multiplicateurs +10%/-5%, double clip [0.05, 0.30]."""
    base_pct = _clip(K_ATR * atr_norm_15m, PCT_MIN, PCT_MAX)

    if di_ratio_15m >= DI_BULLISH_THRESHOLD:
        force = min((di_ratio_15m - DI_BULLISH_THRESHOLD) / (2.0 - DI_BULLISH_THRESHOLD), 1.0)
        gul_mult, gll_mult = 1 + 0.10 * force, 1 - 0.05 * force
    elif di_ratio_15m <= DI_BEARISH_THRESHOLD:
        force = min((DI_BEARISH_THRESHOLD - di_ratio_15m) / DI_BEARISH_THRESHOLD, 1.0)
        gul_mult, gll_mult = 1 - 0.05 * force, 1 + 0.10 * force
    else:
        force, gul_mult, gll_mult = 0.0, 1.0, 1.0

    gul_pct = _clip(base_pct * gul_mult, PCT_MIN, PCT_MAX)
    gll_pct = _clip(base_pct * gll_mult, PCT_MIN, PCT_MAX)

    return {"base_pct": base_pct, "force": force, "gul_mult": gul_mult, "gll_mult": gll_mult, "gul_pct": gul_pct, "gll_pct": gll_pct}


def build_optimized_config(adapter, inst_id: str, allocated_capital_usdc: Decimal | None = None) -> PreflightConfig:
    """Construit le PreflightConfig d'une NOUVELLE grille, via la policy
    V1 déjà validée : MarketReader 15 min -> ATR normalisé + DI ratio ->
    GLL/GUL -> validation économique -> PreflightConfig.

    N'est appelée qu'UNE SEULE FOIS, avant la boucle -- jamais depuis
    run(). P0 est lu ici, une seule fois (get_ticker), jamais relu ensuite
    pendant la vie de cette grille.

    allocated_capital_usdc : consigne d'allocation économique explicite --
    capital USDC nouvellement mis à disposition de CETTE grille, à
    l'exclusion de la valeur de tout token déjà détenu dans le wallet
    (jamais additionné à celle-ci). None (par défaut) : repli sur l'ancien
    calcul alpha × patrimoine total -- préserve tel quel tout appelant qui
    n'a pas encore adopté ce mécanisme (l'exigence "toujours explicite à
    la création" est appliquée par l'appelant CLI, pas ici).

    Lève GridCreationRefused si les données de marché sont insuffisantes
    (MarketReaderError), si allocated_capital_usdc dépasse le quote
    réellement disponible MAINTENANT (vérifié une seule fois, ici, jamais
    revérifié une fois la grille active -- les propres ordres BUY de la
    grille immobiliseront légitimement du quote par la suite), ou si le
    candidat est économiquement INVALID -- aucune grille n'est alors
    retournée, conformément au principe déjà établi : le candidat est
    simplement rejeté, jamais ajusté silencieusement.
    """
    ticker = adapter.get_ticker(inst_id)
    p0 = ticker.last
    fees = adapter.get_fee_rates(inst_id)

    try:
        conditions = market_reader.read(adapter, inst_id)
    except market_reader.MarketReaderError as error:
        raise GridCreationRefused(f"MarketReader a refusé : {error}") from error

    sim = _simulate_policy(conditions.atr_norm_15m, conditions.di_ratio_15m)

    p0_float = float(p0)
    gul = Decimal(str(p0_float * (1 + sim["gul_pct"])))
    gll = Decimal(str(p0_float * (1 - sim["gll_pct"])))

    draft_config = PreflightConfig(
        inst_id=inst_id, gul=gul, gll=gll, nu=NU, nl=NL, p0=p0,
        geometry=CREATE_GEOMETRY, spacing_h_pct=abs(fees.maker),
        alpha=ALPHA, operational_margin=OPERATIONAL_MARGIN,
        allocated_capital=allocated_capital_usdc,
    )  # q=None -- pas encore figé, ce brouillon ne sert jamais à l'activation

    report = run_preflight(adapter, draft_config)

    if allocated_capital_usdc is not None and allocated_capital_usdc > report.balances.quote_available:
        raise GridCreationRefused(
            f"allocated_capital_usdc={allocated_capital_usdc} dépasse le quote "
            f"réellement disponible={report.balances.quote_available} -- capital "
            "insuffisant, aucune grille créée"
        )

    validation = validate_candidate(
        fees=fees, anchor_price=p0_float,
        buy_levels=tuple(float(level) for level in report.trellis[:NL]),
        sell_levels=tuple(float(level) for level in report.trellis[NL + 1:]),
    )
    if not validation.valid:
        raise GridCreationRefused(f"Candidat économiquement INVALID : {validation.diagnostic()}")

    # q figé UNE SEULE FOIS ici, à partir de l'allocation (explicite ou
    # legacy) disponible à cet instant précis -- devient l'identité
    # économique immuable de cette grille, au même titre que
    # P0/GLL/GUL/nu/nl, jamais recalculé ensuite.
    return replace(draft_config, q=report.q)


def pending_grid_state_path_for(inst_id: str) -> str:
    """Chemin de persistance d'une grille dont l'initialisation patrimoniale
    est encore en cours.

    Ce fichier contient la même identité immuable que la grille finale,
    mais son existence signifie explicitement : ne pas lancer la boucle
    normale avant que InitializationController atteigne SPOT_READY.
    """
    return f"pending_grid_state_{inst_id}.json"


def save_grid_state(config: PreflightConfig, instrument, path: str = STATE_FILE_PATH) -> None:
    """Écrit l'état de la grille active de façon ATOMIQUE : fichier
    temporaire dans le même répertoire, flush + fsync, puis os.replace().
    Un crash pendant l'écriture ne peut jamais laisser un fichier
    partiellement écrit à l'emplacement final -- os.replace() est une
    opération atomique du système de fichiers.

    Exactement les 14 champs nécessaires à reconstruire un PreflightConfig
    identique -- Decimal sérialisés en chaînes, jamais en flottant. q et
    allocated_capital font partie de l'identité immuable de la grille au
    même titre que P0/GLL/GUL/nu/nl : ne doivent JAMAIS être persistés tant
    qu'ils n'ont pas été figés (voir build_optimized_config) -- écrire
    q=None ou allocated_capital=None serait une régression silencieuse vers
    une valeur recalculée/devinée à chaque reprise.
    """
    if config.q is None:
        raise ValueError(
            "save_grid_state : config.q est None -- l'identité économique "
            "de cette grille n'a jamais été figée. Refus de persister un "
            "état de grille sans q (jamais un q deviné ou recalculé plus "
            "tard à la reprise)."
        )
    if config.allocated_capital is None:
        raise ValueError(
            "save_grid_state : config.allocated_capital est None -- la "
            "consigne d'allocation économique de cette grille n'a jamais "
            "été figée. Refus de persister un état de grille sans "
            "allocated_capital (jamais deviné ni recalculé plus tard à la "
            "reprise)."
        )
    payload = {
        "inst_id": config.inst_id,
        "p0": str(config.p0),
        "gll": str(config.gll),
        "gul": str(config.gul),
        "nu": config.nu,
        "nl": config.nl,
        "geometry": config.geometry.name,
        "spacing_h_pct": str(config.spacing_h_pct),
        "alpha": str(config.alpha),
        "operational_margin": str(config.operational_margin),
        "tick_size": str(instrument.tick_size),
        "lot_size": str(instrument.lot_size),
        "q": str(config.q),
        "allocated_capital": str(config.allocated_capital),
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".active_grid_state_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # atomique : jamais d'état partiel au chemin final
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class LegacyGridStateWithoutQ(Exception):
    """Levée par load_grid_state si un fichier de grille par ailleurs
    valide (les 12 champs historiques présents) ne contient pas de q figé
    -- signe d'une grille créée avant que q ne devienne une identité
    immuable. Cette grille ne doit JAMAIS être reprise avec un q deviné ou
    recalculé depuis les soldes courants : elle doit être arrêtée puis
    recréée explicitement sous le nouveau mécanisme (aucune migration
    automatique)."""


class LegacyGridStateWithoutAllocatedCapital(Exception):
    """Levée par load_grid_state si un fichier de grille contient déjà q
    (donc créé sous feature/q-immutable-grid-identity) mais pas
    allocated_capital -- signe d'une grille créée avant ce chantier
    d'allocation de capital explicite. Même doctrine que
    LegacyGridStateWithoutQ : reprise refusée, aucune migration
    automatique, aucun allocated_capital deviné depuis le q déjà figé ni
    depuis les soldes courants -- arrêter cette grille puis la recréer
    explicitement avec --allocated-capital-usdc."""


def load_grid_state(path: str = STATE_FILE_PATH) -> PreflightConfig | None:
    """Charge le PreflightConfig persisté, ou retourne None si le fichier
    est absent OU structurellement invalide -- NE LÈVE JAMAIS dans ces
    deux cas. Un fichier invalide doit être traité exactement comme un
    fichier absent (repli sur création optimisée), jamais comme une
    configuration historique fiable, jamais comme une erreur fatale qui
    interromprait le service.

    Exceptions délibérées à "ne lève jamais" : un fichier structurellement
    valide mais dépourvu de q (grille legacy, antérieure à
    feature/q-immutable-grid-identity) lève LegacyGridStateWithoutQ ; un
    fichier avec q mais sans allocated_capital (antérieur à ce chantier)
    lève LegacyGridStateWithoutAllocatedCapital -- ni l'un ni l'autre ne
    doit jamais être confondu avec une absence, qui déclencherait
    silencieusement une création optimisée par-dessus une grille déjà
    active sur OKX.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        if not all(field in payload for field in _REQUIRED_STATE_FIELDS):
            return None
        missing_q = "q" not in payload
        missing_allocated_capital = "allocated_capital" not in payload
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, InvalidOperation, OSError):
        return None

    if missing_q:
        raise LegacyGridStateWithoutQ(
            f"{path} ne contient pas de q figé -- grille créée avant "
            "l'immuabilité de q. Reprise refusée. Arrêter cette grille, "
            "nettoyer son état persisté, puis la recréer sous le nouveau "
            "mécanisme (aucun recalcul ni seed automatique de q)."
        )
    if missing_allocated_capital:
        raise LegacyGridStateWithoutAllocatedCapital(
            f"{path} contient q mais pas allocated_capital -- grille créée "
            "avant le chantier d'allocation de capital explicite. Reprise "
            "refusée. Arrêter cette grille, nettoyer son état persisté, "
            "puis la recréer avec --allocated-capital-usdc."
        )

    try:
        return PreflightConfig(
            inst_id=payload["inst_id"],
            gul=Decimal(payload["gul"]),
            gll=Decimal(payload["gll"]),
            nu=int(payload["nu"]),
            nl=int(payload["nl"]),
            p0=Decimal(payload["p0"]),
            geometry=GridGeometry[payload["geometry"]],
            spacing_h_pct=Decimal(payload["spacing_h_pct"]),
            alpha=Decimal(payload["alpha"]),
            operational_margin=Decimal(payload["operational_margin"]),
            q=Decimal(payload["q"]),
            allocated_capital=Decimal(payload["allocated_capital"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, InvalidOperation, OSError):
        return None


def _has_real_orders(activation) -> bool:
    """Détermine, à partir de GridActivationResult UNIQUEMENT (jamais de
    plan.decision.to_materialize, jamais une supposition), si au moins un
    ordre existe réellement sur OKX à l'issue de cette tentative
    d'activation -- placé PENDANT cet appel (placed), ou déjà présent
    avant (already_open/already_filled, cas de sondage idempotent).

    Utilisé pour découpler deux décisions désormais distinctes :
        A. l'étiquette d'exposition (ACTIVATED / PARTIALLY_EXPOSED / ERROR)
        B. la persistance + le démarrage de la boucle (ce doit continuer
           dès qu'un ordre réel existe, pour que le cycle suivant puisse
           gérer ce que l'activation initiale n'a pas terminé -- jamais
           laisser un ordre réel orphelin, sans grille persistée ni
           boucle pour le reprendre).

    activation.activation_result est None pour tout état qui n'a jamais
    atteint GridActivationController (FEE_TYPE_REJECTED,
    INITIALIZATION_INCOMPLETE, EXPOSURE_CONFLICT, ou une ERROR levée
    avant cet appel) -- dans tous ces cas, aucun ordre réel n'a pu être
    placé, retourne False sans ambiguïté.
    """
    result = activation.activation_result
    if result is None:
        return False
    return bool(result.placed or result.already_open or result.already_filled)


def run_auto_mode(
    adapter,
    live: bool,
    inst_id: str = INST_ID,
    state_path: str | None = None,
    allocated_capital_usdc: Decimal | None = None,
) -> int:
    """Cycle complet --mode auto -- chemin NOUVEAU, indépendant de resume/
    create (toutes deux inchangées, jamais appelées par cette fonction que
    build_optimized_config/GridTradingController/run, réutilisées telles quelles).

        état persisté valide ?
            OUI -> reprise EXACTE, aucun MarketReader, aucune activation,
                   allocated_capital_usdc ignoré s'il est fourni (déjà
                   figé dans l'état persisté)
            NON -> allocated_capital_usdc requis -> optimisation 15 min ->
                   validation -> activation -> persistance atomique
                   (uniquement si ACTIVATED confirmé)

    `inst_id` : instrument de cette instance -- transmis à build_optimized_config
    et à get_instrument, jamais codé en dur. Défaut "XRP-USDC" (INST_ID),
    pour compatibilité stricte avec tout appel existant qui ne le fournit pas.

    `state_path` : si None (défaut), dérivé automatiquement de `inst_id` via
    state_file_path_for -- aucun partage possible entre deux instruments.
    Si fourni explicitement (comme le font tous les appels de test déjà
    existants), utilisé tel quel, sans dérivation -- comportement antérieur
    strictement préservé.

    `allocated_capital_usdc` : consigne d'allocation économique explicite,
    requise UNIQUEMENT lorsqu'aucun état persisté valide n'existe (chemin
    de création) -- jamais de repli sur le solde USDC du wallet.

    Retourne le code de sortie de main().
    """
    if state_path is None:
        state_path = state_file_path_for(inst_id)

    instrument = adapter.get_instrument(inst_id)
    pending_path = pending_grid_state_path_for(inst_id)
    try:
        pending_config = load_grid_state(pending_path)
    except (LegacyGridStateWithoutQ, LegacyGridStateWithoutAllocatedCapital) as error:
        print(f"❌ {error}")
        print("\n========== SAFETY ==========")
        print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        return 1

    # Une initialisation patrimoniale interrompue possède priorité sur
    # active_grid_state : on doit reprendre EXACTEMENT la même grille,
    # jamais recalculer P0/GLL/GUL/allocated_capital.
    if pending_config is not None:
        config = pending_config
        print_grid_identity(config, instrument)
        print("\n========== REPRISE D'INITIALISATION (persistance pending) ==========")
        print(
            "Configuration pending valide -- aucun MarketReader, aucune "
            "optimisation, aucun recalcul de P0."
        )
        if allocated_capital_usdc is not None:
            print(
                f"ℹ️  --allocated-capital-usdc fourni ({allocated_capital_usdc}) mais "
                f"ignoré -- initialisation pending déjà figée avec "
                f"allocated_capital={config.allocated_capital}."
            )

        if not live:
            print("\n========== SAFETY ==========")
            print("READ ONLY (--live non fourni)")
            print("NO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
            print("Initialisation non reprise. Relancer avec --live.")
            return 0

        print("\n========== REPRISE DE L'INITIALISATION PATRIMONIALE ==========")
        activation = GridTradingController().run(
            adapter, config, k_buy=K_BUY, k_sell=K_SELL
        )
        print(f"État : {activation.state.name} — {activation.detail}")

        if activation.state == GridTradingState.INITIALIZATION_INCOMPLETE:
            # L'ordre INITBUY peut être encore live/partiellement rempli.
            # On conserve impérativement l'identité de la grille.
            save_grid_state(config, instrument, pending_path)
            print(
                f"Initialisation toujours en cours -- état pending conservé "
                f"({pending_path})."
            )
            return 2

        if activation.state != GridTradingState.ACTIVATED:
            print(
                "Reprise d'initialisation non terminée : état final "
                f"{activation.state.name}. Configuration pending conservée."
            )
            save_grid_state(config, instrument, pending_path)
            return 2

        # ACTIVATED : la grille est maintenant réellement exploitable.
        save_grid_state(config, instrument, state_path)
        try:
            os.unlink(pending_path)
        except FileNotFoundError:
            pass

        print(f"État actif persisté avec succès ({state_path}).")
        print("État pending supprimé -- grille désormais pleinement active.")

        print("\n========== BOUCLE ACTIVE (Ctrl-C pour arrêter) ==========")
        run(adapter, config)
        return 0

    try:
        loaded_config = load_grid_state(state_path)
    except (LegacyGridStateWithoutQ, LegacyGridStateWithoutAllocatedCapital) as error:
        print(f"❌ {error}")
        print("\n========== SAFETY ==========")
        print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        return 1

    if loaded_config is not None:
        config = loaded_config
        print_grid_identity(config, instrument)
        print("\n========== REPRISE AUTOMATIQUE (persistance) ==========")
        print("Fichier d'état valide -- AUCUN MarketReader, AUCUNE optimisation, AUCUNE activation.")
        if allocated_capital_usdc is not None:
            print(
                f"ℹ️  --allocated-capital-usdc fourni ({allocated_capital_usdc}) mais "
                f"ignoré -- grille déjà active, allocated_capital chargé depuis "
                f"l'état persisté ({config.allocated_capital})."
            )

        if not live:
            print("\n========== SAFETY ==========")
            print("READ ONLY (--live non fourni)")
            print("NO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
            print("Boucle active NON démarrée. Relancer avec --live pour démarrer réellement.")
            return 0

        print("\n========== BOUCLE ACTIVE (Ctrl-C pour arrêter) ==========")
        run(adapter, config)
        return 0

    print("Aucun état de grille persistée valide (absent ou invalide) -- création optimisée.")
    if allocated_capital_usdc is None:
        print(
            "❌ --allocated-capital-usdc absent -- impossible de créer une "
            "nouvelle grille sans capital explicitement alloué. Aucune "
            "valeur par défaut (jamais le solde USDC du wallet)."
        )
        print("\n========== SAFETY ==========")
        print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        return 1

    try:
        config = build_optimized_config(adapter, inst_id, allocated_capital_usdc=allocated_capital_usdc)
    except GridCreationRefused as error:
        print(f"❌ Création de grille refusée : {error}")
        print("\n========== SAFETY ==========")
        print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        return 1

    print_grid_identity(config, instrument)
    print("\n========== NOUVELLE GRILLE — CANDIDAT VALID (mode auto) ==========")

    if not live:
        print("\n========== SAFETY ==========")
        print("READ ONLY (--live non fourni)")
        print("NO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        print("Activation initiale NON effectuée. Boucle active NON démarrée. Aucun état persisté.")
        return 0

    print("\n========== ACTIVATION INITIALE (moteur PDF existant) ==========")
    activation = GridTradingController().run(adapter, config, k_buy=K_BUY, k_sell=K_SELL)
    print(f"État : {activation.state.name} — {activation.detail}")
    if activation.state != GridTradingState.ACTIVATED and not _has_real_orders(activation):
        print("Aucun ordre réellement placé — état NON persisté, boucle active non démarrée.")
        return 2
    if activation.state != GridTradingState.ACTIVATED:
        print(f"Activation partielle ({activation.state.name}) mais au moins un ordre réel "
              f"existe déjà sur OKX -- persistance et démarrage de la boucle pour permettre "
              f"au cycle suivant de reprendre ce qui n'a pas pu être matérialisé.")

    save_grid_state(config, instrument, state_path)
    print(f"État persisté avec succès ({state_path}).")

    print("\n========== BOUCLE ACTIVE (Ctrl-C pour arrêter) ==========")
    run(adapter, config)
    return 0


class GridQNotFrozen(ValueError):
    """Levée par run() si config.q est None -- l'identité économique de la
    grille (q) n'a jamais été figée avant l'entrée en boucle. Jamais une
    valeur à deviner ici : tout chemin de construction de config
    (build_optimized_config, ou un state chargé) doit figer q avant
    d'appeler run()."""


class GridAllocatedCapitalNotFrozen(ValueError):
    """Levée par run() si config.allocated_capital est None -- la consigne
    d'allocation économique de la grille n'a jamais été figée avant
    l'entrée en boucle. Dans les chemins réels de ce module, q figé
    implique déjà allocated_capital figé (build_optimized_config fige les
    deux ensemble, load_grid_state exige les deux) -- cette garde reste
    défensive contre un config construit à la main de façon incohérente,
    jamais une valeur à deviner ici."""


def run(
    adapter,
    config: PreflightConfig,
    *,
    k_buy: int = K_BUY,
    k_sell: int = K_SELL,
    cycle_interval: float = CYCLE_INTERVAL_SECONDS,
    max_cycles: int | None = None,
    sleep_fn=time.sleep,
) -> int:
    """Boucle active. `config` n'est JAMAIS recalculé ici -- réutilisé tel
    quel à chaque itération, exactement l'objet reçu en paramètre.

    `max_cycles`/`sleep_fn` : usage test uniquement (None = boucle réelle,
    infinie, jusqu'à SIGINT). En usage réel, ne jamais fournir max_cycles.

    Retourne le nombre de cycles réellement exécutés.

    Lève GridQNotFrozen si config.q est None, ou GridAllocatedCapitalNotFrozen
    si config.allocated_capital est None -- refuse de démarrer plutôt que
    de laisser run_preflight recalculer/deviner une identité économique
    incomplète à chaque cycle depuis les soldes courants.
    """
    if config.q is None:
        raise GridQNotFrozen(
            "config.q est None : l'identité économique de cette grille n'a "
            "jamais été figée. run() refuse de démarrer plutôt que de "
            "recalculer q à partir des soldes courants à chaque cycle."
        )
    if config.allocated_capital is None:
        raise GridAllocatedCapitalNotFrozen(
            "config.allocated_capital est None : la consigne d'allocation "
            "économique de cette grille n'a jamais été figée. run() refuse "
            "de démarrer."
        )
    churn_state_path = state_file_path_for(config.inst_id)
    cycles_run = 0
    try:
        while max_cycles is None or cycles_run < max_cycles:
            # `config` n'est recalculé QUE par grid_replacement.py, et
            # UNIQUEMENT lorsqu'un remplacement complet de grille vient
            # d'aboutir (nouvelle identité déjà persistée et activée) --
            # jamais par run_preflight, qui continue de recevoir `config`
            # tel quel à chaque cycle (report peut varier légitimement --
            # soldes réels, spot_covered -- le treillis et q non, sauf
            # remplacement).
            config, materialize = grid_replacement.run_replacement_check(
                adapter, config, k_buy, k_sell, churn_state_path,
            )
            if not materialize:
                # Remplacement en cours (annulation ou activation pas
                # encore abouties) -- aucune grille officiellement active
                # ce cycle, donc aucune matérialisation possible.
                cycles_run += 1
                if max_cycles is None:
                    sleep_fn(cycle_interval)
                continue

            report = run_preflight(adapter, config)
            orchestration_result = run_exposure_cycle(
                adapter, report, k_buy, k_sell, churn_state_path=churn_state_path,
            )
            if orchestration_result.activation_result.state != GridActivationState.ACTIVE:
                print(
                    f"[{orchestration_result.activation_result.state.name}] "
                    f"{orchestration_result.activation_result.detail}"
                )
            cycles_run += 1
            if max_cycles is None:
                sleep_fn(cycle_interval)
    except StopRequested:
        print("\n========== ARRÊT DEMANDÉ (SIGINT) ==========")
        print("Grille active arrêtée proprement.")
        print("Aucune nouvelle grille recalculée. Aucun ordre modifié.")
    return cycles_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reprise ou création d'une grille active.")
    parser.add_argument("--inst-id", dest="inst_id", default=INST_ID, help=f"Instrument à trader. Défaut : {INST_ID} (compatibilité stricte, comportement inchangé si non fourni).")
    parser.add_argument("--mode", choices=("resume", "create", "auto"), default="resume", help="resume (défaut, inchangé) = grille historique déjà active ; create = nouvelle grille via l'optimiseur ; auto = reprise depuis persistance si valide, sinon création optimisée (usage systemd).")
    parser.add_argument("--live", action="store_true", help="Entre réellement dans la boucle active. Absent = READ-ONLY, affiche l'identité puis s'arrête.")
    parser.add_argument(
        "--allocated-capital-usdc", dest="allocated_capital_usdc", default=None,
        help="Consigne d'allocation économique : capital USDC nouvellement mis à disposition d'une "
             "NOUVELLE grille, à l'exclusion de la valeur de tout token déjà détenu dans le wallet. "
             "Requis pour --mode create. Requis pour --mode auto UNIQUEMENT lors d'une création "
             "(aucun état persisté valide) -- ignoré, avec message explicite, si une grille est reprise "
             "depuis l'état persisté. Aucune valeur par défaut (jamais le solde USDC du wallet).",
    )
    return parser.parse_args()


def _parse_allocated_capital_usdc(raw: str) -> Decimal:
    """Parse strict de --allocated-capital-usdc -- jamais un repli sur une
    valeur devinée en cas d'entrée invalide."""
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise ValueError(f"--allocated-capital-usdc invalide : {raw!r} n'est pas un nombre décimal") from None
    if value <= 0:
        raise ValueError("--allocated-capital-usdc doit être strictement positif")
    return value


def main() -> int:
    args = parse_args()
    _install_sigint_handler()

    adapter = build_adapter(MODE)

    allocated_capital_usdc = None
    if args.allocated_capital_usdc is not None:
        try:
            allocated_capital_usdc = _parse_allocated_capital_usdc(args.allocated_capital_usdc)
        except ValueError as error:
            print(f"❌ {error}")
            print("\n========== SAFETY ==========")
            print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
            return 1

    if args.mode == "auto":
        return run_auto_mode(adapter, args.live, inst_id=args.inst_id, allocated_capital_usdc=allocated_capital_usdc)

    if args.mode == "create":
        if allocated_capital_usdc is None:
            print(
                "❌ --mode create requiert --allocated-capital-usdc (capital USDC "
                "explicitement alloué à cette nouvelle grille). Aucune valeur par "
                "défaut (jamais le solde USDC du wallet)."
            )
            print("\n========== SAFETY ==========")
            print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
            return 1
        try:
            config = build_optimized_config(adapter, args.inst_id, allocated_capital_usdc=allocated_capital_usdc)
        except GridCreationRefused as error:
            print(f"❌ Création de grille refusée : {error}")
            print("\n========== SAFETY ==========")
            print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
            return 1
    else:
        # resume -- HISTORICAL_* volontairement NON généralisées (spécifiques
        # à l'ancien état XRP, cf. contrainte 6) : ce mode reste correct
        # uniquement avec --inst-id XRP-USDC (le défaut) ; un autre --inst-id
        # ici produirait un inst_id incohérent avec des GLL/GUL XRP -- jamais
        # utilisé en production H24 (remplacé par --mode auto), signalé tel quel.
        config = build_historical_config(adapter, args.inst_id)  # reprise -- inchangé

    instrument = adapter.get_instrument(args.inst_id)
    print_grid_identity(config, instrument)

    if args.mode == "create":
        print("\n========== NOUVELLE GRILLE — CANDIDAT VALID ==========")
        print("Config immuable pour toute la vie de cette grille.")
    else:
        print("\n========== REPRISE DE GRILLE ACTIVE ==========")
        print("Grille déjà active sur OKX -- aucune activation initiale effectuée.")
        print("GridTradingController jamais appelé. build_config() jamais appelé.")

    if not args.live:
        print("\n========== SAFETY ==========")
        print("READ ONLY (--live non fourni)")
        print("NO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        if args.mode == "create":
            print("Activation initiale NON effectuée. Boucle active NON démarrée.")
        else:
            print("Boucle active NON démarrée.")
        print("Relancer avec --live pour démarrer réellement.")
        return 0

    if args.mode == "create":
        # Activation via le moteur PDF EXISTANT, inchangé -- même objet
        # `config` que celui construit ci-dessus par build_optimized_config,
        # jamais reconstruit, jamais recalculé.
        print("\n========== ACTIVATION INITIALE (moteur PDF existant) ==========")
        activation = GridTradingController().run(adapter, config, k_buy=K_BUY, k_sell=K_SELL)
        print(f"État : {activation.state.name} — {activation.detail}")

        # INITIALIZATION_INCOMPLETE / BUYING n'est PAS un échec de création
        # de grille : l'identité est valide, mais l'achat patrimonial initial
        # n'est pas encore terminé. On persiste cette identité dans un fichier
        # distinct afin qu'un redémarrage puisse reprendre exactement la même
        # grille sans recalculer P0.
        if activation.state == GridTradingState.INITIALIZATION_INCOMPLETE:
            pending_path = pending_grid_state_path_for(args.inst_id)
            save_grid_state(config, instrument, pending_path)
            print(
                f"Initialisation non terminée -- configuration pending "
                f"persistée ({pending_path})."
            )
            print(
                "Aucun ordre de grille envoyé. Le prochain --mode auto "
                "reprendra cette même grille."
            )
            return 2

        if activation.state != GridTradingState.ACTIVATED and not _has_real_orders(activation):
            print("Aucun ordre réellement placé — état NON persisté, boucle active non démarrée.")
            return 2
        if activation.state != GridTradingState.ACTIVATED:
            print(f"Activation partielle ({activation.state.name}) mais au moins un ordre réel "
                  f"existe déjà sur OKX -- persistance et démarrage de la boucle pour permettre "
                  f"au cycle suivant de reprendre ce qui n'a pas pu être matérialisé.")

        # Persistance -- SEULEMENT si ACTIVATED ou si au moins un ordre réel
        # existe déjà (_has_real_orders ci-dessus), jamais avant, jamais si
        # aucun ordre réel n'existe. Même mécanisme atomique que
        # run_auto_mode() (tempfile+fsync+os.replace, save_grid_state()
        # inchangée) -- reproduit ici, jamais dupliqué en logique.
        save_grid_state(config, instrument, state_file_path_for(args.inst_id))
        pending_path = pending_grid_state_path_for(args.inst_id)
        try:
            os.unlink(pending_path)
        except FileNotFoundError:
            pass
        print(f"État persisté avec succès ({state_file_path_for(args.inst_id)}).")

    print("\n========== BOUCLE ACTIVE (Ctrl-C pour arrêter) ==========")
    try:
        run(adapter, config)
    except GridQNotFrozen as error:
        print(f"❌ {error}")
        print("\n========== SAFETY ==========")
        print("READ ONLY\nNO ORDER PLACED\nNO ORDER CANCELLED\nNO ORDER MODIFIED")
        print(
            "--mode resume (build_historical_config) ne fige jamais q -- "
            "chemin déjà noté comme non utilisé en production H24, "
            "remplacé par --mode auto. Ajouter un q historique explicite "
            "à build_historical_config si ce chemin doit revivre."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
