"""grid_replacement.py — remplacement automatique de la grille active
lorsqu'une borne (GUL ou GLL) est durablement dépassée, validé par 4
rapports de conception successifs (extension par générations rejetée au
profit du remplacement complet ; jonction mathématique P0_new=borne_old ;
comparaison quantitative des formules de largeur).

Mécanisme, EXACTEMENT celui validé, rien de plus :

    N=3 observations consécutives hors [GLL, GUL]
        -> ancienne grille arrêtée (aucun nouveau cycle matérialisé)
        -> ordres ouverts de l'ancienne grille annulés, vérifiés
        -> P0_new = GUL_old (dépassement haut) ou GLL_old (dépassement bas)
           -- JAMAIS le prix courant
        -> largeur du côté franchi = largeur ATR normale + déplacement déjà
           réalisé (gul_pct_new = gul_pct_ATR + déplacement_pct) ;
           l'autre côté conserve sa largeur ATR normale, inchangée
        -> q recalculé (jamais copié de l'ancienne grille)
        -> protection anti-churn de la nouvelle grille : repart à zéro
           (conséquence directe de l'écrasement complet du fichier d'état,
           §PERSISTANCE ci-dessous -- pas une logique séparée à maintenir)
        -> nouvelle grille activée, devient l'UNIQUE grille opérationnelle
        -> ancienne grille conservée en historique, jamais supprimée,
           jamais réactivée telle quelle

Une seule grille active à la fois -- jamais de coexistence. Un nouveau
remplacement (G1 -> G2) est un comportement normal, pas une erreur, si le
marché dépasse réellement la nouvelle borne.

--------------------------------------------------------------------------
POURQUOI CE MODULE DUPLIQUE DEUX PETITS ÉLÉMENTS DE run_active_grid.py
--------------------------------------------------------------------------
run_active_grid.py doit importer ce module (pour appeler
run_replacement_check depuis sa boucle run()) -- toute importation en sens
inverse créerait un cycle. Deux éléments sont donc dupliqués ICI plutôt
qu'importés :

  1. La formule de largeur ATR (_atr_width_pct) -- copie exacte de
     run_active_grid._simulate_policy (même K_ATR=15, mêmes seuils DI
     1.2/0.8, mêmes multiplicateurs +10 %/-5 %, même double clip
     [0.05, 0.30]). Toute évolution de l'une doit être répercutée
     manuellement sur l'autre. Précédent déjà assumé dans ce dépôt pour un
     cas analogue : build_optimized_config (docstring : "Duplication
     assumée (troisième occurrence) : aucune consolidation entre fichiers
     n'a été faite").

  2. Le format d'écriture des 14 champs de configuration -- copie du
     format de run_active_grid.save_grid_state (mêmes noms de champs,
     même sérialisation Decimal->chaîne, même mécanisme atomique
     tempfile+fsync+os.replace). Toute évolution du format doit être
     répercutée des deux côtés.

--------------------------------------------------------------------------
CE QUE CE MODULE NE FAIT JAMAIS
--------------------------------------------------------------------------
- Ne modifie jamais gi, le mécanisme de réarmement au même gi, K_BUY/K_SELL,
  la géométrie d'une grille existante, ni aucune formule du PDF.
- Ne fait jamais P0_new = prix courant -- toujours la borne franchie,
  gelée depuis la configuration encore intacte de l'ancienne grille.
- N'introduit aucun coefficient arbitraire nouveau : le seul paramètre est
  N (REPLACEMENT_N=3, déjà validé), la largeur réutilise intégralement la
  formule ATR déjà en production, sans coefficient additionnel.
- Ne laisse jamais deux grilles matérialiser des ordres simultanément --
  la matérialisation de l'ancienne grille est suspendue dès l'instant où
  le seuil est atteint (ACTIVE -> CANCELLING), avant même la première
  tentative d'annulation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN

import market_reader
from candidate_economic_validator import validate_candidate
from grid_preflight import PreflightConfig, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from trellis_calculator import GridGeometry

# Seul paramètre du mécanisme de déclenchement -- même discipline que
# CHURN_PROTECTION_N (churn_protection.py) : un compteur d'observations
# consécutives, rien de temporel, rien lié à une distance ou à l'ATR.
REPLACEMENT_N = 3

# ---------------------------------------------------------------------------
# Duplication assumée de run_active_grid._simulate_policy -- voir note de
# module ci-dessus.
# ---------------------------------------------------------------------------
_K_ATR = 15.0
_DI_BULLISH_THRESHOLD = 1.2
_DI_BEARISH_THRESHOLD = 0.8
_PCT_MIN = 0.05
_PCT_MAX = 0.30


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _atr_width_pct(atr_norm_15m: float, di_ratio_15m: float) -> dict:
    """Copie exacte de run_active_grid._simulate_policy -- voir note de
    module. Retourne {"gul_pct": ..., "gll_pct": ...}, la largeur ATR
    normale de chaque côté, AVANT tout ajustement lié à un remplacement."""
    base_pct = _clip(_K_ATR * atr_norm_15m, _PCT_MIN, _PCT_MAX)

    if di_ratio_15m >= _DI_BULLISH_THRESHOLD:
        force = min((di_ratio_15m - _DI_BULLISH_THRESHOLD) / (2.0 - _DI_BULLISH_THRESHOLD), 1.0)
        gul_mult, gll_mult = 1 + 0.10 * force, 1 - 0.05 * force
    elif di_ratio_15m <= _DI_BEARISH_THRESHOLD:
        force = min((_DI_BEARISH_THRESHOLD - di_ratio_15m) / _DI_BEARISH_THRESHOLD, 1.0)
        gul_mult, gll_mult = 1 - 0.05 * force, 1 + 0.10 * force
    else:
        gul_mult, gll_mult = 1.0, 1.0

    return {
        "gul_pct": _clip(base_pct * gul_mult, _PCT_MIN, _PCT_MAX),
        "gll_pct": _clip(base_pct * gll_mult, _PCT_MIN, _PCT_MAX),
    }


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


# ---------------------------------------------------------------------------
# État persisté -- même fichier que la configuration de grille
# (active_grid_state_<inst_id>.json), sous une clé de premier niveau dédiée
# "replacement", par fusion stricte -- même schéma et même mécanisme
# atomique que churn_protection.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplacementState:
    """État du mécanisme de remplacement pour l'instrument concerné.

    lifecycle_status :
        "ACTIVE"     -- grille normale, compteurs de dépassement actifs.
        "CANCELLING" -- seuil atteint, annulation des ordres de l'ancienne
                        grille en cours (retentée tant que non vérifiée).
        "ACTIVATING" -- ordres annulés, nouvelle configuration calculée et
                        GELÉE (pending_config), activation en cours
                        (retentée avec la MÊME configuration tant que non
                        ACTIVATED -- ne jamais recalculer entre deux
                        tentatives, sous peine de dériver d'un cycle à
                        l'autre).
    direction : "UP" ou "DOWN", None si ACTIVE.
    pending_config : configuration gelée de la nouvelle grille, présente
        uniquement en ACTIVATING.
    """

    lifecycle_status: str
    direction: str | None
    above_streak: int
    below_streak: int
    pending_config: PreflightConfig | None = None

    def __post_init__(self):
        if self.lifecycle_status not in ("ACTIVE", "CANCELLING", "ACTIVATING"):
            raise ValueError(f"lifecycle_status invalide : {self.lifecycle_status!r}")
        if self.lifecycle_status == "ACTIVE" and self.direction is not None:
            raise ValueError("ACTIVE exige direction=None")
        if self.lifecycle_status != "ACTIVE" and self.direction not in ("UP", "DOWN"):
            raise ValueError("CANCELLING/ACTIVATING exigent direction='UP' ou 'DOWN'")
        if self.lifecycle_status == "ACTIVATING" and self.pending_config is None:
            raise ValueError("ACTIVATING exige pending_config non None")
        if self.lifecycle_status != "ACTIVATING" and self.pending_config is not None:
            raise ValueError("pending_config ne doit être défini qu'en ACTIVATING")


def fresh_state() -> ReplacementState:
    return ReplacementState("ACTIVE", None, 0, 0, None)


def _serialize_config(config: PreflightConfig) -> dict:
    return {
        "inst_id": config.inst_id, "p0": str(config.p0), "gll": str(config.gll),
        "gul": str(config.gul), "nu": config.nu, "nl": config.nl,
        "geometry": config.geometry.name, "spacing_h_pct": str(config.spacing_h_pct),
        "alpha": str(config.alpha), "operational_margin": str(config.operational_margin),
        "q": str(config.q) if config.q is not None else None,
        "allocated_capital": str(config.allocated_capital) if config.allocated_capital is not None else None,
    }


def _deserialize_config(payload: dict) -> PreflightConfig:
    return PreflightConfig(
        inst_id=payload["inst_id"], gul=Decimal(payload["gul"]), gll=Decimal(payload["gll"]),
        nu=int(payload["nu"]), nl=int(payload["nl"]), p0=Decimal(payload["p0"]),
        geometry=GridGeometry[payload["geometry"]], spacing_h_pct=Decimal(payload["spacing_h_pct"]),
        alpha=Decimal(payload["alpha"]), operational_margin=Decimal(payload["operational_margin"]),
        q=Decimal(payload["q"]) if payload["q"] is not None else None,
        allocated_capital=Decimal(payload["allocated_capital"]) if payload["allocated_capital"] is not None else None,
    )


def _serialize_state(state: ReplacementState) -> dict:
    return {
        "lifecycle_status": state.lifecycle_status,
        "direction": state.direction,
        "above_streak": state.above_streak,
        "below_streak": state.below_streak,
        "pending_config": _serialize_config(state.pending_config) if state.pending_config is not None else None,
    }


def _deserialize_state(payload: dict) -> ReplacementState:
    return ReplacementState(
        lifecycle_status=payload["lifecycle_status"],
        direction=payload["direction"],
        above_streak=payload["above_streak"],
        below_streak=payload["below_streak"],
        pending_config=_deserialize_config(payload["pending_config"]) if payload.get("pending_config") is not None else None,
    )


def load_replacement_state(path: str) -> ReplacementState:
    """Lit la section replacement du fichier de grille. Retourne l'état
    initial (ACTIVE, compteurs à zéro) si le fichier est absent, invalide,
    ou dépourvu de cette section -- NE LÈVE JAMAIS, même doctrine que
    load_churn_protection."""
    if not os.path.exists(path):
        return fresh_state()
    try:
        with open(path) as f:
            payload = json.load(f)
        raw = payload.get("replacement")
        if raw is None:
            return fresh_state()
        return _deserialize_state(raw)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError):
        return fresh_state()


def save_replacement_state(state: ReplacementState, path: str) -> None:
    """Écrit l'état de remplacement dans le fichier de grille existant,
    PAR FUSION STRICTE -- préserve tous les autres champs (configuration,
    churn_protection) strictement intacts. Écriture atomique, identique à
    save_churn_protection."""
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing["replacement"] = _serialize_state(state)

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


def _persist_new_active_config(config: PreflightConfig, instrument, path: str) -> None:
    """Écrit la nouvelle grille comme identité ACTIVE -- ÉCRASEMENT COMPLET
    du fichier (même comportement que run_active_grid.save_grid_state,
    format identique, dupliqué ici pour éviter tout import circulaire, cf.
    note de module). Cet écrasement réinitialise gratuitement
    churn_protection et replacement (absents du nouveau payload = état par
    défaut au prochain chargement) -- exactement la règle 11 demandée
    (protection anti-churn repart à zéro), sans logique séparée à
    maintenir."""
    if config.q is None or config.allocated_capital is None:
        raise ValueError("_persist_new_active_config exige q et allocated_capital déjà figés")
    payload = {
        "inst_id": config.inst_id, "p0": str(config.p0), "gll": str(config.gll),
        "gul": str(config.gul), "nu": config.nu, "nl": config.nl,
        "geometry": config.geometry.name, "spacing_h_pct": str(config.spacing_h_pct),
        "alpha": str(config.alpha), "operational_margin": str(config.operational_margin),
        "tick_size": str(instrument.tick_size), "lot_size": str(instrument.lot_size),
        "q": str(config.q), "allocated_capital": str(config.allocated_capital),
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".active_grid_state_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_history(retired_config: PreflightConfig, path_prefix: str) -> None:
    """Ajoute l'identité complète de la grille retirée à un journal
    append-only (jamais réécrit, jamais tronqué) -- audit/historique,
    règle 10/§7. Toujours appelé AVANT _persist_new_active_config : en cas
    d'interruption entre les deux, l'ancienne identité reste de toute
    façon intacte dans le fichier vivant (non encore écrasé) -- aucune
    perte possible, au pire une entrée d'historique dupliquée au prochain
    essai (inoffensif pour un journal d'audit)."""
    history_path = f"{path_prefix}_history.jsonl"
    entry = _serialize_config(retired_config)
    with open(history_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Logique pure -- comptage des dépassements consécutifs.
# ---------------------------------------------------------------------------


def update_breach_streak(
    state: ReplacementState, price: Decimal, gll: Decimal, gul: Decimal, N: int = REPLACEMENT_N,
) -> ReplacementState:
    """PURE. N'est appelée que lorsque state.lifecycle_status == 'ACTIVE'.
    Incrémente le compteur du côté concerné, réinitialise l'autre et
    celui-ci si le prix est revenu dans la zone. Bascule vers CANCELLING +
    direction dès que N est atteint -- c'est la SEULE transition hors de
    ACTIVE produite par cette fonction."""
    if price > gul:
        above, below = state.above_streak + 1, 0
    elif price < gll:
        above, below = 0, state.below_streak + 1
    else:
        above, below = 0, 0

    if above >= N:
        return ReplacementState("CANCELLING", "UP", above, below, None)
    if below >= N:
        return ReplacementState("CANCELLING", "DOWN", above, below, None)
    return ReplacementState("ACTIVE", None, above, below, None)


# ---------------------------------------------------------------------------
# Annulation des ordres de l'ancienne grille.
# ---------------------------------------------------------------------------


def cancel_active_generation_orders(adapter, inst_id: str) -> tuple[str, ...]:
    """Annule tous les ordres de grille actuellement ouverts pour cet
    instrument. Une seule grille étant active à la fois (règle 1), tout
    ordre préfixé "GRID" ouvert appartient nécessairement à la génération
    sortante -- inutile de reconstruire son identité pour les identifier."""
    open_orders = adapter.list_open_orders(inst_id)
    cancelled = []
    for order in open_orders:
        if order.client_order_id.startswith("GRID"):
            adapter.cancel_order(inst_id, client_order_id=order.client_order_id)
            cancelled.append(order.client_order_id)
    return tuple(cancelled)


def _orders_still_open(adapter, inst_id: str) -> bool:
    open_orders = adapter.list_open_orders(inst_id)
    return any(order.client_order_id.startswith("GRID") for order in open_orders)


# ---------------------------------------------------------------------------
# Calcul de la nouvelle grille.
# ---------------------------------------------------------------------------


def compute_new_config(
    adapter, old_config: PreflightConfig, direction: str, allocated_capital_new: Decimal, instrument,
) -> PreflightConfig:
    """Construit la configuration (q=None, pas encore figé) de la nouvelle
    grille. Fait 2 appels réseau (market_reader.read pour ATR/DI,
    get_ticker pour le déplacement déjà réalisé) -- comme
    build_optimized_config, ce module ne sépare pas strictement I/O et
    calcul pour cette fonction précise, par cohérence avec le patron déjà
    en place.

    RÈGLES APPLIQUÉES (chantier validé) :
      - P0_new = borne franchie (old_config.gul ou old_config.gll),
        JAMAIS le prix courant (règle 5) -- arrondi au tick_size, seule
        opération appliquée à cette valeur.
      - largeur du côté franchi = largeur ATR normale + déplacement déjà
        réalisé (règle 6) ; l'autre côté = largeur ATR normale seule,
        inchangée (règles 7-8).
      - aucun coefficient nouveau : réutilise intégralement _atr_width_pct
        (copie de _simulate_policy), sans pondération additionnelle.
    """
    if direction == "UP":
        borne_old = old_config.gul
    elif direction == "DOWN":
        borne_old = old_config.gll
    else:
        raise ValueError(f"direction inconnue : {direction!r}")

    p0_new = _round_down(borne_old, instrument.tick_size)

    ticker = adapter.get_ticker(old_config.inst_id)
    prix_confirmation = ticker.last

    conditions = market_reader.read(adapter, old_config.inst_id)
    sim = _atr_width_pct(conditions.atr_norm_15m, conditions.di_ratio_15m)

    deplacement_pct = abs(float(prix_confirmation) - float(borne_old)) / float(borne_old)

    if direction == "UP":
        gul_pct = sim["gul_pct"] + deplacement_pct
        gll_pct = sim["gll_pct"]
    else:
        gll_pct = sim["gll_pct"] + deplacement_pct
        gul_pct = sim["gul_pct"]

    gul = Decimal(str(float(p0_new) * (1 + gul_pct)))
    gll = Decimal(str(float(p0_new) * (1 - gll_pct)))

    fees = adapter.get_fee_rates(old_config.inst_id)

    return PreflightConfig(
        inst_id=old_config.inst_id, gul=gul, gll=gll,
        nu=old_config.nu, nl=old_config.nl, p0=p0_new,
        geometry=old_config.geometry, spacing_h_pct=abs(fees.maker),
        alpha=old_config.alpha, operational_margin=old_config.operational_margin,
        allocated_capital=allocated_capital_new,
    )


# ---------------------------------------------------------------------------
# Orchestration -- point d'entrée appelé depuis run_active_grid.run().
# ---------------------------------------------------------------------------


def run_replacement_check(
    adapter, config: PreflightConfig, k_buy: int, k_sell: int, state_path: str,
) -> tuple[PreflightConfig, bool]:
    """Point d'entrée unique, appelé depuis run() à CHAQUE cycle, avant
    run_preflight/run_exposure_cycle.

    Retourne (config_à_utiliser_ce_cycle, matérialiser_ce_cycle) :
      - (config inchangé, True)  : rien à faire, cycle normal.
      - (config inchangé, False) : remplacement en cours (annulation ou
        activation pas encore abouties) -- l'appelant NE DOIT PAS appeler
        run_exposure_cycle ce cycle (aucune matérialisation possible tant
        qu'aucune grille n'est officiellement active).
      - (nouveau_config, True)   : le remplacement vient d'aboutir --
        nouveau_config est DÉJÀ persisté et activé, l'appelant poursuit
        directement avec lui.
    """
    instrument = adapter.get_instrument(config.inst_id)

    state = load_replacement_state(state_path)

    if state.lifecycle_status == "ACTIVE":
        ticker = adapter.get_ticker(config.inst_id)
        new_state = update_breach_streak(state, ticker.last, config.gll, config.gul)
        save_replacement_state(new_state, state_path)
        if new_state.lifecycle_status == "ACTIVE":
            return config, True
        state = new_state  # bascule immédiate vers CANCELLING, même cycle

    if state.lifecycle_status == "CANCELLING":
        cancel_active_generation_orders(adapter, config.inst_id)
        if _orders_still_open(adapter, config.inst_id):
            return config, False  # retente l'annulation au cycle suivant

        balances = adapter.get_balances(instrument.base_ccy, instrument.quote_ccy)
        ticker = adapter.get_ticker(config.inst_id)
        allocated_capital_new = balances.quote_available + balances.base_available * ticker.last

        draft = compute_new_config(adapter, config, state.direction, allocated_capital_new, instrument)
        report = run_preflight(adapter, draft)

        validation = validate_candidate(
            fees=adapter.get_fee_rates(config.inst_id), anchor_price=float(draft.p0),
            buy_levels=tuple(float(level) for level in report.trellis[:draft.nl]),
            sell_levels=tuple(float(level) for level in report.trellis[draft.nl + 1:]),
        )
        if not validation.valid:
            # Conditions de marché actuelles ne permettent pas une grille
            # valide -- reste en CANCELLING (ordres déjà annulés), retente
            # le calcul au cycle suivant. Jamais d'activation d'une
            # config invalide.
            return config, False

        frozen = replace(draft, q=report.q)
        new_state = ReplacementState("ACTIVATING", state.direction, 0, 0, frozen)
        save_replacement_state(new_state, state_path)
        state = new_state  # bascule immédiate vers ACTIVATING, même cycle

    if state.lifecycle_status == "ACTIVATING":
        frozen = state.pending_config
        activation = GridTradingController().run(adapter, frozen, k_buy=k_buy, k_sell=k_sell)
        if activation.state != GridTradingState.ACTIVATED:
            return config, False  # retente au cycle suivant, MÊME config gelée

        append_history(config, state_path.rsplit(".json", 1)[0])
        _persist_new_active_config(frozen, instrument, state_path)
        return frozen, True

    return config, True
