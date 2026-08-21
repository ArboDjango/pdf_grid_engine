"""MarketReader — observation pure du marché, aucune décision de grille.

Produit MarketConditions(atr_norm_15m, di_ratio_15m) à partir de bougies
OKX 15 minutes publiques (OkxSpotAdapter.get_klines). Réutilise
exclusivement les formules déjà présentes dans le bot Gate
(ta.volatility.average_true_range, ta.trend.adx_pos, ta.trend.adx_neg) —
aucune formule économique ou statistique n'est réécrite ici.

Frontière stricte, volontairement étroite : ce module ne connaît ni
PreflightConfig, ni GridParameterPolicy, ni CandidateEconomicValidator,
ni GridEconomicValidator, ni Cell, ni GridActivationController, ni
GridOrderOrchestrator, ni aucune stratégie d'exécution. Il ne produit
jamais Gul, Gll, nu ou nl — ces quatre paramètres appartiennent
exclusivement au futur GridParameterPolicy (chantier séparé, non
commencé ici).

Lecture ponctuelle uniquement : ce module ne construit ni base
historique, ni cache, ni collecteur permanent, ni infrastructure de
stockage — chaque appel à read() relit intégralement ce dont il a
besoin, sans conserver aucun état entre deux appels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import ta

from okx_spot_adapter import Candle

# Fenêtre de calcul des trois indicateurs — identique à celle déjà
# utilisée par le bot Gate (window=14), jamais recalculée différemment
# entre ATR et DI+/DI-.
INDICATOR_WINDOW = 14

# Nombre de bougies demandées à OKX — reprise directe de la pratique déjà
# éprouvée par le bot Gate pour son calcul 15 minutes (exchange.get_klines
# (SYMBOL, KLINE_15M, 50)) — pas une valeur arbitraire.
CANDLES_REQUESTED = 50

# Seuil en-deçà duquel les données sont jugées insuffisantes : fixé
# exactement à CANDLES_REQUESTED, pas au minimum théorique (window+1=15).
# Justification empirique, pas seulement théorique : à 15 bougies, DI+ et
# DI- sont tous deux calculés à EXACTEMENT 0.0 par la bibliothèque ta --
# une valeur finie en apparence, mais un artefact d'amorçage insuffisant,
# pas un signal réel (vérifié empiriquement avant cette implémentation).
# Exiger le compte plein demandé (déjà la pratique réelle de Gate) évite
# d'introduire un second seuil, distinct et non justifié, entre le minimum
# théorique et la pratique réelle.
MIN_CANDLES_REQUIRED = CANDLES_REQUESTED

BAR = "15m"


class MarketReaderError(RuntimeError):
    """Les données de marché sont insuffisantes ou invalides pour produire une observation fiable."""


@dataclass(frozen=True)
class MarketConditions:
    """Deux observations de marché, rien de plus. Ne contient ni Gul, ni
    Gll, ni nu, ni nl, ni ADX, ni volume, ni spread, ni aucun score
    composite — volontairement minimal."""
    atr_norm_15m: float
    di_ratio_15m: float


def read(adapter, inst_id: str) -> MarketConditions:
    """Lit CANDLES_REQUESTED bougies OKX 15 minutes et produit MarketConditions.

    `adapter` doit exposer get_klines(inst_id, bar, limit) -> tuple[Candle, ...]
    (OkxSpotAdapter, ou tout objet compatible pour les tests). Lecture
    ponctuelle : aucun état n'est conservé après le retour de cet appel.
    """
    candles = adapter.get_klines(inst_id, bar=BAR, limit=CANDLES_REQUESTED)
    return _compute_conditions(candles)


def _compute_conditions(candles: tuple[Candle, ...]) -> MarketConditions:
    if len(candles) < MIN_CANDLES_REQUIRED:
        raise MarketReaderError(
            f"Données insuffisantes : {len(candles)} bougie(s) reçue(s), "
            f"{MIN_CANDLES_REQUIRED} requises."
        )

    # OKX retourne les bougies en ordre antichronologique (la plus récente
    # en premier) -- normalisation EXPLICITE en ordre chronologique
    # croissant, jamais une dépendance implicite à l'ordre fourni par
    # l'API (l'ordre de get_klines() lui-même reste un miroir fidèle de
    # la réponse OKX brute — c'est ici, et seulement ici, que la
    # normalisation a lieu).
    ordered = tuple(sorted(candles, key=lambda c: c.ts))

    high = pd.Series([float(c.high) for c in ordered])
    low = pd.Series([float(c.low) for c in ordered])
    close = pd.Series([float(c.close) for c in ordered])

    # 1. Finitude d'abord — jamais de comparaison ni d'arithmétique sur
    # une valeur potentiellement NaN/Infinity avant cette vérification.
    # (Un close NaN comparé via `<= 0` sur un Decimal lève
    # decimal.InvalidOperation, pas une MarketReaderError propre — c'est
    # précisément ce que cet ordre évite.)
    _require_all_finite(high, "high")
    _require_all_finite(low, "low")
    _require_all_finite(close, "close")

    # 2. Dernier close, seulement une fois la finitude garantie.
    last_close = ordered[-1].close

    # 3. Positivité stricte, seulement maintenant.
    if last_close <= 0:
        raise MarketReaderError(f"Dernier close invalide (doit être strictement positif) : {last_close}")

    # 4. ATR/DI, seulement après toutes les vérifications ci-dessus.
    atr_series = ta.volatility.average_true_range(high, low, close, window=INDICATOR_WINDOW)
    dip_series = ta.trend.adx_pos(high, low, close, window=INDICATOR_WINDOW)
    dim_series = ta.trend.adx_neg(high, low, close, window=INDICATOR_WINDOW)

    atr = _require_finite_last(atr_series, "ATR")
    dip = _require_finite_last(dip_series, "DI+")
    dim = _require_finite_last(dim_series, "DI-")

    atr_norm_15m = atr / float(last_close)
    di_ratio_15m = dip / max(dim, 0.001)

    return MarketConditions(atr_norm_15m=atr_norm_15m, di_ratio_15m=di_ratio_15m)


def _require_all_finite(series: pd.Series, name: str) -> None:
    if not all(math.isfinite(value) for value in series):
        raise MarketReaderError(f"Valeur non finie (NaN/infini) détectée dans la série d'entrée '{name}'.")


def _require_finite_last(series: pd.Series, name: str) -> float:
    value = float(series.iloc[-1])
    if not math.isfinite(value):
        raise MarketReaderError(f"{name} : valeur non finie en fin de série — données insuffisantes ou invalides.")
    return value
