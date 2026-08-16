"""Validation économique des configurations candidates de grille — V1.

Réutilise GridEconomicValidator (grid_economic_validator.py, brique du bot
Gate, importée telle quelle, jamais modifiée). Cette brique se contente
d'assembler l'hypothèse économique conservatrice V1 et d'appeler
minimum_sell_price (l'UNIQUE formule économique existante) — aucune
formule économique n'est réimplémentée ni dupliquée ici.

Frontière stricte avec l'existant, déjà validée dans le document de
conception de ce chantier :
  grid_preflight.py::_cycle_check  -> reste l'UNIQUE garde-fou de la
                                       grille RÉELLEMENT construite
                                       (frais MAKER, operational_margin
                                       inchangé, jamais touché ici).
  candidate_economic_validator.py  -> valide des CANDIDATS, avant leur
                                       transformation en PreflightConfig,
                                       jamais après.

Hypothèse conservatrice V1 :
  fee_buy = fee_sell = abs(fees.taker)
    La grille réelle s'exécute exclusivement en POST-ONLY (maker) —
    GridActivationController n'appelle jamais que place_post_only_limit
    pour les ordres de grille. Utiliser le taux TAKER ici est donc
    délibérément plus pessimiste que l'exécution réelle attendue : si
    un candidat reste rentable sous cette hypothèse, il dispose d'une
    marge supplémentaire en conditions réelles (maker, moins coûteux).
  slippage_buy = slippage_sell = SLIPPAGE_CONSERVATIVE
    Hypothèse V1 FIXE, PAS une mesure OKX réelle. Aucun historique
    d'exécutions OKX n'existe aujourd'hui pour calculer un slippage
    observé (contrairement au mécanisme EMA du bot Gate, qui mesure le
    slippage réel après chaque exécution — inapplicable ici en l'absence
    de collecte historique, volontairement hors périmètre de ce
    chantier). À ajuster explicitement si une mesure réelle devient
    disponible plus tard.
  safety_margin
    JAMAIS surchargé — le défaut de GridEconomicAssumptions (0.0) est
    utilisé tel quel, conformément à la décision de garder cette V1
    aussi simple que possible. operational_margin (grid_preflight.py)
    n'est ni réutilisé, ni dupliqué ici : les deux formules (additive
    côté _cycle_check, multiplicative en chaîne côté GridEconomicValidator)
    ne sont pas interchangeables — partager une même constante entre les
    deux produirait des seuils de rentabilité incohérents.

PnL net STRICTEMENT positif (décision de conception) :
  GridEconomicValidator.validate_grid (Gate, verbatim) accepte
  sell_price >= required_sell_price (comparaison LARGE — au seuil exact,
  PnL net == 0, le candidat est déclaré valide par la primitive Gate
  telle quelle). Ceci diverge de la décision de conception "PnL net > 0"
  (strictement positif).

  Correction appliquée ICI, à cette couche uniquement, sans jamais
  modifier grid_economic_validator.py ni y dupliquer la formule
  économique : _walk_adjacent_pairs_strict ci-dessous reproduit
  UNIQUEMENT la mécanique de tri/appariement des niveaux adjacents,
  déjà présente dans GridEconomicValidator.validate_grid (pur Python de
  structure : construire les triplets (prix, nature), trier, parcourir
  par paires adjacentes -- AUCUN contenu économique). Pour chaque paire,
  le seuil requis est obtenu par un appel direct à
  GridEconomicValidator.minimum_sell_price -- la seule et unique formule
  économique de ce projet, jamais réécrite -- puis comparé avec une
  inégalité STRICTE (>) au lieu de la comparaison large (>=) native de
  validate_grid.

  Compromis assumé explicitement : cette duplication de la mécanique de
  tri/appariement crée un couplage avec l'implémentation interne de
  GridEconomicValidator.validate_grid. Si cette dernière évoluait un
  jour (nouvelle règle de tri, nouveau type de niveau), la copie ci-
  dessous devrait être mise à jour en conséquence -- risque accepté ici
  car grid_economic_validator.py est explicitement gelé et importé
  verbatim dans ce projet, jamais appelé à évoluer indépendamment.
"""

from __future__ import annotations

from grid_economic_validator import GridEconomicAssumptions, GridEconomicValidation, GridEconomicValidator, GridEconomicViolation
from okx_spot_adapter import FeeRates

# Hypothèse V1 conservatrice, non mesurée — voir docstring de module.
# Ordre de grandeur choisi par analogie avec TRADING_FEE_RT (bot Gate,
# 0.001 = 0.10%), pas une mesure OKX. À valider/ajuster explicitement.
SLIPPAGE_CONSERVATIVE: float = 0.001


def validate_candidate(
    fees: FeeRates,
    anchor_price: float,
    buy_levels: tuple[float, ...],
    sell_levels: tuple[float, ...],
) -> GridEconomicValidation:
    """Valide une configuration de grille candidate sous l'hypothèse V1.

    Fonction pure : `fees` doit être déjà lu par l'appelant (aucun accès
    réseau ici). Ne modifie jamais anchor_price/buy_levels/sell_levels,
    ne recalcule jamais le treillis.

    PnL net STRICTEMENT positif exigé (voir docstring de module) — au
    seuil exact de couverture des coûts, le résultat est INVALID, pas
    VALID (contrairement à GridEconomicValidator.validate_grid appelé
    seul, qui accepte le seuil exact).
    """
    fee_taker = abs(float(fees.taker))
    assumptions = GridEconomicAssumptions(
        fee_buy=fee_taker,
        fee_sell=fee_taker,
        slippage_buy=SLIPPAGE_CONSERVATIVE,
        slippage_sell=SLIPPAGE_CONSERVATIVE,
        # safety_margin non fourni : reste au défaut 0.0 de GridEconomicAssumptions.
    )
    validator = GridEconomicValidator(assumptions)
    return _walk_adjacent_pairs_strict(validator, anchor_price, buy_levels, sell_levels)


def _walk_adjacent_pairs_strict(
    validator: GridEconomicValidator,
    anchor_price: float,
    buy_levels: tuple[float, ...],
    sell_levels: tuple[float, ...],
) -> GridEconomicValidation:
    """Reproduit UNIQUEMENT la mécanique de tri/appariement de
    GridEconomicValidator.validate_grid (structure pure, aucun contenu
    économique) — pour appliquer une inégalité stricte (PnL net > 0) au
    lieu de la comparaison large (>=) native de cette méthode.

    L'unique appel à du contenu économique est validator.minimum_sell_price
    -- la formule elle-même, jamais dupliquée, jamais réécrite.
    """
    if anchor_price <= 0:
        raise ValueError("anchor_price must be positive")

    levels = [(float(price), "BUY") for price in buy_levels]
    levels.append((float(anchor_price), "ANCHOR"))
    levels.extend((float(price), "SELL") for price in sell_levels)
    levels.sort(key=lambda item: item[0])

    violations: list[GridEconomicViolation] = []
    for (lower, lower_kind), (upper, upper_kind) in zip(levels, levels[1:]):
        required = validator.minimum_sell_price(lower)  # formule unique, réutilisée telle quelle
        if upper <= required:  # strict : le seuil exact n'est plus accepté (PnL net > 0)
            violations.append(GridEconomicViolation(
                lower_price=lower, upper_price=upper, required_upper_price=required,
                shortfall=required - upper, cycle="BUY_TO_SELL",
                lower_kind=lower_kind, upper_kind=upper_kind,
            ))
    return GridEconomicValidation(valid=not violations, violations=tuple(violations))
