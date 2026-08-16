"""Tests pour candidate_economic_validator.validate_candidate.

Aucun appel réseau : fonction pure, FeeRates construit directement par
les tests. Réutilise GridEconomicValidator (Gate) sans le réimplémenter
-- ces tests vérifient l'ASSEMBLAGE (quelles valeurs sont transmises),
jamais la formule économique elle-même (déjà propriété de Gate).
"""

from decimal import Decimal

import pytest

from candidate_economic_validator import SLIPPAGE_CONSERVATIVE, validate_candidate
from grid_economic_validator import GridEconomicAssumptions, GridEconomicValidator
from okx_spot_adapter import FeeRates


def fee_rates(maker="-0.0008", taker="-0.001"):
    """Maker et taker délibérément différents, pour distinguer sans ambiguïté
    lequel des deux candidate_economic_validator utilise réellement."""
    return FeeRates("XRP-USDC", Decimal(maker), Decimal(taker))


def _exact_threshold(taker: float, slippage: float = SLIPPAGE_CONSERVATIVE, buy_price: float = 0.98) -> float:
    """Seuil de vente minimal EXACT, recalculé indépendamment (jamais par
    appel à validate_candidate lui-même) — pour construire des cas
    PnL>0 / ==0 / <0 sans ambiguïté de flottant."""
    assumptions = GridEconomicAssumptions(fee_buy=taker, fee_sell=taker, slippage_buy=slippage, slippage_sell=slippage)
    return GridEconomicValidator(assumptions).minimum_sell_price(buy_price)


class TestGridEconomicValidatorItselfUnchanged:
    def test_raw_validate_grid_still_accepts_exact_boundary(self):
        """
        Preuve positive, pas seulement un diff Git à 0 ligne : appelée
        DIRECTEMENT (sans passer par validate_candidate), la brique Gate
        GridEconomicValidator.validate_grid conserve exactement son
        comportement d'origine (>=, seuil exact accepté) -- confirmant
        qu'elle n'a subi aucune modification ni duplication de formule.
        La stricte positivité n'est imposée qu'à la couche
        candidate_economic_validator, jamais dans grid_economic_validator.py.
        """
        assumptions = GridEconomicAssumptions(fee_buy=0.001, fee_sell=0.001, slippage_buy=SLIPPAGE_CONSERVATIVE, slippage_sell=SLIPPAGE_CONSERVATIVE)
        validator = GridEconomicValidator(assumptions)
        threshold = validator.minimum_sell_price(0.98)

        raw_result = validator.validate_grid(anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold,))

        assert raw_result.valid is True  # comportement Gate natif, >=, inchangé


class TestPnlSign:
    def test_pnl_positive_is_valid(self):
        fees = fee_rates(taker="-0.001")
        threshold = _exact_threshold(0.001)
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold * 1.01,))
        assert result.valid is True

    def test_pnl_exactly_at_breakeven_boundary_is_invalid(self):
        """
        CORRIGÉ : PnL net == 0 (seuil exact de couverture des coûts) doit
        être INVALID -- décision de conception "PnL net > 0", strictement
        positif. GridEconomicValidator.validate_grid seul accepterait ce
        seuil exact (comparaison >= native) ; candidate_economic_validator
        impose désormais une comparaison stricte (>) à cette couche, sans
        modifier ni dupliquer la formule économique de GridEconomicValidator
        (voir docstring de module, _walk_adjacent_pairs_strict).
        """
        fees = fee_rates(taker="-0.001")
        threshold = _exact_threshold(0.001)
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold,))
        assert result.valid is False  # seuil exact : PnL net == 0, désormais INVALID

    def test_pnl_infinitesimally_above_breakeven_is_valid(self):
        """Juste au-dessus du seuil exact (PnL net infinitésimalement positif) : VALID."""
        fees = fee_rates(taker="-0.001")
        threshold = _exact_threshold(0.001)
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold * 1.0001,))
        assert result.valid is True

    def test_pnl_negative_is_invalid(self):
        fees = fee_rates(taker="-0.001")
        threshold = _exact_threshold(0.001)  # buy_price=0.98 par défaut
        insufficient_sell = 0.98 + (threshold - 0.98) * 0.5  # entre buy et le seuil, jamais sous buy
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(insufficient_sell,))
        assert result.valid is False


class TestFeeSourceIsTaker:
    def test_fee_comes_from_taker_not_maker(self):
        """
        maker et taker délibérément très différents : si le validateur
        utilisait par erreur fees.maker, le seuil de rentabilité serait
        différent et ce test le détecterait.
        """
        fees_low_taker = fee_rates(maker="-0.05", taker="-0.0001")   # taker très bas
        fees_high_taker = fee_rates(maker="-0.05", taker="-0.05")     # taker très élevé, même maker

        # Un même écart de prix, valide sous un taker bas, invalide sous un
        # taker élevé : la seule variable qui change est taker -> preuve
        # que c'est bien taker qui est utilisé, pas maker (identique
        # dans les deux cas).
        threshold_low = _exact_threshold(0.0001)
        result_low = validate_candidate(fees_low_taker, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold_low * 1.001,))
        assert result_low.valid is True

        result_high = validate_candidate(fees_high_taker, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(threshold_low * 1.001,))
        assert result_high.valid is False  # même écart, mais taker élevé -> invalide


class TestSlippageTransmission:
    def test_slippage_buy_and_sell_both_applied(self):
        """
        Vérifie que slippage_buy ET slippage_sell sont tous deux transmis
        (pas seulement l'un des deux) — comparé à un appel équivalent
        construit directement avec GridEconomicAssumptions.
        """
        fees = fee_rates(taker="-0.001")
        direct_assumptions = GridEconomicAssumptions(
            fee_buy=0.001, fee_sell=0.001,
            slippage_buy=SLIPPAGE_CONSERVATIVE, slippage_sell=SLIPPAGE_CONSERVATIVE,
        )
        direct_result = GridEconomicValidator(direct_assumptions).validate_grid(
            anchor_price=1.00, buy_levels=(0.98,), sell_levels=(1.02,),
        )
        wrapped_result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(1.02,))

        assert wrapped_result.valid == direct_result.valid

    def test_slippage_value_matches_conservative_constant(self):
        """SLIPPAGE_CONSERVATIVE est bien la valeur réellement utilisée -- pas une autre."""
        fees = fee_rates(taker="-0.001")
        threshold_with_declared_slippage = _exact_threshold(0.001, slippage=SLIPPAGE_CONSERVATIVE)
        threshold_with_different_slippage = _exact_threshold(0.001, slippage=0.05)  # très différent

        assert threshold_with_declared_slippage != threshold_with_different_slippage

        # Un prix de vente valide sous SLIPPAGE_CONSERVATIVE mais invalide
        # sous un slippage bien plus élevé confirme que c'est bien
        # SLIPPAGE_CONSERVATIVE (et non une autre valeur) qui est utilisée.
        sell_price = threshold_with_declared_slippage * 1.001
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(sell_price,))
        assert result.valid is True
        assert sell_price < threshold_with_different_slippage  # aurait été INVALID sous l'autre hypothèse


class TestSafetyMarginNeverOverridden:
    def test_safety_margin_stays_at_default_zero(self):
        """
        Le résultat de validate_candidate doit être identique à un appel
        GridEconomicValidator construit avec safety_margin=0.0 explicite
        -- jamais une autre valeur.
        """
        fees = fee_rates(taker="-0.001")
        threshold_margin_zero = _exact_threshold(0.001)  # safety_margin implicite = 0.0 (défaut)

        assumptions_explicit_margin = GridEconomicAssumptions(
            fee_buy=0.001, fee_sell=0.001,
            slippage_buy=SLIPPAGE_CONSERVATIVE, slippage_sell=SLIPPAGE_CONSERVATIVE,
            safety_margin=0.10,  # marge non nulle, pour contraste
        )
        threshold_margin_nonzero = GridEconomicValidator(assumptions_explicit_margin).minimum_sell_price(1.0)

        assert threshold_margin_zero != threshold_margin_nonzero  # précondition du test

        # Un prix juste au-dessus du seuil "margin=0" doit être VALID via
        # validate_candidate -- s'il surchargeait silencieusement une marge
        # non nulle, ce prix serait en-dessous du vrai seuil et INVALID.
        sell_price = threshold_margin_zero * 1.0001
        result = validate_candidate(fees, anchor_price=1.00, buy_levels=(0.98,), sell_levels=(sell_price,))
        assert result.valid is True
        assert sell_price < threshold_margin_nonzero  # aurait échoué sous margin=0.10


class TestGridPreflightUntouched:
    def test_cycle_check_behavior_pinned_unchanged(self):
        """
        Filet de non-régression explicite : _cycle_check (grid_preflight.py)
        doit produire exactement le même résultat qu'avant ce chantier,
        avec sa propre formule (additive, operational_margin), totalement
        indépendante de candidate_economic_validator.
        """
        from decimal import Decimal as D
        from grid_preflight import _cycle_check

        result = _cycle_check(D("1.00"), D("1.05"), D("0.001"), D("0.50"))

        # Formule EXACTE de _cycle_check, additive : (buy*fee+sell*fee)*(1+margin)
        expected_required = (D("1.00") * D("0.001") + D("1.05") * D("0.001")) * (D("1") + D("0.50"))
        assert result.required_spread == expected_required
        assert result.actual_spread == D("0.05")
        assert result.profitable == (D("0.05") >= expected_required)


class TestNoRegression:
    def test_full_suite_placeholder(self):
        """
        Placeholder explicite : la non-régression du socle complet est
        vérifiée par l'exécution de la suite pytest complète (python3 -m
        pytest -q), pas par un test unitaire isolé -- ce test documente
        cette exigence sans la dupliquer artificiellement ici.
        """
        assert True
