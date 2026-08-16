"""Economic invariants for executable grid prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


@dataclass(frozen=True)
class GridEconomicAssumptions:
    """Worst-case costs expressed as fractions (0.001 means 0.1%)."""

    fee_buy: float
    fee_sell: float
    slippage_buy: float = 0.0
    slippage_sell: float = 0.0
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.fee_sell >= 1 or self.slippage_sell >= 1:
            raise ValueError("sell fee and slippage must be below 100%")


@dataclass(frozen=True)
class GridEconomicViolation:
    lower_price: float
    upper_price: float
    required_upper_price: float
    shortfall: float
    cycle: str
    lower_kind: str
    upper_kind: str

    def describe(self) -> str:
        return (
            f"{self.cycle}: {self.lower_kind} {self.lower_price:.8f} -> "
            f"{self.upper_kind} {self.upper_price:.8f}; minimum executable "
            f"sell price is {self.required_upper_price:.8f} "
            f"(shortfall {self.shortfall:.8f})"
        )


@dataclass(frozen=True)
class GridEconomicValidation:
    valid: bool
    violations: tuple[GridEconomicViolation, ...] = ()

    def diagnostic(self) -> str:
        return "grid is economically valid" if self.valid else "; ".join(
            violation.describe() for violation in self.violations
        )


class GridEconomicValidator:
    """Validate executable grid gaps with compounded worst-case economics.

    A BUY trigger ``B`` is executed at ``B * (1 + slippage_buy)`` and a
    SELL trigger ``S`` at ``S * (1 - slippage_sell)``.  Fees apply to those
    execution values. A pair is accepted only when::

      S(1-slip_sell)(1-fee_sell) >=
      B(1+slip_buy)(1+fee_buy)(1+safety_margin)
    """

    def __init__(self, assumptions: GridEconomicAssumptions) -> None:
        self.assumptions = assumptions

    def minimum_sell_price(self, buy_price: float) -> float:
        if buy_price <= 0:
            raise ValueError("buy_price must be positive")
        a = self.assumptions
        return buy_price * (
            (1 + a.slippage_buy) * (1 + a.fee_buy) * (1 + a.safety_margin)
            / ((1 - a.slippage_sell) * (1 - a.fee_sell))
        )

    def minimum_gap_ratio(self) -> float:
        return self.minimum_sell_price(1.0) - 1.0

    def validate_cycle(self, buy_price: float, sell_price: float,
                       cycle: Literal["BUY_TO_SELL", "SELL_TO_BUY"] = "BUY_TO_SELL",
                       *, lower_kind: str = "BUY", upper_kind: str = "SELL") -> GridEconomicValidation:
        required = self.minimum_sell_price(buy_price)
        if sell_price >= required:
            return GridEconomicValidation(valid=True)
        return GridEconomicValidation(False, (GridEconomicViolation(
            lower_price=buy_price, upper_price=sell_price,
            required_upper_price=required, shortfall=required - sell_price,
            cycle=cycle, lower_kind=lower_kind, upper_kind=upper_kind,
        ),))

    def validate_grid(self, *, anchor_price: float, buy_levels: Iterable[float],
                      sell_levels: Iterable[float]) -> GridEconomicValidation:
        """Validate every adjacent rounded price; this covers all wider pairs.

        Chaque paire adjacente doit permettre un cycle BUY -> SELL rentable
        -- une seule vérification par paire, jamais dupliquée sous un
        second nom de cycle (SELL_TO_BUY, retiré : il ne recalculait
        jamais rien, il réutilisait la même violation que BUY_TO_SELL
        sous une étiquette différente -- confirmé en production sur
        INJUSDT, 10 désactivations de grille en 2 jours pour un écart
        de rentabilité marginal, chacune comptant deux fois la même
        violation).
        """
        if anchor_price <= 0:
            raise ValueError("anchor_price must be positive")
        levels = [(float(price), "BUY") for price in buy_levels]
        levels.append((float(anchor_price), "ANCHOR"))
        levels.extend((float(price), "SELL") for price in sell_levels)
        levels.sort(key=lambda item: item[0])

        violations: list[GridEconomicViolation] = []
        for (lower, lower_kind), (upper, upper_kind) in zip(levels, levels[1:]):
            result = self.validate_cycle(
                lower, upper, lower_kind=lower_kind, upper_kind=upper_kind,
            )
            violations.extend(result.violations)
        return GridEconomicValidation(valid=not violations, violations=tuple(violations))
