"""Pré-vol de simulation pour une grille SPOT.

Construit une grille exploitable à partir de données de compte et de règles
d'instrument, sans jamais appeler une opération de placement ou d'annulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Protocol

from cell import Cell
from cell_instantiator import CellInstantiator
from grid_order_projection import GridOrderProjection, OrderInstruction
from okx_spot_adapter import Balances, FeeRates, InstrumentRules
from trellis_calculator import GridGeometry, TrellisCalculator, TrellisParameterError


class PreflightError(ValueError):
    """Le scénario ne peut pas produire une grille réellement exploitable."""


class OkxPreflightSource(Protocol):
    def get_instrument(self, inst_id: str) -> InstrumentRules: ...
    def get_fee_rates(self, inst_id: str) -> FeeRates: ...
    def get_balances(self, base_ccy: str, quote_ccy: str) -> Balances: ...


@dataclass(frozen=True)
class PreflightConfig:
    inst_id: str
    gul: Decimal
    gll: Decimal
    nu: int
    nl: int
    p0: Decimal
    geometry: GridGeometry
    spacing_h_pct: Decimal
    alpha: Decimal
    operational_margin: Decimal


@dataclass(frozen=True)
class CycleCheck:
    buy_price: Decimal
    sell_price: Decimal
    required_spread: Decimal
    actual_spread: Decimal
    profitable: bool


@dataclass(frozen=True)
class PreflightReport:
    instrument: InstrumentRules
    fees: FeeRates
    balances: Balances
    config: PreflightConfig
    total_capital: Decimal
    allocated_capital: Decimal
    trellis: tuple[Decimal, ...]
    q_raw: Decimal
    q: Decimal
    required_spot: Decimal
    required_cash: Decimal
    spot_covered: bool
    cash_covered: bool
    cycles: tuple[CycleCheck, ...]
    cells: tuple[Cell, ...]
    orders: tuple[OrderInstruction, ...]

    @property
    def economically_valid(self) -> bool:
        return all(cycle.profitable for cycle in self.cycles)

    @property
    def constraints_valid(self) -> bool:
        return self.q >= self.instrument.min_size and all(
            _multiple_of(level, self.instrument.tick_size) for level in self.trellis
        ) and _multiple_of(self.q, self.instrument.lot_size)


def run_preflight(source: OkxPreflightSource, config: PreflightConfig) -> PreflightReport:
    """Exécute le pré-vol. Cette fonction ne place aucun ordre."""
    _validate_config(config)
    instrument = source.get_instrument(config.inst_id)
    fees = source.get_fee_rates(config.inst_id)
    balances = source.get_balances(instrument.base_ccy, instrument.quote_ccy)
    if instrument.state != "live":
        raise PreflightError(f"Instrument {config.inst_id} non exploitable : state={instrument.state}")
    if not _multiple_of(config.p0, instrument.tick_size):
        raise PreflightError("P0 doit déjà respecter le tick size ; aucun arrondi silencieux de P0")

    raw_trellis = _calculate_trellis(config)
    trellis = tuple(
        config.p0 if index == config.nl else _round_down(level, instrument.tick_size)
        for index, level in enumerate(raw_trellis)
    )
    _validate_final_trellis(trellis, config.p0, instrument.tick_size)

    total_capital = balances.quote_total + balances.base_total * config.p0
    allocated_capital = config.alpha * total_capital
    lower_levels = trellis[:config.nl]
    denominator = Decimal(config.nu) * config.p0 + sum(lower_levels, Decimal("0"))
    q_raw = allocated_capital / denominator
    q = _round_down(q_raw, instrument.lot_size)
    if q < instrument.min_size:
        raise PreflightError(f"q={q} est inférieur à min_size={instrument.min_size}")
    if instrument.min_notional is not None and q * trellis[0] < instrument.min_notional:
        raise PreflightError("Le niveau le plus bas ne satisfait pas min_notional")


    maker_fee = abs(fees.maker)
    cycles = tuple(
        _cycle_check(left, right, maker_fee, config.operational_margin)
        for left, right in zip(trellis, trellis[1:])
    )

    cells = CellInstantiator().instantiate(tuple(float(level) for level in trellis), float(config.p0))
    orders = GridOrderProjection().project_initial(cells, float(q))
    required_spot = Decimal(config.nu) * q
    required_cash = sum(lower_levels, Decimal("0")) * q
    return PreflightReport(instrument, fees, balances, config, total_capital, allocated_capital, trellis, q_raw, q, required_spot, required_cash, balances.base_available >= required_spot, balances.quote_available >= required_cash, cycles, cells, orders)


def format_report(report: PreflightReport) -> str:
    """Rendu texte exclusivement simulé du pré-vol."""
    lines = [
        "SIMULATION PRE-VOL — aucun ordre envoyé",
        f"Instrument: {report.instrument.inst_id} ({report.instrument.base_ccy}/{report.instrument.quote_ccy})",
        f"Contraintes: tick={report.instrument.tick_size} lot={report.instrument.lot_size} min_size={report.instrument.min_size} min_notional={report.instrument.min_notional}",
        f"Frais: maker={report.fees.maker} taker={report.fees.taker}",
        f"Patrimoine: base={report.balances.base_total} quote={report.balances.quote_total} F_total={report.total_capital}",
        f"P0={report.config.p0} niveaux={report.config.nl + report.config.nu} treillis={', '.join(map(str, report.trellis))}",
        f"q brut={report.q_raw} q final={report.q}",
        f"Couverture: spot {report.required_spot} ({report.spot_covered}), cash {report.required_cash} ({report.cash_covered})",
        "Cycles:",
    ]
    lines.extend(f"BUY {check.buy_price} → SELL {check.sell_price}: spread={check.actual_spread}, requis={check.required_spread}, rentable={check.profitable}" for check in report.cycles)
    lines.append("Ordres initiaux simulés:")
    lines.extend(f"{side:<4} {quantity} {report.instrument.base_ccy} @ {price}" for side, price, quantity in report.orders)
    return "\n".join(lines)


def _calculate_trellis(config: PreflightConfig) -> tuple[Decimal, ...]:
    try:
        values = TrellisCalculator().calculate(float(config.gul), float(config.gll), config.nu, config.nl, float(config.p0), config.geometry, float(config.spacing_h_pct))
    except TrellisParameterError as error:
        raise PreflightError(str(error)) from error
    return tuple(Decimal(str(value)) for value in values)


def _validate_config(config: PreflightConfig) -> None:
    if not Decimal("0") <= config.alpha <= Decimal("1"):
        raise PreflightError("alpha doit appartenir à [0, 1]")
    if config.operational_margin < Decimal("0"):
        raise PreflightError("operational_margin doit être positif ou nul")


def _validate_final_trellis(trellis: tuple[Decimal, ...], p0: Decimal, tick_size: Decimal) -> None:
    if p0 not in trellis:
        raise PreflightError("P0 n'est plus identifiable dans le treillis final")
    if any(not _multiple_of(level, tick_size) for level in trellis):
        raise PreflightError("Un niveau final ne respecte pas le tick size")
    if any(left >= right for left, right in zip(trellis, trellis[1:])):
        raise PreflightError("Le treillis arrondi ne reste pas strictement croissant")


def _round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise PreflightError("tick_size et lot_size doivent être strictement positifs")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _multiple_of(value: Decimal, step: Decimal) -> bool:
    return step > 0 and value % step == 0


def _cycle_check(buy: Decimal, sell: Decimal, maker_fee: Decimal, margin: Decimal) -> CycleCheck:
    actual = sell - buy
    required = (buy * maker_fee + sell * maker_fee) * (Decimal("1") + margin)
    return CycleCheck(buy, sell, required, actual, actual >= required)
