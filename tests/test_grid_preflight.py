"""Tests du pré-vol : uniquement des données simulées, aucun ordre réel."""

from decimal import Decimal

import pytest

from grid_preflight import PreflightConfig, PreflightError, format_report, run_preflight
from okx_spot_adapter import Balances, FeeRates, InstrumentRules
from trellis_calculator import GridGeometry


class FakeSource:
    def __init__(self, *, base="0", quote="1000", maker="0.001", tick="0.1", lot="0.1", minimum="0.1", min_notional=None):
        self.instrument = InstrumentRules("ANY-QUOTE", "ANY", "QUOTE", Decimal(tick), Decimal(lot), Decimal(minimum), None if min_notional is None else Decimal(min_notional), 1, 1, "live")
        self.fees = FeeRates("ANY-QUOTE", Decimal(maker), Decimal("0.002"))
        self.balances = Balances(Decimal(base), Decimal(quote), Decimal(base), Decimal(quote))
        self.calls = []

    def get_instrument(self, inst_id):
        self.calls.append("instrument")
        return self.instrument

    def get_fee_rates(self, inst_id):
        self.calls.append("fees")
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        self.calls.append("balances")
        return self.balances


def config(**changes):
    values = dict(inst_id="ANY-QUOTE", gul=Decimal("110"), gll=Decimal("90"), nu=2, nl=2, p0=Decimal("100"), geometry=GridGeometry.EQUIDISTANT, spacing_h_pct=Decimal("0.001"), alpha=Decimal("0.5"), operational_margin=Decimal("0.5"))
    values.update(changes)
    return PreflightConfig(**values)


class TestFundingScenarios:
    def test_cash_only_reports_missing_spot_coverage(self):
        report = run_preflight(FakeSource(base="0", quote="1000"), config())
        assert report.spot_covered is False
        assert report.cash_covered is True

    def test_base_only_reports_missing_cash_coverage(self):
        report = run_preflight(FakeSource(base="10", quote="0"), config())
        assert report.spot_covered is True
        assert report.cash_covered is False

    def test_cash_and_base_can_cover_initial_projection(self):
        report = run_preflight(FakeSource(base="10", quote="1000"), config())
        assert report.spot_covered is True
        assert report.cash_covered is True


class TestConstraintsAndEconomics:
    def test_q_is_rounded_down_to_lot_size(self):
        report = run_preflight(FakeSource(base="0", quote="1000", lot="0.1"), config())
        assert report.q_raw == Decimal("1.298701298701298701298701299")
        assert report.q == Decimal("1.2")

    def test_q_under_min_size_is_rejected(self):
        with pytest.raises(PreflightError, match="min_size"):
            run_preflight(FakeSource(quote="10", minimum="1"), config())

    def test_final_levels_respect_tick_size(self):
        report = run_preflight(FakeSource(tick="0.1"), config())
        assert all(level % Decimal("0.1") == 0 for level in report.trellis)

    def test_p0_remains_identifiable_after_tick_rounding(self):
        p0 = Decimal("1.0071")
        report = run_preflight(
            FakeSource(tick="0.0001", lot="0.001", minimum="1"),
            config(
                gul=p0 * Decimal("1.15"),
                gll=p0 * Decimal("0.85"),
                nu=5,
                nl=5,
                p0=p0,
                geometry=GridGeometry.EQUIDISTANT,
            ),
        )

        assert report.trellis[5] == p0
        assert p0 in report.trellis

    def test_trellis_collapsed_by_tick_is_rejected(self):
        with pytest.raises(PreflightError, match="strictement croissant"):
            run_preflight(FakeSource(tick="10"), config())

    def test_min_notional_is_checked_at_lowest_level(self):
        with pytest.raises(PreflightError, match="min_notional"):
            run_preflight(FakeSource(min_notional="200"), config())

    def test_profitable_cycles_use_maker_fee(self):
        report = run_preflight(FakeSource(maker="0.001"), config())
        assert all(cycle.profitable for cycle in report.cycles)

    def test_negative_okx_maker_fee_is_treated_as_cost_magnitude(self):
        positive = run_preflight(FakeSource(maker="0.002"), config())
        negative = run_preflight(FakeSource(maker="-0.002"), config())
        assert negative.cycles == positive.cycles

    def test_non_profitable_cycles_are_explicitly_reported(self):
        report = run_preflight(FakeSource(maker="0.1"), config())
        assert any(not cycle.profitable for cycle in report.cycles)

    def test_operational_margin_changes_required_spread(self):
        without_margin = run_preflight(FakeSource(), config(operational_margin=Decimal("0"))).cycles[0]
        with_margin = run_preflight(FakeSource(), config(operational_margin=Decimal("0.5"))).cycles[0]
        assert with_margin.required_spread > without_margin.required_spread

    def test_different_maker_fees_change_economic_check(self):
        low = run_preflight(FakeSource(maker="0.001"), config()).cycles[0]
        high = run_preflight(FakeSource(maker="0.01"), config()).cycles[0]
        assert high.required_spread > low.required_spread


class TestProjectionAndSimulation:
    def test_uses_cd003_and_cd004_to_project_initial_orders(self):
        report = run_preflight(FakeSource(), config())
        assert report.orders == (("BUY", 90.0, 1.2), ("BUY", 95.0, 1.2), ("SELL", 105.0, 1.2), ("SELL", 110.0, 1.2))

    def test_no_order_is_sent_to_source(self):
        source = FakeSource()
        run_preflight(source, config())
        assert source.calls == ["instrument", "fees", "balances"]
        assert not hasattr(source, "place_post_only_limit")

    def test_report_is_explicitly_a_simulation(self):
        report = run_preflight(FakeSource(), config())
        text = format_report(report)
        assert "SIMULATION" in text
        assert "BUY" in text and "SELL" in text

    def test_p0_must_be_tick_aligned_without_silent_rounding(self):
        with pytest.raises(PreflightError, match="P0"):
            run_preflight(FakeSource(tick="0.1"), config(p0=Decimal("100.05"), geometry=GridGeometry.FLEXIBLE))
