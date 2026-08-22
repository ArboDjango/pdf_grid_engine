"""Tests pour dry_run_replacement.py -- harnais DRY-RUN de validation du
remplacement dynamique de grille (grid_replacement.py).

Ne teste PAS grid_replacement.py lui-même (déjà couvert par
tests/test_grid_replacement.py) : vérifie uniquement que le harnais lui-même
(1) ne laisse jamais un appel d'écriture atteindre l'adaptateur réel, (2)
simule correctement le bookkeeping d'un fill de market buy, et (3) permet au
scénario de remplacement d'aboutir jusqu'à ACTIVATED quand un achat
patrimonial initial est nécessaire.
"""

import json
import os
from decimal import Decimal

import pytest

import dry_run_replacement
import grid_replacement
from test_grid_replacement import FakeAdapter, grid_config, open_order


def write_real_state(path, config, tick="0.0001", lot="0.001"):
    payload = {
        "inst_id": config.inst_id, "p0": str(config.p0), "gll": str(config.gll),
        "gul": str(config.gul), "nu": config.nu, "nl": config.nl,
        "geometry": config.geometry.name, "spacing_h_pct": str(config.spacing_h_pct),
        "alpha": str(config.alpha), "operational_margin": str(config.operational_margin),
        "tick_size": tick, "lot_size": lot,
        "q": str(config.q), "allocated_capital": str(config.allocated_capital),
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# =============================================================================
# Bookkeeping simulé -- get_balances / place_market_buy.
# =============================================================================


class TestSimulatedBalances:
    def test_get_balances_is_strict_passthrough_before_any_fill(self):
        fake = FakeAdapter()
        dry = dry_run_replacement.DryRunAdapter(
            fake, dry_run_replacement.SimulatedPriceFeed([Decimal("1.20")]), log=lambda msg: None,
        )
        real = fake.get_balances("XRP", "USDC")
        wrapped = dry.get_balances("XRP", "USDC")
        assert wrapped == real

    def test_place_market_buy_never_reaches_real_adapter(self):
        fake = FakeAdapter()
        dry = dry_run_replacement.DryRunAdapter(
            fake, dry_run_replacement.SimulatedPriceFeed([Decimal("1.20")]), log=lambda msg: None,
        )
        dry.get_ticker("XRP-USDC")  # fixe last_price_seen à 1.20
        dry.place_market_buy("XRP-USDC", Decimal("100"), "INITBUY-TEST")
        assert fake.placed_orders == []  # aucun appel réel

    def test_place_market_buy_updates_shadow_balances_by_exact_fill_arithmetic(self):
        fake = FakeAdapter()
        real_before = fake.get_balances("XRP", "USDC")
        dry = dry_run_replacement.DryRunAdapter(
            fake, dry_run_replacement.SimulatedPriceFeed([Decimal("1.20")]), log=lambda msg: None,
        )
        dry.get_ticker("XRP-USDC")
        dry.place_market_buy("XRP-USDC", Decimal("100"), "INITBUY-TEST")

        wrapped = dry.get_balances("XRP", "USDC")
        assert wrapped.base_available == real_before.base_available + Decimal("100")
        assert wrapped.quote_available == real_before.quote_available - Decimal("100") * Decimal("1.20")
        # Le solde réel sous-jacent (FakeAdapter) reste totalement inchangé --
        # seul le ledger local de DryRunAdapter a bougé.
        assert fake.get_balances("XRP", "USDC") == real_before

    def test_multiple_fills_accumulate(self):
        fake = FakeAdapter()
        real_before = fake.get_balances("XRP", "USDC")
        dry = dry_run_replacement.DryRunAdapter(
            fake, dry_run_replacement.SimulatedPriceFeed([Decimal("1.10")]), log=lambda msg: None,
        )
        dry.get_ticker("XRP-USDC")
        dry.place_market_buy("XRP-USDC", Decimal("50"), "A")
        dry.place_market_buy("XRP-USDC", Decimal("30"), "B")
        wrapped = dry.get_balances("XRP", "USDC")
        assert wrapped.base_available == real_before.base_available + Decimal("80")
        assert len(dry.simulated_fills) == 2


# =============================================================================
# Bout en bout -- le remplacement doit pouvoir atteindre ACTIVATED.
# =============================================================================


class TestFullMaterializationReachesActivated:
    def test_replacement_materializes_and_no_real_write_happens(self, tmp_path, monkeypatch, capsys):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_real_state(state_path, config)

        fake = FakeAdapter(ticker_price="1.20")
        fake.open_orders = (open_order(1.02, side="SELL", coid="GRID-EXISTING-1"),)

        monkeypatch.setattr(dry_run_replacement, "build_adapter", lambda mode: fake)
        monkeypatch.setattr(
            "sys.argv",
            [
                "dry_run_replacement.py", "--inst-id", "XRP-USDC", "--state-file", state_path,
                "--max-cycles", "30",
                "--prices", ",".join(["1.02"] * 2 + ["1.08", "1.15"] + ["1.20"] * 26),
            ],
        )

        rc = dry_run_replacement.main()
        out = capsys.readouterr().out

        assert rc == 0
        assert "REMPLACEMENT MATÉRIALISÉ" in out
        assert "P0_new == round_down(borne_old, tick_size) : OK" in out
        assert "NIVEAUX DE LA NOUVELLE GRILLE" in out

        # Aucun ordre réel : le FakeAdapter (adaptateur "réel" simulé ici)
        # n'a jamais vu passer d'écriture -- tout est resté dans le ledger
        # et le carnet fictif de DryRunAdapter.
        assert fake.placed_orders == []
        assert fake.cancelled_orders == []

    def test_invariant_p0_new_equals_rounded_old_boundary(self, tmp_path, monkeypatch, capsys):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_real_state(state_path, config)

        fake = FakeAdapter(ticker_price="1.20")

        monkeypatch.setattr(dry_run_replacement, "build_adapter", lambda mode: fake)
        monkeypatch.setattr(
            "sys.argv",
            [
                "dry_run_replacement.py", "--inst-id", "XRP-USDC", "--state-file", state_path,
                "--max-cycles", "30",
                "--prices", ",".join(["1.02"] * 2 + ["1.08", "1.15"] + ["1.20"] * 26),
            ],
        )
        dry_run_replacement.main()
        out = capsys.readouterr().out

        instrument = fake.get_instrument("XRP-USDC")
        expected_p0 = grid_replacement._round_down(config.gul, instrument.tick_size)
        assert f"round_down(borne_old, tick_size)   = {expected_p0}" in out
        assert "dépassement HAUT : P0_new == GUL_old ?" in out

    def test_downward_breach_direction_and_invariant(self, tmp_path, monkeypatch, capsys):
        state_path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_real_state(state_path, config)

        fake = FakeAdapter(ticker_price="0.90")

        monkeypatch.setattr(dry_run_replacement, "build_adapter", lambda mode: fake)
        monkeypatch.setattr(
            "sys.argv",
            [
                "dry_run_replacement.py", "--inst-id", "XRP-USDC", "--state-file", state_path,
                "--max-cycles", "30",
                "--prices", ",".join(["1.00", "1.00", "0.94", "0.92"] + ["0.90"] * 26),
            ],
        )
        dry_run_replacement.main()
        out = capsys.readouterr().out

        assert "direction        = DOWN" in out
        assert "dépassement BAS : P0_new == GLL_old ?" in out
        assert fake.placed_orders == []
        assert fake.cancelled_orders == []
