"""Tests de initialize_fresh_cell_memory.py -- 15 cas demandés.

Aucun accès réseau réel. Adaptateur factice dont toute méthode
d'écriture/historique lève si appelée -- preuve structurelle, pas
seulement une assertion sur le résultat.
"""

import json
from decimal import Decimal

import pytest

from cell import CellState
from cell_memory import OrderStatus, ReconciliationStatus, load_cell_memory, save_cell_memory, initial_entry
from grid_preflight import PreflightConfig
from initialize_fresh_cell_memory import build_fresh_cell_memory, main
from okx_spot_adapter import Balances, FeeRates, InstrumentRules
from trellis_calculator import GridGeometry


class FakeAdapter:
    """N'implémente NI list_fills, NI list_open_orders, NI get_order --
    tout appel à l'une de ces méthodes lève AttributeError, preuve
    structurelle qu'elles ne sont jamais sollicitées par ce module."""

    def __init__(self):
        self.instrument = InstrumentRules("XRP-USDC", "XRP", "USDC", Decimal("0.0001"), Decimal("0.001"), Decimal("0.1"), None, 4, 1, "live")
        self.fees = FeeRates("XRP-USDC", Decimal("-0.0008"), Decimal("-0.001"))
        self.balances = Balances(Decimal("1000"), Decimal("1000"), Decimal("1000"), Decimal("1000"))

    def get_instrument(self, inst_id):
        return self.instrument

    def get_fee_rates(self, inst_id):
        return self.fees

    def get_balances(self, base_ccy, quote_ccy):
        return self.balances

    def place_post_only_limit(self, *a, **k):
        raise AssertionError("ÉCRITURE INTERDITE -- placement jamais autorisé ici")

    def cancel_order(self, *a, **k):
        raise AssertionError("ÉCRITURE INTERDITE -- annulation jamais autorisée ici")


def grid_config(**changes):
    values = dict(
        inst_id="XRP-USDC", gul=Decimal("1.0563715"), gll=Decimal("0.951235"), nu=5, nl=5,
        p0=Decimal("1.0013"), geometry=GridGeometry.FLEXIBLE,
        spacing_h_pct=Decimal("0.0008"), alpha=Decimal("0.95"), operational_margin=Decimal("0.50"),
    )
    values.update(changes)
    return PreflightConfig(**values)


def write_state_file(path, config):
    payload = {
        "inst_id": config.inst_id, "p0": str(config.p0), "gll": str(config.gll), "gul": str(config.gul),
        "nu": config.nu, "nl": config.nl, "geometry": config.geometry.name,
        "spacing_h_pct": str(config.spacing_h_pct), "alpha": str(config.alpha),
        "operational_margin": str(config.operational_margin), "tick_size": "0.0001", "lot_size": "0.001",
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# 1-2 : gi < P0 -> CASH_HELD, gi > P0 -> SPOT_HELD
# ---------------------------------------------------------------------------

class TestGeometricInitialization:
    def test_1_gi_below_p0_is_cash_held(self):
        adapter = FakeAdapter()
        config = grid_config()
        memory = build_fresh_cell_memory(adapter, config)
        for gi, entry in memory.items():
            if gi < float(config.p0):
                assert entry.economic_state == CellState.CASH_HELD

    def test_2_gi_above_p0_is_spot_held(self):
        adapter = FakeAdapter()
        config = grid_config()
        memory = build_fresh_cell_memory(adapter, config)
        for gi, entry in memory.items():
            if gi > float(config.p0):
                assert entry.economic_state == CellState.SPOT_HELD

    def test_p0_never_a_cell(self):
        adapter = FakeAdapter()
        config = grid_config()
        memory = build_fresh_cell_memory(adapter, config)
        assert float(config.p0) not in memory


# ---------------------------------------------------------------------------
# 3 : target_qty == report.q
# ---------------------------------------------------------------------------

class TestTargetQty:
    def test_3_target_qty_matches_report_q(self):
        from grid_preflight import run_preflight
        adapter = FakeAdapter()
        config = grid_config()
        report = run_preflight(adapter, config)
        memory = build_fresh_cell_memory(adapter, config)
        for entry in memory.values():
            assert entry.target_qty == report.q


# ---------------------------------------------------------------------------
# 4-7 : aucun order_id/client_order_id/trade_id, order_status NONE
# ---------------------------------------------------------------------------

class TestNoTechnicalIdentity:
    def test_4_no_order_id(self):
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        assert all(entry.order_id is None for entry in memory.values())

    def test_5_no_client_order_id(self):
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        assert all(entry.client_order_id is None for entry in memory.values())

    def test_6_no_trade_ids(self):
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        assert all(entry.trade_ids_applied == () for entry in memory.values())

    def test_7_order_status_none(self):
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        assert all(entry.order_status == OrderStatus.NONE for entry in memory.values())

    def test_reconciliation_status_ok_not_needs_reconciliation(self):
        """reconciliation_status='NONE' n'existe pas dans le schéma déjà
        livré (seuls OK/NEEDS_RECONCILIATION) -- OK est la valeur correcte
        pour une cellule neuve, jamais un troisième état inventé ici."""
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        assert all(entry.reconciliation_status == ReconciliationStatus.OK for entry in memory.values())


# ---------------------------------------------------------------------------
# 8-9 : idempotence, refus par défaut, --force explicite
# ---------------------------------------------------------------------------

class TestIdempotence:
    def test_8_second_run_without_force_refuses(self, tmp_path, monkeypatch, capsys):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_state_file(path, config)

        adapter = FakeAdapter()
        first_memory = build_fresh_cell_memory(adapter, config)
        save_cell_memory(first_memory, path)

        import initialize_fresh_cell_memory as mod
        monkeypatch.setattr(mod, "state_file_path_for", lambda inst_id: path)
        monkeypatch.setattr(mod, "build_adapter", lambda mode: adapter)
        monkeypatch.setattr("sys.argv", ["initialize_fresh_cell_memory.py", "XRP-USDC"])

        exit_code = mod.main()
        captured = capsys.readouterr()
        assert "CELL_MEMORY_ALREADY_EXISTS" in captured.out
        assert exit_code != 0

        # Aucune modification : mémoire strictement identique après le refus.
        reloaded = load_cell_memory(path)
        assert reloaded == first_memory

    def test_9_force_flag_allows_overwrite(self, tmp_path, monkeypatch):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_state_file(path, config)

        adapter = FakeAdapter()
        stale_memory = {0.9999: initial_entry(CellState.SPOT_HELD, Decimal("999"))}
        save_cell_memory(stale_memory, path)

        import initialize_fresh_cell_memory as mod
        monkeypatch.setattr(mod, "state_file_path_for", lambda inst_id: path)
        monkeypatch.setattr(mod, "build_adapter", lambda mode: adapter)
        monkeypatch.setattr("sys.argv", ["initialize_fresh_cell_memory.py", "XRP-USDC", "--force"])

        exit_code = mod.main()
        assert exit_code == 0

        reloaded = load_cell_memory(path)
        assert 0.9999 not in reloaded  # ancienne entrée périmée remplacée
        assert len(reloaded) > 1


# ---------------------------------------------------------------------------
# 10-13 : aucune méthode réseau d'ordre/historique appelée
# ---------------------------------------------------------------------------

class TestNoOrderOrHistoryNetworkCalls:
    def test_10_no_placement_method_referenced(self):
        import inspect
        import initialize_fresh_cell_memory as mod
        source = inspect.getsource(mod)
        assert "adapter.place_post_only_limit(" not in source
        assert "adapter.cancel_order(" not in source

    def test_11_no_list_fills_referenced(self):
        import inspect
        import initialize_fresh_cell_memory as mod
        source = inspect.getsource(mod)
        assert "adapter.list_fills(" not in source

    def test_12_no_list_open_orders_referenced(self):
        import inspect
        import initialize_fresh_cell_memory as mod
        source = inspect.getsource(mod)
        assert "adapter.list_open_orders(" not in source

    def test_13_no_get_order_referenced(self):
        import inspect
        import initialize_fresh_cell_memory as mod
        source = inspect.getsource(mod)
        assert "adapter.get_order(" not in source

    def test_fake_adapter_lacks_these_methods_entirely_proof(self):
        """FakeAdapter n'implémente structurellement pas ces méthodes --
        si build_fresh_cell_memory les appelait, ceci lèverait AttributeError."""
        adapter = FakeAdapter()
        assert not hasattr(adapter, "list_fills")
        assert not hasattr(adapter, "list_open_orders")
        assert not hasattr(adapter, "get_order")
        # Et pourtant, la construction réussit sans erreur :
        build_fresh_cell_memory(adapter, grid_config())


# ---------------------------------------------------------------------------
# 14 : grille / PreflightConfig inchangés
# ---------------------------------------------------------------------------

class TestConfigUnchanged:
    def test_14_existing_config_fields_untouched(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        config = grid_config()
        write_state_file(path, config)
        with open(path) as f:
            original = json.load(f)

        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, config)
        save_cell_memory(memory, path)

        with open(path) as f:
            after = json.load(f)
        for key, value in original.items():
            assert after[key] == value  # strictement inchangé


# ---------------------------------------------------------------------------
# 15 : écriture atomique (réutilise le mécanisme déjà testé en Phase 1)
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_15_no_tmp_file_left_behind(self, tmp_path):
        import os
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        write_state_file(path, grid_config())
        adapter = FakeAdapter()
        memory = build_fresh_cell_memory(adapter, grid_config())
        save_cell_memory(memory, path)
        remaining = os.listdir(tmp_path)
        assert all(not name.startswith(".active_grid_state_") for name in remaining)
