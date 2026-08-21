"""Tests de cell_memory.py -- schéma de persistance (Phase 1).

Aucun accès réseau. Vérifie sérialisation, désérialisation, atomicité,
fusion stricte avec le fichier de configuration existant.
"""

import json
import os
from decimal import Decimal

import pytest

from cell import CellState
from cell_memory import (
    CellMemoryEntry, OrderStatus, ReconciliationStatus,
    deserialize_cell_memory, deserialize_entry, initial_entry,
    load_cell_memory, save_cell_memory, serialize_cell_memory, serialize_entry,
)


def sample_entry(**overrides):
    values = dict(
        economic_state=CellState.CASH_HELD,
        materialization_side=None,
        order_status=OrderStatus.NONE,
        client_order_id=None,
        order_id=None,
        trade_ids_applied=(),
        filled_qty=Decimal("0"),
        target_qty=Decimal("18.266"),
        last_transition_ts=None,
        reconciliation_status=ReconciliationStatus.OK,
        reconciliation_since_ts=None,
    )
    values.update(overrides)
    return CellMemoryEntry(**values)


class TestEntryConstruction:
    def test_initial_entry_defaults(self):
        entry = initial_entry(CellState.SPOT_HELD, Decimal("21.038"))
        assert entry.economic_state == CellState.SPOT_HELD
        assert entry.order_status == OrderStatus.NONE
        assert entry.trade_ids_applied == ()
        assert entry.filled_qty == Decimal("0")
        assert entry.reconciliation_status == ReconciliationStatus.OK

    def test_trade_ids_applied_must_be_tuple(self):
        with pytest.raises(TypeError):
            sample_entry(trade_ids_applied=["840664"])  # liste, jamais accepté

    def test_invalid_materialization_side_rejected(self):
        with pytest.raises(ValueError):
            sample_entry(materialization_side="HOLD")


class TestSerializationRoundTrip:
    def test_round_trip_preserves_all_fields(self):
        entry = sample_entry(
            materialization_side="SELL",
            order_status=OrderStatus.PARTIAL,
            client_order_id="GRIDabc123",
            order_id="3840876819823546368",
            trade_ids_applied=("840664", "838447"),
            filled_qty=Decimal("10.5"),
            last_transition_ts=1787021169070,
            reconciliation_status=ReconciliationStatus.NEEDS_RECONCILIATION,
            reconciliation_since_ts=1787021169070,
        )
        restored = deserialize_entry(serialize_entry(entry))
        assert restored == entry

    def test_decimal_serialized_as_string_never_float(self):
        entry = sample_entry(filled_qty=Decimal("0.1"))
        payload = serialize_entry(entry)
        assert payload["filled_qty"] == "0.1"
        assert isinstance(payload["filled_qty"], str)

    def test_cell_memory_dict_round_trip(self):
        memory = {
            0.9896: sample_entry(economic_state=CellState.SPOT_HELD),
            0.9795: sample_entry(economic_state=CellState.CASH_HELD),
        }
        restored = deserialize_cell_memory(serialize_cell_memory(memory))
        assert restored == memory


class TestAtomicPersistence:
    def test_save_creates_file_with_cell_memory_key(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        memory = {0.9896: sample_entry()}
        save_cell_memory(memory, path)
        with open(path) as f:
            payload = json.load(f)
        assert "cell_memory" in payload
        assert "0.9896" in payload["cell_memory"]

    def test_save_never_leaves_tmp_file_behind(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        save_cell_memory({0.9896: sample_entry()}, path)
        remaining = os.listdir(tmp_path)
        assert all(not name.startswith(".active_grid_state_") for name in remaining)

    def test_load_round_trip(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        memory = {0.9896: sample_entry(economic_state=CellState.SPOT_HELD)}
        save_cell_memory(memory, path)
        loaded = load_cell_memory(path)
        assert loaded == memory

    def test_load_missing_file_returns_empty_dict_never_raises(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        assert load_cell_memory(path) == {}

    def test_load_invalid_json_returns_empty_dict_never_raises(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert load_cell_memory(path) == {}

    def test_load_file_without_cell_memory_key_returns_empty(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        with open(path, "w") as f:
            json.dump({"inst_id": "XRP-USDC", "p0": "1.0"}, f)
        assert load_cell_memory(path) == {}


class TestStrictMergeWithExistingConfig:
    """Preuve que cell_memory ne touche JAMAIS aux champs de
    configuration existants (p0, gll, gul, etc.) -- exigence explicite
    de compatibilité avec run_active_grid.py::save_grid_state."""

    def test_existing_config_fields_untouched_after_save(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        original_config = {
            "inst_id": "XRP-USDC", "p0": "0.9999", "gll": "0.949905",
            "gul": "1.049895", "nu": 5, "nl": 5, "geometry": "FLEXIBLE",
            "spacing_h_pct": "0.002", "alpha": "0.95", "operational_margin": "0.50",
            "tick_size": "0.0001", "lot_size": "0.001",
        }
        with open(path, "w") as f:
            json.dump(original_config, f)

        save_cell_memory({0.9896: sample_entry()}, path)

        with open(path) as f:
            after = json.load(f)

        for key, value in original_config.items():
            assert after[key] == value  # strictement inchangé
        assert "cell_memory" in after  # ajouté, sans rien retirer

    def test_second_save_only_updates_cell_memory_key(self, tmp_path):
        path = str(tmp_path / "active_grid_state_XRP-USDC.json")
        with open(path, "w") as f:
            json.dump({"inst_id": "XRP-USDC", "p0": "0.9999"}, f)

        save_cell_memory({0.9896: sample_entry()}, path)
        save_cell_memory({0.9896: sample_entry(order_status=OrderStatus.FILLED)}, path)

        with open(path) as f:
            after = json.load(f)
        assert after["inst_id"] == "XRP-USDC"
        assert after["cell_memory"]["0.9896"]["order_status"] == "FILLED"
