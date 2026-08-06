"""
tests/test_cell.py

Tests pour cell.py -- vérifient exclusivement ce que CD-001 (version
finale) contractualise. Rien de plus : aucune assertion sur une
méthode ou une validation qui dépasserait le contrat.
"""

import pytest

from cell import Cell, CellState, CellStateError


class TestIdentity:
    """CD-001 §3 : gi et state, rien d'autre -- tous deux publics."""

    def test_gi_is_stored_exactly(self):
        cell = Cell(gi=105.123456, state=CellState.SPOT_HELD)
        assert cell.gi == 105.123456

    def test_state_is_stored_exactly(self):
        cell = Cell(gi=105.0, state=CellState.CASH_HELD)
        assert cell.state is CellState.CASH_HELD

    def test_no_other_attribute_exists_at_construction(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        assert vars(cell).keys() == {"gi", "state"}


class TestConstructionReceivesStateDirectly:
    """
    CD-001 §5 : la construction reçoit un état déjà déterminé --
    aucune référence à P0, aucune méthode dédiée qui en dépendrait.
    """

    def test_can_be_constructed_directly_in_spot_held(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        assert cell.state is CellState.SPOT_HELD

    def test_can_be_constructed_directly_in_cash_held(self):
        cell = Cell(gi=95.0, state=CellState.CASH_HELD)
        assert cell.state is CellState.CASH_HELD

    def test_no_p0_referencing_constructors_exist(self):
        assert not hasattr(Cell, "create_above_p0")
        assert not hasattr(Cell, "create_below_p0")


class TestInvariants:
    """
    CD-001 §4 : exactement deux invariants. Aucun test sur la
    positivité de gi -- ce n'est pas un invariant listé par le
    contrat, donc pas une garantie que le composant doive offrir.
    """

    def test_gi_never_changes_across_transitions(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        cell.on_sell_completed()
        cell.on_buy_completed()
        assert cell.gi == 105.0

    def test_always_exactly_one_state(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        assert cell.state in (CellState.SPOT_HELD, CellState.CASH_HELD)
        cell.on_sell_completed()
        assert cell.state in (CellState.SPOT_HELD, CellState.CASH_HELD)


class TestOnSellCompleted:
    """CD-001 §5-§6."""

    def test_toggles_to_cash_held(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        cell.on_sell_completed()
        assert cell.state is CellState.CASH_HELD

    def test_raises_when_cell_is_cash_held(self):
        cell = Cell(gi=95.0, state=CellState.CASH_HELD)
        with pytest.raises(CellStateError):
            cell.on_sell_completed()

    def test_raises_leaves_state_unchanged(self):
        cell = Cell(gi=95.0, state=CellState.CASH_HELD)
        try:
            cell.on_sell_completed()
        except CellStateError:
            pass
        assert cell.state is CellState.CASH_HELD


class TestOnBuyCompleted:
    """CD-001 §5-§6."""

    def test_toggles_to_spot_held(self):
        cell = Cell(gi=95.0, state=CellState.CASH_HELD)
        cell.on_buy_completed()
        assert cell.state is CellState.SPOT_HELD

    def test_raises_when_cell_is_spot_held(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        with pytest.raises(CellStateError):
            cell.on_buy_completed()


class TestFullCycle:
    def test_sell_then_buy_returns_to_original_state(self):
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        cell.on_sell_completed()
        cell.on_buy_completed()
        assert cell.state is CellState.SPOT_HELD
        assert cell.gi == 105.0

    def test_same_object_identity_is_preserved_across_transitions(self):
        # CD-001 §8 : seule la cellule modifie elle-même son état --
        # elle n'est jamais recréée par une transition.
        cell = Cell(gi=105.0, state=CellState.SPOT_HELD)
        identity_before = id(cell)
        cell.on_sell_completed()
        assert id(cell) == identity_before


class TestNoDependencies:
    """Vérifie, au niveau du module, l'absence de dépendance externe."""

    def test_module_imports_only_stdlib(self):
        import cell as cell_module
        import inspect

        source = inspect.getsource(cell_module)
        import_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        allowed = {"from __future__ import annotations", "from enum import Enum"}
        assert set(import_lines) <= allowed


class TestNoUnauthorizedMethods:
    """
    CD-001 §5 : exactement trois événements (construction,
    on_sell_completed, on_buy_completed). Aucune autre méthode
    publique ne doit exister -- vérifié explicitement plutôt que
    supposé.
    """

    def test_cell_exposes_no_method_beyond_the_three_contractual_events(self):
        public_methods = {
            name for name in dir(Cell)
            if not name.startswith("_") and callable(getattr(Cell, name))
        }
        assert public_methods == {"on_sell_completed", "on_buy_completed"}
