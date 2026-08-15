from decimal import Decimal

from cell_order_identity import (
    ROOT,
    derive_grid_occurrence_id,
    derive_next_order_id,
    derive_root_order_id,
)
from grid_preflight import PreflightConfig
from trellis_calculator import GridGeometry


def config():
    return PreflightConfig(
        inst_id="XRP-USDC",
        gul=Decimal("1.05"),
        gll=Decimal("0.95"),
        nu=3,
        nl=3,
        p0=Decimal("100"),
        geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=Decimal("1"),
        alpha=Decimal("0.5"),
        operational_margin=Decimal("0.001"),
    )


def test_root_keeps_existing_identity():
    c = config()

    old_id = derive_root_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
    )

    assert old_id.startswith("GRID")
    assert len(old_id) == 28


def test_occurrences_at_same_gi_are_distinct():
    c = config()

    root = derive_root_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
    )

    sell_1 = derive_next_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "SELL",
        95.0,
        root,
    )

    buy_2 = derive_next_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
        sell_1,
    )

    assert root != sell_1
    assert sell_1 != buy_2
    assert root != buy_2


def test_same_predecessor_is_idempotent():
    c = config()

    root = derive_root_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
    )

    a = derive_next_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "SELL",
        95.0,
        root,
    )

    b = derive_next_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "SELL",
        95.0,
        root,
    )

    assert a == b


def test_q_does_not_affect_identity():
    c = config()

    a = derive_grid_occurrence_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
        ROOT,
    )

    # La quantité n'est volontairement pas un argument.
    b = derive_grid_occurrence_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
        ROOT,
    )

    assert a == b


def test_same_gi_different_side_have_distinct_roots():
    c = config()

    buy = derive_root_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "BUY",
        95.0,
    )

    sell = derive_root_order_id(
        c,
        Decimal("0.1"),
        Decimal("0.1"),
        "SELL",
        95.0,
    )

    assert buy != sell
