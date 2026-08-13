"""CD-004 — Projection de grille en ordres.

Ce service projette les cellules initiales en instructions d'ordres et
calcule, après un fill complet, l'ordre opposé au niveau voisin du treillis.
Une instruction est le tuple immuable ``(side, price, q)``. Ce module ne
crée aucun ordre exchange, ne conserve aucun état et ne modifie aucune Cell.

CD-002 désigne mathématiquement P0 comme le niveau à l'index ``nl``.
CD-004 exige seulement que tout niveau opérationnel reçu, y compris P0,
soit identifiable sans ambiguïté dans le treillis par l'appelant. Il ne
résout pas la question de l'égalité numérique de P0.
"""

from __future__ import annotations

from collections.abc import Sequence

from cell import Cell, CellState


BUY = "BUY"
SELL = "SELL"

OrderInstruction = tuple[str, float, float]


class GridOrderProjection:
    """Service sans état de CD-004."""

    def project_initial(
        self,
        cells: Sequence[Cell],
        q: float,
    ) -> tuple[OrderInstruction, ...]:
        """Projette les cellules CD-003 en ordres initiaux.

        CASH_HELD produit un BUY à ``gi`` ; SPOT_HELD produit un SELL à
        ``gi``. P0 n'est pas une Cell et ne peut donc produire aucun ordre
        initial dans ce service.
        """
        orders: list[OrderInstruction] = []
        for cell in cells:
            if cell.state is CellState.CASH_HELD:
                orders.append((BUY, cell.gi, q))
            elif cell.state is CellState.SPOT_HELD:
                orders.append((SELL, cell.gi, q))
        return tuple(orders)

    def after_buy_completed(
        self,
        trellis: Sequence[float],
        executed_level: float,
        q: float,
        open_orders: Sequence[OrderInstruction] = (),
    ) -> tuple[OrderInstruction, ...]:
        """Projette SELL au voisin supérieur après un BUY complètement exécuté."""
        return self._rearm(
            trellis, executed_level, q, SELL, 1, open_orders
        )

    def after_sell_completed(
        self,
        trellis: Sequence[float],
        executed_level: float,
        q: float,
        open_orders: Sequence[OrderInstruction] = (),
    ) -> tuple[OrderInstruction, ...]:
        """Projette BUY au voisin inférieur après un SELL complètement exécuté."""
        return self._rearm(
            trellis, executed_level, q, BUY, -1, open_orders
        )

    def after_partial_fill(self) -> tuple[OrderInstruction, ...]:
        """Un fill partiel ne produit aucun réarmement (CD-004)."""
        return ()

    def _rearm(
        self,
        trellis: Sequence[float],
        executed_level: float,
        q: float,
        successor_side: str,
        offset: int,
        open_orders: Sequence[OrderInstruction],
    ) -> tuple[OrderInstruction, ...]:
        """Calcule le successeur voisin, sans modifier le treillis."""
        try:
            index = trellis.index(executed_level)
        except ValueError as error:
            raise ValueError(
                "Le niveau exécuté doit être identifiable sans ambiguïté "
                "dans le treillis opérationnel"
            ) from error

        successor_index = index + offset
        if successor_index < 0 or successor_index >= len(trellis):
            return ()

        successor: OrderInstruction = (
            successor_side,
            trellis[successor_index],
            q,
        )
        if successor in open_orders:
            return ()
        return (successor,)
