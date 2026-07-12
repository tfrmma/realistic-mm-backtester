from __future__ import annotations

from dataclasses import dataclass, field

from mmbt.core.types import Fill, InventoryState


@dataclass
class Portfolio:
    """Multi-symbol inventory. One InventoryState per symbol, lazily created."""

    _positions: dict[str, InventoryState] = field(default_factory=dict, repr=False)

    def get(self, symbol: str) -> InventoryState:
        if symbol not in self._positions:
            self._positions[symbol] = InventoryState(symbol=symbol)
        return self._positions[symbol]

    def apply_fill(self, fill: Fill, fee_rate_maker: float = 0.0, fee_rate_taker: float = 0.0) -> None:
        self.get(fill.symbol).apply_fill(fill, fee_rate_maker, fee_rate_taker)

    def total_realized_pnl(self) -> float:
        return sum(inv.realized_pnl for inv in self._positions.values())

    def total_fees(self) -> float:
        return sum(inv.fees_paid for inv in self._positions.values())

    def total_unrealized_pnl(self, mids: dict[str, float]) -> float:
        return sum(
            inv.unrealized_pnl(mids[s])
            for s, inv in self._positions.items()
            if s in mids
        )

    def total_pnl(self, mids: dict[str, float]) -> float:
        return self.total_realized_pnl() + self.total_unrealized_pnl(mids) - self.total_fees()

    def positions(self) -> dict[str, float]:
        return {s: inv.position for s, inv in self._positions.items()}

    def symbols(self) -> list[str]:
        return list(self._positions.keys())

    def summary(self) -> dict:
        return {
            "symbols": self.symbols(),
            "positions": {s: round(p, 6) for s, p in self.positions().items()},
            "total_realized_pnl": round(self.total_realized_pnl(), 8),
            "total_fees": round(self.total_fees(), 8),
        }
