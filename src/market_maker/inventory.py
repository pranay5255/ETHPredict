"""Inventory book management for market making."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

@dataclass
class Position:
    """Represents a trading position."""
    size: float
    entry_price: float
    timestamp: datetime
    pnl: float = 0.0

class InventoryBook:
    """Manages inventory positions and risk limits."""
    
    def __init__(
        self,
        max_position: float,
        max_drawdown: float,
        target_inventory: float = 0.0
    ):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.target_inventory = target_inventory
        self.positions: Dict[str, Position] = {}
        self.total_pnl = 0.0
        self.peak_pnl = 0.0
        
    def add_position(
        self,
        symbol: str,
        size: float,
        price: float,
        timestamp: datetime
    ) -> bool:
        """
        Add a new position to the inventory.
        
        Args:
            symbol: Trading pair symbol
            size: Position size (positive for long, negative for short)
            price: Entry price
            timestamp: Trade timestamp
            
        Returns:
            bool: True if position was added successfully
        """
        # Check position limits
        current_size = self.get_net_position(symbol)
        if abs(current_size + size) > self.max_position:
            return False
            
        # Create new position
        position = Position(
            size=size,
            entry_price=price,
            timestamp=timestamp
        )
        
        # Update positions
        if symbol in self.positions:
            # Update existing position
            current = self.positions[symbol]
            new_size = current.size + size
            if abs(new_size) < 1e-8:  # Position closed
                del self.positions[symbol]
            else:
                # Update average price
                current.entry_price = (
                    current.entry_price * current.size + price * size
                ) / new_size
                current.size = new_size
        else:
            # Add new position
            self.positions[symbol] = position
            
        return True
        
    def update_pnl(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> float:
        """
        Update PnL for a position.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            float: Updated PnL
        """
        if symbol not in self.positions:
            return 0.0
            
        position = self.positions[symbol]
        position.pnl = position.size * (current_price - position.entry_price)
        
        # Update total PnL
        self.total_pnl = sum(p.pnl for p in self.positions.values())
        self.peak_pnl = max(self.peak_pnl, self.total_pnl)
        
        return position.pnl
        
    def get_net_position(self, symbol: str) -> float:
        """Get net position size for a symbol."""
        return self.positions.get(symbol, Position(0, 0, datetime.now())).size
        
    def get_total_position(self) -> float:
        """Get total absolute position size across all symbols."""
        return sum(abs(p.size) for p in self.positions.values())
        
    def get_drawdown(self) -> float:
        """Calculate current drawdown."""
        if self.peak_pnl == 0:
            return 0.0
        return (self.peak_pnl - self.total_pnl) / abs(self.peak_pnl)
        
    def check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if current positions violate risk limits.
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check total position size
        total_pos = self.get_total_position()
        if total_pos > self.max_position:
            return False, f"Total position {total_pos} exceeds limit {self.max_position}"
            
        # Check drawdown
        drawdown = self.get_drawdown()
        if drawdown > self.max_drawdown:
            return False, f"Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}"
            
        return True, "OK" 