"""Limit Order Book (LOB) matching engine with price-time FIFO."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

@dataclass
class Order:
    """Represents a limit order."""
    order_id: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    timestamp: datetime
    filled: float = 0.0
    status: str = "open"  # "open", "filled", "cancelled"

class LOBMatchingEngine:
    """Limit Order Book matching engine with price-time FIFO."""
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: Dict[float, List[Order]] = defaultdict(list)  # price -> orders
        self.asks: Dict[float, List[Order]] = defaultdict(list)  # price -> orders
        self.orders: Dict[str, Order] = {}  # order_id -> order
        
    def _round_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self.tick_size) * self.tick_size
        
    def add_order(
        self,
        order_id: str,
        side: str,
        price: float,
        size: float,
        timestamp: datetime
    ) -> Order:
        """
        Add a new order to the book.
        
        Args:
            order_id: Unique order identifier
            side: "buy" or "sell"
            price: Limit price
            size: Order size
            timestamp: Order timestamp
            
        Returns:
            Order: The created order
        """
        # Round price to tick size
        price = self._round_price(price)
        
        # Create order
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            timestamp=timestamp
        )
        
        # Add to book
        if side == "buy":
            self.bids[price].append(order)
        else:
            self.asks[price].append(order)
            
        self.orders[order_id] = order
        return order
        
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            bool: True if order was cancelled
        """
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        order.status = "cancelled"
        
        # Remove from book
        if order.side == "buy":
            self.bids[order.price].remove(order)
            if not self.bids[order.price]:
                del self.bids[order.price]
        else:
            self.asks[order.price].remove(order)
            if not self.asks[order.price]:
                del self.asks[order.price]
                
        return True
        
    def match_orders(self) -> List[Tuple[Order, Order, float, float]]:
        """
        Match orders in the book.
        
        Returns:
            List of (buy_order, sell_order, price, size) tuples
        """
        matches = []
        
        # Get best bid and ask
        best_bid = max(self.bids.keys()) if self.bids else 0
        best_ask = min(self.asks.keys()) if self.asks else float('inf')
        
        while best_bid >= best_ask and self.bids and self.asks:
            # Get orders at best prices
            bid_orders = self.bids[best_bid]
            ask_orders = self.asks[best_ask]
            
            # Match orders
            while bid_orders and ask_orders:
                bid = bid_orders[0]
                ask = ask_orders[0]
                
                # Calculate match size
                match_size = min(
                    bid.size - bid.filled,
                    ask.size - ask.filled
                )
                
                # Record match
                matches.append((bid, ask, best_bid, match_size))
                
                # Update filled amounts
                bid.filled += match_size
                ask.filled += match_size
                
                # Remove filled orders
                if bid.filled >= bid.size:
                    bid.status = "filled"
                    bid_orders.pop(0)
                if ask.filled >= ask.size:
                    ask.status = "filled"
                    ask_orders.pop(0)
                    
            # Clean up empty price levels
            if not bid_orders:
                del self.bids[best_bid]
            if not ask_orders:
                del self.asks[best_ask]
                
            # Update best prices
            best_bid = max(self.bids.keys()) if self.bids else 0
            best_ask = min(self.asks.keys()) if self.asks else float('inf')
            
        return matches
        
    def get_order_book(self, depth: int = 10) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Get current order book state.
        
        Args:
            depth: Number of price levels to return
            
        Returns:
            Tuple of (bids, asks) where each is a list of (price, size) tuples
        """
        # Get bids
        bid_prices = sorted(self.bids.keys(), reverse=True)[:depth]
        bids = [
            (price, sum(order.size - order.filled for order in self.bids[price]))
            for price in bid_prices
        ]
        
        # Get asks
        ask_prices = sorted(self.asks.keys())[:depth]
        asks = [
            (price, sum(order.size - order.filled for order in self.asks[price]))
            for price in ask_prices
        ]
        
        return bids, asks
        
    def get_mid_price(self) -> float:
        """Get current mid price."""
        best_bid = max(self.bids.keys()) if self.bids else 0
        best_ask = min(self.asks.keys()) if self.asks else float('inf')
        return (best_bid + best_ask) / 2 if best_ask != float('inf') else best_bid 