"""GLFT (Guéant-Lehalle-Fernandez-Tapia) quote calculator with inventory skew."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class GLFTParams:
    """Parameters for GLFT quote calculator."""
    gamma: float  # Risk aversion parameter
    kappa: float  # Inventory skew parameter
    sigma: float  # Volatility
    dt: float     # Time step
    max_inventory: float  # Maximum inventory limit
    min_spread: float    # Minimum spread in basis points

class GLFTQuoteCalculator:
    """GLFT quote calculator with inventory skew."""
    
    def __init__(self, params: GLFTParams):
        self.params = params
        
    def calculate_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate bid and ask quotes using GLFT formula with inventory skew.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            volatility: Optional volatility override
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Use provided volatility or default from params
        sigma = volatility or self.params.sigma
        
        # Calculate inventory skew
        skew = self.params.kappa * inventory / self.params.max_inventory
        
        # Calculate base spread
        base_spread = np.sqrt(self.params.gamma * sigma**2 * self.params.dt)
        
        # Apply inventory skew to spread
        spread = base_spread * (1 + skew)
        
        # Ensure minimum spread
        spread = max(spread, self.params.min_spread / 10000)
        
        # Calculate quotes
        bid_price = mid_price * (1 - spread/2)
        ask_price = mid_price * (1 + spread/2)
        
        return bid_price, ask_price
    
    def calculate_optimal_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: Optional[float] = None,
        target_inventory: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate optimal quotes considering target inventory.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            volatility: Optional volatility override
            target_inventory: Target inventory level
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate base quotes
        bid, ask = self.calculate_quotes(mid_price, inventory, volatility)
        
        # Adjust quotes based on inventory target
        inventory_diff = inventory - target_inventory
        if abs(inventory_diff) > 0:
            # Adjust spread to encourage mean reversion
            adjustment = self.params.kappa * inventory_diff / self.params.max_inventory
            bid *= (1 - adjustment)
            ask *= (1 + adjustment)
            
        return bid, ask 