"""AMM (constant-product) swap simulator for Aerodrome and PancakeSwap."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

class AMMType(Enum):
    """Supported AMM types."""
    AERODROME = "aerodrome"
    PANCAKESWAP = "pancakeswap"

@dataclass
class AMMParams:
    """Parameters for AMM simulation."""
    amm_type: AMMType
    fee_bps: int  # Fee in basis points
    initial_x: float  # Initial token X reserves
    initial_y: float  # Initial token Y reserves
    slippage_tolerance: float = 0.01  # 1% default slippage tolerance

class AMMSimulator:
    """Simulates AMM (constant-product) swaps."""
    
    def __init__(self, params: AMMParams):
        self.params = params
        self.x_reserves = params.initial_x
        self.y_reserves = params.initial_y
        self.k = params.initial_x * params.initial_y  # Constant product
        
    def get_price(self) -> float:
        """Get current price (y/x)."""
        return self.y_reserves / self.x_reserves
        
    def calculate_swap_output(
        self,
        amount_in: float,
        is_x_to_y: bool,
        with_slippage: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate swap output amount.
        
        Args:
            amount_in: Input amount
            is_x_to_y: True if swapping X to Y, False otherwise
            with_slippage: Whether to apply slippage tolerance
            
        Returns:
            Tuple of (output_amount, price_impact)
        """
        # Calculate fee
        fee = amount_in * (self.params.fee_bps / 10000)
        amount_in_after_fee = amount_in - fee
        
        # Calculate output using constant product formula
        if is_x_to_y:
            # X to Y swap
            amount_out = (self.y_reserves * amount_in_after_fee) / (
                self.x_reserves + amount_in_after_fee
            )
            price_impact = amount_out / self.y_reserves
        else:
            # Y to X swap
            amount_out = (self.x_reserves * amount_in_after_fee) / (
                self.y_reserves + amount_in_after_fee
            )
            price_impact = amount_out / self.x_reserves
            
        # Apply slippage tolerance if requested
        if with_slippage:
            amount_out *= (1 - self.params.slippage_tolerance)
            
        return amount_out, price_impact
        
    def execute_swap(
        self,
        amount_in: float,
        is_x_to_y: bool,
        with_slippage: bool = True
    ) -> Tuple[float, float, float]:
        """
        Execute a swap and update reserves.
        
        Args:
            amount_in: Input amount
            is_x_to_y: True if swapping X to Y, False otherwise
            with_slippage: Whether to apply slippage tolerance
            
        Returns:
            Tuple of (output_amount, price_impact, fee)
        """
        # Calculate output
        amount_out, price_impact = self.calculate_swap_output(
            amount_in, is_x_to_y, with_slippage
        )
        
        # Calculate fee
        fee = amount_in * (self.params.fee_bps / 10000)
        
        # Update reserves
        if is_x_to_y:
            self.x_reserves += amount_in
            self.y_reserves -= amount_out
        else:
            self.x_reserves -= amount_out
            self.y_reserves += amount_in
            
        return amount_out, price_impact, fee
        
    def get_liquidity_depth(
        self,
        price_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculate liquidity depth for a price range.
        
        Args:
            price_range: Tuple of (min_price, max_price)
            
        Returns:
            Tuple of (x_liquidity, y_liquidity)
        """
        min_price, max_price = price_range
        
        # Calculate liquidity at price bounds
        x_at_min = np.sqrt(self.k / min_price)
        x_at_max = np.sqrt(self.k / max_price)
        y_at_min = np.sqrt(self.k * min_price)
        y_at_max = np.sqrt(self.k * max_price)
        
        # Calculate depth
        x_depth = abs(x_at_max - x_at_min)
        y_depth = abs(y_at_max - y_at_min)
        
        return x_depth, y_depth 