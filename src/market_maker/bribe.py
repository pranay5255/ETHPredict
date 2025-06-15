"""Bribe optimization and inclusion probability model."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize

@dataclass
class BribeParams:
    """Parameters for bribe optimization."""
    min_bribe: float  # Minimum bribe amount
    max_bribe: float  # Maximum bribe amount
    base_fee: float   # Base transaction fee
    urgency: float    # Urgency factor (0-1)
    percentile: float # Target inclusion percentile

class BribeOptimizer:
    """Optimizes bribe amounts for transaction inclusion."""
    
    def __init__(self, params: BribeParams):
        self.params = params
        
    def inclusion_probability(
        self,
        bribe: float,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None
    ) -> float:
        """
        Calculate probability of transaction inclusion.
        
        Args:
            bribe: Bribe amount
            base_fee: Optional base fee override
            urgency: Optional urgency factor override
            
        Returns:
            float: Probability of inclusion
        """
        base = base_fee or self.params.base_fee
        u = urgency or self.params.urgency
        
        # Logistic function for inclusion probability
        x = (bribe - base) / base
        prob = 1 / (1 + np.exp(-u * x))
        
        return prob
        
    def optimize_bribe(
        self,
        target_prob: Optional[float] = None,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Optimize bribe amount for target inclusion probability.
        
        Args:
            target_prob: Target inclusion probability
            base_fee: Optional base fee override
            urgency: Optional urgency factor override
            
        Returns:
            Tuple of (optimal_bribe, inclusion_probability)
        """
        target = target_prob or self.params.percentile
        base = base_fee or self.params.base_fee
        u = urgency or self.params.urgency
        
        def objective(bribe):
            prob = self.inclusion_probability(bribe, base, u)
            return abs(prob - target)
            
        # Optimize bribe amount
        result = minimize(
            objective,
            x0=self.params.min_bribe,
            bounds=[(self.params.min_bribe, self.params.max_bribe)],
            method='L-BFGS-B'
        )
        
        optimal_bribe = result.x[0]
        prob = self.inclusion_probability(optimal_bribe, base, u)
        
        return optimal_bribe, prob
        
    def calculate_optimal_bribe(
        self,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None
    ) -> float:
        """
        Calculate optimal bribe amount for current market conditions.
        
        Args:
            base_fee: Optional base fee override
            urgency: Optional urgency factor override
            
        Returns:
            float: Optimal bribe amount
        """
        bribe, _ = self.optimize_bribe(
            target_prob=self.params.percentile,
            base_fee=base_fee,
            urgency=urgency
        )
        return bribe 