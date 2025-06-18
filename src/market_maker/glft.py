from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class GLFTParams:
    gamma: float
    kappa: float
    sigma: float
    dt: float
    max_inventory: float
    min_spread: float


class GLFTQuoteCalculator:
    def __init__(self, params: GLFTParams):
        self.params = params

    def calculate_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float]:
        sigma = volatility or self.params.sigma
        skew = self.params.kappa * inventory / self.params.max_inventory
        base_spread = np.sqrt(self.params.gamma * sigma**2 * self.params.dt)
        spread = base_spread * (1 + skew)
        spread = max(spread, self.params.min_spread / 10000)
        bid = mid_price * (1 - spread / 2)
        ask = mid_price * (1 + spread / 2)
        return bid, ask

    def calculate_optimal_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: Optional[float] = None,
        target_inventory: float = 0.0,
    ) -> Tuple[float, float]:
        bid, ask = self.calculate_quotes(mid_price, inventory, volatility)
        diff = inventory - target_inventory
        if diff:
            adj = self.params.kappa * diff / self.params.max_inventory
            bid *= 1 - adj
            ask *= 1 + adj

        return bid, ask

