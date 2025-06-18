from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class BribeParams:
    min_bribe: float
    max_bribe: float
    base_fee: float
    urgency: float
    percentile: float


class BribeOptimizer:
    def __init__(self, params: BribeParams):
        self.params = params

    def inclusion_probability(
        self,
        bribe: float,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None,
    ) -> float:
        base = base_fee or self.params.base_fee
        u = urgency or self.params.urgency
        x = (bribe - base) / base
        return 1 / (1 + np.exp(-u * x))

    def optimize_bribe(
        self,
        target_prob: Optional[float] = None,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None,
    ) -> Tuple[float, float]:
        target = target_prob or self.params.percentile
        base = base_fee or self.params.base_fee
        u = urgency or self.params.urgency

        def objective(b):
            return abs(self.inclusion_probability(b[0], base, u) - target)

        result = minimize(
            objective,
            x0=[self.params.min_bribe],
            bounds=[(self.params.min_bribe, self.params.max_bribe)],
            method="L-BFGS-B",
        )
        b = result.x[0]
        p = self.inclusion_probability(b, base, u)
        return b, p

    def calculate_optimal_bribe(
        self,
        base_fee: Optional[float] = None,
        urgency: Optional[float] = None,
    ) -> float:
        b, _ = self.optimize_bribe(None, base_fee, urgency)

        return b

