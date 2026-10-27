"""Initial flat control for harness setup; no trading edge is claimed."""
import numpy as np

from grader.baselines.common import Baseline


class Strategy(Baseline):
    name = 'initial-flat-control'

    def predict(self, history, decision_times):
        return self.positions(history, decision_times, lambda data: np.zeros(len(data)))
