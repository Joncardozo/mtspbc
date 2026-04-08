from typing import List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from ..State import State
from .OperatorSelectionScheme import OperatorSelectionScheme


class RouletteWheel(OperatorSelectionScheme):
    def __init__(self, scores, decay, num_destroy, num_repair, op_coupling=None):
        super().__init__(num_destroy, num_repair, op_coupling)

        if any(s < 0 for s in scores):
            raise ValueError("Negative scores are not understood.")
        if len(scores) < 4:
            raise ValueError(f"Expected four scores, found {len(scores)}")
        if not (0 <= decay <= 1):
            raise ValueError("decay outside [0, 1] not understood.")

        self._scores = scores
        self._d_weights = np.ones(num_destroy, dtype=float)
        self._r_weights = np.ones(num_repair, dtype=float)
        self._decay = decay

    @property
    def scores(self):
        return self._scores

    @property
    def destroy_weights(self):
        return self._d_weights

    @property
    def repair_weights(self):
        return self._r_weights

    @property
    def decay(self):
        return self._decay

    def reset_weights(self):
        self._d_weights = np.ones(self.num_destroy, dtype=float)
        self._r_weights = np.ones(self.num_repair, dtype=float)

    def __call__(self, rng: Generator, best: State, curr: State) -> Tuple[int, int]:
        def select(weights):
            probs = weights / np.sum(weights)
            return rng.choice(len(weights), p=probs)

        d_idx = select(self._d_weights)
        coupled = np.flatnonzero(self.op_coupling[d_idx])
        r_idx = coupled[select(self._r_weights[coupled])]
        return d_idx, r_idx

    def update(self, cand, d_idx, r_idx, outcome):
        self._d_weights[d_idx] = self._decay * self._d_weights[d_idx] + (1 - self._decay) * self._scores[outcome]
        self._r_weights[r_idx] = self._decay * self._r_weights[r_idx] + (1 - self._decay) * self._scores[outcome]
