from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator

from ..Outcome import Outcome
from ..State import State


class OperatorSelectionScheme(ABC):
    def __init__(self, num_destroy, num_repair, op_coupling=None):
        if op_coupling is not None:
            op_coupling = np.asarray(op_coupling, dtype=bool)
            op_coupling = np.atleast_2d(op_coupling)
        else:
            op_coupling = np.ones((num_destroy, num_repair), dtype=bool)

        self._validate_arguments(num_destroy, num_repair, op_coupling)
        self._num_destroy = num_destroy
        self._num_repair = num_repair
        self._op_coupling = op_coupling

    @property
    def num_destroy(self):
        return self._num_destroy

    @property
    def num_repair(self):
        return self._num_repair

    @property
    def op_coupling(self):
        return self._op_coupling

    @abstractmethod
    def __call__(self, rng: Generator, best: State, curr: State) -> Tuple[int, int]: ...

    @abstractmethod
    def update(self, candidate: State, d_idx: int, r_idx: int, outcome: Outcome): ...

    @staticmethod
    def _validate_arguments(num_destroy, num_repair, op_coupling):
        if num_destroy <= 0 or num_repair <= 0:
            raise ValueError("Missing destroy or repair operators.")
        if op_coupling.shape != (num_destroy, num_repair):
            raise ValueError(
                f"Coupling matrix shape {op_coupling.shape}, expected {(num_destroy, num_repair)}."
            )
        d_idcs = np.flatnonzero(np.count_nonzero(op_coupling, axis=1) == 0)
        if d_idcs.size != 0:
            raise ValueError(f"Destroy op. {d_idcs[0]} has no coupled repair operators.")
