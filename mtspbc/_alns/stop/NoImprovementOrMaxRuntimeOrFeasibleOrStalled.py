import time
from typing import Optional

from numpy.random import Generator

from ..State import State


class NoImprovementOrMaxRuntimeOrFeasibleOrStalled:
    def __init__(self, max_iterations: int, max_runtime: float):
        if max_iterations < 0:
            raise ValueError("max_iterations < 0 not understood.")
        self._max_iterations = max_iterations
        self._max_runtime = max_runtime
        self._target: Optional[float] = None
        self._counter_it = 0
        self._start_time = None

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def max_runtime(self):
        return self._max_runtime

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        if self._start_time is None:
            self._start_time = time.perf_counter()
        if self._target is None or best.objective() < self._target:
            self._target = best.objective()
            self._counter_it = 0
        else:
            self._counter_it += 1
        return (
            self._counter_it >= self._max_iterations
            or (time.perf_counter() - self._start_time > self._max_runtime)
            or best.feasible
        )
