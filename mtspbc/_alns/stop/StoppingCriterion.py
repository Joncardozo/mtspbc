from typing import Protocol

from numpy.random import Generator

from ..State import State


class StoppingCriterion(Protocol):
    def __call__(self, rng: Generator, best: State, current: State) -> bool: ...
