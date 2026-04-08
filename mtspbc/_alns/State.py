from typing import Protocol

import numpy as np


class State(Protocol):
    def objective(self) -> float: ...


class ContextualState(State, Protocol):
    def get_context(self) -> np.ndarray: ...
