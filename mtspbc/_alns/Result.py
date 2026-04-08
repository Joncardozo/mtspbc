from .State import State
from .Statistics import Statistics


class Result:
    def __init__(self, best: State, statistics: Statistics):
        self._best = best
        self._statistics = statistics

    @property
    def best_state(self) -> State:
        return self._best

    @property
    def statistics(self) -> Statistics:
        return self._statistics
