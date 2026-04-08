from collections import defaultdict

import numpy as np

from .Outcome import Outcome


class Statistics:
    def __init__(self):
        self._objectives = []
        self._runtimes = []
        self._destroy_operator_counts = defaultdict(lambda: [0, 0, 0, 0])
        self._repair_operator_counts = defaultdict(lambda: [0, 0, 0, 0])

    @property
    def objectives(self):
        return np.array(self._objectives)

    @property
    def start_time(self):
        return self._runtimes[0]

    @property
    def total_runtime(self):
        return self._runtimes[-1] - self._runtimes[0]

    @property
    def runtimes(self):
        return np.diff(self._runtimes)

    @property
    def destroy_operator_counts(self):
        return self._destroy_operator_counts

    @property
    def repair_operator_counts(self):
        return self._repair_operator_counts

    def collect_objective(self, objective):
        self._objectives.append(objective)

    def collect_runtime(self, time):
        self._runtimes.append(time)

    def collect_destroy_operator(self, operator_name, outcome):
        self._destroy_operator_counts[operator_name][outcome] += 1

    def collect_repair_operator(self, operator_name, outcome):
        self._repair_operator_counts[operator_name][outcome] += 1
