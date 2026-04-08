import logging

from .update import update

logger = logging.getLogger(__name__)


class RecordToRecordTravel:
    def __init__(self, start_threshold, end_threshold, step, method="linear", cmp_best=True):
        if start_threshold < 0 or end_threshold < 0 or step < 0:
            raise ValueError("Thresholds and step must be non-negative.")
        if start_threshold < end_threshold:
            raise ValueError("start_threshold < end_threshold not understood.")
        if method == "exponential" and step > 1:
            raise ValueError("Exponential updating cannot have step > 1.")
        if method not in ("linear", "exponential"):
            raise ValueError("Method must be one of ['linear', 'exponential']")

        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._step = step
        self._method = method
        self._cmp_best = cmp_best
        self._threshold = start_threshold

    @property
    def start_threshold(self):
        return self._start_threshold

    @property
    def end_threshold(self):
        return self._end_threshold

    @property
    def step(self):
        return self._step

    @property
    def method(self):
        return self._method

    def __call__(self, rng, best, current, candidate):
        baseline = best if self._cmp_best else current
        res = candidate.objective() - baseline.objective() <= self._threshold
        self._threshold = max(self.end_threshold, update(self._threshold, self.step, self.method))
        return res

    @classmethod
    def autofit(cls, init_obj, start_gap, end_gap, num_iters, method="linear"):
        if not (0 <= end_gap <= start_gap):
            raise ValueError("Must have 0 <= end_gap <= start_gap")
        if num_iters <= 0:
            raise ValueError("Non-positive num_iters not understood.")
        if method not in ("linear", "exponential"):
            raise ValueError("Method must be one of ['linear', 'exponential']")

        start_threshold = start_gap * init_obj
        end_threshold = end_gap * init_obj

        if method == "linear":
            step = (start_threshold - end_threshold) / num_iters
        else:
            step = (end_threshold / start_threshold) ** (1 / num_iters)

        return cls(start_threshold, end_threshold, step, method=method)
