import logging
import time
from typing import Dict, List, Optional, Protocol, Tuple

import numpy.random as rnd

from .Outcome import Outcome
from .Result import Result
from .State import State
from .Statistics import Statistics
from .accept import AcceptanceCriterion
from .select import OperatorSelectionScheme
from .stop import StoppingCriterion

logger = logging.getLogger(__name__)


class _OperatorType(Protocol):
    __name__: str

    def __call__(self, state: State, rng: rnd.Generator, **kwargs) -> State: ...


class _CallbackType(Protocol):
    __name__: str

    def __call__(self, state: State, rng: rnd.Generator, **kwargs): ...


class ALNS:
    """
    Adaptive Large Neighbourhood Search (ALNS) algorithm.
    Minimisation variant following Pisinger & Røpke (2010).
    """

    def __init__(self, rng: rnd.Generator = rnd.default_rng()):
        self._rng = rng
        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}
        self._on_outcome: Dict[Outcome, _CallbackType] = {}

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._r_ops.items())

    def add_destroy_operator(self, op: _OperatorType, name: Optional[str] = None):
        self._d_ops[op.__name__ if name is None else name] = op

    def add_repair_operator(self, op: _OperatorType, name: Optional[str] = None):
        self._r_ops[name if name else op.__name__] = op

    def iterate(
        self,
        initial_solution: State,
        op_select: OperatorSelectionScheme,
        accept: AcceptanceCriterion,
        stop: StoppingCriterion,
        phase: int,
        **kwargs,
    ) -> Result:
        if not self.destroy_operators or not self.repair_operators:
            raise ValueError("Missing destroy or repair operators.")

        curr = best = initial_solution

        stats = Statistics()
        stats.collect_objective(initial_solution.objective())
        stats.collect_runtime(time.perf_counter())

        while not stop(self._rng, best, curr):
            d_idx, r_idx = op_select(self._rng, best, curr)
            _, d_op = self.destroy_operators[d_idx]
            _, r_op = self.repair_operators[r_idx]

            destroyed = d_op(curr, self._rng, **kwargs)
            cand = r_op(destroyed, self._rng, **kwargs)

            best, curr, outcome = self._eval_cand(accept, best, curr, cand, **kwargs)
            op_select.update(cand, d_idx, r_idx, outcome)

            stats.collect_objective(curr.objective())
            stats.collect_destroy_operator(self.destroy_operators[d_idx][0], outcome)
            stats.collect_repair_operator(self.repair_operators[r_idx][0], outcome)
            stats.collect_runtime(time.perf_counter())

        return Result(best, stats)

    def on_best(self, func: _CallbackType):
        self._on_outcome[Outcome.BEST] = func

    def on_better(self, func: _CallbackType):
        self._on_outcome[Outcome.BETTER] = func

    def on_accept(self, func: _CallbackType):
        self._on_outcome[Outcome.ACCEPT] = func

    def on_reject(self, func: _CallbackType):
        self._on_outcome[Outcome.REJECT] = func

    def _eval_cand(self, accept, best, curr, cand, **kwargs):
        outcome = self._determine_outcome(accept, best, curr, cand)
        func = self._on_outcome.get(outcome)
        if callable(func):
            func(cand, self._rng, **kwargs)

        if outcome == Outcome.BEST:
            return cand, cand, outcome
        if outcome == Outcome.REJECT:
            return best, curr, outcome
        return best, cand, outcome

    def _determine_outcome(self, accept, best, curr, cand):
        outcome = Outcome.REJECT
        if accept(self._rng, best, curr, cand):
            outcome = Outcome.ACCEPT
            if cand.objective() < curr.objective():
                outcome = Outcome.BETTER
        if cand.objective() < best.objective():
            outcome = Outcome.BEST
        return outcome
