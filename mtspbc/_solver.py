"""Single-thread ALNS solver for mTSPBC (reference settings from the paper)."""

import numpy as np
import numpy.random as rnd

from . import vrpbc_state as _vrpbc_state
from . import destroy_native as _destroy
from . import repair_native as _repair
from ._constructive import convex_hull_constructive_cover_removal
from ._alns import ALNS
from ._alns.accept import RecordToRecordTravel
from ._alns.select import RouletteWheel
from ._alns.stop import NoImprovementOrMaxRuntime, NoImprovementOrMaxRuntimeOrFeasibleOrStalled


# ---------------------------------------------------------------------------
# Destroy operator wrappers
# ---------------------------------------------------------------------------

def string_removal_op(state, rng, **kw):
    return _destroy.string_removal(state, int(rng.integers(0, 2**32)), 7, 8)

def string_removal_close_events_op(state, rng, **kw):
    return _destroy.string_removal_close_events(state, int(rng.integers(0, 2**32)), 7, 8)

def random_removal_varying_degree_op(state, rng, **kw):
    return _destroy.random_removal_varying_degree(state, int(rng.integers(0, 2**32)), 0.05, 0.1, 0.2)

def worst_removal_op(state, rng, **kw):
    return _destroy.worst_removal(state, int(rng.integers(0, 2**32)))

def cross_route_shuffle_light_op(state, rng, **kw):
    return _destroy.cross_route_shuffle(state, int(rng.integers(0, 2**32)), 0.3)

def cross_route_shuffle_heavy_op(state, rng, **kw):
    return _destroy.cross_route_shuffle(state, int(rng.integers(0, 2**32)), 0.5)

def swap_destroy_nearest_neighbour_op(state, rng, **kw):
    return _destroy.swap_destroy_nearest_neighbour(state, int(rng.integers(0, 2**32)))

def swap_destroy_random_op(state, rng, **kw):
    return _destroy.swap_destroy_random(state, int(rng.integers(0, 2**32)))

def invert_routes_op(state, rng, **kw):
    return _destroy.invert_routes(state, int(rng.integers(0, 2**32)))

def opt_2_op(state, rng, **kw):
    return _destroy.opt_2(state, int(rng.integers(0, 2**32)), 8)

def opt_3_op(state, rng, **kw):
    return _destroy.opt_3(state, int(rng.integers(0, 2**32)), 8)


# ---------------------------------------------------------------------------
# Repair operator wrappers
# ---------------------------------------------------------------------------

def greedy_repair_op(state, rng, **kw):
    return _repair.greedy_repair(state, int(rng.integers(0, 2**32)))

def greedy_repair_max_d_op(state, rng, **kw):
    return _repair.greedy_repair_max_d(state, int(rng.integers(0, 2**32)))

def greedy_repair_sum_d_op(state, rng, **kw):
    return _repair.greedy_repair_sum_d(state, int(rng.integers(0, 2**32)))

def regret_repair_op(state, rng, **kw):
    return _repair.regret_repair(state, int(rng.integers(0, 2**32)))

def empty_route_repair_op(state, rng, **kw):
    return _repair.empty_route_repair(state, int(rng.integers(0, 2**32)))

def coverage_aware_repair_op(state, rng, **kw):
    return _repair.coverage_aware_repair(state, int(rng.integers(0, 2**32)))

def local_search_op(state, rng, **kw):
    return _repair.local_search(state, int(rng.integers(0, 2**32)))


# ---------------------------------------------------------------------------
# Perturbation helpers (called directly, not via ALNS framework)
# ---------------------------------------------------------------------------

def _perturb(state, ruin_fn, repair_fn, rng):
    return repair_fn(ruin_fn(state, rng), rng)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve_instance(data: dict, tot_runtime: float, seed: int):
    """
    Run the ALNS solver on *data* for at most *tot_runtime* seconds.

    Parameters
    ----------
    data
        Instance dictionary (keys: node_coord, distances, D, k, dimension).
    tot_runtime
        Wall-clock budget in seconds.
    seed
        Integer random seed.

    Returns
    -------
    Best state found (C++ VrpbcState).
    """
    master_rng = rnd.default_rng(seed)

    # ---------- build ALNS instance ----------
    alns = ALNS(rnd.default_rng(master_rng.integers(0, 2**32)))

    alns.add_destroy_operator(string_removal_op)
    alns.add_destroy_operator(string_removal_close_events_op)
    alns.add_destroy_operator(random_removal_varying_degree_op)
    alns.add_destroy_operator(worst_removal_op)
    alns.add_destroy_operator(cross_route_shuffle_light_op)
    alns.add_destroy_operator(cross_route_shuffle_heavy_op)

    nb_destroy = len(alns.destroy_operators)

    alns.add_destroy_operator(swap_destroy_nearest_neighbour_op)
    alns.add_destroy_operator(swap_destroy_random_op)
    alns.add_destroy_operator(invert_routes_op)
    alns.add_destroy_operator(opt_2_op)
    alns.add_destroy_operator(opt_3_op)

    nb_local_search = len(alns.destroy_operators) - nb_destroy

    alns.add_repair_operator(greedy_repair_op)
    alns.add_repair_operator(greedy_repair_max_d_op)
    alns.add_repair_operator(greedy_repair_sum_d_op)
    alns.add_repair_operator(regret_repair_op)
    alns.add_repair_operator(empty_route_repair_op)
    alns.add_repair_operator(coverage_aware_repair_op)

    nb_repair = len(alns.repair_operators)
    alns.add_repair_operator(local_search_op)

    # Operator coupling: local-search destroys only pair with local_search repair
    sup  = [1] * nb_repair + [0]
    inf_ = [0] * nb_repair + [1]
    coupling = np.array([sup] * nb_destroy + [inf_] * nb_local_search)

    nb_d_total = len(alns.destroy_operators)
    nb_r_total = len(alns.repair_operators)

    # ---------- constructive solution ----------
    dimension = data["dimension"]
    k = data["k"]
    num_iters = max(5000, min(50000, dimension * k * 50))

    init_py = convex_hull_constructive_cover_removal(data)
    init = _vrpbc_state.VrpbcState(
        data["D"], data["distances"], data["node_coord"], init_py.routes, []
    )

    # ---------- Phase 1: seek feasibility ----------
    remaining_RT = float(tot_runtime)
    total_RT_used = 0.0
    run_count = 0

    repair_pool  = [greedy_repair_op, greedy_repair_max_d_op]
    ruin_pool_p1 = [random_removal_varying_degree_op]

    select = RouletteWheel([8, 2, 1, 0], 0.9, nb_d_total, nb_r_total, coupling)
    accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0.00, num_iters)
    stop   = NoImprovementOrMaxRuntimeOrFeasibleOrStalled(num_iters, remaining_RT)

    result   = alns.iterate(init, select, accept, stop, 1)
    rt       = result.statistics.total_runtime
    total_RT_used += rt
    remaining_RT  -= rt

    feasible  = result.best_state.feasible
    incumbent = result.best_state

    while remaining_RT > 0 and not feasible:
        run_count += 1
        repair_fn = repair_pool[int(master_rng.integers(0, len(repair_pool)))]
        ruin_fn   = ruin_pool_p1[int(master_rng.integers(0, len(ruin_pool_p1)))]

        stop   = NoImprovementOrMaxRuntimeOrFeasibleOrStalled(num_iters, remaining_RT)
        select = RouletteWheel([8, 2, 1, 0], 0.9, nb_d_total, nb_r_total, coupling)
        accept = RecordToRecordTravel.autofit(incumbent.objective(), 0.03, 0.01, num_iters)

        perturb_rng = rnd.default_rng(master_rng.integers(0, 2**32))
        if run_count % 5 == 0:
            restart = _destroy.random_removal(incumbent, int(perturb_rng.integers(0, 2**32)), 0.3)
            restart = _repair.greedy_repair_max_d(restart, int(perturb_rng.integers(0, 2**32)))
            result = alns.iterate(restart, select, accept, stop, 1)
        else:
            perturbation = _perturb(incumbent, ruin_fn, repair_fn, perturb_rng)
            result = alns.iterate(perturbation, select, accept, stop, 1)

        rt = result.statistics.total_runtime
        total_RT_used += rt
        remaining_RT  -= rt
        feasible = result.best_state.feasible
        if result.best_state.objective() < incumbent.objective():
            incumbent = result.best_state

    # ---------- Phase 2: optimise ----------
    ini_delta      = 0.03
    end_delta      = 0.01
    decrement      = 0.01
    stagnation     = 0
    max_stagnation = 3
    deep_stagnation = 6

    curr_state = incumbent

    repair_pool_p2 = [greedy_repair_op, regret_repair_op, greedy_repair_max_d_op, greedy_repair_sum_d_op]
    ruin_pool_p2   = [string_removal_op, string_removal_close_events_op, random_removal_varying_degree_op]

    while remaining_RT > 0:
        run_count += 1
        repair_fn = repair_pool_p2[int(master_rng.integers(0, len(repair_pool_p2)))]
        ruin_fn   = ruin_pool_p2[int(master_rng.integers(0, len(ruin_pool_p2)))]

        stop   = NoImprovementOrMaxRuntime(num_iters, remaining_RT)
        select = RouletteWheel([5, 2, 1, 0], 0.9, nb_d_total, nb_r_total, coupling)
        accept = RecordToRecordTravel.autofit(
            curr_state.objective(), ini_delta, max(0.001, ini_delta - end_delta), num_iters
        )

        perturb_rng = rnd.default_rng(master_rng.integers(0, 2**32))
        if stagnation >= deep_stagnation:
            perturbation = cross_route_shuffle_heavy_op(incumbent, perturb_rng)
            perturbation = _repair.greedy_repair_max_d(
                perturbation, int(perturb_rng.integers(0, 2**32))
            )
        else:
            perturbation = _perturb(incumbent, ruin_fn, repair_fn, perturb_rng)

        result = alns.iterate(perturbation, select, accept, stop, 2)
        rt = result.statistics.total_runtime
        total_RT_used += rt
        remaining_RT  -= rt

        improved = result.best_state.objective() < incumbent.objective()
        if improved:
            incumbent  = result.best_state
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= deep_stagnation:
            ini_delta  = 0.03
            stagnation = 0
        elif stagnation >= max_stagnation:
            ini_delta = min(0.03, ini_delta + decrement)
        elif improved:
            ini_delta = max(end_delta, ini_delta - decrement)

        curr_state = result.best_state

    return incumbent
