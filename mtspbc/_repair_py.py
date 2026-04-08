"""Pure-Python repair operators used by the constructive heuristic."""

import numpy as np


def greedy_repair_max_d(state, rng):
    rng.shuffle(state.unassigned)
    while state.unassigned:
        customer = state.unassigned.pop()
        route, idx = _best_insert_max_d(customer, state)
        if route is None:
            route, idx = _best_insert(customer, state)
        if route is not None:
            route.insert(idx, customer)
    state.events_update()
    state.distances_update()
    return state


def _best_insert(customer, state):
    best_cost, best_route, best_idx = None, None, None
    for route in state.routes:
        for idx in range(len(route) + 1):
            cost = _insert_cost(customer, route, idx, state)
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx
    return best_route, best_idx


def _best_insert_max_d(customer, state):
    data = state.data
    best_cost, best_route, best_idx = None, None, None
    best_max_d = 1e9

    for ir, route in enumerate(state.routes):
        for idx in range(len(route) + 1):
            cost = _insert_cost(customer, route, idx, state)
            new_state = state.copy()
            new_state.routes[ir].insert(idx, customer)
            new_state.events_update()
            new_state.distances_update()
            new_max_d = new_state.max_d

            if new_max_d <= data["D"]:
                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx
                    best_max_d = new_max_d
            else:
                if (best_cost is None or cost < best_cost) and new_max_d <= best_max_d:
                    best_cost, best_route, best_idx = cost, route, idx
                    best_max_d = new_max_d

    return best_route, best_idx


def _insert_cost(customer, route, idx, state):
    dist = state.data["distances"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]
