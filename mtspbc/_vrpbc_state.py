"""Pure-Python VrpbcState used during the constructive phase only."""

import copy

import numpy as np


class VrpbcState:
    def __init__(self, data, routes, unassigned=None, max_dist=None, feasible=False, events=None):
        self.data = data
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
        self.feasible = feasible
        self.distances = []
        self.max_d = max_dist if max_dist is not None else 0
        self.events_update()
        self.distances_update()

    def copy(self):
        return VrpbcState(self.data, copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        penalty_criteria = max(0, self.max_d - self.data["D"])
        penalty_sum = (
            1e6 * penalty_criteria
            * sum(item for sub in self.distances for item in sub if item > self.data["D"])
        )
        penalty_count_criteria = max(0, self.max_d > self.data["D"])
        penalty_count = (
            1e6 * penalty_count_criteria
            * sum(1 for sub in self.distances for item in sub if item > self.data["D"])
        )
        penalty_max = 1e6 * (
            max(self.max_d, penalty_criteria) if self.max_d > self.data["D"] else 0
        )
        penalty_empty = 1e12 * sum(1 for r in self.routes if len(r) == 0)
        routes_cost = sum(_route_cost(self, r) for r in self.routes)
        if self.max_d > self.data["D"]:
            return penalty_empty + penalty_count + penalty_max + routes_cost
        return routes_cost

    def events_update(self):
        self.events = [_compute_events(self, r) for r in self.routes]

    def distances_update(self):
        self.distances = _compute_distances(self, self.routes, self.events)
        self.max_d = max((item for sub in self.distances for item in sub), default=0.0)
        self.feasible = self.max_d <= self.data["D"]

    def find_route(self, customer):
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"Customer {customer} not in solution.")


def _route_cost(state, route):
    dist = state.data["distances"]
    tour = [0] + route + [0]
    return int(sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)))


def _compute_events(state, route):
    dist = state.data["distances"]
    tour = [0] + route + [0]
    events = [0]
    for i in range(1, len(tour)):
        events.append(int(events[-1] + dist[tour[i]][tour[i - 1]]))
    return events


def _compute_distances(state, routes, events):
    data = state.data
    n_vehicles = len(routes)
    distances = []
    for i in range(n_vehicles):
        distances.append([])
        tour_i = [0] + routes[i] + [0]
        for idx, e in enumerate(events[i]):
            max_dist = 0.0
            for j in range(n_vehicles):
                if i == j:
                    continue
                tour_j = [0] + routes[j] + [0]
                j_event_idx = len(events[j]) - 1
                try:
                    j_event_next = next(item for item in events[j] if item > e)
                    j_event_idx = events[j].index(j_event_next) - 1
                    j_event = events[j][j_event_idx]
                    dt_j = e - j_event
                    arc = (tour_j[j_event_idx], tour_j[j_event_idx + 1])
                    pos = _real_position(data, arc, dt_j)
                except StopIteration:
                    pos = data["node_coord"][0]
                dist_val = float(np.linalg.norm(data["node_coord"][tour_i[idx]] - pos))
                if dist_val > max_dist:
                    max_dist = dist_val
            distances[i].append(max_dist)
    return distances


def _real_position(data, arc, diff_time):
    dep = data["node_coord"][arc[0]]
    arr = data["node_coord"][arc[1]]
    vec = np.array(arr - dep, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return dep
    return dep + (vec / norm) * diff_time
