"""Convex-hull constructive heuristic with covered-node removal."""

import math

import numpy as np
import numpy.random as rnd

from ._vrpbc_state import VrpbcState
from ._repair_py import greedy_repair_max_d
from ._util import get_point_to_segment_distance_numpy

_SEED = 42


class _Node:
    __slots__ = ("x", "y", "index")

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def _cross(p1, p2, p3):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)


def _convex_hull(nodes):
    if not nodes:
        return []
    start = min(nodes[1:], key=lambda n: nodes[0].distance(n))
    sorted_nodes = sorted(
        nodes,
        key=lambda n: (math.atan2(n.y - start.y, n.x - start.x), n.distance(start)),
    )
    hull = []
    for node in sorted_nodes:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], node) <= 0:
            hull.pop()
        hull.append(node)
    return hull


def _covered_removal_hull(data, route_Node):
    route = [0] + [n.index for n in route_Node] + [0]
    remove_nodes = []

    for prev_idx in range(len(route) - 2):
        if len(route) <= 3:
            break
        if route[prev_idx] in remove_nodes:
            continue
        for next_idx in range(prev_idx + 2, len(route)):
            distance = 0
            for cover_idx in range(prev_idx + 1, next_idx):
                cover_node = route[cover_idx]
                distance_tmp = get_point_to_segment_distance_numpy(
                    data["node_coord"][route[prev_idx]],
                    data["node_coord"][route[next_idx]],
                    data["node_coord"][cover_node],
                )
                if distance_tmp > distance:
                    distance = distance_tmp
                if distance < data["D"]:
                    remove_nodes.append(cover_node)
                if distance > data["D"]:
                    break

    remove_nodes = set(remove_nodes)
    idxs = [i for i, n in enumerate(route) if n in remove_nodes]
    for i in reversed(idxs):
        route.pop(i)
    route.pop(0)
    route.pop()
    return route


def convex_hull_constructive_cover_removal(data):
    nodes = [_Node(c[0], c[1], i) for i, c in enumerate(data["node_coord"])]
    depot = nodes[0]
    unassigned = nodes[1:]

    salesmen_tours = [[] for _ in range(data["k"])]

    for i in range(data["k"]):
        if len(unassigned) < 3:
            break
        hull = _convex_hull(unassigned)
        hull_int = _covered_removal_hull(data, hull)
        salesmen_tours[i] = hull_int
        unassigned = [n for n in unassigned if n.index not in hull_int]

    unassigned_int = [n.index for n in unassigned]
    state = VrpbcState(data, salesmen_tours, unassigned_int)
    state = greedy_repair_max_d(state, rnd.default_rng(_SEED))
    return state
