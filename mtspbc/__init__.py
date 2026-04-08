"""
mtspbc — ALNS solver for the multiple Travelling Salesman Problem
with Backup Coverage (mTSP-BC).

Usage
-----
    import mtspbc
    mtspbc.solve("path/to/instance.BC")
"""

import sys

from ._read import build_distance_matrix, read_instance
from ._solver import solve_instance


def solve(instance_path: str) -> None:
    """
    Solve a mTSPBC instance and print the routes and objective value.

    Parameters
    ----------
    instance_path
        Path to the instance file. Format: first line is ``m n d``
        (vehicles, clients, max coverage distance); followed by ``n+1``
        lines of ``x y`` coordinates (depot first).
    """
    import time

    data = read_instance(instance_path)
    data["distances"] = build_distance_matrix(data["node_coord"])

    n, k, D = data["dimension"], data["k"], data["D"]
    tot_RT = max(10, int(n * k / 25 * 60))

    print(f"Instance : {instance_path}")
    print(f"Vehicles : {k}  |  Nodes : {n + 1}  |  Max coverage : {D}")
    print("Solving …")

    t0 = time.perf_counter()
    best = solve_instance(data, tot_RT, seed=int(t0 * 1000) & 0xFFFFFFFF)
    elapsed = time.perf_counter() - t0

    print(f"\nObjective : {best.objective():.2f}")
    print(f"Feasible  : {best.feasible}")
    print(f"Max dist  : {best.max_d:.2f}  (limit {D})")
    print(f"Routes cost: {best.get_routes_cost():.2f}")
    print(f"Runtime   : {elapsed:.1f}s")
    print("\nRoutes (depot = 0):")
    for i, route in enumerate(best.routes):
        tour = [0] + list(route) + [0]
        print(f"  Route {i}: {' -> '.join(map(str, tour))}")


def _cli():
    """Entry-point for ``mtspbc <instance>`` command."""
    if len(sys.argv) < 2:
        print("Usage: mtspbc <instance_path>")
        sys.exit(1)
    solve(sys.argv[1])
