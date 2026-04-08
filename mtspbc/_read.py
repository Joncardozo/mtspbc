import math

import numpy as np


def read_instance(file_path: str) -> dict:
    """
    Read a mTSPBC instance file.

    File format
    -----------
    Line 1 : ``m  n  d``  — number of vehicles, clients, max coverage distance.
    Lines 2…n+2 : ``x  y`` coordinates (depot at line 2, clients after).

    Returns
    -------
    dict with keys: node_coord, distances, D, k, dimension.
    """
    with open(file_path) as f:
        header = f.readline().strip().split()
        num_vehicles = int(header[0])
        num_clients = int(header[1])
        D_max = float(header[2])

        coords = []
        for _ in range(num_clients + 1):
            x, y = map(float, f.readline().strip().split())
            coords.append([x, y])

    coordinates = np.array(coords)

    # Euclidean distance matrix (rounded to nearest integer, TSPLIB convention)
    n = num_clients + 1
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i][j] = round(np.linalg.norm(coordinates[i] - coordinates[j]))

    return {
        "node_coord": coordinates,
        "distances": dists,
        "D": D_max,
        "k": num_vehicles,
        "dimension": num_clients,
    }


def build_distance_matrix(node_coord, mode: str = "integer") -> np.ndarray:
    """Recompute the distance matrix (exact or integer-rounded Euclidean)."""
    n = len(node_coord)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = node_coord[i][0] - node_coord[j][0]
            dy = node_coord[i][1] - node_coord[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dists[i][j] = int(d + 0.5) if mode == "integer" else d
    return dists
