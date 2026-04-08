import numpy as np


def get_point_to_segment_distance_numpy(a, b, c) -> float:
    """Minimum distance from point c to segment [a, b]."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v = b - a
    w = c - a
    dot_vv = np.dot(v, v)
    if dot_vv == 0:
        return float(np.linalg.norm(w))
    t = np.dot(w, v) / dot_vv
    if t <= 0:
        return float(np.linalg.norm(w))
    if t >= 1:
        return float(np.linalg.norm(c - b))
    return float(np.linalg.norm(c - (a + t * v)))
