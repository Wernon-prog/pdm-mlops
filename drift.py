"""Small drift helpers used by the API monitoring layer."""
import numpy as np


def psi(reference, current, edges, eps=1e-4):
    """Population Stability Index between a reference and current sample."""
    edges = np.asarray(edges, dtype=float).copy()

    if edges.size < 2:
        raise ValueError("PSI needs at least two bin edges")

    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)

    ref_share = np.clip(ref_share, eps, None)
    cur_share = np.clip(cur_share, eps, None)

    return float(
        np.sum(
            (cur_share - ref_share)
            * np.log(cur_share / ref_share)
        )
    )
