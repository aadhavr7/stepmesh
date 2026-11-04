"""Curvature and length based sizing, shared by edges and surface isoparms.

A polyline sampled along a curve gets a per-segment weight
    w = max(turn / angle_tol, length / L_ref)
where L_ref is the max edge length if set, otherwise the total length (so a
straight run contributes exactly one division). The division count is the
rounded-up weight sum and points are placed at equal steps of cumulative
weight, which is a curvature-aware arc length reparameterization.
"""
from __future__ import annotations

import math

import numpy as np


def angle_between(a, b):
    """Row-wise angle between vectors, 0 where either is near zero."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    c = np.zeros(len(a))
    c[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    ang = np.arccos(np.clip(c, -1.0, 1.0))
    ang[~ok] = 0.0
    return ang


def segment_weights(points, tangents, angle_tol, max_edge, extra_dirs=None):
    """Per-segment weights for a sampled curve.

    points:   (N, 3) samples
    tangents: (N, 3) derivative vectors (used for turning)
    extra_dirs: optional (N, 3) second direction field whose turning also
                counts (surface normals along an isoparm)
    Returns (weights (N-1,), total_length)
    """
    seg = np.diff(points, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    total = float(lens.sum())
    turn = angle_between(tangents[:-1], tangents[1:])
    if extra_dirs is not None:
        turn = np.maximum(turn, angle_between(extra_dirs[:-1], extra_dirs[1:]))
    lref = max_edge if max_edge > 0 else max(total, 1e-12)
    w = np.maximum(turn / angle_tol, lens / lref)
    return w, total


def count_from_weights(wsum, length, min_edge):
    n = max(1, int(math.ceil(wsum - 1e-6)))
    if min_edge > 0 and length > 0:
        n = min(n, max(1, int(math.floor(length / min_edge))))
    return n


def distribute(params, weights, n):
    """Return n+1 parameter values at equal cumulative-weight steps.

    params: (N,) increasing sample parameters, weights: (N-1,) segment weights.
    """
    params = np.asarray(params, dtype=float)
    w = np.asarray(weights, dtype=float)
    if n <= 0:
        return np.array([params[0], params[-1]])
    # small floor so zero-weight stretches still advance (avoids flat cdf)
    w = w + 1e-9 + 1e-6 * (w.sum() / max(len(w), 1))
    cum = np.concatenate([[0.0], np.cumsum(w)])
    cum /= cum[-1]
    targets = np.linspace(0.0, 1.0, n + 1)
    out = np.interp(targets, cum, params)
    out[0], out[-1] = params[0], params[-1]
    return out
