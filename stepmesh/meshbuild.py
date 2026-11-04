"""Flat numpy arrays from a polygon Mesh: triangles for display and for
triangle-only formats, polygon edges for wireframe, per-corner normals.

Ngons (which can be non-convex and curved) are projected onto their best-fit
plane and ear-clipped, never fanned.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .export_obj import cluster_normals
from .meshdata import Mesh

import mapbox_earcut as _earcut


@dataclass
class MeshArrays:
    points: np.ndarray        # (V,3) used positions
    normals: np.ndarray       # (N,3) clustered corner normals
    tri_v: np.ndarray         # (T,3) indices into points
    tri_n: np.ndarray         # (T,3) indices into normals
    tri_poly: np.ndarray      # (T,) source polygon index
    tri_face: np.ndarray      # (T,) B-rep face index
    tri_sides: np.ndarray     # (T,) side count of the source polygon (3, 4, 5+)
    edges: np.ndarray         # (E,2) unique polygon edges, indices into points
    open_edges: np.ndarray    # (K,2) edges used by one polygon only
    poly_sides: np.ndarray    # (P,) side count per polygon
    poly_v: list              # per polygon: indices into points
    poly_n: list              # per polygon: indices into normals
    poly_face: np.ndarray
    poly_body: np.ndarray


def _ngon_tris(P):
    """Triangulate one polygon given its 3D corners; returns (k,3) local indices."""
    n = len(P)
    Q = P - P.mean(axis=0)
    R = np.roll(Q, -1, axis=0)
    C = np.empty((n, 3))
    C[:, 0] = Q[:, 1] * R[:, 2] - Q[:, 2] * R[:, 1]
    C[:, 1] = Q[:, 2] * R[:, 0] - Q[:, 0] * R[:, 2]
    C[:, 2] = Q[:, 0] * R[:, 1] - Q[:, 1] * R[:, 0]
    nx, ny, nz = (float(x) for x in C.sum(axis=0))   # Newell normal
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln < 1e-30:
        return np.array([(0, i, i + 1) for i in range(1, n - 1)])
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    ex = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
    ax = (ny * ex[2] - nz * ex[1], nz * ex[0] - nx * ex[2], nx * ex[1] - ny * ex[0])
    la = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2)
    ax = (ax[0] / la, ax[1] / la, ax[2] / la)
    ay = (ny * ax[2] - nz * ax[1], nz * ax[0] - nx * ax[2], nx * ax[1] - ny * ax[0])
    S = Q @ np.array([ax, ay]).T
    idx = np.asarray(_earcut.triangulate_float64(S, np.array([n], dtype=np.uint32)), dtype=np.int64)
    if len(idx) == 3 * (n - 2):
        return idx.reshape(-1, 3)
    try:
        from .cdt import cdt
        t, _ = cdt(S, [(i, (i + 1) % n) for i in range(n)])
        t = [tt for tt in t if max(tt) < n]
        if len(t) == n - 2:
            return np.array(t)
    except Exception:
        pass
    return np.array([(0, i, i + 1) for i in range(1, n - 1)])


def build_arrays(mesh: Mesh, sharp_deg: float = 20.0) -> MeshArrays:
    used = sorted({v for p in mesh.polys for v in p.verts})
    remap = {g: i for i, g in enumerate(used)}
    pts = np.array([mesh.points[g] for g in used], dtype=np.float64).reshape(-1, 3)
    nlist, poly_nidx = cluster_normals(mesh, sharp_deg)
    normals = np.array(nlist, dtype=np.float64).reshape(-1, 3)

    poly_v = [[remap[v] for v in p.verts] for p in mesh.polys]
    poly_n = poly_nidx
    sides = np.array([len(v) for v in poly_v], dtype=np.int32)
    pface = np.array([p.face for p in mesh.polys], dtype=np.int32)
    pbody = np.array([p.body for p in mesh.polys], dtype=np.int32)

    # triangles, quads (vectorized) and ngons, then put back in polygon order
    parts = []   # (poly index, sub index, tri_v, tri_n)
    i3 = np.nonzero(sides == 3)[0]
    if len(i3):
        parts.append((i3, np.zeros(len(i3), np.int64),
                      np.array([poly_v[i] for i in i3], np.int64), np.array([poly_n[i] for i in i3], np.int64)))
    i4 = np.nonzero(sides == 4)[0]
    if len(i4):
        V4 = np.array([poly_v[i] for i in i4], np.int64)
        N4 = np.array([poly_n[i] for i in i4], np.int64)
        d02 = ((pts[V4[:, 0]] - pts[V4[:, 2]]) ** 2).sum(axis=1)
        d13 = ((pts[V4[:, 1]] - pts[V4[:, 3]]) ** 2).sum(axis=1)
        short02 = (d02 <= d13)[:, None]
        for sub, (la, lb) in enumerate((((0, 1, 2), (0, 1, 3)), ((0, 2, 3), (1, 2, 3)))):
            loc = np.where(short02, np.array(la), np.array(lb))
            parts.append((i4, np.full(len(i4), sub, np.int64),
                          np.take_along_axis(V4, loc, axis=1), np.take_along_axis(N4, loc, axis=1)))
    for pi in np.nonzero(sides > 4)[0].tolist():
        vv, nn = poly_v[pi], poly_n[pi]
        loc = _ngon_tris(pts[vv])
        vv, nn = np.asarray(vv, np.int64), np.asarray(nn, np.int64)
        parts.append((np.full(len(loc), pi, np.int64), np.arange(len(loc), dtype=np.int64), vv[loc], nn[loc]))
    if parts:
        P_ = np.concatenate([p[0] for p in parts])
        S_ = np.concatenate([p[1] for p in parts])
        order = np.lexsort((S_, P_))
        tri_v = np.concatenate([p[2] for p in parts])[order]
        tri_n = np.concatenate([p[3] for p in parts])[order]
        tri_p = P_[order]
    else:
        tri_v = np.zeros((0, 3), np.int64)
        tri_n = np.zeros((0, 3), np.int64)
        tri_p = np.zeros(0, np.int64)

    # unique polygon edges, with use counts
    e = []
    for vv in poly_v:
        k = len(vv)
        for i in range(k):
            a, b = vv[i], vv[(i + 1) % k]
            e.append((min(a, b), max(a, b)))
    if e:
        E, cnt = np.unique(np.array(e, dtype=np.int64), axis=0, return_counts=True)
        open_e = E[cnt == 1]
    else:
        E = np.zeros((0, 2), dtype=np.int64)
        open_e = E
    return MeshArrays(points=pts, normals=normals, tri_v=tri_v, tri_n=tri_n, tri_poly=tri_p,
                      tri_face=pface[tri_p] if len(tri_p) else np.zeros(0, np.int32),
                      tri_sides=sides[tri_p] if len(tri_p) else np.zeros(0, np.int32),
                      edges=E, open_edges=open_e, poly_sides=sides, poly_v=poly_v, poly_n=poly_n,
                      poly_face=pface, poly_body=pbody)
