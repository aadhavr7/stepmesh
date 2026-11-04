"""Flat numpy arrays from a polygon Mesh: triangles for display and for
triangle-only formats, polygon edges for wireframe, per-corner normals.

Ngons (which can be non-convex and curved) are projected onto their best-fit
plane and ear-clipped, never fanned.
"""
from __future__ import annotations

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
    c = P.mean(axis=0)
    nrm = np.zeros(3)
    for i in range(n):
        nrm += np.cross(P[i] - c, P[(i + 1) % n] - c)
    ln = np.linalg.norm(nrm)
    if ln < 1e-30:
        return np.array([(0, i, i + 1) for i in range(1, n - 1)])
    nrm /= ln
    ax = np.cross(nrm, [1.0, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1.0, 0])
    ax /= np.linalg.norm(ax)
    ay = np.cross(nrm, ax)
    S = np.stack([(P - c) @ ax, (P - c) @ ay], axis=1)
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

    tv, tn, tp = [], [], []
    for pi, (vv, nn) in enumerate(zip(poly_v, poly_n)):
        k = len(vv)
        if k == 3:
            loc = ((0, 1, 2),)
        elif k == 4:
            d02 = np.sum((pts[vv[0]] - pts[vv[2]]) ** 2)
            d13 = np.sum((pts[vv[1]] - pts[vv[3]]) ** 2)
            loc = ((0, 1, 2), (0, 2, 3)) if d02 <= d13 else ((0, 1, 3), (1, 2, 3))
        else:
            loc = _ngon_tris(pts[vv])
        for a, b, c in loc:
            tv.append((vv[a], vv[b], vv[c]))
            tn.append((nn[a], nn[b], nn[c]))
            tp.append(pi)
    tri_v = np.array(tv, dtype=np.int64).reshape(-1, 3)
    tri_n = np.array(tn, dtype=np.int64).reshape(-1, 3)
    tri_p = np.array(tp, dtype=np.int64)

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
