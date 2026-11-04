"""Constrained Delaunay wrapper (Shewchuk's Triangle) and triangle-pair to quad merging."""
from __future__ import annotations

import math

import numpy as np
import triangle as tr


class CDTError(RuntimeError):
    pass


def cdt(points, segments, holes=None, opts="pQ"):
    """Constrained triangulation of a planar straight line graph.

    Returns (triangles (T,3) int, vertices (V,2)); V > len(points) means
    Triangle had to insert vertices (crossing segments). Callers decide.
    """
    pts = np.asarray(points, dtype=float)
    data = {"vertices": pts}
    if len(segments):
        data["segments"] = np.asarray(segments, dtype=np.int32)
    if holes is not None and len(holes):
        data["holes"] = np.asarray(holes, dtype=float)
    try:
        out = tr.triangulate(data, opts)
    except Exception as exc:  # Triangle raises plain RuntimeError on bad input
        raise CDTError(str(exc)) from exc
    if "triangles" not in out:
        return np.zeros((0, 3), dtype=int), pts
    return np.asarray(out["triangles"], dtype=int), np.asarray(out["vertices"], dtype=float)


def point_in_loops(q, loops):
    """Even-odd point in polygon test of points q (N,2) against closed loops."""
    q = np.atleast_2d(q)
    inside = np.zeros(len(q), dtype=bool)
    for L in loops:
        a = L
        b = np.roll(L, -1, axis=0)
        ay, by = a[:, 1][None, :], b[:, 1][None, :]
        ax, bx = a[:, 0][None, :], b[:, 0][None, :]
        qx, qy = q[:, 0][:, None], q[:, 1][:, None]
        cond = (ay > qy) != (by > qy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (bx - ax) * (qy - ay) / (by - ay) + ax
        hit = cond & (qx < xint)
        inside ^= (hit.sum(axis=1) % 2).astype(bool)
    return inside


def dist_to_segments(q, A, B):
    """Min distance from points q (N,2) to segments A->B (M,2)."""
    q = np.atleast_2d(q)
    if len(A) == 0:
        return np.full(len(q), np.inf)
    d = B - A
    dd = np.einsum("ij,ij->i", d, d)
    dd[dd < 1e-30] = 1e-30
    rel = q[:, None, :] - A[None, :, :]
    t = np.clip(np.einsum("nmj,mj->nm", rel, d) / dd[None, :], 0.0, 1.0)
    proj = A[None, :, :] + t[..., None] * d[None, :, :]
    return np.sqrt(((q[:, None, :] - proj) ** 2).sum(axis=2)).min(axis=1)


def interior_point(loop):
    """A point strictly inside a simple closed polygon (N,2)."""
    L = np.asarray(loop)
    seg = np.roll(L, -1, axis=0) - L
    lens = np.linalg.norm(seg, axis=1)
    order = np.argsort(-lens)
    span = max(np.ptp(L[:, 0]), np.ptp(L[:, 1]), 1e-12)
    for k in order[: min(len(order), 8)]:
        if lens[k] < 1e-14:
            continue
        mid = L[k] + 0.5 * seg[k]
        nrm = np.array([-seg[k][1], seg[k][0]]) / lens[k]
        for eps in (1e-3, 1e-2, 1e-4):
            for s in (1.0, -1.0):
                p = mid + s * eps * span * nrm
                if point_in_loops(p, [L])[0]:
                    return p
    return L.mean(axis=0)


def _angle(a, b, c):
    """Interior angle at b (2D or 3D)."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-15 or nv < 1e-15:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / (nu * nv)))))


def pair_triangles(tris, coords2d, coords3d=None, max_angle_deg=170.0, max_fold_deg=20.0):
    """Greedily merge adjacent triangles into convex quads.

    tris: list of (a,b,c) local indices, CCW in coords2d.
    Pairs whose shared edge is longest are merged first (removes the long
    diagonals of minimal planar triangulations).
    Returns list of polygons (3 or 4 local indices), CCW.
    """
    tris = [tuple(t) for t in tris]
    edge_owner = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            edge_owner.setdefault((min(a, b), max(a, b)), []).append((ti, k))
    cands = []
    max_ang = math.radians(max_angle_deg)
    for (a, b), owners in edge_owner.items():
        if len(owners) != 2:
            continue
        (t0, k0), (t1, k1) = owners
        A = tris[t0]
        B = tris[t1]
        # quad: A[k0], opposite vertex of B, A[k0+1], opposite of A
        p = A[k0]
        q = A[(k0 + 1) % 3]
        ra = A[(k0 + 2) % 3]
        rb = B[(k1 + 2) % 3]
        quad = [ra, p, rb, q]
        P = [coords2d[i] for i in quad]
        ok = True
        for i in range(4):
            cr = np.cross(P[(i + 1) % 4] - P[i], P[(i + 2) % 4] - P[(i + 1) % 4])
            if cr <= 1e-14:
                ok = False
                break
            if _angle(P[i - 1], P[i], P[(i + 1) % 4]) > max_ang:
                ok = False
                break
        if not ok:
            continue
        if coords3d is not None:
            X = [coords3d[i] for i in (p, q, ra, rb)]
            n0 = np.cross(X[1] - X[0], X[2] - X[0])
            n1 = np.cross(X[3] - X[0], X[1] - X[0])
            fold = _angle(n0, np.zeros(3), n1) if np.linalg.norm(n0) > 0 and np.linalg.norm(n1) > 0 else 0.0
            if math.degrees(fold) > max_fold_deg:
                continue
        L = float(np.linalg.norm(coords2d[p] - coords2d[q]))
        cands.append((L, t0, t1, quad))
    cands.sort(key=lambda c: -c[0])
    used = set()
    polys = []
    for L, t0, t1, quad in cands:
        if t0 in used or t1 in used:
            continue
        used.add(t0)
        used.add(t1)
        polys.append(quad)
    for ti, t in enumerate(tris):
        if ti not in used:
            polys.append(list(t))
    return polys


def _simple_cycle(edges):
    """True if directed edges form exactly one simple closed cycle."""
    nxt = {}
    for a, b in edges:
        if a in nxt:
            return False
        nxt[a] = b
    if len(set(nxt.values())) != len(nxt):
        return False
    start = next(iter(nxt))
    v, steps = start, 0
    while True:
        v = nxt.get(v)
        steps += 1
        if v is None or steps > len(nxt):
            return False
        if v == start:
            return steps == len(nxt)


def _walk(edges):
    nxt = dict(edges)
    start = min(nxt)
    cyc, v = [start], nxt[start]
    while v != start:
        cyc.append(v)
        v = nxt[v]
    return cyc


def grow_regions(tris, coords=None):
    """Merge a triangulation into as few simple polygons (ngons) as possible.

    Regions grow triangle by triangle across shared edges and only accept a
    triangle when the region boundary stays one simple cycle (no pinch, no
    hole). A face with one hole typically ends up as two ngons, without any
    new vertex. Returns boundary cycles (vertex index lists, same winding as
    the input triangles).
    """
    tris = [tuple(int(x) for x in t) for t in tris]
    edge_tri = {}
    for ti, t in enumerate(tris):
        for k in range(3):
            edge_tri[(t[k], t[(k + 1) % 3])] = ti

    def area(t):
        if coords is None:
            return 0.0
        a, b, c = (np.asarray(coords[i]) for i in t)
        return abs(float(np.cross(b - a, c - a)))

    unassigned = set(range(len(tris)))
    regions = []
    while unassigned:
        seed = max(unassigned, key=lambda i: area(tris[i]))
        unassigned.discard(seed)
        t = tris[seed]
        bnd = {(t[k], t[(k + 1) % 3]) for k in range(3)}
        grown = True
        while grown:
            grown = False
            for a, b in sorted(bnd):
                if (a, b) not in bnd:
                    continue
                ni = edge_tri.get((b, a))
                if ni is None or ni not in unassigned:
                    continue
                tt = tris[ni]
                te = {(tt[k], tt[(k + 1) % 3]) for k in range(3)}
                cand = {e for e in bnd if (e[1], e[0]) not in te} | {e for e in te if (e[1], e[0]) not in bnd}
                if cand and _simple_cycle(cand):
                    bnd = cand
                    unassigned.discard(ni)
                    grown = True
        regions.append(_walk(bnd))
    return regions
