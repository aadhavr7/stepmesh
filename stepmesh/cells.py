"""Trimmed curved faces as clipped grid cells.

A trimmed face keeps its isoparametric grid. Every cell cut by a trim keeps
the part inside the trim as one polygon, and boundary points between two
grid lines become extra corners of that cell (an ngon) instead of being
triangulated into a band.

Two steps keep the result welded:
  1. insert_crossings(): before any vertex exists, every place where a grid
     line of a trimmed face crosses one of its trim edges becomes a required
     point of that edge. The edge is still discretized once and shared, so
     the neighbour face sees the same point (as an extra ngon corner when it
     is planar or trimmed).
  2. mesh_trimmed_cells(): each grid cell is clipped against the face domain
     in scaled UV (shapely). Clip vertices are mapped back to grid nodes or
     to the shared edge points, never to new points on a trim.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import shapely
from shapely.geometry import Polygon as SPoly

from .cdt import _simple_cycle, _walk, cdt, dist_to_segments, grow_regions, interior_point, pair_triangles

SKIP_FRAC = 0.15      # crossings this close (in cell sizes) to an existing edge point are not inserted
COLLAPSE_FRAC = 0.0   # interior grid nodes closer than this to a trim snap onto it (off: folding risk; slivers are merged instead)
SMALL_FRAC = 0.15     # clipped pieces smaller than this fraction of their cell merge into a neighbour


def _spacing(lines, x):
    i = int(np.clip(np.searchsorted(lines, x), 1, len(lines) - 1))
    gaps = [lines[i] - lines[i - 1]]
    if i + 1 < len(lines):
        gaps.append(lines[i + 1] - lines[i])
    if i - 2 >= 0:
        gaps.append(lines[i - 1] - lines[i - 2])
    return max(min(gaps), 1e-300)


def _bisect(c2d, axis, target, ta, tb, iters=48):
    fa = c2d.uv(ta)[axis] - target
    fb = c2d.uv(tb)[axis] - target
    if fa == 0:
        return ta
    if fb == 0:
        return tb
    if fa * fb > 0:  # not bracketed (non-monotonic pcurve piece): linear guess
        return ta + (tb - ta) * (fa / (fa - fb)) if fa != fb else 0.5 * (ta + tb)
    for _ in range(iters):
        tm = 0.5 * (ta + tb)
        fm = c2d.uv(tm)[axis] - target
        if fa * fm <= 0:
            tb, fb = tm, fm
        else:
            ta, fa = tm, fm
    return 0.5 * (ta + tb)


def refine_for_holes(fc, U, V):
    """Add a grid line through every inner trim loop that no grid line
    crosses, so no cell ever contains a whole hole (such cells would need a
    triangulated fallback). The line goes along the axis with the larger
    local spacing, through the middle of the hole."""
    U, V = list(U), list(V)
    for is_outer, uses in fc.wires:
        if is_outer:
            continue
        pts = np.vstack([u.uv_samples for u in uses])
        u0, u1 = pts[:, 0].min(), pts[:, 0].max()
        v0, v1 = pts[:, 1].min(), pts[:, 1].max()
        if any(u0 < x < u1 for x in U) or any(v0 < y < v1 for y in V):
            continue
        su = _spacing(np.array(sorted(U)), 0.5 * (u0 + u1)) * fc.su
        sv = _spacing(np.array(sorted(V)), 0.5 * (v0 + v1)) * fc.sv
        if su >= sv:
            U.append(0.5 * (u0 + u1))
        else:
            V.append(0.5 * (v0 + v1))
    return np.array(sorted(set(U))), np.array(sorted(set(V)))


def structured_fallback(m):
    """Structured faces whose opposite sides will not match fall back to the
    trimmed path anyway: decide that now so they take part in insertion.
    Returns the edges that belong to faces staying structured."""
    for fc in m.faces:
        if fc is None or fc.kind not in ("quad", "tri"):
            continue
        cnt = [sum(len(m.edges[u.eidx].params) - 1 for u in side) for side in fc.sides]
        bad = cnt[0] != cnt[2] if fc.kind == "tri" else (cnt[0] != cnt[2] or cnt[1] != cnt[3])
        if bad:
            fc.note = f"side counts {cnt}, trimmed"
            fc.kind = "trimmed"
            m._apply_aspect(fc, m.p.trim_aspect) if m.p.trim_aspect > 0 else None
    structured_edges = set()
    for fc in m.faces:
        if fc is not None and fc.kind in ("quad", "tri"):
            for side in fc.sides:
                for u in side:
                    structured_edges.add(u.eidx)
    return structured_edges


def face_crossings(m, fc, structured_edges):
    """Grid lines of one trimmed face and the edge parameters where they cross
    its trims. Returns (U, V, [(edge index, t), ...])."""
    U, V = refine_for_holes(fc, m._grid_lines(fc, None, 0), m._grid_lines(fc, None, 1))
    lines = (U * fc.su, V * fc.sv)
    scale = (fc.su, fc.sv)
    found = []
    for _, uses in fc.wires:
        for use in uses:
            ed = m.edges[use.eidx]
            if ed.degenerate or use.eidx in structured_edges:
                continue
            ts = ed.params
            S = m._scaled(fc, np.array([use.c2d.uv(t) for t in ts]))
            for k in range(len(ts) - 1):
                for axis in (0, 1):
                    a, b = S[k, axis], S[k + 1, axis]
                    lo, hi = min(a, b), max(a, b)
                    if hi - lo < 1e-12:
                        continue
                    L = lines[axis]
                    for x in L[(L > lo) & (L < hi)]:
                        h = min(_spacing(lines[0], S[k, 0]), _spacing(lines[1], S[k, 1]))
                        seg = np.linalg.norm(S[k + 1] - S[k])
                        f = (x - a) / (b - a)
                        if min(f, 1 - f) * seg < SKIP_FRAC * h:
                            continue
                        found.append((use.eidx, _bisect(use.c2d, axis, x / scale[axis], ts[k], ts[k + 1])))
    return U, V, found


def insert_crossings(m, pool=None):
    """Add grid-line / trim crossings of trimmed faces to the edge params.
    With a worker pool the per-face part runs on several cores; merging is
    order-independent, so the result is the same either way."""
    structured_edges = structured_fallback(m)
    trimmed = [fc for fc in m.faces if fc is not None and fc.kind == "trimmed"]
    if pool is not None:
        results = pool.crossings(trimmed, structured_edges)
    else:
        results = {fc.idx: face_crossings(m, fc, structured_edges) for fc in trimmed}
    pending = defaultdict(list)
    for fc in trimmed:
        U, V, found = results[fc.idx]
        fc.grid = (U, V)
        for ei, t in found:
            pending[ei].append(t)
    added = 0
    for ei, extra in pending.items():
        ed = m.edges[ei]
        span = abs(ed.t1 - ed.t0) or 1.0
        params = list(ed.params)
        for t in sorted(extra):
            if min(abs(t - q) for q in params) > 1e-7 * span:
                params.append(t)
                added += 1
        ed.params = np.array(sorted(params))
    m.crossings_added = added


def mesh_trimmed_cells(m, fc) -> bool:
    loops = m._loops(fc)
    if getattr(fc, "grid", None) is not None:
        U, V = fc.grid
    else:
        U, V = refine_for_holes(fc, m._grid_lines(fc, loops, 0), m._grid_lines(fc, loops, 1))
    if len(U) < 2 or len(V) < 2:
        return False
    Us, Vs = U * fc.su, V * fc.sv
    LP_uv = np.vstack([uv for _, uv, _ in loops])
    LP_g = [g for _, _, gs in loops for g in gs]
    LP_s = m._scaled(fc, LP_uv)
    rings = [m._scaled(fc, uv) for _, uv, _ in loops]
    # loop membership of every boundary point: (loop start, loop length)
    loop_of = []
    start = 0
    for _, uv, _ in loops:
        loop_of += [(start, len(uv))] * len(uv)
        start += len(uv)
    outer = [i for i, (o, _, _) in enumerate(loops) if o]
    if len(outer) != 1:
        areas = [abs(SPoly(r).area) for r in rings]
        outer = [int(np.argmax(areas))]
    oi = outer[0]
    domain = SPoly(rings[oi], [r for i, r in enumerate(rings) if i != oi])
    if not domain.is_valid:
        m.log.append(f"face {fc.idx}: trim loops invalid in UV, triangle band fallback")
        return False
    shapely.prepare(domain)
    segA, segB, seg0, seg1 = [], [], [], []
    start = 0
    for r in rings:
        n_r = len(r)
        segA.append(r)
        segB.append(np.roll(r, -1, axis=0))
        seg0 += [start + k for k in range(n_r)]
        seg1 += [start + (k + 1) % n_r for k in range(n_r)]
        start += n_r
    A, B = np.vstack(segA), np.vstack(segB)
    seg0, seg1 = np.array(seg0), np.array(seg1)
    D = B - A
    DD = np.maximum(np.einsum("ij,ij->i", D, D), 1e-300)
    diag = float(np.hypot(*(LP_s.max(axis=0) - LP_s.min(axis=0)))) + 1e-30
    tol = 1e-7 * diag
    on_tol = 1e-6 * diag
    nu, nv = len(U), len(V)
    NI, NJ = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    Q = np.stack([Us[NI], Vs[NJ]], axis=-1).reshape(-1, 2)
    d2 = ((Q[:, None, :] - LP_s[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1).reshape(nu, nv)
    near_d = np.sqrt(d2.min(axis=1)).reshape(nu, nv)
    dist = dist_to_segments(Q, A, B).reshape(nu, nv)

    nk_cache = {}

    def node_key(i, j):
        k = nk_cache.get((i, j))
        if k is not None:
            return k
        if near_d[i, j] <= tol:
            k = ("L", int(nearest[i, j]))
        # a node lying on a trim but not on an edge point would be a
        # T-junction: it has to become the nearest edge point
        elif dist[i, j] <= 10 * on_tol or (
                COLLAPSE_FRAC > 0 and dist[i, j] < COLLAPSE_FRAC * min(_spacing(Us, Us[i]), _spacing(Vs, Vs[j]))):
            k = ("L", int(nearest[i, j]))
        else:
            k = ("N", i, j)
        nk_cache[(i, j)] = k
        return k

    LPx, LPy = LP_s[:, 0].tolist(), LP_s[:, 1].tolist()
    Usl, Vsl = Us.tolist(), Vs.tolist()

    def key_xy(k):
        return (LPx[k[1]], LPy[k[1]]) if k[0] == "L" else (Usl[k[1]], Vsl[k[2]])

    # per-cell search windows: trim points and segments near the current cell
    seg_lo = np.minimum(A, B)
    seg_hi = np.maximum(A, B)
    win = {"lp": None, "seg": None, "margin": 0.0}

    def set_window(i, j):
        mrg = max(Us[i + 1] - Us[i], Vs[j + 1] - Vs[j])
        x0, x1, y0, y1 = Us[i] - mrg, Us[i + 1] + mrg, Vs[j] - mrg, Vs[j + 1] + mrg
        win["lp"] = np.nonzero((LP_s[:, 0] >= x0) & (LP_s[:, 0] <= x1) & (LP_s[:, 1] >= y0) & (LP_s[:, 1] <= y1))[0]
        win["seg"] = np.nonzero((seg_hi[:, 0] >= x0) & (seg_lo[:, 0] <= x1) & (seg_hi[:, 1] >= y0) & (seg_lo[:, 1] <= y1))[0]
        win["margin"] = mrg

    def key_s(k):
        return LP_s[k[1]] if k[0] == "L" else np.array([Us[k[1]], Vs[k[2]]])

    def key_uv(k):
        return LP_uv[k[1]] if k[0] == "L" else np.array([U[k[1]], V[k[2]]])

    node_gid = {}

    def key_gid(k):
        if k[0] == "L":
            return LP_g[k[1]]
        if k not in node_gid:
            node_gid[k] = m._new_point(fc, (U[k[1]], V[k[2]]))
        return node_gid[k]

    snap_worst = 0.0

    def map_vertex(q, i, j):
        nonlocal snap_worst
        for x, y in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
            if abs(q[0] - Us[x]) <= tol and abs(q[1] - Vs[y]) <= tol:
                return node_key(x, y)
        cand = win["lp"]
        if cand is not None and len(cand):
            dc = np.hypot(LP_s[cand, 0] - q[0], LP_s[cand, 1] - q[1])
            a = int(dc.argmin())
            if dc[a] <= win["margin"]:
                k, dk = int(cand[a]), float(dc[a])
            else:
                cand = None
        if cand is None or not len(cand):
            d = np.hypot(LP_s[:, 0] - q[0], LP_s[:, 1] - q[1])
            k = int(d.argmin())
            dk = float(d[k])
        h = max(Us[i + 1] - Us[i], Vs[j + 1] - Vs[j])
        if dk > tol:
            # a crossing of a cell side with a trim segment: snap it the same
            # way in every cell that sees it, using the exact crossing on the
            # segment instead of the clipper's (noisy) coordinates
            axis, X = None, None
            for xs in (Us[i], Us[i + 1]):
                if abs(q[0] - xs) <= tol:
                    axis, X = 0, xs
            if axis is None:
                for ys in (Vs[j], Vs[j + 1]):
                    if abs(q[1] - ys) <= tol:
                        axis, X = 1, ys
            if axis is not None:
                sidx = None
                sc = win["seg"]
                for pool in ((sc,) if sc is not None and len(sc) else ()) + (None,):
                    Ap = A if pool is None else A[pool]
                    Dp = D if pool is None else D[pool]
                    DDp = DD if pool is None else DD[pool]
                    rel = q[None, :] - Ap
                    t = np.clip(np.einsum("ij,ij->i", rel, Dp) / DDp, 0.0, 1.0)
                    dseg = np.linalg.norm(rel - t[:, None] * Dp, axis=1)
                    a = int(dseg.argmin())
                    if pool is None or dseg[a] <= win["margin"]:
                        sidx = a if pool is None else int(pool[a])
                        tsel = float(t[a])
                        break
                da = D[sidx, axis]
                ts = (X - A[sidx, axis]) / da if abs(da) > 1e-300 else tsel
                k = int(seg0[sidx] if ts < 0.5 else seg1[sidx])
                dk = float(np.hypot(*(LP_s[k] - q)))
        snap_worst = max(snap_worst, dk / h)
        return ("L", k)

    def ring_keys(coords, i, j):
        """Map a clipped ring to keys. Ring sides that follow the trim (not a
        cell side) get every edge point between their ends restored: the
        clipper may merge nearly coincident points, which would leave
        T-junctions against the neighbour face."""
        x0, x1, y0, y1 = Us[i], Us[i + 1], Vs[j], Vs[j + 1]
        pad = 1e-6 * diag

        def on_rect_side(a, b):
            for xs in (x0, x1):
                if abs(a[0] - xs) <= tol and abs(b[0] - xs) <= tol:
                    return True
            for ys in (y0, y1):
                if abs(a[1] - ys) <= tol and abs(b[1] - ys) <= tol:
                    return True
            return False

        keys = [map_vertex(np.array(c), i, j) for c in coords]
        out = []
        n = len(keys)
        for a in range(n):
            ka, kb = keys[a], keys[(a + 1) % n]
            out.append(ka)
            if ka[0] != "L" or kb[0] != "L" or ka == kb or on_rect_side(coords[a], coords[(a + 1) % n]):
                continue
            sa, ln = loop_of[ka[1]]
            sb, _ = loop_of[kb[1]]
            if sa != sb:
                continue
            pa, pb = ka[1] - sa, kb[1] - sa
            fwd = (pb - pa) % ln
            bwd = ln - fwd
            if fwd <= bwd:
                arc = [sa + (pa + t) % ln for t in range(1, fwd)]
            else:
                arc = [sa + (pa - t) % ln for t in range(1, bwd)]
            if not arc or len(arc) > 64:
                continue
            P = LP_s[arc]
            if (P[:, 0] < x0 - pad).any() or (P[:, 0] > x1 + pad).any() or (P[:, 1] < y0 - pad).any() or (P[:, 1] > y1 + pad).any():
                continue
            out += [("L", int(k)) for k in arc]
        return out

    def dedupe(keys):
        out = []
        for k in keys:
            if not out or out[-1] != k:
                out.append(k)
        while len(out) > 1 and out[0] == out[-1]:
            out.pop()
        return out

    # trim points bucketed by grid cell (with tolerance), for densify
    bucket = {}
    i_lo = np.clip(np.searchsorted(Us, LP_s[:, 0] - on_tol, side="left") - 1, 0, len(Us) - 2)
    i_hi = np.clip(np.searchsorted(Us, LP_s[:, 0] + on_tol, side="right") - 1, 0, len(Us) - 2)
    j_lo = np.clip(np.searchsorted(Vs, LP_s[:, 1] - on_tol, side="left") - 1, 0, len(Vs) - 2)
    j_hi = np.clip(np.searchsorted(Vs, LP_s[:, 1] + on_tol, side="right") - 1, 0, len(Vs) - 2)
    for k in range(len(LP_s)):
        for bi in range(int(i_lo[k]), int(i_hi[k]) + 1):
            for bj in range(int(j_lo[k]), int(j_hi[k]) + 1):
                bucket.setdefault((bi, bj), []).append(k)

    def densify(keys, ci, cj):
        """Insert boundary points lying on a polygon side (straight trims along grid lines)."""
        if not any(k[0] == "L" for k in keys):
            return dedupe(keys)
        # the cell's bucket is enough only if the whole piece lies in the cell;
        # corners snapped to trim points outside it need the full search
        xy = [key_xy(k) for k in keys]
        if (min(c[0] for c in xy) >= Usl[ci] - on_tol and max(c[0] for c in xy) <= Usl[ci + 1] + on_tol
                and min(c[1] for c in xy) >= Vsl[cj] - on_tol and max(c[1] for c in xy) <= Vsl[cj + 1] + on_tol):
            cand = bucket.get((ci, cj))
            if not cand:
                return dedupe(keys)
        else:
            x0, x1 = min(c[0] for c in xy) - on_tol, max(c[0] for c in xy) + on_tol
            y0, y1 = min(c[1] for c in xy) - on_tol, max(c[1] for c in xy) + on_tol
            cand = np.nonzero((LP_s[:, 0] >= x0) & (LP_s[:, 0] <= x1) & (LP_s[:, 1] >= y0) & (LP_s[:, 1] <= y1))[0].tolist()
        out = []
        n = len(keys)
        for a in range(n):
            ka, kb = keys[a], keys[(a + 1) % n]
            out.append(ka)
            if ka[0] != "L" and kb[0] != "L":
                continue
            ax, ay = key_xy(ka)
            bx, by = key_xy(kb)
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-30:
                continue
            hits = []
            for k in cand:
                rx, ry = LPx[k] - ax, LPy[k] - ay
                t = (rx * dx + ry * dy) / L2
                if 1e-9 < t < 1 - 1e-9:
                    px, py = rx - t * dx, ry - t * dy
                    if math.sqrt(px * px + py * py) < on_tol:
                        hits.append((t, k))
            for _, k in sorted(hits):
                kk = ("L", k)
                if kk != ka and kk != kb:
                    out.append(kk)
        return dedupe(out)

    pieces = []        # [keys, cell_area, mergeable, i, j]
    # classify and clip all cells in batched shapely calls
    CI, CJ = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    CI, CJ = CI.ravel(), CJ.ravel()
    boxes = shapely.box(Us[CI], Vs[CJ], Us[CI + 1], Vs[CJ + 1])
    inside_cells = shapely.contains(domain, boxes)
    cut_cells = shapely.intersects(domain, boxes) & ~inside_cells
    clipped = dict(zip(np.nonzero(cut_cells)[0].tolist(), shapely.intersection(boxes[cut_cells], domain)))
    for c in range(len(CI)):
        i, j = int(CI[c]), int(CJ[c])
        if True:
            carea = (Us[i + 1] - Us[i]) * (Vs[j + 1] - Vs[j])
            if inside_cells[c]:
                keys = [node_key(i, j), node_key(i + 1, j), node_key(i + 1, j + 1), node_key(i, j + 1)]
                pieces.append([dedupe(keys), carea, True, i, j])
                continue
            if not cut_cells[c]:
                continue
            set_window(i, j)
            inter = clipped[c]
            for g in getattr(inter, "geoms", [inter]):
                if g.geom_type != "Polygon" or g.area < 1e-9 * carea:
                    continue
                ext = ring_keys(list(g.exterior.coords)[:-1], i, j)
                if len(g.interiors) == 0:
                    pieces.append([dedupe(ext), carea, True, i, j])
                    continue
                # a whole hole inside one cell: triangulate the piece, regrow into ngons
                ringkeys = [dedupe(ext)] + [dedupe(ring_keys(list(r.coords)[:-1], i, j)) for r in g.interiors]
                for poly in _holed_piece(ringkeys, key_s):
                    pieces.append([poly, carea, False, i, j])

    # shapely does not promise a winding: make every piece CCW in UV so
    # neighbours share edges in opposite directions
    for p in pieces:
        if len(p[0]) >= 3 and _area(p[0], key_xy) < 0:
            p[0] = p[0][::-1]
    pieces = [[densify(k, ci, cj), a, mg] for k, a, mg, ci, cj in pieces if len(k) >= 3]
    pieces = _merge_small(pieces, key_xy, {})

    out = m.p.output
    for keys, _, _ in pieces:
        if len(keys) < 3:
            continue
        gids = [key_gid(k) for k in keys]
        # collapse consecutive equal gids (poles)
        g2, k2 = [], []
        for g, k in zip(gids, keys):
            if not g2 or g2[-1] != g:
                g2.append(g)
                k2.append(k)
        while len(g2) > 1 and g2[0] == g2[-1]:
            g2.pop()
            k2.pop()
        if len(set(g2)) < 3:
            continue
        # an outline that touches itself at a vertex is two polygons sharing it
        for part in _split_pinches(list(range(len(g2))), g2):
            pg = [g2[a] for a in part]
            pk = [k2[a] for a in part]
            if len(set(pg)) < 3:
                continue
            if out == "ngons" or len(pg) <= 4:
                m._emit(fc, pg, [key_uv(k) for k in pk], sign=1.0)
                continue
            S = np.array([key_s(k) for k in pk])
            if len({(round(x, 12), round(y, 12)) for x, y in S}) < len(S):
                tris = [(0, a, a + 1) for a in range(1, len(pk) - 1)]
            else:
                try:
                    tris, _ = cdt(S, [(a, (a + 1) % len(pk)) for a in range(len(pk))])
                    tris = [t for t in tris if max(t) < len(pk)]
                except Exception:
                    tris = [(0, a, a + 1) for a in range(1, len(pk) - 1)]
            polys = pair_triangles(tris, S) if out == "quads" else [list(t) for t in tris]
            for poly in polys:
                m._emit(fc, [pg[a] for a in poly], [key_uv(pk[a]) for a in poly], sign=1.0)
    if snap_worst > 0.6:
        m.log.append(f"face {fc.idx}: trim snap moved a clip point {snap_worst:.2f} cells")
    return True


def _holed_piece(ringkeys, key_s):
    allk = []
    index = {}
    segs = []
    for r in ringkeys:
        ids = []
        for k in r:
            if k not in index:
                index[k] = len(allk)
                allk.append(k)
            ids.append(index[k])
        segs += [(ids[a], ids[(a + 1) % len(ids)]) for a in range(len(ids))]
    pts = np.array([key_s(k) for k in allk])
    holes = [interior_point(np.array([key_s(k) for k in r])) for r in ringkeys[1:]]
    tris, verts = cdt(pts, segs, holes)
    tris = [t for t in tris if max(t) < len(allk)]
    return [[allk[a] for a in cyc] for cyc in grow_regions(tris, pts)]


def _area(keys, key_xy, cache=None):
    """Signed area in scaled UV. Kept on numpy's summation: the exact rounding
    decides the merge order of equal-sized slivers, so it must not change.
    Results are cached per outline (the merge pass asks repeatedly)."""
    t = tuple(keys)
    if cache is not None:
        a = cache.get(t)
        if a is not None:
            return a
    P = np.array([key_xy(k) for k in keys], dtype=float)
    x, y = P[:, 0], P[:, 1]
    a = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if cache is not None:
        cache[t] = a
    return a


def _merge_small(pieces, key_s, cache=None):
    """Merge slivers left by trims passing near grid lines into a neighbour."""
    def edges_of(keys):
        return [(keys[a], keys[(a + 1) % len(keys)]) for a in range(len(keys))]

    alive = [True] * len(pieces)
    owner = {}
    for pi, (keys, _, _) in enumerate(pieces):
        for e in edges_of(keys):
            owner[e] = pi
    order = sorted(range(len(pieces)), key=lambda i: abs(_area(pieces[i][0], key_s, cache)) / pieces[i][1])
    for pi in order:
        keys, carea, mergeable = pieces[pi]
        if not alive[pi] or not mergeable:
            continue
        if abs(_area(keys, key_s, cache)) >= SMALL_FRAC * carea:
            break
        best, best_n = None, 0
        cnt = defaultdict(int)
        for a, b in edges_of(keys):
            q = owner.get((b, a))
            if q is not None and q != pi and alive[q] and pieces[q][2]:
                cnt[q] += 1
        for q, n in cnt.items():
            if n > best_n:
                best, best_n = q, n
        if best is None:
            continue
        ea = set(edges_of(keys))
        eb = set(edges_of(pieces[best][0]))
        cand = {e for e in ea if (e[1], e[0]) not in eb} | {e for e in eb if (e[1], e[0]) not in ea}
        if not cand or not _simple_cycle(cand):
            continue
        merged = _walk(cand)
        alive[pi] = False
        pieces[best] = [merged, max(pieces[best][1], carea), True]
        for e in edges_of(merged):
            owner[e] = best
    return [p for p, a in zip(pieces, alive) if a]


def _split_pinches(idx, ids):
    """Split a closed index loop wherever the same id appears twice."""
    seen = {}
    for pos, a in enumerate(idx):
        g = ids[a]
        if g in seen:
            i = seen[g]
            first = idx[i:pos]
            second = idx[pos:] + idx[:i]
            return _split_pinches(first, ids) + _split_pinches(second, ids)
        seen[g] = pos
    return [idx]
