"""Post-pass safety net for watertightness.

1. T-junction repair: a polygon side a->b that has no partner, where a and b
   both sit on the boundary of the polygon's B-rep face, is replaced by the
   path of shared edge points between them (along B-rep edges, through B-rep
   vertices). The polygon simply gains corners, which ngon output absorbs
   naturally, the same way boundary points between grid lines are absorbed.
2. Gap fill: remaining small closed loops of unpartnered sides become one
   polygon each.
Polygons that grew beyond 4 corners are re-split for quads / tris output.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np

from .cdt import cdt, pair_triangles
from .meshdata import Polygon

MAX_PATH = 16
MAX_FILL = 32


def _half_edges(polys):
    he = Counter()
    for p in polys:
        n = len(p.verts)
        for i in range(n):
            he[(p.verts[i], p.verts[(i + 1) % n])] += 1
    return he


def _open(he):
    return [(a, b) for (a, b), c in he.items() if c == 1 and (b, a) not in he]


def repair(m) -> dict:
    mesh = m.mesh
    # boundary graph of every face: consecutive shared edge points
    face_adj = defaultdict(lambda: defaultdict(set))
    for fc in m.faces:
        if fc is None:
            continue
        for _, uses in fc.wires:
            for u in uses:
                g = m.edges[u.eidx].gids
                for a, b in zip(g[:-1], g[1:]):
                    if a != b:
                        face_adj[fc.idx][a].add(b)
                        face_adj[fc.idx][b].add(a)

    he = _half_edges(mesh.polys)
    where = {}
    for pi, p in enumerate(mesh.polys):
        n = len(p.verts)
        for i in range(n):
            where[(p.verts[i], p.verts[(i + 1) % n])] = pi

    inserted = 0
    touched = set()
    for a, b in _open(he):
        pi = where.get((a, b))
        if pi is None:
            continue
        p = mesh.polys[pi]
        adj = face_adj.get(p.face)
        if not adj or a not in adj or b not in adj or b in adj[a]:
            continue
        # BFS a -> b over the face boundary, avoiding vertices already in the polygon
        used = set(p.verts) - {b}
        prev = {a: None}
        dq = deque([(a, 0)])
        found = False
        while dq and not found:
            x, d = dq.popleft()
            if d >= MAX_PATH:
                continue
            for y in adj[x]:
                if y in prev or (y in used and y != b):
                    continue
                prev[y] = x
                if y == b:
                    found = True
                    break
                dq.append((y, d + 1))
        if not found:
            continue
        path = []
        y = prev[b]
        while y != a:
            path.append(y)
            y = prev[y]
        path.reverse()
        k = p.verts.index(a)
        nrm = p.normals[k]
        p.verts[k + 1:k + 1] = path
        p.normals[k + 1:k + 1] = [nrm] * len(path)
        inserted += len(path)
        touched.add(pi)
        # keep the lookup current for later sides of the same polygon
        n = len(p.verts)
        for i in range(n):
            where[(p.verts[i], p.verts[(i + 1) % n])] = pi

    # gap fill (never across free B-rep edges: those are real open boundaries)
    edge_faces = defaultdict(int)
    for fc in m.faces:
        if fc is None:
            continue
        for ei in {u.eidx for _, uses in fc.wires for u in uses}:
            edge_faces[ei] += 1
    free_pairs = set()
    for ei, ed in enumerate(m.edges):
        if edge_faces[ei] == 1 and ed.gids:
            for a, b in zip(ed.gids[:-1], ed.gids[1:]):
                free_pairs.add((min(a, b), max(a, b)))
    he = _half_edges(mesh.polys)
    nxt = defaultdict(list)
    for a, b in _open(he):
        nxt[b].append(a)
    vnormal = {}
    vface = {}
    for p in mesh.polys:
        for v, n in zip(p.verts, p.normals):
            vnormal.setdefault(v, n)
            vface.setdefault(v, (p.face, p.body))
    filled = 0
    seen = set()
    for start in list(nxt):
        if start in seen or len(nxt[start]) != 1:
            continue
        cyc = [start]
        x = nxt[start][0]
        ok = True
        while x != start:
            if x in cyc or len(nxt.get(x, [])) != 1 or len(cyc) > MAX_FILL:
                ok = False
                break
            cyc.append(x)
            x = nxt[x][0]
        seen.update(cyc)
        if not ok or len(cyc) < 3:
            continue
        if any((min(x, y), max(x, y)) in free_pairs for x, y in zip(cyc, cyc[1:] + cyc[:1])):
            continue
        face, body = vface[cyc[0]]
        normals = [vnormal[v] for v in cyc]
        if mesh.add_poly(cyc, normals, face, body):
            filled += 1
            touched.add(len(mesh.polys) - 1)

    if m.p.output != "ngons" and touched:
        _resplit(mesh, touched, m.p.output)
    return {"inserted": inserted, "polys_fixed": len(touched) - filled, "filled": filled}


def _resplit(mesh, touched, output):
    keep, extra = [], []
    for pi, p in enumerate(mesh.polys):
        if pi not in touched or len(p.verts) <= (4 if output == "quads" else 3):
            keep.append(p)
            continue
        P = np.array([mesh.points[v] for v in p.verts])
        c = P.mean(axis=0)
        nrm = np.zeros(3)
        for i in range(len(P)):
            nrm += np.cross(P[i] - c, P[(i + 1) % len(P)] - c)
        nrm /= (np.linalg.norm(nrm) or 1.0)
        ax = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < 0.9 else [0, 1, 0])
        ax /= np.linalg.norm(ax)
        ay = np.cross(nrm, ax)
        S = np.stack([(P - c) @ ax, (P - c) @ ay], axis=1)
        n = len(P)
        try:
            tris, verts = cdt(S, [(i, (i + 1) % n) for i in range(n)])
            tris = [tuple(t) for t in tris if max(t) < n]
            if len(tris) != n - 2:
                raise ValueError
        except Exception:
            tris = [(0, i, i + 1) for i in range(1, n - 1)]
        polys = pair_triangles(tris, S) if output == "quads" else [list(t) for t in tris]
        for q in polys:
            extra.append(Polygon([p.verts[i] for i in q], [p.normals[i] for i in q], p.face, p.body))
    mesh.polys = keep + extra
