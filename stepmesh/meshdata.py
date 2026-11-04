"""Polygon mesh container, statistics and watertightness check."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Polygon:
    verts: list            # global vertex ids, CCW seen from outside
    normals: list          # one (3,) normal per corner
    face: int              # B-rep face index
    body: int              # solid index


@dataclass
class Mesh:
    points: list = field(default_factory=list)      # list of (3,) arrays
    polys: list = field(default_factory=list)
    face_names: dict = field(default_factory=dict)  # face index -> group name
    body_names: dict = field(default_factory=dict)

    def add_point(self, p) -> int:
        self.points.append(np.asarray(p, dtype=float))
        return len(self.points) - 1

    def add_poly(self, verts, normals, face, body):
        # collapse consecutive duplicates (poles, degenerate edges)
        vv, nn = [], []
        for v, n in zip(verts, normals):
            if vv and vv[-1] == v:
                continue
            vv.append(v)
            nn.append(n)
        while len(vv) > 1 and vv[0] == vv[-1]:
            vv.pop()
            nn.pop()
        # fewer than 3 distinct corners, or a non-consecutive repeat (bow tie): drop
        if len(vv) < 3 or len(set(vv)) < len(vv):
            return False
        self.polys.append(Polygon(vv, nn, face, body))
        return True


def edge_usage(polys):
    """Directed and undirected edge counts."""
    und = Counter()
    directed = Counter()
    for p in polys:
        n = len(p.verts)
        for i in range(n):
            a, b = p.verts[i], p.verts[(i + 1) % n]
            und[(min(a, b), max(a, b))] += 1
            directed[(a, b)] += 1
    return und, directed


def stats(mesh_or_polys, npoints=None) -> dict:
    polys = mesh_or_polys.polys if isinstance(mesh_or_polys, Mesh) else mesh_or_polys
    used = set()
    tris = quads = ngons = 0
    for p in polys:
        used.update(p.verts)
        k = len(p.verts)
        if k == 3:
            tris += 1
        elif k == 4:
            quads += 1
        else:
            ngons += 1
    und, directed = edge_usage(polys)
    open_edges = sum(1 for c in und.values() if c == 1)
    nonmanifold = sum(1 for c in und.values() if c > 2)
    flipped = sum(1 for (a, b), c in directed.items() if c > 1 or (b, a) in directed and directed[(b, a)] != c)
    total = max(len(polys), 1)
    tri_equiv = tris + 2 * quads + sum(len(p.verts) - 2 for p in polys if len(p.verts) > 4)
    return {
        "verts": len(used),
        "polys": len(polys),
        "tris": tris,
        "quads": quads,
        "ngons": ngons,
        "quad_pct": round(100.0 * quads / total, 1),
        "tri_equiv": tri_equiv,
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold,
        "orientation_errors": flipped,
        "watertight": open_edges == 0 and nonmanifold == 0,
    }
