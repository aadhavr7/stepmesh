"""OBJ writer with per-body objects, per-face groups and split normals."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .meshdata import Mesh


def triangulate_polys(mesh: Mesh):
    """Split quads along the shorter diagonal and ngons as fans (for 'tris' output)."""
    from .meshdata import Polygon
    out = []
    P = mesh.points
    for p in mesh.polys:
        v, n = p.verts, p.normals
        if len(v) == 3:
            out.append(p)
        elif len(v) == 4:
            d02 = np.linalg.norm(P[v[0]] - P[v[2]])
            d13 = np.linalg.norm(P[v[1]] - P[v[3]])
            if d02 <= d13:
                idx = [(0, 1, 2), (0, 2, 3)]
            else:
                idx = [(0, 1, 3), (1, 2, 3)]
            for t in idx:
                out.append(Polygon([v[k] for k in t], [n[k] for k in t], p.face, p.body))
        else:
            for k in range(1, len(v) - 1):
                t = (0, k, k + 1)
                out.append(Polygon([v[i] for i in t], [n[i] for i in t], p.face, p.body))
    mesh.polys = out
    return mesh


def cluster_normals(mesh: Mesh, sharp_deg: float):
    """Per vertex, merge corner normals within sharp_deg of each other.

    Exact surface normals already agree across tangent B-rep edges, so this
    keeps smooth joins smooth and leaves sharp edges split.
    Returns (normal list, per-poly list of normal indices).
    """
    cos_t = math.cos(math.radians(sharp_deg))
    per_vertex = {}
    refs = []
    # plain floats: this runs once per polygon corner (hundreds of thousands)
    for p in mesh.polys:
        row = []
        for v, n in zip(p.verts, p.normals):
            nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
            clusters = per_vertex.get(v)
            if clusters is None:
                clusters = per_vertex[v] = []
            hit = -1
            for ci, c in enumerate(clusters):
                r = c[3]
                if r[0] * nx + r[1] * ny + r[2] * nz >= cos_t:
                    hit = ci
                    break
            if hit < 0:
                clusters.append([nx, ny, nz, (nx, ny, nz)])
                hit = len(clusters) - 1
            else:
                c = clusters[hit]
                c[0] += nx
                c[1] += ny
                c[2] += nz
            row.append((v, hit))
        refs.append(row)
    normals = []
    index = {}
    for v, clusters in per_vertex.items():
        for ci, c in enumerate(clusters):
            ln = math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
            index[(v, ci)] = len(normals)
            normals.append(np.array((c[0] / ln, c[1] / ln, c[2] / ln)) if ln > 0 else np.array(c[:3]))
    poly_n = [[index[k] for k in row] for row in refs]
    return normals, poly_n


def write_obj(mesh: Mesh, path, sharp_deg: float = 20.0, group_by: str = "face", header: str = "",
              weld="all"):
    """weld: "all"    one OBJ vertex per mesh vertex (closed mesh)
             "smooth" split vertices where normals split (sharp B-rep edges),
                      shared across tangent joins
             "none"   every B-rep face is its own island; copies along edges
                      are bit-identical, so merge-by-distance with 0 tolerance
                      restores the closed mesh
    True/False are accepted as "all"/"none"."""
    path = Path(path)
    if weld is True:
        weld = "all"
    elif weld is False:
        weld = "none"
    normals, poly_n = cluster_normals(mesh, sharp_deg)
    order = sorted(range(len(mesh.polys)), key=lambda i: (mesh.polys[i].body, mesh.polys[i].face))

    def key(i, k):
        p = mesh.polys[i]
        v = p.verts[k]
        if weld == "none":
            return (p.face, v)
        if weld == "smooth":
            return ("n", poly_n[i][k])   # normal index is unique per (vertex, normal cluster)
        return v

    order_keys, seen, src = [], set(), []
    for i in order:
        for k, v in enumerate(mesh.polys[i].verts):
            kk = key(i, k)
            if kk not in seen:
                seen.add(kk)
                order_keys.append(kk)
                src.append(v)
    remap = {kk: n + 1 for n, kk in enumerate(order_keys)}
    used = src
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# stepmesh OBJ\n")
        for line in header.splitlines():
            f.write(f"# {line}\n")
        for g in used:
            x, y, z = mesh.points[g]
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        cur_body = cur_face = None
        for i in order:
            p = mesh.polys[i]
            if p.body != cur_body:
                cur_body = p.body
                cur_face = None
                f.write(f"o {mesh.body_names.get(p.body, f'body_{p.body}')}\n")
            if group_by == "face" and p.face != cur_face:
                cur_face = p.face
                f.write(f"g {mesh.face_names.get(p.face, f'face_{p.face}')}\n")
            f.write("f " + " ".join(f"{remap[key(i, k)]}//{poly_n[i][k] + 1}" for k, v in enumerate(p.verts)) + "\n")
    return path
