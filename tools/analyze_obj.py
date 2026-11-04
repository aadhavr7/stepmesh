"""Black-box analysis of duplicate (coincident) vertices in an OBJ.

Meant for comparing any exporter's output, for example a MoI export the user
made themselves, against stepmesh. It answers:

  * How many vertices sit exactly on top of another one, and how many are
    only nearly coincident (float noise)?
    Exact copies mean both sides came from one shared edge discretization.
    Near copies mean each side was discretized on its own.
  * Where are they? Between different groups (B-rep face boundaries), and if
    so across smooth (tangent) or sharp joins? Or inside one group (seams,
    poles)?
  * Does merging them by position close the mesh (watertight), or are there
    T-junctions left, meaning the two sides never matched?

Usage: python tools/analyze_obj.py mesh.obj [--tol 1e-6] [--sharp 20]
"""
import argparse
import math
from collections import Counter, defaultdict

import numpy as np


def load(path):
    V, F, G = [], [], []
    group = "default"
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith(("g ", "o ", "usemtl ")):
            parts = line.split(maxsplit=1)
            if parts[0] == "g" or (parts[0] == "o" and group == "default"):
                group = parts[1].strip() if len(parts) > 1 else group
        elif line.startswith("f "):
            idx = []
            for t in line.split()[1:]:
                i = int(t.split("/")[0])
                idx.append(i - 1 if i > 0 else len(V) + i)
            F.append(idx)
            G.append(group)
    return np.array(V, dtype=float), F, G


def poly_normal(P):
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else n


def open_edges(F):
    c = Counter()
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            if a != b:
                c[(min(a, b), max(a, b))] += 1
    return sum(1 for v in c.values() if v == 1), sum(1 for v in c.values() if v > 2)


def cluster(V, tol):
    """Map each vertex to a cluster id by quantized position (tol 0 = exact)."""
    if tol <= 0:
        keys = [tuple(v) for v in V]
    else:
        keys = [tuple(np.round(v / tol).astype(np.int64)) for v in V]
    ids = {}
    return np.array([ids.setdefault(k, len(ids)) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--tol", type=float, default=None, help="near-duplicate tolerance (default 1e-6 x bbox diagonal)")
    ap.add_argument("--sharp", type=float, default=20.0, help="dihedral angle above which a join counts as sharp, degrees")
    a = ap.parse_args()

    V, F, G = load(a.obj)
    used = sorted({i for f in F for i in f})
    diag = float(np.linalg.norm(V[used].max(axis=0) - V[used].min(axis=0))) if used else 1.0
    tol = a.tol if a.tol is not None else 1e-6 * diag

    exact = cluster(V, 0.0)
    near = cluster(V, tol)
    n_used = len(used)
    n_exact = len({exact[i] for i in used})
    n_near = len({near[i] for i in used})

    # per vertex: groups and polygon normals that use it
    vgroups = defaultdict(set)
    vnormals = defaultdict(list)
    for f, g in zip(F, G):
        n = poly_normal(V[f])
        for i in f:
            vgroups[i].add(g)
            vnormals[i].append(n)

    members = defaultdict(list)
    for i in used:
        members[near[i]].append(i)
    cos_sharp = math.cos(math.radians(a.sharp))
    kinds = Counter()
    sizes = Counter()
    for cid, idx in members.items():
        if len(idx) < 2:
            continue
        sizes[len(idx)] += 1
        groups = [frozenset(vgroups[i]) for i in idx]
        same_group = len(set().union(*groups)) == 1
        avg = []
        for i in idx:
            s = np.sum(vnormals[i], axis=0)
            ln = np.linalg.norm(s)
            avg.append(s / ln if ln > 0 else s)
        mind = min(float(np.dot(avg[p], avg[q])) for p in range(len(avg)) for q in range(p + 1, len(avg)))
        where = "inside one group (seam or pole)" if same_group else "between groups (face boundary)"
        crease = "sharp" if mind < cos_sharp else "smooth"
        kinds[f"{where}, {crease}"] += 1

    o0, nm0 = open_edges(F)
    Fm = [[int(near[i]) for i in f] for f in F]
    o1, nm1 = open_edges(Fm)

    print(f"file: {a.obj}")
    print(f"polygons: {len(F)}  groups: {len(set(G))}")
    print(f"vertices used: {n_used}")
    print(f"  unique positions, exact:            {n_exact}  ({n_used - n_exact} exact duplicates)")
    print(f"  unique positions, within {tol:.2e}: {n_near}  ({n_exact - n_near} more that are only nearly equal)")
    print("duplicate clusters by location:")
    for k, v in sorted(kinds.items(), key=lambda t: -t[1]):
        print(f"  {v:7d}  {k}")
    print("cluster sizes: " + ", ".join(f"{s} copies x {c}" for s, c in sorted(sizes.items())))
    print(f"open edges as written:        {o0}  non-manifold: {nm0}")
    print(f"open edges after merge-by-pos: {o1}  non-manifold: {nm1}")
    if o1 == 0 and nm1 == 0:
        print("verdict: duplicates are a pure export split; merging them gives a closed mesh (no T-junctions).")
    elif o1 < o0:
        print("verdict: merging closes some seams; the rest are real gaps or T-junctions (sides discretized differently).")
    else:
        print("verdict: open boundary is real geometry or T-junctions, not just split vertices.")


if __name__ == "__main__":
    main()
