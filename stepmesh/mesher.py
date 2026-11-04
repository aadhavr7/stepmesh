"""Edge-first, isoparametric, quad-dominant B-rep mesher.

Pipeline
  1. Index topology. Build per-face context: wires as ordered edge uses with
     their pcurves, UV scale factors, curvature sizing in u and v.
  2. Classify each face: planar, quad (4 sides, structured grid), tri
     (3 sides, structured grid with one collapsed side), or trimmed.
  3. Decide edge division counts: own curvature/length, plus requirements
     from adjacent curved faces where the edge is an isoparm, plus
     equality constraints between opposite sides of structured faces
     (propagated with union-find, max wins).
  4. Discretize every edge once. Edge points get global vertex ids shared by
     all faces using the edge, so the result is welded by construction.
  5. Mesh each face in its UV domain using the shared boundary points.
  6. Normals from the exact surface at each corner's UV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import occ
from .cdt import CDTError, cdt, dist_to_segments, grow_regions, interior_point, pair_triangles, point_in_loops
from .meshdata import Mesh
from .params import MeshParams
from .sizing import angle_between, count_from_weights, distribute, grid_weights, segment_weights

CORNER_DEG = 40.0


# ---------------------------------------------------------------- data classes

@dataclass
class EdgeUse:
    edge: object          # oriented TopoDS_Edge as found in the wire
    eidx: int
    reversed: bool
    c2d: object
    t0: float
    t1: float
    uv_samples: np.ndarray = None   # samples in traversal order, raw UV
    iso: str = None                 # "u" (u const), "v" (v const) or None


@dataclass
class EdgeData:
    edge: object
    degenerate: bool
    t0: float
    t1: float
    own: int = 1
    req: int = 1
    params: np.ndarray = None
    gids: list = None
    weights: tuple = None   # (sample params, segment weights) for distribution
    samples: tuple = None   # (params, points, tangents) of the 3D curve
    length: float = 0.0


@dataclass
class FaceCtx:
    idx: int
    face: object
    fwd: object
    reversed: bool
    body: int
    surf: occ.Surface
    wires: list = field(default_factory=list)   # list of (is_outer, [EdgeUse])
    kind: str = "trimmed"
    su: float = 1.0
    sv: float = 1.0
    ubox: tuple = (0.0, 1.0)
    vbox: tuple = (0.0, 1.0)
    nu: int = 1
    nv: int = 1
    wu: tuple = None
    wv: tuple = None
    sides: list = None      # for quad/tri: list of lists of EdgeUse
    virtual_corner: int = None
    note: str = ""


# ---------------------------------------------------------------- mesher

class Cancelled(Exception):
    pass


class Mesher:
    def __init__(self, shape, params: MeshParams, progress=None):
        """progress: optional callable(fraction 0..1, message) -> bool.
        Returning False cancels the run (raises Cancelled)."""
        self.p = params
        self._progress = progress
        self.topo = occ.Topology(shape)
        self.mesh = Mesh()
        self.edges: list[EdgeData] = []
        self.faces: list[FaceCtx] = []
        self.vgid = {}
        self.log = []
        self._ncache = {}

    # ------------------------------------------------------------ main
    def _tick(self, frac, msg):
        if self._progress is not None and self._progress(frac, msg) is False:
            raise Cancelled()

    def run(self) -> Mesh:
        from .parallel import FacePool, worker_count
        workers = worker_count(self.p, len(self.topo.faces))
        if workers > 1:
            with FacePool(self, workers) as pool:
                sizing = pool.start_sizing()          # workers size faces ...
                self._tick(0.0, "analysing edges")
                self._init_vertices()
                self._init_edges()                    # ... while this runs here
                self._setup_faces(pool.finish_sizing(sizing))
                self._prepare_edges(pool)
                pool.mesh_faces()
        else:
            self._tick(0.0, "analysing edges")
            self._init_vertices()
            self._init_edges()
            self._setup_faces({})
            self._prepare_edges()
            self._mesh_faces_serial()
        self.mesh.body_names = {i: n for i, n in enumerate(self.topo.body_names)}
        from .repair import repair
        self._tick(0.96, "checking watertightness")
        r = repair(self)
        if r["inserted"] or r["filled"]:
            self.log.append(f"repair: {r['inserted']} shared edge points inserted into {r['polys_fixed']} polygons, {r['filled']} gaps filled")
        return self.mesh

    def _setup_faces(self, pre):
        nf = max(len(self.topo.faces), 1)
        for fi, f in enumerate(self.topo.faces):
            if fi % 25 == 0:
                self._tick(0.25 + 0.05 * fi / nf, f"setting up faces {fi}/{nf}")
            try:
                self.faces.append(self._init_face(fi, f, pre=pre.get(fi)))
            except Exception as exc:
                self.log.append(f"face {fi}: init failed: {exc}")
                self.faces.append(None)

    def _prepare_edges(self, pool=None):
        self._edge_own_counts()
        self._edge_counts()
        self._edge_params()
        if self.p.band == "cells":
            from .cells import insert_crossings
            self._tick(0.3, "inserting trim crossings")
            insert_crossings(self, pool)
        self._tick(0.32, "discretizing edges")
        self._edge_points()

    def _mesh_faces_serial(self):
        nf = max(len(self.topo.faces), 1)
        for k, fc in enumerate(self.faces):
            if k % 10 == 0:
                self._tick(0.35 + 0.6 * k / nf, f"meshing faces {k}/{nf}")
            if fc is None:
                continue
            self.mesh.face_names[fc.idx] = f"face_{fc.idx}_{fc.kind}"
            try:
                self._mesh_face(fc)
            except Exception as exc:
                self.log.append(f"face {fc.idx} ({fc.kind}): meshing failed: {exc}")

    # ------------------------------------------------------------ setup
    def _init_vertices(self):
        for vi, v in enumerate(self.topo.verts):
            self.vgid[vi] = self.mesh.add_point(occ.vertex_point(v))

    def _init_edges(self):
        for e in self.topo.edges:
            t0, t1 = occ.edge_range(e)
            deg = occ.is_degenerated(e)
            ed = EdgeData(edge=e, degenerate=deg, t0=t0, t1=t1)
            if not deg:
                c = occ.Curve3(e)
                n = 2 if c.is_line else 64
                ts = np.linspace(t0, t1, n + 1) if n > 2 else np.array([t0, t1])
                if c.is_line:
                    ts = np.linspace(t0, t1, 33)
                pts, tans = zip(*(c.d1(t) for t in ts))
                pts, tans = np.array(pts), np.array(tans)
                ed.samples = (ts, pts, tans)
                ed.length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            ed.own = 1
            self.edges.append(ed)

    def _scope_hits(self, planar: bool) -> bool:
        s = self.p.max_edge_scope
        return self.p.max_edge > 0 and (s == "all" or (s == "planar") == planar)

    def _edge_own_counts(self):
        """Own division count of each edge: curve turning, plus the max edge
        length when an adjacent face is inside the max_edge scope."""
        p = self.p
        in_scope = [False] * len(self.edges)
        for fc in self.faces:
            if fc is None or not self._scope_hits(fc.surf.is_plane):
                continue
            for _, uses in fc.wires:
                for u in uses:
                    in_scope[u.eidx] = True
        for ei, ed in enumerate(self.edges):
            if ed.degenerate:
                continue
            ts, pts, tans = ed.samples
            w, L = segment_weights(pts, tans, p.angle, p.max_edge if in_scope[ei] else 0.0)
            ed.own = count_from_weights(w.sum(), L, p.min_edge)
            ed.weights = (ts, w)

    def _init_face(self, fi, f, pre=None) -> FaceCtx:
        """pre: sizing numbers already computed by a worker process (see parallel.py)."""
        fwd = occ.to_face(f.Oriented(occ.TopAbs_FORWARD))
        surf = occ.Surface(fwd)
        fc = FaceCtx(idx=fi, face=f, fwd=fwd, reversed=occ.is_reversed(f),
                     body=self.topo.face_body[fi], surf=surf)
        allpts = []
        for is_outer, edges in occ.wires_of(fwd):
            uses = []
            for e in edges:
                ei = self.topo.edge_index(e)
                ed = self.edges[ei]
                c2 = occ.Curve2(e, fwd)
                rev = e.Orientation() == occ.TopAbs_REVERSED
                ts = np.linspace(ed.t0, ed.t1, 9)
                if rev:
                    ts = ts[::-1]
                uvs = np.array([c2.uv(t) for t in ts])
                uses.append(EdgeUse(edge=e, eidx=ei, reversed=rev, c2d=c2, t0=ed.t0, t1=ed.t1, uv_samples=uvs))
                allpts.append(uvs)
            fc.wires.append((is_outer, uses))
        allpts = np.vstack(allpts)
        fc.ubox = (allpts[:, 0].min(), allpts[:, 0].max())
        fc.vbox = (allpts[:, 1].min(), allpts[:, 1].max())
        fc.pre = pre
        if pre is not None:
            fc.su, fc.sv = pre["su"], pre["sv"]
        else:
            self._face_scale(fc)
        for _, uses in fc.wires:
            for u in uses:
                u.iso = self._iso_kind(fc, u.uv_samples)
        if surf.is_plane and not self._planar_grid():
            fc.kind = "planar"
            return fc
        if pre is not None and "nu" in pre:
            fc.nu, fc.nv, fc.wu, fc.wv, fc.Lu, fc.Lv = pre["nu"], pre["nv"], pre["wu"], pre["wv"], pre["Lu"], pre["Lv"]
        else:
            self._face_sizing(fc)
        self._classify(fc)
        if fc.kind == "trimmed" and self.p.trim_aspect > 0:
            self._apply_aspect(fc, self.p.trim_aspect)
        if self.p.aspect > 0:
            self._apply_aspect(fc, self.p.aspect)
        self._face_requirements(fc)
        return fc

    def _planar_grid(self) -> bool:
        """Planar faces get a grid only when an aspect limit or a planar
        max edge length asks for it; otherwise ngon / minimal triangulation."""
        return self.p.aspect > 0 or self._scope_hits(True)

    def _face_scale(self, fc):
        """Average |Su|, |Sv| over the UV box so UV can be made roughly metric."""
        us = np.linspace(*fc.ubox, 7)
        vs = np.linspace(*fc.vbox, 7)
        a, b = [], []
        for u in us:
            for v in vs:
                try:
                    _, du, dv = fc.surf.d1(u, v)
                except Exception:
                    continue
                a.append(np.linalg.norm(du))
                b.append(np.linalg.norm(dv))
        fc.su = max(float(np.mean(a)) if a else 1.0, 1e-9)
        fc.sv = max(float(np.mean(b)) if b else 1.0, 1e-9)

    def _iso_kind(self, fc, uvs):
        du = np.ptp(uvs[:, 0]) * fc.su
        dv = np.ptp(uvs[:, 1]) * fc.sv
        tol = 1e-7 * max(du, dv, 1e-9) + 1e-9
        if du <= tol and dv > tol:
            return "u"
        if dv <= tol and du > tol:
            return "v"
        return None

    def _face_sizing(self, fc):
        """Division counts and weight distributions along u and v, from curvature."""
        p = self.p
        K = 25
        us = np.linspace(*fc.ubox, K)
        vs = np.linspace(*fc.vbox, K)
        P = np.zeros((K, K, 3))
        SU = np.zeros((K, K, 3))
        SV = np.zeros((K, K, 3))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                P[i, j], SU[i, j], SV[i, j] = fc.surf.d1(u, v)
        # zero out derivatives and normals at singular points (poles, apex) so
        # their arbitrary directions do not count as turning
        for D in (SU, SV):
            dl = np.linalg.norm(D, axis=2, keepdims=True)
            D *= dl > 1e-6 * max(float(np.median(dl)), 1e-300)
        N = np.cross(SU, SV)
        nn = np.linalg.norm(N, axis=2, keepdims=True)
        good = nn > 1e-6 * max(float(np.median(nn)), 1e-300)
        N = np.where(good, N / np.maximum(nn, 1e-300), 0.0)
        # Angle criterion = max angle between surface normals at polygon
        # corners, applied separately along u and along v.
        max_edge = p.max_edge if self._scope_hits(fc.surf.is_plane) else 0.0
        # along u (rows at fixed v) and along v, all rows at once
        # (sums kept in the original per-row order so results stay bit-identical)
        w, Lu = grid_weights(P, N, p.angle, max_edge)
        wr = np.ascontiguousarray(w.T)
        wsum_u, Lu = [r.sum() for r in wr], list(Lu)
        Wu = np.zeros(K - 1)
        for r in wr:
            Wu += r
        w, Lv = grid_weights(P.transpose(1, 0, 2), N.transpose(1, 0, 2), p.angle, max_edge)
        wr = np.ascontiguousarray(w.T)
        wsum_v, Lv = [r.sum() for r in wr], list(Lv)
        Wv = np.zeros(K - 1)
        for r in wr:
            Wv += r
        fc.nu = count_from_weights(max(wsum_u), max(Lu), p.min_edge)
        fc.nv = count_from_weights(max(wsum_v), max(Lv), p.min_edge)
        fc.wu = (us, Wu)
        fc.wv = (vs, Wv)
        fc.Lu = float(np.mean(Lu))
        fc.Lv = float(np.mean(Lv))

    def _apply_aspect(self, fc, ratio):
        hu = fc.Lu / fc.nu
        hv = fc.Lv / fc.nv
        if hu <= 1e-12 or hv <= 1e-12:
            return
        if hu > ratio * hv:
            fc.nu = max(fc.nu, int(math.ceil(fc.Lu / (ratio * hv))))
        elif hv > ratio * hu:
            fc.nv = max(fc.nv, int(math.ceil(fc.Lv / (ratio * hu))))

    def _weight_frac(self, w, a, b):
        """Fraction of total weight between parameters a and b."""
        ts, ws = w
        cum = np.concatenate([[0.0], np.cumsum(ws + 1e-12)])
        cum /= cum[-1]
        fa, fb = np.interp([a, b], ts, cum)
        return abs(fb - fa)

    def _face_requirements(self, fc):
        """An edge that is an isoparm of this face must carry at least the
        face's division count over its span, so the grid can align with it."""
        for _, uses in fc.wires:
            for u in uses:
                ed = self.edges[u.eidx]
                if u.iso == "v":
                    frac = self._weight_frac(fc.wu, u.uv_samples[0, 0], u.uv_samples[-1, 0])
                    req = int(math.ceil(fc.nu * frac - 1e-6))
                elif u.iso == "u":
                    frac = self._weight_frac(fc.wv, u.uv_samples[0, 1], u.uv_samples[-1, 1])
                    req = int(math.ceil(fc.nv * frac - 1e-6))
                else:
                    continue
                ed.req = max(ed.req, req)

    # ------------------------------------------------------------ classification
    def _scaled(self, fc, uv):
        uv = np.asarray(uv, dtype=float)
        return np.stack([uv[..., 0] * fc.su, uv[..., 1] * fc.sv], axis=-1)

    def _classify(self, fc):
        if len(fc.wires) != 1:
            fc.kind = "trimmed"
            return
        uses = fc.wires[0][1]
        n = len(uses)
        corners = []
        angles = []
        for k in range(n):
            a = self._scaled(fc, uses[k - 1].uv_samples)
            b = self._scaled(fc, uses[k].uv_samples)
            ta = a[-1] - a[-2]
            tb = b[1] - b[0]
            ang = angle_between(ta[None], tb[None])[0]
            if np.linalg.norm(ta) < 1e-14 or np.linalg.norm(tb) < 1e-14:
                ang = math.pi / 2
            if ang > math.radians(CORNER_DEG):
                corners.append(k)
                angles.append(ang)
        if len(corners) not in (3, 4):
            fc.kind = "trimmed"
            return
        sides = []
        for ci, c in enumerate(corners):
            nxt = corners[(ci + 1) % len(corners)]
            side = []
            k = c
            while True:
                side.append(uses[k])
                k = (k + 1) % n
                if k == nxt:
                    break
            sides.append(side)
        fc.sides = sides
        if len(corners) == 4:
            fc.kind = "quad"
        else:
            # collapse the sharpest corner (largest turn = smallest interior angle)
            # whose two adjacent sides are single edges
            order = np.argsort(-np.array(angles))
            pick = None
            for ci in order:
                s_after = sides[ci]
                s_before = sides[ci - 1]
                if len(s_after) == 1 and len(s_before) == 1:
                    pick = int(ci)
                    break
            if pick is None:
                fc.kind = "trimmed"
                return
            # rotate so the virtual corner is corner 0: sides S0 (after), S1, S2 (before)
            fc.sides = sides[pick:] + sides[:pick]
            fc.kind = "tri"

    # ------------------------------------------------------------ edge counts
    def _edge_counts(self):
        parent = list(range(len(self.edges)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for fc in self.faces:
            if fc is None or fc.kind not in ("quad", "tri"):
                continue
            S = fc.sides
            if fc.kind == "quad":
                pairs = [(0, 2), (1, 3)]
            else:
                pairs = [(0, 2)]
            for a, b in pairs:
                if len(S[a]) == 1 and len(S[b]) == 1:
                    union(S[a][0].eidx, S[b][0].eidx)
        best = {}
        for i, ed in enumerate(self.edges):
            n = max(ed.own, ed.req)
            r = find(i)
            best[r] = max(best.get(r, 1), n)
        for i, ed in enumerate(self.edges):
            ed.n = best[find(i)]

    def _edge_params(self):
        """Parameter values of every edge's points (no vertices yet, so faces
        can still add required points such as grid crossings)."""
        for ed in self.edges:
            if ed.degenerate:
                ed.params = np.linspace(ed.t0, ed.t1, ed.n + 1)
            else:
                ts, w = ed.weights
                ed.params = distribute(ts, w, ed.n)

    def _edge_points(self):
        """Create the shared mesh vertices of every edge, once."""
        for ed in self.edges:
            v0, v1 = occ.edge_vertices(ed.edge)
            g0 = self.vgid[self.topo.vert_index(v0)]
            g1 = self.vgid[self.topo.vert_index(v1)]
            if ed.degenerate:
                ed.gids = [g0] * len(ed.params)
                continue
            c = occ.Curve3(ed.edge)
            gids = [g0]
            for t in ed.params[1:-1]:
                gids.append(self.mesh.add_point(c.point(t)))
            gids.append(g1)
            ed.gids = gids

    # ------------------------------------------------------------ face meshing helpers
    def _use_points(self, fc, use: EdgeUse):
        """(uv (n,2), gids) of an edge use in traversal order, endpoints included."""
        ed = self.edges[use.eidx]
        ts = ed.params
        g = ed.gids
        if use.reversed:
            ts = ts[::-1]
            g = g[::-1] if g is not None else None
        uv = np.array([use.c2d.uv(t) for t in ts])
        return uv, (list(g) if g is not None else None)

    def _wire_loop(self, fc, uses):
        uvs, gids = [], []
        for u in uses:
            uv, g = self._use_points(fc, u)
            uvs.append(uv[:-1])
            gids.extend(g[:-1])
        return np.vstack(uvs), gids

    def _normal(self, fc, uv):
        # cached per exact UV: a vertex is shared by about four polygons
        key = (fc.idx, float(uv[0]), float(uv[1]))
        n = self._ncache.get(key)
        if n is None:
            c = ((fc.ubox[0] + fc.ubox[1]) / 2, (fc.vbox[0] + fc.vbox[1]) / 2)
            n = fc.surf.normal(uv[0], uv[1], toward=c)
            if fc.reversed:
                n = -n
            self._ncache[key] = n
        return n

    def _new_point(self, fc, uv):
        return self.mesh.add_point(fc.surf.point(uv[0], uv[1]))

    def _emit(self, fc, gids, uvs, sign=None):
        """Add a polygon given CCW-in-UV corners; flips for reversed faces."""
        if sign is None:
            su, sv = fc.su, fc.sv
            n = len(uvs)
            area = 0.0
            for k in range(n):
                a, b = uvs[k], uvs[(k + 1) % n]
                area += (a[0] * su) * (b[1] * sv) - (a[1] * sv) * (b[0] * su)
            sign = 1.0 if area >= 0 else -1.0
        order = list(range(len(gids)))
        if sign < 0:
            order.reverse()
        if fc.reversed:
            order.reverse()
        if fc.kind == "planar":
            nrm = self._normal(fc, uvs[0])
            normals = [nrm] * len(gids)
        else:
            normals = [self._normal(fc, uvs[k]) for k in order]
        return self.mesh.add_poly([gids[k] for k in order], normals, fc.idx, fc.body)

    # ------------------------------------------------------------ face meshing
    def _mesh_face(self, fc):
        if fc.kind == "planar":
            self._mesh_planar(fc)
            return
        if fc.kind in ("quad", "tri"):
            if self._mesh_structured(fc):
                return
            fc.note = f"structured failed ({fc.note}), trimmed fallback"
            fc.kind = "trimmed"
            self.mesh.face_names[fc.idx] = f"face_{fc.idx}_trimmed"
        self._mesh_trimmed(fc)

    def _loops(self, fc):
        loops = []
        for is_outer, uses in fc.wires:
            uv, g = self._wire_loop(fc, uses)
            loops.append((is_outer, uv, g))
        return loops

    def _mesh_planar(self, fc):
        loops = self._loops(fc)
        out = self.p.output
        if len(loops) == 1 and out == "ngons":
            _, uv, g = loops[0]
            self._emit(fc, g, uv)
            return
        if out == "ngons":
            self._cdt_loops(fc, loops, grow=True)
            return
        self._cdt_loops(fc, loops, pair=out == "quads" and self.p.planar_quads)

    def _cdt_loops(self, fc, loops, extra_uv=None, pair=False, grow=False):
        pts, gids, segs, holes = [], [], [], []
        for is_outer, uv, g in loops:
            base = len(pts)
            n = len(uv)
            pts.extend(list(uv))
            gids.extend(g)
            segs.extend([(base + k, base + (k + 1) % n) for k in range(n)])
            if not is_outer:
                holes.append(interior_point(self._scaled(fc, uv)))
        sp = self._scaled(fc, np.array(pts))
        tris, verts = cdt(sp, segs, holes)
        if len(verts) > len(pts):
            self.log.append(f"face {fc.idx}: CDT inserted {len(verts) - len(pts)} points (crossing trims)")
            for k in range(len(pts), len(verts)):
                uv = np.array([verts[k][0] / fc.su, verts[k][1] / fc.sv])
                pts.append(uv)
                gids.append(self._new_point(fc, uv))
        pts = np.array(pts)
        if grow:
            polys = grow_regions(tris, self._scaled(fc, pts))
            for poly in polys:
                self._emit(fc, [gids[k] for k in poly], pts[poly])
            return
        if pair:
            polys = pair_triangles([tuple(t) for t in tris], self._scaled(fc, pts))
        else:
            polys = [list(t) for t in tris]
        for poly in polys:
            self._emit(fc, [gids[k] for k in poly], pts[poly], sign=1.0)

    # ---- structured (transfinite interpolation in UV)
    def _side_points(self, fc, side):
        uvs, gids = [], []
        for k, u in enumerate(side):
            uv, g = self._use_points(fc, u)
            if k > 0:
                uv, g = uv[1:], g[1:]
            uvs.append(uv)
            gids.extend(g)
        return np.vstack(uvs), gids

    def _mesh_structured(self, fc) -> bool:
        S = [self._side_points(fc, s) for s in fc.sides]
        if fc.kind == "tri":
            # virtual collapsed side at corner 0 (between S2 and S0)
            c0uv, c0g = S[0][0][0], S[0][1][0]
            m = len(S[1][1])
            S.append((np.repeat(c0uv[None], m, axis=0), [c0g] * m))
        (b_uv, b_g), (r_uv, r_g), (t_uv, t_g), (l_uv, l_g) = S
        t_uv, t_g = t_uv[::-1], t_g[::-1]
        l_uv, l_g = l_uv[::-1], l_g[::-1]
        n, m = len(b_g) - 1, len(r_g) - 1
        if len(t_g) - 1 != n or len(l_g) - 1 != m:
            fc.note = f"side counts {n}/{len(t_g)-1} x {m}/{len(l_g)-1}"
            return False

        def chord(uv):
            s = self._scaled(fc, uv)
            d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))])
            return d / d[-1] if d[-1] > 1e-14 else np.linspace(0, 1, len(uv))

        sb, st = chord(b_uv), chord(t_uv)
        sl, sr = chord(l_uv), chord(r_uv)
        c00, c10, c11, c01 = b_uv[0], b_uv[-1], t_uv[-1], t_uv[0]
        G = np.zeros((n + 1, m + 1, 2))
        gid = [[None] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                tj = (1 - sb[i]) * sl[j] + sb[i] * sr[j]
                si = (1 - tj) * sb[i] + tj * st[i]
                G[i, j] = ((1 - tj) * b_uv[i] + tj * t_uv[i] + (1 - si) * l_uv[j] + si * r_uv[j]
                           - ((1 - si) * (1 - tj) * c00 + si * (1 - tj) * c10
                              + si * tj * c11 + (1 - si) * tj * c01))
        for i in range(n + 1):
            gid[i][0], gid[i][m] = b_g[i], t_g[i]
            G[i, 0], G[i, m] = b_uv[i], t_uv[i]
        for j in range(m + 1):
            gid[0][j], gid[n][j] = l_g[j], r_g[j]
            G[0, j], G[n, j] = l_uv[j], r_uv[j]
        # validate: consistent cell orientation in scaled UV
        Gs = self._scaled(fc, G)
        a = Gs[:-1, :-1]
        b = Gs[1:, :-1]
        c = Gs[1:, 1:]
        d = Gs[:-1, 1:]

        def tri_area(p, q, r):
            return 0.5 * ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

        area = tri_area(a, b, c) + tri_area(a, c, d)
        a1 = tri_area(a, b, c)
        a2 = tri_area(a, c, d)
        scale = np.abs(area).max() + 1e-30
        sgn = 1.0 if area.sum() >= 0 else -1.0
        tol = -1e-9 * scale
        if (sgn * a1 < tol).any() or (sgn * a2 < tol).any():
            fc.note = "inverted cells"
            return False
        for i in range(1, n):
            for j in range(1, m):
                gid[i][j] = self._new_point(fc, G[i, j])
        for i in range(n):
            for j in range(m):
                corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                self._emit(fc, [gid[x][y] for x, y in corners], [G[x, y] for x, y in corners], sign=sgn)
        return True

    # ---- trimmed: structured grid inside, CDT band to the trims
    def _grid_lines(self, fc, loops, axis):
        """Grid lines along one UV axis. axis 0 -> u values, 1 -> v values."""
        lo, hi = (fc.ubox if axis == 0 else fc.vbox)
        n = fc.nu if axis == 0 else fc.nv
        w = fc.wu if axis == 0 else fc.wv
        base = distribute(w[0], w[1], n)
        # align to the widest isoparametric boundary edge running along this axis
        want_iso = "v" if axis == 0 else "u"
        best, best_span = None, 0.0
        for _, uses in fc.wires:
            for u in uses:
                if u.iso != want_iso:
                    continue
                vals, _ = self._use_points(fc, u)
                vals = vals[:, axis]
                span = np.ptp(vals)
                if span > best_span:
                    best, best_span = vals, span
        rng = hi - lo
        if best is None or best_span < 0.5 * rng:
            return base
        lines = list(np.sort(best))
        h = rng / max(n, 1)
        bmin, bmax = lines[0], lines[-1]
        lines += [x for x in base if x < bmin - 0.3 * h or x > bmax + 0.3 * h]
        # the grid must always cover the whole UV box of the trims, even when
        # the aligned edge stops short (thin rows are merged away later)
        eps = 1e-9 * max(rng, 1e-12)
        if all(abs(x - lo) > eps for x in lines):
            lines.append(lo)
        if all(abs(x - hi) > eps for x in lines):
            lines.append(hi)
        lines = np.unique(np.round(np.array(lines), 12))
        keep = [lines[0]]
        for x in lines[1:]:
            if x - keep[-1] > 1e-9 * max(rng, 1e-12):
                keep.append(x)
        return np.array(keep)

    def _mesh_trimmed(self, fc):
        if self.p.band == "cells":
            from .cells import mesh_trimmed_cells
            try:
                if mesh_trimmed_cells(self, fc):
                    return
            except Exception as exc:
                self.log.append(f"face {fc.idx}: cell clipping failed ({exc}), triangle band fallback")
        self._mesh_trimmed_band(fc)

    def _mesh_trimmed_band(self, fc):
        p = self.p
        loops = self._loops(fc)
        U = self._grid_lines(fc, loops, 0)
        V = self._grid_lines(fc, loops, 1)
        # boundary arrays (scaled UV)
        LP_uv = np.vstack([uv for _, uv, _ in loops])
        LP_g = [g for _, _, gs in loops for g in gs]
        LP_s = self._scaled(fc, LP_uv)
        loops_s = [self._scaled(fc, uv) for _, uv, _ in loops]
        seg_idx = []
        base = 0
        for _, uv, _ in loops:
            k = len(uv)
            for a in range(k):
                seg_idx.append((base + a, base + (a + 1) % k))
            base += k
        seg_idx = np.array(seg_idx)
        A, B = LP_s[seg_idx[:, 0]], LP_s[seg_idx[:, 1]]
        diag = float(np.hypot(*(LP_s.max(axis=0) - LP_s.min(axis=0)))) + 1e-30
        tol = 1e-7 * diag
        Us, Vs = U * fc.su, V * fc.sv
        nu, nv = len(U), len(V)
        # nodes
        NI, NJ = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        Q = np.stack([Us[NI], Vs[NJ]], axis=-1).reshape(-1, 2)
        d2 = ((Q[:, None, :] - LP_s[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        snapped = np.where(np.sqrt(d2[np.arange(len(Q)), nearest]) <= tol, nearest, -1).reshape(nu, nv)
        inside = point_in_loops(Q, loops_s).reshape(nu, nv)
        dist = dist_to_segments(Q, A, B).reshape(nu, nv)
        ok_node = inside | (snapped >= 0)

        # cells
        ncu, ncv = nu - 1, nv - 1
        bad = np.zeros((ncu, ncv), dtype=bool)

        def cell_range(vals, lo_, hi_):
            i0 = max(int(np.searchsorted(vals, lo_ - tol, side="right")) - 1, 0)
            i1 = min(int(np.searchsorted(vals, hi_ + tol, side="left")), len(vals) - 1)
            return range(i0, i1)

        for a, b in zip(A, B):
            for i in cell_range(Us, min(a[0], b[0]), max(a[0], b[0])):
                for j in cell_range(Vs, min(a[1], b[1]), max(a[1], b[1])):
                    if bad[i, j]:
                        continue
                    if _seg_hits_open_rect(a, b, Us[i] + tol, Us[i + 1] - tol, Vs[j] + tol, Vs[j + 1] - tol):
                        bad[i, j] = True
        for k, q in enumerate(LP_s):
            for i in cell_range(Us, q[0], q[0]):
                for j in cell_range(Vs, q[1], q[1]):
                    if bad[i, j]:
                        continue
                    if not (Us[i] - tol <= q[0] <= Us[i + 1] + tol and Vs[j] - tol <= q[1] <= Vs[j + 1] + tol):
                        continue
                    corner_ids = (snapped[i, j], snapped[i + 1, j], snapped[i + 1, j + 1], snapped[i, j + 1])
                    if k not in corner_ids:
                        bad[i, j] = True
        keep = np.zeros((ncu, ncv), dtype=bool)
        cent = np.stack([(Us[:-1, None] + Us[1:, None]) / 2 + 0 * Vs[None, :-1],
                         0 * Us[:-1, None] + (Vs[None, :-1] + Vs[None, 1:]) / 2], axis=-1)
        cent_in = point_in_loops(cent.reshape(-1, 2), loops_s).reshape(ncu, ncv)
        for i in range(ncu):
            for j in range(ncv):
                if bad[i, j] or not cent_in[i, j]:
                    continue
                h = min(Us[i + 1] - Us[i], Vs[j + 1] - Vs[j])
                good = True
                for x, y in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    if not ok_node[x, y]:
                        good = False
                        break
                    if snapped[x, y] < 0 and dist[x, y] < p.band_margin * h:
                        good = False
                        break
                if good:
                    keep[i, j] = True

        # node gids
        node_gid = {}

        def ngid(x, y):
            key = (x, y)
            if key not in node_gid:
                s = snapped[x, y]
                node_gid[key] = LP_g[s] if s >= 0 else self._new_point(fc, (U[x], V[y]))
            return node_gid[key]

        # emit kept quads
        for i in range(ncu):
            for j in range(ncv):
                if keep[i, j]:
                    cs = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                    self._emit(fc, [ngid(x, y) for x, y in cs], [(U[x], V[y]) for x, y in cs], sign=1.0)

        # band: CDT between trims and the kept region
        in_region = np.zeros((nu, nv), dtype=bool)
        for i in range(ncu):
            for j in range(ncv):
                if keep[i, j]:
                    in_region[i, j] = in_region[i + 1, j] = in_region[i, j + 1] = in_region[i + 1, j + 1] = True
        pts = list(LP_s)
        pgid = list(LP_g)
        puv = list(LP_uv)
        segs = set()
        for a, b in seg_idx:
            segs.add((min(a, b), max(a, b)))
        node_idx = {}

        def nidx(x, y):
            s = snapped[x, y]
            if s >= 0:
                return int(s)
            if (x, y) not in node_idx:
                node_idx[(x, y)] = len(pts)
                pts.append(np.array([Us[x], Vs[y]]))
                puv.append(np.array([U[x], V[y]]))
                pgid.append(ngid(x, y))
            return node_idx[(x, y)]

        def kept(i, j):
            return 0 <= i < ncu and 0 <= j < ncv and keep[i, j]

        holes = []
        # region boundary edges
        for i in range(ncu):
            for j in range(nv):
                if kept(i, j - 1) != kept(i, j):
                    a, b = nidx(i, j), nidx(i + 1, j)
                    segs.add((min(a, b), max(a, b)))
        for i in range(nu):
            for j in range(ncv):
                if kept(i - 1, j) != kept(i, j):
                    a, b = nidx(i, j), nidx(i, j + 1)
                    segs.add((min(a, b), max(a, b)))
        for i in range(ncu):
            for j in range(ncv):
                if keep[i, j]:
                    holes.append(cent[i, j])
        # free interior nodes in the band improve shape and chordal accuracy
        for i in range(nu):
            for j in range(nv):
                if in_region[i, j] or snapped[i, j] >= 0 or not inside[i, j]:
                    continue
                hu = min([Us[k + 1] - Us[k] for k in (i - 1, i) if 0 <= k < nu - 1] or [diag])
                hv = min([Vs[k + 1] - Vs[k] for k in (j - 1, j) if 0 <= k < nv - 1] or [diag])
                if dist[i, j] >= p.band_margin * min(hu, hv):
                    nidx(i, j)
        for is_outer, uv, _ in loops:
            if not is_outer:
                holes.append(interior_point(self._scaled(fc, uv)))
        segs = [s for s in segs if s[0] != s[1]]
        try:
            tris, verts = cdt(np.array(pts), segs, holes)
        except CDTError as exc:
            self.log.append(f"face {fc.idx}: band CDT failed: {exc}")
            return
        if len(verts) > len(pts):
            self.log.append(f"face {fc.idx}: band CDT inserted {len(verts) - len(pts)} points")
            for k in range(len(pts), len(verts)):
                uv = np.array([verts[k][0] / fc.su, verts[k][1] / fc.sv])
                puv.append(uv)
                pgid.append(self._new_point(fc, uv))
        puv = np.array(puv)
        for poly in tris:
            self._emit(fc, [pgid[k] for k in poly], puv[poly], sign=1.0)


def _seg_hits_open_rect(a, b, x0, x1, y0, y1):
    """Liang-Barsky: does segment a-b pass through the open rectangle?"""
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(pk) < 1e-300:
            if qk <= 0:
                return False
        else:
            r = qk / pk
            if pk < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t1 - t0 > 1e-12


def mesh_shape(shape, params: MeshParams, progress=None):
    m = Mesher(shape, params, progress)
    mesh = m.run()
    return mesh, m
