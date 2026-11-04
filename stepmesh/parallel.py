"""Use several CPU cores for the per-face work.

Two phases run in one pool of worker processes:
  1. sizing: each face's curvature sampling (the expensive part of setting
     up a face), sent back as plain numbers;
  2. meshing: after the shared edges are discretized, every face is meshed
     on its own and sent back as polygons plus the new points it created.
Workers get the healed shape as a binary BREP (exact doubles) and run the
same code as a single process. The main process stitches results together
in face order and numbers new points exactly as a serial run would, so the
output is identical.
"""
from __future__ import annotations

import math
import os
import pickle
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from .meshdata import Polygon

MIN_FACES = 150        # below this, starting workers costs more than it saves

_W = {}


def worker_count(params, nfaces):
    if params.workers == 1 or nfaces < MIN_FACES:
        return 1
    cpu = os.cpu_count() or 1
    n = params.workers if params.workers > 1 else min(max(cpu - 1, 1), 16)
    return max(1, min(n, math.ceil(nfaces / 40)))


# ---------------------------------------------------------------- worker side

def _init_worker(shape_path, params):
    from OCP.BinTools import BinTools
    from OCP.TopoDS import TopoDS_Shape

    from . import occ
    from .mesher import EdgeData, Mesher

    shape = TopoDS_Shape()
    BinTools.Read_s(shape, shape_path)
    m = Mesher(shape, params)
    for e in m.topo.edges:
        t0, t1 = occ.edge_range(e)
        m.edges.append(EdgeData(edge=e, degenerate=occ.is_degenerated(e), t0=t0, t1=t1))
    _W.update(m=m, state_path=None)


def _size_chunk(face_ids):
    m = _W["m"]
    out = {}
    for fi in face_ids:
        try:
            fc = m._init_face(fi, m.topo.faces[fi])
        except Exception:
            out[fi] = None
            continue
        r = {"su": fc.su, "sv": fc.sv}
        if fc.wu is not None:
            r.update(nu=fc.nu, nv=fc.nv, wu=fc.wu, wv=fc.wv, Lu=fc.Lu, Lv=fc.Lv)
        out[fi] = r
    return out


def _load_state(state_path):
    """Edge points, face decisions and extras for one phase, loaded once per worker."""
    if _W["state_path"] == state_path:
        return
    with open(state_path, "rb") as f:
        st = pickle.load(f)
    m = _W["m"]
    for ed, (params_e, gids) in zip(m.edges, st["edges"]):
        ed.params = np.asarray(params_e, dtype=float)
        ed.gids = None if gids is None else list(gids)
    if st.get("n0") is not None:
        m.mesh.points = [None] * st["n0"]
    _W.update(face_state=st["faces"], n0=st.get("n0"), structured=st.get("structured"), state_path=state_path)


def _face_from_state(fi):
    m, st = _W["m"], _W["face_state"][fi]
    fc = m._init_face(fi, m.topo.faces[fi], pre=st["pre"])
    fc.kind, fc.nu, fc.nv, fc.note = st["kind"], st["nu"], st["nv"], st["note"]
    if st.get("grid") is not None:
        fc.grid = (np.asarray(st["grid"][0]), np.asarray(st["grid"][1]))
    return fc


def _cross_chunk(face_ids, state_path):
    from .cells import face_crossings
    _load_state(state_path)
    return {fi: face_crossings(_W["m"], _face_from_state(fi), _W["structured"]) for fi in face_ids}


def _mesh_chunk(face_ids, state_path):
    _load_state(state_path)
    m, face_state, n0 = _W["m"], _W["face_state"], _W["n0"]
    out = []
    for fi in face_ids:
        st = face_state[fi]
        p0, l0 = len(m.mesh.polys), len(m.log)
        try:
            fc = _face_from_state(fi)
            m.mesh.face_names[fi] = f"face_{fi}_{fc.kind}"
            m._mesh_face(fc)
            kind, note = fc.kind, fc.note
        except Exception as exc:
            m.log.append(f"face {fi} ({st['kind']}): meshing failed: {exc}")
            kind, note = st["kind"], st["note"]
        pts = [np.asarray(p) for p in m.mesh.points[n0:]]
        polys = [(p.verts, p.normals, p.face, p.body) for p in m.mesh.polys[p0:]]
        out.append((fi, kind, note, m.mesh.face_names.get(fi), pts, polys, m.log[l0:]))
        # forget this face so the next one numbers its new points from n0 again
        del m.mesh.points[n0:]
        del m.mesh.polys[p0:]
        del m.log[l0:]
    return out


# ---------------------------------------------------------------- main side

class FacePool:
    """Worker processes for one meshing run (use as a context manager)."""

    def __init__(self, m, workers):
        self.m, self.workers = m, workers
        self.tmp = []

    def _tempfile(self, suffix):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.close()
        self.tmp.append(f.name)
        return f.name

    def __enter__(self):
        from OCP.BinTools import BinTools
        path = self._tempfile(".bbrep")
        BinTools.Write_s(self.m.topo.shape, path)   # binary BREP: exact doubles
        self.pool = ProcessPoolExecutor(max_workers=self.workers, initializer=_init_worker,
                                        initargs=(path, self.m.p))
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True, cancel_futures=True)
        for p in self.tmp:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _run(self, fn, ids, extra, tick):
        size = max(4, math.ceil(len(ids) / (self.workers * 6)))
        chunks = [ids[i:i + size] for i in range(0, len(ids), size)]
        pending = {self.pool.submit(fn, c, *extra) for c in chunks}
        results, done_n = [], 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                r = f.result()
                results.append(r)
                done_n += len(r)
            try:
                tick(done_n, len(ids))
            except Exception:
                for f in pending:
                    f.cancel()
                raise
        return results

    def start_sizing(self):
        """Queue curvature sizing of every face; collect with finish_sizing."""
        ids = list(range(len(self.m.topo.faces)))
        size = max(4, math.ceil(len(ids) / (self.workers * 6)))
        return [self.pool.submit(_size_chunk, ids[i:i + size]) for i in range(0, len(ids), size)]

    def finish_sizing(self, futures):
        m, n = self.m, len(self.m.topo.faces)
        out, pending = {}, set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                out.update(f.result())
            try:
                m._tick(0.05 + 0.2 * len(out) / max(n, 1), f"sizing faces {len(out)}/{n}")
            except Exception:
                for f in pending:
                    f.cancel()
                raise
        return {k: v for k, v in out.items() if v is not None}

    def _face_state(self, faces, with_grid):
        out = {}
        for fc in faces:
            g = getattr(fc, "grid", None) if with_grid else None
            out[fc.idx] = {"kind": fc.kind, "nu": fc.nu, "nv": fc.nv, "note": fc.note,
                           "pre": getattr(fc, "pre", None),
                           "grid": None if g is None else (np.asarray(g[0]).tolist(), np.asarray(g[1]).tolist())}
        return out

    def _write_state(self, st):
        path = self._tempfile(".pkl")
        with open(path, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def crossings(self, trimmed, structured_edges):
        """face_crossings for every trimmed face, computed by the workers."""
        m = self.m
        path = self._write_state({"edges": [(np.asarray(ed.params).tolist(), None) for ed in m.edges],
                                  "faces": self._face_state(trimmed, with_grid=False),
                                  "structured": structured_edges})
        out = {}
        for r in self._run(_cross_chunk, [fc.idx for fc in trimmed], (path,),
                           lambda d, n: m._tick(0.3, f"inserting trim crossings {d}/{n}")):
            out.update(r)
        return out

    def mesh_faces(self):
        """Mesh every face in the workers; same result as the serial loop."""
        m = self.m
        faces = [fc for fc in m.faces if fc is not None]
        n0 = len(m.mesh.points)
        state_path = self._write_state({"edges": [(np.asarray(ed.params).tolist(), list(ed.gids)) for ed in m.edges],
                                        "faces": self._face_state(faces, with_grid=True), "n0": n0})
        ids = [fc.idx for fc in faces]
        results = {}
        for chunk in self._run(_mesh_chunk, ids, (state_path,),
                               lambda d, n: m._tick(0.35 + 0.6 * d / n, f"meshing faces {d}/{n}")):
            for r in chunk:
                results[r[0]] = r

        # stitch together in face order, numbering new points as a serial run would
        by_idx = {fc.idx: fc for fc in faces}
        for fi in ids:
            _, kind, note, name, pts, polys, log = results[fi]
            fc = by_idx[fi]
            fc.kind, fc.note = kind, note
            m.mesh.face_names[fi] = name or f"face_{fi}_{kind}"
            base = len(m.mesh.points)
            m.mesh.points.extend(pts)
            for verts, normals, face, body in polys:
                m.mesh.polys.append(Polygon([v if v < n0 else base + (v - n0) for v in verts], normals, face, body))
            m.log.extend(log)
