"""Thin helpers over OCP (OpenCASCADE) for reading STEP and walking B-rep topology."""
from __future__ import annotations

import math

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Plane
from OCP.gp import gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import (
    TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SHELL,
    TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS


def _cast(name):
    # OCP 7.x exposes to_face, OCP 8 exposes TopoDS.Face
    return getattr(TopoDS, name + "_s", None) or getattr(TopoDS, name)


to_face, to_edge, to_vertex, to_wire = (_cast(n) for n in ("Face", "Edge", "Vertex", "Wire"))
try:
    from OCP.TopTools import TopTools_IndexedMapOfShape
except ImportError:  # OCP 8 exposes NCollection templates under OCP.collections
    from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape


def read_step(path: str, heal: bool = True):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for {path} (status {status})")
    reader.TransferRoots()
    shape = reader.OneShape()
    if heal:
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()
    return shape


def index_map(shape, kind):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


class Topology:
    """Stable 0-based indices for faces, edges, vertices and bodies."""

    def __init__(self, shape):
        self.shape = shape
        self.fmap = index_map(shape, TopAbs_FACE)
        self.emap = index_map(shape, TopAbs_EDGE)
        self.vmap = index_map(shape, TopAbs_VERTEX)
        self.faces = [to_face(self.fmap.FindKey(i)) for i in range(1, self.fmap.Extent() + 1)]
        self.edges = [to_edge(self.emap.FindKey(i)) for i in range(1, self.emap.Extent() + 1)]
        self.verts = [to_vertex(self.vmap.FindKey(i)) for i in range(1, self.vmap.Extent() + 1)]
        # body per face: solids first, then free shells, then loose faces
        self.face_body = [-1] * len(self.faces)
        self.body_names = []
        for kind, label in ((TopAbs_SOLID, "solid"), (TopAbs_SHELL, "shell")):
            exp = TopExp_Explorer(shape, kind)
            while exp.More():
                sub = exp.Current()
                body = len(self.body_names)
                hit = False
                fexp = TopExp_Explorer(sub, TopAbs_FACE)
                while fexp.More():
                    fi = self.face_index(fexp.Current())
                    if self.face_body[fi] < 0:
                        self.face_body[fi] = body
                        hit = True
                    fexp.Next()
                if hit:
                    self.body_names.append(f"{label}_{body}")
                exp.Next()
        for fi, b in enumerate(self.face_body):
            if b < 0:
                self.face_body[fi] = len(self.body_names)
                self.body_names.append(f"face_body_{fi}")

    def face_index(self, f):
        return self.fmap.FindIndex(f) - 1

    def edge_index(self, e):
        return self.emap.FindIndex(e) - 1

    def vert_index(self, v):
        return self.vmap.FindIndex(v) - 1


def wires_of(face_fwd):
    """List of (is_outer, [oriented edges in order]) for a FORWARD face."""
    outer = BRepTools.OuterWire_s(face_fwd)
    out = []
    exp = TopExp_Explorer(face_fwd, TopAbs_WIRE)
    while exp.More():
        w = to_wire(exp.Current())
        edges = []
        wexp = BRepTools_WireExplorer(w, face_fwd)
        while wexp.More():
            edges.append(wexp.Current())
            wexp.Next()
        if not edges:  # wire explorer can fail on badly ordered wires; fall back to raw order
            eexp = TopExp_Explorer(w, TopAbs_EDGE)
            while eexp.More():
                edges.append(to_edge(eexp.Current()))
                eexp.Next()
        out.append((w.IsSame(outer), edges))
        exp.Next()
    # outer first
    out.sort(key=lambda t: not t[0])
    return out


def is_reversed(shape):
    return shape.Orientation() == TopAbs_REVERSED


def edge_vertices(edge):
    """(first, last) vertices of the edge in its natural parameter direction."""
    fwd = edge.Oriented(TopAbs_FORWARD)
    return TopExp.FirstVertex_s(to_edge(fwd)), TopExp.LastVertex_s(to_edge(fwd))


def edge_range(edge):
    return BRep_Tool.Range_s(edge)


def is_degenerated(edge):
    return BRep_Tool.Degenerated_s(edge)


def vertex_point(v):
    p = BRep_Tool.Pnt_s(v)
    return np.array([p.X(), p.Y(), p.Z()])


class Curve3:
    def __init__(self, edge):
        self.ad = BRepAdaptor_Curve(edge)
        self.is_line = self.ad.GetType() == GeomAbs_Line

    def point(self, t):
        p = self.ad.Value(t)
        return np.array([p.X(), p.Y(), p.Z()])

    def d1(self, t):
        p, v = gp_Pnt(), gp_Vec()
        self.ad.D1(t, p, v)
        return (p.X(), p.Y(), p.Z()), (v.X(), v.Y(), v.Z())


class Curve2:
    def __init__(self, edge, face):
        self.ad = BRepAdaptor_Curve2d(edge, face)

    def uv(self, t):
        p = self.ad.Value(t)
        return np.array([p.X(), p.Y()])


class Surface:
    """Fast evaluator over BRepAdaptor_Surface. Returns plain tuples: these are
    called millions of times, and small numpy arrays cost far more than the
    geometry itself."""

    def __init__(self, face):
        self.ad = BRepAdaptor_Surface(face)
        self.is_plane = self.ad.GetType() == GeomAbs_Plane
        self.u0, self.u1 = self.ad.FirstUParameter(), self.ad.LastUParameter()
        self.v0, self.v1 = self.ad.FirstVParameter(), self.ad.LastVParameter()
        self._p, self._du, self._dv = gp_Pnt(), gp_Vec(), gp_Vec()

    def point(self, u, v):
        p = self.ad.Value(u, v)
        return np.array([p.X(), p.Y(), p.Z()])

    def d1(self, u, v):
        p, du, dv = self._p, self._du, self._dv
        self.ad.D1(u, v, p, du, dv)
        return ((p.X(), p.Y(), p.Z()), (du.X(), du.Y(), du.Z()), (dv.X(), dv.Y(), dv.Z()))

    def _cross(self, u, v):
        du, dv = self._du, self._dv
        self.ad.D1(u, v, self._p, du, dv)
        ax, ay, az = du.X(), du.Y(), du.Z()
        bx, by, bz = dv.X(), dv.Y(), dv.Z()
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        return nx, ny, nz, math.sqrt(nx * nx + ny * ny + nz * nz),             max(math.sqrt(ax * ax + ay * ay + az * az), math.sqrt(bx * bx + by * by + bz * bz))

    def normal(self, u, v, toward=None):
        """Unit normal Su x Sv. Near singular points (poles, apex) step toward `toward` (a UV point)."""
        nx, ny, nz, ln, big = self._cross(u, v)
        # singular when one derivative collapses relative to the other
        # (sphere pole: |Su| ~ 1e-16 while |Sv| = r); the cross product
        # direction is then noise
        if ln > 1e-8 * big * big and ln > 1e-14:
            return np.array((nx / ln, ny / ln, nz / ln))
        if toward is not None:
            for f in (1e-4, 1e-3, 1e-2, 5e-2):
                nx, ny, nz, ln, _ = self._cross(u + f * (toward[0] - u), v + f * (toward[1] - v))
                if ln > 1e-14:
                    return np.array((nx / ln, ny / ln, nz / ln))
        return np.array([0.0, 0.0, 1.0])
