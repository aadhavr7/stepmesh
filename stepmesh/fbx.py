"""Binary FBX 7.4 writer: polygon meshes with ngons and per-corner normals.

One Model + Geometry per body. Written from the publicly documented binary
FBX layout (node records, typed properties, zlib arrays, header and footer)
and checked by importing into Blender.

Units: OpenCASCADE gives millimetres. FBX UnitScaleFactor is "centimetres
per file unit", so raw CAD output declares 0.1, and a user scale s (e.g.
0.001 for metres) declares 0.1 / s. Importers then size the model correctly.
"""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import numpy as np

from .export_obj import cluster_normals
from .meshdata import Mesh

FBX_VERSION = 7400
HEADER = b"Kaydara FBX Binary  \x00\x1a\x00"
FILE_ID = bytes.fromhex("28b32aebb624ccc2bfc8b02aa92bfcf1")
FOOT_ID = bytes.fromhex("fabcab09d0c8d466b176fb831cf7267e")
FOOT_MAGIC = bytes.fromhex("f85a8c6adef5d97eece90ce3758f290b")


class N:
    """FBX node: name, typed properties, children."""

    __slots__ = ("name", "props", "kids")

    def __init__(self, name, *props, kids=None):
        self.name, self.props, self.kids = name, list(props), list(kids or [])

    def add(self, *nodes):
        self.kids.extend(nodes)
        return self


class Raw(bytes):
    pass


def _prop(p) -> bytes:
    if isinstance(p, bool):
        return b"C" + struct.pack("<?", p)
    if isinstance(p, Raw):
        return b"R" + struct.pack("<I", len(p)) + bytes(p)
    if isinstance(p, (bytes, bytearray)):
        return b"S" + struct.pack("<I", len(p)) + bytes(p)
    if isinstance(p, str):
        b = p.encode("utf-8")
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(p, np.ndarray):
        code = {np.dtype("<f8"): b"d", np.dtype("<f4"): b"f", np.dtype("<i4"): b"i", np.dtype("<i8"): b"l"}[p.dtype]
        raw = p.tobytes()
        comp = zlib.compress(raw, 6)
        if len(comp) < len(raw):
            return code + struct.pack("<III", len(p), 1, len(comp)) + comp
        return code + struct.pack("<III", len(p), 0, len(raw)) + raw
    if isinstance(p, I64):
        return b"L" + struct.pack("<q", int(p))
    if isinstance(p, int):
        return b"I" + struct.pack("<i", p)
    if isinstance(p, float):
        return b"D" + struct.pack("<d", p)
    raise TypeError(f"unsupported FBX property {type(p)}")


class I64(int):
    pass


def _write_node(out: bytearray, n: N):
    start = len(out)
    out += b"\0" * 13  # end offset, prop count, prop length, name length
    name = n.name.encode("ascii")
    out += name
    pstart = len(out)
    for p in n.props:
        out += _prop(p)
    plen = len(out) - pstart
    if n.kids or not n.props:
        for k in n.kids:
            _write_node(out, k)
        out += b"\0" * 13
    struct.pack_into("<IIIB", out, start, len(out), len(n.props), plen, len(name))


def _P(name, ptype, label, flags, *vals):
    return N("P", name, ptype, label, flags, *vals)


def _props70(*ps):
    return N("Properties70", kids=list(ps))


def _objname(name, cls):
    return name.encode("utf-8") + b"\x00\x01" + cls.encode("ascii")


def write_fbx(path, mesh: Mesh, scale=1.0, up="z", weld="all", sharp_deg=20.0, app="StepMesh"):
    path = Path(path)
    normals, poly_n = cluster_normals(mesh, sharp_deg)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    P = np.asarray(mesh.points, dtype=np.float64) * scale
    if up == "y":
        P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1)
        normals = np.stack([normals[:, 0], normals[:, 2], -normals[:, 1]], axis=1)

    bodies = {}
    for pi, p in enumerate(mesh.polys):
        bodies.setdefault(p.body, []).append(pi)

    next_id = [1_000_000]

    def uid():
        next_id[0] += 1
        return I64(next_id[0])

    objects, conns = N("Objects"), N("Connections")
    n_geo = 0
    for body, plist in sorted(bodies.items()):
        key_index, src = {}, []
        pvi, nidx = [], []
        for pi in plist:
            p = mesh.polys[pi]
            k = len(p.verts)
            for c, v in enumerate(p.verts):
                if weld == "none":
                    key = (p.face, v)
                elif weld == "smooth":
                    key = ("n", poly_n[pi][c])
                else:
                    key = v
                if key not in key_index:
                    key_index[key] = len(src)
                    src.append(v)
                vi = key_index[key]
                pvi.append(vi if c < k - 1 else -vi - 1)   # last corner is stored as ~index
                nidx.append(poly_n[pi][c])
        used_n, ninv = np.unique(np.array(nidx, dtype=np.int64), return_inverse=True)
        verts = P[np.array(src, dtype=np.int64)].ravel().astype("<f8")
        name = mesh.body_names.get(body, f"body_{body}")
        gid, mid = uid(), uid()
        geo = N("Geometry", gid, _objname(name, "Geometry"), "Mesh").add(
            _props70(),
            N("GeometryVersion", 124),
            N("Vertices", verts),
            N("PolygonVertexIndex", np.array(pvi, dtype="<i4")),
            N("LayerElementNormal", 0).add(
                N("Version", 101), N("Name", ""),
                N("MappingInformationType", "ByPolygonVertex"),
                N("ReferenceInformationType", "IndexToDirect"),
                N("Normals", normals[used_n].ravel().astype("<f8")),
                N("NormalsIndex", ninv.astype("<i4"))),
            N("Layer", 0).add(N("Version", 100), N("LayerElement").add(
                N("Type", "LayerElementNormal"), N("TypedIndex", 0))),
        )
        model = N("Model", mid, _objname(name, "Model"), "Mesh").add(
            N("Version", 232),
            _props70(_P("DefaultAttributeIndex", "int", "Integer", "", 0),
                     _P("InheritType", "enum", "", "", 1)),
            N("MultiLayer", 0), N("MultiTake", 0), N("Shading", True), N("Culling", "CullingOff"),
        )
        objects.add(geo, model)
        conns.add(N("C", "OO", mid, I64(0)), N("C", "OO", gid, mid))
        n_geo += 1

    unit = 0.1 / scale  # centimetres per file unit (CAD data is millimetres)
    if up == "y":
        axes = (1, 1, 2, 1, 0, 1)
    else:
        axes = (2, 1, 1, -1, 0, 1)
    lt = time.localtime()
    creator = f"{app} FBX writer"
    root = [
        N("FBXHeaderExtension").add(
            N("FBXHeaderVersion", 1003), N("FBXVersion", FBX_VERSION), N("EncryptionType", 0),
            N("CreationTimeStamp").add(N("Version", 1000), N("Year", lt.tm_year), N("Month", lt.tm_mon),
                                       N("Day", lt.tm_mday), N("Hour", lt.tm_hour), N("Minute", lt.tm_min),
                                       N("Second", lt.tm_sec), N("Millisecond", 0)),
            N("Creator", creator)),
        N("FileId", Raw(FILE_ID)),
        N("CreationTime", "1970-01-01 10:00:00:000"),
        N("Creator", creator),
        N("GlobalSettings").add(N("Version", 1000), _props70(
            _P("UpAxis", "int", "Integer", "", axes[0]), _P("UpAxisSign", "int", "Integer", "", axes[1]),
            _P("FrontAxis", "int", "Integer", "", axes[2]), _P("FrontAxisSign", "int", "Integer", "", axes[3]),
            _P("CoordAxis", "int", "Integer", "", axes[4]), _P("CoordAxisSign", "int", "Integer", "", axes[5]),
            _P("OriginalUpAxis", "int", "Integer", "", axes[0]), _P("OriginalUpAxisSign", "int", "Integer", "", 1),
            _P("UnitScaleFactor", "double", "Number", "", float(unit)),
            _P("OriginalUnitScaleFactor", "double", "Number", "", float(unit)),
            _P("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0),
            _P("DefaultCamera", "KString", "", "", "Producer Perspective"),
            _P("TimeMode", "enum", "", "", 11),
            _P("TimeSpanStart", "KTime", "Time", "", I64(0)),
            _P("TimeSpanStop", "KTime", "Time", "", I64(46186158000)),
            _P("CustomFrameRate", "double", "Number", "", 24.0))),
        N("Documents").add(N("Count", 1), N("Document", uid(), "Scene", "Scene").add(
            _props70(_P("SourceObject", "object", "", ""), _P("ActiveAnimStackName", "KString", "", "", "")),
            N("RootNode", I64(0)))),
        N("References"),
        N("Definitions").add(
            N("Version", 100), N("Count", 1 + 2 * n_geo),
            N("ObjectType", "GlobalSettings").add(N("Count", 1)),
            N("ObjectType", "Geometry").add(N("Count", n_geo)),
            N("ObjectType", "Model").add(N("Count", n_geo))),
        objects,
        conns,
        N("Takes").add(N("Current", "")),
    ]
    out = bytearray(HEADER + struct.pack("<I", FBX_VERSION))
    for n in root:
        _write_node(out, n)
    out += b"\0" * 13
    out += FOOT_ID
    pad = (16 - len(out) % 16) % 16
    out += b"\0" * (pad if pad else 16)
    out += struct.pack("<I", FBX_VERSION) + b"\0" * 120 + FOOT_MAGIC
    path.write_bytes(bytes(out))
    return path
